"""Prompt 098-102：Data 修改类工具。

data.filter / data.clean / data.transform / data.aggregate / data.merge。
全部通过 DataEngineService 执行（产生新版本），工具自身不触碰 DataFrame。
data.merge 强制走 Plan -> Validate -> Permission -> Execute 流程。
"""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult


class _DataOpTool(Tool):
    """公共：版本化数据操作基类。"""

    category = "data"
    permission = "modify_data"
    risk_level = "medium"
    op_type = ""

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        input_version = params.get("version")
        op_params = {
            k: v
            for k, v in params.items()
            if k not in ("dataset_id", "version")
        }
        version, _operation = engine.run_operation(
            dataset_id,
            self.op_type,
            op_params,
            input_version=int(input_version) if input_version else None,
        )
        return ToolResult.ok(
            {
                "dataset_id": dataset_id,
                "new_version": version.version,
                "rows": version.row_count,
                "columns": version.column_count,
            },
            summary=f"操作 {self.op_type} 完成，生成版本 v{version.version}",
        )


class DataFilterTool(_DataOpTool):
    name = "data.filter"
    description = "按条件过滤数据集行，生成新版本。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "op": {"type": "string"},
                        "value": {},
                    },
                    "required": ["column", "op"],
                },
            },
            "logic": {"type": "string", "enum": ["and", "or"]},
        },
        "required": ["dataset_id", "conditions"],
    }
    op_type = "filter"


class DataCleanTool(Tool):
    """数据清洗：缺失处理 + 去重（两步顺序执行，各生成版本）。"""

    name = "data.clean"
    description = "清洗数据：先处理缺失值（mean/median/mode/constant/drop），再按需去重。"
    category = "data"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "missing": {
                "type": "object",
                "description": "handle_missing 参数（strategy/columns/value）",
            },
            "deduplicate": {
                "type": "object",
                "description": "drop_duplicates 参数（subset/keep）",
            },
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "modify_data"
    risk_level = "high"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        input_version = params.get("version")
        versions: list[int] = []
        steps: list[str] = []
        if params.get("missing"):
            v, _ = engine.run_operation(
                dataset_id,
                "missing",
                params["missing"],
                input_version=int(input_version) if input_version else None,
            )
            versions.append(v.version)
            input_version = v.version
            steps.append(f"missing->v{v.version}")
        if params.get("deduplicate"):
            v, _ = engine.run_operation(
                dataset_id,
                "duplicate",
                params["deduplicate"],
                input_version=int(input_version) if input_version else None,
            )
            versions.append(v.version)
            steps.append(f"deduplicate->v{v.version}")
        if not versions:
            return ToolResult.fail("未提供任何清洗步骤（missing/deduplicate）")
        return ToolResult.ok(
            {"dataset_id": dataset_id, "versions": versions},
            summary=f"清洗完成：{', '.join(steps)}",
        )


class DataTransformTool(_DataOpTool):
    name = "data.transform"
    description = "新增/覆盖派生列（数值运算、日期部分、常量）。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "name": {"type": "string", "description": "新增/覆盖的列名"},
            "expression": {
                "type": "object",
                "description": (
                    "结构化表达式：{'type': 'column'|'value'|'math'|'date_part', ...}"
                ),
            },
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["dataset_id", "name", "expression"],
    }
    op_type = "transform"


class DataAggregateTool(_DataOpTool):
    name = "data.aggregate"
    description = "分组聚合统计（count/sum/mean/median/min/max/std）。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "group_by": {"type": "array", "items": {"type": "string"}},
            "aggregations": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["dataset_id", "group_by", "aggregations"],
    }
    op_type = "aggregate"


# 常见的外键后缀（维度表主键），用于「事实表外键 = <前缀> + <维度主键名>」的识别。
# 例：PULocationID / DOLocationID → 右表 LocationID；airport_code → 右表 code。
_FK_SUFFIXES = ("id", "code", "no", "num", "key", "name")


def _infer_join_keys(left_df: Any, right_df: Any) -> list[dict[str, str]]:
    """未显式指定 Join Key 时推断：优先同名列，其次后缀/前缀包含关系。

    只在调用方未给 keys 时兜底。此前 keys 是必填，LLM / 规则规划器往往拿不到列名
    （规划阶段不读 Schema），导致多表关联请求必然失败；推断结果会随 plan 一起返回，
    便于前端与 Agent 复核。推断不出时返回空列表，由调用方给出明确的列清单错误。

    匹配策略（按可靠性降序）：
      1. 同名列（归一化后相等，如 user_id ↔ UserID）；
      2. 后缀包含：左表某列以右表某列名**结尾**（PULocationID ↔ LocationID），
         或反之（右表列是左表列名的一部分）。这是维度表主键 + 事实表外键前缀的通用形态；
      3. `<stem>_id` 约定（user_id ↔ user，仅当后缀是短外键词时才用）。
    """
    # 归一化：忽略大小写与 _ / - 分隔符，这样 userId 与 user_id 也能对上
    def _norm(name: str) -> str:
        return str(name).lower().replace("_", "").replace("-", "").strip()

    left_cols = [str(c) for c in left_df.columns]
    right_cols = [str(c) for c in right_df.columns]
    right_by_norm = {_norm(c): c for c in right_cols}
    if not right_by_norm:
        return []

    # 右表列的唯一性画像（供「同名列是否可靠关联键」判断）。
    # 测试里的假对象只有 .columns，无 .n_unique()，此处防御性回退为 None。
    def _right_unique(col: str) -> int | None:
        try:
            s = right_df[col]
            return int(s.n_unique())
        except Exception:
            return None

    def _right_rows() -> int | None:
        try:
            return int(right_df.height)
        except Exception:
            return None

    # 1. 同名列 —— 但**只有「在右表唯一」的同名列才是可靠 join key**。
    #    否则（如事实表历史上已 join 过一次维度表、带入了 Borough/Zone/service_zone
    #    等冗余维度列）会贪心堆叠多个非唯一列组成 composite key，导致 many-to-many。
    #    这类场景应让位给「后缀包含」分支去匹配真正唯一的主键（LocationID）。
    same_name = [
        {"left": col, "right": right_by_norm[_norm(col)]}
        for col in left_cols
        if _norm(col) in right_by_norm
    ]
    # 裸 id（两表各自的主键）不是可靠的关联键，有其他同名列时优先排除
    if len(same_name) > 1:
        filtered = [k for k in same_name if _norm(k["left"]) != "id"]
        if filtered:
            same_name = filtered

    if same_name:
        rows = _right_rows()
        unique_keys: list[dict[str, str]] = []
        for k in same_name:
            u = _right_unique(k["right"])
            # 无法取唯一性（假对象只有 .columns）时保守视为候选，保留原行为；
            # 能取到唯一性时，只有「右表完全唯一」的同名列才作为可靠 join key。
            if u is None or (rows is not None and u == rows):
                unique_keys.append(k)
        # 优先采用「右表唯一」的同名列（至多取 1 个，避免堆叠成 composite 制造 many-to-many）
        if unique_keys:
            return unique_keys[:1]
        # 所有同名列在右表都不唯一 → 它们是冗余维度列（历史上 join 带入），
        # 不可靠，让位给「后缀包含」分支去匹配真正唯一的主键（LocationID）。
        same_name = []

    left_by_norm = {_norm(c): c for c in left_cols}

    # 2. 后缀包含：事实表外键 = <前缀> + <维度主键名>（PULocationID ↔ LocationID）。
    #    只接受「维度主键名 ≥ 2 字符」的包含，避免把 id 这类单字符碎片误判成主键。
    #    要求包含关系的列名本身以常见外键后缀结尾，进一步降低误匹配。
    suffix_hits: list[dict[str, str]] = []
    seen_right: set[str] = set()
    for rnorm, rcol in right_by_norm.items():
        if len(rnorm) < 2 or not any(rnorm.endswith(s) for s in _FK_SUFFIXES):
            continue
        for lnorm, lcol in left_by_norm.items():
            if lcol == rcol or len(lnorm) <= len(rnorm):
                continue
            if lnorm.endswith(rnorm):
                # 同一维度主键只保留一个外键：data.merge 是单次 join，
                # 若把 PULocationID / DOLocationID 都指向 LocationID 会形成两个
                # right=LocationID 的 JoinKey，执行层 rename 会产生重复列名而崩溃。
                # 保留左表列序靠前的第一个，其余外键由用户/Agent 显式指定做二次合并。
                if rnorm not in seen_right:
                    seen_right.add(rnorm)
                    suffix_hits.append({"left": lcol, "right": rcol})
    if suffix_hits:
        return suffix_hits[:3]

    # 3. `<stem>_id` 约定（裸 id 主键匹配）
    for col in left_cols:
        norm = _norm(col)
        if not norm.endswith("id") or norm == "id":
            continue
        stem = norm[:-2]
        if not stem:
            continue
        for candidate in (norm, f"{stem}_id", stem):
            match = right_by_norm.get(candidate.replace("_", ""))
            if match is not None:
                return [{"left": col, "right": match}]
    return []


class DataMergeTool(Tool):
    """数据合并：Plan -> Validate -> Permission -> Execute。"""

    name = "data.merge"
    description = (
        "将两个数据集按 Key 合并：自动建议字段映射（Plan），"
        "校验合并计划（Validate），权限通过后执行（Execute）并产生新版本。"
        "未指定 keys 时按同名列自动推断 Join Key（推断结果会写入返回的 plan）。"
    )
    category = "data"
    input_schema = {
        "type": "object",
        "properties": {
            "left_dataset_id": {"type": "integer"},
            "right_dataset_id": {"type": "integer"},
            "keys": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
                    "required": ["left", "right"],
                },
            },
            "join_type": {"type": "string", "enum": ["inner", "left", "right", "outer"]},
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "right_column": {"type": "string"},
                        "output_column": {"type": "string"},
                    },
                },
            },
        },
        # keys 不再强制：缺省时由工具按同名列推断，避免「未指定 Join Key」直接失败
        "required": ["left_dataset_id", "right_dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "create_version"
    risk_level = "high"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        from app.data_engine.merge.plan import ColumnMapping, JoinKey, MergePlan

        engine = services.require("data_engine_service")
        ds = services.require("dataset_service")
        left_id = int(params["left_dataset_id"])
        right_id = int(params["right_dataset_id"])
        self.assert_dataset_access(context, left_id)
        self.assert_dataset_access(context, right_id)

        left_version = params.get("left_version")
        right_version = params.get("right_version")
        left_df = ds.load_version(left_id, int(left_version) if left_version else None)
        right_df = ds.load_version(right_id, int(right_version) if right_version else None)

        # 1. Plan：构造计划；mapping 未提供时自动建议
        raw_keys = [k for k in (params.get("keys") or []) if isinstance(k, dict) and k.get("left") and k.get("right")]
        inferred_keys: list[dict[str, str]] = []
        if not raw_keys:
            inferred_keys = _infer_join_keys(left_df, right_df)
            if not inferred_keys:
                return ToolResult.fail(
                    "未能确定 Join Key：请显式传入 keys（如 [{\"left\": \"user_id\", \"right\": \"user_id\"}]）。"
                    f"左表列：{list(left_df.columns)}；右表列：{list(right_df.columns)}",
                    metadata={"stage": "plan"},
                )
        plan = MergePlan(
            left={"dataset_id": left_id},
            right={"dataset_id": right_id},
            keys=[JoinKey(left=k["left"], right=k["right"]) for k in (raw_keys or inferred_keys)],
            join_type=params.get("join_type", "inner"),
        )
        if inferred_keys:
            plan.warnings.append(
                "未指定 Join Key，已按同名列自动推断："
                + "、".join(f"{k['left']} = {k['right']}" for k in inferred_keys)
            )
        suggestions = []
        if params.get("mappings"):
            plan.mapping = [
                ColumnMapping(
                    right_column=m["right_column"], output_column=m["output_column"]
                )
                for m in params["mappings"]
            ]
        else:
            # 建议仅作参考信息返回：Join 由 keys 决定，
            # 右表非键列默认全部引入（同名列由执行器加后缀）。
            suggestions = engine.suggest_mappings(left_df, right_df)

        # 2. Validate（权限由 Registry 入口已裁决）
        validation = engine.validate_merge(left_df, right_df, plan)
        if not validation["ok"]:
            return ToolResult.fail(
                validation["errors"],
                data={"plan": plan.to_dict(), "validation": validation},
                metadata={"stage": "validate"},
            )

        # 3/4. Execute（run_merge 内部会再次强制校验）
        output_version, report = engine.run_merge(
            left_id, right_id, plan,
            input_version=int(left_version) if left_version else None,
            right_version=int(right_version) if right_version else None,
        )
        return ToolResult.ok(
            {
                "dataset_id": left_id,
                "new_version": output_version.version,
                "report": report.to_dict(),
                "suggested_mappings": suggestions,
                "plan": plan.to_dict(),
                "validation": validation,
            },
            summary=(
                f"合并完成：匹配 {report.matched_rows} 行，"
                f"生成版本 v{output_version.version}"
            ),
        )
