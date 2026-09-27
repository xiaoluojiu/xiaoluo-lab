"""Prompt 098-102：Data 修改类工具。

data.filter / data.clean / data.transform / data.merge 通过 DataEngineService
执行（产生新版本），工具自身不触碰 DataFrame。

例外是 data.aggregate：它是**只读统计**，就地算完直接回结果，不产生新版本
（详见该类文档串里记的事故）。

data.merge 强制走 Plan -> Validate -> Permission -> Execute 流程。
"""

from __future__ import annotations

from typing import Any

from app.data_engine.operations import aggregate
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

#: 聚合结果最多回传多少行。分组基数很高时（按用户 id 分组）不必把几万行塞进答案，
#: 总数由 ``rows`` 字段如实给出。
_AGG_RESULT_ROWS = 50


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
        data: dict[str, Any] = {
            "dataset_id": dataset_id,
            "new_version": version.version,
            "rows": version.row_count,
            "columns": version.column_count,
            # 回执实际应用的操作参数：否则 AI 只能回「本次未获取筛选条件/聚合配置」，
            # 无法向用户确认「确实按你说的执行了」。
            "params": op_params,
        }
        # 聚合/转换/筛选这类「产出新数据」的操作，回执结果前几行，
        # 否则聚合出的各分组数值永远到不了用户眼前（只能看到「生成了 30 行 × 2 列」）。
        try:
            ds = services.require("dataset_service")
            out_df = ds.load_version(dataset_id, version.version)
            data["preview"] = out_df.head(10).to_dicts()
        except Exception:  # noqa: BLE001 - 预览是锦上添花，失败不阻断
            pass
        return ToolResult.ok(
            data,
            summary=(
                f"操作 {self.op_type} 完成，生成版本 v{version.version}"
                f"（{version.row_count} 行 × {version.column_count} 列）"
            ),
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
                "description": "筛选条件列表，每个条件用 column/op/value 三个键表达",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "要筛选的列名，必须是数据集的真实列名"},
                        "op": {
                            "type": "string",
                            "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in", "is_null"],
                            "description": "比较运算符：eq=等于、neq=不等于、gt=大于、gte=大于等于、lt=小于、lte=小于等于、contains=包含子串、in=在集合中、is_null=为空",
                        },
                        "value": {"description": "比较值，数字直接写数字，字符串写字符串；is_null 不需要 value"},
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
        # 回执最终版本的行数：用户问「清洗后还剩多少行」时 AI 才有据可答，
        # 否则只能回「本次未获取行数」。
        rows = None
        try:
            ds = services.require("dataset_service")
            row = ds.get_version_row(dataset_id, versions[-1])
            if row is not None:
                rows = int(row.row_count)
        except Exception:  # noqa: BLE001 - 行数是回执，读不到不阻断
            pass
        return ToolResult.ok(
            {"dataset_id": dataset_id, "versions": versions, "rows": rows},
            summary=f"清洗完成：{', '.join(steps)}" + (f"（{rows} 行）" if rows is not None else ""),
        )


class DataTransformTool(_DataOpTool):
    name = "data.transform"
    description = "新增/覆盖派生列（数值运算、日期部分、常量）。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "name": {"type": "string", "description": "新增/覆盖的列名，不能与已有列重名"},
            "expression": {
                "type": "object",
                "description": (
                    "结构化表达式。常见形态："
                    "①直接引用列 {'type':'column','column':'真实列名'}；"
                    "②常量 {'type':'value','value':123}；"
                    "③二元运算 {'type':'math','op':'add|sub|mul|div','left':{操作数},'right':{操作数}}；"
                    "④一元运算 {'type':'math','op':'abs|neg','args':[{操作数}]}（abs=绝对值）。"
                ),
            },
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["dataset_id", "name", "expression"],
    }
    op_type = "transform"


class DataAggregateTool(Tool):
    """分组聚合统计（**只读**）。

    ★ 这里曾经继承 ``_DataOpTool``，于是把 7 行 × 2 列的聚合结果**写成了数据集
      的新版本**——而新版本即 latest。用户问完「按 purpose 分组求 credit_amount
      的平均值」，他 1220 行 × 14 列的数据集就变成了一张分组小表，接着问的每一
      句（相关性/质量检查/age 的分布）都跑在这张废表上：
         · eda.correlation 报「至少需要 2 个数值字段」（明明有 7 个数值列）
         · dataset.quality 回答「共 7 行、2 列」
         · 点名 age，槽位抽取器从 latest 的列名里只能挑到 purpose
      聚合是**看数字**，不是**改数据**：就地算完直接把结果回给用户，数据集不动。
    """

    name = "data.aggregate"
    description = "分组聚合统计（count/sum/mean/median/min/max/std），只返回统计结果，不改动数据集。"
    category = "data"
    permission = "analyze_data"
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "group_by": {"type": "array", "items": {"type": "string"}, "description": "分组列名列表"},
            "aggregations": {
                "type": "array",
                "description": "聚合配置列表，每个用 column+func 两个键表达（func 是聚合函数）",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "要聚合的列名"},
                        "func": {
                            "type": "string",
                            "enum": ["count", "sum", "mean", "median", "min", "max", "std"],
                            "description": "聚合函数：mean=平均值、sum=求和、count=计数、median=中位数、min/max=最小/最大、std=标准差",
                        },
                    },
                    "required": ["column", "func"],
                },
            },
        },
        "required": ["dataset_id", "group_by", "aggregations"],
    }
    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        # 聚合配置的形状很松（槽位抽取既可能给 {"column","func"}，也可能给
        # {"credit_amount": "mean"}），缺 func 时直接往下传只会得到
        # 「unsupported aggregation func: None」—— 用户看到的是一次毫无意义的失败。
        params = _normalize_aggregations(params)
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        version = params.get("version")
        df = ds.load_version(dataset_id, int(version) if version else None)
        group_by = [str(c) for c in (params.get("group_by") or []) if c]
        aggregations = params.get("aggregations") or []
        if not aggregations:
            # ★「按 Contract 分组统计」没说聚合什么。此前链路挂起反问
            #   「要对哪些列做什么聚合？」且一个候选都不给，用户改口后再问一遍
            #   （Telco 压测实测）。分组计数是「统计」最自然的默认。
            aggregations = [{"column": group_by[0], "func": "count"}]
            defaulted = True
        else:
            defaulted = False
        # ★ 非数值列配 mean 会直接抛
        #   「aggregation func 'mean' only applies to numeric columns」
        #   （「按 Contract 分组统计流失率」里 Churn 是 Yes/No，实测踩到）。
        #   这里按列类型纠偏成计数，并把这次改动留痕。
        retuned: list[str] = []
        fixed_aggs: list[dict[str, Any]] = []
        for item in aggregations:
            if not isinstance(item, dict):
                fixed_aggs.append(item)
                continue
            column = str(item.get("column") or "")
            func = str(item.get("func") or "")
            if column in df.columns and not df.schema[column].is_numeric() and func != "count":
                retuned.append(f"{column} 是非数值列，{func} 改为 count")
                item = {**item, "func": "count"}
            fixed_aggs.append(item)
        aggregations = fixed_aggs
        out = aggregate(df, group_by=group_by, aggregations=aggregations)
        preview = out.head(_AGG_RESULT_ROWS).to_dicts()
        return ToolResult.ok(
            {
                "dataset_id": dataset_id,
                "rows": int(out.height),
                "columns": int(out.width),
                # 回执实际应用的操作参数：否则答案说不出「确实按你说的分组算了」
                "params": {"group_by": group_by, "aggregations": aggregations},
                "preview": preview,
            },
            summary=(
                f"聚合完成：{out.height} 行 × {out.width} 列"
                + ("（未指定聚合方式，默认按分组计数）" if defaulted else "")
                + ("；" + "；".join(retuned) if retuned else "")
                + "（统计结果，数据集未改动）"
            ),
        )


#: 合法的聚合函数（与 ``data_engine.operations.AGG_FUNCTIONS`` 同口径）
_ALLOWED_AGG_FUNCS = ("count", "sum", "mean", "median", "min", "max", "std")
#: 中文说法 → 标准函数名。槽位抽取常常原样带回「平均值」「笔数」这类词。
_AGG_FUNC_ALIASES = {
    "平均值": "mean", "均值": "mean", "平均": "mean", "mean": "mean",
    "求和": "sum", "总和": "sum", "合计": "sum", "sum": "sum",
    "计数": "count", "笔数": "count", "数量": "count", "个数": "count", "条数": "count", "count": "count",
    "中位数": "median", "median": "median",
    "最大值": "max", "最大": "max", "max": "max",
    "最小值": "min", "最小": "min", "min": "min",
    "标准差": "std", "std": "std",
}


def _normalize_aggregations(params: dict[str, Any]) -> dict[str, Any]:
    """把聚合配置补成 ``{"column": ..., "func": ...}``。

    能接受的三种输入：
    - ``{"column": "credit_amount", "func": "mean"}``（标准形）
    - ``{"credit_amount": "mean"}``（抽取模型最常给的单键形）
    - 缺 func / func 是中文说法（``{"column": "x", "func": "平均值"}``）

    补不出来的项直接丢弃——留下它只会让引擎抛一句用户看不懂的错。
    """
    items = params.get("aggregations")
    if not isinstance(items, list):
        return params

    fixed: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        column = item.get("column")
        func = item.get("func")
        if column is None and func is None and len(item) == 1:
            # 单键形：{"credit_amount": "mean"} —— 抽取模型最常给的形态
            column, func = next(iter(item.items()))
        func = _AGG_FUNC_ALIASES.get(str(func).strip().lower(), str(func).strip().lower())
        if func not in _ALLOWED_AGG_FUNCS:
            # 没写函数：有点名列 → 求均值（最常见的诉求）；连列都没有 → 计数
            func = "mean" if column else "count"
        if not column:
            continue
        fixed.append({"column": str(column), "func": func})

    if not fixed:
        # 一个都没救回来：原样交回，让必填槽位/引擎给出它自己的错误
        return params
    return {**params, "aggregations": fixed}


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
