"""Prompt 092-097：Dataset 只读工具。

dataset.list / dataset.inspect / dataset.preview / dataset.schema /
dataset.profile / dataset.quality / dataset.relation —— 全部只读，不产生新版本。
"""

from __future__ import annotations

from typing import Any

import polars as pl

from app.analysis import is_identifier_like
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

#: 关系探索时，两侧各取多少个唯一值样本估算重叠度。
#: 取样本而不是全量 join：300 万行的表做全量交集既不必要也可能把内存打满，
#: 而「能不能关联」这个问题用样本覆盖率已经足够回答。
_RELATION_SAMPLE_VALUES = 200
#: 最多对多少个共同列做重叠度估算（按左表列顺序取前 N 个）。
_RELATION_MAX_COLUMNS = 8


def _key_score(column: dict[str, Any]) -> float:
    """一个共同列配不配当关联键。

    只看匹配率会把「取值少、碰巧全覆盖」的列顶到第一（实测 tenure_months
    100% 命中、customer_id 只有 45%，系统却推荐前者，然后自己警告
    「两侧都不是唯一键、多对多会膨胀」）。先奖励**像键**的特征：
    名字像主键、且至少一侧接近唯一；匹配率只作同类之间的次要排序。
    """
    if float(column.get("match_coverage") or 0.0) <= 0:
        return -1.0
    score = 1.0 if is_identifier_like(str(column.get("column") or "")) else 0.0
    score += max(
        float(column.get("left_unique_ratio") or 0.0),
        float(column.get("right_unique_ratio") or 0.0),
    )
    # 同分时再用匹配率区分
    return score * 10 + float(column.get("match_coverage") or 0.0)
#: 规模提示阈值：超过就提醒先聚合再关联。
_RELATION_LARGE_ROWS = 1_000_000


class DatasetListTool(Tool):
    """列出数据集（只能读取列表）。"""

    name = "dataset.list"
    description = "列出当前用户可见的数据集（分页），仅读取元信息。"
    category = "dataset"
    input_schema = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 20},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}, "total": {"type": "integer"}},
    }
    permission = "read_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        page = int(params.get("page", 1))
        page_size = int(params.get("page_size", 20))
        items, total = ds.list(page=page, page_size=page_size)
        # 列出的是**元信息**（名称/描述），不是数据内容。若再按会话绑定的
        # dataset_ids 过滤，用户「列出所有数据集」就只能看到已绑定的那一个，
        # 其余数据集永远无法被发现/选择 —— 单租户平台下这是错误过滤。
        # 数据级访问控制由各读取工具内部的 assert_dataset_access 负责，不在这里。
        data = {
            "items": [
                {"id": d.id, "name": d.name, "description": d.description}
                for d in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        return ToolResult.ok(data, summary=f"共 {total} 个数据集，当前页 {len(items)} 个")


class DatasetInspectTool(Tool):
    """查看数据集详情与版本信息。"""

    name = "dataset.inspect"
    description = "查看数据集元信息、版本列表与指定版本（不给则最新版本）的规模。"
    category = "dataset"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            # ★ 必须有 version 这一项，否则引擎不会把会话基线版本填进来，
            #   工具就会去读 latest。latest 可能已被一次空结果操作顶成 0 行的表
            #   —— 实测：筛选「customer_id > 99999999」得到 0 行后，用户再问
            #   「这份数据有多少行」答的是 0 行，而基线明明还是 3035 行（第十二轮）。
            "version": {"type": "integer", "description": "要查看的版本号；不给则看最新版本"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "read_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        dataset = ds.get(dataset_id)
        versions, total = ds.get_versions(dataset_id, page=1, page_size=20)
        wanted = params.get("version")
        try:
            focus = ds.get_version_row(dataset_id, int(wanted)) if wanted else ds.get_version_row(dataset_id)
        except Exception:  # noqa: BLE001 - 指定版本不存在就退回最新，别让「看概况」失败
            focus = ds.get_version_row(dataset_id)
        data = {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "version_count": total,
            "latest_version": {
                "version": focus.version,
                "rows": focus.row_count,
                "columns": focus.column_count,
                "created_at": str(focus.created_at),
            },
            "versions": [
                {"version": v.version, "rows": v.row_count, "columns": v.column_count}
                for v in versions
            ],
        }
        # ★ 「当前版本」必须写进 summary：它是事实摘要里唯一进答案的口径，
        #   数据里还挂着 versions 列表时，模型会挑错一个版本报给用户。
        summary = (
            f"{dataset.name} 当前版本 v{focus.version}：{focus.row_count} 行 × {focus.column_count} 列"
            f"（共 {total} 个版本）"
        )
        return ToolResult.ok(data, summary=summary)


class _VersionDataTool(Tool):
    """公共：按 dataset_id(+version) 读取 DataFrame 的只读工具基类。"""

    category = "dataset"
    permission = "read_data"

    def _load_df(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ):
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        version = params.get("version")
        return ds.load_version(dataset_id, int(version) if version else None), dataset_id


class DatasetPreviewTool(_VersionDataTool):
    name = "dataset.preview"
    description = "预览数据集内容（分页，可筛选列与行条件）。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "page": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 20},
            "columns": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        preview_params = {
            k: params[k]
            for k in ("page", "page_size", "columns", "sort", "filter_logic")
            if k in params
        }
        if params.get("filters"):
            # 工具面向 Agent 的参数名为 filters，preview 内部参数为 filter_conditions
            preview_params["filter_conditions"] = params["filters"]
        result = engine.preview(df, **preview_params)
        return ToolResult.ok(
            result,
            summary=(
                f"数据集 {dataset_id} 预览：{result['total']} 行，"
                f"第 {result['page']}/{result['total_pages']} 页"
            ),
        )


class DatasetSchemaTool(_VersionDataTool):
    name = "dataset.schema"
    description = "查看数据集列结构（列名、类型、空值、唯一值、样本值）。"
    input_schema = {
        "type": "object",
        "properties": {"dataset_id": {"type": "integer"}, "version": {"type": "integer"}},
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        return ToolResult.ok(engine.schema(df), summary=f"数据集 {dataset_id} 共 {df.width} 列")


class DatasetProfileTool(_VersionDataTool):
    name = "dataset.profile"
    description = "生成数据集统计画像（数值分布、缺失、类别 Top 等）。"
    input_schema = {
        "type": "object",
        "properties": {"dataset_id": {"type": "integer"}, "version": {"type": "integer"}},
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        return ToolResult.ok(engine.profile(df), summary=f"数据集 {dataset_id} 画像完成")


class DatasetQualityTool(_VersionDataTool):
    name = "dataset.quality"
    description = (
        "数据质量检查（缺失、重复、异常值、Schema）。"
        "传 target 时该列只做描述、不产出清洗建议；异常值方法会按字段语义标注适用性。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            # 第二层：目标列保护。用户/Agent 已知目标列时必须传，
            # 否则报告会把「要预测的对象」当成需要清洗的脏数据。
            "target": {"type": "string", "description": "目标列名（可选，传入后该列不做清洗建议）"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        target = str(params.get("target") or "") or None
        report = engine.quality(df, target=target)
        issue_count = len(report.get("issues", []))
        # 补充按列缺失值明细：质量报告里只有「缺失单元格总数」，没有分列数字。
        # 用户追问「哪列缺失最多」时，Agent 手里没有数据可答，只能回「本次未获取」。
        # 这里一次性算出各列缺失数（一次 null_count 扫描），缺失为 0 也如实列出。
        nulls = df.null_count().row(0)
        report["missing_by_column"] = [
            {
                "column": str(c),
                "missing": int(nulls[i]),
                "missing_rate": round(int(nulls[i]) / df.height, 4) if df.height else 0.0,
            }
            for i, c in enumerate(df.columns)
        ]
        summary = f"数据集 {dataset_id} 质量检查：{issue_count} 个问题"
        # 缺失值结论必须写进 summary：data 里的明细会被事实摘要的字段上限挤掉，
        # 而 summary 直接进事实摘要 —— 用户追问「哪列缺失最多」时 AI 才能有据可答。
        missing_cols = [(str(c), int(nulls[i])) for i, c in enumerate(df.columns) if int(nulls[i]) > 0]
        if missing_cols:
            top = max(missing_cols, key=lambda x: x[1])
            summary += f"；缺失值最多的列是 {top[0]}（{top[1]} 个，另有 {len(missing_cols) - 1} 列也有缺失）"
        else:
            summary += "；各列均无缺失值"
        if target:
            report.setdefault("target_protection", {
                "target": target,
                "note": f"{target} 已标记为目标列，只做描述、未进入清洗建议",
            })
            summary += f"；目标列 {target} 已排除出清洗建议"
        return ToolResult.ok(report, summary=summary)


class DatasetRelationTool(Tool):
    """两个数据集之间的关系探索（只读）。

    回答的是「这两个数据集有什么联系、能不能关联、按什么键关联」。

    为什么需要它
    ------------
    用户会问「探索这两个数据集间的联系」，而在它之前没有任何工具能回答：
    要么只能看单个数据集的结构（dataset.schema），要么就得真的去合并
    （data.merge，高风险、会产出新版本）。「先看看能不能关联」这个最自然的
    中间步骤是缺的，于是这类请求只能被路由到一个不相干的意图上（实测就被
    「机器学习」三个字吸到了训练链路上，最后跑出一个 kmeans 了事）。

    只读性
    ------
    全程走 ``scan_version``（惰性扫描）+ 聚合下推，不物化全表、不写任何版本。
    重叠度用两侧各 200 个唯一值样本估算 —— 全量 join 对百万行表既慢又占内存，
    而「能不能关联」用覆盖率已经能回答。
    """

    name = "dataset.relation"
    description = (
        "分析两个数据集之间的联系：共同字段、各字段值域重叠度、可关联键建议与规模对比。"
        "只读，不产生新版本；适合「这两个数据集有什么关系 / 能不能关联 / 按什么键合并」这类探索。"
    )
    category = "dataset"
    input_schema = {
        "type": "object",
        "properties": {
            "left_dataset_id": {"type": "integer", "description": "左侧数据集 id（默认会话主数据集）"},
            "right_dataset_id": {"type": "integer", "description": "右侧数据集 id（默认会话第二个数据集）"},
            "sample_values": {"type": "integer", "description": f"每侧参与重叠度估算的唯一值样本数，默认 {_RELATION_SAMPLE_VALUES}"},
        },
        "required": ["left_dataset_id", "right_dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"
    risk_level = "low"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        left_id = int(params["left_dataset_id"])
        right_id = int(params["right_dataset_id"])
        self.assert_dataset_access(context, left_id)
        self.assert_dataset_access(context, right_id)
        if left_id == right_id:
            return ToolResult.fail("需要两个不同的数据集才能分析它们之间的关系")

        sample = int(params.get("sample_values") or _RELATION_SAMPLE_VALUES)
        sample = max(20, min(sample, 2000))

        left_name = self._name(services, left_id)
        right_name = self._name(services, right_id)
        left_frame = ds.scan_version(left_id)
        right_frame = ds.scan_version(right_id)
        left_schema = left_frame.collect_schema()
        right_schema = right_frame.collect_schema()

        left_info = self._side(left_id, left_name, left_frame, left_schema)
        right_info = self._side(right_id, right_name, right_frame, right_schema)

        shared = [c for c in left_schema.names() if c in right_schema.names()]
        columns: list[dict[str, Any]] = []
        for column in shared[:_RELATION_MAX_COLUMNS]:
            columns.append(
                self._compare(
                    left_frame, right_frame, left_schema, right_schema,
                    column, sample, left_info["rows"], right_info["rows"],
                )
            )

        # ★ 排序不能只看匹配率。实测：tenure_months（取值少、碰巧 100% 命中）
        #   压过了 customer_id（真实主键，但因主表被筛选过只匹配 45%）——
        #   系统推荐了一个「两侧都不是唯一键」的列，还顺带警告自己
        #   「多对多会大幅膨胀」。关联键要先看**像不像键**：名字像主键，
        #   且至少一侧接近唯一。
        ranked = sorted(columns, key=_key_score, reverse=True)
        keys = [c for c in ranked if c["match_coverage"] > 0][:3]
        advice = self._advice(left_info, right_info, shared, ranked, len(shared) > _RELATION_MAX_COLUMNS)

        data = {
            "left": left_info,
            "right": right_info,
            "shared_columns": columns,
            "shared_column_count": len(shared),
            "recommended_keys": [
                {"column": c["column"], "match_coverage": c["match_coverage"], "join_hint": c["join_hint"]}
                for c in keys
            ],
            "advice": advice,
        }
        summary = (
            f"两个数据集共有 {len(shared)} 个同名字段"
            + (f"；推荐关联键 {keys[0]['column']}（左→右匹配率 {keys[0]['match_coverage']:.0%}）" if keys else "；未找到可用的关联键")
        )
        return ToolResult.ok(
            data,
            summary=summary,
            warnings=self._warnings(left_info, right_info, ranked),
            signals=["shared_columns_found"] if shared else ["no_shared_columns"],
        )

    # ---- 内部 --------------------------------------------------------
    @staticmethod
    def _name(services: ToolServices, dataset_id: int) -> str:
        try:
            return str(services.require("dataset_service").get(dataset_id).name or "")
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _side(dataset_id: int, name: str, frame: Any, schema: Any) -> dict[str, Any]:
        rows = int(frame.select(pl.len()).collect().item())
        names = list(schema.names())
        return {
            "dataset_id": dataset_id,
            "name": name,
            "rows": rows,
            "column_count": len(names),
            "column_names": names[:40],
        }

    @classmethod
    def _compare(
        cls, left_frame: Any, right_frame: Any, left_schema: Any, right_schema: Any,
        column: str, sample: int, left_rows: int, right_rows: int,
    ) -> dict[str, Any]:
        left_unique = int(left_frame.select(pl.col(column).drop_nulls().n_unique()).collect().item())
        right_unique = int(right_frame.select(pl.col(column).drop_nulls().n_unique()).collect().item())
        left_values = cls._sample(left_frame, column, sample)
        right_values = cls._sample(right_frame, column, sample)
        hit = len(left_values & right_values)
        coverage = round(hit / len(left_values), 4) if left_values else 0.0

        left_dtype = str(left_schema[column])
        right_dtype = str(right_schema[column])
        left_ratio = left_unique / left_rows if left_rows else 0.0
        right_ratio = right_unique / right_rows if right_rows else 0.0
        if coverage == 0:
            hint = "值域几乎不重叠，不建议作为关联键"
        elif left_ratio > 0.95 and right_ratio > 0.95:
            hint = "两侧都接近唯一，可作为 1:1 主键关联"
        elif left_ratio > 0.95 or right_ratio > 0.95:
            hint = "一侧接近唯一，可作为 1:N 外键关联（注意行数膨胀）"
        else:
            hint = "两侧都不是唯一键，关联后行数可能大幅膨胀（多对多）"
        return {
            "column": column,
            "left_type": left_dtype,
            "right_type": right_dtype,
            "type_matches": left_dtype == right_dtype,
            "left_unique": left_unique,
            "right_unique": right_unique,
            "left_unique_ratio": round(left_ratio, 4),
            "right_unique_ratio": round(right_ratio, 4),
            "match_coverage": coverage,
            "join_hint": hint,
        }

    @staticmethod
    def _sample(frame: Any, column: str, sample: int) -> set[str]:
        try:
            series = frame.select(pl.col(column).drop_nulls().unique().head(sample)).collect().to_series()
        except Exception:  # noqa: BLE001 - 单列失败不该让整次关系探索失败
            return set()
        return {str(v) for v in series.to_list()[:sample]}

    @classmethod
    def _advice(
        cls, left: dict[str, Any], right: dict[str, Any],
        shared: list[str], ranked: list[dict[str, Any]], truncated: bool,
    ) -> list[str]:
        advice: list[str] = []
        label_left = left["name"] or f"数据集 {left['dataset_id']}"
        label_right = right["name"] or f"数据集 {right['dataset_id']}"
        if not shared:
            advice.append(
                f"{label_left} 与 {label_right} 没有同名字段，无法直接按列关联。"
                "可先确认是否存在语义相同但命名不同的字段（如 id / ID / 编号），"
                "或改用时间粒度对齐等派生键。"
            )
            return advice

        best = ranked[0] if ranked else None
        if best and best["match_coverage"] > 0:
            advice.append(
                f"建议按「{best['column']}」关联：左→右匹配率 {best['match_coverage']:.0%}，{best['join_hint']}。"
            )
            if not best["type_matches"]:
                advice.append(
                    f"注意：「{best['column']}」两侧类型不同（{best['left_type']} vs {best['right_type']}），"
                    "关联前需要先统一类型。"
                )
        else:
            advice.append("共同字段的值域几乎没有重叠，直接关联会丢掉绝大多数行，建议先核对字段语义与单位。")

        if truncated:
            advice.append(f"共同字段超过 {_RELATION_MAX_COLUMNS} 个，仅对前 {_RELATION_MAX_COLUMNS} 个做了重叠度估算。")

        big = [s for s in (left, right) if s["rows"] >= _RELATION_LARGE_ROWS]
        for side in big:
            advice.append(
                f"{side['name'] or ('数据集 ' + str(side['dataset_id']))} 有 {side['rows']} 行，"
                "关联与建模前建议先按分析目标做筛选或聚合抽样，避免内存与耗时失控。"
            )
        return advice

    @staticmethod
    def _warnings(left: dict[str, Any], right: dict[str, Any], ranked: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        for item in ranked:
            if item["match_coverage"] and item["match_coverage"] < 0.3:
                warnings.append(
                    f"字段「{item['column']}」两侧值域重叠仅 {item['match_coverage']:.0%}，按它关联会丢大量行"
                )
            if "Float" in item["left_type"] or "Float" in item["right_type"]:
                warnings.append(f"字段「{item['column']}」是浮点类型，作为关联键容易因精度问题匹配不上")
        for side in (left, right):
            if side["rows"] >= _RELATION_LARGE_ROWS:
                warnings.append(f"数据集 {side['dataset_id']} 达 {side['rows']} 行，全量关联代价很高")
        return warnings[:5]
