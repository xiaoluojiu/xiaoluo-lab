"""Prompt 103-107：EDA 分析工具（只读）。

eda.describe / eda.distribution / eda.correlation / eda.outlier / eda.visualize。
直接复用 Phase 4 的 EDA 模块，工具层只做参数与权限编排。
"""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from app.analysis import (
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    DistributionAnalyzer,
    DistributionOverviewAnalyzer,
    EdaOutlierAnalyzer,
    VisualizationBuilder,
)
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult


def _extract_signals(result: Any) -> list[str]:
    """从分析结果里提取结构化 signals（供 DecisionProvider 驱动下一步）。

    只认**机器可读**的 signal 标记，不解析自然语言。分析器若已显式产出
    ``signals`` 字段则直接采用；否则按结果结构做确定性推断。
    """
    if not isinstance(result, dict):
        return []
    # 分析器已显式产出 signals（如 DistributionOverviewAnalyzer）。
    explicit = result.get("signals")
    if isinstance(explicit, list):
        return [str(s) for s in explicit]
    # 其他分析器的确定性推断（异常值检测 → outliers_detected）。
    #
    # ★ 历史缺陷：这里判的是**顶层** `outliers` / `outlier_count` 键，而
    #   `EdaOutlierAnalyzer.analyze` 返回的是 `{"method": ..., "columns": [...]}`，
    #   异常数嵌在 `columns[i]["outlier_count"]` 里 —— 条件永远不成立，
    #   `outliers_detected` 信号**从未被产出过**。
    #   `decision/signal.py` 的白名单因此有一条永远走不到的分支，
    #   「分布 → 异常」这类信号驱动的下一步实际不会发生。
    if "outliers" in result or "outlier_count" in result:
        return ["outliers_detected"]
    columns = result.get("columns")
    if isinstance(columns, list):
        for item in columns:
            if isinstance(item, dict) and int(item.get("outlier_count") or 0) > 0:
                return ["outliers_detected"]
    return []


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _eda_summary(result: dict[str, Any]) -> str:
    """从 EDA 结果里提取关键信息生成人话摘要。

    旧实现用 ``self.description``（工具说明）当 summary，结果是「描述性统计：
    数值列均值/分位数…」这句废话 —— 模型只拿到描述，拿不到这次到底算出了什么。
    这里按各分析器的返回结构提取关键数字，让答案有据可依。
    """
    parts: list[str] = []
    # 异常值检测：columns[i].outlier_count 求和
    cols = result.get("columns")
    if isinstance(cols, list) and cols and isinstance(cols[0], dict):
        with_count = [
            c for c in cols
            if isinstance(c, dict) and int(c.get("outlier_count") or 0) > 0
        ]
        if with_count:
            total = sum(int(c.get("outlier_count") or 0) for c in with_count)
            parts.append(f"{len(with_count)} 列检出异常，共 {total} 个异常值")
    # 单列分布：column + mean/std
    if isinstance(result.get("column"), str):
        col = result["column"]
        stat = []
        if result.get("mean") is not None:
            stat.append(f"均值 {_fmt(result['mean'])}")
        if result.get("std") is not None:
            stat.append(f"标准差 {_fmt(result['std'])}")
        parts.append(f"{col}：{'，'.join(stat)}" if stat else f"{col} 的分布")
    # 相关性：matrix 是 dict
    if isinstance(result.get("matrix"), dict):
        mcols = result.get("columns") or list(result["matrix"].keys())
        parts.append(f"{len(mcols)} 个数值列的相关性矩阵")
        # ★ 二分类列被编码后必须写进 summary：否则答案会说「Churn 与 tenure 的
        #   相关性」，却不提它是按 Yes=1 编码算出来的，读者无从判断符号含义。
        encoded = result.get("binary_encoded")
        if isinstance(encoded, dict) and encoded:
            parts.append(
                "二分类列已编码参与："
                + "、".join(f"{k}（{v}）" for k, v in encoded.items())
            )
    # 行数
    if isinstance(result.get("row_count"), int):
        parts.append(f"{result['row_count']} 行")
    return "；".join(parts) if parts else "分析完成"


class _EdaTool(Tool):
    """公共：加载数据 -> EDA 模块分析。"""

    category = "eda"
    permission = "analyze_data"
    module = None  # EdaModule 子类

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        version = params.get("version")
        df = ds.load_version(dataset_id, int(version) if version else None)
        options = {
            k: v for k, v in params.items() if k not in ("dataset_id", "version")
        }
        result = self.module().analyze(df, **options)
        # 结构化 signals（任务 7）：从分析结果里提取可被 DecisionProvider 消费的标记。
        signals = _extract_signals(result)
        return ToolResult.ok(result, summary=_eda_summary(result), signals=signals)


class EdaDescribeTool(_EdaTool):
    name = "eda.describe"
    description = "描述性统计：数值列均值/分位数、类别列频次 Top。"
    module = DescriptiveAnalyzer
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}


class EdaDistributionTool(_EdaTool):
    name = "eda.distribution"
    description = "单列分布：数值直方分桶 / 类别频次占比。"
    module = DistributionAnalyzer
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "column": {"type": "string"},
            "bins": {"type": "integer", "default": 20},
            "top_n": {"type": "integer", "default": 10},
        },
        "required": ["dataset_id", "column"],
    }
    output_schema = {"type": "object"}


class EdaDistributionOverviewTool(_EdaTool):
    """数据集级分布总览（「看看这批数据的分布」的正确落点，无需 column）。"""

    name = "eda.distribution_overview"
    description = (
        "数据集级分布总览：一次给出全表数值列（均值/中位数/偏态/直方图摘要）"
        "与类别列（Top 类别/唯一值数）的分布概况，无需指定列。"
    )
    module = DistributionOverviewAnalyzer
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "bins": {"type": "integer", "default": 10},
            "top_n": {"type": "integer", "default": 10},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}


class EdaCorrelationTool(_EdaTool):
    name = "eda.correlation"
    description = "数值列相关性矩阵（Pearson/Spearman 自动选择）。"
    module = CorrelationAnalyzer
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "method": {"type": "string", "enum": ["auto", "pearson", "spearman"]},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}


class EdaOutlierTool(_EdaTool):
    name = "eda.outlier"
    description = "数值列异常检测（IQR / Z-Score）。"
    module = EdaOutlierAnalyzer
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "method": {"type": "string", "enum": ["iqr", "zscore"]},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}


def _chart_artifact(data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """把 VisualizationBuilder 的结果渲染成可直接展示的 SVG 图表。

    ★ 历史缺陷：``eda.visualize`` 只返回 ``{chart, column, x, y}`` 这类**数据**，
      答案层只能把它翻译成一段文字描述（「最低箱是…，计数 284」），
      前端 ChatPanel 又只渲染纯文本 —— 于是用户要的图**从头到尾没有出现过**，
      这就是「有图看不到」。

    这里复用报告模块已有的 ``chart_svg.to_svg``（同一套图表字典格式），
    把图真正画出来并挂在 ``metadata["chart"]`` 上，由引擎带进会话历史，
    前端在对话框里内联渲染。

    渲染不出来时**必须如实上报**（``rendered=False`` + ``reason``），
    绝不能假装已经生成 —— 那比不出图更糟，用户会以为是自己的浏览器坏了。
    """
    from app.reports.chart_svg import _chart_title, to_svg  # 延迟导入：避免工具层与报告层循环依赖

    kind = str(data.get("chart") or params.get("chart") or "")
    spec = {**data, "chart": kind}
    try:
        title = _chart_title(spec)
    except Exception:  # noqa: BLE001 - 标题只是修饰
        title = kind
    try:
        svg = to_svg(spec)
    except Exception as exc:  # noqa: BLE001 - 画图失败不该让整次运行失败
        return {
            "kind": kind,
            "title": title,
            "svg": "",
            "rendered": False,
            "reason": f"图表渲染失败：{exc}",
        }
    return {"kind": kind, "title": title, "svg": svg, "rendered": bool(svg)}


def _is_numeric_dtype(dtype: Any) -> bool:
    name = str(dtype or "").lower()
    return name.startswith(("int", "uint", "float", "decimal"))


#: 图表类型别名。槽位抽取和 LLM 都会吐出缩写或中文名，而 VisualizationBuilder
#: 只认枚举里那一个英文词 —— 「箱线图」被抽成 ``box`` 后直接撞上
#: 「不支持的图表类型：box」，用户完全不知道自己哪里说错了（第十轮实测）。
#: 归一化后照常画图，并在 warnings 里留痕。
_CHART_ALIASES: dict[str, str] = {
    "hist": "histogram", "直方图": "histogram", "histogram": "histogram",
    "bar": "bar", "柱状图": "bar", "柱图": "bar", "bar_chart": "bar",
    "box": "boxplot", "箱线图": "boxplot", "箱型图": "boxplot", "box_plot": "boxplot",
    "scatter": "scatter", "散点图": "scatter", "scatter_plot": "scatter",
    "line": "line", "折线图": "line", "line_chart": "line",
    "heatmap": "heatmap", "热力图": "heatmap", "相关矩阵图": "heatmap",
    "qq": "qq", "qq图": "qq", "qq_plot": "qq",
    "grouped_bar": "grouped_bar", "分组柱状图": "grouped_bar", "分组柱图": "grouped_bar",
    "area": "area", "面积图": "area",
    # 饼图没有渲染实现（chart_svg 里没有对应 builder）。与其抛「不支持的图表类型」
    # 让用户自己猜该换什么，不如降级成柱状图 —— 占比从频次里一样读得出来，
    # 同时如实标注降级。
    "pie": "bar", "饼图": "bar", "pie_chart": "bar",
}

#: 只需要一个主列的图表类型
_SINGLE_COLUMN_CHARTS = ("histogram", "bar", "area", "qq", "boxplot")

#: 列名里出现这些词元说明它是标识列 —— 给它画分布图毫无意义。
_ID_TOKENS = frozenset({"id", "ids", "code", "no", "num", "index", "key", "uuid", "编号", "序号", "主键"})


def _id_like(name: str) -> bool:
    """列名是不是标识列（customer_id / 序号 / order_no …）。"""
    lowered = str(name or "").lower()
    parts = re.split(r"[^a-z0-9\u4e00-\u9fff]+|(?<=[a-z0-9])(?=[A-Z])", lowered)
    return bool(_ID_TOKENS.intersection(p for p in parts if p))


def _auto_pick_column(chart: str, dtypes: dict[str, str], counts: dict[str, int] | None = None) -> str:
    """用户没说画哪一列时，按图表类型挑一个**说得通**的列。

    ★ 历史缺陷（第十轮实测）：「画个直方图看看」连续三轮挂起在
      「要对哪一列做这个分析？」—— 一个候选列都不给，用户只能自己猜列名。
      反问而不给选项，是「把本该系统承担的推理推回给用户」，
      与「模糊指令时基于已有字段给具体选项」的要求正好相反。

    挑列的规则：直方图/QQ/面积图要数值列，柱状图/箱线图优先类别列；
    两类都跳过标识列（给 customer_id 画直方图是纯噪音）。
    """
    numeric = [c for c, t in dtypes.items() if _is_numeric_dtype(t)]
    others = [c for c in dtypes if c not in numeric]
    numeric = [c for c in numeric if not _id_like(c)] or numeric
    others = [c for c in others if not _id_like(c)] or others
    if chart == "bar":
        pool = others or numeric
    else:
        pool = numeric or others
    # ★ 常量列（只有一个取值）画出来是一根孤零零的柱子 / 一个点，毫无信息量。
    #   实测数据集里 region 只有「华东」一个值，自动挑列正好挑中它 ——
    #   用户看到的图等于什么都没说。
    if counts:
        varying = [c for c in pool if counts.get(c, 2) > 1]
        pool = varying or pool
    return pool[0] if pool else ""


def _normalize_chart_params(
    params: dict[str, Any], dtypes: dict[str, str], counts: dict[str, int] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """按图表类型与**真实列类型**补齐/纠正参数。

    ★ 历史缺陷：用户说「画 region 和 monthly_charges 的关系图」，链路抽出
      ``chart=grouped_bar, column=monthly_charges``，工具回一句
      「该图表类型必须指定 y 轴字段」—— 两列都点名了，却一个都没落对位置。
      原因是分组柱状图的三个槽（分类列 / 数值列 / 分组列）必须由**类型**决定，
      不能靠用户说话的顺序猜。

    工具最清楚自己需要什么形状的参数，纠偏的责任在这里，不在通用抽取器。
    纠偏必须**留痕**（写进 warnings），不能悄悄改。
    """
    raw_chart = str(params.get("chart") or "").strip()
    chart = _CHART_ALIASES.get(raw_chart.lower(), raw_chart)
    fixed = dict(params)
    notes: list[str] = []
    if chart and chart != raw_chart:
        fixed["chart"] = chart
        if raw_chart.lower() in ("pie", "饼图", "pie_chart"):
            notes.append("饼图暂不支持渲染，已按各类别频次出柱状图（占比可从柱高读出）")
        else:
            notes.append(f"图表类型「{raw_chart}」已归一化为 {chart}")

    if chart in ("scatter", "line"):
        x, y = fixed.get("x"), fixed.get("y")
        # 只给了一列时无从补齐（需要两列），交给工具明确报错。
        if x and y and not _is_numeric_dtype(dtypes.get(x)) and _is_numeric_dtype(dtypes.get(y)):
            fixed["x"], fixed["y"] = y, x
            notes.append(f"x/y 与列类型不匹配，已交换为 x={y}、y={x}")
        return fixed, notes

    if chart == "grouped_bar":
        candidates = [fixed.get("column"), fixed.get("x"), fixed.get("y")]
        candidates = [c for c in candidates if c and c in dtypes]
        numeric = [c for c in candidates if _is_numeric_dtype(dtypes.get(c))]
        others = [c for c in candidates if not _is_numeric_dtype(dtypes.get(c))]
        value_col = fixed.get("y") if _is_numeric_dtype(dtypes.get(fixed.get("y"))) else ""
        if not value_col and numeric:
            value_col = numeric[0]
        cat_col = fixed.get("column") if not _is_numeric_dtype(dtypes.get(fixed.get("column"))) else ""
        if not cat_col and others:
            cat_col = others[0]
        if not (value_col and cat_col and value_col != cat_col):
            return fixed, notes
        group_by = fixed.get("group_by")
        if group_by and group_by != cat_col and group_by in dtypes:
            fixed["column"] = cat_col
            fixed["y"] = value_col
            fixed.pop("bins", None)
            return fixed, notes
        # ★ 只有一个分类维度时，分组柱状图画不出来 —— group_by 只能等于 column，
        #   而工具会直接报「分组字段与字段不能是同一列」（实测：「画 region 和
        #   monthly_charges 的关系图」连续两轮卡在这里）。
        #   「类别列 vs 数值列的关系」本来就该用**箱线图**：按类别分组看数值分布。
        fixed["chart"] = "boxplot"
        fixed["column"] = value_col
        fixed["group_by"] = cat_col
        fixed.pop("y", None)
        fixed.pop("bins", None)
        notes.append(
            f"分组柱状图需要两个分类维度，当前只有 {cat_col}；"
            f"已改用箱线图按 {cat_col} 分组看 {value_col} 的分布"
        )
        return fixed, notes

    if chart == "boxplot":
        # ★ 「画一个按 Contract 分组的箱线图」里 Contract 是**分类列**，
        #   它该是 group_by 而不是主列。按字面把分类列塞进 column，工具直接报
        #   「箱线图的列必须是数值列：Contract」，一张图都没出（Telco 压测实测）。
        column = fixed.get("column")
        group_by = fixed.get("group_by")
        if column and not _is_numeric_dtype(dtypes.get(column)):
            if group_by and _is_numeric_dtype(dtypes.get(group_by)):
                fixed["column"], fixed["group_by"] = group_by, column
                notes.append(f"{column} 是分类列，已改为按 {column} 分组看 {group_by} 的分布")
            else:
                picked = _auto_pick_column("boxplot", dtypes, counts)
                if picked and picked != column:
                    fixed["group_by"] = column
                    fixed["column"] = picked
                    notes.append(f"箱线图主列需为数值列，已改为按 {column} 分组看 {picked} 的分布")

    if chart in _SINGLE_COLUMN_CHARTS:
        column = fixed.get("column")
        if not column or column not in dtypes:
            picked = _auto_pick_column(chart, dtypes, counts)
            if picked:
                fixed["column"] = picked
                notes.append(f"未指定要画哪一列，已按 {picked} 绘制；要看其他列直接说列名即可")

    return fixed, notes


class EdaVisualizeTool(_EdaTool):
    name = "eda.visualize"
    description = (
        "生成图表数据：histogram/bar/area/qq 需要 column；line/scatter 需要 x,y；"
        "grouped_bar 需要 column(分类)+y(数值)+group_by(分类)；boxplot/heatmap 需要 column(s)。"
    )
    module = VisualizationBuilder
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "chart": {
                "type": "string",
                "enum": [
                    "histogram", "bar", "line", "scatter",
                    "boxplot", "heatmap", "qq", "grouped_bar", "area",
                ],
            },
            "column": {"type": "string", "description": "histogram/bar/qq/area/boxplot 主列；grouped_bar 的分类列"},
            "x": {"type": "string", "description": "line/scatter 的 x 列"},
            "y": {"type": "string", "description": "line/scatter/grouped_bar 的 y（数值）列"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "group_by": {"type": "string", "description": "grouped_bar 的分组列"},
            "agg": {
                "type": "string",
                "enum": ["mean", "sum", "count", "median"],
                "description": "grouped_bar 的聚合方式",
            },
            "bins": {"type": "integer", "default": 10},
            "top_n": {"type": "integer", "default": 20},
        },
        "required": ["dataset_id", "chart"],
    }
    output_schema = {"type": "object"}

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        # 图表参数是**类型敏感**的（散点图的 x/y 必须都是数值列、分组柱状图的
        # y 必须是数值列），所以这里要拿到 schema 再跑，而不是沿用父类的通用路径。
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        version = params.get("version")
        df = ds.load_version(dataset_id, int(version) if version else None)
        dtypes = {c: str(t) for c, t in df.schema.items()}
        # 各列唯一值个数（一次查询算完）：自动挑列时要跳过常量列。
        counts: dict[str, int] = {}
        try:
            row = df.select([pl.col(c).n_unique().alias(c) for c in dtypes]).row(0, named=True)
            counts = {c: int(v or 0) for c, v in row.items()}
        except Exception:  # noqa: BLE001 - 只是挑列的辅助信息，算不出就不过滤
            counts = {}

        normalized, notes = _normalize_chart_params(params, dtypes, counts)

        # 双列图表缺列时，工具原本只报「该图表类型必须指定x 轴字段（x）」——
        # 用户不知道「x 轴字段」是什么，更不知道该怎么改口。这里把话说成人话。
        chart_kind = str(normalized.get("chart") or "")
        if chart_kind in ("scatter", "line") and not (
            normalized.get("x") and normalized.get("y")
        ):
            label = "散点图" if chart_kind == "scatter" else "折线图"
            numeric_cols = [c for c, t in dtypes.items() if _is_numeric_dtype(t)][:8]
            return ToolResult.fail(
                f"{label}需要两个数值列（x 轴和 y 轴各一列）。"
                f"当前可用的数值列：{'、'.join(numeric_cols) or '（无）'}。"
                "可以直接说「画 tenure_months 和 monthly_charges 的散点图」。",
                data={"chart": chart_kind, "numeric_columns": numeric_cols},
            )

        options = {k: v for k, v in normalized.items() if k not in ("dataset_id", "version")}
        result_data = VisualizationBuilder().analyze(df, **options)
        result = ToolResult.ok(
            result_data,
            summary=_eda_summary(result_data) if isinstance(result_data, dict) else "",
            signals=_extract_signals(result_data),
        )
        for note in notes:
            result.warnings.append(note)

        if not isinstance(result.data, dict):
            return result
        artifact = _chart_artifact(result.data, normalized)
        result.metadata["chart"] = artifact
        if artifact["rendered"]:
            # summary 会直接进事实摘要（不受 _collect_facts 条数限制），
            # 图是否真的画出来必须写在这里，否则答案层无从知晓。
            result.summary = f"{result.summary}；图表已生成（{artifact['title']}）".strip("；")
        else:
            result.warnings.append(
                f"本次未能在对话中内嵌图表（{artifact.get('reason') or '未知原因'}）"
            )
        return result
