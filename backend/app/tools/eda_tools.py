"""Prompt 103-107：EDA 分析工具（只读）。

eda.describe / eda.distribution / eda.correlation / eda.outlier / eda.visualize。
直接复用 Phase 4 的 EDA 模块，工具层只做参数与权限编排。
"""

from __future__ import annotations

from typing import Any

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
    if "outliers" in result or "outlier_count" in result:
        return ["outliers_detected"]
    return []


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
        # 例如 distribution_overview 会产出 high_missing / high_skew，驱动「下一步
        # 是否要深入缺失/异常分析」。不是自然语言，是机器可读信号。
        signals = _extract_signals(result)
        return ToolResult.ok(result, summary=f"{self.description}", signals=signals)


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
