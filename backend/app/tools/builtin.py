"""内置 Agent 工具注册。"""
from __future__ import annotations

from app.tools.data_tools import DataAggregateTool, DataCleanTool, DataFilterTool, DataMergeTool, DataTransformTool
from app.tools.dataset_tools import DatasetInspectTool, DatasetListTool, DatasetPreviewTool, DatasetProfileTool, DatasetQualityTool, DatasetSchemaTool
from app.tools.eda_tools import EdaCorrelationTool, EdaDescribeTool, EdaDistributionTool, EdaOutlierTool, EdaVisualizeTool
from app.tools.ml_tools import MlCompareTool, MlDetectTaskTool, MlEvaluateTool, MlExplainConfigTool, MlExplainTool, MlPredictTool, MlPrepareTool, MlTrainTool
from app.tools.workflow_tools import WorkflowBuildAndRunTool, WorkflowCreateTool, WorkflowInspectTool, WorkflowListTool, WorkflowRunTool
from app.tools.report_tools import ReportGenerateTool
from app.tools.registry import TOOL_REGISTRY

_BUILTIN_TOOLS = (
    DatasetListTool, DatasetInspectTool, DatasetPreviewTool, DatasetSchemaTool, DatasetProfileTool, DatasetQualityTool,
    DataFilterTool, DataCleanTool, DataTransformTool, DataAggregateTool, DataMergeTool,
    EdaDescribeTool, EdaDistributionTool, EdaCorrelationTool, EdaOutlierTool, EdaVisualizeTool,
    MlDetectTaskTool, MlPrepareTool, MlTrainTool, MlPredictTool, MlEvaluateTool, MlCompareTool, MlExplainTool,
    MlExplainConfigTool,
    WorkflowListTool, WorkflowCreateTool, WorkflowInspectTool, WorkflowRunTool, WorkflowBuildAndRunTool,
    ReportGenerateTool,
)

_REGISTERED = False

def register_builtin_tools() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for tool_cls in _BUILTIN_TOOLS:
        TOOL_REGISTRY.register(tool_cls())
    _REGISTERED = True

register_builtin_tools()
