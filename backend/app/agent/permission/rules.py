"""Prompt 115：Risk Level 与 Tool-Risk 映射规则。"""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """操作风险等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 需要人工确认的风险等级（>= HIGH）
CONFIRMATION_RISKS = {RiskLevel.HIGH, RiskLevel.CRITICAL}


def risk_requires_confirmation(risk: RiskLevel) -> bool:
    return risk in CONFIRMATION_RISKS


# Tool name -> Risk Level 映射规则（注册表以此校验工具声明）
DEFAULT_TOOL_RISKS: dict[str, RiskLevel] = {
    # 只读：低风险
    "dataset.list": RiskLevel.LOW,
    "dataset.inspect": RiskLevel.LOW,
    "dataset.preview": RiskLevel.LOW,
    "dataset.schema": RiskLevel.LOW,
    "dataset.profile": RiskLevel.LOW,
    "dataset.quality": RiskLevel.LOW,
    "eda.describe": RiskLevel.LOW,
    "eda.distribution": RiskLevel.LOW,
    "eda.correlation": RiskLevel.LOW,
    "eda.outlier": RiskLevel.LOW,
    "eda.visualize": RiskLevel.LOW,
    "ml.detect_task": RiskLevel.LOW,
    "ml.evaluate": RiskLevel.LOW,
    "ml.compare": RiskLevel.LOW,
    "ml.explain": RiskLevel.LOW,
    # 新增工具（含调参手册 / 推理参数说明）常忘记登记 —— 与 `ml.explain` 同为只读查询，LOW。
    "ml.explain_config": RiskLevel.LOW,
    # 修改数据（产生新版本）：中风险
    "data.filter": RiskLevel.MEDIUM,
    "data.transform": RiskLevel.MEDIUM,
    "data.aggregate": RiskLevel.MEDIUM,
    "ml.prepare": RiskLevel.MEDIUM,
    "ml.predict": RiskLevel.MEDIUM,
    # 破坏性/重操作：高风险
    "data.clean": RiskLevel.HIGH,
    "ml.train": RiskLevel.HIGH,
    # 跨数据集合并：高风险
    "data.merge": RiskLevel.HIGH,
    # Workflow 编排：创建实体中风险（工具自身 requires_confirmation），
    # 执行编排高风险（会级联触发数据处理/训练节点）
    "workflow.list": RiskLevel.LOW,
    "workflow.inspect": RiskLevel.LOW,
    "workflow.create": RiskLevel.MEDIUM,
    "workflow.run": RiskLevel.HIGH,
    "workflow.build_and_run": RiskLevel.HIGH,
    # 报告：只读分析 + 产出报告文件，低风险
    "report.generate": RiskLevel.LOW,
}
