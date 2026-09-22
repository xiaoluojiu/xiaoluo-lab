"""本地 AI Router：把「小洛实验室」的真实能力封装成可训练、可校验的封闭契约。

只暴露契约层。推理实现（本地模型接入）与数据构建（离线脚本）分别在
后续阶段落到本包内与 `scripts/router/`，都只依赖这里定义的类型。
"""

from app.local_router.escalation_rules import detect_escalation
from app.local_router.contract import (
    EscalationReason,
    Intent,
    RouteAction,
    RouterDecision,
    RouterRequest,
    ValidationOutcome,
    category_consistency_report,
    decide_route,
    decision_json_schema,
    input_schema,
    intent_of_tool,
    known_params,
    optional_params,
    param_type,
    required_params,
    tool_label_space,
    tool_spec,
    tool_specs,
    tools_of_intent,
    validate_decision,
)

__all__ = [
    "EscalationReason",
    "Intent",
    "RouteAction",
    "RouterDecision",
    "RouterRequest",
    "ValidationOutcome",
    "category_consistency_report",
    "decide_route",
    "detect_escalation",
    "decision_json_schema",
    "input_schema",
    "intent_of_tool",
    "known_params",
    "optional_params",
    "param_type",
    "required_params",
    "tool_label_space",
    "tool_spec",
    "tool_specs",
    "tools_of_intent",
    "validate_decision",
]
