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
from app.local_router.qwen import (
    QWEN_SOURCE,
    QwenRouterModel,
    available as qwen_available,
    get_model as get_qwen_model,
    reset_model_cache as reset_qwen_cache,
    unavailable_reason as qwen_unavailable_reason,
)
from app.local_router.qwen_protocol import (
    QWEN_MODES,
    QwenDecision,
    QwenMode,
    build_messages,
    parse_decision,
    to_router_decision,
)
from app.local_router.router import (
    ROUTE_SOURCE_LEXICAL,
    ROUTE_SOURCE_QWEN,
    ROUTE_SOURCE_RULE,
    RouteOutcome,
    route_request_detailed,
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
    # ---- Qwen 神经路由 ----
    "QWEN_MODES",
    "QWEN_SOURCE",
    "ROUTE_SOURCE_LEXICAL",
    "ROUTE_SOURCE_QWEN",
    "ROUTE_SOURCE_RULE",
    "QwenDecision",
    "QwenMode",
    "QwenRouterModel",
    "RouteOutcome",
    "build_messages",
    "get_qwen_model",
    "parse_decision",
    "qwen_available",
    "qwen_unavailable_reason",
    "reset_qwen_cache",
    "route_request_detailed",
    "to_router_decision",
]
