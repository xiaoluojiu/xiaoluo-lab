"""本地 Router 的**三层决策合成** —— 线上推理与离线评测共用这一个实现。

三层顺序（即真实调用顺序，改动顺序等于改动线上行为）
----------------------------------------------------
| 顺序 | 决策 | 谁来做 | 依据 |
| --- | --- | --- | --- |
| ① | **该不该升级** | `escalation_rules.detect_escalation()` | 结构/词表信号；规则 F1 **98.2%** ≫ 学习版 51.3% |
| ② | **选哪个工具 / 判哪种走法** | `qwen.QwenRouterModel`（Qwen3-0.6B + LoRA） | 语义理解；受 `LOCAL_ROUTER_MODE` 控制 |
| ③ | 同上（② 未命中时的基线） | `model.LexicalRouterModel` | TF-IDF + LinearSVC，部署态 route 80.7% |
| ④ | **工具已定但缺必填槽位** | 本模块的纯函数 | `required_params ∖ resolved` 是符号计算，学习版 F1 仅 8.9~47% |

② 与 ③ 是**同一层的两种实现**，不是两个层：Qwen 命中就用 Qwen，不命中
（未启用 / 依赖缺失 / 权重缺失 / 超时 / 解析失败 / 工具不在注册表）就退回词法。
两段产出都送进同一个 `_finalize()` 过结构校验，再往下游走。

`L0 → L1 → 反问` 的顺序不可调换：升级必须最先判定，否则会把「平台根本做不了的事」
喂给一个只会选工具的小模型，得到一个**看起来有把握的错误工具**。

两个诚实标注（不要当成实现完整）
--------------------------------
1. `PARAM_EXTRACTION_IMPLEMENTED = False`：**参数抽取尚未实现**。所以「缺哪些必填槽位」
   目前只能由规则可判定的 `RULE_RESOLVABLE` 给出；其余必填参数**假设由下游补齐**
   （现状是 LLM planner 抽取）。这一行是离线评测口径（80.7%）与线上行为一致的前提，
   参数抽取落地后必须改回由 `decision.params` 驱动，否则线上会比评测**看起来更好**。
2. `confidence_threshold` 默认 **0.0（不门控）**，与离线评测口径一致。
   打开门控（如 0.55）会让低置信请求转升级 —— 这是**行为变更**，
   必须靠 `scripts/router/analyze_fusion.py` 的覆盖率-准确率曲线来选值，而不是拍脑袋。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.local_router import contract as C
from app.local_router.contract import (
    EscalationReason,
    Intent,
    RouterDecision,
    RouterRequest,
    decide_route,
    validate_decision,
)
from app.local_router.escalation_rules import detect_escalation
from app.local_router.model import CALL_PREFIX, CHAT_LABEL, get_model, tool_of_label
from app.local_router.qwen import QWEN_SOURCE, get_model as get_qwen_model
from app.local_router.qwen import unavailable_reason as qwen_unavailable_reason
from app.local_router.qwen_protocol import (
    QwenDecision,
    build_messages,
    parse_decision,
    to_router_decision,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "PARAM_EXTRACTION_IMPLEMENTED",
    "ROUTE_SOURCE_LEXICAL",
    "ROUTE_SOURCE_QWEN",
    "ROUTE_SOURCE_RULE",
    "RULE_RESOLVABLE",
    "ConstantModel",
    "RouteOutcome",
    "decision_to_route",
    "route_request",
    "route_request_detailed",
]

#: 一次路由最终由谁拍板。进 DecisionTrace，用来回答「这条是 Qwen 判的还是词法判的」。
ROUTE_SOURCE_QWEN = QWEN_SOURCE
ROUTE_SOURCE_LEXICAL = "local_router"
ROUTE_SOURCE_RULE = "rule"

# 规则能**无条件判定**的槽位：RouterRequest 里直接读得出「有没有」。
# 与 `contract.decide_route` 的职责一致 —— 它是「反问用户」的依据，不是升级依据。
RULE_RESOLVABLE = {"dataset_id"}

# ⚠️ 见模块 docstring 第 1 条。改成 True 之前必须先实现参数抽取。
PARAM_EXTRACTION_IMPLEMENTED = False

DEFAULT_CONFIDENCE_THRESHOLD = 0.0

_UNSET = object()


def _default_threshold() -> float:
    """从配置读门控阈值；配置层不可用时退回不门控。

    延迟 + 容错 import：离线脚本（`.venv-l1`）也会调用本模块，
    不该因为少一个 pydantic-settings 就整条链路失败。
    """
    try:
        from app.core.config import settings

        return float(getattr(settings, "LOCAL_ROUTER_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))
    except Exception:  # noqa: BLE001
        return DEFAULT_CONFIDENCE_THRESHOLD


def _as_request(req: Any) -> RouterRequest:
    """接受 dict 或 RouterRequest（离线样本里 request 是 dict）。"""
    if isinstance(req, RouterRequest):
        return req
    if not isinstance(req, dict):
        return RouterRequest(utterance=str(req))
    allowed = set(RouterRequest.model_fields)
    return RouterRequest(**{k: v for k, v in req.items() if k in allowed})


def _normalize_label(label: str | None) -> str | None:
    """`ask::<tool>` 按 `call::<tool>` 处理（容错，与 assemble_route 的历史行为一致）：
    是否反问由第 ③ 层规则决定，不由上游标签决定，否则会重复套用规则。"""
    if not label:
        return None
    text = str(label)
    if text.startswith("ask::"):
        return CALL_PREFIX + text[len("ask::"):]
    return text


def _missing_by_rule(req: RouterRequest, tool: str) -> list[str]:
    """规则口径下「工具已定但缺必填槽位」。

    只认 `RULE_RESOLVABLE`：其他必填参数的缺失在**当前实现下无法判定**
    （既不解析用户话里的值，也就不知道有没有给），只能假设下游补齐。
    """
    resolved = {"dataset_id"} if req.bound_dataset_id is not None else set()
    return [p for p in C.required_params(tool) if p not in resolved and p in RULE_RESOLVABLE]


@dataclass
class ConstantModel:
    """固定输出的假模型 —— 让离线评测能把「交叉验证折外预测」注入同一条合成路径。

    有了它，评测跑的就是**线上那段代码**，而不是一个长得像它的副本。
    """

    label: str | None
    confidence: float = 0.0

    def predict_for(self, req: Any) -> tuple[str | None, float]:  # noqa: ARG002 — 与真模型同签名
        return self.label, self.confidence


@dataclass
class RouteOutcome:
    """一次路由的完整结果：决策 + **谁拍板** + Qwen 的原始判定。

    为什么要多带这两个字段：DecisionTrace 与 shadow trace 必须能回答
    「这条判定是 Qwen 给的、词法给的，还是 L0 规则直接拦下的」——
    否则积累再多真实数据也分不清该优化哪一层。
    """

    decision: RouterDecision
    source: str = ROUTE_SOURCE_LEXICAL
    qwen: QwenDecision | None = None
    #: Qwen 不可用 / 失败的原因。**与「Qwen 判了 escalate」是两回事**：
    #: 前者说明这一层本次不存在（应当降级），后者是这一层的有效结论（应当升级）。
    qwen_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "qwen": self.qwen.to_dict() if self.qwen is not None else None,
            "qwen_error": self.qwen_error,
        }


def _qwen_mode() -> str:
    """当前档位。配置层不可用时按 `off` —— 与词法路径「读不到配置就退回默认」同款容错。"""
    try:
        from app.core.config import settings

        return settings.local_router_mode
    except Exception:  # noqa: BLE001
        return "off"


def _qwen_runtime() -> tuple[int, float, int]:
    """`(max_new_tokens, timeout_s, max_datasets)`，全部夹紧到安全区间。"""
    try:
        from app.core.config import settings

        return (
            max(16, int(settings.LOCAL_ROUTER_MAX_NEW_TOKENS)),
            max(0.05, float(settings.LOCAL_ROUTER_TIMEOUT_MS) / 1000.0),
            max(0, int(settings.LOCAL_ROUTER_MAX_CONTEXT_DATASETS)),
        )
    except Exception:  # noqa: BLE001
        return 160, 0.8, 30


def _route_with_qwen(
    request: RouterRequest, *, record_only: bool
) -> tuple[RouterDecision | None, QwenDecision | None, str | None]:
    """跑一次 Qwen。返回 `(可用决策, 原始判定, 失败原因)`。

    `record_only=True`（shadow 档）时**永远不返回可用决策**，只把原始判定带回去
    落 trace —— 这是「shadow 对线上零行为改动」的实现处，不是靠调用方自觉。
    """
    model = get_qwen_model()
    if model is None:
        return None, None, qwen_unavailable_reason() or "Qwen 模型不可用"

    max_new_tokens, timeout_s, max_datasets = _qwen_runtime()
    raw, error = model.predict(
        build_messages(request, max_datasets=max_datasets),
        max_new_tokens=max_new_tokens,
        timeout_s=timeout_s,
    )
    if error is not None:
        return None, None, error
    parsed = parse_decision(raw or "")
    if parsed is None:
        return None, None, "模型输出无法解析为合法决策"
    if record_only:
        return None, parsed, None
    decision = to_router_decision(parsed, request)
    if decision is None:
        return None, parsed, "模型给出的工具不在当前注册表"
    return decision, parsed, None


def _finalize(
    decision: RouterDecision, request: RouterRequest, threshold: float
) -> RouterDecision:
    """③ 反问规则 + 确定性校验 + 置信度门控。

    Qwen 路径与词法路径**共用这一段**：无论哪一层的产出，都要过同一套结构校验
    才能往下走 —— 这是「模型输出不能直接执行」的第一道闸门。
    """
    confidence = float(decision.confidence)
    if decision.tool is not None:
        # 合并而不是覆盖：Qwen 的 clarification 会自带 missing，而规则能判定
        # dataset_id 是否缺失。覆盖任一边都会漏掉一类反问理由。
        decision.missing = list(decision.missing) + [
            item
            for item in _missing_by_rule(request, decision.tool)
            if item not in decision.missing
        ]

    outcome = validate_decision(decision)
    if not PARAM_EXTRACTION_IMPLEMENTED:
        # ⚠️ 见模块 docstring 第 1 条。改成 True 之前必须先实现参数抽取。
        outcome.missing_required = decision.missing

    action, _detail = decide_route(outcome, confidence=confidence, confidence_threshold=threshold)
    if action == "escalate":
        # 「结构不合格 / 置信度不足」都归到 ambiguous：现有五类原因是给**结构信号**用的，
        # 这两类不是「用户说的事超出能力」，而是「这一次没把握」。
        decision.escalate = True
        decision.escalate_reason = EscalationReason.AMBIGUOUS
        decision.tool = None
        decision.missing = []
    elif action == "ask_user":
        decision.missing = list(outcome.missing_required)
    decision.confidence = confidence
    return decision


def route_request_detailed(
    req: Any,
    *,
    model: Any = _UNSET,
    confidence_threshold: float | None = None,
) -> RouteOutcome:
    """一次完整的本地路由决策，连「谁拍板」一起返回。

    `model` 语义：不传 ⇒ 用进程级模型；传 `ConstantModel(...)` ⇒ 注入固定预测
    （离线评测，此时**不跑 Qwen**，保证评测口径与历史完全一致）；传 `None` ⇒
    模拟「模型不可用」，走保守升级。
    """
    request = _as_request(req)
    threshold = _default_threshold() if confidence_threshold is None else float(confidence_threshold)

    # ① L0 结构规则：零成本、零参数，先于模型。命中即升级，intent 留空（尚未判定能力域）。
    reason = detect_escalation(request.utterance)
    if reason is not None:
        return RouteOutcome(
            RouterDecision(escalate=True, escalate_reason=reason, confidence=1.0),
            ROUTE_SOURCE_RULE,
        )

    # ② L1 神经路由（Qwen）：命中就直接用，不命中再退回词法。
    #    仅在「没有显式注入模型」时参与 —— 离线评测注入 ConstantModel 是要精确控制
    #    词法层的预测，此时再跑一次 Qwen 会让评测口径漂移。
    qwen_decision: QwenDecision | None = None
    qwen_error: str | None = None
    if model is _UNSET:
        mode = _qwen_mode()
        if mode in ("shadow", "active"):
            qwen_route, qwen_decision, qwen_error = _route_with_qwen(
                request, record_only=(mode == "shadow")
            )
            if qwen_route is not None:
                return RouteOutcome(
                    _finalize(qwen_route, request, threshold),
                    ROUTE_SOURCE_QWEN,
                    qwen_decision,
                )

    # ③ L1 词法模型：选工具 / 判闲聊（Qwen 不可用时的既有基线）
    active = get_model() if model is _UNSET else model
    if active is None:
        # 模型缺失/过期：保守升级，绝不猜（宁多花一次远程调用，不要静默做错）。
        return RouteOutcome(
            RouterDecision(
                escalate=True, escalate_reason=EscalationReason.AMBIGUOUS, confidence=0.0
            ),
            ROUTE_SOURCE_LEXICAL,
            qwen_decision,
            qwen_error,
        )

    raw_label, raw_confidence = active.predict_for(request)
    label = _normalize_label(raw_label)
    confidence = float(raw_confidence)
    if not label:
        # L1 未表态属异常：同样保守升级。
        return RouteOutcome(
            RouterDecision(
                escalate=True, escalate_reason=EscalationReason.AMBIGUOUS, confidence=confidence
            ),
            ROUTE_SOURCE_LEXICAL,
            qwen_decision,
            qwen_error,
        )

    tool = tool_of_label(label)
    decision = RouterDecision(
        intent=Intent.CHAT if tool is None else (C.intent_of_tool(tool) or Intent.CHAT),
        tool=tool,
        confidence=confidence,
    )
    return RouteOutcome(
        _finalize(decision, request, threshold),
        ROUTE_SOURCE_LEXICAL,
        qwen_decision,
        qwen_error,
    )


def route_request(
    req: Any,
    *,
    model: Any = _UNSET,
    confidence_threshold: float | None = None,
) -> RouterDecision:
    """一次完整的本地路由决策（只要决策本身；来源信息见 `route_request_detailed`）。"""
    return route_request_detailed(
        req, model=model, confidence_threshold=confidence_threshold
    ).decision


def decision_to_route(decision: RouterDecision) -> str:
    """把决策压成离线口径的 route 串：`chat` / `call::<tool>` / `ask::<tool>` / `escalate::<reason>`。

    这是**评测口径的唯一定义**（`eval_harness.route_label` 生成金标用的是同一套形状）。
    离线分析全部走它，因此线上线与离线永远可比。
    """
    if decision.escalate:
        reason = decision.escalate_reason.value if decision.escalate_reason else EscalationReason.AMBIGUOUS.value
        return "escalate::" + reason
    if decision.tool is None:
        return CHAT_LABEL
    if decision.missing:
        return "ask::" + decision.tool
    return "call::" + decision.tool
