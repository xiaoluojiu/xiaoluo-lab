"""本地 Router 的**三层决策合成** —— 线上推理与离线评测共用这一个实现。

三层顺序（即真实调用顺序，改动顺序等于改动线上行为）
----------------------------------------------------
| 顺序 | 决策 | 谁来做 | 依据 |
| --- | --- | --- | --- |
| ① | **该不该升级** | `escalation_rules.detect_escalation()` | 结构/词表信号；规则 F1 **98.2%** ≫ 学习版 51.3% |
| ② | **选哪个工具 / 是否闲聊** | `model.LexicalRouterModel` | 唯一真正需要语义的部分 |
| ③ | **工具已定但缺必填槽位** | 本模块的纯函数 | `required_params ∖ resolved` 是符号计算，学习版 F1 仅 8.9~47% |

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

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "PARAM_EXTRACTION_IMPLEMENTED",
    "RULE_RESOLVABLE",
    "ConstantModel",
    "decision_to_route",
    "route_request",
]

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


def route_request(
    req: Any,
    *,
    model: Any = _UNSET,
    confidence_threshold: float | None = None,
) -> RouterDecision:
    """一次完整的本地路由决策。

    `model` 语义：不传 ⇒ 用进程级模型（`model.get_model()`）；
    传 `ConstantModel(...)` ⇒ 注入固定预测（离线评测）；
    传 `None` ⇒ 模拟「模型不可用」，走保守升级。
    """
    request = _as_request(req)
    threshold = _default_threshold() if confidence_threshold is None else float(confidence_threshold)

    # ① L0 结构规则：零成本、零参数，先于模型。命中即升级，intent 留空（尚未判定能力域）。
    reason = detect_escalation(request.utterance)
    if reason is not None:
        return RouterDecision(escalate=True, escalate_reason=reason, confidence=1.0)

    # ② L1 模型：选工具 / 判闲聊
    active = get_model() if model is _UNSET else model
    if active is None:
        # 模型缺失/过期：保守升级，绝不猜（宁多花一次远程调用，不要静默做错）。
        return RouterDecision(
            escalate=True, escalate_reason=EscalationReason.AMBIGUOUS, confidence=0.0
        )

    raw_label, raw_confidence = active.predict_for(request)
    label = _normalize_label(raw_label)
    confidence = float(raw_confidence)
    if not label:
        # L1 未表态属异常：同样保守升级。
        return RouterDecision(
            escalate=True, escalate_reason=EscalationReason.AMBIGUOUS, confidence=confidence
        )

    tool = tool_of_label(label)
    decision = RouterDecision(
        intent=Intent.CHAT if tool is None else (C.intent_of_tool(tool) or Intent.CHAT),
        tool=tool,
        confidence=confidence,
    )

    # ③ 反问规则（确定性）
    if tool is not None:
        decision.missing = _missing_by_rule(request, tool)

    outcome = validate_decision(decision)
    if not PARAM_EXTRACTION_IMPLEMENTED:
        # ⚠️ validate_decision 的 missing_required 由 decision.params 推出，而 params 现在是空的
        # （不抽取）。若不覆盖，任何带必填参数的工具都会被判成「缺槽位」而反问用户，
        # 与离线评测口径（只认 dataset_id）不一致，线上会显著更差且更容易疲劳用户。
        # ⇒ 用规则口径覆盖，并保留 decision.missing 作为对外字段。
        # 参数抽取落地后，删除这两行。
        outcome.missing_required = decision.missing

    action, detail = decide_route(outcome, confidence=confidence, confidence_threshold=threshold)
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
