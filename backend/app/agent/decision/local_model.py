"""LocalModelDecisionProvider —— 用本地 Router 模型做工具选择决策。

复用仓库已有的 ``app.local_router``（TF-IDF + LinearSVC 词法模型，三层合成：
L0 升级规则 → L1 工具选择 → 反问规则）。**不新造一套模型**，只把它包装成
DecisionProvider 的 ``local_router`` / ``local_model`` 来源。

关键诚实性约束（任务 18.8 / 18.11）：
* 模型不可用（缺失 / 过期 / 加载失败）时**不得静默吞错**，必须显式返回
  ``confidence=0.0`` 的保守升级决策，并在 reasons 里写明原因；
* 不把「接入接口」当成「模型真正参与运行」—— 这里真正调用 ``route_request``，
  拿真实模型的预测，而非返回一个假置信度。
"""

from __future__ import annotations

from app.agent.decision.provider import (
    DecisionAction,
    DecisionContext,
    DecisionProvider,
    DecisionSource,
    AgentDecision,
)


class LocalModelDecisionProvider(DecisionProvider):
    """本地 Router 模型决策：把用户语言映射到「工具 + 置信度 + 候选」。"""

    def decide(self, context: DecisionContext) -> AgentDecision:
        from app.local_router.router import decision_to_route, route_request

        try:
            decision = route_request(
                {
                    "utterance": context.user_request,
                    "bound_dataset_id": context.bound_dataset_id,
                    "available_columns": context.available_columns,
                    "recent_tools": context.recent_tools,
                }
            )
        except Exception as exc:  # noqa: BLE001 — 本地模型异常不能静默吞掉
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=DecisionSource.LOCAL_ROUTER,
                confidence=0.0,
                reasons=[f"本地 Router 推理异常：{type(exc).__name__}: {exc}"],
            )

        route = decision_to_route(decision)
        confidence = float(decision.confidence)

        # 升级：L0 规则或模型置信度不足 / 模型不可用。
        if decision.escalate:
            reason = decision.escalate_reason.value if decision.escalate_reason else "unknown"
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=DecisionSource.LOCAL_ROUTER,
                confidence=confidence,
                reasons=[f"本地 Router 判定升级：{reason}"],
                evidence={"route": route, "escalate_reason": reason},
            )

        # 反问：工具已定但缺必填槽位。
        if decision.missing:
            return AgentDecision(
                action=DecisionAction.ASK_USER,
                tool=decision.tool,
                source=DecisionSource.LOCAL_ROUTER,
                confidence=confidence,
                reasons=[f"工具 {decision.tool} 缺少必填参数：{', '.join(decision.missing)}"],
                evidence={"route": route, "missing": list(decision.missing)},
            )

        # 闲聊。
        if decision.tool is None:
            return AgentDecision(
                action=DecisionAction.CHAT,
                source=DecisionSource.LOCAL_ROUTER,
                confidence=confidence,
                reasons=["本地 Router 判定为闲聊"],
                evidence={"route": route},
            )

        # 执行工具。
        return AgentDecision(
            action=DecisionAction.EXECUTE_TOOL,
            tool=decision.tool,
            source=DecisionSource.LOCAL_ROUTER,
            confidence=confidence,
            reasons=[f"本地 Router 选择工具 {decision.tool}"],
            evidence={"route": route, "confidence": confidence},
        )
