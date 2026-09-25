"""LocalModelDecisionProvider —— 用本地 Router 模型做工具选择决策。

复用仓库已有的 ``app.local_router``（L0 升级规则 → L1 工具选择 → 反问规则）。
**不新造一套模型**，只把它包装成 DecisionProvider 的来源。

L1 现在是两种实现的合体
----------------------
* ``local_qwen``   —— Qwen3-0.6B + LoRA（语义理解，输出 6 类 mode + 工具 + 参数）
* ``local_router`` —— TF-IDF + LinearSVC（词法基线，部署态 route 80.7%）

Qwen 命中就用 Qwen，不命中自动退回词法；两层产出的都是同一个 ``RouterDecision``，
所以本文件**不需要为它们分别写分支**，只需要如实上报来源。

关键诚实性约束（任务 18.8 / 18.11）：
* 模型不可用（缺失 / 过期 / 加载失败 / 推理超时 / 输出解析失败）时**不得静默吞错**，
  必须显式返回 ``confidence=0.0`` 的保守升级决策，并在 reasons 里写明原因；
* 不把「接入接口」当成「模型真正参与运行」—— 这里真正调用
  ``route_request_detailed``，拿真实模型的预测，而非返回一个假置信度。
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
        from app.local_router.router import (
            ROUTE_SOURCE_QWEN,
            decision_to_route,
            route_request_detailed,
        )

        try:
            outcome = route_request_detailed(
                {
                    "utterance": context.user_request,
                    "bound_dataset_id": context.bound_dataset_id,
                    "bound_dataset_name": context.bound_dataset_name,
                    "available_datasets": context.available_datasets,
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

        decision = outcome.decision
        route = decision_to_route(decision)
        confidence = float(decision.confidence)
        # 如实上报这一条是 Qwen 判的还是词法判的 —— DecisionTrace 靠它分层评估。
        source = (
            DecisionSource.LOCAL_QWEN
            if outcome.source == ROUTE_SOURCE_QWEN
            else DecisionSource.LOCAL_ROUTER
        )
        evidence: dict = {"route": route, "router_source": outcome.source}
        # Qwen 的原始判定（含 steps / raw_reason）在契约里装不下，但对离线分析
        # 「模型到底想干什么」很有价值，一并带进 trace。
        if outcome.qwen is not None:
            evidence["qwen"] = outcome.qwen.to_dict()
        if outcome.qwen_error:
            evidence["qwen_error"] = outcome.qwen_error

        # 升级：L0 规则或模型置信度不足 / 模型不可用。
        if decision.escalate:
            reason = decision.escalate_reason.value if decision.escalate_reason else "unknown"
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=source,
                confidence=confidence,
                reasons=[f"本地 Router 判定升级：{reason}"],
                evidence={**evidence, "escalate_reason": reason},
            )

        # 反问：工具已定但缺必填槽位。
        if decision.missing:
            return AgentDecision(
                action=DecisionAction.ASK_USER,
                tool=decision.tool,
                source=source,
                confidence=confidence,
                reasons=[f"工具 {decision.tool} 缺少必填参数：{', '.join(decision.missing)}"],
                evidence={**evidence, "missing": list(decision.missing)},
            )

        # 闲聊。
        if decision.tool is None:
            return AgentDecision(
                action=DecisionAction.CHAT,
                source=source,
                confidence=confidence,
                reasons=["本地 Router 判定为闲聊"],
                evidence=evidence,
            )

        # 执行工具。
        return AgentDecision(
            action=DecisionAction.EXECUTE_TOOL,
            tool=decision.tool,
            params=dict(decision.params or {}),
            source=source,
            confidence=confidence,
            reasons=[f"本地 Router 选择工具 {decision.tool}"],
            evidence={**evidence, "confidence": confidence},
        )
