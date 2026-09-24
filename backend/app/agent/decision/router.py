"""DecisionRouter —— 按优先级组合三个 DecisionProvider 的统一决策入口。

这是「谁来 Decide」的**唯一入口**。调用方不直接碰三个 Provider，
而是调 ``route(context)``，由本模块按优先级合成：

    LocalModel  →  必要时 Rule fallback  →  必要时 Remote escalation

组合逻辑（可替换、可观测、可训练）：
--------------------------------------------------------------------
1. 先跑 RuleDecisionProvider：它只表态**确定性**动作（取消/必填参数/边界），
   命中即返回（零模型调用）；
2. 未命中则跑 LocalModelDecisionProvider（本地 Router 模型）；
3. 本地升级 / 分歧 / 失败时，若需要，再跑 RemoteLLMDecisionProvider。

每一步的判定都带 ``source`` + ``confidence`` + ``evidence``，全部可进
DecisionTrace —— 这正是后续训练闭环的数据来源。
"""

from __future__ import annotations

from typing import Any

from app.agent.decision.provider import (
    DecisionAction,
    DecisionContext,
    DecisionProvider,
    DecisionSource,
    AgentDecision,
)
from app.agent.decision.rule import RuleDecisionProvider
from app.agent.decision.local_model import LocalModelDecisionProvider
from app.agent.decision.remote import RemoteLLMDecisionProvider
from app.agent.decision.signal import SignalDecisionProvider


class DecisionRouter:
    """统一决策入口。优先级：Rule（确定性）→ Signal（数据信号）→ LocalModel → Remote（升级）。"""

    def __init__(
        self,
        *,
        rule: DecisionProvider | None = None,
        signal: DecisionProvider | None = None,
        local_model: DecisionProvider | None = None,
        remote: DecisionProvider | None = None,
        llm: Any | None = None,
        #: 本地置信度低于此值视为「本地不足」，可升级远程（默认 0.0 = 不门控，
        #: 与 local_router 的默认口径一致）。
        confidence_threshold: float = 0.0,
    ) -> None:
        self.rule = rule or RuleDecisionProvider()
        self.signal = signal or SignalDecisionProvider()
        self.local_model = local_model or LocalModelDecisionProvider()
        self.remote = remote or RemoteLLMDecisionProvider(llm)
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def route(self, context: DecisionContext, *, allow_remote: bool = True) -> AgentDecision:
        """合成一次决策。返回带完整证据链的 AgentDecision。

        ``allow_remote=False`` 时禁止升级远程（用于 PoC「简单任务零远程调用」验证）。
        """
        # 1) 确定性规则（零模型调用，命中即返回）。
        rule_decision = self.rule.decide(context)
        if rule_decision.action in (DecisionAction.STOP, DecisionAction.ASK_USER) and \
           rule_decision.confidence >= 1.0:
            return rule_decision

        # 2) 数据信号（Observe→Decide→Act 的「Observe」：上一步 ToolResult.signals
        #    驱动的确定性下一步）。首轮无 signals 时它不表态（confidence=0），自然跳过。
        signal_decision = self.signal.decide(context)
        if signal_decision.action == DecisionAction.EXECUTE_TOOL and \
           signal_decision.tool is not None:
            return signal_decision

        # 3) 本地模型（真正参与运行，不是摆设）。
        local = self.local_model.decide(context)

        # 本地成功决定工具 → 直接返回。
        if local.action == DecisionAction.EXECUTE_TOOL and local.tool is not None:
            return local
        if local.action == DecisionAction.CHAT:
            return local
        if local.action == DecisionAction.ASK_USER:
            return local

        # 本地升级：先判断是否允许远程，再决定走远程还是保守返回。
        if not allow_remote or self.llm is None:
            return local  # 保留本地的「升级」结论（诚实：本地确实没把握）

        # 4) 远程战略指导（仅复杂任务升级时调用一次）。
        remote = self.remote.decide(context)
        # 远程给出的是方向性指导；若远程也没结论，保留本地升级结论。
        if remote.action == DecisionAction.ESCALATE and remote.confidence == 0.0:
            return local
        return remote

    def decide(self, context: DecisionContext, *, allow_remote: bool = True) -> AgentDecision:
        """别名，与 DecisionProvider.decide 语义一致。"""
        return self.route(context, allow_remote=allow_remote)


__all__ = [
    "DecisionAction",
    "DecisionContext",
    "DecisionProvider",
    "DecisionRouter",
    "DecisionSource",
    "AgentDecision",
    "RuleDecisionProvider",
    "SignalDecisionProvider",
    "LocalModelDecisionProvider",
    "RemoteLLMDecisionProvider",
]
