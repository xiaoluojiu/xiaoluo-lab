"""RemoteLLMDecisionProvider —— 复杂任务升级到远程大模型做**战略级**指导。

定位（任务 10）
--------------
远程 LLM 是**升级机制，不是默认大脑**。它只在本地能力不足时被调用一次，
产出战略级指导（复杂度判断 / 下一步方向 / 高层次的计划建议），
**不控制整个 Tool 执行循环**。工具执行仍由本地 Agent 完成。

升级触发条件（任务 10 列出的，由调用方判断后传入本 Provider）：
* 本地置信度明显不足；
* 本地模型高度分歧；
* ToolResult 非预期；
* 参数无法解析；
* 任务开放性过高；
* 多次执行失败；
* 本地无法形成有效决策。

本 Provider 的职责：**只做一次远程调用**，把结果转成 AgentDecision。
它不做重试循环、不做多轮远程追问 —— 那些会退化成「Planner Remote → Tool →
Remote → …」的反模式（任务 10 明确反对）。
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


class RemoteLLMDecisionProvider(DecisionProvider):
    """远程战略指导。要求注入 ``llm``（LLMProvider），否则返回保守升级。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def decide(self, context: DecisionContext) -> AgentDecision:
        if self.llm is None:
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=DecisionSource.REMOTE_LLM,
                confidence=0.0,
                reasons=["远程 LLM 未注入，无法做战略指导"],
            )

        from app.agent.llm.base import LLMMessage

        prompt = (
            "你是小洛实验室的 Agent 战略指导者。当前本地模型无法完成一个复杂任务，"
            "请给出**一次**战略级判断：下一步最该做什么。\n"
            "不要给出完整的工具执行循环，只给方向性的下一步动作与理由。\n"
            f"用户请求：{context.user_request}\n"
            f"任务理解：{context.task_spec.to_dict() if hasattr(context.task_spec, 'to_dict') else context.task_spec}\n"
            f"当前 ToolResult 信号：{context.signals}\n"
        )
        try:
            response = self.llm.chat([LLMMessage(role="user", content=prompt)])
            content = (response.content or "").strip()
        except Exception as exc:  # noqa: BLE001 — 远程失败不能静默吞掉
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=DecisionSource.REMOTE_LLM,
                confidence=0.0,
                reasons=[f"远程 LLM 调用失败：{type(exc).__name__}: {exc}"],
            )

        if not content:
            return AgentDecision(
                action=DecisionAction.ESCALATE,
                source=DecisionSource.REMOTE_LLM,
                confidence=0.0,
                reasons=["远程 LLM 返回空内容"],
            )

        # 远程给出的是战略级指导，不是最终工具决策。这里把它作为 evidence 记录，
        # 具体工具选择仍回落到本地 Agent / Rule（任务 10：本地执行）。
        return AgentDecision(
            action=DecisionAction.EXECUTE_TOOL,
            tool=None,
            source=DecisionSource.REMOTE_LLM,
            confidence=0.5,
            reasons=[f"远程战略指导：{content[:200]}"],
            evidence={"remote_guidance": content},
        )
