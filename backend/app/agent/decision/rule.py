"""RuleDecisionProvider —— 只做确定性、无歧义的流程控制。

职责被**严格限制**（任务 5.3）：只能承担以下类别的判定，禁止继续增加语义映射：

* 明确的系统命令（「取消」「停止」等）；
* 取消 / 确认 / 帮助 / 退出；
* 状态校验（是否已绑定数据集、是否还有剩余步骤）；
* 必填参数检查（工具已定、但缺必填槽位 → 反问）；
* 确定性边界（超过重试上限 → 终止）；
* 无歧义的安全 / 流程控制。

**不能**在这里写：

    「分布」→ xxx
    「相关性」→ xxx
    「异常值」→ xxx

这类「用户说了什么 → 选哪个工具」的**语义映射**，一律交给
:class:`LocalModelDecisionProvider`（本地 Router 模型）。把语义判断塞进规则，
就是把 Agent 变回硬编码关键词引擎 —— 这正是本次要收敛的病灶。
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

#: 明确的系统命令（取消 / 停止 / 退出）。只有这些是规则可判定的确定性行为。
_CANCEL_UTTERANCES = frozenset({"取消", "停止", "退出", "算了", "不做了", "停下"})


class RuleDecisionProvider(DecisionProvider):
    """确定性规则决策。**零模型调用**，只在无歧义时表态。"""

    def decide(self, context: DecisionContext) -> AgentDecision:
        # 1) 明确的取消 / 停止命令 —— 无歧义，规则直接判。
        raw = (context.user_request or "").strip()
        if raw in _CANCEL_UTTERANCES or (len(raw) <= 3 and raw in {"取消", "停止", "退出"}):
            return AgentDecision(
                action=DecisionAction.STOP,
                source=DecisionSource.RULE,
                confidence=1.0,
                reasons=["命中明确的取消/停止命令"],
                evidence={"utterance": raw},
            )

        # 2) 必填参数检查：任务理解已经定下工具、但缺 dataset_id 这类规则可判定的槽位。
        #    注意：这里只做「规则能无条件判定」的槽位（与 contract.RULE_RESOLVABLE 一致），
        #    不做语义性参数抽取（那是本地模型 / 远程的职责）。
        spec = context.task_spec
        if spec is not None and getattr(spec, "sub_goal", ""):
            tool = _tool_of_spec(spec)
            if tool and context.bound_dataset_id is None:
                # 需要 dataset_id 的工具、且未绑定数据集 → 反问，而非硬猜。
                from app.local_router.contract import required_params
                if "dataset_id" in required_params(tool):
                    return AgentDecision(
                        action=DecisionAction.ASK_USER,
                        source=DecisionSource.RULE,
                        confidence=1.0,
                        reasons=[f"工具 {tool} 需要 dataset_id，但当前未绑定数据集"],
                        evidence={"tool": tool, "missing": ["dataset_id"]},
                    )

        # 3) 确定性边界：尝试次数超限 → 终止（防死循环的结构性兜底，不是语义判断）。
        if context.attempts > 0 and context.extra.get("max_attempts"):
            max_attempts = int(context.extra["max_attempts"])
            if context.attempts >= max_attempts:
                return AgentDecision(
                    action=DecisionAction.STOP,
                    source=DecisionSource.RULE,
                    confidence=1.0,
                    reasons=[f"尝试次数 {context.attempts} 已达上限 {max_attempts}"],
                    evidence={"attempts": context.attempts, "max_attempts": max_attempts},
                )

        # 规则没有可判定的结论 —— 不表态，交由上层（本地模型 / 远程）。
        return AgentDecision(
            action=DecisionAction.EXECUTE_TOOL,
            tool=None,
            source=DecisionSource.RULE,
            confidence=0.0,
            reasons=["规则无可判定的确定性动作，交本地模型决定"],
        )


def _tool_of_spec(spec: Any) -> str | None:
    """从 TaskSpec 里取出「已确定要执行的工具」（若有）。"""
    tool = getattr(spec, "entities", {}).get("tool")
    return str(tool) if tool else None
