"""SignalDecisionProvider —— 基于 ToolResult.signals 的**数据驱动**下一步决策。

定位（任务 6/7 的 Observe→Decide→Act 落点）
--------------------------------------------
这是「本地 Router 选工具」之后的**第二步及以后**决策来源。它不读自然语言语义，
只读 ``DecisionContext.signals``（工具执行后产出的**机器可读结构化标记**，如
``outliers_detected`` / ``high_missing`` / ``high_skew``），据其给出确定性下一步。

为什么它不算「语义关键词路由」
--------------------------------
``signals`` 不是「用户说了什么」，而是「数据长什么样」的客观事实 —— 由
``DistributionOverviewAnalyzer._signals`` 这类分析器在真实结果上计算得出
（如某列缺失率 ≥ 30% 才标 ``high_missing``）。据此决定「要不要深入缺失/异常分析」，
是平台能力边界内的**数据驱动**判定，与「if '分布' in text」这类自然语言关键词
有本质区别（后者才是被禁止的病灶）。

边界
----
* 只认 ``signals`` 白名单里的标记，不扩展自然语言；
* 信号 → 工具 的映射是**平台能力边界**（哪类信号该深入哪个只读分析工具），
  不是语义；且只给「下一步该看什么」，不硬编多步链条；
* 没有 signals / 信号不在白名单 → 返回 ``confidence=0`` 不表态，交本地 Router 兜底。
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


#: 机器可读信号 → 该深入的**无需 column**的只读分析工具（平台能力边界，非自然语言语义）。
#: 只覆盖「分布总览之后，数据客观形状提示的、且工具无需列名即可执行」的下一步。
#: 注意：``eda.distribution`` 需要 column，而 signals 层面没有列名，故不在此映射
#: （否则会因缺 column 参数而失败）。``eda.outlier`` / ``data.clean`` 均为数据集级。
_SIGNAL_TO_TOOL: dict[str, str] = {
    "outliers_detected": "eda.outlier",
    "high_skew": "eda.outlier",
    "high_missing": "data.clean",
    "strong_correlation": "eda.correlation",
}


class SignalDecisionProvider(DecisionProvider):
    """数据信号驱动的确定性下一步。零模型调用，只在信号白名单内表态。"""

    def decide(self, context: DecisionContext) -> AgentDecision:
        signals = context.signals or {}
        # 兼容两种形状：{signal: True} 或 {signals: [...]}。
        if isinstance(signals, dict):
            signal_list = list(signals.keys())
            if not signal_list and signals.get("signals"):
                signal_list = [str(s) for s in signals.get("signals") or []]
        else:
            signal_list = [str(s) for s in signals] if signals else []

        if not signal_list:
            return AgentDecision(
                action=DecisionAction.EXECUTE_TOOL, tool=None,
                source=DecisionSource.LOCAL_MODEL, confidence=0.0,
                reasons=["无 signals，交本地 Router 决定"],
            )

        # 按白名单顺序取第一个可行动的信号（确定性，不硬编多步链条）。
        last_tool = context.recent_tools[-1] if context.recent_tools else None
        for sig in signal_list:
            tool = _SIGNAL_TO_TOOL.get(str(sig))
            if tool and tool != last_tool:
                return AgentDecision(
                    action=DecisionAction.EXECUTE_TOOL, tool=tool,
                    source=DecisionSource.LOCAL_MODEL, confidence=0.9,
                    reasons=[f"信号 {sig} 提示深入 {tool}"],
                    evidence={"signal": str(sig), "tool": tool},
                )
        # 信号不在白名单 → 不表态。
        return AgentDecision(
            action=DecisionAction.EXECUTE_TOOL, tool=None,
            source=DecisionSource.LOCAL_MODEL, confidence=0.0,
            reasons=[f"信号 {signal_list} 不在可行动白名单，交本地 Router 决定"],
        )
