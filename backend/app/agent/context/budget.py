"""Agent Context 预算配置。

这里控制发送给 LLM 的上下文字符预算。它是防止上下文无界增长的
运行时约束，而不是数据处理逻辑；默认值保持与旧版 Agent 行为兼容。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """一次 Agent Prompt 的分区预算。

    同时用**字符**与**token**双限：中文 1 字≈1 token、英文 4 字符≈1 token，
    只按字符卡会导致中文场景悄悄吃满 LLM 输入窗口。二者谁先到就按谁截断，
    见 :mod:`app.agent.context.tokens` 的估算口径。
    """

    max_chars: int = 6000
    # token 上限为 0 表示关闭 token 约束（退回纯字符行为）
    max_tokens: int = 6000
    user_request: int = 800
    dataset: int = 1600
    task: int = 600
    permissions: int = 400
    tools: int = 1400
    history: int = 1000
    history_messages: int = 6

    def normalized(self) -> "ContextBudget":
        """保证外部配置异常时仍得到安全、正数的预算。"""
        return ContextBudget(
            max_chars=max(int(self.max_chars), 1000),
            max_tokens=max(int(self.max_tokens), 0),
            user_request=max(int(self.user_request), 100),
            dataset=max(int(self.dataset), 200),
            task=max(int(self.task), 100),
            permissions=max(int(self.permissions), 100),
            tools=max(int(self.tools), 100),
            history=max(int(self.history), 100),
            history_messages=max(int(self.history_messages), 0),
        )
