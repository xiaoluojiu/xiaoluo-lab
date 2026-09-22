"""Agent Context 预算配置。

这里控制发送给 LLM 的上下文字符预算。它是防止上下文无界增长的
运行时约束，而不是数据处理逻辑；默认值保持与旧版 Agent 行为兼容。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """一次 Agent Prompt 的分区预算（字符数）。"""

    max_chars: int = 6000
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
            user_request=max(int(self.user_request), 100),
            dataset=max(int(self.dataset), 200),
            task=max(int(self.task), 100),
            permissions=max(int(self.permissions), 100),
            tools=max(int(self.tools), 100),
            history=max(int(self.history), 100),
            history_messages=max(int(self.history_messages), 0),
        )
