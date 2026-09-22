"""AgentContext：受控的 Agent 上下文容器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.context.budget import ContextBudget

TRUNCATION_MARK = "…【已截断】"


def clip_text(text: str, max_chars: int) -> str:
    """按字符数截断文本并附加标记。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_MARK


def clip_obj(obj: Any, max_chars: int) -> Any:
    """把对象序列化为 JSON 并截断；失败时退回 str 截断。"""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(obj)
    return clip_text(text, max_chars)


@dataclass
class AgentContext:
    """Agent 上下文（全部为受控大小的摘要，不含完整数据）。"""

    user_request: str = ""
    dataset_context: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    permission_context: dict[str, Any] = field(default_factory=dict)
    tool_context: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    budget: ContextBudget | None = None

    def to_prompt_text(
        self,
        *,
        max_chars: int = 6000,
        budget: ContextBudget | None = None,
    ) -> str:
        """渲染受控上下文；优先使用 Builder 注入的预算，且严格限制总字符数。"""
        budget = budget or self.budget
        if budget is None:
            per_section = max(max_chars // 6, 120)
            budget = ContextBudget(
                max_chars=max_chars,
                user_request=per_section,
                dataset=per_section,
                task=per_section,
                permissions=per_section,
                tools=per_section,
                history=per_section,
                history_messages=6,
            )
        budget = budget.normalized()

        sections = {
            "user_request": budget.user_request,
            "dataset": budget.dataset,
            "task": budget.task,
            "permissions": budget.permissions,
            "tools": budget.tools,
            "history": budget.history,
        }
        total = sum(sections.values())
        if total > budget.max_chars:
            # 先给每个分区保留最低空间，再按原预算比例分配剩余空间。
            minimum = 80
            remaining = max(budget.max_chars - minimum * len(sections), 0)
            weighted = {key: max(value - minimum, 0) for key, value in sections.items()}
            weight_total = sum(weighted.values())
            if weight_total:
                allocated = {
                    key: minimum + int(remaining * weight / weight_total)
                    for key, weight in weighted.items()
                }
            else:
                allocated = {key: minimum for key in sections}
            # 消除整数舍入造成的 1~5 字符超额。
            overflow = sum(allocated.values()) - budget.max_chars
            for key in reversed(list(allocated)):
                if overflow <= 0:
                    break
                reducible = max(allocated[key] - minimum, 0)
                delta = min(reducible, overflow)
                allocated[key] -= delta
                overflow -= delta
            sections = allocated

        history = self.conversation_history[-budget.history_messages :]
        parts = [
            f"[用户请求]\n{clip_text(self.user_request, sections['user_request'])}",
            f"[数据集]\n{clip_obj(self.dataset_context, sections['dataset'])}",
            f"[任务]\n{clip_obj(self.task_context, sections['task'])}",
            f"[权限]\n{clip_obj(self.permission_context, sections['permissions'])}",
            f"[可用工具]\n{clip_obj({'tools': self.tool_context.get('tools', [])}, sections['tools'])}",
        ]
        if history and budget.history_messages > 0:
            history_text = "\n".join(
                f"{m.get('role', 'user')}: {clip_text(m.get('content', ''), 300)}"
                for m in history
            )
            parts.append(f"[历史对话]\n{clip_text(history_text, sections['history'])}")
        return "\n\n".join(parts)

    def dataset_ids(self) -> list[int]:
        """上下文中出现过的数据集 id。"""
        ids: list[int] = []
        for key in self.dataset_context.keys():
            try:
                ids.append(int(key))
            except (TypeError, ValueError):
                continue
        return ids
