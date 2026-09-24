"""Prompt 090：ToolExecutionContext。

工具执行时的上下文：谁在调用、在哪个会话、允许访问哪些数据集、
拥有哪些权限。工具与注册表据此做权限与数据范围校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.permission.models import Permission


@dataclass
class ToolExecutionContext:
    """工具执行上下文。"""

    user_id: str
    session_id: str = ""
    # 允许访问的数据集 id。
    #   None  = 不限制（unrestricted，仅用于不绑定数据集的系统级调用）
    #   set() = 没有数据集可访问（当前没有关联数据集）
    #   {7}   = 只能访问 7
    # 三种状态必须可区分：「没有关联数据集」与「允许访问全部数据集」不是一回事，
    # 混在同一个空集合里会让 Agent 在用户还没选数据集时静默读任意数据集。
    dataset_ids: set[int] | None = None
    # 用户拥有的权限
    permissions: set[Permission] = field(default_factory=set)
    request_id: str = ""
    # 其他扩展信息
    extra: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def can_access_dataset(self, dataset_id: int) -> bool:
        # None 才是不限制；空集合表示「一个都不许访问」。
        if self.dataset_ids is None:
            return True
        return dataset_id in self.dataset_ids
