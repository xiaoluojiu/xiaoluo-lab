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
    # 允许访问的数据集 id（空集合表示不限制）
    dataset_ids: set[int] = field(default_factory=set)
    # 用户拥有的权限
    permissions: set[Permission] = field(default_factory=set)
    request_id: str = ""
    # 其他扩展信息
    extra: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def can_access_dataset(self, dataset_id: int) -> bool:
        return not self.dataset_ids or dataset_id in self.dataset_ids
