"""Prompt 086：工作流状态定义。"""

from __future__ import annotations

from enum import StrEnum


class NodeStatus(StrEnum):
    """节点 / 工作流运行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# 终态集合（进入后不再变化）
TERMINAL_STATUSES = {
    NodeStatus.SUCCESS,
    NodeStatus.FAILED,
    NodeStatus.CANCELLED,
    NodeStatus.SKIPPED,
}
