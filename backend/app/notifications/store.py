"""通知中心：线程安全的进程内通知队列 + 通知偏好设置。"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# 通知类型：training / report / workflow / permission / system
# 对应前端铃铛列表的图标与跳转目标。
NOTIFICATION_TYPES = ("training", "report", "workflow", "permission", "system")

# 队列容量：超出后淘汰最旧的通知，避免内存无限增长。
_MAX_QUEUE = 200


@dataclass
class Notification:
    """一条站内通知。"""

    id: str
    type: str
    title: str
    body: str
    created_at: float = field(default_factory=time.time)
    read: bool = False
    # 跳转提示：前端据此拼出「去查看」链接（如 /ml、/reports、/workflow）。
    link: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at,
            "read": self.read,
            "link": self.link,
        }


class NotificationStore:
    """进程级单例：emit / list / 未读计数 / 标记已读 + 通知偏好。

    通知偏好只存进程内（与 Agent 策略一致），重启后回到默认值。
    邮箱等站外渠道为「展望」：这里只保存偏好，不做真实推送。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: list[Notification] = []
        # ★ 变更版本号：**任何**写操作都自增。
        # 前端 SSE 靠它与上一次推送的版本号比对来判断要不要重新拉列表，
        # 从而避免在浏览器里反复轮询完整列表（原先是固定 15s 一轮）。
        self._version = 0
        # 通知偏好：email 占位 + 各类型开关。
        self._prefs: dict[str, Any] = {
            "email": "",
            "notify_training": True,
            "notify_report": True,
            "notify_workflow": True,
            "notify_permission": False,
            "notify_system": True,
        }

    # ---------- emit ----------

    def emit(
        self,
        type_: str,
        title: str,
        body: str,
        *,
        link: str | None = None,
    ) -> Notification:
        """写入一条通知。若该类型在偏好里被关闭，则不产生通知（返回 None 由调用方忽略）。"""
        if not self._is_enabled(type_):
            raise _DisabledNotification(type_)
        item = Notification(
            id=uuid.uuid4().hex[:12],
            type=type_,
            title=title,
            body=body,
            link=link,
        )
        with self._lock:
            self._items.append(item)
            # 淘汰最旧
            if len(self._items) > _MAX_QUEUE:
                self._items = self._items[-_MAX_QUEUE:]
            self._version += 1
        return item

    def _is_enabled(self, type_: str) -> bool:
        key = f"notify_{type_}"
        return bool(self._prefs.get(key, True))

    # ---------- query ----------

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items)
        # 最新在前
        items.sort(key=lambda n: n.created_at, reverse=True)
        return [n.to_dict() for n in items[:limit]]

    def unread_count(self) -> int:
        with self._lock:
            return sum(1 for n in self._items if not n.read)

    def version(self) -> int:
        """当前版本号；与上一次读到的不同即意味着需要重新拉取列表。"""
        with self._lock:
            return self._version

    def snapshot(self) -> dict[str, Any]:
        """一次加锁拿到「列表 + 未读数 + 版本号」，避免三者之间互相错位。"""
        with self._lock:
            items = sorted(self._items, key=lambda n: n.created_at, reverse=True)
            unread = sum(1 for n in items if not n.read)
            return {
                "items": [n.to_dict() for n in items[:50]],
                "unread": unread,
                "version": self._version,
            }

    def mark_read(self, notification_id: str) -> bool:
        with self._lock:
            for n in self._items:
                if n.id == notification_id and not n.read:
                    n.read = True
                    self._version += 1
                    return True
        return False

    def mark_all_read(self) -> int:
        count = 0
        with self._lock:
            for n in self._items:
                if not n.read:
                    n.read = True
                    count += 1
            if count:
                self._version += 1
        return count

    # ---------- prefs ----------

    def get_prefs(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._prefs)

    def update_prefs(self, partial: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for key, value in partial.items():
                if key in self._prefs:
                    self._prefs[key] = value
            return dict(self._prefs)


class _DisabledNotification(Exception):
    """该类型通知已被用户在偏好里关闭，调用方应静默忽略。"""

    def __init__(self, type_: str):
        super().__init__(f"通知类型已关闭：{type_}")
        self.type = type_


# 进程级单例（与 deps.py 的 AGENT_STORE 同构）。
NOTIFICATION_STORE = NotificationStore()


def notify(
    type_: str,
    title: str,
    body: str,
    *,
    link: str | None = None,
    store: NotificationStore | None = None,
) -> Notification | None:
    """便捷入口：写入通知；类型被关闭时静默忽略，绝不打断业务主流程。"""
    store = store or NOTIFICATION_STORE
    try:
        return store.emit(type_, title, body, link=link)
    except _DisabledNotification:
        return None
