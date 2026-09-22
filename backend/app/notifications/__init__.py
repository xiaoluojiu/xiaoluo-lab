"""通知模块：进程内通知中心 + 通知偏好设置。

单一进程内队列（不落库），与 AgentStore / WORKFLOW_SERVICE 同为进程级单例。
通知偏好（邮箱、各类型开关）同样只存进程内，重启后回到 .env / 默认值。
"""
from app.notifications.store import NOTIFICATION_STORE, NotificationStore, notify

__all__ = ["NOTIFICATION_STORE", "NotificationStore", "notify"]
