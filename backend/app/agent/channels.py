"""SSE 事件通道。

为什么需要它
------------
前端在「等待确认」期间**保持 SSE 连接不关闭**，并明确要求：确认之后的
``completed`` 事件必须**经同一条流补发** —— 那是最终答案进入聊天区的唯一出口。
如果后端在挂起时就关闭流，前端永远收不到 ``final_answer``。

因此通道的生命周期要跨过挂起：

- 创建于 ``POST /sessions/{id}/messages``
- 复用 于 ``POST /runs/{id}/confirm`` / ``clarify``（这两个端点会重新启动执行线程）
- 关闭于 ``completed`` / ``failed`` 事件发出之后

去重
----
连接建立时先把**已有事件**补发一遍（切会话、断线重连要靠它重建时间线），
随后注册的监听者必须跳过 seq 不大于已补发最大值的事件，否则会重复推送。
``start_seq`` 就是这道闸门。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from app.agent.models import AgentEvent, EventType

logger = logging.getLogger(__name__)

#: poll 的空结果哨兵。不能用 None —— None 同时表示「通道已关闭」。
EMPTY = object()


class RunChannel:
    """单个运行的一条事件广播通道。线程安全。"""

    def __init__(self, run_id: str, start_seq: int = 0) -> None:
        self.run_id = run_id
        self.start_seq = start_seq
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()

    def publish(self, event: AgentEvent) -> None:
        """推入事件。``start_seq`` 之前的（已补发过的）直接丢弃。"""
        with self._lock:
            if self._closed or event.seq <= self.start_seq:
                return
        self._queue.put(event)

    def close(self) -> None:
        """关闭通道。放入 None 哨兵唤醒阻塞中的 poll。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def mark_started(self, seq: int) -> None:
        """标记「该 seq 及之前的事件已推送过」，之后的 publish 会跳过它们。"""
        with self._lock:
            if seq > self.start_seq:
                self.start_seq = seq

    def poll(self, timeout: float = 1.0) -> Any:
        """取一条事件。返回 ``EMPTY`` 表示超时，``None`` 表示通道已关闭。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return EMPTY


class ChannelManager:
    """运行 id → 通道。"""

    def __init__(self) -> None:
        self._channels: dict[str, RunChannel] = {}
        self._lock = threading.Lock()

    def get_or_create(self, run_id: str, start_seq: int = 0) -> RunChannel:
        with self._lock:
            channel = self._channels.get(run_id)
            if channel is None or channel.closed:
                channel = RunChannel(run_id, start_seq=start_seq)
                self._channels[run_id] = channel
            return channel

    def get(self, run_id: str) -> RunChannel | None:
        with self._lock:
            return self._channels.get(run_id)

    def discard(self, run_id: str) -> None:
        with self._lock:
            channel = self._channels.pop(run_id, None)
        if channel is not None:
            channel.close()

    def close_all(self) -> None:
        with self._lock:
            channels = list(self._channels.values())
            self._channels.clear()
        for channel in channels:
            channel.close()


CHANNELS = ChannelManager()


def sse_frame(event_type: str, data: str) -> str:
    """拼一帧 SSE。前端按 ``\\n\\n`` 切帧，逐行读 ``event:`` / ``data:``。"""
    # data 里的换行会破坏帧结构，这里压成空格（事件内容是单行 JSON）
    payload = (data or "").replace("\r", " ").replace("\n", " ")
    return f"event: {event_type}\ndata: {payload}\n\n"


def is_terminal_event(event: AgentEvent) -> bool:
    return event.type in (EventType.COMPLETED, EventType.FAILED)


def wait_deadline(seconds: float) -> float:
    return time.time() + max(1.0, seconds)
