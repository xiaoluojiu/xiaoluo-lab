"""数据集 Agent 上下文缓存。

缓存边界使用 dataset_id + version；数据版本变化后旧缓存自然失效。
只缓存已经计算好的 schema/profile/quality/sample 摘要，不缓存原始数据。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class DatasetContextCache:
    """线程安全、有界的进程内数据集摘要缓存。"""

    def __init__(self, max_items: int = 32) -> None:
        self.max_items = max(int(max_items), 1)
        self._items: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, dataset_id: int, version: int) -> dict[str, Any] | None:
        key = (int(dataset_id), int(version))
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return dict(value)

    def set(self, dataset_id: int, version: int, value: dict[str, Any]) -> None:
        key = (int(dataset_id), int(version))
        with self._lock:
            self._items[key] = dict(value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._items), "max_items": self.max_items}


DATASET_CONTEXT_CACHE = DatasetContextCache()
