"""版本级 DataFrame 缓存。

背景
----
``DatasetVersion`` 在语义上**不可变**（DatasetService 的约定："原版本永远不覆盖"），
因此 (dataset_id, version) -> DataFrame 的映射不会失效，只需要淘汰。

引入前的问题：每次调用 ``load_version()`` 都要
``storage.read()``（全量 bytes）→ ``io.BytesIO`` → ``pl.read_parquet``（全量解码）。
一轮 EDA 浏览（选数据集 → 看样本 → schema → 运行分析 → 切 4 个结果页签）
会对同一份 Parquet 重复解码 8~10 次，其中多数请求只需要几十行或几个统计量。

设计要点
--------
1. **单例、进程级**：与 AGENT_STORE / WORKFLOW_SERVICE 一样是进程内状态，
   与「必须单 worker」的既有前提一致。
2. **两级容量控制**：条数上限 + 字节上限，任一超限即开始淘汰最久未使用的条目。
3. **返回不可变副本**：Polars 的 ``clone()`` 成本很低（列是 Arc 共享），
   但能防止调用方原地修改污染缓存。分析模块本身是只读的，
   这里只是把"只读"从约定变成结构性保证。
4. **命中失败不影响正确性**：任何异常都回退到直接读盘，缓存只是加速层。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import polars as pl

from app.core.logging import get_logger

logger = get_logger("app.data_engine.cache")


def _configured_limits() -> tuple[int, int]:
    """从 settings 读取容量上限；配置层异常时退回默认值。"""
    try:
        from app.core.config import settings

        return (
            int(settings.DATASET_FRAME_CACHE_MAX_ITEMS),
            int(settings.DATASET_FRAME_CACHE_MAX_BYTES),
        )
    except Exception:  # pragma: no cover
        return DEFAULT_MAX_ITEMS, DEFAULT_MAX_BYTES


# 默认容量：条数 + 字节。实际生效值来自 settings（DATASET_FRAME_CACHE_*）。
DEFAULT_MAX_ITEMS = 16
DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MB


def _frame_bytes(df: pl.DataFrame) -> int:
    """估算 DataFrame 的常驻内存占用（字节）。"""
    try:
        return int(df.estimated_size("b"))
    except Exception:  # pragma: no cover - 极老版本 Polars 无该 API
        return int(df.height * max(df.width, 1) * 8)


class VersionFrameCache:
    """按 (dataset_id, version) 缓存不可变版本快照。

    - ``get`` 命中时返回数据的浅拷贝（clone），未命中返回 None。
    - ``put`` 超限时按 LRU 顺序淘汰，直到容纳得下。
    - ``invalidate_dataset`` 用于数据集被删除时主动清理。
    """

    def __init__(
        self,
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        cfg_items, cfg_bytes = _configured_limits()
        self.max_items = max(1, int(max_items if max_items is not None else cfg_items))
        self.max_bytes = max(0, int(max_bytes if max_bytes is not None else cfg_bytes))
        self._lock = threading.RLock()
        self._entries: OrderedDict[tuple[int, int], tuple[pl.DataFrame, int]] = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def get(self, dataset_id: int, version: int) -> pl.DataFrame | None:
        key = (int(dataset_id), int(version))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            # 命中后移到队尾（最近使用）
            self._entries.move_to_end(key)
            self.hits += 1
            frame = entry[0]
        # clone 在锁外做，减少临界区
        return frame.clone()

    def put(self, dataset_id: int, version: int, df: pl.DataFrame) -> None:
        key = (int(dataset_id), int(version))
        size = _frame_bytes(df)

        # 单个对象就超过总预算：不缓存（缓存它只会立刻被淘汰）
        if self.max_bytes and size > self.max_bytes:
            return

        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing[1]

            self._entries[key] = (df, size)
            self._total_bytes += size

            while self._entries and (
                len(self._entries) > self.max_items
                or (self.max_bytes and self._total_bytes > self.max_bytes)
            ):
                _evicted_key, (_frame, evicted_size) = self._entries.popitem(last=False)
                self._total_bytes -= evicted_size
                self.evictions += 1

            self._total_bytes = max(self._total_bytes, 0)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def invalidate_dataset(self, dataset_id: int) -> None:
        """数据集被删除时清理其所有版本。"""
        target = int(dataset_id)
        with self._lock:
            keys = [k for k in self._entries if k[0] == target]
            for key in keys:
                _frame, size = self._entries.pop(key)
                self._total_bytes -= size
            self._total_bytes = max(self._total_bytes, 0)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": len(self._entries),
                "bytes": self._total_bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(
                    self.hits / (self.hits + self.misses), 4
                ) if (self.hits + self.misses) else 0.0,
            }


# 进程级单例（与 AGENT_STORE / WORKFLOW_SERVICE 同属进程内状态，需单 worker 运行）。
VERSION_FRAME_CACHE = VersionFrameCache()


def get_version_frame_cache() -> VersionFrameCache:
    return VERSION_FRAME_CACHE
