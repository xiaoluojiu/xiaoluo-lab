"""StorageService。

业务代码统一通过 StorageService 操作存储。

具体后端：

LocalStorage

未来可以替换为：

MinIO
S3
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from app.storage.base import (
    Storage,
    StoredObjectMeta,
)
from app.storage.local import LocalStorage


class StorageService:
    def __init__(
        self,
        backend: Storage,
    ) -> None:
        self._backend = backend

    def save(
        self,
        key: str,
        data: bytes,
    ) -> StoredObjectMeta:
        return self._backend.save(
            key,
            data,
        )

    def save_stream(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        return self._backend.save_stream(
            key,
            source,
        )

    # ---- 本地直通（大数据接入的性能前提，见 app/data_engine/ingest.py）----

    def local_path(
        self,
        key: str,
    ) -> Path | None:
        """对象的本地绝对路径；后端不支持本地直通时返回 None。"""
        return self._backend.local_path(key)

    def promote(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        """把本地临时文件提升为存储对象（同盘零拷贝，跨盘自动退回拷贝）。"""
        return self._backend.promote(key, source)

    def supports_local_path(self) -> bool:
        """当前后端是否支持「按路径读」，决定能否走 scan_parquet 懒执行。"""
        try:
            return self._backend.local_path("__probe__") is not None
        except Exception:  # noqa: BLE001 - 探测失败一律按不支持处理
            return False

    def read(
        self,
        key: str,
    ) -> bytes:
        return self._backend.read(
            key
        )

    def delete(
        self,
        key: str,
    ) -> None:
        self._backend.delete(
            key
        )

    def exists(
        self,
        key: str,
    ) -> bool:
        return self._backend.exists(
            key
        )

    def metadata(
        self,
        key: str,
    ) -> StoredObjectMeta:
        return self._backend.stat(
            key
        )

    def list(
        self,
        prefix: str = "",
    ) -> Iterator[StoredObjectMeta]:
        return self._backend.list(
            prefix
        )


def get_storage() -> StorageService:
    """默认使用本地 Storage。"""

    return StorageService(
        LocalStorage()
    )
