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
