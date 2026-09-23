"""本地文件系统 Storage 实现。

功能：

- Storage key 安全校验
- 路径穿越防护
- 自动创建目录
- bytes 保存
- 临时文件流式保存
- 文件读取
- 文件删除
- metadata
- prefix 列表
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import StorageException
from app.storage.base import (
    Storage,
    StoredObjectMeta,
)
from app.storage.security import (
    ensure_inside_root,
    validate_key,
)


class LocalStorage(Storage):
    def __init__(
        self,
        root: Path | None = None,
    ) -> None:
        self.root = (
            root or settings.data_root_path
        ).resolve()

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==================================================
    # 内部方法
    # ==================================================

    def _resolve(
        self,
        key: str,
    ) -> Path:
        clean_key = validate_key(
            key
        )

        return ensure_inside_root(
            self.root,
            self.root / clean_key,
        )

    @staticmethod
    def _meta(
        key: str,
        path: Path,
    ) -> StoredObjectMeta:
        stat_result = path.stat()

        return StoredObjectMeta(
            key=key,
            size=stat_result.st_size,
            modified_at=datetime.fromtimestamp(
                stat_result.st_mtime
            ),
        )

    # ==================================================
    # 保存 bytes
    # ==================================================

    def save(
        self,
        key: str,
        data: bytes,
    ) -> StoredObjectMeta:
        path = self._resolve(
            key
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_name(
            f".{path.name}.tmp"
        )

        try:
            temp_path.write_bytes(
                data
            )

            os.replace(
                temp_path,
                path,
            )

        except OSError as exc:
            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            raise StorageException(
                "failed to save object",
                code="STORAGE_WRITE_FAILED",
                details={
                    "key": key,
                },
            ) from exc

        return self._meta(
            key,
            path,
        )

    # ==================================================
    # 流式保存
    # ==================================================

    def save_stream(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        """从临时文件流式复制到 Storage。

        使用临时目标文件 + os.replace，
        避免程序中途异常留下一个看似正常的半截文件。
        """

        if not source.is_file():
            raise StorageException(
                "source file not found",
                code="SOURCE_NOT_FOUND",
                details={
                    "source": str(source),
                },
            )

        path = self._resolve(
            key
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_name(
            f".{path.name}.tmp"
        )

        try:
            with (
                source.open("rb") as src,
                temp_path.open("wb") as dst,
            ):
                shutil.copyfileobj(
                    src,
                    dst,
                    length=1024 * 1024,
                )

            os.replace(
                temp_path,
                path,
            )

        except OSError as exc:
            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            raise StorageException(
                "failed to save object",
                code="STORAGE_WRITE_FAILED",
                details={
                    "key": key,
                },
            ) from exc

        return self._meta(
            key,
            path,
        )

    # ==================================================
    # 读取
    # ==================================================

    def read(
        self,
        key: str,
    ) -> bytes:
        path = self._resolve(
            key
        )

        if not path.is_file():
            raise StorageException(
                "object not found",
                code="NOT_FOUND",
                details={
                    "key": key,
                },
            )

        try:
            return path.read_bytes()

        except OSError as exc:
            raise StorageException(
                "failed to read object",
                code="STORAGE_READ_FAILED",
                details={
                    "key": key,
                },
            ) from exc

    # ==================================================
    # 删除
    # ==================================================

    def delete(
        self,
        key: str,
    ) -> None:
        path = self._resolve(
            key
        )

        if not path.is_file():
            raise StorageException(
                "object not found",
                code="NOT_FOUND",
                details={
                    "key": key,
                },
            )

        try:
            path.unlink()

        except OSError as exc:
            raise StorageException(
                "failed to delete object",
                code="STORAGE_DELETE_FAILED",
                details={
                    "key": key,
                },
            ) from exc

    # ==================================================
    # 存在性
    # ==================================================

    def exists(
        self,
        key: str,
    ) -> bool:
        path = self._resolve(
            key
        )

        return path.is_file()

    # ==================================================
    # Metadata
    # ==================================================

    def stat(
        self,
        key: str,
    ) -> StoredObjectMeta:
        path = self._resolve(
            key
        )

        if not path.is_file():
            raise StorageException(
                "object not found",
                code="NOT_FOUND",
                details={
                    "key": key,
                },
            )

        return self._meta(
            key,
            path,
        )

    # ==================================================
    # List
    # ==================================================

    def list(
        self,
        prefix: str = "",
    ) -> Iterator[StoredObjectMeta]:

        clean_prefix = ""

        if prefix:
            clean_prefix = validate_key(
                prefix.rstrip("/")
            )

        base = (
            self._resolve(
                clean_prefix
            )
            if clean_prefix
            else self.root
        )

        if not base.exists():
            return

        if base.is_file():
            yield self._meta(
                clean_prefix,
                base,
            )
            return

        for path in sorted(
            base.rglob("*")
        ):
            if not path.is_file():
                continue

            key = (
                path.relative_to(
                    self.root
                )
                .as_posix()
            )

            if (
                not clean_prefix
                or key.startswith(
                    clean_prefix + "/"
                )
            ):
                yield self._meta(
                    key,
                    path,
                )

    # ==================================================
    # 本地直通（大数据接入）
    # ==================================================

    def local_path(
        self,
        key: str,
    ) -> Path | None:
        """返回对象的本地绝对路径（已做路径穿越校验）。

        仅做路径解析，不要求对象已存在——入库时需要「先拿目标路径、再写文件」，
        若此处强制 exists 检查，流式写入就没有落点。
        """
        return self._resolve(key)

    def promote(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        """原子提升本地临时文件为存储对象。

        同盘时 ``os.replace`` 是零拷贝的元数据操作；跨盘（Windows 跨盘符）
        会抛 ``OSError``，此时退回 ``save_stream`` 的流式拷贝。
        """
        if not source.is_file():
            raise StorageException(
                "source file not found",
                code="SOURCE_NOT_FOUND",
                details={"source": str(source)},
            )

        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.replace(source, path)
        except OSError:
            # 跨文件系统 / 目标被占用：退回拷贝（仍然先写临时文件再替换）。
            return self.save_stream(key, source)

        return self._meta(key, path)
