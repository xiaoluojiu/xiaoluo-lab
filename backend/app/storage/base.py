"""Storage 抽象接口。

只定义契约，不绑定具体实现。

未来可以使用：

LocalStorage
MinIO
S3

而不影响上层业务。

key：

相对存储根目录的 POSIX 路径。

例如：

raw/abc.csv
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredObjectMeta:
    """存储对象元数据。"""

    key: str
    size: int
    modified_at: datetime


class Storage(ABC):
    """存储后端抽象。"""

    @abstractmethod
    def save(
        self,
        key: str,
        data: bytes,
    ) -> StoredObjectMeta:
        """保存 bytes 数据。"""
        raise NotImplementedError

    @abstractmethod
    def save_stream(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        """从本地临时文件保存对象。"""
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        key: str,
    ) -> bytes:
        """读取对象。"""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        key: str,
    ) -> None:
        """删除对象。"""
        raise NotImplementedError

    @abstractmethod
    def exists(
        self,
        key: str,
    ) -> bool:
        """判断对象是否存在。"""
        raise NotImplementedError

    @abstractmethod
    def stat(
        self,
        key: str,
    ) -> StoredObjectMeta:
        """获取对象元数据。"""
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        prefix: str = "",
    ) -> Iterator[StoredObjectMeta]:
        """按照前缀列出对象。"""
        raise NotImplementedError
