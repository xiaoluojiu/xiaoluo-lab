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

    # ---------------------------------------------------------
    # 可选能力：本地直通（大数据接入的性能前提）
    # ---------------------------------------------------------
    # 下面两个方法**不是**抽象方法：远程后端（S3 / MinIO）无法提供本地路径，
    # 保持默认实现即可让上层用「能力探测」的方式降级。

    def local_path(
        self,
        key: str,
    ) -> Path | None:
        """返回对象的本地绝对路径；不支持本地直通的后端返回 None。

        存在的意义：Parquet 走 ``pl.scan_parquet(path)`` 可以内存映射 + 谓词/投影
        下推；若必须先 ``read()`` 出完整 bytes，大数据量下会白拷贝一份内存。
        """
        return None

    def promote(
        self,
        key: str,
        source: Path,
    ) -> StoredObjectMeta:
        """把本地临时文件「就地」提升为存储对象（优先原子 rename，避免二次拷贝）。

        默认实现退回 ``save_stream``（一次完整拷贝），保证远程后端仍可工作。
        """
        return self.save_stream(key, source)
