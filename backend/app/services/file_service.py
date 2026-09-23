"""FileService。

负责文件上传相关业务逻辑：

- 文件名校验
- 文件格式校验
- 文件大小校验
- SHA-256 内容指纹
- 重复文件检测
- 临时文件管理
- Storage 写入
- File 数据库记录
- Storage / DB 失败后的补偿清理
"""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    NotFoundException,
    ValidationException,
)
from app.models.file import File
from app.storage.service import StorageService

# 单文件上传上限与读取块大小：均由配置驱动。
# 历史上这里是硬编码的 ``100 * 1024 * 1024``——「大数据平台只收 100 MB」
# 本身就是自相矛盾的定位，而且它并不能真正保护进程（真正的瓶颈见
# app/data_engine/ingest.py 的说明）。现在默认 2 GiB，可按部署环境调整。
_FALLBACK_MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024
_FALLBACK_CHUNK_SIZE = 4 * 1024 * 1024


def get_max_upload_size() -> int:
    """当前生效的单文件上传上限（字节）。"""
    try:
        return int(settings.MAX_UPLOAD_SIZE_BYTES)
    except Exception:  # pragma: no cover - 配置层异常时退回默认
        return _FALLBACK_MAX_UPLOAD_SIZE


def get_upload_chunk_size() -> int:
    """上传流的分块大小（字节）。"""
    try:
        return max(64 * 1024, int(settings.UPLOAD_CHUNK_SIZE_BYTES))
    except Exception:  # pragma: no cover
        return _FALLBACK_CHUNK_SIZE


# 兼容旧引用：保留常量名，但值不再写死。
MAX_UPLOAD_SIZE = _FALLBACK_MAX_UPLOAD_SIZE
UPLOAD_CHUNK_SIZE = _FALLBACK_CHUNK_SIZE

# 第一阶段允许的数据文件格式
ALLOWED_UPLOAD_FORMATS = {
    "csv",
    "json",
    "xlsx",
    "xls",
    "parquet",
    "arff",
}


class FileService:
    def __init__(
        self,
        db: Session,
        storage: StorageService,
        *,
        max_upload_size: int | None = None,
    ) -> None:
        self.db = db
        self.storage = storage
        # 允许注入：测试可以用极小的上限验证「超限拒绝」这一契约，
        # 而不用真的构造一个 2 GiB 的字节串（历史测试为此分配了 100 MB）。
        self.max_upload_size = (
            int(max_upload_size) if max_upload_size is not None else get_max_upload_size()
        )

    @property
    def chunk_size(self) -> int:
        return get_upload_chunk_size()

    # ==================================================
    # 上传
    # ==================================================

    def upload_stream(
        self,
        original_name: str,
        file_obj: BinaryIO,
    ) -> File:
        """流式上传文件。

        流程：

        上传流
          ↓
        临时文件
          ↓
        SHA256
          ↓
        大小检查
          ↓
        格式检查
          ↓
        重复检测
          ↓
        Storage
          ↓
        DB
        """

        # --------------------------------------------------
        # 1. 文件名
        # --------------------------------------------------

        if not original_name or not original_name.strip():
            raise ValidationException(
                "file name is required",
                code="FILE_NAME_REQUIRED",
            )

        original_name = original_name.strip()

        safe_name = _sanitize_filename(
            original_name
        )

        # --------------------------------------------------
        # 2. 文件格式
        # --------------------------------------------------

        file_format = _get_file_format(
            safe_name
        )

        if file_format not in ALLOWED_UPLOAD_FORMATS:
            raise ValidationException(
                "file format is not supported",
                code="FILE_FORMAT_NOT_SUPPORTED",
                details={
                    "format": file_format,
                    "allowed_formats": sorted(
                        ALLOWED_UPLOAD_FORMATS
                    ),
                },
            )

        # --------------------------------------------------
        # 3. 创建临时文件
        # --------------------------------------------------

        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix="xiaoluo_lab_",
                suffix=".upload",
                delete=False,
            ) as temp_file:

                temp_path = Path(
                    temp_file.name
                )

                sha256 = hashlib.sha256()
                total_size = 0

                # --------------------------------------------------
                # 4. 流式读取
                # --------------------------------------------------

                while True:
                    chunk = file_obj.read(
                        self.chunk_size
                    )

                    if not chunk:
                        break

                    total_size += len(chunk)

                    # 超过限制立即停止
                    if total_size > self.max_upload_size:
                        raise ValidationException(
                            "file too large",
                            code="FILE_TOO_LARGE",
                            details={
                                "max_size": self.max_upload_size,
                                "actual_size": total_size,
                            },
                        )

                    sha256.update(chunk)

                    temp_file.write(chunk)

            # --------------------------------------------------
            # 5. 空文件
            # --------------------------------------------------

            if total_size == 0:
                raise ValidationException(
                    "file content is empty",
                    code="FILE_EMPTY",
                )

            checksum = sha256.hexdigest()

            # --------------------------------------------------
            # 6. 重复文件检测
            # --------------------------------------------------

            stmt = (
                select(File)
                .where(
                    File.checksum == checksum,
                    File.size == total_size,
                )
                .order_by(File.id.asc())
            )

            existing = self.db.scalars(stmt).first()

            if existing is not None:
                return existing

            # --------------------------------------------------
            # 7. Storage Key
            # --------------------------------------------------

            stored_name = (
                f"{uuid.uuid4().hex[:12]}_"
                f"{safe_name}"
            )

            key = f"raw/{stored_name}"

            # --------------------------------------------------
            # 8. 保存 Storage
            # --------------------------------------------------

            self.storage.promote(
                key,
                temp_path,
            )

            # --------------------------------------------------
            # 9. 创建数据库记录
            # --------------------------------------------------

            file_record = File(
                name=stored_name,
                original_name=original_name,
                path=key,
                size=total_size,
                format=file_format,
                checksum=checksum,
            )

            try:
                self.db.add(file_record)
                self.db.commit()
                self.db.refresh(file_record)

                return file_record

            except Exception:
                # DB 写入失败：
                # 删除已经成功保存的 Storage 对象。

                try:
                    if self.storage.exists(key):
                        self.storage.delete(key)
                except Exception:
                    # 不覆盖数据库原始异常。
                    pass

                raise

        finally:
            # --------------------------------------------------
            # 10. 永远清理临时文件
            # --------------------------------------------------

            if temp_path is not None:
                try:
                    temp_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

    # ==================================================
    # 兼容旧接口
    # ==================================================

    def upload(
        self,
        original_name: str,
        content: bytes,
    ) -> File:
        """兼容旧的 bytes 上传方式。

        旧测试或内部代码如果仍然调用 upload()
        不会立即全部失效。

        新 API 必须使用 upload_stream()。
        """

        if not content:
            raise ValidationException(
                "file content is empty",
                code="FILE_EMPTY",
            )

        if len(content) > self.max_upload_size:
            raise ValidationException(
                "file too large",
                code="FILE_TOO_LARGE",
                details={
                    "max_size": self.max_upload_size,
                    "actual_size": len(content),
                },
            )

        return self.upload_stream(
            original_name=original_name,
            file_obj=BytesIO(content),
        )

    # ==================================================
    # 查询
    # ==================================================

    def get(
        self,
        file_id: int,
    ) -> File:
        file = self.db.get(
            File,
            file_id,
        )

        if file is None:
            raise NotFoundException(
                "file not found",
                details={
                    "file_id": file_id,
                },
            )

        return file

    # ==================================================
    # 读取
    # ==================================================

    def local_path(
        self,
        file_id: int,
    ) -> Path | None:
        """返回文件在本地磁盘上的绝对路径；后端不支持本地直通时返回 None。

        这是大数据接入的关键一环：拿到路径后可以「原地解析 + 流式转 Parquet」，
        而不是先把整个文件读成 bytes（2 GiB 的文件那样做等于白占 2 GiB 内存）。
        """
        file = self.get(file_id)

        try:
            return self.storage.local_path(file.path)
        except Exception:  # noqa: BLE001 - 探测失败按不支持处理
            return None

    def read(
        self,
        file_id: int,
    ) -> bytes:
        """读取文件内容（兼容接口，慎用于大文件）。

        注意：
        当前 Storage 抽象仍然返回 bytes。
        需要处理大文件时请改用 ``local_path()``（配合 ingest 流式入库）
        或 ``open_stream()``（配合流式下载）。
        """

        file = self.get(
            file_id
        )

        return self.storage.read(
            file.path
        )

    def open_stream(
        self,
        file_id: int,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> tuple[File, "Iterator[bytes]"]:
        """以分块迭代器方式读取文件（用于流式下载，内存占用与文件体积无关）。

        后端不支持本地直通时退回「一次性读入内存再分块吐出」——语义一致，
        只是大文件下会占内存，属于后端能力差异而非本层退化。
        """
        file = self.get(file_id)

        path = self.local_path(file_id)

        if path is not None and path.is_file():

            def _iter_local() -> Iterator[bytes]:
                with path.open("rb") as handle:
                    while True:
                        block = handle.read(chunk_size)
                        if not block:
                            break
                        yield block

            return file, _iter_local()

        content = self.storage.read(file.path)

        def _iter_memory() -> Iterator[bytes]:
            for offset in range(0, len(content), chunk_size):
                yield content[offset : offset + chunk_size]

        return file, _iter_memory()

    # ==================================================
    # 列表
    # ==================================================

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[File], int]:

        offset = (
            page - 1
        ) * page_size

        stmt = (
            select(File)
            .order_by(File.id.desc())
            .limit(page_size)
            .offset(offset)
        )

        items = list(self.db.scalars(stmt))

        total = self.db.scalar(
            select(func.count(File.id))
        ) or 0

        return items, total

    # ==================================================
    # 删除
    # ==================================================

    def delete(
        self,
        file_id: int,
    ) -> None:
        """删除文件。

        顺序：

        Storage
          ↓
        DB

        Storage 删除失败时，
        不继续删除数据库记录。
        """

        file = self.get(
            file_id
        )

        if self.storage.exists(
            file.path
        ):
            self.storage.delete(
                file.path
            )

        self.db.delete(file)
        self.db.commit()


# ======================================================
# 工具函数
# ======================================================

def _get_file_format(
    filename: str,
) -> str:
    """获取扩展名。"""

    suffix = Path(
        filename
    ).suffix.lower()

    if not suffix:
        return ""

    return suffix[1:]


def _sanitize_filename(
    name: str,
) -> str:
    """安全处理用户文件名。"""

    base = (
        name
        .replace("\\", "/")
        .rsplit("/", 1)[-1]
        .strip()
    )

    cleaned = "".join(
        "_"
        if (
            ch in '<>:"|?*'
            or ord(ch) < 32
        )
        else ch
        for ch in base
    )

    cleaned = cleaned.strip(
        ". "
    )

    if not cleaned:
        raise ValidationException(
            "invalid file name",
            code="FILE_NAME_INVALID",
        )

    # Windows 保留设备名
    stem = Path(
        cleaned
    ).stem.lower()

    reserved_names = {
        "con",
        "prn",
        "aux",
        "nul",
        *{
            f"com{i}"
            for i in range(1, 10)
        },
        *{
            f"lpt{i}"
            for i in range(1, 10)
        },
    }

    if stem in reserved_names:
        cleaned = f"_{cleaned}"

    # 防止极端长文件名
    return cleaned[:200]
