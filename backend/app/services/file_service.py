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
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import (
    NotFoundException,
    ValidationException,
)
from app.models.file import File
from app.storage.service import StorageService

# 单文件最大上传大小：100 MB
MAX_UPLOAD_SIZE = 100 * 1024 * 1024

# 每次从上传流读取 1 MB
UPLOAD_CHUNK_SIZE = 1024 * 1024

# 第一阶段允许的数据文件格式
ALLOWED_UPLOAD_FORMATS = {
    "csv",
    "json",
    "xlsx",
    "xls",
    "parquet",
}


class FileService:
    def __init__(
        self,
        db: Session,
        storage: StorageService,
    ) -> None:
        self.db = db
        self.storage = storage

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
                        UPLOAD_CHUNK_SIZE
                    )

                    if not chunk:
                        break

                    total_size += len(chunk)

                    # 超过限制立即停止
                    if total_size > MAX_UPLOAD_SIZE:
                        raise ValidationException(
                            "file too large",
                            code="FILE_TOO_LARGE",
                            details={
                                "max_size": MAX_UPLOAD_SIZE,
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

            self.storage.save_stream(
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

        if len(content) > MAX_UPLOAD_SIZE:
            raise ValidationException(
                "file too large",
                code="FILE_TOO_LARGE",
                details={
                    "max_size": MAX_UPLOAD_SIZE,
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

    def read(
        self,
        file_id: int,
    ) -> bytes:
        """读取文件内容。

        注意：
        当前 Storage 抽象仍然返回 bytes。

        对于毕设规模的数据文件足够使用。
        后续真正进入大文件数据分析阶段时，
        再增加流式读取/分块解析能力。
        """

        file = self.get(
            file_id
        )

        return self.storage.read(
            file.path
        )

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
