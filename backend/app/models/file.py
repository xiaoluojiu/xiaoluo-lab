"""File 数据库模型。

只记录上传文件的元信息，不存储实际文件内容（内容由 Storage 层管理）。
"""

from __future__ import annotations

from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class File(BaseModel):
    __tablename__ = "files"

    # 存储名（唯一、含防碰撞前缀），如 8f3a1b2c_sales.csv
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    # 用户上传时的原始文件名
    original_name: Mapped[str] = mapped_column(String(255))
    # Storage 层 key（如 raw/8f3a1b2c_sales.csv）
    path: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    # 文件格式（小写扩展名，如 csv / xlsx / json / parquet）
    format: Mapped[str] = mapped_column(String(32), default="")
    # SHA-256 内容指纹（用于去重与完整性校验）
    checksum: Mapped[str] = mapped_column(String(64), index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<File id={self.id} name={self.name!r}>"
