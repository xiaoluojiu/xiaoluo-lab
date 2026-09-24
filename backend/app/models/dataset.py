"""Dataset 数据库模型。

Dataset 是逻辑数据集：只保存元信息。
DataFrame 内容一律通过 DatasetVersion 快照（parquet）持久化，
绝不把 DataFrame 直接存进数据库。
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Dataset(BaseModel):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # 来源上传文件（可选）
    source_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("files.id"), nullable=True
    )

    source_file = relationship("File", foreign_keys=[source_file_id])
    versions = relationship(
        "DatasetVersion",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Dataset id={self.id} name={self.name!r}>"
