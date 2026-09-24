"""Operation 数据库模型。

记录对数据集执行过的每次操作（输入版本 -> 操作 -> 输出版本），
用于实现数据操作的审计（audit）与复现（reproduce）。
"""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Operation(BaseModel):
    __tablename__ = "operations"

    # 所属数据集（冗余存储便于按 dataset 查询）
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    # 输入 / 输出版本（首个操作可能没有输入版本）
    input_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, index=True
    )
    output_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, index=True
    )
    # 操作类型（filter / transform / aggregate / merge / ...）
    operation_type: Mapped[str] = mapped_column(String(64), index=True)
    # 操作参数快照（JSON），用于复现
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    # success / failed
    status: Mapped[str] = mapped_column(String(32), default="success", index=True)
    # 失败时的错误信息
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    input_version = relationship(
        "DatasetVersion", foreign_keys=[input_version_id]
    )
    output_version = relationship(
        "DatasetVersion", foreign_keys=[output_version_id]
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Operation id={self.id} type={self.operation_type!r} "
            f"status={self.status!r}>"
        )
