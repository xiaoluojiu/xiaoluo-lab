"""DatasetVersion 数据库模型。

重要：DatasetVersion 默认不可变（immutable）。
任何数据修改都产生新版本（新行 + 新 storage_path），
本模型不提供 update 方法，Repository 层也禁止更新版本数据内容。
"""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class DatasetVersion(BaseModel):
    __tablename__ = "dataset_versions"
    # 「同 dataset 内版本号唯一」必须由数据库裁决，不能靠应用层先读 max()+1。
    #
    # 背景：`storage_path UNIQUE` 拦不住并发——并发请求算出同一个 version 时，
    # 派生出的 storage key 也相同，两个请求写的是同一个文件，约束无从生效。
    # 只有 (dataset_id, version) 唯一才能真正保证「v8 只有一个」。
    # 版本号预定就是靠这条约束的 IntegrityError 做仲裁（见 DatasetService）。
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "version",
            name="uq_dataset_versions_dataset_id_version",
        ),
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"), index=True
    )
    # 单调递增版本号（从 1 开始，dataset 内唯一）
    version: Mapped[int] = mapped_column(Integer, index=True)
    # 父版本（首个版本为 NULL）
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True
    )
    # 快照在 Storage 层的 key（parquet 文件）
    storage_path: Mapped[str] = mapped_column(String(512), unique=True)
    format: Mapped[str] = mapped_column(String(32), default="parquet")
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    # 列 schema：{"col": "Int64", ...}（数据库列名为 schema）
    schema_json: Mapped[dict] = mapped_column("schema", JSON, default=dict)

    dataset = relationship("Dataset", back_populates="versions")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DatasetVersion id={self.id} dataset={self.dataset_id} "
            f"v={self.version}>"
        )
