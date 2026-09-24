"""Experiment 数据库模型。

一次实验 = 绑定一个 DatasetVersion + 任务类型 + 模型 + 参数 + 预处理配置，
保证实验可复现（同一数据版本、同一参数、同一 seed）。
"""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Experiment(BaseModel):
    __tablename__ = "experiments"

    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    # 绑定的数据版本行 id（不可变快照）
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    # classification / regression / clustering
    task: Mapped[str] = mapped_column(String(32), index=True)
    # Model Registry 中的模型名
    model: Mapped[str] = mapped_column(String(64))
    # 目标列（聚类为 NULL）
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 模型超参数（透传给 ModelAdapter）
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    # 预处理配置（missing/encoding/scaling），由 PreprocessingPipeline 消费
    preprocessing: Mapped[dict] = mapped_column(JSON, default=dict)
    # 随机种子（保证可复现；支持 random_state 的模型会注入）
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")

    dataset = relationship("Dataset")
    version = relationship("DatasetVersion")
    runs = relationship(
        "ExperimentRun",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Experiment id={self.id} task={self.task!r} "
            f"model={self.model!r}>"
        )
