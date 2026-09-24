"""ExperimentRun 数据库模型。

一次实验运行的状态与产物：
- status: pending / running / success / failed
- metrics: 评估指标（JSON）
- artifacts: 产物元信息（JSON；如特征清单、预处理报告）
- runtime: 运行耗时（秒）
- error: 失败原因
"""

from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class ExperimentRun(BaseModel):
    __tablename__ = "experiment_runs"

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifacts: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment = relationship("Experiment", back_populates="runs")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ExperimentRun id={self.id} experiment={self.experiment_id} "
            f"status={self.status!r}>"
        )
