"""LearningProgress：学习中心的学习进度记录。

设计取舍
--------
为什么建表而不是写 localStorage：
    学习中心的核心价值在于「学习者能看到自己的进度」。localStorage 只能
    存在单台浏览器上，换设备 / 清缓存即丢，也无法在做完成率统计。
    一张表换来的是：真正的进度闭环 + 可统计 + 可写进论文的量化数据。

为什么用 (learner_id, experiment_key) 唯一约束而不是直接挂 Experiment：
    学习实验是「教学任务」，与训练出来的 Experiment 是两件事——
    一个学习任务可以反复做（多次训练），而进度只关心「最后是否达标」。
    用 experiment_key 关联目录里的任务定义，两者解耦。

为什么把 attempts 单独一张表：
    attempts 承载「做过几次、每次什么结果」，是过程数据；
    progress 承载「当前状态」，是结果数据。分开后 progress 永远只有一行，
    更新是幂等的，不会被并发写坏。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

# 学习状态机：未开始 -> 进行中 -> 已完成（可回退到进行中继续改）
LEARNING_STATUSES = ("not_started", "in_progress", "completed")


class LearningProgress(BaseModel):
    """一个学习者在某个学习实验上的当前进度（每个实验恒为一行）。"""

    __tablename__ = "learning_progress"
    __table_args__ = (
        UniqueConstraint("learner_id", "experiment_key", name="uq_learning_learner_exp"),
        Index("ix_learning_learner_status", "learner_id", "status"),
    )

    learner_id: Mapped[str] = mapped_column(String(64), index=True)
    # 对应 learning/catalog.py 中的 experiment key（如 "iris-classification"）
    experiment_key: Mapped[str] = mapped_column(String(64), index=True)
    # not_started / in_progress / completed
    status: Mapped[str] = mapped_column(String(24), default="in_progress")
    # 任务要求的逐条勾选状态：{"明确特征X与目标y": true, ...}
    checklist: Mapped[dict] = mapped_column(JSON, default=dict)
    # 最近一次提交的代码（用于恢复现场；不做版本历史）
    code: Mapped[str] = mapped_column(Text, default="")
    # 最近一次选用的数据集
    dataset_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 累计尝试次数（提交/运行次数）
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    # 最近一次是否通过全部硬性检查
    last_passed: Mapped[bool | None] = mapped_column(nullable=True)
    # 最近一次检查的完整结果（checks / hints / score）
    last_result: Mapped[dict] = mapped_column(JSON, default=dict)
    # 首次达标时间（用于「完成于」展示）
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    attempts_log = relationship(
        "LearningAttempt",
        back_populates="progress",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LearningProgress learner={self.learner_id!r} "
            f"key={self.experiment_key!r} status={self.status!r}>"
        )


class LearningAttempt(BaseModel):
    """一次学习提交的过程记录（保留最近若干次，用于「练习曲线」）。"""

    __tablename__ = "learning_attempts"

    progress_id: Mapped[int] = mapped_column(
        ForeignKey("learning_progress.id", ondelete="CASCADE"), index=True
    )
    # 这次提交的检查得分（0-100）
    score: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[bool] = mapped_column(default=False)
    # 每条检查项的通过情况，便于回看「哪一步一直卡住」
    checks: Mapped[list] = mapped_column(JSON, default=list)
    # 运行时输出摘要（stdout 前若干行 / 错误信息）
    output_summary: Mapped[str] = mapped_column(Text, default="")

    progress = relationship("LearningProgress", back_populates="attempts_log")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LearningAttempt progress={self.progress_id} score={self.score}>"


# 自定义学习卡片的形态
CUSTOM_CARD_KINDS = ("practice", "checklist")


class LearningCustomCard(BaseModel):
    """学习者自建的学习卡片（与内置实验目录解耦）。

    设计取舍
    --------
    内置实验是「有判定规则的客观任务」（AST + 数据探针），自建卡片则是
    「辅助学习」的轻量载体：练习卡（practice）写代码 + 交给 AI 点评，
    清单卡（checklist）记录要点 + 手动勾选。两者都不做客观判分——
    因为一旦判分，就必须让用户自己写规则，等于把「导入式」的复杂度
    换个地方塞回来。自建卡片的定位是「轻量、可随时建、不打断主线」。
    """

    __tablename__ = "learning_custom_cards"

    learner_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(128), default="")
    # practice（练习卡）/ checklist（清单卡）
    kind: Mapped[str] = mapped_column(String(16), default="practice")
    # 目标描述：练习卡用于 AI 点评上下文；清单卡可为空
    goal: Mapped[str] = mapped_column(Text, default="")
    # 起始 / 当前代码（仅练习卡使用）
    code: Mapped[str] = mapped_column(Text, default="")
    # 要点列表（仅清单卡使用）：[{"text": "...", "done": false}, ...]
    items: Mapped[list] = mapped_column(JSON, default=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LearningCustomCard id={self.id} learner={self.learner_id!r} "
            f"kind={self.kind!r} title={self.title!r}>"
        )
