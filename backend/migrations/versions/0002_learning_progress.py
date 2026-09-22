"""learning progress + attempts

Revision ID: 0002_learning_progress
Revises: 0001_initial
Create Date: 2026-09-21

新增学习中心的两张表：
  - learning_progress：每个 (learner_id, experiment_key) 一行，记录当前状态
  - learning_attempts：每次提交的过程记录，用于「练习曲线」

设计说明见 app/models/learning.py 的模块 docstring。
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_learning_progress"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_progress",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("learner_id", sa.String(length=64), nullable=False),
        sa.Column("experiment_key", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=24), nullable=False, server_default="in_progress"
        ),
        sa.Column("checklist", sa.JSON(), nullable=True),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_passed", sa.Boolean(), nullable=True),
        sa.Column("last_result", sa.JSON(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "learner_id", "experiment_key", name="uq_learning_learner_exp"
        ),
    )
    op.create_index(
        "ix_learning_progress_learner_id", "learning_progress", ["learner_id"]
    )
    op.create_index(
        "ix_learning_progress_experiment_key", "learning_progress", ["experiment_key"]
    )
    op.create_index(
        "ix_learning_learner_status", "learning_progress", ["learner_id", "status"]
    )

    op.create_table(
        "learning_attempts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "progress_id",
            sa.Integer(),
            sa.ForeignKey("learning_progress.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("score", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "passed", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("checks", sa.JSON(), nullable=True),
        sa.Column("output_summary", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_learning_attempts_progress_id", "learning_attempts", ["progress_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_learning_attempts_progress_id", table_name="learning_attempts")
    op.drop_table("learning_attempts")
    op.drop_index("ix_learning_learner_status", table_name="learning_progress")
    op.drop_index(
        "ix_learning_progress_experiment_key", table_name="learning_progress"
    )
    op.drop_index("ix_learning_progress_learner_id", table_name="learning_progress")
    op.drop_table("learning_progress")
