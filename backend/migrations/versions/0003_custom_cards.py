"""learning custom cards

Revision ID: 0003_custom_cards
Revises: 0002_learning_progress
Create Date: 2026-09-21

新增学习中心的自建卡片表 learning_custom_cards：
  - 与内置实验目录解耦，承载「辅助学习」的轻量载体
  - kind：practice（练习卡，写代码 + AI 点评）/ checklist（清单卡，手动勾选）
  - 不做客观判分（判分需要用户自写规则，复杂度又绕回来了）

设计说明见 app/models/learning.py 的 LearningCustomCard 注释。
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_custom_cards"
down_revision: Union[str, None] = "0002_learning_progress"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_custom_cards",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("learner_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False, server_default=""),
        sa.Column(
            "kind", sa.String(length=16), nullable=False, server_default="practice"
        ),
        sa.Column("goal", sa.Text(), nullable=True),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("items", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_learning_custom_cards_learner_id",
        "learning_custom_cards",
        ["learner_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_learning_custom_cards_learner_id", table_name="learning_custom_cards")
    op.drop_table("learning_custom_cards")
