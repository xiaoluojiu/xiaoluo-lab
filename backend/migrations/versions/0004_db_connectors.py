"""database connectors

Revision ID: 0004_db_connectors
Revises: 0003_custom_cards
Create Date: 2026-09-23

新增「拓展功能 · 数据库连接器」表 db_connectors：
  - 存外部数据库的连接配置（host / port / database / schema / username）
  - password_enc 只存密文（Fernet，见 app/connectors/crypto.py），
    任何 API 出参都不含该列
  - last_status / last_error / last_checked_at：把「最近一次试连结论」落在表上，
    用户回到页面即可看到故障，而不用反复手点「测试连接」
  - dataset_id / last_import_json：记录最近一次导入的目标与抽取指标
    （strategy / batches / warnings），配合 DatasetVersion 可复盘数据来源

设计说明见 app/models/connector.py 的 DbConnector 注释。
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_db_connectors"
down_revision: Union[str, None] = "0003_custom_cards"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "db_connectors",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("dialect", sa.String(length=32), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=True),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("database", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("schema_name", sa.String(length=128), nullable=True),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("password_enc", sa.Text(), nullable=False, server_default=""),
        sa.Column("options_json", sa.JSON(), nullable=True),
        sa.Column(
            "last_status", sa.String(length=16), nullable=False, server_default="unknown"
        ),
        sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_checked_at", sa.DateTime(), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("last_import_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_db_connectors_dataset_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_db_connectors_name", "db_connectors", ["name"], unique=True)
    op.create_index("ix_db_connectors_dialect", "db_connectors", ["dialect"])


def downgrade() -> None:
    op.drop_index("ix_db_connectors_dialect", table_name="db_connectors")
    op.drop_index("ix_db_connectors_name", table_name="db_connectors")
    op.drop_table("db_connectors")
