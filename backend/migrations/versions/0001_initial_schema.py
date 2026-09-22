"""initial schema (Prompt 206)

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-16

覆盖全部数据库模型：
files / datasets / dataset_versions / operations / experiments / experiment_runs
（Workflow 与 AgentRun 第一版为进程内持久化，不入库）
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _base_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    # ---- files ----
    op.create_table(
        "files",
        *_base_columns(),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False, unique=True),
        sa.Column(
            "size", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "format", sa.String(length=32), nullable=False, server_default=""
        ),
        sa.Column("checksum", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_files_name", "files", ["name"], unique=True)
    op.create_index("ix_files_path", "files", ["path"], unique=True)
    op.create_index("ix_files_checksum", "files", ["checksum"])

    # ---- datasets ----
    op.create_table(
        "datasets",
        *_base_columns(),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_file_id", sa.Integer(), sa.ForeignKey("files.id"), nullable=True),
    )
    op.create_index("ix_datasets_name", "datasets", ["name"])

    # ---- dataset_versions ----
    op.create_table(
        "dataset_versions",
        *_base_columns(),
        sa.Column(
            "dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "parent_version_id",
            sa.Integer(),
            sa.ForeignKey("dataset_versions.id"),
            nullable=True,
        ),
        sa.Column("storage_path", sa.String(length=512), nullable=False, unique=True),
        sa.Column(
            "format", sa.String(length=32), nullable=False, server_default="parquet"
        ),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "column_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("schema", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.create_index("ix_dataset_versions_dataset_id", "dataset_versions", ["dataset_id"])
    op.create_index("ix_dataset_versions_version", "dataset_versions", ["version"])
    op.create_index(
        "ix_dataset_versions_storage_path", "dataset_versions", ["storage_path"], unique=True
    )

    # ---- operations ----
    op.create_table(
        "operations",
        *_base_columns(),
        sa.Column(
            "dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False
        ),
        sa.Column(
            "input_version_id",
            sa.Integer(),
            sa.ForeignKey("dataset_versions.id"),
            nullable=True,
        ),
        sa.Column(
            "output_version_id",
            sa.Integer(),
            sa.ForeignKey("dataset_versions.id"),
            nullable=True,
        ),
        sa.Column("operation_type", sa.String(length=64), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="success"
        ),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_operations_dataset_id", "operations", ["dataset_id"])
    op.create_index("ix_operations_input_version_id", "operations", ["input_version_id"])
    op.create_index("ix_operations_output_version_id", "operations", ["output_version_id"])
    op.create_index("ix_operations_operation_type", "operations", ["operation_type"])
    op.create_index("ix_operations_status", "operations", ["status"])

    # ---- experiments ----
    op.create_table(
        "experiments",
        *_base_columns(),
        sa.Column(
            "dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False
        ),
        sa.Column(
            "dataset_version_id",
            sa.Integer(),
            sa.ForeignKey("dataset_versions.id"),
            nullable=False,
        ),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("target_column", sa.String(length=255), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("preprocessing", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_experiments_dataset_id", "experiments", ["dataset_id"])
    op.create_index(
        "ix_experiments_dataset_version_id", "experiments", ["dataset_version_id"]
    )
    op.create_index("ix_experiments_task", "experiments", ["task"])

    # ---- experiment_runs ----
    op.create_table(
        "experiment_runs",
        *_base_columns(),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("artifacts", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("runtime", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_experiment_runs_experiment_id", "experiment_runs", ["experiment_id"])
    op.create_index("ix_experiment_runs_status", "experiment_runs", ["status"])


def downgrade() -> None:
    op.drop_table("experiment_runs")
    op.drop_table("experiments")
    op.drop_table("operations")
    op.drop_table("dataset_versions")
    op.drop_table("datasets")
    op.drop_table("files")
