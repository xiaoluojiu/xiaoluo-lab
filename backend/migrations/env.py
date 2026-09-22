"""Alembic 环境（Prompt 206）。

数据库 URL 统一来自 app.core.config.settings.DATABASE_URL；
元数据来自 app.core.database.Base，显式导入全部模型模块以注册表结构。
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from app.core.config import settings
from app.core.database import Base

# 显式导入全部模型（确保 __tablename__ 注册进 Base.metadata）
import app.models.file  # noqa: F401
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.operation  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.learning  # noqa: F401

config = context.config

# 迁移脚本内可用 config.attributes / logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 覆盖 alembic.ini 中的占位 URL
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL，不连接数据库。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行迁移。"""
    connectable = config.attributes.get("connection", None)
    if connectable is None:
        from sqlalchemy import engine_from_config, pool

        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite 修改列需要 batch 模式
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
