"""SQLAlchemy Base Model。

提供 id / created_at / updated_at 基础能力，但不强制所有模型继承：
- 需要 id 与时间戳的模型 -> 继承 BaseModel
- 只需要部分能力的模型 -> 直接继承 app.core.database.Base 并自行选择 Mixin
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class IdMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class BaseModel(IdMixin, TimestampMixin, Base):
    """带 id + created_at + updated_at 的可选基类。"""

    __abstract__ = True
