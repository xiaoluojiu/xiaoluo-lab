"""Prompt 009/010 测试：数据库基础设施与 Base Model。"""

from __future__ import annotations

from datetime import datetime

import pytest
from app.core.database import Base, SessionLocal, engine, get_db
from app.models.base import BaseModel
from sqlalchemy import String, create_engine, select, text
from sqlalchemy.orm import Mapped, mapped_column


def test_engine_select_1():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_get_db_yields_session_and_closes():
    from unittest import mock

    from sqlalchemy.orm import Session

    with mock.patch.object(Session, "close", autospec=True) as spy:
        gen = get_db()
        db = next(gen)
        assert db.execute(text("SELECT 1")).scalar() == 1
        with pytest.raises(StopIteration):
            next(gen)
        spy.assert_called_once_with(db)


def test_sessionmaker_usable():
    with SessionLocal() as session:
        assert session.execute(text("SELECT 1")).scalar() == 1


def test_base_model_columns():
    """BaseModel 提供 id / created_at / updated_at。"""

    class TmpItem(BaseModel):
        __tablename__ = "tmp_item_base_test"
        name: Mapped[str] = mapped_column(String(50))

    tmp_engine = create_engine("sqlite://")
    Base.metadata.create_all(tmp_engine)
    with SessionLocal(bind=tmp_engine) as session:
        item = TmpItem(name="hello")
        session.add(item)
        session.commit()
        row = session.execute(select(TmpItem)).scalar_one()
        assert row.id == 1
        assert isinstance(row.created_at, datetime)
        assert isinstance(row.updated_at, datetime)


def test_base_allows_partial_models():
    """不强制所有模型拥有相同字段：只用 IdMixin 也可建表。"""
    from app.core.database import Base
    from app.models.base import IdMixin

    class TmpOnlyId(IdMixin, Base):
        __tablename__ = "tmp_only_id_test"
        label: Mapped[str] = mapped_column(String(20))

    tmp_engine = create_engine("sqlite://")
    Base.metadata.create_all(tmp_engine)
    with SessionLocal(bind=tmp_engine) as session:
        obj = TmpOnlyId(label="x")
        session.add(obj)
        session.commit()
        assert obj.id == 1
        assert not hasattr(obj, "created_at")
