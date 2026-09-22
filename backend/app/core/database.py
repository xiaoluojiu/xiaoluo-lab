"""SQLAlchemy 数据库基础设施。"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import BACKEND_ROOT, settings


def _ensure_sqlite_dir(url: str) -> None:
    """确保 SQLite 数据库目录存在。"""
    if not url.startswith("sqlite"):
        return

    raw_path = (
        url.split("///", 1)[-1]
        if "///" in url
        else url
    )

    path = Path(raw_path)

    if not path.is_absolute():
        path = (BACKEND_ROOT / path).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _make_engine(url: str):
    """创建数据库 Engine。"""
    kwargs = {
        "pool_pre_ping": True,
        "future": True,
    }

    if url.startswith("sqlite"):
        _ensure_sqlite_dir(url)
        kwargs["connect_args"] = {
            "check_same_thread": False,
        }

    return create_engine(
        url,
        **kwargs,
    )


engine = _make_engine(
    settings.database_url
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """ORM 基类。"""


def get_db() -> Generator[Session, None, None]:
    """提供请求级数据库 Session。"""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
