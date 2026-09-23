"""SQLAlchemy 数据库基础设施。"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import BACKEND_ROOT, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# SQLite 在默认 journal 模式下一个写事务会锁住整个库：Agent 运行过程中要持续
# 落 tire ipynb / SSE 事件与工具记录，读请求同时又在刷列表 —— 表现为莫名的
# `database is locked`。WAL 让读写不再互斥，是所有 SQLite 部署的第一步。
_SQLITE_PRAGMAS = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("busy_timeout", "5000"),
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
    """为每条 SQLite 连接设置 PRAGMA。

    用 ``connect`` 事件而不是启动时一次性执行：连接池会新建/复用连接，
    只对某一条生效时另一条仍会退回默认行为。

    ``synchronous=NORMAL`` 而非 ``OFF``：NORMAL 在 WAL 下已经足够快，
    且崩溃时最多丢最后一个 checkpoint 之后的事务，不会静默损坏库文件。
    """
    if type(dbapi_connection).__module__ != "sqlite3":  # pragma: no cover - 方言差异
        return
    cursor = dbapi_connection.cursor()
    try:
        for name, value in _SQLITE_PRAGMAS:
            cursor.execute(f"PRAGMA {name}={value}")
    finally:
        cursor.close()


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


def check_runtime_configuration() -> list[str]:
    """启动体检：返回一组「能跑但跑不好」的配置告警。

    刻意返回列表而非直接抛异常——缺 Key 时平台仍然可用（规则规划器 + 本地
    静态分析都能跑），硬启动失败只会让排障更难。调用方负责把结果写进日志。
    """
    warnings: list[str] = []
    env = (settings.APP_ENV or "").strip().lower()

    if env in {"prod", "production"}:
        if not settings.LLM_API_KEY:
            warnings.append(
                "APP_ENV=prod 但未配置 LLM_API_KEY：Agent 会静默退化为平台规则应答，"
                "用户看到的是『能问能答但不会调工具』，请在 backend/.env 或部署环境中配置。"
            )
        if settings.DEBUG:
            warnings.append("生产环境开启了 DEBUG=true：错误堆栈可能随响应返回给浏览器。")
        if "*" in settings.cors_allowed_origins:
            warnings.append("CORS_ALLOW_ORIGINS 包含通配符 *：任意站点都能调用本机写接口。")
        if "sqlite" in settings.database_url:
            warnings.append(
                "生产环境仍使用 SQLite：单实例可用，但写并发受限且无法水平扩展，"
                "请在正式部署前评估切换到 PostgreSQL/MySQL。"
            )
    elif not settings.LLM_API_KEY:
        warnings.append(
            "未配置 LLM_API_KEY（当前 APP_ENV=%s）：远程大模型不可用，"
            "Agent 将走平台自带的规则规划器。" % (settings.APP_ENV or "dev")
        )
    return warnings
