"""数据库方言注册表。

为什么要有这一层
----------------
「支持 Postgres / MySQL / SQL Server」并不只是拼一个连接串——每个方言在
四件事上都不一样，写成一堆 ``if dialect == "mysql"`` 会散落到抽取、内省、
分页、引号处理各处：

1. **URL 形态**：SQLite 是本地文件路径，其余是 ``dialect+driver://user:pw@host:port/db``；
2. **驱动依赖**：``psycopg``/``pymysql``/``pyodbc`` 都是可选依赖，缺失时必须给出
   「装哪个包」的可操作提示，而不是 ``ModuleNotFoundError`` 堆栈；
3. **表结构内省**：SQLAlchemy 的 ``Inspector`` 已能跨库，但「当前 schema / 数据库」
   的默认值不同（Postgres 是 ``public``，MySQL 是库名，SQLite 无 schema 概念）；
4. **标识符引用**：``"col"`` 与 `` `col` `` 的差异；同时这是**注入面**——
   表名/列名来自用户输入，必须走 ``prepare_identifier`` 白名单校验而非字符串拼接。

把差异收敛到 ``DialectSpec`` 之后，``extract`` 与 ``service`` 都只面向抽象，
新增一个方言 = 新增一行注册，不需要改抽取逻辑。
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.data_engine.exceptions import DataEngineException

# 标识符白名单：字母/下划线开头，允许字母数字下划线，允许 库名.表名 两级。
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_QUALIFIED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*\.[A-Za-z_][A-Za-z0-9_$]*$")


class ConnectorDialectError(DataEngineException):
    """方言不支持 / 驱动缺失 / 标识符非法。"""

    default_code = "CONNECTOR_DIALECT_ERROR"
    default_message = "数据库方言配置错误"


@dataclass(frozen=True)
class DialectSpec:
    """一个数据库方言的静态描述。"""

    name: str
    label: str
    # SQLAlchemy URL 前缀。``None`` 表示该方言不走 SQLAlchemy（保留给原生驱动）。
    url_scheme: str
    # 候选驱动包（按优先级）。第一个为推荐。
    drivers: tuple[str, ...] = ()
    default_port: int | None = None
    # 是否需要 host / port（SQLite 这类本地文件库不需要）。
    requires_host: bool = True
    # 标识符引号字符。
    quote: str = '"'
    # 默认 schema：None 表示「由驱动决定」，空串表示「显式留空」。
    default_schema: str | None = None
    # 是否支持服务端游标（真流式）；不支持则走 keyset / limit-offset。
    server_side_cursor: bool = False
    notes: str = ""
    # 是否允许出现在配置白名单里时自动放开（保留字段，便于将来做灰度）。
    extra: dict = field(default_factory=dict)

    # ---- 能力探测 ----

    @property
    def driver_installed(self) -> bool:
        """是否至少有一个候选驱动可用。"""
        if not self.drivers:
            # 无外部驱动依赖（如 SQLite 走标准库）。
            return True
        return any(importlib.util.find_spec(driver) is not None for driver in self.drivers)

    def missing_driver_hint(self) -> str:
        """驱动缺失时的可操作提示。"""
        if self.drivers:
            return f"pip install {' '.join(self.drivers)}"
        return ""

    # ---- URL 构造 ----

    def build_url(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        options: dict | None = None,
    ) -> str:
        from app.connectors.dialects import build_url as _build_url

        return _build_url(
            self,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            options=options,
        )

    # ---- 标识符安全 ----

    def quote_identifier(self, name: str) -> str:
        """给标识符加引号（并做白名单校验，防注入）。

        注意：这里**不接受任意字符串**。``name`` 必须匹配
        ``[A-Za-z_][A-Za-z0-9_$]*``（可含一级 ``schema.`` 前缀）。
        SQLAlchemy 的 ``quoted_name`` 只解决大小写语义，不解决注入，
        因此校验必须发生在拼 SQL 之前。
        """
        parts = name.split(".")
        if len(parts) > 2 or not all(_IDENTIFIER_RE.match(p) for p in parts):
            raise ConnectorDialectError(
                f"非法标识符：{name!r}（只允许字母/数字/下划线，可带一级 schema 前缀）",
                details={"identifier": name, "quote": self.quote},
            )
        q = self.quote
        return ".".join(f"{q}{p}{q}" for p in parts)

    def qualify(self, table: str, schema: str | None = None) -> str:
        """拼出「schema.table」形式的引用标识符。"""
        target = (schema or self.default_schema or "").strip()
        if target and "." not in table:
            return self.quote_identifier(f"{target}.{table}")
        return self.quote_identifier(table)


def _url_escape(value: str) -> str:
    """URL 组件转义（口令里常见 @ : / # ，不转义会把 URL 拆坏）。"""
    from urllib.parse import quote_plus

    return quote_plus(value)


def build_url(
    spec: DialectSpec,
    *,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    options: dict | None = None,
) -> str:
    """构造 SQLAlchemy 连接 URL。

    SQLite 特殊：``database`` 就是文件路径，且允许 ``:memory:``。
    """
    opts = {k: v for k, v in (options or {}).items() if v not in (None, "")}

    if spec.name == "sqlite":
        if not database:
            raise ConnectorDialectError(
                "SQLite 连接需要提供数据库文件路径（database 字段）",
                details={"dialect": spec.name},
            )
        if database == ":memory:":
            return "sqlite+pysqlite:///:memory:"
        path = Path(database).expanduser()
        # 统一成 POSIX 风格绝对路径：Windows 反斜杠在 sqlite URL 里会被吞。
        return f"sqlite+pysqlite:///{path.resolve().as_posix()}"

    if not database:
        raise ConnectorDialectError(
            f"{spec.label} 连接需要提供数据库名（database 字段）",
            details={"dialect": spec.name},
        )

    credentials = ""
    if username:
        credentials = _url_escape(username)
        if password:
            credentials += f":{_url_escape(password)}"
        credentials += "@"

    netloc = f"{credentials}{host or 'localhost'}"

    effective_port = port or spec.default_port
    if effective_port:
        netloc += f":{int(effective_port)}"

    url = f"{spec.url_scheme}://{netloc}/{_url_escape(database)}"

    # 超时与连接池相关参数通过 query string 传递（各驱动命名不同，按方言映射）。
    query: list[str] = []
    timeout = int(getattr(settings, "CONNECTOR_CONNECT_TIMEOUT_SECONDS", 10) or 10)
    if spec.name == "postgresql":
        query.append(f"connect_timeout={timeout}")
    elif spec.name == "mysql":
        query.append(f"connect_timeout={timeout}")
        query.append("charset=utf8mb4")
    elif spec.name == "mssql":
        query.append(f"timeout={timeout}")
    elif spec.name == "oracle":
        query.append(f"tcp_connect_timeout={timeout}")

    for key, value in opts.items():
        query.append(f"{key}={_url_escape(str(value))}")

    if query:
        url += "?" + "&".join(query)

    return url


# ---------------------------------------------------------
# 注册表
# ---------------------------------------------------------

DIALECTS: dict[str, DialectSpec] = {
    "sqlite": DialectSpec(
        name="sqlite",
        label="SQLite",
        url_scheme="sqlite+pysqlite",
        drivers=(),  # 标准库 sqlite3，无需外部依赖
        requires_host=False,
        quote='"',
        server_side_cursor=False,
        notes="本地文件库，零外部依赖；适合演示与轻量场景。",
    ),
    "postgresql": DialectSpec(
        name="postgresql",
        label="PostgreSQL",
        url_scheme="postgresql+psycopg",
        drivers=("psycopg", "psycopg2"),
        default_port=5432,
        quote='"',
        default_schema="public",
        server_side_cursor=True,
        notes="支持服务端游标（named cursor），可对超大表做真流式抽取。",
    ),
    "mysql": DialectSpec(
        name="mysql",
        label="MySQL / MariaDB",
        url_scheme="mysql+pymysql",
        drivers=("pymysql", "MySQLdb"),
        default_port=3306,
        quote="`",
        server_side_cursor=True,
        notes="使用 SSCursor（无缓冲游标）实现流式，避免整结果集驻留客户端内存。",
    ),
    "mssql": DialectSpec(
        name="mssql",
        label="SQL Server",
        url_scheme="mssql+pyodbc",
        drivers=("pyodbc",),
        default_port=1433,
        quote='"',
        default_schema="dbo",
        server_side_cursor=False,
        notes="需要系统安装 ODBC Driver；通过 pyodbc 连接。",
    ),
    "oracle": DialectSpec(
        name="oracle",
        label="Oracle",
        url_scheme="oracle+oracledb",
        drivers=("oracledb",),
        default_port=1521,
        quote='"',
        server_side_cursor=False,
        notes="使用 python-oracledb 瘦客户端模式，无需安装 Oracle Client。",
    ),
}

# 公开可枚举的顺序（前端下拉框按此渲染）
SUPPORTED_DIALECTS: tuple[str, ...] = tuple(DIALECTS)


def get_dialect(name: str) -> DialectSpec:
    """按名称取方言；不存在或未在白名单内时抛出可操作的错误。"""
    key = (name or "").strip().lower()

    spec = DIALECTS.get(key)
    if spec is None:
        raise ConnectorDialectError(
            f"不支持的数据库类型：{name!r}",
            details={"dialect": name, "supported": list(SUPPORTED_DIALECTS)},
        )

    allowed_value = getattr(settings, "allowed_connector_dialects", None)
    if callable(allowed_value):
        allowed_tuple = tuple(allowed_value())
    elif allowed_value is None:
        allowed_tuple = SUPPORTED_DIALECTS
    else:
        allowed_tuple = tuple(allowed_value)

    if allowed_tuple and key not in allowed_tuple:
        raise ConnectorDialectError(
            f"数据库类型 {spec.label} 未在部署白名单中启用",
            details={
                "dialect": key,
                "allowed": list(allowed_tuple),
                "how_to_enable": "设置环境变量 CONNECTOR_ALLOWED_DIALECTS 后重启后端",
            },
        )

    return spec


def dialect_catalog() -> list[dict]:
    """方言清单（供前端下拉框展示，含驱动是否就绪）。"""
    allowed = set(getattr(settings, "allowed_connector_dialects", ()) or ())
    catalog: list[dict] = []
    for spec in DIALECTS.values():
        catalog.append(
            {
                "name": spec.name,
                "label": spec.label,
                "default_port": spec.default_port,
                "requires_host": spec.requires_host,
                "driver_installed": spec.driver_installed,
                "driver_hint": spec.missing_driver_hint(),
                "enabled": (not allowed) or spec.name in allowed,
                "server_side_cursor": spec.server_side_cursor,
                "notes": spec.notes,
            }
        )
    return catalog
