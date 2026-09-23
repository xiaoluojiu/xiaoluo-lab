"""分块抽取引擎：把外部数据库的整表/查询变成 Parquet 快照。

这是「突破文件体积限制」的落点
------------------------------
文件上传的瓶颈在「一次请求内搬运完整个文件」；而关系库的读取天然可分页，
因此可以做成：**拉一批 → 转一批 → 写一批**，全程不做全量物化。

抽取策略（按能力自动选择，并如实记录在结果里）
--------------------------------------------
1. ``server-side-cursor``（PostgreSQL / MySQL）：SQLAlchemy
   ``stream_results=True`` 让驱动使用服务端游标（命名游标 / SSCursor），
   数据留在服务端，客户端只按批取。这是唯一真正的「流式」，内存与表行数无关。
2. ``paged``（SQLite 等无服务端游标支持）：``LIMIT n OFFSET m`` 逐页取。
   正确但深层 OFFSET 会变慢（DB 仍需跳过前 m 行），因此额外支持 ``keyset``
   形式——若调用方给出单调唯一列（``order_by``），改写成
   ``WHERE key > :last ORDER BY key LIMIT n``，代价与页深无关。

批次落盘用 ``pyarrow.parquet.ParquetWriter`` **追加 row group**：
- schema 只在第一批确定，之后各批按同一 schema 写入，避免「不同批次各自推断
  类型」造成 Parquet 列类型漂移（这在「前几批全整数、突然出现 NULL」时很常见）；
- 写入是增量 flush 的，进程在中途失败最多损失最后一批，而不是整份数据。

安全边界
--------
表名 / 列名一律走 ``DialectSpec.quote_identifier`` 的白名单校验后再拼接；
``where`` 允许原始 SQL 片段（这是连接器的正常能力），但**拒绝分号**以阻断
多语句注入（``1=1; DROP TABLE ...``）。生产环境还应给连接器配置只读账号。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.pool import NullPool

from app.connectors.dialects import DialectSpec
from app.core.config import settings
from app.core.logging import get_logger
from app.data_engine.exceptions import DataEngineException

logger = get_logger("app.connectors.extract")

# WHERE 片段里禁止出现的模式：多语句 / 注释截断。
_FORBIDDEN_WHERE = re.compile(r";|--|/\*|\*/", re.IGNORECASE)


class ConnectorExtractError(DataEngineException):
    """抽取过程错误。"""

    default_code = "CONNECTOR_EXTRACT_ERROR"
    default_message = "数据库抽取失败"


# ---------------------------------------------------------
# 结果对象
# ---------------------------------------------------------


@dataclass
class ExtractResult:
    """一次抽取的统计与策略记录。"""

    dest_path: Path
    table: str
    strategy: str
    row_count: int
    column_count: int
    schema: dict[str, str] = field(default_factory=dict)
    batches: int = 0
    elapsed_seconds: float = 0.0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return round(self.row_count / self.elapsed_seconds, 1)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source": "database",
            "table": self.table,
            "strategy": self.strategy,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "batches": self.batches,
            "truncated": self.truncated,
            "elapsed_seconds": self.elapsed_seconds,
            "rows_per_second": self.rows_per_second,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------
# 连接
# ---------------------------------------------------------


def open_engine(url: str, spec: DialectSpec, *, pooled: bool = False) -> Engine:
    """按 URL 创建 SQLAlchemy Engine。

    ``pooled=False``（抽取场景默认）：用 ``NullPool`` —— 一次抽取就是一个短
    生命周期的连接，池化只会把「远端连接」和「本进程」绑得更久，既不提速
    也更容易在远端超时后留下坏连接。

    ``pooled=True``：试连 / 预览等高频短查询走连接池。
    """
    connect_args: dict[str, Any] = {}

    if spec.name == "sqlite":
        connect_args["check_same_thread"] = False
    elif spec.name == "postgresql":
        # 语句级超时由数据库端强制，避免一条慢查询把后端工作线程钉死。
        timeout_s = int(getattr(settings, "CONNECTOR_STATEMENT_TIMEOUT_SECONDS", 300) or 300)
        connect_args["options"] = f"-c statement_timeout={timeout_s * 1000}"

    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}

    if pooled and spec.name != "sqlite":
        kwargs["pool_size"] = max(1, int(getattr(settings, "CONNECTOR_POOL_SIZE", 5) or 5))
        kwargs["pool_recycle"] = 1800
    else:
        kwargs["poolclass"] = NullPool

    if connect_args:
        kwargs["connect_args"] = connect_args

    try:
        return create_engine(url, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 驱动缺失 / URL 非法
        hint = spec.missing_driver_hint()
        raise ConnectorExtractError(
            f"无法创建 {spec.label} 连接：{exc}"
            + (f"（可尝试：{hint}）" if hint else ""),
            details={"dialect": spec.name, "driver_hint": hint},
        ) from exc


def close_engine(engine: Engine | None) -> None:
    """释放连接池（NullPool 下等价于关闭已开连接）。"""
    if engine is None:
        return
    try:
        engine.dispose()
    except Exception:  # noqa: BLE001
        logger.warning("释放数据库引擎失败（已忽略）")


# ---------------------------------------------------------
# SQL 构造（标识符一律经方言校验）
# ---------------------------------------------------------


def _validate_where(where: str | None) -> str:
    fragment = (where or "").strip()
    if not fragment:
        return ""
    if _FORBIDDEN_WHERE.search(fragment):
        raise ConnectorExtractError(
            "WHERE 条件中包含分号或 SQL 注释，已拒绝（防多语句注入）",
            details={"where": fragment[:200]},
        )
    return fragment


def build_select(
    spec: DialectSpec,
    *,
    table: str,
    schema: str | None = None,
    columns: list[str] | None = None,
    where: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    keyset: tuple[str, Any] | None = None,
) -> str:
    """拼一条只读 SELECT。

    所有标识符（表 / 列 / keyset 列）都经过 ``quote_identifier`` 白名单校验，
    因此这里的字符串拼接不构成注入面；``where`` 片段单独做多语句检查。
    """
    projection = (
        ", ".join(spec.quote_identifier(col) for col in columns)
        if columns
        else "*"
    )

    sql = f"SELECT {projection} FROM {spec.qualify(table, schema)}"

    conditions: list[str] = []
    clause = _validate_where(where)
    if clause:
        conditions.append(f"({clause})")

    if keyset is not None:
        key_col, key_value = keyset
        conditions.append(f"{spec.quote_identifier(key_col)} > :_keyset_value")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    if order_by:
        sql += f" ORDER BY {spec.quote_identifier(order_by)}"

    # LIMIT/OFFSET 的占位符各驱动语法不同（MSSQL 用 TOP / OFFSET FETCH），
    # 这里只覆盖 LIMIT 方言；不支持的方言由调用方走服务端游标路径。
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
        if offset:
            sql += f" OFFSET {int(offset)}"

    return sql


# ---------------------------------------------------------
# 批次迭代
# ---------------------------------------------------------


def _rows_to_frame(rows: list[Any], columns: list[str], schema: pl.Schema | None) -> pl.DataFrame:
    """把一批 Row 转成 DataFrame；给定 schema 时强制对齐列类型。"""
    data = {name: [] for name in columns}
    for row in rows:
        mapping = row._mapping if hasattr(row, "_mapping") else row
        for name in columns:
            data[name].append(mapping.get(name) if hasattr(mapping, "get") else mapping[name])

    if schema is None:
        return pl.DataFrame(data)

    # strict=False：个别值超出统一 schema 时置 null 而不是整批失败。
    # 这在真实库上很常见（例如某几行时间字段存的是空字符串）。
    return pl.DataFrame(data, schema=schema, strict=False)


def iter_batches(
    engine: Engine,
    spec: DialectSpec,
    *,
    table: str,
    schema: str | None = None,
    columns: list[str] | None = None,
    where: str | None = None,
    order_by: str | None = None,
    batch_size: int | None = None,
    max_rows: int | None = None,
    keyset_column: str | None = None,
) -> Iterator[tuple[list[str], pl.DataFrame, str]]:
    """逐批产出 ``(列名, 数据帧, 策略名)``。

    策略选择：
    - 方言支持服务端游标 → ``server-side-cursor``（真流式）；
    - 否则若给出 ``keyset_column`` → ``keyset``（页深无关）；
    - 否则 → ``paged``（LIMIT/OFFSET）。
    """
    size = max(1, int(batch_size or getattr(settings, "CONNECTOR_BATCH_ROWS", 50000) or 50000))
    cap = int(max_rows) if max_rows else 0
    produced = 0

    common = {
        "table": table,
        "schema": schema,
        "columns": columns,
        "where": where,
        "order_by": order_by,
    }

    if spec.server_side_cursor:
        sql = build_select(spec, **common)
        conn = engine.connect()
        try:
            # stream_results + max_row_buffer：让驱动使用服务端游标并按批取回。
            result = conn.execution_options(
                stream_results=True, max_row_buffer=size
            ).execute(text(sql))

            keys = list(result.keys())
            frame_schema: pl.Schema | None = None
            for partition in result.partitions(size):
                rows = list(partition)
                if not rows:
                    continue
                if cap and produced + len(rows) > cap:
                    rows = rows[: max(0, cap - produced)]
                frame = _rows_to_frame(rows, keys, frame_schema)
                if frame_schema is None:
                    frame_schema = frame.schema
                produced += frame.height
                yield keys, frame, "server-side-cursor"
                if cap and produced >= cap:
                    break
        except DataEngineException:
            raise
        except Exception as exc:  # noqa: BLE001 - 归一为业务异常，附上 SQL 便于排查
            raise ConnectorExtractError(
                f"流式抽取失败：{exc}",
                details={"sql": sql, "strategy": "server-side-cursor"},
            ) from exc
        finally:
            conn.close()
        return

    # ---- 分页路径 ----
    keyset_col = (keyset_column or "").strip() or None
    use_keyset = bool(keyset_col)
    last_value: Any = None
    has_last_value = False
    offset = 0
    frame_schema = None

    while True:
        page_limit = size
        if cap:
            page_limit = min(page_limit, max(1, cap - produced))

        # 首页还没有游标值：不能加 `key > :_keyset_value`，否则绑定参数缺失。
        uses_keyset_condition = use_keyset and has_last_value

        sql = build_select(
            spec,
            **common,
            limit=page_limit,
            offset=None if use_keyset else offset,
            keyset=(keyset_col, last_value) if uses_keyset_condition else None,
        )

        # 游标值必须真正绑定；标识符由 build_select 校验并加引号，
        # 这里只把**值**作为绑定参数传入。
        params = {"_keyset_value": last_value} if uses_keyset_condition else None

        try:
            with engine.connect() as conn:
                rows = list(conn.execute(text(sql), params or {}).fetchall())
        except DataEngineException:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一归一为业务异常，附 SQL 便于排查
            raise ConnectorExtractError(
                f"分页抽取失败：{exc}",
                details={"sql": sql, "strategy": "keyset" if use_keyset else "paged"},
            ) from exc

        if not rows:
            break

        keys = list(rows[0]._mapping.keys()) if hasattr(rows[0], "_mapping") else []

        if use_keyset:
            mapping = rows[-1]._mapping
            last_value = mapping.get(keyset_col)
            has_last_value = True
            if last_value is None:
                # keyset 列出现空值 → 无法继续单调推进，退化为一次性 LIMIT 取完。
                raise ConnectorExtractError(
                    f"keyset 列 {keyset_col!r} 出现空值，无法继续分页；"
                    "请改用不分页方式或指定其他唯一列",
                    details={"keyset_column": keyset_col, "strategy": "keyset"},
                )
        else:
            offset += len(rows)

        frame = _rows_to_frame(rows, keys, frame_schema)
        if frame_schema is None:
            frame_schema = frame.schema

        produced += frame.height
        yield keys, frame, "keyset" if use_keyset else "paged"

        if cap and produced >= cap:
            return
        if len(rows) < page_limit:
            return


# ---------------------------------------------------------
# 抽取 → Parquet
# ---------------------------------------------------------


def extract_table_to_parquet(
    engine: Engine,
    spec: DialectSpec,
    dest: Path | str,
    *,
    table: str,
    schema: str | None = None,
    columns: list[str] | None = None,
    where: str | None = None,
    order_by: str | None = None,
    batch_size: int | None = None,
    max_rows: int | None = None,
    keyset_column: str | None = None,
) -> ExtractResult:
    """把表/查询抽成 Parquet，增量追加 row group。

    ``dest`` 应是临时路径：由 ``ConnectorService`` 决定最终存储 key 并通过
    ``storage.promote`` 原子落位（与 ``ingest`` 保持一致的约定）。
    """
    dest = Path(str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    warnings: list[str] = []
    writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None
    strategy = "unknown"
    row_count = 0
    batch_count = 0
    truncated = False

    row_group_rows = max(1024, int(getattr(settings, "INGEST_ROW_GROUP_ROWS", 262144) or 262144))

    try:
        for keys, frame, batch_strategy in iter_batches(
            engine,
            spec,
            table=table,
            schema=schema,
            columns=columns,
            where=where,
            order_by=order_by,
            batch_size=batch_size,
            max_rows=max_rows,
            keyset_column=keyset_column,
        ):
            strategy = batch_strategy
            table_arrow = frame.to_arrow()

            if writer is None:
                arrow_schema = table_arrow.schema
                writer = pq.ParquetWriter(
                    str(dest),
                    arrow_schema,
                    compression="zstd",
                )

            # 各批按同一 schema 写入；列顺序由第一批固定。
            if table_arrow.schema != arrow_schema:
                table_arrow = table_arrow.cast(arrow_schema)

            # 按目标行组大小切分，避免单批过大形成巨型 row group。
            for offset in range(0, table_arrow.num_rows, row_group_rows):
                writer.write_table(table_arrow.slice(offset, row_group_rows))

            row_count += table_arrow.num_rows
            batch_count += 1

        if writer is None:
            # 空表：没有任何批次，但必须产出一个合法 Parquet（否则后续读取报错）。
            columns_hint = columns or []
            empty_schema = pa.schema([(c, pa.string()) for c in columns_hint]) or pa.schema(
                [("__empty__", pa.string())]
            )
            writer = pq.ParquetWriter(str(dest), empty_schema, compression="zstd")
            warnings.append(
                "查询结果为空：已产出零行 Parquet，列类型按字符串占位"
                + ("（未指定 columns，无法还原真实类型）" if not columns_hint else "")
            )
    finally:
        if writer is not None:
            writer.close()

    if max_rows and row_count >= int(max_rows):
        truncated = True
        warnings.append(f"已达单次抽取行数上限 {max_rows}，结果被截断")

    schema_map = (
        {field.name: str(field.type) for field in arrow_schema} if arrow_schema else {}
    )

    return ExtractResult(
        dest_path=dest,
        table=table,
        strategy=strategy,
        row_count=row_count,
        column_count=len(schema_map),
        schema=schema_map,
        batches=batch_count,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        truncated=truncated,
        warnings=warnings,
    )


# ---------------------------------------------------------
# 轻量查询（试连 / 列表 / 列结构 / 预览）
# ---------------------------------------------------------


def list_schemas(engine: Engine, spec: DialectSpec) -> list[str]:
    """列出可见 schema（SQLite 无此概念，返回空列表）。"""
    if spec.name == "sqlite":
        return []
    try:
        return list(inspect(engine).get_schema_names())
    except Exception:  # noqa: BLE001 - 权限不足时不应阻断
        return []


def list_tables(
    engine: Engine,
    spec: DialectSpec,
    *,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    """列出表与视图。"""
    inspector = inspect(engine)
    target = schema or spec.default_schema
    try:
        tables = inspector.get_table_names(schema=target)
        views = inspector.get_view_names(schema=target)
    except Exception as exc:  # noqa: BLE001
        raise ConnectorExtractError(
            f"读取表清单失败：{exc}",
            details={"dialect": spec.name, "schema": target},
        ) from exc

    return [
        {"name": name, "type": "table", "schema": target} for name in sorted(tables)
    ] + [{"name": name, "type": "view", "schema": target} for name in sorted(views)]


def describe_table(
    engine: Engine,
    spec: DialectSpec,
    *,
    table: str,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    """列出列结构（名称 / 类型 / 可空 / 主键）。"""
    inspector = inspect(engine)
    target = schema or spec.default_schema

    try:
        columns = inspector.get_columns(table, schema=target)
        pk = []
        try:
            pk = list(inspector.get_pk_constraint(table, schema=target).get("constrained_columns") or [])
        except Exception:  # noqa: BLE001 - 某些视图无主键信息
            pk = []
    except Exception as exc:  # noqa: BLE001
        raise ConnectorExtractError(
            f"读取表结构失败：{exc}",
            details={"dialect": spec.name, "table": table, "schema": target},
        ) from exc

    return [
        {
            "name": col.get("name"),
            "type": str(col.get("type")),
            "nullable": bool(col.get("nullable", True)),
            "primary_key": col.get("name") in pk,
        }
        for col in columns
    ]


def fetch_scalar(engine: Engine, sql: str, params: dict | None = None) -> Any:
    """执行标量查询（健康检查用）。"""
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def preview_rows(
    engine: Engine,
    spec: DialectSpec,
    *,
    table: str,
    schema: str | None = None,
    columns: list[str] | None = None,
    where: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """预览若干行（不落盘、不建数据集）。"""
    size = max(1, int(limit or getattr(settings, "CONNECTOR_PREVIEW_ROWS", 200) or 200))

    sql = build_select(spec, table=table, schema=schema, columns=columns, where=where, limit=size)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            keys = list(result.keys())
            rows = [dict(row._mapping) for row in result.fetchall()]
    except DataEngineException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ConnectorExtractError(
            f"预览失败：{exc}", details={"sql": sql}
        ) from exc

    return {
        "columns": keys,
        "rows": _jsonable(rows),
        "row_count": len(rows),
        "limit": size,
        "sql": sql,
    }


def _jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把非 JSON 原生类型（Decimal / date / bytes）转成可序列化形式。"""
    import datetime
    import decimal

    out: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
                converted[key] = value.isoformat()
            elif isinstance(value, decimal.Decimal):
                converted[key] = float(value)
            elif isinstance(value, (bytes, bytearray, memoryview)):
                converted[key] = f"<binary {len(bytes(value))} bytes>"
            else:
                converted[key] = value
        out.append(converted)
    return out
