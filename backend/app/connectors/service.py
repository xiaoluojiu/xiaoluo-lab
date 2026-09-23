"""ConnectorService：外部数据库连接器的业务编排。

职责边界
--------
- **只做编排**：把「配置记录」「连接引擎」「分块抽取」「数据集版本」串起来；
  每个环节的具体实现分别在 ``dialects`` / ``extract`` / ``DatasetService``，
  本模块不含 SQL 拼接，也不碰存储布局。
- **口令单向**：写入时加密，读取时只为建连而解密，任何对外结构都不含原文。

导入语义（关键设计）
--------------------
``import_table`` 复用 ``DatasetService.stage_version``：先在存储目标位置预留
版本号与暂存路径，抽取引擎直接把 Parquet 写在那里，退出上下文时原子落位并
登记 DatasetVersion。因此：

- 抽取规模不受进程内存限制（分块 + 增量写 row group）；
- 中途失败不会留下「半截版本」——暂存文件被清理，数据集版本号没被消耗；
- 成功时得到的是一个与其他数据集完全等价的一等对象（可 EDA / 合并 / 建模）。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.connectors.crypto import CredentialCipher, get_cipher, mask_secret
from app.connectors.dialects import (
    DialectSpec,
    build_url as build_dialect_url,
    dialect_catalog,
    get_dialect,
)
from app.connectors.extract import (
    ConnectorExtractError,
    close_engine,
    describe_table as extract_describe_table,
    extract_table_to_parquet,
    fetch_scalar,
    list_schemas,
    list_tables as extract_list_tables,
    open_engine,
    preview_rows as extract_preview_rows,
)
from app.core.config import settings
from app.core.exceptions import NotFoundException, ValidationException
from app.core.logging import get_logger
from app.models.connector import DbConnector
from app.schemas.connector import (
    ConnectorCreate,
    ConnectorImportRequest,
    ConnectorImportResponse,
    ConnectorPreviewRequest,
    ConnectorResponse,
    ConnectorTestRequest,
    ConnectorTestResponse,
    ConnectorUpdate,
)
from app.services.dataset_service import DatasetService

logger = get_logger("app.connectors.service")

# 各方言的「服务器版本」查询语句（试连时回显，帮助用户确认连对了库）。
_VERSION_SQL: dict[str, str] = {
    "sqlite": "SELECT sqlite_version()",
    "postgresql": "SHOW server_version",
    "mysql": "SELECT VERSION()",
    "mssql": "SELECT @@VERSION",
    "oracle": "SELECT banner FROM v$version WHERE ROWNUM = 1",
}


class ConnectorService:
    """外部数据库连接器服务。"""

    def __init__(
        self,
        db: Session,
        dataset_service: DatasetService,
        *,
        cipher: CredentialCipher | None = None,
    ) -> None:
        self.db = db
        self.dataset_service = dataset_service
        self.cipher = cipher or get_cipher()

    # =========================================================
    # 目录 / 配置
    # =========================================================

    @staticmethod
    def catalog() -> list[dict]:
        """可用数据库类型清单（含驱动是否就绪）。"""
        return dialect_catalog()

    # =========================================================
    # CRUD
    # =========================================================

    def create(self, payload: ConnectorCreate) -> DbConnector:
        spec = get_dialect(payload.dialect)

        name = payload.name.strip()
        if self._name_taken(name):
            raise ValidationException(
                f"连接器名称已存在：{name}",
                code="CONNECTOR_NAME_CONFLICT",
                details={"name": name},
            )

        self._assert_capacity()

        record = DbConnector(
            name=name,
            dialect=spec.name,
            host=(payload.host or "").strip() or None,
            port=payload.port or spec.default_port,
            database=(payload.database or "").strip(),
            schema_name=(payload.schema_name or "").strip() or spec.default_schema,
            username=(payload.username or "").strip() or None,
            password_enc=self.cipher.encrypt(payload.password),
            options_json=dict(payload.options or {}),
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        if payload.test_on_create:
            self.test_saved(record.id)

        return record

    def get(self, connector_id: int) -> DbConnector:
        record = self.db.get(DbConnector, connector_id)
        if record is None:
            raise NotFoundException(
                "连接器不存在",
                details={"connector_id": connector_id},
            )
        return record

    def list(self, page: int = 1, page_size: int = 20) -> tuple[list[DbConnector], int]:
        page = max(1, int(page))
        page_size = min(max(1, int(page_size)), 200)

        stmt = (
            select(DbConnector)
            .order_by(DbConnector.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        items = list(self.db.scalars(stmt))
        total = self.db.scalar(select(func.count(DbConnector.id))) or 0
        return items, total

    def update(self, connector_id: int, payload: ConnectorUpdate) -> DbConnector:
        record = self.get(connector_id)

        if payload.name is not None:
            name = payload.name.strip()
            if name and name != record.name and self._name_taken(name):
                raise ValidationException(
                    f"连接器名称已存在：{name}",
                    code="CONNECTOR_NAME_CONFLICT",
                    details={"name": name},
                )
            record.name = name

        if payload.host is not None:
            record.host = payload.host.strip() or None
        if payload.port is not None:
            record.port = payload.port
        if payload.database is not None:
            record.database = payload.database.strip()
        if payload.schema_name is not None:
            record.schema_name = payload.schema_name.strip() or None
        if payload.username is not None:
            record.username = payload.username.strip() or None
        if payload.options is not None:
            record.options_json = dict(payload.options)
        if payload.password is not None:
            # None = 不改；"" = 清空；其余 = 覆盖
            record.password_enc = self.cipher.encrypt(payload.password)

        self.db.commit()
        self.db.refresh(record)
        return record

    def delete(self, connector_id: int) -> None:
        record = self.get(connector_id)
        # 只删配置，不动已经导入的数据集——数据已经独立成为一等对象，
        # 删除连接配置不应该悄悄带走别人的分析成果。
        self.db.delete(record)
        self.db.commit()

    # =========================================================
    # 连接
    # =========================================================

    def resolve_spec(self, record: DbConnector) -> DialectSpec:
        return get_dialect(record.dialect)

    def build_url(self, record: DbConnector) -> str:
        spec = self.resolve_spec(record)
        if not spec.driver_installed:
            raise ConnectorExtractError(
                f"{spec.label} 驱动未安装，无法连接。安装方式：{spec.missing_driver_hint()}",
                details={"dialect": spec.name, "driver_hint": spec.missing_driver_hint()},
            )

        password = self.cipher.decrypt(record.password_enc)

        return build_dialect_url(
            spec,
            host=record.host,
            port=record.port,
            database=record.database,
            username=record.username,
            password=password,
            options=record.options_json or {},
        )

    def test_request(self, payload: ConnectorTestRequest) -> ConnectorTestResponse:
        """试连：支持「未保存的配置」，也支持「已保存 + 局部覆盖」。"""
        spec = get_dialect(payload.dialect)

        if payload.use_saved and payload.connector_id:
            record = self.get(payload.connector_id)
            password = (
                payload.password
                if payload.password is not None
                else self.cipher.decrypt(record.password_enc)
            )
            url = build_dialect_url(
                spec,
                host=payload.host or record.host,
                port=payload.port or record.port,
                database=payload.database or record.database,
                username=payload.username or record.username,
                password=password,
                options={**(record.options_json or {}), **(payload.options or {})},
            )
        else:
            url = build_dialect_url(
                spec,
                host=payload.host,
                port=payload.port,
                database=payload.database,
                username=payload.username,
                password=payload.password,
                options=payload.options or {},
            )

        return self._probe(spec, url)

    def test_saved(self, connector_id: int) -> ConnectorTestResponse:
        """试连已保存的连接器，并把结论写回记录（供列表页展示）。"""
        record = self.get(connector_id)
        spec = self.resolve_spec(record)

        try:
            response = self._probe(spec, self.build_url(record))
        except Exception as exc:  # noqa: BLE001 - 建 URL 阶段的失败也要落状态
            response = ConnectorTestResponse(
                ok=False,
                message=str(exc),
                dialect=spec.name,
                driver_installed=spec.driver_installed,
            )

        self._record_status(record, response)
        return response

    def _probe(self, spec: DialectSpec, url: str) -> ConnectorTestResponse:
        if not spec.driver_installed:
            return ConnectorTestResponse(
                ok=False,
                message=f"{spec.label} 驱动未安装。安装方式：{spec.missing_driver_hint()}",
                dialect=spec.name,
                driver_installed=False,
            )

        engine = None
        started = time.perf_counter()

        try:
            engine = open_engine(url, spec, pooled=True)
            version_sql = _VERSION_SQL.get(spec.name)
            version = str(fetch_scalar(engine, version_sql)) if version_sql else None
            latency = round((time.perf_counter() - started) * 1000, 2)

            return ConnectorTestResponse(
                ok=True,
                latency_ms=latency,
                server_version=version,
                message=f"连接成功（{latency} ms）",
                dialect=spec.name,
                driver_installed=True,
            )
        except Exception as exc:  # noqa: BLE001 - 连接失败是正常业务结果，不抛异常
            return ConnectorTestResponse(
                ok=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                message=self._humanize_error(exc, spec),
                dialect=spec.name,
                driver_installed=spec.driver_installed,
            )
        finally:
            close_engine(engine)

    def _record_status(self, record: DbConnector, response: ConnectorTestResponse) -> None:
        record.last_status = "ok" if response.ok else "error"
        record.last_error = "" if response.ok else (response.message or "连接失败")[:2000]
        record.last_checked_at = datetime.now()
        self.db.commit()
        self.db.refresh(record)

    @staticmethod
    def _humanize_error(exc: Exception, spec: DialectSpec) -> str:
        """把驱动原始异常压成一句可操作的话。

        ``ModuleNotFoundError: No module named 'psycopg2'`` 对用户毫无意义，
        但「缺少驱动，请 pip install psycopg」他立刻能照做。
        """
        text = str(exc)
        lowered = text.lower()

        if "no module named" in lowered or "importerror" in lowered:
            hint = spec.missing_driver_hint()
            return f"缺少 {spec.label} 驱动：{text}。安装方式：{hint}" if hint else text
        if "password authentication failed" in lowered:
            return "账号或口令错误（password authentication failed）"
        if "access denied" in lowered:
            return "账号或口令错误（access denied）"
        if "could not connect" in lowered or "connection refused" in lowered:
            return f"无法连接到 {spec.label} 服务，请检查 host / port 与网络放通：{text}"
        if "timeout" in lowered or "timed out" in lowered:
            return f"连接超时（{getattr(settings, 'CONNECTOR_CONNECT_TIMEOUT_SECONDS', 10)}s），请检查网络与防火墙"
        if "does not exist" in lowered or "unknown database" in lowered:
            return f"库或表不存在：{text}"

        return text

    # =========================================================
    # 内省
    # =========================================================

    def list_tables(self, connector_id: int, schema: str | None = None) -> dict[str, Any]:
        record = self.get(connector_id)
        spec = self.resolve_spec(record)
        engine = None

        try:
            engine = open_engine(self.build_url(record), spec)
            tables = extract_list_tables(engine, spec, schema=schema)

            return {
                "connector_id": record.id,
                "dialect": spec.name,
                "schema": schema or spec.default_schema,
                "schemas": list_schemas(engine, spec),
                "tables": tables,
                "total": len(tables),
            }
        finally:
            close_engine(engine)

    def describe_table(
        self,
        connector_id: int,
        table: str,
        schema: str | None = None,
    ) -> dict[str, Any]:
        record = self.get(connector_id)
        spec = self.resolve_spec(record)
        engine = None

        try:
            engine = open_engine(self.build_url(record), spec)
            columns = extract_describe_table(engine, spec, table=table, schema=schema)
            return {
                "connector_id": record.id,
                "table": table,
                "schema": schema or spec.default_schema,
                "columns": columns,
                "total": len(columns),
            }
        finally:
            close_engine(engine)

    def preview(
        self,
        connector_id: int,
        payload: ConnectorPreviewRequest,
    ) -> dict[str, Any]:
        record = self.get(connector_id)
        spec = self.resolve_spec(record)
        engine = None

        try:
            engine = open_engine(self.build_url(record), spec)
            return extract_preview_rows(
                engine,
                spec,
                table=payload.table,
                schema=payload.schema_name,
                columns=payload.columns,
                where=payload.where,
                limit=payload.limit,
            )
        finally:
            close_engine(engine)

    # =========================================================
    # 导入为数据集
    # =========================================================

    def import_table(
        self,
        connector_id: int,
        payload: ConnectorImportRequest,
    ) -> ConnectorImportResponse:
        record = self.get(connector_id)
        spec = self.resolve_spec(record)

        dataset_id = self._resolve_target_dataset(record, payload)

        engine = None
        try:
            engine = open_engine(self.build_url(record), spec)

            with self.dataset_service.stage_version(dataset_id) as staging:
                result = extract_table_to_parquet(
                    engine,
                    spec,
                    staging.staging_path,
                    table=payload.table,
                    schema=payload.schema_name,
                    columns=payload.columns,
                    where=payload.where,
                    order_by=payload.order_by,
                    batch_size=payload.batch_size,
                    max_rows=payload.max_rows or (getattr(settings, "CONNECTOR_MAX_ROWS", 0) or None),
                    keyset_column=payload.keyset_column,
                )
                staging.set_stats(
                    row_count=result.row_count,
                    column_count=result.column_count,
                    schema_json=result.schema,
                )
            # 注意：stage_version 是在 with 块**退出时**才 promote + 登记的，
            # 因此 version_row 必须在这里读，而不是在 with 块内部。
            dataset_version = staging.version
            version_row = staging.version_row
        finally:
            close_engine(engine)

        # stage_version 已在退出时提交，version_row 必然有值；这里再兜一层防御。
        if version_row is not None:
            dataset_version = version_row.version

        metadata = result.to_metadata()
        metadata["connector_id"] = record.id
        metadata["connector_name"] = record.name

        record.dataset_id = dataset_id
        record.last_import_json = metadata
        self.db.commit()
        self.db.refresh(record)

        logger.info(
            "连接器导入完成 | connector=%s table=%s rows=%s strategy=%s elapsed=%.2fs",
            record.name,
            payload.table,
            result.row_count,
            result.strategy,
            result.elapsed_seconds,
        )

        return ConnectorImportResponse(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            row_count=result.row_count,
            column_count=result.column_count,
            strategy=result.strategy,
            batches=result.batches,
            elapsed_seconds=result.elapsed_seconds,
            rows_per_second=result.rows_per_second,
            truncated=result.truncated,
            warnings=result.warnings,
        )

    # =========================================================
    # 出参
    # =========================================================

    @staticmethod
    def to_response(record: DbConnector) -> ConnectorResponse:
        return ConnectorResponse(
            id=record.id,
            name=record.name,
            dialect=record.dialect,
            host=record.host,
            port=record.port,
            database=record.database,
            schema_name=record.schema_name,
            username=record.username,
            options=record.options_json or {},
            has_password=bool(record.password_enc),
            last_status=record.last_status or "unknown",
            last_error=record.last_error or "",
            last_checked_at=record.last_checked_at,
            dataset_id=record.dataset_id,
            last_import=record.last_import_json or {},
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def password_hint(record: DbConnector) -> str:
        """给日志/事件流用的口令占位（永不返回明文）。"""
        return mask_secret(record.password_enc)

    # =========================================================
    # 内部
    # =========================================================

    def _name_taken(self, name: str) -> bool:
        stmt = select(DbConnector.id).where(DbConnector.name == name).limit(1)
        return self.db.scalars(stmt).first() is not None

    def _assert_capacity(self) -> None:
        limit = int(getattr(settings, "CONNECTOR_MAX_CONNECTORS", 100) or 0)
        if not limit:
            return
        total = self.db.scalar(select(func.count(DbConnector.id))) or 0
        if total >= limit:
            raise ValidationException(
                f"连接器数量已达上限（{limit}）",
                code="CONNECTOR_LIMIT_REACHED",
                details={"limit": limit, "current": total},
            )

    def _resolve_target_dataset(
        self,
        record: DbConnector,
        payload: ConnectorImportRequest,
    ) -> int:
        """确定导入目标：追加到已有数据集，或新建一个。"""
        if payload.dataset_id is not None:
            self.dataset_service.get(payload.dataset_id)
            return int(payload.dataset_id)

        name = (payload.dataset_name or "").strip()
        if not name:
            source = payload.table.split(".")[-1]
            name = f"{record.name} · {source}"

        dataset = self.dataset_service.create(
            name=name[:255],
            description=payload.dataset_description
            or f"由数据库连接器「{record.name}」（{record.dialect}）导入",
        )
        return dataset.id


__all__ = ["ConnectorService"]
