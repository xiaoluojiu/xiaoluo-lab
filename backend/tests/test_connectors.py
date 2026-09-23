"""数据库连接器测试。

策略：**SQLite 端到端**。SQLite 走标准库、零外部依赖，因此这套用例在任何环境
都能真跑（不 mock 数据库），能真正验证「试连 → 列表 → 结构 → 预览 → 分块抽取 →
落成 DatasetVersion → 可被 DatasetService 读回」这条完整链路。

PostgreSQL / MySQL 的真实连接无法在 CI 里假造，因此那部分只测**可失败路径**：
驱动缺失时必须给出「装哪个包」的可操作提示，而不是抛裸的 ModuleNotFoundError。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.connectors.crypto import (
    CIPHER_PREFIX,
    ConnectorCryptoUnavailable,
    crypto_available,
    get_cipher,
    reset_cipher,
)
from app.connectors.dialects import build_url, dialect_catalog, get_dialect
from app.connectors.extract import _validate_where, build_select
from app.connectors.service import ConnectorService
from app.core.config import settings
from app.core.exceptions import ValidationException
from app.data_engine.exceptions import DataEngineException
from app.schemas.connector import (
    ConnectorCreate,
    ConnectorImportRequest,
    ConnectorPreviewRequest,
    ConnectorTestRequest,
)
from app.services.dataset_service import DatasetService

ROW_COUNT = 5000


@pytest.fixture(autouse=True)
def _fixed_connector_secret(monkeypatch):
    """固定加密密钥，避免测试去真实 MODEL_ROOT 下生成密钥文件。"""
    monkeypatch.setattr(settings, "CONNECTOR_SECRET_KEY", "unit-test-secret", raising=False)
    reset_cipher()
    yield
    reset_cipher()


@pytest.fixture()
def source_db(tmp_path) -> Path:
    """造一个真实的 SQLite 数据源。"""
    path = tmp_path / "source.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT, score REAL, city TEXT)"
        )
        conn.executemany(
            "INSERT INTO events (id, name, score, city) VALUES (?, ?, ?, ?)",
            [
                (i, f"user_{i}", i * 1.5, "北京" if i % 2 == 0 else "上海")
                for i in range(1, ROW_COUNT + 1)
            ],
        )
        conn.execute("CREATE VIEW v_events AS SELECT id, name FROM events")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture()
def services(db, storage):
    dataset_service = DatasetService(db, storage)
    return ConnectorService(db, dataset_service), dataset_service


def _make_connector(service: ConnectorService, source_db: Path, **overrides):
    payload = {
        "name": "本地测试库",
        "dialect": "sqlite",
        "database": str(source_db),
    }
    payload.update(overrides)
    return service.create(ConnectorCreate(**payload))


# =========================================================
# 方言层
# =========================================================


def test_dialect_catalog_reports_driver_status():
    catalog = {item["name"]: item for item in dialect_catalog()}

    assert "sqlite" in catalog
    # SQLite 走标准库，永远可用
    assert catalog["sqlite"]["driver_installed"] is True
    assert catalog["sqlite"]["requires_host"] is False
    # 每个方言都要能给出「装什么」的提示（无外部依赖的除外）
    for item in catalog.values():
        if item["driver_hint"]:
            assert item["driver_hint"].startswith("pip install")


def test_dialect_whitelist_rejects_unlisted_dialect():
    """默认白名单不含 oracle ⇒ 必须拒绝，并告诉用户怎么放开。"""
    with pytest.raises(DataEngineException) as excinfo:
        get_dialect("oracle")

    details = excinfo.value.details
    assert details["dialect"] == "oracle"
    assert "CONNECTOR_ALLOWED_DIALECTS" in details["how_to_enable"]


def test_unknown_dialect_lists_supported():
    with pytest.raises(DataEngineException) as excinfo:
        get_dialect("not-a-db")
    assert "sqlite" in excinfo.value.details["supported"]


def test_sqlite_url_uses_absolute_posix_path(source_db: Path):
    url = build_url(get_dialect("sqlite"), database=str(source_db))
    assert url.startswith("sqlite+pysqlite:///")
    assert "\\" not in url  # Windows 反斜杠会把 sqlite URL 拆坏


def test_identifier_quoting_differs_per_dialect():
    assert get_dialect("postgresql").quote_identifier("user_id") == '"user_id"'
    assert get_dialect("mysql").quote_identifier("user_id") == "`user_id`"
    assert get_dialect("postgresql").quote_identifier("public.orders") == '"public"."orders"'


@pytest.mark.parametrize(
    "bad",
    [
        "events; DROP TABLE events",
        "events' OR '1'='1",
        "events--",
        "1events",
        "a.b.c",
        "",
    ],
)
def test_identifier_whitelist_blocks_injection(bad: str):
    """标识符只能走白名单——这是本模块唯一的注入面，必须堵死。"""
    with pytest.raises(DataEngineException):
        get_dialect("sqlite").quote_identifier(bad)


def test_where_fragment_rejects_statement_terminator():
    with pytest.raises(DataEngineException):
        _validate_where("1=1; DROP TABLE events")
    with pytest.raises(DataEngineException):
        _validate_where("1=1 -- comment")
    # 正常条件放行
    assert _validate_where("score > 10") == "score > 10"


def test_build_select_quotes_table_and_columns():
    sql = build_select(
        get_dialect("postgresql"),
        table="events",
        columns=["id", "name"],
        where="score > 1",
        limit=10,
    )
    assert 'FROM "public"."events"' in sql
    assert '"id", "name"' in sql
    assert "WHERE (score > 1)" in sql
    assert sql.endswith("LIMIT 10")


# =========================================================
# CRUD + 口令
# =========================================================


def test_password_never_stored_in_plaintext(services, source_db):
    service, _ = services

    if not crypto_available():
        # 未安装 cryptography 时，系统必须**明确拒绝**而不是悄悄明文落库
        with pytest.raises(ConnectorCryptoUnavailable):
            _make_connector(service, source_db, username="u", password="s3cret")
        return

    record = _make_connector(service, source_db, username="u", password="s3cret")

    assert record.password_enc != "s3cret"
    assert record.password_enc.startswith(CIPHER_PREFIX)

    response = ConnectorService.to_response(record)
    dumped = response.model_dump()
    # 出参结构里绝不能出现任何口令字段
    assert "password" not in dumped
    assert "password_enc" not in dumped
    assert dumped["has_password"] is True
    # 但内部仍能解回原文用于建连
    assert get_cipher().decrypt(record.password_enc) == "s3cret"


def test_to_response_masks_password_when_absent(services, source_db):
    service, _ = services
    record = _make_connector(service, source_db)
    response = ConnectorService.to_response(record)
    assert response.has_password is False
    assert not hasattr(response, "password")


def test_duplicate_name_rejected(services, source_db):
    service, _ = services
    _make_connector(service, source_db)
    with pytest.raises(ValidationException):
        _make_connector(service, source_db, name="本地测试库")


def test_driver_missing_message_is_actionable(services):
    """PostgreSQL 驱动未必装：报错必须告诉用户装什么。"""
    service, _ = services
    record = service.create(
        ConnectorCreate(name="pg", dialect="postgresql", host="127.0.0.1", database="x")
    )

    result = service.test_saved(record.id)
    assert result.ok is False

    if not result.driver_installed:
        assert "pip install" in result.message
        assert "驱动" in result.message

    # 无论哪种失败，状态都要落到记录上，供列表页展示
    service.db.refresh(record)
    assert record.last_status == "error"
    assert record.last_error


# =========================================================
# 端到端：试连 → 列表 → 结构 → 预览 → 导入
# =========================================================


def test_sqlite_connection_and_introspection(services, source_db):
    service, _ = services
    record = _make_connector(service, source_db)

    test_result = service.test_saved(record.id)
    assert test_result.ok is True, test_result.message
    assert test_result.driver_installed is True
    assert test_result.server_version

    service.db.refresh(record)
    assert record.last_status == "ok"

    tables = service.list_tables(record.id)
    names = {item["name"] for item in tables["tables"]}
    assert "events" in names
    assert "v_events" in names

    described = service.describe_table(record.id, "events")
    columns = {item["name"]: item for item in described["columns"]}
    assert set(columns) == {"id", "name", "score", "city"}
    assert columns["id"]["primary_key"] is True


def test_preview_returns_rows_and_sql(services, source_db):
    service, _ = services
    record = _make_connector(service, source_db)

    data = service.preview(
        record.id,
        ConnectorPreviewRequest(table="events", limit=10, columns=["id", "city"]),
    )

    assert data["columns"] == ["id", "city"]
    assert data["row_count"] == 10
    assert "LIMIT 10" in data["sql"]
    assert data["rows"][0]["city"] in {"北京", "上海"}


def test_sqlite_import_end_to_end(services, source_db, storage):
    """完整链路：外部表 → 分块抽取 → Parquet 快照 → 数据集版本 → 可读回。"""
    service, dataset_service = services
    record = _make_connector(service, source_db)

    result = service.import_table(
        record.id,
        ConnectorImportRequest(table="events", dataset_name="外部事件表"),
    )

    assert result.row_count == ROW_COUNT
    assert result.column_count == 4
    # SQLite 无服务端游标 ⇒ 走分页路径
    assert result.strategy == "paged"
    assert result.batches >= 1
    assert result.truncated is False
    assert result.elapsed_seconds >= 0

    # 落成的是与其他数据集完全等价的 DatasetVersion
    version_row = dataset_service.get_version_row(result.dataset_id, result.dataset_version)
    assert version_row.row_count == ROW_COUNT
    assert version_row.column_count == 4
    assert storage.exists(version_row.storage_path)

    frame = dataset_service.load_version(result.dataset_id, result.dataset_version)
    assert frame.height == ROW_COUNT
    assert set(frame.columns) == {"id", "name", "score", "city"}
    # 中文与浮点都要原样过来（验证编码与类型没在批量搬运中丢失）
    assert frame.filter(frame["id"] == 1)["city"][0] == "上海"
    assert float(frame.filter(frame["id"] == 2)["score"][0]) == 3.0

    # 连接器记录要记下最近一次导入，便于复盘
    service.db.refresh(record)
    assert record.dataset_id == result.dataset_id
    assert record.last_import_json["strategy"] == "paged"
    assert record.last_import_json["row_count"] == ROW_COUNT


def test_import_with_keyset_uses_keyset_strategy(services, source_db):
    service, _ = services
    record = _make_connector(service, source_db)

    result = service.import_table(
        record.id,
        ConnectorImportRequest(
            table="events",
            keyset_column="id",
            order_by="id",
            batch_size=1000,
        ),
    )

    assert result.strategy == "keyset"
    assert result.row_count == ROW_COUNT
    # 1000 行一批 ⇒ 至少 5 批
    assert result.batches >= 5


def test_import_respects_columns_where_and_max_rows(services, source_db):
    service, _ = services
    record = _make_connector(service, source_db)

    result = service.import_table(
        record.id,
        ConnectorImportRequest(
            table="events",
            columns=["id", "city"],
            where="id <= 500",
            max_rows=100,
            keyset_column="id",
            order_by="id",
            batch_size=64,
        ),
    )

    assert result.column_count == 2
    assert result.row_count == 100
    assert result.truncated is True
    assert any("截断" in item for item in result.warnings)


def test_import_appends_new_version_to_existing_dataset(services, source_db):
    service, dataset_service = services
    record = _make_connector(service, source_db)

    first = service.import_table(
        record.id, ConnectorImportRequest(table="events", where="id <= 10")
    )
    second = service.import_table(
        record.id,
        ConnectorImportRequest(
            table="events",
            where="id <= 20",
            dataset_id=first.dataset_id,
        ),
    )

    assert second.dataset_id == first.dataset_id
    assert second.dataset_version == first.dataset_version + 1
    assert dataset_service.get_version_row(first.dataset_id).row_count == 20


def test_failed_import_leaves_no_version_and_no_staging_file(
    services, source_db, storage
):
    """抽取中途失败时，不能留下半截版本，也不能消耗版本号。"""
    service, dataset_service = services
    record = _make_connector(service, source_db)

    ok = service.import_table(
        record.id, ConnectorImportRequest(table="events", where="id <= 10")
    )
    assert ok.dataset_version == 1

    with pytest.raises(DataEngineException):
        # 表不存在 → 抽取阶段抛错
        service.import_table(record.id, ConnectorImportRequest(table="no_such_table"))

    # 仍然只有 v1，没有产生 v2
    versions, total = dataset_service.get_versions(ok.dataset_id)
    assert total == 1

    # 存储里也不该有残留的暂存文件
    leftovers = [
        item.key
        for item in storage.list(f"datasets/{ok.dataset_id}/")
        if ".writing" in item.key
    ]
    assert leftovers == []


def test_delete_connector_keeps_imported_dataset(services, source_db):
    service, dataset_service = services
    record = _make_connector(service, source_db)

    result = service.import_table(
        record.id, ConnectorImportRequest(table="events", where="id <= 5")
    )

    service.delete(record.id)

    # 数据集是一等对象，不因连接配置被删而消失
    dataset = dataset_service.get(result.dataset_id)
    assert dataset.name
    assert dataset_service.load_version(result.dataset_id).height == 5


def test_test_request_works_before_saving(services, source_db):
    """表单填完先测一下：不需要先保存连接器。"""
    service, _ = services

    result = service.test_request(
        ConnectorTestRequest(dialect="sqlite", database=str(source_db))
    )

    assert result.ok is True
    assert result.latency_ms >= 0


def test_services_expose_connector_tool_risk_mapping():
    """新增工具必须登记风险等级，否则 Agent 链路会在权限裁决处报错。"""
    from app.agent.permission.rules import DEFAULT_TOOL_RISKS, RiskLevel
    from app.tools.builtin import register_builtin_tools
    from app.tools.registry import TOOL_REGISTRY

    # 单跑本文件时 app.tools.builtin 未必已被导入（它只在 app 启动链路里被拉起），
    # 显式注册一次；该函数幂等。
    register_builtin_tools()

    for name in ("connector.list", "connector.tables", "connector.preview", "connector.import"):
        assert name in TOOL_REGISTRY.names()
        assert name in DEFAULT_TOOL_RISKS

    assert DEFAULT_TOOL_RISKS["connector.import"] == RiskLevel.HIGH
    assert DEFAULT_TOOL_RISKS["connector.list"] == RiskLevel.LOW
