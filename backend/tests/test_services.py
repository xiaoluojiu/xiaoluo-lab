"""Prompt 023-027 测试：Schemas、FileService、DatasetService。"""

from __future__ import annotations

import polars as pl
import pytest
from app.core.exceptions import DatasetException, NotFoundException, ValidationException
from app.schemas.common import ApiError, ApiResponse, PageInfo, Pagination
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetVersionResponse
from app.schemas.file import FileResponse
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService


# ---------- Prompt 023-025：Schemas ----------
def test_api_response_wrappers():
    ok = ApiResponse[int](data=42)
    assert ok.success is True and ok.data == 42
    bad = ApiResponse[int](success=False, error=ApiError(code="X", message="m"))
    assert bad.error.code == "X" and bad.data is None


def test_pagination_and_page_info():
    page = PageInfo.build(page=2, page_size=10, total=35)
    assert page.total_pages == 4
    p = Pagination[str](items=["a", "b"], page_info=page)
    assert p.page_info.total == 35 and len(p.items) == 2


def test_file_and_dataset_schemas():
    f = FileResponse(
        id=1, name="n", original_name="o", path="raw/n", size=1,
        format="csv", checksum="c", created_at="2026-01-01T00:00:00Z",
    )
    assert f.format == "csv"
    dc = DatasetCreate(name="d1")
    assert dc.source_file_id is None
    v = DatasetVersionResponse(
        id=1, dataset_id=1, version=1, parent_version_id=None,
        storage_path="p", format="parquet", row_count=0, column_count=0, schema={},
    )
    assert v.schema == {}
    dr = DatasetResponse(
        id=1, name="n", description="", source_file_id=None,
        created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
    )
    assert dr.latest_version is None


# ---------- Prompt 026：FileService ----------
def test_file_service_upload_flow(db, storage):
    svc = FileService(db, storage)
    f = svc.upload("我的 数据.csv", b"a,b\n1,2\n")
    assert f.path.startswith("raw/")
    assert f.format == "csv"
    assert f.size == 8
    assert storage.exists(f.path)


def test_file_service_rejects_bad_input(db, storage):
    svc = FileService(db, storage)
    with pytest.raises(ValidationException):
        svc.upload("", b"x")
    with pytest.raises(ValidationException):
        svc.upload("a.csv", b"")
    with pytest.raises(ValidationException):
        svc.upload("a.csv", b"x" * (100 * 1024 * 1024 + 1))


def test_file_service_dedupes_same_content(db, storage):
    svc = FileService(db, storage)
    f1 = svc.upload("one.csv", b"same")
    f2 = svc.upload("two.csv", b"same")
    assert f1.id == f2.id


def test_file_service_never_overwrites_same_name(db, storage):
    svc = FileService(db, storage)
    f1 = svc.upload("data.csv", b"v1")
    f2 = svc.upload("data.csv", b"different content v2")
    assert f1.path != f2.path
    assert storage.read(f1.path) == b"v1"
    assert storage.read(f2.path) == b"different content v2"


# ---------- Prompt 027：DatasetService ----------
def test_dataset_service_create_get_list(db, storage):
    svc = DatasetService(db, storage)
    ds = svc.create("sales", description="d")
    assert svc.get(ds.id).name == "sales"
    items, total = svc.list()
    assert total == 1 and items[0].id == ds.id
    with pytest.raises(NotFoundException):
        svc.get(9999)


def test_dataset_service_versions(db, storage):
    svc = DatasetService(db, storage)
    ds = svc.create("d")
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    v1 = svc.create_version(ds.id, df1)
    v2 = svc.create_version(ds.id, df2)

    assert v1.version == 1 and v2.version == 2
    assert v2.parent_version_id == v1.id
    assert v1.row_count == 3 and v1.column_count == 2
    assert v1.schema_json == {"a": "Int64", "b": "String"}

    versions, total = svc.get_versions(ds.id)
    assert total == 2 and versions[0].version == 2

    # 快照可读回
    back = svc.load_version(ds.id, 1)
    assert back.equals(df1)
    assert svc.load_version(ds.id).equals(df2)

    # 原始数据不可覆盖：v1 的存储内容不受 v2 影响
    assert svc.storage.read(v1.storage_path) != svc.storage.read(v2.storage_path)


def test_dataset_service_rejects_snapshot_conflict(db, storage, monkeypatch):
    svc = DatasetService(db, storage)
    ds = svc.create("d")
    # 模拟目标 key 已存在（版本快照冲突）
    monkeypatch.setattr(svc.storage._backend, "exists", lambda key: True, raising=True)
    with pytest.raises(DatasetException):
        svc.create_version(ds.id, pl.DataFrame({"a": [1]}))


def test_dataset_service_load_missing_version(db, storage):
    svc = DatasetService(db, storage)
    ds = svc.create("d")
    with pytest.raises(NotFoundException):
        svc.load_version(ds.id)
