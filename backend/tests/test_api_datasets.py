"""Prompt 062-063：Dataset API 与 Dataset Analysis API 测试。"""

import io

import polars as pl
import pytest
from app.api.deps import get_storage_service
from app.core.database import get_db
from app.main import app
from app.services.dataset_service import DatasetService
from fastapi.testclient import TestClient


@pytest.fixture()
def api(db, storage) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_storage_service] = lambda: storage
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture()
def service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


def df_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


@pytest.fixture()
def dataset_with_version(service) -> int:
    ds = service.create("demo", "测试数据集")
    df = pl.DataFrame(
        {"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"], "score": [10, 20, 30, 40, 50]}
    )
    service.create_version(ds.id, df)
    return ds.id


class TestDatasetAPI:
    def test_create(self, api):
        resp = api.post("/api/v1/datasets", json={"name": "test", "description": "d"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["name"] == "test"

    def test_create_validation(self, api):
        resp = api.post("/api/v1/datasets", json={"name": ""})
        assert resp.status_code == 422

    def test_list_pagination(self, api, service):
        for i in range(3):
            service.create(f"ds-{i}")
        resp = api.get("/api/v1/datasets", params={"page": 1, "page_size": 2})
        body = resp.json()
        assert len(body["data"]["items"]) == 2
        assert body["data"]["page_info"]["total"] == 3

    def test_get(self, api, dataset_with_version):
        resp = api.get(f"/api/v1/datasets/{dataset_with_version}")
        body = resp.json()
        assert body["data"]["id"] == dataset_with_version
        assert body["data"]["latest_version"]["version"] == 1

    def test_get_not_found(self, api):
        resp = api.get("/api/v1/datasets/9999")
        assert resp.status_code == 404
        assert resp.json()["code"] == "NOT_FOUND"

    def test_patch(self, api, dataset_with_version):
        resp = api.patch(
            f"/api/v1/datasets/{dataset_with_version}", json={"description": "new"}
        )
        assert resp.json()["data"]["description"] == "new"

    def test_delete(self, api, service, dataset_with_version):
        resp = api.delete(f"/api/v1/datasets/{dataset_with_version}")
        assert resp.json()["data"]["deleted"] is True
        # 存储快照一并清理
        assert list(service.storage.list(f"datasets/{dataset_with_version}/")) == []
        resp = api.get(f"/api/v1/datasets/{dataset_with_version}")
        assert resp.status_code == 404


class TestAnalysisAPI:
    def test_preview(self, api, dataset_with_version):
        resp = api.get(
            f"/api/v1/datasets/{dataset_with_version}/preview",
            params={"page": 1, "page_size": 2, "sort_column": "id", "sort_desc": True},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total"] == 5
        assert body["data"]["items"][0]["id"] == 5
        assert body["data"]["version"] == 1

    def test_preview_with_filter(self, api, dataset_with_version):
        resp = api.get(
            f"/api/v1/datasets/{dataset_with_version}/preview",
            params={"filter": '[{"column":"score","op":"gte","value":30}]'},
        )
        assert resp.json()["data"]["total"] == 3

    def test_preview_invalid_filter(self, api, dataset_with_version):
        resp = api.get(
            f"/api/v1/datasets/{dataset_with_version}/preview", params={"filter": "{bad"}
        )
        assert resp.status_code == 422

    def test_schema(self, api, dataset_with_version):
        body = api.get(f"/api/v1/datasets/{dataset_with_version}/schema").json()
        columns = {c["column"]: c for c in body["data"]["columns"]}
        assert columns["score"]["dtype"] == "Int64"

    def test_profile(self, api, dataset_with_version):
        body = api.get(f"/api/v1/datasets/{dataset_with_version}/profile").json()
        assert body["data"]["row_count"] == 5
        assert body["data"]["missing_cells"] == 0

    def test_quality(self, api, dataset_with_version):
        body = api.get(f"/api/v1/datasets/{dataset_with_version}/quality").json()
        assert body["data"]["statistics"]["issue_count"] == len(body["data"]["issues"])

    def test_specific_version(self, api, service, dataset_with_version):
        df = pl.DataFrame({"id": [9], "name": ["z"], "score": [99]})
        service.create_version(dataset_with_version, df)
        resp = api.get(
            f"/api/v1/datasets/{dataset_with_version}/preview", params={"version": 1}
        )
        assert resp.json()["data"]["total"] == 5
        resp = api.get(
            f"/api/v1/datasets/{dataset_with_version}/preview", params={"version": 2}
        )
        assert resp.json()["data"]["total"] == 1

    def test_dataset_not_found(self, api):
        resp = api.get("/api/v1/datasets/9999/schema")
        assert resp.status_code == 404

    def test_no_versions(self, api, service):
        ds = service.create("empty")
        resp = api.get(f"/api/v1/datasets/{ds.id}/preview")
        assert resp.status_code == 404
