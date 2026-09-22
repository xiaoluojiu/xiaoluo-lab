"""Prompt 196-205：Files / Processing / Merge / EDA / ML / Experiments /
Workflow / Reports API 测试。"""

import polars as pl
import pytest
from app.api.deps import (
    get_data_engine_service,
    get_dataset_service,
    get_experiment_service,
    get_file_service,
    get_workflow_service,
)
from app.core.database import get_db
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.main import app
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService
from fastapi.testclient import TestClient


@pytest.fixture()
def services(db, storage) -> tuple[DatasetService, DataEngineService, ExperimentService]:
    ds_service = DatasetService(db, storage)
    engine = DataEngineService(ds_service)
    exp_service = ExperimentService(db, ds_service)
    return ds_service, engine, exp_service


@pytest.fixture()
def api(db, storage, services) -> TestClient:
    ds_service, engine, exp_service = services
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_file_service] = lambda: FileService(db, storage)
    app.dependency_overrides[get_dataset_service] = lambda: ds_service
    app.dependency_overrides[get_data_engine_service] = lambda: engine
    app.dependency_overrides[get_experiment_service] = lambda: exp_service
    from app.api.deps import WORKFLOW_SERVICE

    app.dependency_overrides[get_workflow_service] = lambda: WORKFLOW_SERVICE
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _mk_dataset(service: DatasetService, name: str, df: pl.DataFrame) -> int:
    ds = service.create(name)
    service.create_version(ds.id, df)
    return ds.id


class TestFilesAPI:
    def test_upload_list_get_delete(self, api):
        resp = api.post(
            "/api/v1/files/upload",
            files={"file": ("hello.csv", b"a,b\n1,2\n", "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["original_name"] == "hello.csv"
        assert body["format"] == "csv"

        listed = api.get("/api/v1/files").json()["data"]
        assert listed["page_info"]["total"] == 1

        got = api.get(f"/api/v1/files/{body['id']}").json()["data"]
        assert got["checksum"] == body["checksum"]

        content = api.get(f"/api/v1/files/{body['id']}/content")
        assert content.content == b"a,b\n1,2\n"

        assert api.delete(f"/api/v1/files/{body['id']}").json()["data"]["deleted"] is True
        assert api.get(f"/api/v1/files/{body['id']}").status_code == 404

    def test_upload_dedup(self, api):
        for _ in range(2):
            api.post(
                "/api/v1/files/upload",
                files={"file": ("same.csv", b"x\n1\n", "text/csv")},
            )
        assert api.get("/api/v1/files").json()["data"]["page_info"]["total"] == 1

    def test_upload_empty_rejected(self, api):
        resp = api.post("/api/v1/files/upload", files={"file": ("e.csv", b"", "text/csv")})
        assert resp.status_code == 422


class TestProcessingAPI:
    def test_operations_catalog(self, api):
        ops = api.get("/api/v1/processing/operations").json()["data"]
        types = {o["op_type"] for o in ops}
        assert {"missing", "filter", "transform", "aggregate"} <= types

    def test_clean_and_aggregate(self, api, services):
        ds_service, _, _ = services
        df = pl.DataFrame({"city": ["bj", "sh", "bj", "sz"], "amount": [10, None, 30, 50]})
        ds = _mk_dataset(ds_service, "ops", df)

        clean = api.post(
            f"/api/v1/processing/datasets/{ds}/clean",
            json={"params": {"strategy": "drop"}},
        ).json()["data"]
        assert clean["version"]["row_count"] == 3
        assert clean["operation"]["status"] == "success"

        agg = api.post(
            f"/api/v1/processing/datasets/{ds}/aggregate",
            json={
                "params": {
                    "group_by": ["city"],
                    "aggregations": [{"column": "amount", "func": "sum", "alias": "total"}],
                }
            },
        ).json()["data"]
        assert agg["version"]["row_count"] == 2  # 清洗后剩余 bj / sz 两城

    def test_invalid_filter_params(self, api, services):
        ds_service, _, _ = services
        ds = _mk_dataset(ds_service, "ops2", pl.DataFrame({"a": [1, 2]}))
        resp = api.post(
            f"/api/v1/processing/datasets/{ds}/filter",
            json={"params": {"conditions": [{"column": "nope", "op": "gt", "value": 0}]}},
        )
        assert resp.status_code == 400


class TestMergeAPI:
    @pytest.fixture()
    def two_datasets(self, services):
        ds_service, _, _ = services
        left = _mk_dataset(
            ds_service, "left", pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
        )
        right = _mk_dataset(
            ds_service, "right", pl.DataFrame({"id": [2, 3, 4], "w": ["x", "y", "z"]})
        )
        return left, right

    def test_mapping_keys_validate_execute(self, api, two_datasets):
        left, right = two_datasets
        mapping = api.post(
            "/api/v1/merge/mapping",
            json={"left_dataset_id": left, "right_dataset_id": right},
        ).json()["data"]
        assert any(m["source_column"] == "id" for m in mapping)

        keys = api.post(
            "/api/v1/merge/keys",
            json={
                "left_dataset_id": left,
                "right_dataset_id": right,
                "left_key": "id",
                "right_key": "id",
            },
        ).json()["data"]
        assert keys["cardinality"] == "one-to-one"

        # T0-M4：preview 返回固定版本号，execute 必须显式传入
        plan = {"keys": [{"left": "id", "right": "id"}], "join_type": "left"}
        preview = api.post(
            "/api/v1/merge/preview",
            json={"left_dataset_id": left, "right_dataset_id": right, "plan": plan},
        ).json()["data"]
        # left={1,2,3} right={2,3,4} -> left join -> 3 行
        assert preview["output_rows"] == 3
        left_version = preview["left_version"]
        right_version = preview["right_version"]
        assert left_version == 1 and right_version == 1

        validation = api.post(
            "/api/v1/merge/validate",
            json={
                "left_dataset_id": left,
                "right_dataset_id": right,
                "left_version": left_version,
                "right_version": right_version,
                "plan": plan,
            },
        ).json()["data"]
        assert validation["ok"] is True

        # T0-M4：execute 必须显式版本号；缺省将被拒绝
        result = api.post(
            "/api/v1/merge/execute",
            json={
                "left_dataset_id": left,
                "right_dataset_id": right,
                "left_version": left_version,
                "right_version": right_version,
                "plan": plan,
            },
        ).json()["data"]
        assert result["report"]["output_rows"] == 3
        assert result["version"]["version"] == 2

        # T0-M4 验证：缺省版本号时 execute 必须返回 400/422
        no_ver = api.post(
            "/api/v1/merge/execute",
            json={"left_dataset_id": left, "right_dataset_id": right, "plan": plan},
        )
        assert no_ver.status_code in (400, 422)

    def test_execute_bad_key_fails_validation(self, api, two_datasets):
        left, right = two_datasets
        resp = api.post(
            "/api/v1/merge/validate",
            json={
                "left_dataset_id": left,
                "right_dataset_id": right,
                "left_version": 1,
                "right_version": 1,
                "plan": {"keys": [{"left": "nope", "right": "id"}]},
            },
        )
        assert resp.json()["data"]["ok"] is False


class TestEdaAPI:
    def test_all_modules(self, api, services):
        ds_service, _, _ = services
        df = pl.DataFrame({"a": [1, 2, 3, 4, 100], "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
        ds = _mk_dataset(ds_service, "eda-ds", df)

        desc = api.get(f"/api/v1/datasets/{ds}/eda/descriptive").json()["data"]
        assert desc["row_count"] == 5

        corr = api.get(f"/api/v1/datasets/{ds}/eda/correlation").json()["data"]
        assert set(corr["matrix"]) == {"a", "b"}

        dist = api.get(
            f"/api/v1/datasets/{ds}/eda/distribution", params={"column": "a"}
        ).json()["data"]
        assert dist["type"] == "numeric"

        out = api.get(f"/api/v1/datasets/{ds}/eda/outlier").json()["data"]
        cols = {r["column"]: r for r in out["columns"]}
        assert cols["a"]["outlier_count"] >= 1

    def test_dataset_not_found(self, api):
        assert api.get("/api/v1/datasets/9999/eda/descriptive").status_code == 404


class TestMLAndExperimentsAPI:
    @pytest.fixture()
    def ml_dataset(self, services):
        ds_service, _, _ = services
        df = pl.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "x2": [0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.8, 0.6],
                "y": [0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        return _mk_dataset(ds_service, "ml-ds", df)

    def test_models_catalog(self, api):
        models = api.get("/api/v1/ml/models").json()["data"]
        names = {m["name"] for m in models}
        assert {"logistic_regression", "linear_regression", "kmeans"} <= names

    def test_train_and_experiment_flow(self, api, ml_dataset):
        body = api.post(
            "/api/v1/ml/train",
            json={
                "dataset_id": ml_dataset,
                "task": "classification",
                "model": "logistic_regression",
                "target_column": "y",
                "seed": 42,
            },
        ).json()["data"]
        assert body["run"]["status"] == "success"
        assert "accuracy" in body["run"]["metrics"]

        exp_id = body["experiment"]["id"]
        runs = api.get(f"/api/v1/experiments/{exp_id}/runs").json()["data"]
        assert len(runs) == 1

        created = api.post(
            "/api/v1/experiments",
            json={
                "dataset_id": ml_dataset,
                "task": "regression",
                "model": "linear_regression",
                "target_column": "x1",
            },
        )
        assert created.status_code == 200
        run = api.post(
            f"/api/v1/experiments/{created.json()['data']['id']}/run"
        ).json()["data"]
        assert run["status"] == "success"

        compare = api.post(
            "/api/v1/experiments/compare",
            json={"run_ids": [body["run"]["id"], run["id"]]},
        ).json()["data"]
        assert "entries" in compare

    def test_train_task_validation(self, api, ml_dataset):
        resp = api.post(
            "/api/v1/ml/train",
            json={"dataset_id": ml_dataset, "task": "bad", "model": "kmeans"},
        )
        assert resp.status_code == 422


class TestWorkflowAPI:
    def test_crud_run_clone_delete(self, api):
        created = api.post(
            "/api/v1/workflows",
            json={
                "name": "demo-flow",
                "nodes": [{"id": "n1", "type": "noop", "config": {}}],
                "edges": [],
            },
        ).json()["data"]
        wid = created["id"]

        assert any(
            w["name"] == "demo-flow" for w in api.get("/api/v1/workflows").json()["data"]
        )

        clone = api.post(f"/api/v1/workflows/{wid}/clone").json()["data"]
        assert clone["name"] == "demo-flow (copy)"

        run = api.post(f"/api/v1/workflows/{wid}/run", json={}).json()["data"]
        # noop runner 已注册：执行成功
        assert run["status"] == "success"
        assert run["result"]["logs"][0]["node_id"] == "n1"

        assert api.get(f"/api/v1/workflows/runs/{run['run_id']}").status_code == 200

        updated = api.put(
            f"/api/v1/workflows/{wid}",
            json={"name": "demo-2", "nodes": [], "edges": []},
        ).json()["data"]
        assert updated["name"] == "demo-2"

        assert api.delete(f"/api/v1/workflows/{wid}").json()["data"]["deleted"] is True

    def test_invalid_dag_rejected(self, api):
        resp = api.post(
            "/api/v1/workflows",
            json={
                "name": "cycle",
                "nodes": [{"id": "a", "type": "noop"}, {"id": "b", "type": "noop"}],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                ],
            },
        )
        assert resp.status_code == 422


class TestReportsAPI:
    def test_generate_and_export(self, api, services):
        ds_service, _, _ = services
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        ds = _mk_dataset(ds_service, "report-ds", df)

        report = api.post(
            "/api/v1/reports/generate",
            json={"dataset_id": ds, "title": "测试报告"},
        ).json()["data"]
        assert report["title"] == "测试报告"
        assert report["sections"], "至少包含概览章节"
        assert report["conclusions"]

        md = api.post(
            "/api/v1/reports/export", json={"report": report, "format": "markdown"}
        )
        assert "text/markdown" in md.headers["content-type"]
        assert "测试报告" in md.text

        html = api.post(
            "/api/v1/reports/export", json={"report": report, "format": "html"}
        )
        assert html.headers["content-type"].startswith("text/html")

    def test_generate_not_found(self, api):
        assert (
            api.post("/api/v1/reports/generate", json={"dataset_id": 9999}).status_code
            == 404
        )
