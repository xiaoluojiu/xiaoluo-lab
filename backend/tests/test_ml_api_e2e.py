"""T0-9：ML API 黑盒 E2E 测试。

通过真实 TestClient 请求验证 /api/v1/ml/train 全链路：
- classification（mixed numeric/string，无手动 preprocessing）
- excluded_columns 真正影响最终 features
- regression（linear_regression）
- clustering（kmeans）
- 非法模型参数 → 明确失败，不允许静默成功
- 极小数据集 → 明确失败
"""

from __future__ import annotations

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


def _create_version(service: DatasetService, name: str, df: pl.DataFrame) -> tuple[int, int]:
    """创建数据集 + 版本，返回 (dataset_id, version_id)。"""
    ds = service.create(name)
    version = service.create_version(ds.id, df)
    return ds.id, version.id


# ----------------------------------------------------------------------
# T0-9.1：classification + mixed numeric/string，无手动 preprocessing
# ----------------------------------------------------------------------
def test_train_classification_mixed_types_no_preprocessing(api, service):
    """普通 mixed-type CSV → 直接选 target + 模型 → 训练成功。

    覆盖 T0-2：默认预处理必须自动可用（类别列 one-hot，数值列 scaling）。
    """
    ds_id, _ = _create_version(
        service,
        "cls-mixed",
        pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "age": [25, 32, 45, 22, 38, 50, 28, 33, 41, 36],
                "city": ["BJ", "SH", "BJ", "GZ", "SH", "BJ", "GZ", "SH", "BJ", "GZ"],
                "income": [5.0, 8.5, 12.0, 4.5, 9.0, 15.0, 6.0, 10.0, 13.0, 7.5],
                "label": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "parameters": {"max_iter": 2000},
            "excluded_columns": ["id"],
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True, f"训练应成功，实际响应: {body}"
    run = body["data"]["run"]
    assert run["status"] == "success", f"run 状态应为 success，实际: {run['status']}"
    assert {"accuracy", "precision", "recall", "f1"} <= set(run["metrics"])
    # T0-1 验收：excluded_columns 不在模型 parameters 中
    exp = body["data"]["experiment"]
    assert "excluded_columns" not in exp["parameters"]
    # artifacts 必须记录实际特征不含被排除的列
    assert "id" not in run["artifacts"]["features"]
    # T0-7：必须保存 pipeline.pkl
    assert run["artifacts"].get("pipeline_key")


# ----------------------------------------------------------------------
# T0-9.2：不传 excluded_columns → 成功
# ----------------------------------------------------------------------
def test_train_classification_without_excluded_columns(api, service):
    ds_id, _ = _create_version(
        service,
        "cls-simple",
        pl.DataFrame(
            {
                "a": [1.0, 1.5, 2.0, 8.0, 8.5, 9.0, 1.2, 8.2],
                "b": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5, 1.6, 8.6],
                "label": [0, 0, 0, 1, 1, 1, 0, 1],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["run"]["status"] == "success"


# ----------------------------------------------------------------------
# T0-9.3：excluded_columns 真正影响最终 features
# ----------------------------------------------------------------------
def test_excluded_columns_truly_affect_features(api, service):
    ds_id, _ = _create_version(
        service,
        "cls-excluded",
        pl.DataFrame(
            {
                "id": list(range(1, 11)),
                "age": [25, 32, 45, 22, 38, 50, 28, 33, 41, 36],
                "income": [5.0, 8.5, 12.0, 4.5, 9.0, 15.0, 6.0, 10.0, 13.0, 7.5],
                "label": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            }
        ),
    )

    # 第一次：不排除 id
    resp1 = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "seed": 42,
        },
    )
    features_without_exclude = resp1.json()["data"]["run"]["artifacts"]["features"]
    assert "id" in features_without_exclude

    # 第二次：排除 id
    resp2 = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "excluded_columns": ["id"],
            "seed": 42,
        },
    )
    body2 = resp2.json()
    assert body2["success"] is True
    features_with_exclude = body2["data"]["run"]["artifacts"]["features"]
    assert "id" not in features_with_exclude
    # 排除后 features 集合应该比排除前少一列
    assert len(features_with_exclude) == len(features_without_exclude) - 1
    # 实验的 parameters 不含 excluded_columns
    assert "excluded_columns" not in body2["data"]["experiment"]["parameters"]


# ----------------------------------------------------------------------
# T0-9.4：regression（linear_regression）
# ----------------------------------------------------------------------
def test_train_regression_success(api, service):
    ds_id, _ = _create_version(
        service,
        "reg-data",
        pl.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                "y": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "regression",
            "model": "linear_regression",
            "target_column": "y",
            "seed": 0,
        },
    )
    body = resp.json()
    assert body["success"] is True
    run = body["data"]["run"]
    assert run["status"] == "success"
    assert {"mae", "mse", "rmse", "r2"} <= set(run["metrics"])


# ----------------------------------------------------------------------
# T0-9.5：clustering（kmeans）
# ----------------------------------------------------------------------
def test_train_clustering_success(api, service):
    ds_id, _ = _create_version(
        service,
        "clu-data",
        pl.DataFrame(
            {
                "a": [1.0, 1.2, 1.1, 9.0, 9.2, 9.1, 1.3, 9.3],
                "b": [1.0, 1.1, 0.9, 9.0, 9.1, 8.9, 1.2, 9.2],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "clustering",
            "model": "kmeans",
            "parameters": {"n_clusters": 2},
            "seed": 0,
        },
    )
    body = resp.json()
    assert body["success"] is True
    run = body["data"]["run"]
    assert run["status"] == "success"
    assert run["metrics"]["cluster_count"] == 2


# ----------------------------------------------------------------------
# T0-9.6：非法模型参数 → 明确失败
# ----------------------------------------------------------------------
def test_train_invalid_model_params_returns_failure(api, service):
    """kmeans n_clusters=999 远超样本数 → run.status == 'failed'，不允许静默成功。"""
    ds_id, _ = _create_version(
        service,
        "clu-bad",
        pl.DataFrame(
            {
                "a": [1.0, 1.2, 1.1, 9.0, 9.2, 9.1],
                "b": [1.0, 1.1, 0.9, 9.0, 9.1, 8.9],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "clustering",
            "model": "kmeans",
            "parameters": {"n_clusters": 999},
            "seed": 0,
        },
    )
    # API 仍返回 200，但 run.status 必须是 failed
    body = resp.json()
    assert body["success"] is True  # API 层成功（创建了实验+运行）
    run = body["data"]["run"]
    assert run["status"] == "failed", "非法参数必须导致 run 失败，不允许静默成功"
    assert run["error"]  # 必须有错误信息


# ----------------------------------------------------------------------
# T0-9.7：极小数据集（无法分层切分）→ 明确失败
# ----------------------------------------------------------------------
def test_train_too_small_dataset_fails_clearly(api, service):
    """只有 1 行数据 → preflight 必须拦截，错误信息明确。"""
    ds_id, _ = _create_version(
        service,
        "tiny",
        pl.DataFrame({"a": [1.0], "label": [0]}),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "seed": 0,
        },
    )
    body = resp.json()
    run = body["data"]["run"]
    assert run["status"] == "failed"
    assert run["error"]


# ----------------------------------------------------------------------
# T0-9.8：target 在 excluded_columns 中 → 明确失败
# ----------------------------------------------------------------------
def test_target_in_excluded_columns_fails(api, service):
    ds_id, _ = _create_version(
        service,
        "cls-target-excluded",
        pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "label": [0, 1, 0, 1],
            }
        ),
    )

    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds_id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "excluded_columns": ["label"],  # 错误：排除 target
            "seed": 0,
        },
    )
    body = resp.json()
    run = body["data"]["run"]
    assert run["status"] == "failed"
    assert "target" in run["error"].lower() or "excluded" in run["error"].lower()


# ----------------------------------------------------------------------
# T0-9.9：models 列表不返回 dimensionality（PCA 隐藏）
# ----------------------------------------------------------------------
def test_models_list_hides_dimensionality(api):
    resp = api.get("/api/v1/ml/models")
    body = resp.json()
    assert body["success"] is True
    tasks = {m.get("task") for m in body["data"]}
    assert "dimensionality" not in tasks
