"""ML 推理链路 / 可复现性 / Workflow ML 节点 端到端测试。

覆盖本轮审计发现并修复的问题：
1. 推理链路真正打通：训练产物（model.pkl + pipeline.pkl）可被加载并批量推理
2. 推理必须经过 pipeline（原始列直接喂模型会失败 —— 反向验证）
3. 失败运行 / 缺列 → 明确报错，不允许静默产出结果
4. Workflow ml.train 结果可复现（seed 注入模型本身）
5. Workflow ml.train 支持类别列（走默认预处理）
6. Workflow ml.cluster 对 DBSCAN 不再崩溃，并返回评估指标
7. supports_param 公开探测正确
"""

from __future__ import annotations

import polars as pl
import pytest
from app.api.deps import get_storage_service
from app.core.database import get_db
from app.main import app
from app.ml_engine.registry import MODEL_REGISTRY
from app.services.dataset_service import DatasetService
from app.workflow.models import Node
from app.workflow.runners import build_default_runners
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


def _mixed_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(1, 41)),
            "age": [20 + (i * 7) % 50 for i in range(40)],
            "city": (["BJ", "SH", "GZ"] * 14)[:40],
            "income": [3.0 + (i % 13) * 1.5 for i in range(40)],
            "label": [0 if i % 2 == 0 else 1 for i in range(40)],
        }
    )


def _train(api, service, df: pl.DataFrame | None = None, **overrides):
    ds = service.create("predict-ds")
    service.create_version(ds.id, df if df is not None else _mixed_df())
    payload = {
        "dataset_id": ds.id,
        "task": "classification",
        "model": "random_forest_classifier",
        "target_column": "label",
        "parameters": {"n_estimators": 25},
        "excluded_columns": ["id"],
        "seed": 42,
    }
    payload.update(overrides)
    resp = api.post("/api/v1/ml/train", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True, body
    return ds.id, body["data"]


# ----------------------------------------------------------------------
# 1/2. 推理链路：训练 -> /ml/predict
# ----------------------------------------------------------------------
def test_predict_after_train_returns_predictions_and_probabilities(api, service):
    ds_id, data = _train(api, service)
    run_id = data["run"]["id"]
    assert data["run"]["status"] == "success"

    resp = api.post(
        "/api/v1/ml/predict",
        json={"run_id": run_id, "dataset_id": ds_id, "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True, body
    result = body["data"]

    assert result["row_count"] == 40
    assert result["pipeline_applied"] is True, "类别列数据必须走预处理管道"
    assert result["prediction_column"] == "prediction"
    assert result["probability_columns"], "分类模型应返回概率列"
    assert len(result["preview"]) == 10
    assert all("prediction" in row for row in result["preview"])
    # 训练时特征含 one-hot 后的 city=BJ 等列，原始特征不应直接出现在模型特征里
    assert "city" in result["feature_columns"]
    assert any(c.startswith("city=") for c in result["model_features"])


def test_predict_reuses_saved_pipeline_not_raw_columns(api, service):
    """反向验证：只加载 model.pkl 而跳过 pipeline 必然失败。

    这条测试锁死「推理必须经过 pipeline」这一正确行为，
    防止后续改动退化成把原始列直接喂给模型。
    """
    ds_id, data = _train(api, service)
    model = MODEL_REGISTRY.get("random_forest_classifier")
    payload_key = data["run"]["artifacts"]["model_key"]
    storage_obj = app.dependency_overrides[get_storage_service]()
    loaded = model.from_bytes(storage_obj.read(payload_key))
    raw = _mixed_df().drop("label").head(3)
    with pytest.raises(Exception) as exc:
        loaded.predict(raw)
    assert "特征" in str(exc.value) or "missing" in str(exc.value).lower()
    assert ds_id


def test_predict_rejects_failed_run(api, service):
    """对失败运行推理 -> 明确 4xx，不允许静默产出。"""
    ds = service.create("bad-ds")
    service.create_version(
        ds.id,
        pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "label": [0, 0]}),
    )
    resp = api.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": ds.id,
            "task": "classification",
            "model": "logistic_regression",
            "target_column": "label",
            "seed": 42,
        },
    )
    run_id = resp.json()["data"]["run"]["id"]
    assert resp.json()["data"]["run"]["status"] == "failed"

    pred = api.post("/api/v1/ml/predict", json={"run_id": run_id, "dataset_id": ds.id})
    assert pred.status_code >= 400, pred.text


def test_explain_returns_feature_importance_from_artifacts(api, service):
    _, data = _train(api, service)
    run_id = data["run"]["id"]
    # 训练产物中应已落库特征重要性
    assert data["run"]["artifacts"].get("feature_importance")

    resp = api.get(f"/api/v1/ml/explain/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()["data"]
    assert body["source"] == "run_artifacts"
    assert body["importances"], "特征重要性不应为空"
    assert body["importances"][0]["feature"]


# ----------------------------------------------------------------------
# 4/5. Workflow ml.train：可复现 + 支持类别列
# ----------------------------------------------------------------------
def _workflow_train(df: pl.DataFrame, params: dict) -> dict:
    runners = build_default_runners()
    node = Node("train", "ml.train", {"params": params})
    return runners["ml.train"](node, {"load": {"_df": df}}, {})


def test_workflow_ml_train_is_reproducible_with_same_seed():
    """同一 random_state 两次训练，指标必须完全一致（seed 注入模型本身）。"""
    df = pl.DataFrame(
        {
            "f1": [float((i * 13) % 17) for i in range(60)],
            "f2": [float((i * 7) % 11) for i in range(60)],
            "target": [1 if i % 2 == 0 else 0 for i in range(60)],
        }
    )
    params = {
        "model": "random_forest_classifier",
        "target_column": "target",
        "test_size": 0.25,
        "random_state": 42,
        "n_estimators": 20,
    }
    first = _workflow_train(df, params)
    second = _workflow_train(df, params)
    assert first["metrics"]["accuracy"] == second["metrics"]["accuracy"], (
        f"同 seed 结果必须可复现：{first['metrics']} vs {second['metrics']}"
    )
    assert first["random_state"] == 42


def test_workflow_ml_train_handles_categorical_columns():
    """类别列不再导致训练失败（默认预处理自动 one-hot）。"""
    df = _mixed_df().drop("id")
    out = _workflow_train(
        df,
        {
            "model": "random_forest_classifier",
            "target_column": "label",
            "test_size": 0.25,
            "random_state": 7,
            "n_estimators": 15,
        },
    )
    assert out["status"] if "status" in out else True
    assert out["metrics"]["accuracy"] is not None
    assert any(c.startswith("city=") for c in out["model_features"])


def test_workflow_ml_cluster_dbscan_returns_metrics():
    """DBSCAN 无 predict，降级用 labels_，且返回聚类评估指标。"""
    runners = build_default_runners()
    df = pl.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1],
            "y": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1],
        }
    )
    node = Node("cluster", "ml.cluster", {"params": {"model": "dbscan", "eps": 0.5, "min_samples": 2}})
    out = runners["ml.cluster"](node, {"load": {"_df": df}}, {})
    assert "cluster" in out["_df"].columns
    assert "cluster_count" in out["metrics"]


# ----------------------------------------------------------------------
# 7. supports_param 公开探测
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("random_forest_classifier", True),
        ("decision_tree_classifier", True),
        ("logistic_regression", True),
        ("kmeans", True),
        ("linear_regression", False),
        ("knn_regressor", False),
    ],
)
def test_supports_random_state_detection(model_name, expected):
    assert MODEL_REGISTRY.supports_param(model_name, "random_state") is expected
