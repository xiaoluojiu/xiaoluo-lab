"""ML 模块第一轮优化：推理结果导出 / 模型产物完整性 / 新模型 / 阈值统一。

这四件事的共同点是「跨层」：训练产物 → 存储 → 推理 → 导出，以及
后端元数据 → 目录接口 → 前端判定。任何一层单独看都没错，错的是接缝，
所以这里全部走真实的 create → run → 落盘 → 读回，不用 mock 替代存储。

数据沿用 `test_ml_engine_smoke.py` 的造法（polars 现造 200 行），
不依赖仓库里任何数据集文件。
"""

from __future__ import annotations

import io
import pathlib
import tempfile

import numpy as np
import polars as pl
import pytest

from app.api.deps import get_storage_service
from app.core.database import SessionLocal
from app.experiments.service import ExperimentService
from app.ml_engine.exceptions import MLEngineException
from app.ml_engine.metadata import SIGNAL_RULES, build_catalog
from app.ml_engine.registry import MODEL_REGISTRY
from app.services.dataset_service import DatasetService

ROWS = 200
HEADER_SIZE = len(b"XLB1") + 32  # magic(4) + hmac_sha256(32)


def _classification_frame(rows: int = ROWS) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=rows)
    f2 = rng.normal(size=rows)
    cat = np.array(["a", "b", "c"])[rng.integers(0, 3, rows)]
    logit = 1.4 * f1 - 0.9 * f2 + (cat == "a") * 0.8 + rng.normal(scale=0.5, size=rows)
    return pl.DataFrame(
        {"f1": f1, "f2": f2, "cat": cat, "y": (logit > 0).astype(int)}
    )


@pytest.fixture()
def ml_env(tmp_path: pathlib.Path):
    storage = get_storage_service()
    db = SessionLocal()
    try:
        dataset_service = DatasetService(db, storage)
        dataset = dataset_service.create("ml-round1")
        parquet = tmp_path / "seed.parquet"
        _classification_frame().write_parquet(parquet)
        dataset_service.create_version_from_source(dataset.id, parquet)
        yield db, dataset_service, dataset.id
    finally:
        db.close()


def _train(db, dataset_service, dataset_id: int, **kwargs):
    path = pathlib.Path(tempfile.mkdtemp(prefix="ml-round1-")) / "f.parquet"
    _classification_frame().write_parquet(path)
    version, _ = dataset_service.create_version_from_source(dataset_id, path)
    service = ExperimentService(db, dataset_service)
    exp = service.create(
        dataset_id=dataset_id,
        dataset_version_id=version.id,
        seed=42,
        description="round1",
        **kwargs,
    )
    run = service.run(exp.id)
    assert run.status == "success", f"训练失败：{run.error}"
    return service, run


def _trained_classifier(ml_env):
    """跑一次二分类训练，返回 (service, run, dataset_id)。"""
    db, dataset_service, dataset_id = ml_env
    service, run = _train(
        db,
        dataset_service,
        dataset_id,
        task="classification",
        model="logistic_regression",
        target_column="y",
    )
    return service, run, dataset_id


def test_predict_export_csv_covers_every_row(ml_env):
    """导出必须覆盖每一行 —— 预览只给前 200 行，导出少一行就等于漏一个样本。"""
    service, run, dataset_id = _trained_classifier(ml_env)
    content, ext = service.export_predictions(run.id, dataset_id=dataset_id, format="csv")
    assert ext == "csv"
    frame = pl.read_csv(io.BytesIO(content))
    assert frame.height == ROWS, f"导出只有 {frame.height} 行，应为 {ROWS}"
    assert "prediction" in frame.columns
    # 概率列由类别数决定：二分类 → prob_0 / prob_1
    proba = [c for c in frame.columns if c.startswith("prob_")]
    assert len(proba) == 2, frame.columns
    # 原始列必须保留，否则导出文件没法回溯到具体样本
    for col in ("f1", "f2", "cat", "y"):
        assert col in frame.columns, frame.columns


def test_predict_export_parquet_roundtrip(ml_env):
    service, run, dataset_id = _trained_classifier(ml_env)
    content, ext = service.export_predictions(run.id, dataset_id=dataset_id, format="parquet")
    assert ext == "parquet"
    assert content[:4] == b"PAR1", "不是 Parquet 文件"
    frame = pl.read_parquet(io.BytesIO(content))
    assert frame.height == ROWS
    assert "prediction" in frame.columns


def test_predict_export_rejects_unknown_format(ml_env):
    from app.core.exceptions import ValidationException

    service, run, dataset_id = _trained_classifier(ml_env)
    with pytest.raises(ValidationException):
        service.export_predictions(run.id, dataset_id=dataset_id, format="xlsx")


def test_export_and_preview_agree_on_prediction(ml_env):
    """预览与导出走同一次推理：两条路径各跑一遍模型就可能给出两份对不上的结果。"""
    service, run, dataset_id = _trained_classifier(ml_env)
    content, _ = service.export_predictions(run.id, dataset_id=dataset_id, format="csv")
    exported = pl.read_csv(io.BytesIO(content))["prediction"].to_list()
    preview = service.predict(run.id, dataset_id=dataset_id, limit=20)["preview"]
    assert [row["prediction"] for row in preview] == exported[:20]


def test_model_artifact_is_signed_on_save(ml_env):
    service, run, dataset_id = _trained_classifier(ml_env)
    storage = getattr(service.dataset_service, "storage")
    raw = storage.read(run.artifacts["model_key"])
    assert raw[:4] == b"XLB1", "模型产物应带完整性信封"
    # 带签名的产物必须能被正常读回（否则等于把推理功能弄坏了）
    assert service.predict(run.id, dataset_id=dataset_id)["row_count"] == ROWS


def test_tampered_model_artifact_is_rejected(ml_env):
    """被改动过的产物必须拒绝加载，而不是解出一个「能跑但结果不对」的模型。"""
    service, run, dataset_id = _trained_classifier(ml_env)
    storage = getattr(service.dataset_service, "storage")
    key = run.artifacts["model_key"]
    raw = bytearray(storage.read(key))
    raw[-1] ^= 0xFF
    storage.save(key, bytes(raw))
    with pytest.raises(MLEngineException) as exc:
        service.predict(run.id, dataset_id=dataset_id)
    assert exc.value.code == "MODEL_ARTIFACT_TAMPERED"


def test_unsigned_legacy_artifact_still_loads(ml_env):
    """升级前保存的裸 pickle 没有信封，必须仍能读 —— 否则历史运行集体失效。"""
    service, run, dataset_id = _trained_classifier(ml_env)
    storage = getattr(service.dataset_service, "storage")
    key = run.artifacts["model_key"]
    storage.save(key, storage.read(key)[HEADER_SIZE:])
    assert service.predict(run.id, dataset_id=dataset_id)["row_count"] == ROWS


def test_hist_gradient_boosting_is_registered_and_trainable(ml_env):
    db, dataset_service, dataset_id = ml_env
    service, run = _train(
        db,
        dataset_service,
        dataset_id,
        task="classification",
        model="hist_gradient_boosting_classifier",
        target_column="y",
    )
    assert run.status == "success"
    assert run.metrics["accuracy"] > 0.7, run.metrics
    # 支持概率输出 ⇒ 阈值与校准都对它生效
    assert run.artifacts.get("calibration"), run.artifacts.keys()


def test_hist_gradient_boosting_params_are_real_signature_params():
    """目录里写的每个参数都必须是 sklearn 构造函数的形参。

    写错一个名字的后果是：前端允许填、后端构造时 TypeError，
    而报错指向 sklearn 内部，排查成本极高。
    """
    klass = MODEL_REGISTRY.get("hist_gradient_boosting_classifier")
    catalog = build_catalog(MODEL_REGISTRY.list())
    entry = next(m for m in catalog["models"] if m["name"] == "hist_gradient_boosting_classifier")
    assert [p["name"] for p in entry["params"]], "新模型必须带参数说明"
    for param in entry["params"]:
        assert klass.supports_param(param["name"]), f"{param['name']} 不是真实参数"


def test_predict_export_endpoint_returns_file(ml_env, client):
    """端点层：Content-Disposition 与媒体类型都要能让浏览器直接存成文件。"""
    service, run, dataset_id = _trained_classifier(ml_env)
    resp = client.get(
        f"/api/v1/experiments/runs/{run.id}/predict-export",
        params={"dataset_id": dataset_id, "format": "csv"},
    )
    assert resp.status_code == 200, resp.text[:400]
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith(".csv")
    assert "text/csv" in resp.headers["content-type"]
    assert pl.read_csv(io.BytesIO(resp.content)).height == ROWS

    bad = client.get(
        f"/api/v1/experiments/runs/{run.id}/predict-export",
        params={"dataset_id": dataset_id, "format": "xlsx"},
    )
    assert bad.status_code == 422, "格式应在网关层就被挡下"


def test_signal_rules_are_exposed_and_match_playbook_text():
    """阈值只有一个事实源：catalog 透出的数字 与 detect 说明文字必须一致。"""
    catalog = build_catalog(MODEL_REGISTRY.list())
    assert catalog["signal_rules"] == SIGNAL_RULES
    overfit = next(r for r in catalog["tuning_playbook"] if r["signal"] == "overfit")
    threshold = SIGNAL_RULES["overfit"]["classification_metric_gap_above"]
    assert f"{threshold:g}" in overfit["detect"], overfit["detect"]
