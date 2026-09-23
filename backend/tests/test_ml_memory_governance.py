"""ML 链路内存治理回归测试。

对应线上事故：在 dataset 9（airlines，10,000,000 行 × 10 列）上跑 Experiment + Run，
报 ``numpy._core._exceptions._ArrayMemoryError: Unable to allocate 57.1 GiB for an
array with shape (10000000, 767)``。

根因不是「数据太大」，而是**四处同源缺陷**：全都默认「这张表装得进稠密内存」。
   ① 预处理 one-hot 不控基数 → 3 个标称列从 30/369/368 类别展开成 767 列；
   ② 训练不做规模治理 → 1000 万行直接进 ColumnTransformer；
   ③ ``arr[:, i].tolist()`` 把矩阵装箱成上亿个 Python 对象；
   ④ 轮廓系数 O(n²) 在全量上算；
   ⑤ 推理路径同样一次性稠密化（推理不能抽样，只能分块）。
本文件把「治理后的不变量」钉住，防止任何一处回退。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from app.core.config import settings
from app.experiments.service import ExperimentService
from app.ml_engine.evaluation import evaluate_clustering
from app.ml_engine.exceptions import MLEngineException
from app.ml_engine.inference import (
    batch_evaluate,
    batch_predict,
    batch_predict_proba,
    batch_transform,
    plan_chunk_rows,
)
from app.ml_engine.preprocessing import (
    _assert_dense_fits,
    _human_bytes,
    build_pipeline,
    cap_training_rows,
    estimate_output_width,
)
from app.ml_engine.registry import MODEL_REGISTRY
from app.services.dataset_service import DatasetService
from app.workflow.models import Node
from app.workflow.runners import build_default_runners
from app.core.exceptions import WorkflowException


@pytest.fixture()
def limits(monkeypatch):
    """临时改写 ML 内存治理配置（settings 是 lru_cache 单例，直接改属性即可）。"""

    def _apply(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setattr(settings, key, value, raising=False)

    return _apply


def _frame(rows: int, *, high_cardinality: bool = True, seed: int = 3) -> pl.DataFrame:
    """构造一张「像是 1000 万行 airlines」的小样本：数值列 + 高低基数标称列。"""
    rng = np.random.default_rng(seed)
    mod = 240 if high_cardinality else 4
    return pl.DataFrame(
        {
            "num_a": rng.normal(0, 1, rows),
            "num_b": rng.integers(0, 100, rows).astype("float64"),
            "cat_hi": [f"c{i % mod}" for i in range(rows)],
            "cat_lo": [f"k{i % 3}" for i in range(rows)],
            "label": rng.integers(0, 2, rows),
        }
    )


# ======================================================================
# 分块行数规划
# ======================================================================
def test_plan_chunk_rows_divides_budget_by_row_width():
    rows = plan_chunk_rows(100, budget_bytes=2 * 1024**3)
    assert rows == (2 * 1024**3) // (100 * 8 * 4)


def test_plan_chunk_rows_returns_zero_when_budget_disabled():
    """预算 <=0 表示「不分块」，调用方据此走原路径。"""
    assert plan_chunk_rows(100, budget_bytes=0) == 0
    assert plan_chunk_rows(100, budget_bytes=-1) == 0


def test_plan_chunk_rows_never_returns_less_than_one():
    assert plan_chunk_rows(10_000, budget_bytes=8) == 1


# ======================================================================
# 训练集规模治理
# ======================================================================
def test_cap_training_rows_samples_and_reports(limits):
    limits(ML_MAX_TRAIN_ROWS=100)
    df = _frame(400)
    X, y, info = cap_training_rows(df.drop("label"), df["label"], seed=42)
    assert X.height == 100
    assert y is not None and y.len() == 100
    assert info["sampled"] is True
    assert info["original_rows"] == 400
    assert info["used_rows"] == 100
    assert info["sample_rate"] == 0.25
    # 抽样不得改变 schema（列名/列序/类型）
    assert X.columns == df.drop("label").columns
    assert X.schema == df.drop("label").schema


def test_cap_training_rows_is_a_noop_below_limit(limits):
    limits(ML_MAX_TRAIN_ROWS=1000)
    df = _frame(400)
    X, y, info = cap_training_rows(df.drop("label"), df["label"], seed=42)
    assert X.equals(df.drop("label"))
    assert info == {"sampled": False, "original_rows": 400, "used_rows": 400}


def test_cap_training_rows_is_reproducible_for_same_seed(limits):
    limits(ML_MAX_TRAIN_ROWS=50)
    df = _frame(500)
    first, _, _ = cap_training_rows(df, seed=7)
    second, _, _ = cap_training_rows(df, seed=7)
    third, _, _ = cap_training_rows(df, seed=8)
    assert first.equals(second)
    assert not first.equals(third)


def test_cap_training_rows_keeps_x_and_y_row_aligned(limits):
    limits(ML_MAX_TRAIN_ROWS=64)
    df = _frame(300)
    X, y, _ = cap_training_rows(df.drop("label"), df["label"], seed=1)
    # 抽样后 X 与 y 必须仍然逐行对应 —— 错位比「样本变少」严重得多。
    # 通过「(数值列, 标称列, 标签) 三元组必须能在原表中找到」来钉住对齐关系。
    originals = set(
        zip(df["num_a"].to_list(), df["cat_hi"].to_list(), df["label"].to_list())
    )
    triples = list(zip(X["num_a"].to_list(), X["cat_hi"].to_list(), y.to_list()))
    assert len(triples) == 64
    assert all(t in originals for t in triples)
    # 同一位置的标签也要能与原表对上（错位会让三元组落在「合法但错位」的组合上）
    aligned = set(zip(df["num_a"].to_list(), df["cat_hi"].to_list()))
    assert all((a, c) in aligned for a, c, _ in triples)


# ======================================================================
# one-hot 基数控制
# ======================================================================
def test_onehot_respects_max_categories(limits):
    """高基数标称列必须被合并低频类别，否则 369 个机场会放大成 369 列。"""
    limits(ML_ONEHOT_MAX_CATEGORIES=25)
    df = _frame(1200, high_cardinality=True)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    out = pipeline.fit_transform(X)
    # cat_hi 有 240 个不同取值，cat_lo 有 3 个；展开后 cat_hi 最多 25 列
    assert sum(c.startswith("cat_hi=") for c in out.columns) <= 25
    assert sum(c.startswith("cat_lo=") for c in out.columns) <= 3


def test_estimate_output_width_matches_fitted_width(limits):
    limits(ML_ONEHOT_MAX_CATEGORIES=10)
    df = _frame(600)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    planned = estimate_output_width(X, pipeline.encoding)
    actual = pipeline.fit_transform(X).width
    assert planned == actual


# ======================================================================
# 规模预检：拒绝要可读，而不是让 numpy 抛 GiB
# ======================================================================
def test_human_bytes_never_renders_zero():
    """提示存在的唯一意义就是让用户知道「差多少」，显示 0.0 GiB 等于没说。"""
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(1024) == "1.0 KiB"
    assert _human_bytes(1024**2) == "1.0 MiB"
    assert _human_bytes(5 * 1024**3) == "5.0 GiB"
    assert _human_bytes(62_000_000_000) == "57.7 GiB"


def test_dense_guard_message_is_actionable(limits):
    limits(ML_MAX_DENSE_BYTES=1 * 1024**2)
    with pytest.raises(MLEngineException) as excinfo:
        _assert_dense_fits(30_000, 56, stage="测试", hint="提示语")
    message = str(excinfo.value)
    assert "30,000 行 × 56 列" in message
    assert "MiB" in message and "0.0 GiB" not in message
    assert "提示语" in message
    assert excinfo.value.details["needed_bytes"] == 30_000 * 56 * 8


def test_dense_guard_is_skipped_when_budget_disabled(limits):
    limits(ML_MAX_DENSE_BYTES=0)
    _assert_dense_fits(10_000_000, 767, stage="测试", hint="")  # 不应抛错


def test_guard_rejects_the_actual_accident_shape(limits):
    """★ 直接钉住事故的那一行报错：10,000,000 × 767 ≈ 57.1 GiB。"""
    limits(ML_MAX_DENSE_BYTES=2 * 1024**3)
    with pytest.raises(MLEngineException) as excinfo:
        _assert_dense_fits(
            10_000_000, 767, stage="预处理输出（含 one-hot 展开）", hint="提示"
        )
    message = str(excinfo.value)
    assert "10,000,000 行 × 767 列" in message
    assert "57.1 GiB" in message
    assert "2.0 GiB" in message


def test_full_data_transform_is_rejected_before_allocating(limits):
    """真实数据版：一次性 transform 必须先被拦下，而不是先分配再崩。"""
    limits(ML_MAX_DENSE_BYTES=4 * 1024**2)
    df = _frame(40_000, high_cardinality=True)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    with pytest.raises(MLEngineException, match="MiB|GiB"):
        pipeline.fit_transform(X)
    # 同一份数据在预算放开后必须能跑通（证明拒绝的理由确实是内存，不是数据本身有问题）
    limits(ML_MAX_DENSE_BYTES=0)
    assert pipeline.fit_transform(X).height == 40_000


def test_model_predict_is_rejected_before_allocating(limits):
    limits(ML_MAX_DENSE_BYTES=0)
    df = _frame(20_000)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    Xp = pipeline.fit_transform(X)
    model = MODEL_REGISTRY.create("linear_regression", {})
    model.fit(Xp, df["label"].cast(pl.Float64))

    limits(ML_MAX_DENSE_BYTES=2 * 1024**2)
    with pytest.raises(MLEngineException, match="MiB|GiB"):
        model.predict(Xp)


# ======================================================================
# 分块推理：与一次性调用逐元素等价
# ======================================================================
def _trained_classifier(rows: int = 6000, seed: int = 5):
    df = _frame(rows, seed=seed)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    Xp = pipeline.fit_transform(X)
    model = MODEL_REGISTRY.create(
        "random_forest_classifier", {"n_estimators": 8, "random_state": 1}
    )
    model.fit(Xp, df["label"])
    return df, X, pipeline, model


def test_batch_predict_matches_one_shot():
    df, X, pipeline, model = _trained_classifier()
    reference = model.predict(pipeline.transform(X))
    chunked = batch_predict(model, X, pipeline=pipeline, chunk_rows=257)
    assert chunked.len() == reference.len()
    assert chunked.equals(reference)


def test_batch_predict_handles_ragged_final_chunk():
    """总行数不是块大小整数倍时，尾块必须完整 —— 少一行就是少一行结果。"""
    df, X, pipeline, model = _trained_classifier(rows=1333)
    reference = model.predict(pipeline.transform(X))
    chunked = batch_predict(model, X, pipeline=pipeline, chunk_rows=200)
    assert chunked.len() == 1333
    assert chunked.equals(reference)


def test_batch_predict_proba_matches_one_shot():
    df, X, pipeline, model = _trained_classifier()
    reference = model.predict_proba(pipeline.transform(X))
    chunked = batch_predict_proba(model, X, pipeline=pipeline, chunk_rows=257)
    assert chunked.shape == reference.shape
    assert chunked.equals(reference)


def test_batch_transform_matches_one_shot():
    df = _frame(3000)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    Xp = pipeline.fit_transform(X)
    pca = MODEL_REGISTRY.create("pca", {"n_components": 3})
    pca.fit(Xp)
    reference = pca.transform(Xp)
    chunked = batch_transform(pca, X, pipeline=pipeline, chunk_rows=311)
    assert chunked.shape == reference.shape
    assert chunked.equals(reference)


def test_batch_evaluate_matches_one_shot():
    df, X, pipeline, model = _trained_classifier()
    reference = model.evaluate(pipeline.transform(X), df["label"])
    chunked = batch_evaluate(model, X, df["label"], pipeline=pipeline, chunk_rows=257)
    assert chunked == reference


def test_batch_predict_rescues_an_oversized_job(limits):
    """★ 事故的正面修复：同一份超预算数据，一次性调用被拒绝，分块调用跑得通且等价。"""
    df = _frame(12_000)
    X = df.drop("label")
    pipeline = build_pipeline(None, X)
    Xp = pipeline.fit_transform(X)
    model = MODEL_REGISTRY.create(
        "random_forest_classifier", {"n_estimators": 8, "random_state": 1}
    )
    model.fit(Xp, df["label"])

    limits(ML_MAX_DENSE_BYTES=0)
    reference = model.predict(pipeline.transform(X))

    limits(ML_MAX_DENSE_BYTES=2 * 1024**2)
    with pytest.raises(MLEngineException):
        model.predict(pipeline.transform(X))
    chunked = batch_predict(model, X, pipeline=pipeline)
    assert chunked.equals(reference)


# ======================================================================
# 轮廓系数：O(n²) 指标必须先在取数组之前抽样
# ======================================================================
def test_evaluate_clustering_samples_silhouette(limits):
    limits(ML_MAX_SILHOUETTE_SAMPLES=500)
    df = _frame(4000)
    X = df.select(["num_a", "num_b"])
    labels = pl.Series("cluster", (np.arange(4000) % 4))
    result = evaluate_clustering(X, labels)
    assert result["cluster_count"] == 4
    assert isinstance(result["silhouette"], float)
    assert "500" in result["silhouette_note"]
    assert "4,000" in result["silhouette_note"]


def test_evaluate_clustering_skips_silhouette_when_k_invalid():
    df = _frame(50)
    X = df.select(["num_a", "num_b"])
    result = evaluate_clustering(X, pl.Series("cluster", [0] * 50))
    assert result["silhouette"] is None
    assert "无法计算轮廓系数" in result["silhouette_note"]


# ======================================================================
# 入口一：ExperimentService
# ======================================================================
@pytest.fixture()
def dataset_service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


@pytest.fixture()
def experiment_service(db, dataset_service) -> ExperimentService:
    return ExperimentService(db, dataset_service)


def _seeded_experiment(service: ExperimentService, dataset_service: DatasetService, rows: int):
    df = _frame(rows)
    ds = dataset_service.create("mem-guard")
    version = dataset_service.create_version(ds.id, df)
    return service.create(
        dataset_id=ds.id,
        dataset_version_id=version.id,
        task="classification",
        model="logistic_regression",
        target_column="label",
        seed=42,
        description="内存治理回归",
    )


def test_experiment_train_records_sampling_in_artifacts(
    experiment_service, dataset_service, limits
):
    """抽样是有损的：必须写进 artifacts，否则用户会以为指标是在全量上算的。"""
    limits(ML_MAX_TRAIN_ROWS=200)
    exp = _seeded_experiment(experiment_service, dataset_service, rows=900)
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    sampling = run.artifacts["sampling"]
    assert sampling["sampled"] is True
    assert sampling["original_rows"] == 900
    assert sampling["used_rows"] == 200


def test_experiment_train_does_not_sample_small_data(
    experiment_service, dataset_service
):
    exp = _seeded_experiment(experiment_service, dataset_service, rows=120)
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    assert run.artifacts["sampling"]["sampled"] is False


def test_experiment_large_run_stays_within_memory_budget(
    experiment_service, dataset_service, limits
):
    """★ 端到端复现：曾经必炸的数据规模，如今必须成功且带上抽样回执。"""
    limits(ML_MAX_TRAIN_ROWS=300, ML_ONEHOT_MAX_CATEGORIES=20)
    exp = _seeded_experiment(experiment_service, dataset_service, rows=5000)
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    assert run.artifacts["sampling"]["used_rows"] == 300
    assert run.metrics["accuracy"] >= 0.0


# ======================================================================
# 入口二：Workflow runners
# ======================================================================
def test_workflow_ml_train_reports_sampling(limits):
    limits(ML_MAX_TRAIN_ROWS=150)
    runners = build_default_runners()
    df = _frame(800)
    node = Node(
        "train",
        "ml.train",
        {"params": {"model": "logistic_regression", "target_column": "label", "random_state": 42}},
    )
    out = runners["ml.train"](node, {"load": {"_df": df}}, {})
    assert out["sampling"]["sampled"] is True
    assert out["sampling"]["original_rows"] == 800
    # 分类分支先抽样再切分：150 行里 80% 进训练集
    assert out["train_rows"] == 120


def test_workflow_ml_cluster_keeps_every_row_when_sampled(limits):
    """★ 聚类抽样只用于学结构；打标必须覆盖全量，否则下游图少 98% 的点。"""
    limits(ML_MAX_TRAIN_ROWS=120)
    runners = build_default_runners()
    df = _frame(600)
    node = Node(
        "cluster",
        "ml.cluster",
        {"params": {"model": "kmeans", "n_clusters": 3, "output_column": "cluster"}},
    )
    out = runners["ml.cluster"](node, {"load": {"_df": df}}, {})
    assert out["sampling"]["sampled"] is True
    assert out["_df"].height == 600
    assert out["_df"]["cluster"].null_count() == 0


def test_workflow_ml_pca_keeps_every_row_when_sampled(limits):
    limits(ML_MAX_TRAIN_ROWS=100)
    runners = build_default_runners()
    df = _frame(400)
    node = Node("pca", "ml.pca", {"params": {"n_components": 2}})
    out = runners["ml.pca"](node, {"load": {"_df": df}}, {})
    assert out["sampling"]["sampled"] is True
    assert out["_df"].height == 400
    assert out["_df"].columns == ["pc_1", "pc_2"]


def test_workflow_ml_cluster_dbscan_without_sampling_still_works():
    """DBSCAN 无 predict：未抽样时必须保留「降级用 labels_」的旧行为。"""
    runners = build_default_runners()
    df = _frame(60)
    node = Node("cluster", "ml.cluster", {"params": {"model": "dbscan"}})
    out = runners["ml.cluster"](node, {"load": {"_df": df}}, {})
    assert out["sampling"]["sampled"] is False
    assert out["_df"].height == 60
    assert out["_df"]["cluster"].null_count() == 0


def test_workflow_ml_cluster_dbscan_sampled_gives_actionable_error(limits):
    """抽样后 DBSCAN 无法给样本外行打标 —— 必须明确报错，而不是静默丢 98% 的行。"""
    limits(ML_MAX_TRAIN_ROWS=50)
    runners = build_default_runners()
    df = _frame(400)
    node = Node("cluster", "ml.cluster", {"params": {"model": "dbscan"}})
    with pytest.raises(WorkflowException, match="不支持对样本外数据预测簇标签"):
        runners["ml.cluster"](node, {"load": {"_df": df}}, {})


def test_workflow_ml_predict_chunks_without_losing_rows():
    runners = build_default_runners()
    df = _frame(2500)
    train_node = Node(
        "train",
        "ml.train",
        {"params": {"model": "logistic_regression", "target_column": "label", "random_state": 42}},
    )
    trained = runners["ml.train"](train_node, {"load": {"_df": df}}, {})
    predict_node = Node("predict", "ml.predict", {"params": {"output_column": "pred"}})
    predicted = runners["ml.predict"](predict_node, {"train": trained}, {})
    assert predicted["_df"].height == 2500
    assert predicted["_df"]["pred"].null_count() == 0


def test_workflow_ml_train_surfaces_memory_guard_as_workflow_error(limits):
    """预检的中文提示不能被上层当成「未知内部错误」吞掉。"""
    limits(ML_MAX_DENSE_BYTES=256 * 1024, ML_MAX_TRAIN_ROWS=0)
    runners = build_default_runners()
    df = _frame(3000)
    node = Node(
        "train",
        "ml.train",
        {"params": {"model": "logistic_regression", "target_column": "label"}},
    )
    with pytest.raises(WorkflowException, match="训练失败"):
        runners["ml.train"](node, {"load": {"_df": df}}, {})


def test_workflow_ml_evaluate_chunks_without_losing_rows():
    runners = build_default_runners()
    df = _frame(1800)
    train_node = Node(
        "train",
        "ml.train",
        {"params": {"model": "logistic_regression", "target_column": "label", "random_state": 42}},
    )
    trained = runners["ml.train"](train_node, {"load": {"_df": df}}, {})
    eval_node = Node("eval", "ml.evaluate", {"params": {"target_column": "label"}})
    out = runners["ml.evaluate"](eval_node, {"train": trained}, {})
    assert "accuracy" in out["metrics"]
    assert out["rows"] == 1800
