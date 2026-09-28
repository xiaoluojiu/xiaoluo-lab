"""ML 引擎冒烟：分类 / 回归 / 聚类三条链路各跑一次真实训练。

为什么需要它：ML 模块此前**零测试**，而它的失败模式集中在「数据/切分/管道」
的接缝上（高基数列进特征、分层切分的行数下限、预处理只 fit 训练集），
单看函数签名全都正常，只有真跑一遍才暴露。

数据全部在用例里用 polars 现造（200 行），不依赖仓库里任何数据集文件，
也不假设某个 dataset_id 存在 —— 数据集与版本走真实的 DatasetService 落盘，
这样 `ExperimentService.run()` 读到的是与线上同构的 Parquet 版本快照。
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import polars as pl
import pytest

from app.api.deps import get_storage_service
from app.core.database import SessionLocal
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService

ROWS = 200


def _classification_frame(rows: int = ROWS) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=rows)
    f2 = rng.normal(size=rows)
    cat = np.array(["a", "b", "c"])[rng.integers(0, 3, rows)]
    # 标签与特征真的相关 —— 否则指标接近随机水平，什么都验证不了
    logit = 1.4 * f1 - 0.9 * f2 + (cat == "a") * 0.8 + rng.normal(scale=0.5, size=rows)
    return pl.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "cat": cat,
            "y": (logit > 0).astype(int),
        }
    )


def _regression_frame(rows: int = ROWS) -> pl.DataFrame:
    rng = np.random.default_rng(13)
    f1 = rng.normal(size=rows)
    f2 = rng.normal(size=rows)
    return pl.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "y": 3.0 * f1 - 2.0 * f2 + rng.normal(scale=0.3, size=rows),
        }
    )


def _clustering_frame(rows: int = ROWS) -> pl.DataFrame:
    """三团明显分开的点：轮廓系数必须显著大于 0，否则等于没验证聚类。"""
    rng = np.random.default_rng(17)
    centers = np.array([[-6.0, -6.0], [0.0, 6.0], [6.0, -4.0]])
    labels = rng.integers(0, 3, rows)
    points = centers[labels] + rng.normal(scale=0.7, size=(rows, 2))
    return pl.DataFrame({"x1": points[:, 0], "x2": points[:, 1]})


def _dump(frame: pl.DataFrame) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp(prefix="ml-smoke-")) / "frame.parquet"
    frame.write_parquet(path)
    return path


@pytest.fixture()
def ml_env(tmp_path: pathlib.Path):
    """一个真实的数据集（带一个初始版本），各用例再按需追加自己的版本。"""
    storage = get_storage_service()
    db = SessionLocal()
    try:
        dataset_service = DatasetService(db, storage)
        dataset = dataset_service.create("ml-engine-smoke")
        parquet = tmp_path / "seed.parquet"
        _classification_frame().write_parquet(parquet)
        dataset_service.create_version_from_source(dataset.id, parquet)
        yield db, dataset_service, dataset.id
    finally:
        db.close()


def _train(db, dataset_service, dataset_id: int, frame: pl.DataFrame, **kwargs):
    """把给定数据另存为一个版本，然后走完整的 create → run。"""
    version, _ = dataset_service.create_version_from_source(dataset_id, _dump(frame))
    service = ExperimentService(db, dataset_service)
    exp = service.create(
        dataset_id=dataset_id,
        dataset_version_id=version.id,
        seed=42,
        description="smoke",
        **kwargs,
    )
    run = service.run(exp.id)
    assert run.status == "success", f"训练失败：{run.error}"
    return run


def test_classification_run_produces_full_metrics(ml_env):
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _classification_frame(),
        task="classification",
        model="logistic_regression",
        target_column="y",
    )
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in run.metrics, f"缺少指标 {key}：{run.metrics}"
        assert run.metrics[key] is not None, f"{key} 不应为 None"

    # 混淆矩阵对角线合计 / 测试行数 必须等于 accuracy —— 两个独立算出来的数互相印证，
    # 任何一边用错数据（比如拿训练集算的矩阵）都会立刻对不上。
    matrix = run.artifacts["confusion_matrix"]
    correct = sum(matrix["matrix"][i][i] for i in range(len(matrix["labels"])))
    test_rows = run.artifacts["test_rows"]
    assert abs(correct / test_rows - run.metrics["accuracy"]) < 1e-6


def test_regression_run_produces_error_metrics(ml_env):
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _regression_frame(),
        task="regression",
        model="linear_regression",
        target_column="y",
    )
    for key in ("mae", "rmse", "r2"):
        assert key in run.metrics, f"缺少指标 {key}：{run.metrics}"
    # 特征与目标的关系是刻意造出来的强线性关系，r2 必须接近 1；
    # 若这里拿到 0.0 甚至负数，说明预处理或切分把 y 弄丢了。
    assert run.metrics["r2"] > 0.8, run.metrics


def test_clustering_run_produces_silhouette(ml_env):
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _clustering_frame(),
        task="clustering",
        model="kmeans",
        target_column=None,
        parameters={"n_clusters": 3, "n_init": 10},
    )
    assert run.metrics["cluster_count"] == 3
    assert run.metrics["silhouette"] is not None
    assert run.metrics["silhouette"] > 0.5, run.metrics
    # 聚类没有留出集，train_metrics 必须显式为 null（前端据此跳过过拟合诊断）
    assert run.artifacts["train_metrics"] is None


def test_learning_curve_endpoint_matches_train_metrics(ml_env):
    """学习曲线 100% 点与 artifacts.train_metrics 必须是同一个数。

    两者都是「同参数模型在全量训练集上的分数」，只是计算路径不同：
    train_metrics 用已拟合的 adapter 直接 predict，学习曲线用 clone 重新拟合一次。
    一旦它们对不上（抽样口径不同、重复 fit 带了不同数据、或精度被截断），
    前端过拟合诊断与学习曲线就会给出互相矛盾的两个数。
    """
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _classification_frame(),
        task="classification",
        model="logistic_regression",
        target_column="y",
        enable_learning_curve=True,
    )
    curve = run.artifacts.get("learning_curve")
    assert curve, "勾选后必须产出学习曲线"
    assert len(curve["points"]) <= 5

    last = curve["points"][-1]
    train_metrics = run.artifacts["train_metrics"]
    assert last["rows"] == run.artifacts["train_rows"]
    assert abs(train_metrics["accuracy"] - last["train_score"]) < 1e-9, (
        f"train_metrics.accuracy={train_metrics['accuracy']} 与 "
        f"学习曲线终点 train_score={last['train_score']} 不一致"
    )
    # 终点测试集分数与 run.metrics 的主指标同样必须一致
    assert abs(run.metrics["accuracy"] - last["test_score"]) < 1e-6


def test_learning_curve_absent_when_not_requested(ml_env):
    """未勾选时不得产生任何额外拟合 —— 训练产物里连键都不该出现。"""
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _classification_frame(),
        task="classification",
        model="logistic_regression",
        target_column="y",
    )
    assert "learning_curve" not in run.artifacts


def test_binary_classification_always_carries_calibration(ml_env):
    """二分类的概率校准默认开启（无需开关），坐标必须落在 [0,1] 内。"""
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _classification_frame(),
        task="classification",
        model="logistic_regression",
        target_column="y",
    )
    calibration = run.artifacts.get("calibration")
    assert calibration, "二分类必须产出校准结果"
    assert 0.0 <= calibration["ece"] <= 1.0
    assert 0.0 <= calibration["mce"] <= 1.0
    assert calibration["points"], "可靠性曲线不能为空"
    for point in calibration["points"]:
        assert 0.0 <= point["mean_predicted"] <= 1.0
        assert 0.0 <= point["observed_frequency"] <= 1.0


def test_regression_has_no_calibration(ml_env):
    """校准只对二分类有意义，回归结果里不该出现这个区块。"""
    db, dataset_service, dataset_id = ml_env
    run = _train(
        db,
        dataset_service,
        dataset_id,
        _regression_frame(),
        task="regression",
        model="linear_regression",
        target_column="y",
    )
    assert "calibration" not in run.artifacts
