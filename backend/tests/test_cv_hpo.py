"""交叉验证（改动 6）与自动超参搜索（改动 7）的回归测试。

数据全部现场用 polars 构造（300 行二分类，2 个数值特征 + 1 个三值字符串列，
标签与特征真实相关），不依赖数据库、不读磁盘 —— 这两块是纯计算逻辑，
挂上数据库只会让用例变慢变脆。

三条必须钉住的口径：
1. **预处理必须在折内拟合**：`cv_evaluate` 用的是「未拟合管道 + 未拟合 estimator」
   组成的一条 sklearn Pipeline；任何一处先拟合过，CV 指标都会虚高。
2. **回归误差类指标的符号**：sklearn 的 neg_* scorer 是负值，翻不回来
   就会出现「MAE 越大越好」这种反向结论。
3. **搜索结果的键要能直接回填**：`best_params` 不能带 `model__` 前缀，
   否则「应用最佳参数」会把一条 sklearn 不认识的参数塞进训练请求。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from app.ml_engine.cv import SEARCH_SPACES, cv_evaluate, random_search
from app.ml_engine.preprocessing import build_pipeline
from app.ml_engine.registry import MODEL_REGISTRY


def _frame(n: int = 300, seed: int = 7) -> tuple[pl.DataFrame, pl.Series]:
    """构造一份标签与特征真实相关的二分类小数据。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # 三值字符串列：顺带覆盖 one-hot 编码在折内的行为
    cat = rng.choice(["a", "b", "c"], size=n)
    label = ((x1 + 0.8 * x2 + (cat == "a") * 0.6) > 0.3).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "cat": cat}), pl.Series("y", label)


def _estimator(name: str = "logistic_regression") -> object:
    return MODEL_REGISTRY.create(name, {"random_state": 42}).new_estimator()


def test_cv_evaluate_returns_five_folds_with_finite_spread():
    X, y = _frame()
    # 管道刻意**不 fit**：交叉验证要的是未拟合骨架，拟合发生在每折内部
    pipeline = build_pipeline({}, X)
    assert not pipeline.fitted_, "测试前提：管道必须是未拟合的"

    result = cv_evaluate(pipeline, _estimator(), X, y, task="classification", folds=5, seed=42)

    assert result["folds"] == 5
    assert 0.5 <= result["mean"]["accuracy"] <= 1.0, result["mean"]
    # 每一折都要有值，且标准差必须是有限数（NaN 会让前端渲染出空白区间）
    assert len(result["per_fold"]) >= 4
    for name, value in result["std"].items():
        assert np.isfinite(value), f"{name} 的标准差非有限值：{value}"
    folded = {item["metric"]: item["values"] for item in result["per_fold"]}
    assert len(folded["accuracy"]) == 5
    assert all(0.0 <= v <= 1.0 for v in folded["accuracy"])
    # 口径说明必须写出来：只给一堆数字，读者无从判断指标是怎么算的
    assert "折" in result["note"] and "拟合" in result["note"]


def test_cv_as_matrix_does_not_fit_the_pipeline():
    """`as_matrix` 只做「原始帧 -> 矩阵」，绝不能顺手拟合（那是泄漏的入口）。"""
    X, _y = _frame()
    pipeline = build_pipeline({}, X)
    matrix = pipeline.as_matrix(X)
    assert matrix.shape[0] == X.height
    assert not pipeline.fitted_


def test_cv_regression_errors_are_positive_and_rmse_present():
    """回归：neg_* scorer 必须翻回正号，并补出 rmse。"""
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    X = pl.DataFrame({"x": x})
    y = pl.Series("y", 2.0 * x + rng.normal(scale=0.4, size=n))
    pipeline = build_pipeline({}, X)
    result = cv_evaluate(
        pipeline,
        MODEL_REGISTRY.create("linear_regression", {}).new_estimator(),
        X,
        y,
        task="regression",
        folds=5,
        seed=42,
    )
    assert result["mean"]["mae"] > 0, "neg_mean_absolute_error 未翻回正号"
    assert result["mean"]["mse"] > 0, "neg_mean_squared_error 未翻回正号"
    assert result["mean"]["rmse"] > 0
    assert result["mean"]["r2"] > 0.8


def test_random_search_params_are_directly_applicable():
    """best_params 的键必须是模型参数名（能直接回填给训练接口）。"""
    X, y = _frame()
    pipeline = build_pipeline({}, X)
    # 6 组 × 3 折：用例只验证口径，不需要跑满默认 20 组
    result = random_search(
        pipeline, "knn_classifier", X, y,
        task="classification", folds=3, seed=42, n_iter=6,
    )
    assert result["model"] == "knn_classifier"
    assert result["top_results"], "至少应返回一组结果"
    for key in result["best_params"]:
        assert not key.startswith("model__"), f"参数名残留管道前缀：{key}"
        assert key in SEARCH_SPACES["knn_classifier"], f"搜出了搜索空间外的参数：{key}"

    ranks = [item["rank"] for item in result["top_results"]]
    assert ranks == sorted(ranks), f"top_results 未按 rank 升序：{ranks}"
    # 第一名就是 best：两处数字同源，对不上说明「展示的」与「应用的」不是一套
    assert abs(result["best_score"] - result["top_results"][0]["mean_score"]) < 1e-9
    assert result["best_params"] == result["top_results"][0]["params"]


def test_random_search_rejects_model_without_space():
    X, y = _frame()
    pipeline = build_pipeline({}, X)
    from app.ml_engine.exceptions import MLEngineException

    with pytest.raises(MLEngineException) as exc:
        random_search(pipeline, "linear_regression", X, y, task="classification", n_iter=3)
    assert "搜索空间" in str(exc.value)
