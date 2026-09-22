"""推理阶段参数（决策阈值）测试：`app/ml_engine/threshold.py`。

刻意不训练任何模型：阈值语义只依赖「概率数组 + 类别列表」，
用构造数据就能把语义钉死，所以这组用例的运行时间可以忽略。
唯一需要与外部对齐的事实是 sklearn 自己的判定边界，用 argmax 不变式锁定。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from app.ml_engine import threshold as T


class _FakeModel:
    """只带 classes_ 的假模型：用于验证类别还原，不触发任何训练。"""

    def __init__(self, classes: list[object]) -> None:
        self.estimator = type("E", (), {"classes_": np.array(classes)})()


@pytest.fixture()
def proba() -> pl.DataFrame:
    """含一个「正类概率恰好等于 0.5」的并列样本，用来钉住边界口径。"""
    return pl.DataFrame(
        {"prob_0": [0.9, 0.6, 0.5, 0.4, 0.1], "prob_1": [0.1, 0.4, 0.5, 0.6, 0.9]}
    )


@pytest.fixture()
def binary_classes() -> list[int]:
    return [0, 1]


def test_model_classes_returns_native_values() -> None:
    classes = T.model_classes(_FakeModel([0, 1]))
    assert classes == [0, 1]
    # numpy 标量必须在模块边界上转成原生类型，否则 JSON 序列化与相等比较都会出问题
    assert all(type(c) is int for c in classes)


def test_model_classes_missing_estimator_returns_none() -> None:
    assert T.model_classes(object()) is None


def test_is_binary() -> None:
    assert T.is_binary(T.model_classes(_FakeModel([0, 1]))) is True
    assert T.is_binary(T.model_classes(_FakeModel([0, 1, 2]))) is False
    assert T.is_binary(None) is False


def test_threshold_compares_strictly(proba: pl.DataFrame, binary_classes: list[int]) -> None:
    """边界是严格大于：概率恰为 0.5 判负类，与 sklearn 的 predict 口径一致。"""
    assert T.apply_threshold(proba, binary_classes, 0.5).to_list() == [0, 0, 0, 1, 1]


def test_threshold_direction(proba: pl.DataFrame, binary_classes: list[int]) -> None:
    assert T.apply_threshold(proba, binary_classes, 0.6).to_list() == [0, 0, 0, 0, 1]
    assert T.apply_threshold(proba, binary_classes, 0.2).to_list() == [0, 1, 1, 1, 1]


def test_default_threshold_matches_argmax() -> None:
    """不变式：threshold=0.5 必须与 argmax 逐样本一致，含并列样本。

    这条是「取默认值＝与原来行为完全一致」这个承诺的唯一保障，
    一旦破坏，用户就无法把标签变化归因到阈值上。
    """
    rng = np.random.default_rng(7)
    p1 = rng.random(200)
    p1[:5] = 0.5  # 刻意制造并列：sklearn 在这里取 classes_[0]
    p0 = 1.0 - p1
    frame = pl.DataFrame({"prob_0": p0, "prob_1": p1})
    expected = np.argmax(np.column_stack([p0, p1]), axis=1).tolist()
    assert T.apply_threshold(frame, [0, 1], T.DEFAULT_THRESHOLD).to_list() == expected


def test_label_shift_ratio(proba: pl.DataFrame, binary_classes: list[int]) -> None:
    # 默认阈值必须给出 0：否则「没改任何东西」也会显示标签被改动
    assert T.label_shift_ratio(proba, binary_classes, 0.5) == 0.0
    assert T.label_shift_ratio(proba, binary_classes, 0.9) == 0.4


def test_grid_excludes_extremes() -> None:
    """0 与 1 会把所有样本判成同一类，指标退化成常量，故不参与扫描。"""
    assert T.THRESHOLD_GRID[0] > 0.0
    assert T.THRESHOLD_GRID[-1] < 1.0
    assert T.DEFAULT_THRESHOLD in T.THRESHOLD_GRID


def test_curve_tradeoff_and_suggestion(binary_classes: list[int]) -> None:
    rng = np.random.default_rng(0)
    n = 400
    score = rng.normal(0, 1, n)
    y_true = (score + rng.normal(0, 1, n) > 0).astype(int)
    p1 = 1 / (1 + np.exp(-score))
    frame = pl.DataFrame({"prob_0": 1 - p1, "prob_1": p1})

    curve = T.threshold_curve(pl.Series("target", y_true), frame, binary_classes)
    points = curve["points"]
    assert len(points) == len(T.THRESHOLD_GRID)
    # 口径必须显式标注，否则曲线数值会被当成 run.metrics 的 macro 口径去比较
    assert curve["metric_average"] == "positive"
    assert curve["positive_class"] == 1
    assert curve["basis_rows"] == n

    low, high = points[0], points[-1]
    assert high["precision"] >= low["precision"] - 1e-9
    assert high["recall"] <= low["recall"] + 1e-9
    assert high["positive_rate"] <= low["positive_rate"] + 1e-9

    best = max(points, key=lambda point: point["f1"])
    assert curve["suggested_threshold"] == best["threshold"]


def test_curve_metrics_are_json_safe(binary_classes: list[int]) -> None:
    frame = pl.DataFrame({"prob_0": [0.9, 0.6, 0.3, 0.1], "prob_1": [0.1, 0.4, 0.7, 0.9]})
    points = T.threshold_curve(pl.Series("t", [0, 0, 1, 1]), frame, binary_classes)["points"]
    keys = ("precision", "recall", "f1", "accuracy", "positive_rate")
    # NaN / Inf 会破坏 JSON 序列化，指标一律是 [0,1] 内的原生 float
    assert all(
        isinstance(p[k], float) and 0.0 <= p[k] <= 1.0
        for p in points
        for k in keys
    )


def test_curve_rejects_multiclass() -> None:
    frame = pl.DataFrame(
        {"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3], "c": [0.8, 0.6, 0.4]}
    )
    curve = T.threshold_curve(pl.Series("t", [0, 1, 2]), frame, [0, 1, 2])
    assert curve["points"] == []
    assert curve["reason"]


def test_curve_rejects_missing_positive_class(
    proba: pl.DataFrame, binary_classes: list[int]
) -> None:
    curve = T.threshold_curve(pl.Series("t", [0, 0, 0, 0, 0]), proba, binary_classes)
    assert curve["points"] == []
    assert curve["reason"]


def test_curve_drops_null_labels(proba: pl.DataFrame, binary_classes: list[int]) -> None:
    curve = T.threshold_curve(pl.Series("t", [0, None, 1, 1, 0]), proba, binary_classes)
    assert len(curve["points"]) == len(T.THRESHOLD_GRID)
    assert curve["basis_rows"] == 4
