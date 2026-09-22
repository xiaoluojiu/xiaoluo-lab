"""推理阶段参数：决策阈值。

超参数决定「模型学成什么样」，决策阈值决定「概率怎么变成类别」——
同一个模型、同一组概率，换个阈值就能用召回换精确率，**不需要重新训练**。
sklearn 的 `predict` 等价于「正类概率 > 0.5 才算正类」，也就是 0.5 阈值下的 argmax。
换句话说：概率列早就返回给前端了，却没有任何入口能作用在它上面 —— 这是它必须被
显式暴露（而不是埋在默认值里）的原因。

两条边界：
1. **只适用二分类**。类数 > 2 时「一个阈值」没有明确语义（argmax 是对全部类别一起比较的），
   给出一个看起来能调、实际含义不明的旋钮，比不给更糟。因此多分类不由本模块提供。
2. **正类固定为 `classes_` 的最后一个**。sklearn 的 `classes_` 按升序排列，
   `predict_proba` 的列与之一一对应（见 BaseSklearnClassifier.predict_proba），
   因此最后一列就是正类概率，与 sklearn 自身 `predict` 的口径一致。
3. **比较用严格大于（`>`），不是 `>=`**。sklearn 二分类走的是 `decision_function > 0`，
   概率恰好 0.5 时判为 `classes_[0]`（负类）。用 `>=` 会让「阈值 0.5」对决策树这类
   概率会精确取到 0.5 的模型改变标签 —— 而阈值功能最基本的承诺恰恰是
   「取默认值＝与原来行为完全一致」，一旦被打破，用户就没法把改动归因到阈值上。
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn import metrics as sk_metrics

# 概率列前缀（与 BaseSklearnClassifier.predict_proba 的列命名一致）
PROBA_PREFIX = "prob_"

# 默认阈值：与 sklearn 的 argmax 等价（二分类下即 0.5）
DEFAULT_THRESHOLD = 0.5

# 曲线扫描档位：0.05 ~ 0.95，步长 0.05。
# 刻意不含 0 / 1 —— 那两端会把所有样本判成同一类，指标退化成常量，没有选择价值。
THRESHOLD_GRID: list[float] = [round(0.05 * i, 2) for i in range(1, 20)]


def _native(value: Any) -> Any:
    """把 numpy 标量转成 Python 原生类型，保证能安全 JSON 序列化与相等比较。"""
    return value.item() if hasattr(value, "item") else value


def model_classes(model: Any) -> list[Any] | None:
    """取模型训练时见过的类别（真实取值与类型，不是字符串化的列名）。

    取不到时返回 None，调用方据此跳过阈值功能，而不是猜一个出来。
    """
    estimator = getattr(model, "estimator", None)
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return None
    try:
        return [_native(c) for c in classes]
    except TypeError:  # pragma: no cover - 不可迭代的异常实现
        return None


def is_binary(classes: list[Any] | None) -> bool:
    """是否恰好两个类别（阈值功能的前提）。"""
    return classes is not None and len(classes) == 2


def positive_probabilities(y_proba: pl.DataFrame) -> list[float]:
    """取正类概率列（`predict_proba` 输出的最后一列）。"""
    return [float(v) for v in y_proba[y_proba.columns[-1]].to_list()]


def apply_threshold(
    y_proba: pl.DataFrame, classes: list[Any], threshold: float
) -> pl.Series:
    """按阈值把正类概率映射成类别标签：`> threshold` 判为正类。

    严格大于是刻意的（见模块 docstring 第 3 条）：这样 threshold=DEFAULT_THRESHOLD
    与 sklearn 的默认 predict 逐样本一致，包括概率恰为 0.5 的并列样本。
    """
    neg_label, pos_label = classes[0], classes[-1]
    probs = positive_probabilities(y_proba)
    return pl.Series(
        "prediction", [pos_label if p > threshold else neg_label for p in probs]
    )


def label_shift_ratio(
    y_proba: pl.DataFrame, classes: list[Any], threshold: float
) -> float:
    """阈值标签与默认（0.5/argmax）标签不一致的样本占比。

    这是「改阈值到底有没有影响结果」最直接的量化答案：
    占比为 0 说明这个阈值在当前数据上等价于默认行为，用户没必要为它纠结。
    """
    probs = positive_probabilities(y_proba)
    if not probs:
        return 0.0
    changed = sum(
        1
        for p in probs
        if (p > threshold) != (p > DEFAULT_THRESHOLD)
    )
    return round(changed / len(probs), 4)


def _empty_curve(reason: str) -> dict[str, Any]:
    """曲线不可用时的统一返回：带原因，前端据此说明「为什么没有曲线」而不是空白。"""
    return {
        "points": [],
        "reason": reason,
        "positive_class": None,
        "basis_rows": 0,
        "metric_average": "positive",
        "suggested_threshold": None,
        "suggested_f1": None,
    }


def threshold_curve(
    y_true: pl.Series,
    y_proba: pl.DataFrame,
    classes: list[Any],
    grid: list[float] | None = None,
) -> dict[str, Any]:
    """在带真实标签的数据上扫阈值，给出「阈值 → 指标」的取舍曲线。

    指标口径是**正类**（precision / recall / f1 用 pos_label 计算），不是 run.metrics 的
    macro 口径 —— 调阈值的核心动作就是「拿正类的召回换精确率」，用 macro 会把两类的
    变化平均掉、看不出取舍方向。因此返回值里显式带 `metric_average: "positive"`，
    避免它被误当成与 run.metrics 同口径的数字直接比较。

    数据不满足条件（非二分类、长度不一致、正类缺失、全为空）时返回空 points，
    调用方据此不展示曲线，而不是展示一条没有意义的折线。
    """
    if not is_binary(classes):
        return _empty_curve("非二分类任务，单个阈值不具备两类取舍意义")
    if y_proba.width < 2:
        return _empty_curve("概率矩阵不足两列，无法定位正类概率")

    labels = y_true.to_list()
    probs = positive_probabilities(y_proba)
    # 逐行配对时顺手剔除空标签：predict_proba 不产出 None，但真实标签可能缺，
    # 而 sklearn 的指标函数遇到 None 会直接抛错。
    pairs = [(p, v) for p, v in zip(probs, labels) if v is not None]
    if not pairs:
        return _empty_curve("没有带真实标签的样本可评估")
    probs = [p for p, _ in pairs]
    labels = [v for _, v in pairs]

    neg_label, pos_label = classes[0], classes[-1]
    if pos_label not in set(labels):
        return _empty_curve(f"真实标签中没有正类 {pos_label!r}，正类指标无定义")

    points: list[dict[str, Any]] = []
    for t in grid or THRESHOLD_GRID:
        threshold = float(t)
        pred = [pos_label if p > threshold else neg_label for p in probs]
        points.append(
            {
                "threshold": round(threshold, 4),
                "precision": float(
                    sk_metrics.precision_score(
                        labels, pred, pos_label=pos_label, zero_division=0
                    )
                ),
                "recall": float(
                    sk_metrics.recall_score(
                        labels, pred, pos_label=pos_label, zero_division=0
                    )
                ),
                "f1": float(
                    sk_metrics.f1_score(
                        labels, pred, pos_label=pos_label, zero_division=0
                    )
                ),
                "accuracy": float(sk_metrics.accuracy_score(labels, pred)),
                "positive_rate": round(
                    sum(1 for v in pred if v == pos_label) / len(pred), 4
                ),
            }
        )

    # 推荐点＝F1 最优（并列时偏向 accuracy 更高的一侧）
    best = max(points, key=lambda p: (p["f1"], p["accuracy"]))
    return {
        "points": points,
        "reason": None,
        "positive_class": pos_label,
        "basis_rows": len(labels),
        "metric_average": "positive",
        "suggested_threshold": best["threshold"],
        "suggested_f1": best["f1"],
    }
