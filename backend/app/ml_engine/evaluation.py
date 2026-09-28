"""Prompt 076：统一评估指标。

classification：accuracy / precision / recall / f1 / roc_auc
regression：MAE / MSE / RMSE / R2

指标键统一小写：accuracy, precision, recall, f1, roc_auc, mae, mse, rmse, r2。
roc_auc 需要概率输出（y_proba）；无法提供或 y_true 仅一个类别时值为 None 并附说明。
指标永不返回 NaN/Inf（会破坏 JSON 序列化），一律以 None + 说明替代。
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn import metrics as sk_metrics
from sklearn.calibration import calibration_curve as sk_calibration_curve

from app.ml_engine.exceptions import MLEngineException


def evaluate_classification(
    y_true: pl.Series,
    y_pred: pl.Series,
    y_proba: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """分类评估。y_proba 为按类别升序排列的概率列（predict_proba 输出）。"""
    if y_true.len() != y_pred.len():
        raise MLEngineException("y_true 与 y_pred 长度不一致")
    result: dict[str, Any] = {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(
            sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    roc_auc, note = _roc_auc(y_true, y_proba)
    result["roc_auc"] = roc_auc
    if note:
        result["roc_auc_note"] = note
    return result


def evaluate_regression(y_true: pl.Series, y_pred: pl.Series) -> dict[str, Any]:
    """回归评估。"""
    if y_true.len() != y_pred.len():
        raise MLEngineException("y_true 与 y_pred 长度不一致")
    mse = float(sk_metrics.mean_squared_error(y_true, y_pred))
    return {
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }


#: 未指定 seed 时使用的固定抽样种子。固定值保证「同一份数据 + 未指定 seed」的
#: 两次运行拿到同一个子样本，指标可复现；调用方（ExperimentService）会把实验
#: 自己的 seed 传进来，让模型随机性与抽样随机性共用同一个种子口径。
DEFAULT_SILHOUETTE_SEED = 0


def evaluate_clustering(X: pl.DataFrame, labels: pl.Series, *, seed: int | None = None) -> dict[str, Any]:
    """聚类评估：簇数量 + 轮廓系数（簇数不满足条件时为 None）。

    ``seed`` 只影响「行数超过上限时的随机子样本」怎么抽；不传则用
    :data:`DEFAULT_SILHOUETTE_SEED`。实际使用的种子会回传到结果里
    （``silhouette_sample_seed``），便于在 artifacts 中留痕。
    """
    k = labels.n_unique()
    result: dict[str, Any] = {"cluster_count": int(k)}
    if k < 2 or k >= X.height:
        result["silhouette"] = None
        result["silhouette_note"] = "簇数不满足 2 <= k <= n-1，无法计算轮廓系数"
        return result

    # 轮廓系数的复杂度是 O(n²)：1000 万行上既算不下也算不完。
    # 因此在**取数组之前**先抽样，而不是先 to_numpy() 全量再交给 sklearn。
    from app.ml_engine.preprocessing import (
        _assert_dense_fits,
        _max_silhouette_samples,
    )

    cap = _max_silhouette_samples()
    total = X.height
    # seed 与实验 seed 对齐：模型随机性用 42、抽样却用 0 会让「同一 seed 可复现」
    # 这条承诺在轮廓系数上悄悄失效。未指定时退回固定默认值。
    sample_seed = int(seed) if seed is not None else DEFAULT_SILHOUETTE_SEED
    result["silhouette_sample_seed"] = sample_seed
    if cap > 0 and total > cap:
        rng = np.random.default_rng(sample_seed)
        picked = rng.choice(total, size=cap, replace=False)
        picked.sort()
        idx = picked.tolist()
        X_used, labels_used = X[idx], labels[idx]
        result["silhouette_note"] = (
            f"轮廓系数基于 {cap:,} 行随机子样本（共 {total:,} 行；该指标复杂度为 O(n²)）"
        )
    else:
        X_used, labels_used = X, labels

    _assert_dense_fits(
        X_used.height,
        X_used.width,
        stage="轮廓系数输入",
        hint="可调小 ML_MAX_SILHOUETTE_SAMPLES 或 ML_MAX_TRAIN_ROWS。",
    )
    result["silhouette"] = float(
        sk_metrics.silhouette_score(X_used.to_numpy(), labels_used)
    )
    return result


def confusion_matrix(y_true: pl.Series, y_pred: pl.Series) -> dict[str, Any]:
    """混淆矩阵（分类可视化用）。

    返回 labels（类别升序）+ matrix（行=真实，列=预测），
    全部转 int / str 以保证 JSON 序列化安全。
    """
    if y_true.len() != y_pred.len():
        raise MLEngineException("y_true 与 y_pred 长度不一致")
    # 计数交给 sklearn（这里曾经手写了两重循环做同样的事）：
    # 只需显式给出 labels 顺序，输出的行列语义（行=真实，列=预测）与手工版本一致。
    raw_labels = sorted(
        {_jsonable(v) for v in y_true.to_list()} | {_jsonable(v) for v in y_pred.to_list()},
        key=str,
    )
    matrix = sk_metrics.confusion_matrix(y_true, y_pred, labels=raw_labels)
    return {
        "labels": [str(x) for x in raw_labels],
        "matrix": [[int(v) for v in row] for row in matrix],
    }


def classification_report(y_true: pl.Series, y_pred: pl.Series) -> dict[str, Any]:
    """逐类别 precision / recall / f1 / support（macro 之外的补充视角）。"""
    if y_true.len() != y_pred.len():
        raise MLEngineException("y_true 与 y_pred 长度不一致")
    report = sk_metrics.classification_report(
        y_true.to_numpy(),
        y_pred.to_numpy(),
        output_dict=True,
        zero_division=0,
    )
    per_class: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue  # accuracy / macro avg / weighted avg
        per_class.append(
            {
                "label": str(label),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    per_class.sort(key=lambda item: item["support"], reverse=True)
    return {"per_class": per_class}


def regression_residuals(y_true: pl.Series, y_pred: pl.Series) -> dict[str, Any]:
    """回归残差统计（残差 = 真实 - 预测），含直方图分桶便于前端可视化。"""
    if y_true.len() != y_pred.len():
        raise MLEngineException("y_true 与 y_pred 长度不一致")
    residual = np.asarray(y_true.to_numpy(), dtype="float64") - np.asarray(
        y_pred.to_numpy(), dtype="float64"
    )
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "max_abs": None,
            "p50": None,
            "p95": None,
            "histogram": [],
            "note": "残差全部为空/非有限值",
        }
    bins = _histogram(finite)
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)) if finite.size > 1 else 0.0,
        "max_abs": float(np.abs(finite).max()),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "histogram": bins,
    }


def _histogram(values: "np.ndarray", bin_count: int = 12) -> list[dict[str, Any]]:
    """等宽分桶，输出 {range, count}，用于前端柱状图。"""
    low, high = float(values.min()), float(values.max())
    if low == high:
        return [{"range": f"{low:.3f}~{high:.3f}", "count": int(values.size)}]
    # 分桶交给 numpy（这里曾经手写逐元素累加做同样的事）。
    counts, edges = np.histogram(values, bins=bin_count, range=(low, high))
    return [
        {
            "range": f"{edges[i]:.3f}~{edges[i + 1]:.3f}",
            "count": int(counts[i]),
        }
        for i in range(bin_count)
    ]


def _jsonable(value: Any) -> Any:
    """numpy 标量 -> Python 原生类型（保证可作为 dict key 且可 JSON 序列化）。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - 非数值标量直接回落
            return value
    return value


def _roc_auc(
    y_true: pl.Series, y_proba: pl.DataFrame | None
) -> tuple[float | None, str | None]:
    if y_proba is None or y_proba.width == 0:
        return None, "模型无概率输出，roc_auc 不可用"
    if y_true.n_unique() < 2:
        # sklearn 对单一类别返回 NaN（不抛异常），这里显式置空并说明
        return None, "y_true 仅含一个类别，roc_auc 未定义"
    try:
        proba_np = y_proba.to_numpy()
        if proba_np.shape[1] == 2:
            # 二分类：取第二列（类别升序时的正类概率）
            value = float(sk_metrics.roc_auc_score(y_true, proba_np[:, 1]))
        else:
            value = float(
                sk_metrics.roc_auc_score(
                    y_true, proba_np, multi_class="ovr", average="macro"
                )
            )
    except ValueError as exc:
        return None, f"roc_auc 计算失败：{exc}"
    if not math.isfinite(value):
        return None, "roc_auc 结果为 NaN/Inf，已置空"
    return value, None


# ----------------------------------------------------------------------
# Bootstrap 置信区间：单个指标只是点估计，区间才说明它有多稳
# ----------------------------------------------------------------------

def bootstrap_confidence_interval(
    *,
    values_for_indices: Any,
    metric_fn: Callable[[np.ndarray], float],
    n_repeats: int = 1000,
    seed: int = 42,
) -> dict[str, float | None]:
    """有放回重采样，返回指标的 2.5 / 50 / 97.5 分位（95% 置信区间）。

    为什么需要它：测试集指标是一个**点估计** —— 换一份同规模的测试集，
    accuracy 0.83 可能落在 0.79~0.87。只报点估计，用户会把两个差 0.01 的
    模型当成有差别，而那点差别很可能只是抽样噪声。

    ``values_for_indices`` 传 y_true（或任何长度等于样本数的序列），
    这里只取它的长度确定样本规模；``metric_fn(idx)`` 收到一个重采样下标数组，
    返回该重采样样本上的指标值。**两个 run 传同一份 idx 就是配对比较**
    （见 `experiments/comparator.py`）：配对后消掉了「样本本身难易」的影响，
    剩下的才是两个模型的真实差距。

    重采样可能抽到退化样本（例如某个类别一个都没抽到），此时 metric_fn
    会返回 NaN；NaN 一律剔除，全部退化时返回 None —— 不编一个数字出来，
    也绝不让 NaN 进 JSON。
    """
    n = len(values_for_indices)
    if n == 0:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(max(1, int(n_repeats))):
        idx = rng.integers(0, n, size=n)
        try:
            value = float(metric_fn(idx))
        except Exception:  # noqa: BLE001 - 退化重采样不该让整个诊断失败
            continue
        if math.isfinite(value):
            scores.append(value)
    if not scores:
        return {"lower": None, "median": None, "upper": None}
    q = np.quantile(scores, [0.025, 0.5, 0.975])
    return {
        "lower": float(q[0]),
        "median": float(q[1]),
        "upper": float(q[2]),
        "n_repeats": int(len(scores)),
    }


def class_codes(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """把标签编码成 0..K-1，返回 (y_true_codes, y_pred_codes, K)。

    Bootstrap 要跑上千轮，每轮都让 sklearn 重新解析字符串标签会成为瓶颈；
    编码成整数后一轮重采样只剩一次 bincount。
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    _, inverse = np.unique(np.concatenate([yt, yp]), return_inverse=True)
    n = int(yt.size)
    return (
        inverse[:n].astype(np.int64),
        inverse[n:].astype(np.int64),
        int(inverse.max()) + 1,
    )


def macro_f1_from_codes(y_true_codes: np.ndarray, y_pred_codes: np.ndarray, n_classes: int) -> float:
    """macro F1：一次 bincount 出混淆矩阵，避免每轮重采样都走一遍 sklearn。"""
    k = max(1, int(n_classes))
    flat = y_true_codes.astype(np.int64) * k + y_pred_codes.astype(np.int64)
    cm = np.bincount(flat, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def roc_auc_fast(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """二分类 ROC-AUC（按秩计算，等价于 sklearn 的结果但适合放进重采样循环）。"""
    pos = y_true_binary.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ss = scores[order]
    # 同分并列取平均秩（与 sklearn 一致）。分组边界靠「相邻不等」一次算完，
    # 再用 repeat 铺回原长度 —— Bootstrap 里要跑几百上千轮，逐元素 Python 循环会拖垮训练。
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], ss.size]
    avg = (starts + 1 + ends) / 2.0  # 位置从 1 开始计
    ranks = np.empty(scores.size, dtype="float64")
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------
# 概率校准（二分类）：可靠性曲线 + ECE / MCE
# ----------------------------------------------------------------------

def calibration_analysis(
    y_true: pl.Series,
    y_pos_proba: np.ndarray,
    *,
    positive_label: Any = 1,
    n_bins: int = 10,
) -> dict[str, Any]:
    """二分类概率校准分析，数据全部来自留出测试集上的真实概率输出。

    - 可靠性曲线：每个概率桶「平均预测概率」vs「实际正类比例」，贴对角线=校准好；
    - ECE：各桶 |观测比例−平均预测| 按桶样本数加权平均（越小越好）；
    - MCE：各桶偏差的最大值。

    ``y_pos_proba`` 为正类（类别升序第二列）概率的一维数组。
    """
    y_np = np.asarray(y_true.to_numpy())
    p_np = np.asarray(y_pos_proba, dtype="float64")
    if y_np.shape[0] != p_np.shape[0]:
        raise MLEngineException("y_true 与概率长度不一致")
    finite = np.isfinite(p_np)
    y_np, p_np = y_np[finite], np.clip(p_np[finite], 0.0, 1.0)
    if y_np.size == 0:
        raise MLEngineException("概率全部为非有限值，无法做校准分析")

    n_bins = max(2, min(int(n_bins), 50))
    # 曲线点交给 sklearn.calibration（pos_label 显式指定，避免标签猜测）
    frac_pos, mean_pred = sk_calibration_curve(
        y_np, p_np, n_bins=n_bins, strategy="uniform", pos_label=positive_label
    )
    # 逐桶计数（与 calibration_curve 相同的分桶规则），用于 ECE 与前端明细
    bin_idx = np.clip(np.floor(p_np * n_bins).astype(int), 0, n_bins - 1)
    nonempty = [b for b in range(n_bins) if int((bin_idx == b).sum()) > 0]

    points: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total = int(y_np.size)
    for b, observed, predicted in zip(nonempty, frac_pos, mean_pred):
        count = int((bin_idx == b).sum())
        gap = abs(float(observed) - float(predicted))
        ece += gap * count
        mce = max(mce, gap)
        points.append(
            {
                "bin_start": round(b / n_bins, 4),
                "bin_end": round((b + 1) / n_bins, 4),
                "mean_predicted": round(float(predicted), 6),
                "observed_frequency": round(float(observed), 6),
                "count": count,
            }
        )
    return {
        "basis": "holdout_test",
        "positive_class": str(positive_label),
        "n_bins": n_bins,
        "total": total,
        "ece": round(ece / total, 6),
        "mce": round(mce, 6),
        "points": points,
    }


# ----------------------------------------------------------------------
# 学习曲线：同一模型在递增训练规模上重复拟合，测试集固定不变
# ----------------------------------------------------------------------

def learning_curve_scores(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    estimator: Any,
    *,
    task: str,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """学习曲线：在固定留出测试集上，记录模型在不同训练规模下的真实分数。

    每个规模点都用 ``sklearn.base.clone`` 得到**全新未拟合**估计器真实训练一次，
    分别记录训练子集分数与测试集分数。分类主指标用 accuracy，回归用 r2。
    """
    from sklearn.base import clone

    if task == "classification":
        metric_name = "accuracy"

        def score(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(sk_metrics.accuracy_score(y_t, y_p))
    elif task == "regression":
        metric_name = "r2"

        def score(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(sk_metrics.r2_score(y_t, y_p))
    else:
        raise MLEngineException(f"任务 {task!r} 不支持学习曲线（仅分类 / 回归）")

    n_total = int(X_train_np.shape[0])
    sizes: list[int] = []
    for frac in fractions:
        n = min(n_total, max(2, int(round(n_total * float(frac)))))
        if n not in sizes:
            sizes.append(n)

    points: list[dict[str, Any]] = []
    # train_test_split 的输出已整体打乱，取前 n 行等价于大小为 n 的随机子集
    for n in sizes:
        est = clone(estimator)
        est.fit(X_train_np[:n], y_train_np[:n])
        train_score = score(y_train_np[:n], est.predict(X_train_np[:n]))
        test_score = score(y_test_np, est.predict(X_test_np))
        points.append(
            {
                "fraction": round(n / n_total, 6),
                "rows": int(n),
                # 分数刻意保留到 12 位而不是 6 位：100% 那个点的 train_score 与
                # artifacts.train_metrics 的同一指标是**同一个量**（同参数克隆模型、
                # 同一份全量训练集、同一套已拟合管道），两者必须能对到 1e-9 以内。
                # 按 6 位四舍五入时，回归的 r2 这类任意小数会差出 5e-7，
                # 前端「过拟合诊断」与学习曲线就会给出两个对不上的数。
                "train_score": round(train_score, 12),
                "test_score": round(test_score, 12),
            }
        )
    return {
        "basis": "holdout_test",
        "metric": metric_name,
        "points": points,
        "note": "训练子集取打乱后训练集的前 n 行；每个点都是独立拟合的真实结果。",
    }
