"""Prompt 076：统一评估指标。

classification：accuracy / precision / recall / f1 / roc_auc
regression：MAE / MSE / RMSE / R2

指标键统一小写：accuracy, precision, recall, f1, roc_auc, mae, mse, rmse, r2。
roc_auc 需要概率输出（y_proba）；无法提供或 y_true 仅一个类别时值为 None 并附说明。
指标永不返回 NaN/Inf（会破坏 JSON 序列化），一律以 None + 说明替代。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from sklearn import metrics as sk_metrics

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


def evaluate_clustering(X: pl.DataFrame, labels: pl.Series) -> dict[str, Any]:
    """聚类评估：簇数量 + 轮廓系数（簇数不满足条件时为 None）。"""
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
    if cap > 0 and total > cap:
        rng = np.random.default_rng(0)
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
