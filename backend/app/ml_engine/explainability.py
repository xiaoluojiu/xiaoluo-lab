"""Prompt 078：模型解释。

第一版支持：
- feature_importance：树模型 feature_importances_ / 线性模型 coef_
- SHAP：可选依赖；未安装或模型不支持时返回明确说明（不静默失败）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException

SHAP_INSTALL_HINT = "SHAP 依赖未安装，无法提供 SHAP 解释；可执行 pip install shap 后重试"


def explain_feature_importance(model: ModelAdapter) -> dict[str, Any]:
    """特征重要性：树/集成模型取 feature_importances_，线性模型取 |coef_| 均值。"""
    if model.estimator is None:
        raise MLEngineException("模型尚未训练，无法解释")
    est = model.estimator
    importances: np.ndarray | None = None
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if importances is None:
        raise MLEngineException(
            f"模型 {model.name!r} 不支持特征重要性解释"
            "（无 feature_importances_ / coef_）"
        )
    ranked = sorted(
        (
            {"feature": name, "importance": float(value)}
            for name, value in zip(model.feature_names_, importances, strict=False)
        ),
        key=lambda x: x["importance"],
        reverse=True,
    )
    return {"method": "feature_importance", "importances": ranked}


def explain_shap(
    model: ModelAdapter, X: pl.DataFrame, *, sample_limit: int = 100
) -> dict[str, Any]:
    """SHAP 解释（可选依赖）。聚类/降维等无预测目标模型不支持。"""
    if model.estimator is None:
        raise MLEngineException("模型尚未训练，无法解释")
    if model.task in ("clustering", "dimensionality"):
        raise MLEngineException(
            f"任务类型 {model.task!r} 无监督输出，不支持 SHAP 解释"
        )
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise MLEngineException(SHAP_INSTALL_HINT) from exc

    model._validate_columns(X)
    model._check_numeric(X, stage="解释")
    X_np = model._to_numpy(X)
    sample = X_np[:sample_limit]
    try:
        explainer = shap.Explainer(
            model.estimator, X_np, feature_names=model.feature_names_
        )
        explanation = explainer(sample)
        values = np.asarray(explanation.values)
        # 形状归一：(n, features, classes) 或 (n, features) -> 各样本/类别平均绝对值
        if values.ndim == 3:
            mean_abs = np.abs(values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(values).mean(axis=0)
    except Exception as exc:
        raise MLEngineException(
            f"SHAP 计算失败：{exc}（模型 {model.name!r} 可能不受支持）"
        ) from exc

    ranked = sorted(
        (
            {"feature": name, "mean_abs_shap": float(v)}
            for name, v in zip(model.feature_names_, mean_abs, strict=False)
        ),
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )
    return {
        "method": "shap",
        "sample_count": int(sample.shape[0]),
        "mean_abs_shap": ranked,
    }


def explain(
    model: ModelAdapter,
    X: pl.DataFrame | None = None,
    *,
    method: str = "feature_importance",
) -> dict[str, Any]:
    """解释入口分发。method: feature_importance / shap。"""
    if method == "feature_importance":
        return explain_feature_importance(model)
    if method == "shap":
        if X is None:
            raise MLEngineException("SHAP 解释需要提供样本数据 X")
        return explain_shap(model, X)
    raise MLEngineException(
        f"未知解释方法 {method!r}（支持：feature_importance / shap）"
    )
