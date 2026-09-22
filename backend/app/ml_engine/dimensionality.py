"""Prompt 075：PCA 降维。

输出：
- components：主成分载荷（成分 x 特征）
- explained_variance：各主成分方差
- explained_variance_ratio：方差占比
- transform(X)：转换后的数据 pc_1..pc_k
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn import decomposition

from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException


class PCAAdapter(ModelAdapter):
    name = "pca"
    task = "dimensionality"
    estimator_factory = staticmethod(decomposition.PCA)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {"n_components": 2, **(params or {})}
        super().__init__(merged)
        self.components_: pl.DataFrame | None = None
        self.explained_variance_: list[float] = []
        self.explained_variance_ratio_: list[float] = []

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        return decomposition.PCA(**params)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> PCAAdapter:
        super().fit(X, None)
        k = self.estimator.n_components_
        self.components_ = pl.DataFrame(
            self.estimator.components_,
            schema=list(X.columns),
        ).with_row_index("component")
        self.explained_variance_ = [float(v) for v in self.estimator.explained_variance_]
        self.explained_variance_ratio_ = [
            float(v) for v in self.estimator.explained_variance_ratio_
        ]
        if k != len(self.explained_variance_):  # pragma: no cover
            raise MLEngineException("PCA 组件数与解释方差长度不一致")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.estimator is None:
            raise MLEngineException("PCA 尚未训练，请先调用 fit")
        self._validate_columns(X)
        self._check_numeric(X, stage="转换")
        arr = self.estimator.transform(self._to_numpy(X))
        cols = [f"pc_{i + 1}" for i in range(arr.shape[1])]
        return pl.DataFrame(arr, schema=cols)

    def predict(self, X: pl.DataFrame) -> pl.Series:
        raise MLEngineException(
            "PCA 不支持 predict，请使用 transform 获取降维结果"
        )

    def transform_summary(self) -> dict[str, Any]:
        """components / explained_variance / transformed_data 的摘要信息。"""
        if self.components_ is None:
            raise MLEngineException("PCA 尚未训练，无输出摘要")
        return {
            "components": self.components_.to_dicts(),
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
        }
