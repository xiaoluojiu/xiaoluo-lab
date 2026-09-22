"""Prompt 074：聚类模型适配器。

KMeans / DBSCAN。聚类无目标 y；fit 后通过 labels_ 获取训练样本的簇标签。
DBSCAN 不支持对全新样本预测（sklearn 无 predict），predict 给出明确说明。
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn import cluster

from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException


class KMeansAdapter(ModelAdapter):
    name = "kmeans"
    task = "clustering"
    # 暴露工厂，供 supports_param 公开探测模型参数（如 random_state）
    estimator_factory = staticmethod(cluster.KMeans)

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.labels_: pl.Series | None = None

    def _make_estimator(self, params: dict) -> Any:
        return cluster.KMeans(**params)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> KMeansAdapter:
        super().fit(X, None)
        self.labels_ = pl.Series("cluster", self.estimator.labels_)
        return self

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        raise MLEngineException("聚类模型不支持 predict_proba（概率输出）")


class DBSCANAdapter(ModelAdapter):
    name = "dbscan"
    task = "clustering"
    estimator_factory = staticmethod(cluster.DBSCAN)

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.labels_: pl.Series | None = None

    def _make_estimator(self, params: dict) -> Any:
        return cluster.DBSCAN(**params)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> DBSCANAdapter:
        super().fit(X, None)
        self.labels_ = pl.Series("cluster", self.estimator.labels_)
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        raise MLEngineException(
            "DBSCAN 不支持对全新样本预测（无 predict 方法），"
            "训练样本的簇标签请读取 labels_"
        )

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        raise MLEngineException("聚类模型不支持 predict_proba（概率输出）")
