"""Prompt 072：分类模型适配器。

LogisticRegression / KNN / DecisionTree / RandomForest，
统一 ModelAdapter 接口，参数透传给 sklearn，不做模型特判。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
from sklearn import ensemble, linear_model, neighbors, tree

from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException


class BaseSklearnClassifier(ModelAdapter):
    """sklearn 分类器通用适配器：子类只需指定工厂与默认参数。"""

    task = "classification"
    estimator_factory: Callable[..., Any] | None = None
    default_params: dict[str, Any] = {}

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        merged = {**self.default_params, **params}
        return self.estimator_factory(**merged)

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.estimator is None:
            raise MLEngineException("模型尚未训练，请先调用 fit")
        self._validate_columns(X)
        self._check_numeric(X, stage="预测")
        proba = self.estimator.predict_proba(self._to_numpy(X))
        classes = list(self.estimator.classes_)
        data = {
            f"prob_{c}": [float(row[i]) for row in proba]
            for i, c in enumerate(classes)
        }
        return pl.DataFrame(data)


class LogisticRegressionAdapter(BaseSklearnClassifier):
    name = "logistic_regression"
    estimator_factory = staticmethod(linear_model.LogisticRegression)
    default_params = {"max_iter": 1000}


class KNNClassifierAdapter(BaseSklearnClassifier):
    name = "knn_classifier"
    estimator_factory = staticmethod(neighbors.KNeighborsClassifier)


class DecisionTreeClassifierAdapter(BaseSklearnClassifier):
    name = "decision_tree_classifier"
    estimator_factory = staticmethod(tree.DecisionTreeClassifier)


class RandomForestClassifierAdapter(BaseSklearnClassifier):
    name = "random_forest_classifier"
    estimator_factory = staticmethod(ensemble.RandomForestClassifier)
