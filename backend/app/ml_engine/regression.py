"""Prompt 073：回归模型适配器。

LinearRegression / KNN / DecisionTree / RandomForest，
统一 ModelAdapter 接口。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn import ensemble, linear_model, neighbors, tree

from app.ml_engine.base import ModelAdapter


class BaseSklearnRegressor(ModelAdapter):
    """sklearn 回归器通用适配器。"""

    task = "regression"
    estimator_factory: Callable[..., Any] | None = None
    default_params: dict[str, Any] = {}

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        merged = {**self.default_params, **params}
        return self.estimator_factory(**merged)


class LinearRegressionAdapter(BaseSklearnRegressor):
    name = "linear_regression"
    estimator_factory = staticmethod(linear_model.LinearRegression)


class KNNRegressorAdapter(BaseSklearnRegressor):
    name = "knn_regressor"
    estimator_factory = staticmethod(neighbors.KNeighborsRegressor)


class DecisionTreeRegressorAdapter(BaseSklearnRegressor):
    name = "decision_tree_regressor"
    estimator_factory = staticmethod(tree.DecisionTreeRegressor)


class RandomForestRegressorAdapter(BaseSklearnRegressor):
    name = "random_forest_regressor"
    estimator_factory = staticmethod(ensemble.RandomForestRegressor)
