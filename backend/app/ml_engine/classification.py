"""Prompt 072：分类模型适配器。

LogisticRegression / KNN / DecisionTree / RandomForest / HistGradientBoosting，
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
    # 与 random_forest_regressor 同口径：n_jobs=-1 并行 + max_depth=16 限深，
    # 避免默认 max_depth=None 在高基数 one-hot 特征上既慢又过拟合。
    default_params = {"n_jobs": -1, "max_depth": 16}


class HistGradientBoostingClassifierAdapter(BaseSklearnClassifier):
    """直方图梯度提升分类（sklearn 的 LightGBM 同族实现）。

    为什么值得补进模型库：随机森林在高基数 one-hot 特征上要建上百棵树才稳，
    而直方图方法先把连续特征分箱成最多 ``max_bins`` 个桶，再在桶上找分裂点，
    训练复杂度随**桶数**而不是样本数增长 —— 表格数据上通常是「更快且更准」的那个。

    两个与它绑定的教学事实（写死在 metadata.MODEL_PARAMS，不靠前端猜）：
    - ``max_leaf_nodes`` 上限 31：内部用 8 位存叶节点索引，超过会直接报错；
    - ``early_stopping='auto'``：样本过万时自动留一小份做验证，收敛就停。
    """

    name = "hist_gradient_boosting_classifier"
    estimator_factory = staticmethod(ensemble.HistGradientBoostingClassifier)
