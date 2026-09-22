"""Prompt 077：Model Registry。

统一注册 / 查找 / 创建模型，业务层只面向注册表，
不在 ML Engine 里出现 if/else 模型判断。
"""

from __future__ import annotations

from typing import Any

from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException


class ModelRegistry:
    """模型注册表：name -> ModelAdapter 子类。"""

    def __init__(self) -> None:
        self._models: dict[str, type[ModelAdapter]] = {}

    def register(
        self, name: str, adapter_class: type[ModelAdapter]
    ) -> type[ModelAdapter]:
        """注册模型适配器（同名重复注册报错）。"""
        if name in self._models:
            raise MLEngineException(f"模型 {name!r} 已注册")
        self._models[name] = adapter_class
        return adapter_class

    def unregister(self, name: str) -> None:
        if name not in self._models:
            raise MLEngineException(f"模型 {name!r} 未注册")
        del self._models[name]

    def get(self, name: str) -> type[ModelAdapter]:
        if name not in self._models:
            raise MLEngineException(
                f"模型 {name!r} 未注册",
                details={"available": sorted(self._models)},
            )
        return self._models[name]

    def list(self) -> list[dict[str, str]]:
        """已注册模型清单（name + task）。"""
        return [
            {"name": cls.name, "task": cls.task}
            for cls in self._models.values()
        ]

    def create(self, name: str, params: dict[str, Any] | None = None) -> ModelAdapter:
        """按注册名创建模型实例。"""
        return self.get(name)(params)

    def supports_param(self, name: str, param: str) -> bool:
        """模型是否支持某参数（如 random_state），用于 seed 注入前的安全探测。"""
        return self.get(name).supports_param(param)

    def metadata(self, name: str) -> dict[str, Any]:
        """单个模型元信息：名称 / 任务 / 是否支持 random_state / 可注入 seed。"""
        cls = self.get(name)
        return {
            "name": cls.name,
            "task": cls.task,
            "supports_random_state": cls.supports_param("random_state"),
            "supports_probability": cls.task == "classification"
            and hasattr(cls, "predict_proba"),
        }


def _register_builtins(registry: ModelRegistry) -> None:
    """注册内置模型。"""
    from app.ml_engine.classification import (
        DecisionTreeClassifierAdapter,
        KNNClassifierAdapter,
        LogisticRegressionAdapter,
        RandomForestClassifierAdapter,
    )
    from app.ml_engine.clustering import DBSCANAdapter, KMeansAdapter
    from app.ml_engine.dimensionality import PCAAdapter
    from app.ml_engine.regression import (
        DecisionTreeRegressorAdapter,
        KNNRegressorAdapter,
        LinearRegressionAdapter,
        RandomForestRegressorAdapter,
    )

    builtins = (
        LogisticRegressionAdapter,
        KNNClassifierAdapter,
        DecisionTreeClassifierAdapter,
        RandomForestClassifierAdapter,
        LinearRegressionAdapter,
        KNNRegressorAdapter,
        DecisionTreeRegressorAdapter,
        RandomForestRegressorAdapter,
        KMeansAdapter,
        DBSCANAdapter,
        PCAAdapter,
    )
    for cls in builtins:
        registry.register(cls.name, cls)


MODEL_REGISTRY = ModelRegistry()
_register_builtins(MODEL_REGISTRY)
