"""Prompt 077：Model Registry。

统一注册 / 查找 / 创建模型，业务层只面向注册表，
不在 ML Engine 里出现 if/else 模型判断。
"""

from __future__ import annotations

from typing import Any

from app.core.registry import Registry
from app.ml_engine.base import ModelAdapter
from app.ml_engine.exceptions import MLEngineException


class ModelRegistry(Registry[type[ModelAdapter]]):
    """模型注册表：name -> ModelAdapter 子类。

    键值管理与报错复用 :class:`app.core.registry.Registry`（标准化内核），
    与 ``ToolRegistry`` / 特征工程操作注册表共用同一套协议，
    本类只保留模型层特有的 ``create`` / ``metadata`` / ``supports_param``。
    """

    def __init__(self) -> None:
        super().__init__(label="模型")

    # ---- Registry 错误钩子：沿用 ML 层既有异常体系 ----------------------
    def _conflict_error(self, key: str) -> MLEngineException:
        return MLEngineException(f"模型 {key!r} 已注册")

    def _missing_error(self, key: str) -> MLEngineException:
        return MLEngineException(f"模型 {key!r} 未注册", details={"available": self.keys()})

    def list(self) -> list[dict[str, str]]:
        """已注册模型清单（name + task）。"""
        return [
            {"name": cls.name, "task": cls.task}
            for cls in self.values()
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
