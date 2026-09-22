"""Prompt 070：统一 ModelAdapter 接口。

所有模型（分类 / 回归 / 聚合 / 降维）都实现同一接口，
上层（Experiment / Agent）只面向 ModelAdapter 编程，
不在业务代码里出现 if/else 模型判断。

约定：
- X: pl.DataFrame（特征，列名即特征名）
- y: pl.Series | None（目标；聚类 / 降维为 None）
- fit 返回 self，支持链式调用
- save/load 通过 pickle 序列化（bytes），可对接 Storage 层
"""

from __future__ import annotations

import inspect
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from app.ml_engine.exceptions import MLEngineException


class ModelAdapter(ABC):
    """模型适配器基类：屏蔽 sklearn 细节，提供统一训练 / 预测 / 评估 / 持久化。"""

    # 注册表中的模型名（由子类声明）
    name: str = ""
    # 任务类型：classification / regression / clustering / dimensionality
    task: str = ""
    # 底层 estimator 工厂（用于公开地探测模型支持哪些参数，避免 try/except 试探）
    estimator_factory: Callable[..., Any] | None = None

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.estimator: Any | None = None
        # fit 时记录的特征顺序，predict 时按此对齐
        self.feature_names_: list[str] = []
        # 训练样本数（推理时用于校验与展示）
        self.n_samples_: int | None = None
        # 训练耗时（秒），由 fit 记录
        self.fit_seconds_: float | None = None

    # ------------------------------------------------------------------
    # 子类只需实现 _make_estimator；fit/predict 由基类统一编排
    # ------------------------------------------------------------------
    @abstractmethod
    def _make_estimator(self, params: dict[str, Any]) -> Any:
        """根据参数构造底层 estimator（sklearn 模型实例）。"""

    @classmethod
    def supports_param(cls, param: str) -> bool:
        """公开探测底层 estimator 是否支持某个参数（如 random_state）。

        相比「构造一次实例并 catch TypeError」，签名探测不产生副作用、
        不依赖私有方法，LinearRegression / KNN 等不支持的模型会正确返回 False。
        """
        factory = cls.estimator_factory
        if factory is None:
            return False
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):  # pragma: no cover - 内建类型无签名
            return False
        if param in signature.parameters:
            return True
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )

    @property
    def trained(self) -> bool:
        """是否已完成训练。"""
        return self.estimator is not None

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> ModelAdapter:
        self._check_numeric(X, stage="训练")
        if X.width == 0:
            raise MLEngineException("训练特征为空（没有可用列）")
        if X.height == 0:
            raise MLEngineException("训练样本为空（0 行）")
        if y is not None and y.len() != X.height:
            raise MLEngineException(
                "X 与 y 行数不一致",
                details={"x_rows": X.height, "y_rows": y.len()},
            )
        self.feature_names_ = list(X.columns)
        self.estimator = self._make_estimator(self.params)
        X_np = self._to_numpy(X)
        started = time.perf_counter()
        if y is None:
            self.estimator.fit(X_np)
        else:
            self.estimator.fit(X_np, y.to_numpy())
        self.fit_seconds_ = round(time.perf_counter() - started, 6)
        self.n_samples_ = X.height
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """返回预测结果（分类 / 回归为标签或数值；聚类为簇标签）。"""
        if self.estimator is None:
            raise MLEngineException("模型尚未训练，请先调用 fit")
        self._validate_columns(X)
        self._check_numeric(X, stage="预测")
        X_np = self._to_numpy(X)
        values = self.estimator.predict(X_np)
        return pl.Series("prediction", values)

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """类别概率（仅部分分类模型支持；默认不支持并给出明确说明）。"""
        raise MLEngineException(
            f"模型 {self.name!r} 不支持 predict_proba（概率输出）"
        )

    def evaluate(self, X: pl.DataFrame, y: pl.Series) -> dict[str, Any]:
        """在给定数据上评估模型（内部复用 evaluation 模块）。"""
        from app.ml_engine import evaluation

        y_pred = self.predict(X)
        if self.task == "regression":
            return evaluation.evaluate_regression(y, y_pred)
        if self.task == "classification":
            y_proba: pl.DataFrame | None = None
            try:
                y_proba = self.predict_proba(X)
            except MLEngineException:
                y_proba = None
            return evaluation.evaluate_classification(y, y_pred, y_proba)
        raise MLEngineException(
            f"任务类型 {self.task!r} 不支持 evaluate（仅分类 / 回归）"
        )

    # ------------------------------------------------------------------
    # 训练元信息与解释
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """模型摘要：名称 / 任务 / 参数 / 特征 / 训练规模。"""
        return {
            "name": self.name,
            "task": self.task,
            "params": dict(self.params),
            "trained": self.trained,
            "feature_names": list(self.feature_names_),
            "n_features": len(self.feature_names_),
            "n_samples": self.n_samples_,
            "fit_seconds": self.fit_seconds_,
        }

    def feature_importance(self) -> dict[str, Any]:
        """特征重要性（委托 explainability，避免此处循环导入）。"""
        from app.ml_engine.explainability import explain_feature_importance

        return explain_feature_importance(self)

    # ------------------------------------------------------------------
    # 持久化：pickle 序列化
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        if self.estimator is None:
            raise MLEngineException("模型尚未训练，无可保存内容")
        payload = {
            "name": self.name,
            "task": self.task,
            "params": self.params,
            "feature_names_": self.feature_names_,
            "n_samples_": self.n_samples_,
            "fit_seconds_": self.fit_seconds_,
            "estimator": self.estimator,
        }
        return pickle.dumps(payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes())
        return target

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelAdapter:
        payload = pickle.loads(data)  # noqa: S301 - 内部可信数据
        instance = cls(payload.get("params") or {})
        instance.estimator = payload["estimator"]
        instance.feature_names_ = payload.get("feature_names_") or []
        instance.n_samples_ = payload.get("n_samples_")
        instance.fit_seconds_ = payload.get("fit_seconds_")
        return instance

    @classmethod
    def load(cls, path: str | Path) -> ModelAdapter:
        return cls.from_bytes(Path(path).read_bytes())

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _check_numeric(self, X: pl.DataFrame, *, stage: str) -> None:
        non_numeric = [
            name for name, dtype in X.schema.items() if not dtype.is_numeric()
        ]
        if non_numeric:
            raise MLEngineException(
                f"{stage}特征必须为数值列，请先执行预处理（encoding/scaling）",
                details={"non_numeric_columns": non_numeric},
            )

    def _to_numpy(self, X: pl.DataFrame) -> Any:
        if self.feature_names_:
            X = X.select(self.feature_names_)
        return X.to_numpy()

    def _validate_columns(self, X: pl.DataFrame) -> None:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise MLEngineException(
                "预测数据缺少训练时的特征列",
                details={"missing_columns": missing},
            )
