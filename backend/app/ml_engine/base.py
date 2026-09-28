"""Prompt 070：统一 ModelAdapter 接口。

所有模型（分类 / 回归 / 聚合 / 降维）都实现同一接口，
上层（Experiment / Agent）只面向 ModelAdapter 编程，
不在业务代码里出现 if/else 模型判断。

约定：
- X: pl.DataFrame（特征，列名即特征名）
- y: pl.Series | None（目标；聚类 / 降维为 None）
- fit 返回 self，支持链式调用
- save/load 通过 pickle 序列化（bytes），可对接 Storage 层

持久化带一层 HMAC 签名（见 `to_signed_bytes`）：模型产物是 pickle，
反序列化即执行代码，一旦被别的文件顶替或传输出错，症状会是「推理结果莫名其妙」
而不是「文件坏了」。签名把这类问题在加载时就变成一条明确报错。
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from app.ml_engine.exceptions import MLEngineException


# ----------------------------------------------------------------------
# 模型产物的完整性信封
#
# 格式：magic(4) + hmac_sha256(32) + payload
# magic 同时承担「这是不是签名过的产物」的判定 —— 没有它就无法区分
# 「签名不匹配」和「旧版本留下的裸 pickle」，后者必须继续能读。
# ----------------------------------------------------------------------
_MAGIC = b"XLB1"
_DIGEST_SIZE = hashlib.sha256().digest_size  # 32

# 没配置环境变量时用的固定密钥。它**不是**安全边界：只保证「文件被截断 /
# 被别的 pickle 顶替 / 传输中损坏」能被发现，防不住知道这个常量的人伪造。
# 真正要防篡改时设置环境变量 ML_MODEL_SIGNING_SECRET（改密钥后旧产物需要重训）。
_DEFAULT_SIGNING_SECRET = b"xiaoluo-lab-model-artifact-v1"


def _signing_secret() -> bytes:
    """模型签名密钥：优先环境变量 `ML_MODEL_SIGNING_SECRET`，否则回落内置常量。"""
    raw = os.environ.get("ML_MODEL_SIGNING_SECRET") or ""
    return raw.encode("utf-8") if raw else _DEFAULT_SIGNING_SECRET


def _digest_for(payload: bytes) -> bytes:
    return hmac.new(_signing_secret(), payload, hashlib.sha256).digest()


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

    def new_estimator(self) -> Any:
        """构造一份**未拟合**的底层 estimator（交叉验证 / 超参搜索用）。

        这两个场景都要自己掌握 fit 的时机（必须在每折的训练部分里 fit），
        因此不能复用 ``fit`` 创建的 ``self.estimator`` —— 那是已经拟合过的。
        """
        return self._make_estimator(self.params)

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

    # ---- 带完整性信封的存取（存储层一律用它，裸 pickle 仅供向后兼容） ----
    def to_signed_bytes(self) -> bytes:
        """序列化为「magic + HMAC + payload」，供存储层落盘。"""
        payload = self.to_bytes()
        return _MAGIC + _digest_for(payload) + payload

    @classmethod
    def from_signed_bytes(cls, data: bytes) -> ModelAdapter:
        """按信封格式解析并**先验签**再反序列化。

        两种失败给出可区分的原因，上层据此决定「拒绝」还是「按旧格式重试」：
        - 不是签名格式（无 magic）→ `MODEL_ARTIFACT_UNSIGNED`
        - 是签名格式但摘要对不上 → `MODEL_ARTIFACT_TAMPERED`（绝不解 pickle）
        """
        if not isinstance(data, (bytes, bytearray)):
            raise MLEngineException(
                "模型产物不是字节流，无法解析",
                code="MODEL_ARTIFACT_TAMPERED",
                details={"type": type(data).__name__},
            )
        raw = bytes(data)
        if len(raw) < len(_MAGIC) + _DIGEST_SIZE:
            raise MLEngineException(
                "模型产物长度不足，可能已被截断",
                code="MODEL_ARTIFACT_TAMPERED",
                details={"size": len(raw)},
            )
        if raw[: len(_MAGIC)] != _MAGIC:
            raise MLEngineException(
                "模型产物缺少完整性签名（可能是升级前保存的旧产物）",
                code="MODEL_ARTIFACT_UNSIGNED",
                details={"size": len(raw)},
            )
        payload = raw[len(_MAGIC) + _DIGEST_SIZE :]
        expected = raw[len(_MAGIC) : len(_MAGIC) + _DIGEST_SIZE]
        if not hmac.compare_digest(_digest_for(payload), expected):
            raise MLEngineException(
                "模型产物签名校验失败：文件被改动过或与保存时的密钥不一致，"
                "为避免加载不可信内容已拒绝反序列化（请重新训练该运行）",
                code="MODEL_ARTIFACT_TAMPERED",
                details={"size": len(payload)},
            )
        return cls.from_bytes(payload)

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
        # 兜底预检：估计器要的是稠密矩阵，10M 行的宽表在这里同样会爆
        # （与 PreprocessingPipeline 的预检同源，覆盖 predict 在大表上的场景）。
        from app.ml_engine.preprocessing import _assert_dense_fits

        _assert_dense_fits(
            X.height,
            X.width,
            stage=f"模型 {self.name} 的特征矩阵",
            hint=(
                "可采取：① 调小 ML_MAX_TRAIN_ROWS 缩小训练/预测集；"
                "② 对高基数列改用 ordinal 编码；③ 调大 ML_MAX_DENSE_BYTES。"
            ),
        )
        return X.to_numpy()

    def _validate_columns(self, X: pl.DataFrame) -> None:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise MLEngineException(
                "预测数据缺少训练时的特征列",
                details={"missing_columns": missing},
            )
