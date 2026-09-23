"""ML Engine · 特征工程（第三层改造）。

* :mod:`app.ml_engine.feature.registry` —— 操作注册表与执行计划（``FeaturePlan``）
* :mod:`app.ml_engine.feature.operations` —— 内置操作（时间解析 / 循环编码 /
  频次编码 / 目标编码 / 分组统计 / 共线性标记）
* :mod:`app.ml_engine.feature.advisor` —— 基于字段语义的建议器

注册表协议与 ``ToolRegistry`` / ``ModelRegistry`` 完全一致（``app.core.registry.Registry``）。
"""

from __future__ import annotations

from app.ml_engine.feature import operations as _operations  # noqa: F401  注册内置操作
from app.ml_engine.feature.advisor import (
    PRIORITY_ORDER,
    FeatureAdvice,
    recommend_features,
)
from app.ml_engine.feature.registry import (
    FEATURE_OP_REGISTRY,
    FeatureOp,
    FeaturePlan,
    apply_plan,
    apply_plans,
    feature_op,
)

__all__ = [
    "FEATURE_OP_REGISTRY",
    "PRIORITY_ORDER",
    "FeatureAdvice",
    "FeatureOp",
    "FeaturePlan",
    "apply_plan",
    "apply_plans",
    "feature_op",
    "recommend_features",
]
