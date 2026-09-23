"""ML Engine · 特征工程操作注册表（第三层改造）。

为什么需要它
------------
``CRSDepTime`` 是 HHMM 伪数值（2359→0000 有跳变），``Origin`` / ``Dest`` 有
368~369 个取值，``CRSDepTime`` 与 ``CRSArrTime`` 相关系数 r=0.78 ——
这三件事在延误数据集上真实存在，而平台**一件都没处理**：高基数列被 one-hot
展开成 767 列并撑爆内存，共线性没被提示，时间字段直接当数值入模。

根因与 Tool / Model / DataEngine 操作同构：**特征工程没有注册表**，
于是每个需要它的地方各写各的 `if 列名 == 'xxx'`，既不复用也不可发现。

本模块把特征工程标准化为一张注册表（复用 ``app.core.registry.Registry``，
与 ``ToolRegistry`` / ``ModelRegistry`` 同一套协议）：

* 每个操作自己声明 **适用条件**（``applies``）—— 由字段语义驱动，不是由列名硬编码；
* 每个操作自己声明 **是否产生泄漏**（``leaky``）—— 泄漏类操作必须 out-of-fold，
  拿不到 folds 就拒绝执行，而不是静默算出一条泄漏特征；
* 执行产物是 :class:`FeaturePlan`（表达式 + 小表 join），**不在中间步骤物化大表**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from app.core.registry import Registry
from app.quality.semantics import ColumnSemantics

__all__ = [
    "FEATURE_OP_REGISTRY",
    "FeatureOp",
    "FeaturePlan",
    "apply_plan",
    "apply_plans",
    "feature_op",
]


@dataclass
class FeaturePlan:
    """一次特征工程操作的**计划**（尚未执行）。

    刻意分成「表达式」与「小表 join」两类产物：
    前者走 ``with_columns``（零拷贝列运算），后者只 join 聚合出来的小表，
    全过程不产生「行 × 列」的巨型中间 DataFrame。
    """

    op: str
    column: str
    new_columns: list[str] = field(default_factory=list)
    expressions: list[pl.Expr] = field(default_factory=list)
    #: (关联键, 小表, join 方式)
    lookups: list[tuple[str, pl.DataFrame, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "column": self.column,
            "new_columns": list(self.new_columns),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "lookup_count": len(self.lookups),
        }


@dataclass
class FeatureOp:
    """一个特征工程操作。

    ``applies(sem)`` 返回 ``None`` 表示适用，返回字符串表示不适用及原因
    （与 Quality 层的方法适配同一套写法）；``build(df, sem, params)`` 产出计划。
    """

    code: str
    label: str
    category: str
    applies: Callable[[ColumnSemantics], str | None]
    build: Callable[[pl.DataFrame, ColumnSemantics, dict[str, Any]], FeaturePlan]
    #: True ⇒ 该操作会用到目标列统计量，必须 out-of-fold，否则泄漏
    leaky: bool = False
    params_schema: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def check(self, sem: ColumnSemantics) -> str | None:
        return self.applies(sem)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.code,
            "label": self.label,
            "category": self.category,
            "leaky": self.leaky,
            "params": sorted(self.params_schema),
            "note": self.note,
        }


#: 特征工程操作注册表（进程级单例，协议同 ToolRegistry / ModelRegistry）
FEATURE_OP_REGISTRY: Registry[FeatureOp] = Registry(label="特征工程操作")


def feature_op(
    code: str,
    *,
    label: str,
    category: str,
    leaky: bool = False,
    params_schema: dict[str, Any] | None = None,
    note: str = "",
):
    """注册一个特征工程操作（装饰器用法与路由注册一致）。"""

    def _wrap(fn: Callable[[pl.DataFrame, ColumnSemantics, dict[str, Any]], FeaturePlan]) -> Callable[..., FeaturePlan]:
        def applies(sem: ColumnSemantics) -> str | None:
            return getattr(fn, "_applies", lambda _s: None)(sem)

        FEATURE_OP_REGISTRY.register(
            code,
            FeatureOp(
                code=code, label=label, category=category,
                applies=applies, build=fn, leaky=leaky,
                params_schema=dict(params_schema or {}), note=note,
            ),
        )
        return fn

    return _wrap


def when(predicate: Callable[[ColumnSemantics], str | None]):
    """给操作函数挂上适用条件（配合 :func:`feature_op` 使用）。"""

    def _wrap(fn):
        fn._applies = predicate
        return fn

    return _wrap


def apply_plan(df: pl.DataFrame, plan: FeaturePlan) -> pl.DataFrame:
    """执行一个计划。"""
    out = df
    if plan.expressions:
        out = out.with_columns(plan.expressions)
    for key, frame, how in plan.lookups:
        out = out.join(frame, on=key, how=how)
    return out


def apply_plans(df: pl.DataFrame, plans: list[FeaturePlan]) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """按序执行多个计划，返回（新表, 每个计划的执行回执）。"""
    out = df
    receipts: list[dict[str, Any]] = []
    for plan in plans:
        before = out.width
        out = apply_plan(out, plan)
        receipts.append(
            {
                **plan.to_dict(),
                "columns_before": before,
                "columns_after": out.width,
                "rows": out.height,
            }
        )
    return out, receipts
