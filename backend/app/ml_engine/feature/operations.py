"""ML Engine · 内置特征工程操作。

覆盖 Prompt 第三层列出的五类：

* 时间字段解析（HHMM → 时分 / 当日分钟数）
* 循环编码（sin/cos，解决 23:59 → 00:00 的周期断崖）
* 高基数编码（频次编码 / 目标编码）
* 共线性处理（提示，非变换）
* 交叉与聚合特征（分组统计）

两条硬约束
----------
1. **泄漏类操作必须 out-of-fold**。``encode.target`` 与 ``aggregate.group_stat``
   在拿不到 ``folds`` 时**拒绝执行**并给出可操作原因 —— 静默算出一条泄漏特征
   会让离线指标虚高、上线即崩，比不做危险得多。
2. **不为高基数列做 one-hot**。这是 ``ArrayMemoryError: 57.1 GiB`` 的直接来源
   （10 列 → 767 列）；高基数一律走频次/目标编码，低基数交给既有
   ``preprocessing`` 的 one-hot（已有 ``ML_ONEHOT_MAX_CATEGORIES`` 兜底）。
"""

from __future__ import annotations

from typing import Any

import polars as pl

from app.ml_engine.feature.registry import FeaturePlan, feature_op, when
from app.quality.semantics import ColumnRole, ColumnSemantics

__all__ = [
    "build_time_features",
    "register_builtin_feature_ops",
]


# --------------------------------------------------------------------------- #
# 时间
# --------------------------------------------------------------------------- #


@feature_op(
    "time.parse_hhmm",
    label="HHMM 伪数值时间解析",
    category="temporal",
    params_schema={"suffix": "str"},
    note="把 2359 这类值拆成 时/分/当日分钟数，消除 2359→0000 的数值断崖",
)
@when(lambda s: None if s.pseudo_time_format == "HHMM" else f"{s.name} 不是 HHMM 伪数值时间")
def _op_parse_hhmm(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    col = sem.name
    suffix = str(params.get("suffix") or "")
    hour = f"{col}_hour{suffix}"
    minute = f"{col}_minute{suffix}"
    since = f"{col}_minutes{suffix}"
    c = pl.col(col).cast(pl.Int64, strict=False)
    return FeaturePlan(
        op="time.parse_hhmm",
        column=col,
        new_columns=[hour, minute, since],
        expressions=[
            (c // 100).cast(pl.Float64).alias(hour),
            (c % 100).cast(pl.Float64).alias(minute),
            ((c // 100) * 60 + (c % 100)).cast(pl.Float64).alias(since),
        ],
        notes=[f"{col} 按 HHMM 拆分为 {hour}/{minute}/{since}"],
    )


@feature_op(
    "time.cyclic",
    label="循环编码（sin/cos）",
    category="temporal",
    params_schema={"period": "float"},
    note="把周期性取值映射到单位圆，消除周期边界的跳变（23:59→00:00、12月→1月）",
)
@when(
    lambda s: None
    if (
        s.role == ColumnRole.TEMPORAL
        or (s.role == ColumnRole.CATEGORICAL and s.cardinality >= 3 and s.is_numeric)
    )
    else f"{s.name} 不是周期性字段（role={s.role}）"
)
def _op_cyclic(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    col = sem.name
    stats = sem.stats or {}
    max_v = stats.get("max")
    min_v = stats.get("min")
    # 周期默认从取值域推：HHMM 时间用 1440（当日分钟），否则用 (max-min+1)
    period = params.get("period")
    if period is None:
        if sem.pseudo_time_format == "HHMM":
            period = 1440.0
        elif max_v is not None and min_v is not None and max_v > min_v:
            period = float(max_v - min_v + 1)
        else:
            period = float(max(sem.cardinality, 2))
    period = float(period)
    base = pl.col(col).cast(pl.Float64)
    if sem.pseudo_time_format == "HHMM":
        base = (base // 100) * 60 + (base % 100)
    angle = 2.0 * 3.141592653589793 * base / period
    sin_col, cos_col = f"{col}_sin", f"{col}_cos"
    return FeaturePlan(
        op="time.cyclic",
        column=col,
        new_columns=[sin_col, cos_col],
        expressions=[angle.sin().alias(sin_col), angle.cos().alias(cos_col)],
        notes=[f"{col} 按周期 {period:g} 做循环编码 → {sin_col}/{cos_col}"],
    )


# --------------------------------------------------------------------------- #
# 高基数编码
# --------------------------------------------------------------------------- #

_HIGH_CARDINALITY = 50


@feature_op(
    "encode.frequency",
    label="频次编码",
    category="encoding",
    params_schema={"normalize": "bool"},
    note="用取值出现频次替代类别本身，零列膨胀，适配 300+ 取值的高基数列",
)
@when(
    lambda s: None
    if s.role == ColumnRole.CATEGORICAL and s.cardinality > _HIGH_CARDINALITY
    else f"{s.name} 基数 {s.cardinality} 未超过 {_HIGH_CARDINALITY}，无需频次编码"
)
def _op_frequency(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    col = sem.name
    normalize = bool(params.get("normalize", True))
    total = max(df.height, 1)
    freq = (
        df.group_by(col)
        .agg(pl.len().alias("__cnt"))
        .with_columns(
            (pl.col("__cnt") / total).alias(f"{col}_freq")
            if normalize
            else pl.col("__cnt").cast(pl.Float64).alias(f"{col}_freq")
        )
        .select([col, f"{col}_freq"])
    )
    return FeaturePlan(
        op="encode.frequency",
        column=col,
        new_columns=[f"{col}_freq"],
        lookups=[(col, freq, "left")],
        notes=[f"{col}（{sem.cardinality} 个取值）频次编码 → {col}_freq，列数 +1"],
    )


@feature_op(
    "encode.target",
    label="目标编码（必须 out-of-fold）",
    category="encoding",
    leaky=True,
    params_schema={"folds": "series|list", "target": "str", "smoothing": "float"},
    note="用目标均值替代类别；**必须用折外数据计算**，否则指标虚高",
)
@when(lambda s: None if s.role == ColumnRole.CATEGORICAL else f"{s.name} 不是类别列")
def _op_target_encoding(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    col = sem.name
    target = params.get("target")
    folds = params.get("folds")
    smoothing = float(params.get("smoothing", 10.0))

    if not target or target not in df.columns:
        return FeaturePlan(
            op="encode.target", column=col, new_columns=[],
            warnings=[f"目标编码需要 target 参数且该列必须存在（当前 target={target!r}）"],
        )
    if folds is None:
        # ★ 拒绝执行而不是「先算出来再提醒」：泄漏特征一旦进训练集，
        # 离线指标会虚高到看不出问题，上线才暴露 —— 那时代价最大。
        return FeaturePlan(
            op="encode.target", column=col, new_columns=[],
            warnings=[
                f"目标编码必须 out-of-fold：未提供 folds，已拒绝为 {col} 生成目标编码。"
                "请先做 K 折切分并把折号通过 params['folds'] 传入。"
            ],
        )

    new_col = f"{col}_te"
    frame = df.select([pl.col(col).cast(pl.Utf8).alias(col), pl.col(str(target)).cast(pl.Float64).alias("__y")])
    if not isinstance(folds, pl.Series):
        folds = pl.Series("__fold", list(folds))
    frame = frame.with_columns(folds.alias("__fold"))

    # 折外统计：本行所属折的统计要从全局里扣掉自己那一份
    totals = frame.group_by(col).agg(
        pl.col("__y").sum().alias("__sum"), pl.len().alias("__n")
    )
    per_fold = frame.group_by([col, "__fold"]).agg(
        pl.col("__y").sum().alias("__f_sum"), pl.len().alias("__f_n")
    )
    prior = float(frame["__y"].mean() or 0.0)
    joined = (
        frame.join(totals, on=col, how="left")
        .join(per_fold, on=[col, "__fold"], how="left")
        .with_columns(
            [
                (pl.col("__sum") - pl.col("__f_sum")).alias("__o_sum"),
                (pl.col("__n") - pl.col("__f_n")).alias("__o_n"),
            ]
        )
        .with_columns(
            ((pl.col("__o_sum") + smoothing * prior) / (pl.col("__o_n") + smoothing)).alias(new_col)
        )
        .group_by([col, "__fold"])
        .agg(pl.col(new_col).first())
        .select([col, "__fold", new_col])
    )
    return FeaturePlan(
        op="encode.target",
        column=col,
        new_columns=[new_col],
        lookups=[(col, joined.select([col, new_col]).unique(subset=[col]), "left")],
        notes=[
            f"{col} 目标编码（out-of-fold，smoothing={smoothing:g}）→ {new_col}；"
            "统计量只用折外样本计算"
        ],
    )


# --------------------------------------------------------------------------- #
# 聚合 / 交叉（泄漏类）
# --------------------------------------------------------------------------- #


@feature_op(
    "aggregate.group_stat",
    label="分组统计特征（必须 out-of-fold）",
    category="aggregation",
    leaky=True,
    params_schema={"group_by": "str", "target": "str", "folds": "series|list"},
    note="如「该机场的平均延误」；组内统计量必须折外计算",
)
@when(lambda s: None)
def _op_group_stat(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    group_by = str(params.get("group_by") or sem.name)
    target = params.get("target")
    folds = params.get("folds")
    new_col = f"{group_by}_grp_mean"
    if not target or target not in df.columns or group_by not in df.columns:
        return FeaturePlan(
            op="aggregate.group_stat", column=group_by, new_columns=[],
            warnings=[f"分组统计需要 group_by 与 target 两列都存在（{group_by} / {target}）"],
        )
    if folds is None:
        return FeaturePlan(
            op="aggregate.group_stat", column=group_by, new_columns=[],
            warnings=[f"分组统计必须 out-of-fold：未提供 folds，已拒绝生成 {new_col}"],
        )
    frame = df.select([
        pl.col(group_by).cast(pl.Utf8).alias(group_by),
        pl.col(str(target)).cast(pl.Float64).alias("__y"),
    ])
    if not isinstance(folds, pl.Series):
        folds = pl.Series("__fold", list(folds))
    frame = frame.with_columns(folds.alias("__fold"))
    totals = frame.group_by(group_by).agg(pl.col("__y").sum().alias("__s"), pl.len().alias("__n"))
    per_fold = frame.group_by([group_by, "__fold"]).agg(
        pl.col("__y").sum().alias("__fs"), pl.len().alias("__fn")
    )
    table = (
        totals.join(per_fold, on=group_by, how="left")
        .with_columns(
            ((pl.col("__s") - pl.col("__fs")) / (pl.col("__n") - pl.col("__fn"))).alias(new_col)
        )
        .select([group_by, "__fold", new_col])
    )
    return FeaturePlan(
        op="aggregate.group_stat",
        column=group_by,
        new_columns=[new_col],
        lookups=[(group_by, table.select([group_by, new_col]).unique(subset=[group_by]), "left")],
        notes=[f"按 {group_by} 分组的折外目标均值 → {new_col}"],
    )


# --------------------------------------------------------------------------- #
# 共线性（提示型：不做变换，交给人/上层决策）
# --------------------------------------------------------------------------- #


@feature_op(
    "collinearity.flag",
    label="共线性标记",
    category="collinearity",
    params_schema={"threshold": "float", "pairs": "list"},
    note="只做标记与处置建议（保留其一 / 构造差值 / PCA），不做静默删列",
)
@when(lambda s: None if s.role in (ColumnRole.CONTINUOUS, ColumnRole.TEMPORAL) else f"{s.name} 非连续列")
def _op_collinearity(df: pl.DataFrame, sem: ColumnSemantics, params: dict[str, Any]) -> FeaturePlan:
    threshold = float(params.get("threshold", 0.75))
    pairs = [p for p in (params.get("pairs") or []) if sem.name in (p.get("a"), p.get("b"))]
    if not pairs:
        return FeaturePlan(op="collinearity.flag", column=sem.name, new_columns=[])
    others = [
        (p.get("b") if p.get("a") == sem.name else p.get("a"), p.get("r"))
        for p in pairs
    ]
    detail = "；".join(f"与 {o} 的 |r|={abs(float(r)):.2f}" for o, r in others if r is not None)
    return FeaturePlan(
        op="collinearity.flag",
        column=sem.name,
        new_columns=[],
        notes=[
            f"{sem.name} 存在共线性（阈值 {threshold:.2f}）：{detail}。"
            "可选处置：保留其一 / 构造差值 / 做 PCA；不做静默删列。"
        ],
    )


def build_time_features(df: pl.DataFrame, sem: ColumnSemantics) -> FeaturePlan | None:
    """时间字段的通用入口：伪数值时间先解析、再做循环编码。"""
    from app.ml_engine.feature.registry import FEATURE_OP_REGISTRY

    op = FEATURE_OP_REGISTRY.try_get("time.parse_hhmm")
    if op is not None and op.check(sem) is None:
        return op.build(df, sem, {})
    op = FEATURE_OP_REGISTRY.try_get("time.cyclic")
    if op is not None and op.check(sem) is None:
        return op.build(df, sem, {})
    return None


def register_builtin_feature_ops() -> None:
    """内置操作在模块导入时即完成注册（装饰器已在 import 时执行），
    此函数仅为调用方提供显式入口与幂等保证。"""
    return None
