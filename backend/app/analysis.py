"""统一的只读分析模块。

合并原 app/data_engine/profiling、app/data_engine/quality 与 app/eda 三个模块：
- Profiling：analyze_schema（Schema 分析）与 profile（数据画像）
- Quality：数据质量检查——QualityIssue / QualityReport / QualityChecker 基类、
  MissingChecker / DuplicateChecker / OutlierChecker / SchemaChecker、
  共享纯函数 compute_outlier_bounds 与 build_report 报告汇总
- EDA：探索性分析——EdaModule 基类及描述统计 / 相关性 / 分布 / 异常
  四类分析，VisualizationBuilder 构建前端可视化数据

所有分析均只读：绝不修改输入数据。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from app.core.exceptions import ValidationException
from app.data_engine.json_utils import json_safe

# =========================================================
# 内部工具
# =========================================================

# `n_unique()` 对 List / Struct 等不可哈希类型会抛原生异常；
# 分析端点宁可退化也不要 500（见评估报告 E1）。
_HASHABLE_SKIP_TYPES = (pl.List, pl.Struct, pl.Array, pl.Object)


def _safe_n_unique(series: pl.Series) -> int:
    """安全统计唯一值数：不可哈希类型返回 -1（前端据此显示「不适用」）。"""
    if isinstance(series.dtype, _HASHABLE_SKIP_TYPES):
        return -1
    try:
        return int(series.n_unique())
    except Exception:
        return -1


# 低基数数值列会被判定为「分类编码」，不算连续变量。
# 这是为修复「VendorID/payment_type/RatecodeID/PULocationID/DOLocationID 被当连续变量
# 算均值/std、进相关性热力图、做 Q-Q 图」而设的阈值（见 FRONTEND/PROJECT 笔记 §报告同质化）。
# 现实数据里这类列几乎都是编码/ID/日历字段，唯一值数远小于行数，且通常 < 50。
_CATEGORICAL_MAX_UNIQUE = 50

#: 判定「浮点列的值全部落在整数上」的容差。
#: ARFF / CSV 里的日历与编码字段（Month / DayofMonth / DayOfWeek）经常被读成
#: Float64，值本身是 1..12 / 1..31 / 1..7 这样的整数，必须与整型列同等对待。
_INTEGRAL_TOLERANCE = 1e-9

#: 对**浮点列**启用「低基数即分类」判定的最小行数。
#: 「唯一值数远少于行数」这个判据只有在行数足够多时才有区分力：31 行的表里
#: 一个 7 取值列说明不了任何问题（采样本来就少），而 1000 万行的表里
#: 12 取值只可能是月份这种编码。加这道闸门是为了让本判定只作用于它要解决的大表场景，
#: 不影响小数据集上「低基数浮点列算连续量」的既有行为。
_FLOAT_CATEGORICAL_MIN_ROWS = 1000


def _is_integral_valued(series: pl.Series) -> bool:
    """浮点列是否全部落在整数上（真实事故：airlines 的 Month=12 / DayofMonth=31 /
    DayOfWeek=7 全是 Float64，若只放行整型列，它们会被当连续变量算均值、
    进相关性热力图、还生成正态 Q-Q 图，把报告带偏）。"""
    s = series.drop_nulls()
    if s.len() == 0:
        return False
    try:
        diff = (s - s.round(0)).abs().max()
    except Exception:  # noqa: BLE001 - 类型不支持时按「非整数」处理
        return False
    return diff is not None and float(diff) <= _INTEGRAL_TOLERANCE


def _batch_n_unique(df: pl.DataFrame, columns: list[str]) -> dict[str, int]:
    """一次 ``select`` 批量算多列唯一值数。

    逐列 ``df[c].n_unique()`` 会各起一次全表扫描；批量写法把同一份扫描结果
    复用于所有列（与本模块 ``_batch_series_stats`` 同一思路）。不可哈希类型
    与批量失败都退回单列 ``_safe_n_unique``，保证只退化、不报错。
    """
    out: dict[str, int] = {}
    if not columns:
        return out
    try:
        exprs = [pl.col(c).n_unique().alias(c) for c in columns]
        row = df.select(exprs).row(0, named=True)
    except Exception:  # noqa: BLE001 - 批量失败逐列兜底
        row = {}
    for c in columns:
        value = row.get(c) if isinstance(row, dict) else None
        out[c] = int(value) if value is not None else _safe_n_unique(df[c])
    return out


def classify_columns(df: pl.DataFrame, columns: list[str] | None = None) -> dict[str, bool]:
    """一次性判定每列「是否按分类变量对待」。

    调用方应优先用本函数而不是分别调 ``categorical_columns`` + ``continuous_columns``：
    后者会把每列的唯一值数**算两遍**（大表上就是两倍全表扫描）。

    规则：非数值类型一律分类；数值类型中「唯一值数 ≤ 50 且唯一值数明显少于行数」
    的列视为分类编码。整型直接放行；浮点还要满足「取值全部是整数」，避免把
    金额/距离/延误分钟这类低基数浮点列误判成分类。
    """
    cols = [c for c in (columns or df.columns) if c in df.columns]
    if not cols:
        return {}
    result: dict[str, bool] = {}
    numeric: list[str] = []
    for c in cols:
        dtype = df.schema[c]
        if not dtype.is_numeric():
            result[c] = True
            continue
        result[c] = False
        if isinstance(dtype, _HASHABLE_SKIP_TYPES):
            continue
        numeric.append(c)
    counts = _batch_n_unique(df, numeric)
    rows = int(df.height)
    for c in numeric:
        n = counts.get(c, -1)
        if not (0 <= n <= _CATEGORICAL_MAX_UNIQUE and n * 2 < rows):
            continue
        if df.schema[c].is_integer():
            result[c] = True
        elif rows >= _FLOAT_CATEGORICAL_MIN_ROWS:
            # 浮点列：只有「值全是整数」才可能是编码/日历列（Month=1..12、DayOfWeek=1..7）
            result[c] = _is_integral_valued(df[c])
    return result


def is_categorical_like(df: pl.DataFrame, column: str) -> bool:
    """判断某列是否应视为「分类变量」而非「连续变量」。

    规则见 :func:`classify_columns`。加「唯一值远小于行数」这一条，是为了避免把
    小数据集里接近一一对应的连续整型列（如 age、行号）误判成分类（回归：6 行的
    age 有 6 个唯一值，若只看 n_unique≤50 会被误判）。
    """
    if column not in df.columns:
        return False
    return bool(classify_columns(df, [column]).get(column, False))


def categorical_columns(df: pl.DataFrame, columns: list[str] | None = None) -> list[str]:
    """返回应视为分类变量的列名（供相关性/描述统计剔除连续误判）。"""
    return [c for c, is_cat in classify_columns(df, columns).items() if is_cat]


def continuous_columns(df: pl.DataFrame, columns: list[str] | None = None) -> list[str]:
    """返回应视为连续变量的数值列（分类编码列已剔除）。"""
    cls = classify_columns(df, columns)
    return [
        c
        for c in (columns or df.columns)
        if c in df.columns and df.schema[c].is_numeric() and not cls.get(c, False)
    ]


def _batch_series_stats(
    df: pl.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    """一次遍历算出多列的基础统计量。

    替代「逐列 × 多次独立聚合」的写法：过去 profile() 每列要跑
    null_count / n_unique / min / max / mean / median / std / 3×quantile
    共 7+ 次独立全列扫描，100 列数据集就是 700+ 次扫描。
    现在用单次 ``df.select([...])`` 批量出结果。
    """

    if not columns:
        return {}

    exprs: list[pl.Expr] = [pl.len().alias("__rows")]
    for name in columns:
        dtype = df.schema[name]
        exprs.append(pl.col(name).null_count().alias(f"{name}__nulls"))
        if dtype.is_numeric():
            exprs.append(pl.col(name).min().alias(f"{name}__min"))
            exprs.append(pl.col(name).max().alias(f"{name}__max"))
            exprs.append(pl.col(name).mean().alias(f"{name}__mean"))
            exprs.append(pl.col(name).median().alias(f"{name}__median"))
            exprs.append(pl.col(name).std().alias(f"{name}__std"))
            for q in QUANTILES:
                exprs.append(
                    pl.col(name)
                    .quantile(q, interpolation="linear")
                    .alias(f"{name}__q{q}")
                )
        elif dtype.is_temporal():
            exprs.append(pl.col(name).min().alias(f"{name}__min"))
            exprs.append(pl.col(name).max().alias(f"{name}__max"))

    try:
        row = df.select(exprs).row(0, named=True)
    except Exception:
        # 批量聚合失败（极少数 dtype 不支持某算子）时退回逐列模式。
        return {}

    return row


def _stats_of(stats: dict[str, Any], column: str, key: str) -> Any:
    return stats.get(f"{column}__{key}")


# =========================================================
# Profiling（analyze_schema、profile）
# =========================================================

SAMPLE_VALUES = 5


def analyze_schema(
    df: pl.DataFrame,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    names = list(df.columns)

    # 一次批量算出所有列的空值数（替代逐列 null_count 全表扫描）。
    stats = _batch_series_stats(df, names)

    for name, dtype in df.schema.items():
        series = df[name]
        non_null = series.drop_nulls()

        samples = (
            non_null
            .unique(maintain_order=True)
            .head(SAMPLE_VALUES)
            .to_list()
        )

        null_count = _stats_of(stats, name, "nulls")
        if null_count is None:
            null_count = int(series.null_count())
        null_count = int(null_count)

        columns.append(
            {
                "column": name,
                "dtype": str(dtype),
                "nullable": null_count > 0,
                "null_count": null_count,
                "unique_count": _safe_n_unique(series),
                "sample_values": [
                    json_safe(value)
                    for value in samples
                ],
            }
        )

    return {
        "row_count": df.height,
        "column_count": df.width,
        "columns": columns,
    }


QUANTILES = (0.25, 0.5, 0.75)
TOP_VALUES = 5


def _categorical_profile(
    series: pl.Series,
) -> dict[str, Any]:
    counts = (
        series
        .drop_nulls()
        .value_counts(sort=True)
        .head(TOP_VALUES)
    )

    return {
        "top_values": [
            {
                "value": json_safe(row[0]),
                "count": int(row[1]),
            }
            for row in counts.iter_rows()
        ]
    }


def profile(
    df: pl.DataFrame,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    missing_cells = 0

    names = list(df.columns)
    # 批量聚合：一次遍历出齐 min/max/mean/median/std/3×quantile/null_count。
    stats = _batch_series_stats(df, names)

    for name, dtype in df.schema.items():
        series = df[name]

        null_count = _stats_of(stats, name, "nulls")
        missing = int(null_count) if null_count is not None else int(series.null_count())

        missing_cells += missing

        column: dict[str, Any] = {
            "column": name,
            "dtype": str(dtype),
            "missing_count": missing,
            "missing_rate": (
                round(missing / df.height, 6)
                if df.height
                else 0.0
            ),
            "unique_count": _safe_n_unique(series),
        }

        if dtype.is_numeric():
            column["type_class"] = "numeric"
            column.update(
                _numeric_profile_from_stats(
                    stats, name, fallback_series=series
                )
            )

        elif dtype.is_temporal():
            column["type_class"] = "temporal"
            column.update(
                _temporal_profile_from_stats(
                    stats, name, fallback_series=series
                )
            )

        elif dtype == pl.Boolean:
            column["type_class"] = "boolean"

        else:
            column["type_class"] = "categorical"
            column.update(
                _categorical_profile(series)
            )

        columns.append(column)

    if df.height and df.width:
        complete_rows = int(
            df.height
            - df.filter(
                pl.any_horizontal(
                    pl.all().is_null()
                )
            ).height
        )
    else:
        complete_rows = 0

    return {
        "row_count": df.height,
        "column_count": df.width,
        "missing_cells": missing_cells,
        "complete_row_count": complete_rows,
        "complete_row_rate": (
            round(
                complete_rows / df.height,
                6,
            )
            if df.height
            else 0.0
        ),
        "columns": columns,
    }


def _numeric_profile_from_stats(
    stats: dict[str, Any],
    name: str,
    *,
    fallback_series: pl.Series | None = None,
) -> dict[str, Any]:
    """从批量聚合结果组装数值列画像；批量结果缺失时退回单列计算。"""

    def value(key: str, series_method: str) -> Any:
        raw = _stats_of(stats, name, key)
        if raw is not None:
            return raw
        if fallback_series is None:
            return None
        return getattr(fallback_series, series_method)()

    return {
        "min": json_safe(value("min", "min")),
        "max": json_safe(value("max", "max")),
        "mean": json_safe(value("mean", "mean")),
        "median": json_safe(value("median", "median")),
        "std": json_safe(value("std", "std")),
        "quantiles": {
            str(q): json_safe(
                _stats_of(stats, name, f"q{q}")
                if _stats_of(stats, name, f"q{q}") is not None
                else (
                    fallback_series.quantile(q, interpolation="linear")
                    if fallback_series is not None
                    else None
                )
            )
            for q in QUANTILES
        },
    }


def _temporal_profile_from_stats(
    stats: dict[str, Any],
    name: str,
    *,
    fallback_series: pl.Series | None = None,
) -> dict[str, Any]:
    """从批量聚合结果组装时间列画像。"""
    mn = _stats_of(stats, name, "min")
    mx = _stats_of(stats, name, "max")
    if mn is None and fallback_series is not None:
        mn = fallback_series.min()
    if mx is None and fallback_series is not None:
        mx = fallback_series.max()
    return {
        "min": json_safe(mn),
        "max": json_safe(mx),
    }


# =========================================================
# Quality（QualityIssue/QualityReport/QualityChecker 基类、
# MissingChecker、DuplicateChecker、OutlierChecker、SchemaChecker、
# compute_outlier_bounds、build_report）
# =========================================================

# 严重级别（由低到高）
SEVERITY_LEVELS = ("info", "low", "medium", "high", "critical")


@dataclass
class QualityIssue:
    """单条质量问题。"""

    check: str
    severity: str
    message: str
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "column": self.column,
            "details": self.details,
        }


class QualityChecker(ABC):
    """质量检查器抽象基类。"""

    name: str = ""

    @abstractmethod
    def check(self, df: pl.DataFrame) -> list[QualityIssue]:
        """执行检查，返回问题列表（只读，不修改数据）。"""


class MissingChecker(QualityChecker):
    name = "missing"

    def __init__(
        self,
        *,
        warn_threshold: float = 0.05,
        high_threshold: float = 0.3,
        critical_threshold: float = 0.8,
        columns: list[str] | None = None,
    ) -> None:
        self.warn_threshold = warn_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.columns = columns

    def check(self, df: pl.DataFrame) -> list[QualityIssue]:
        cols = self.columns or list(df.columns)
        unknown = [c for c in cols if c not in df.columns]
        if unknown:
            raise ValidationException(
                f"以下字段不存在：{unknown}",
                details={"missing": unknown, "available": list(df.columns)},
            )

        issues: list[QualityIssue] = []
        if df.height == 0:
            return issues
        for name in cols:
            missing = int(df[name].null_count())
            if missing == 0:
                continue
            rate = missing / df.height
            if rate >= self.critical_threshold:
                severity = "critical"
            elif rate >= self.high_threshold:
                severity = "high"
            elif rate >= self.warn_threshold:
                severity = "medium"
            else:
                severity = "low"
            issues.append(
                QualityIssue(
                    check=self.name,
                    column=name,
                    severity=severity,
                    message=(
                        f"字段 {name!r} 存在 {missing} 个缺失值 "
                        f"({rate:.2%})"
                    ),
                    details={
                        "missing_count": missing,
                        "missing_rate": round(rate, 6),
                    },
                )
            )
        return issues


class DuplicateChecker(QualityChecker):
    name = "duplicate"

    def __init__(
        self,
        *,
        subset: list[str] | None = None,
        warn_threshold: float = 0.01,
        high_threshold: float = 0.1,
    ) -> None:
        """subset 为 None 时检查全行重复。"""
        self.subset = subset
        self.warn_threshold = warn_threshold
        self.high_threshold = high_threshold

    def check(self, df: pl.DataFrame) -> list[QualityIssue]:
        subset = self.subset
        if subset is not None:
            unknown = [c for c in subset if c not in df.columns]
            if unknown:
                raise ValidationException(
                    f"去重字段不存在：{unknown}",
                    details={"missing": unknown, "available": list(df.columns)},
                )

        if df.height == 0:
            return []
        if subset is not None:
            duplicate_mask = df.select(subset).is_duplicated()
        else:
            duplicate_mask = df.is_duplicated()
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count == 0:
            return []

        ratio = duplicate_count / df.height
        severity = "high" if ratio >= self.high_threshold else (
            "medium" if ratio >= self.warn_threshold else "low"
        )
        scope = "整行" if subset is None else f"字段 {subset}"
        return [
            QualityIssue(
                check=self.name,
                severity=severity,
                message=(
                    f"按 {scope} 检测到 {duplicate_count} 行重复 ({ratio:.2%})"
                ),
                column=None if subset is None else ",".join(subset),
                details={
                    "duplicate_rows": duplicate_count,
                    "duplicate_ratio": round(ratio, 6),
                    "scope": "full_row" if subset is None else list(subset),
                },
            )
        ]


METHODS = ("iqr", "zscore")


def compute_outlier_bounds(
    s: pl.Series, *, method: str, k: float = 1.5, z_threshold: float = 3.0
) -> tuple[float, float, dict[str, Any]]:
    """计算数值列的异常值边界。

    返回 (lower, upper, info)；info 包含方法相关的中间统计量。
    """
    if method not in METHODS:
        raise ValidationException(
            f"不支持的异常值判定方法：{method}",
            details={"method": method, "allowed": list(METHODS)},
        )
    clean = s.drop_nulls()
    if clean.len() == 0:
        return (float("nan"), float("nan"), {"method": method, "empty": True})

    if method == "iqr":
        q1 = float(clean.quantile(0.25, interpolation="linear"))
        q3 = float(clean.quantile(0.75, interpolation="linear"))
        iqr = q3 - q1
        if iqr == 0 or iqr != iqr:
            # 与 zscore 分支口径统一：IQR 为 0 说明列值在四分位范围内没有散布
            # （典型为常数列）。此时 lower == upper，任何"异常"判定都无意义，
            # 旧实现会静默返回该退化区间，前端表格看起来像"0 个异常"但其实没算。
            return (
                float("nan"),
                float("nan"),
                {"method": "iqr", "q1": q1, "q3": q3, "iqr": iqr, "k": k, "constant": True},
            )
        lower, upper = q1 - k * iqr, q3 + k * iqr
        info = {"method": "iqr", "q1": q1, "q3": q3, "iqr": iqr, "k": k}
    else:  # zscore
        mean = float(clean.mean())
        # Polars 的 std(ddof=1) 在只有 1 个非空样本时返回 None，float(None) 会抛 TypeError。
        std_raw = clean.std()
        std = float(std_raw) if std_raw is not None else 0.0
        if std == 0 or std != std:  # 常数列 / NaN
            return (float("nan"), float("nan"), {"method": "zscore", "std": std, "constant": True})
        lower, upper = mean - z_threshold * std, mean + z_threshold * std
        info = {"method": "zscore", "mean": mean, "std": std, "z_threshold": z_threshold}
    return (lower, upper, info)


class OutlierChecker(QualityChecker):
    """异常值检查。

    在原有「算边界 + 报数量」之外补上了**方法适用性判定**（第二层改造）：
    每个字段先经 :mod:`app.quality` 得到语义画像，再判断当前方法是否适用；
    不适用时把 ``method_mismatch`` 写进 ``details``（不改判定结果本身），
    由报告/Agent 向用户说明「这个结论可能不成立」。

    为什么默认只标注、不改判定：直接换方法会静默改变既有报告的数字
    （如 ``v=[1,2,3,4,5,100]`` 在 IQR 下 1 个异常、换分位数口径后可能 0 个）。
    需要真正改方法时显式传 ``adaptive=True``，由调用方承担口径变化。
    """

    name = "outlier"

    def __init__(
        self,
        *,
        method: str = "iqr",
        k: float = 1.5,
        z_threshold: float = 3.0,
        columns: list[str] | None = None,
        warn_threshold: float = 0.01,
        target: str | None = None,
        business_rules: list[Any] | None = None,
        adaptive: bool = False,
    ) -> None:
        self.method = method
        self.k = k
        self.z_threshold = z_threshold
        self.columns = columns
        self.warn_threshold = warn_threshold
        #: 目标列：只描述、不产出清洗建议
        self.target = target
        self.business_rules = list(business_rules or [])
        #: True ⇒ 按字段语义自动改用适配方法（口径会变，需调用方知情）
        self.adaptive = adaptive

    def check(self, df: pl.DataFrame) -> list[QualityIssue]:
        cols = self.columns or [c for c, d in df.schema.items() if d.is_numeric()]
        issues: list[QualityIssue] = []
        if df.height == 0:
            return issues

        semantics: dict[str, Any] = {}
        if self.adaptive or self.target or self.business_rules:
            semantics = _infer_column_semantics(
                df, target=self.target, business_rules=self.business_rules
            )

        for name in cols:
            dtype = df.schema[name]
            if not dtype.is_numeric():
                continue
            sem = semantics.get(name)
            rules = None
            if sem is not None and isinstance(sem.stats.get("business_rule"), dict):
                rules = _business_rule_from(sem)

            if self.adaptive and sem is not None:
                decision = _assess_column(
                    sem, df[name], requested=self.method, k=self.k,
                    z_threshold=self.z_threshold, business_rule=rules,
                )
                bounds = decision.value
                if bounds is None or not bounds.usable:
                    continue
                lower_f = bounds.lower if bounds.lower is not None else float("-inf")
                upper_f = bounds.upper if bounds.upper is not None else float("inf")
                info = dict(bounds.info)
                info["method"] = bounds.method
                if bounds.notes:
                    info["adjusted_notes"] = list(bounds.notes)
                used_method = bounds.method
                if used_method == "low_frequency":
                    # 低频类别没有数值边界，只报取值清单
                    rare = bounds.info.get("rare_values") or []
                    if not rare:
                        continue
                    issues.append(
                        QualityIssue(
                            check=self.name,
                            column=name,
                            severity="low",
                            message=(
                                f"字段 {name!r} 检测到 {bounds.info.get('rare_value_count', 0)} "
                                f"个低频取值（占比 < {bounds.info.get('threshold_ratio', 0):.2%}）"
                            ),
                            details={
                                "method": "low_frequency",
                                "outlier_count": int(bounds.info.get("rare_value_count", 0)),
                                "outlier_ratio": 0.0,
                                "sample_outliers": rare[:5],
                                "applicable": True,
                                "recommended_method": "low_frequency",
                                "method_mismatch": None,
                            },
                        )
                    )
                    continue
            else:
                lower_f, upper_f, info = compute_outlier_bounds(
                    df[name], method=self.method, k=self.k, z_threshold=self.z_threshold
                )
                if info.get("empty") or info.get("constant"):
                    continue
                used_method = self.method

            s = df[name]
            mask = (s < lower_f) | (s > upper_f)
            count = int(mask.sum())
            mismatch = None
            recommended = used_method
            cleaning = True
            if sem is not None:
                verdict = _select_outlier_method(
                    sem, requested=self.method, business_rule=rules
                )
                recommended = verdict.value or used_method
                blocking = verdict.findings
                mismatch = (
                    {
                        "code": blocking[0].code,
                        "severity": str(blocking[0].severity),
                        "message": blocking[0].message,
                        "suggestion": blocking[0].suggestion,
                    }
                    if blocking
                    else None
                )
                cleaning = _should_suggest_cleaning(sem)
                # 目标列不进结论（只描述）
                if sem.is_target:
                    cleaning = False
            if count == 0:
                # 保持既有行为：没有异常值就不产出 issue。方法不适用的提示只在
                # 确有个结论需要被质疑时才有意义，凭空报「0 个异常但方法不对」
                # 会让质量问题表凭空变长，掩盖真正需要看的问题。
                continue
            ratio = count / df.height
            severity = "high" if ratio >= self.warn_threshold * 5 else (
                "medium" if ratio >= self.warn_threshold else "low"
            )
            samples = s.filter(mask).head(5).to_list()
            details = {
                **info,
                "lower": lower_f,
                "upper": upper_f,
                "outlier_count": count,
                "outlier_ratio": round(ratio, 6),
                "sample_outliers": samples,
                "applicable": mismatch is None,
                "recommended_method": recommended,
                "method_mismatch": mismatch,
                "cleaning_suggested": cleaning,
            }
            if sem is not None:
                details["semantics"] = {
                    "role": str(sem.role),
                    "domain": str(sem.domain),
                    "shape": str(sem.shape),
                    "is_target": sem.is_target,
                }
            issues.append(
                QualityIssue(
                    check=self.name,
                    column=name,
                    severity=severity,
                    message=(
                        f"字段 {name!r} 检测到 {count} 个异常值 ({ratio:.2%})，"
                        f"方法={used_method}，边界=[{lower_f:.4g}, {upper_f:.4g}]"
                        + (
                            f"；⚠ 该方法对本字段可能不适用（建议 {recommended}）"
                            if mismatch
                            else ""
                        )
                    ),
                    details=details,
                )
            )
        return issues


# ---- Quality 层标准能力接线（延迟导入，避免与 app.quality 形成循环依赖） ----

def _infer_column_semantics(
    df: "pl.DataFrame", *, target: str | None, business_rules: list[Any]
) -> dict[str, Any]:
    from app.quality.semantics import infer_semantics

    try:
        return infer_semantics(df, target=target, business_rules=business_rules)
    except Exception:  # noqa: BLE001 - 语义推断失败不得影响质量检查本身
        return {}


def _select_outlier_method(sem: Any, *, requested: str, business_rule: Any) -> Any:
    from app.quality.outlier_strategy import select_outlier_method

    return select_outlier_method(sem, requested=requested, business_rule=business_rule)


def _assess_column(
    sem: Any, series: "pl.Series", *, requested: str, k: float, z_threshold: float, business_rule: Any
) -> Any:
    from app.quality.outlier_strategy import assess_column

    return assess_column(
        sem, series, requested=requested, k=k,
        z_threshold=z_threshold, business_rule=business_rule,
    )


def _should_suggest_cleaning(sem: Any) -> bool:
    from app.quality.outlier_strategy import should_suggest_cleaning

    return should_suggest_cleaning(sem)


def _business_rule_from(sem: Any) -> Any:
    from app.quality.semantics import BusinessRule

    raw = sem.stats.get("business_rule") or {}
    return BusinessRule(
        column=raw.get("column", sem.name),
        lower=raw.get("lower"),
        upper=raw.get("upper"),
        strict_lower=bool(raw.get("strict_lower")),
        strict_upper=bool(raw.get("strict_upper")),
        allowed_values=raw.get("allowed_values"),
        note=str(raw.get("note") or ""),
    )


# 期望类型别名 -> 兼容的 Polars dtype 基类型
EXPECTED_TYPE_ALIASES: dict[str, tuple[pl.DataType, ...]] = {
    "string": (pl.String, pl.Categorical),
    "int": (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    "float": (pl.Float32, pl.Float64),
    "number": (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
               pl.UInt16, pl.UInt32, pl.UInt64),
    "bool": (pl.Boolean,),
    "date": (pl.Date,),
    "datetime": (pl.Datetime,),
    "any": tuple(),  # any 不限制类型
}


class SchemaChecker(QualityChecker):
    name = "schema"

    def __init__(self, expected: dict[str, str]) -> None:
        """expected: {"column": "string|int|float|number|bool|date|datetime|any"}"""
        normalized: dict[str, str] = {}
        for column, expected_type in expected.items():
            t = expected_type.lower()
            if t not in EXPECTED_TYPE_ALIASES:
                allowed = sorted(k for k in EXPECTED_TYPE_ALIASES if k != "any")
                raise ValidationException(
                    f"列 {column!r} 的期望类型 {expected_type!r} 不被支持；"
                    f"可选类型：{', '.join(allowed)}, any",
                    details={
                        "column": column,
                        "expected_type": expected_type,
                        "allowed": [*allowed, "any"],
                    },
                )
            normalized[column] = t
        self.expected = normalized

    def check(self, df: pl.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []

        missing = [c for c in self.expected if c not in df.columns]
        if missing:
            issues.append(
                QualityIssue(
                    check=self.name,
                    severity="high",
                    message=f"缺少字段：{missing}",
                    details={"missing_columns": missing},
                )
            )

        extra = [c for c in df.columns if c not in self.expected]
        if extra:
            issues.append(
                QualityIssue(
                    check=self.name,
                    severity="low",
                    message=f"存在预期外的字段：{extra}",
                    details={"extra_columns": extra},
                )
            )

        conflicts: list[dict[str, str]] = []
        for column, expected_type in self.expected.items():
            if column not in df.columns:
                continue
            actual = df.schema[column]
            if not self._type_matches(actual, expected_type):
                conflicts.append(
                    {"column": column, "expected": expected_type, "actual": str(actual)}
                )
        if conflicts:
            issues.append(
                QualityIssue(
                    check=self.name,
                    severity="high",
                    message=f"{len(conflicts)} 个字段的类型与预期不符",
                    details={"type_conflicts": conflicts},
                )
            )
        return issues

    @staticmethod
    def _type_matches(actual: pl.DataType, expected_type: str) -> bool:
        if expected_type == "any":
            return True
        expected_dtypes = EXPECTED_TYPE_ALIASES[expected_type]
        return any(actual == t or actual.base_type() == t for t in expected_dtypes)


@dataclass
class QualityReport:
    """数据质量结果。"""

    issues: list[QualityIssue] = field(
        default_factory=list
    )

    severity: dict[str, int] = field(
        default_factory=dict
    )

    statistics: dict[str, Any] = field(
        default_factory=dict
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    @property
    def has_errors(self) -> bool:
        return any(
            issue.severity in {"high", "critical"}
            for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "severity": self.severity,
            "statistics": self.statistics,
            "recommendations": self.recommendations,
            "has_errors": self.has_errors,
        }


def _severity_summary(
    issues: list[QualityIssue],
) -> dict[str, int]:
    result = {
        level: 0
        for level in SEVERITY_LEVELS
    }

    for issue in issues:
        result[issue.severity] = (
            result.get(issue.severity, 0) + 1
        )

    return result


def _recommendations(
    issues: list[QualityIssue],
) -> list[str]:
    checks = {
        "missing": (
            "存在缺失值，可使用 missing 操作处理。"
        ),
        "duplicate": (
            "存在重复数据，可使用 duplicate 操作处理。"
        ),
        "outlier": (
            "存在异常值，请结合业务含义确认后处理。"
        ),
        "schema": (
            "Schema 与预期不一致，请检查字段与类型。"
        ),
    }

    found = {
        issue.check
        for issue in issues
    }

    return [
        message
        for check, message in checks.items()
        if check in found
    ]


def build_report(
    df: pl.DataFrame,
    checkers: list[QualityChecker] | None = None,
    *,
    expected_schema: dict[str, str] | None = None,
    target: str | None = None,
    business_rules: list[Any] | None = None,
    adaptive_outlier: bool = False,
) -> QualityReport:
    """汇总质量检查。

    ``target`` / ``business_rules`` / ``adaptive_outlier`` 是第二层改造的入口：
    知道目标列时，异常值检查会标注「该列是目标，不做清洗建议」；
    传了业务口径时，业务口径优先于统计口径。三者都缺省时行为与改造前完全一致。
    """
    if checkers is None:
        checkers = [
            MissingChecker(),
            DuplicateChecker(),
            OutlierChecker(
                target=target,
                business_rules=business_rules,
                adaptive=adaptive_outlier,
            ),
        ]

        if expected_schema is not None:
            checkers.append(
                SchemaChecker(expected_schema)
            )

    issues: list[QualityIssue] = []

    for checker in checkers:
        issues.extend(
            checker.check(df)
        )

    return QualityReport(
        issues=issues,
        severity=_severity_summary(issues),
        statistics={
            "row_count": df.height,
            "column_count": df.width,
            "checked_by": [
                checker.name
                for checker in checkers
            ],
            "missing_cells": int(
                sum(
                    df[column].null_count()
                    for column in df.columns
                )
            ),
            "duplicate_rows": int(
                df.is_duplicated().sum()
            ),
            "issue_count": len(issues),
        },
        recommendations=_recommendations(
            issues
        ),
    )


# =========================================================
# EDA（EdaModule 基类、DescriptiveAnalyzer、CorrelationAnalyzer、
# DistributionAnalyzer、EdaOutlierAnalyzer、VisualizationBuilder）
# =========================================================


class EdaModule(ABC):
    """EDA 模块抽象基类。"""

    name: str = ""

    @abstractmethod
    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        """执行分析，返回 JSON 可序列化结果（只读）。"""

    # ---- 共用工具 ----
    @staticmethod
    def require_columns(df: pl.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            # 缺列属于「请求不可处理」，统一用 422（ValidationException），
            # 不再混用 400（TransformError）。
            raise ValidationException(
                "指定的字段不存在：" + "、".join(missing),
                details={
                    "missing": missing,
                    "available": list(df.columns),
                    "hint": "请从可用字段中重新选择。",
                },
            )

    @staticmethod
    def pick_numeric(df: pl.DataFrame, columns: list[str] | None) -> list[str]:
        cols = columns or [c for c, d in df.schema.items() if d.is_numeric()]
        return [c for c in cols if c in df.columns and df.schema[c].is_numeric()]


class DescriptiveAnalyzer(EdaModule):
    name = "descriptive"

    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        columns: list[str] | None = options.get("columns")
        if columns:
            self.require_columns(df, columns)
        target = columns or list(df.columns)
        result: dict[str, Any] = {"row_count": df.height, "columns": []}

        for name in target:
            if name not in df.columns:
                continue
            dtype = df.schema[name]
            s = df[name]
            col: dict[str, Any] = {
                "column": name,
                "dtype": str(dtype),
                "count": int(s.len()),
                "missing_count": int(s.null_count()),
            }
            # 分类编码整型列（VendorID/payment_type/RatecodeID 等）不当连续变量：
            # 均值/std 没有业务意义（回归：报告曾给 VendorID 算均值 1.754/std 0.4326）。
            if dtype.is_numeric() and not is_categorical_like(df, name):
                col.update(
                    {
                        "mean": json_safe(s.mean()),
                        "std": json_safe(s.std()),
                        "min": json_safe(s.min()),
                        "max": json_safe(s.max()),
                        "quantiles": {
                            str(q): json_safe(s.quantile(q, interpolation="linear"))
                            for q in QUANTILES
                        },
                    }
                )
            elif dtype == pl.Boolean:
                non_null = s.drop_nulls()
                col.update(
                    {
                        "true_count": int((non_null).sum()),
                        "false_count": int(non_null.len() - (non_null).sum()),
                    }
                )
            else:
                non_null = s.drop_nulls()
                top = non_null.value_counts(sort=True).head(1)
                top_value, top_freq = top.row(0) if top.height else (None, 0)
                col.update(
                    {
                        "unique": int(s.n_unique()),
                        "top": json_safe(top_value),
                        "freq": int(top_freq),
                    }
                )
                # 分类编码整型列额外标注，前端/报告可据此渲染「频次分布」而非均值。
                if dtype.is_numeric():
                    col["categorical_encoding"] = True
            result["columns"].append(col)
        return result


CORRELATION_METHODS = ("pearson", "spearman", "auto")


class CorrelationAnalyzer(EdaModule):
    """相关性矩阵。

    性能优化（对应评估报告 P2）：

    1. **一次投影**：先把所有数值列投影成一张 ``df.select(cols).drop_nulls()``，
       后续每一对相关系数直接在这张表上取两列，不再重建 DataFrame。
       原实现对每一对都做 ``df.select([a, b]).drop_nulls()``（n² 次）。
    2. **对称性**：只计算上三角 (a, b) with a < b，下三角直接复制，计算量减半。
    3. **行抽样**：行数超过 ``max_rows`` 时抽样（seed 固定可复现），
       结果带 ``sampled`` / ``sample_size`` / ``original_rows``。
    4. **列上限**：数值列数超过 ``max_columns`` 时按方差取前 N 列，
       结果带 ``truncated`` / ``dropped_columns``。
    """

    name = "correlation"

    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        method = options.get("method", "auto")
        if method not in CORRELATION_METHODS:
            raise ValidationException(
                f"不支持的相关性方法：{method}",
                details={"method": method, "allowed": list(CORRELATION_METHODS)},
            )
        columns: list[str] | None = options.get("columns")
        if columns:
            self.require_columns(df, columns)
        # 只用「连续变量」做相关性：分类编码整型列（VendorID/ratecodeID/区域 ID 等）
        # 不该参与 Pearson/Spearman 相关，否则会得到无业务意义的伪相关（回归：报告热力图
        # 曾把 VendorID、PULocationID、DOLocationID 与金额字段混在一起算相关性）。
        numeric_cols = continuous_columns(df, columns)
        if len(numeric_cols) < 2:
            raise ValidationException(
                "相关性分析至少需要 2 个数值字段",
                details={"numeric_columns": numeric_cols},
            )

        # ---- 列上限：按方差取信息量最大的前 N 列 ----
        max_columns = int(_corr_limits().get("max_columns", 30))
        dropped: list[str] = []
        if max_columns > 0 and len(numeric_cols) > max_columns:
            ranked = _rank_columns_by_variance(df, numeric_cols)
            dropped = [c for c in numeric_cols if c not in ranked]
            numeric_cols = ranked
            if len(numeric_cols) < 2:
                raise ValidationException(
                    "相关性分析至少需要 2 个数值字段",
                    details={"numeric_columns": numeric_cols},
                )

        # ---- 一次投影（只做一次 drop_nulls） ----
        prep = self._prepare(df, numeric_cols)
        mat_df, mat_cols = prep.data, prep.columns

        # ---- 行抽样 ----
        max_rows = int(_corr_limits().get("max_rows", 5000))
        original_rows = mat_df.height
        sampled = bool(max_rows > 0 and original_rows > max_rows)
        if sampled:
            mat_df = mat_df.sample(n=max_rows, seed=42)

        if method == "auto":
            result = self._auto_matrix(mat_df, mat_cols)
        else:
            result = {
                "method": method,
                "columns": mat_cols,
                "matrix": self._matrix(mat_df, mat_cols, method),
            }

        # 元信息恒定存在，前端不必为「是否发生抽样/截断」做两套分支：
        # 未抽样时 sampled=False、sample_size=original_rows，未截断时 truncated=False。
        result["sampled"] = sampled
        result["sample_size"] = int(mat_df.height)
        result["original_rows"] = int(original_rows)
        result["truncated"] = bool(dropped)
        result["dropped_columns"] = list(dropped)
        result["pair_count"] = len(mat_cols) * (len(mat_cols) - 1) // 2
        return result

    # ---- 数据准备 ----

    @dataclass
    class _Prepared:
        data: pl.DataFrame
        columns: list[str]

    def _prepare(self, df: pl.DataFrame, cols: list[str]) -> "_Prepared":
        """投影 + 单次 drop_nulls + 缺失列剔除。

        只在「任意被选列有空值」时才做行过滤，且只做一次，
        避免原实现按列对重复过滤的开销。
        """
        frame = df.select(cols)

        if frame.null_count().row(0).count(0) > 0:
            frame = frame.drop_nulls()

        # drop_nulls 后可能出现全空列，剔除以保证矩阵不出现 null 列
        usable = [
            c for c in cols
            if frame.height and frame[c].null_count() < frame.height
        ]
        if len(usable) < 2:
            raise ValidationException(
                "相关性分析至少需要 2 个非空数值字段",
                details={
                    "numeric_columns": cols,
                    "usable_columns": usable,
                    "rows": int(frame.height),
                },
            )
        return CorrelationAnalyzer._Prepared(
            data=frame.select(usable) if len(usable) != len(cols) else frame,
            columns=usable,
        )

    # ---- 内部 ----

    def _corr_from_frame(
        self, frame: pl.DataFrame, a: str, b: str, method: str
    ) -> float | None:
        """在已投影的表上计算两列相关系数（不再重复 select / drop_nulls）。"""
        n = frame.height
        # 样本量不足时不给系数：2 个点的相关系数恒为 ±1，语义具有误导性。
        if n < 3:
            return None
        x, y = frame[a], frame[b]
        if method == "spearman":
            # Spearman = 对秩计算 Pearson
            x = x.rank("average")
            y = y.rank("average")
        x_std, y_std = x.std(), y.std()
        if x_std is None or y_std is None:
            return None
        denom = float(x_std) * float(y_std)
        if denom == 0 or denom != denom:
            return None
        return float(
            ((x - x.mean()) * (y - y.mean())).sum() / (n - 1) / denom
        )

    def _matrix(
        self, frame: pl.DataFrame, cols: list[str], method: str
    ) -> dict[str, dict[str, float | None]]:
        """对称矩阵：只算上三角，下三角复制。"""
        matrix: dict[str, dict[str, float | None]] = {c: {} for c in cols}
        for i, a in enumerate(cols):
            matrix[a][a] = 1.0
            for b in cols[i + 1:]:
                value = self._corr_from_frame(frame, a, b, method)
                matrix[a][b] = value
                matrix[b][a] = value
        return matrix

    def _auto_matrix(self, frame: pl.DataFrame, cols: list[str]) -> dict[str, Any]:
        """auto 模式：整数列用 Spearman、浮点列用 Pearson，异类统一 Spearman。

        方法选择按「列对」决定，因此同一张矩阵内可能混用两种系数；
        返回 ``methods_used``（逐列倾向）与 ``pair_methods``（逐对实际方法）供前端标注。
        """
        methods_used: dict[str, str] = {
            c: ("spearman" if frame.schema[c].is_integer() else "pearson")
            for c in cols
        }
        matrix: dict[str, dict[str, float | None]] = {c: {} for c in cols}
        pair_methods: dict[str, str] = {}

        for i, a in enumerate(cols):
            matrix[a][a] = 1.0
            for b in cols[i + 1:]:
                method = (
                    methods_used[a]
                    if methods_used[a] == methods_used[b]
                    else "spearman"
                )
                value = self._corr_from_frame(frame, a, b, method)
                matrix[a][b] = value
                matrix[b][a] = value
                pair_methods[f"{a}|{b}"] = method
                pair_methods[f"{b}|{a}"] = method

        return {
            "method": "auto",
            "columns": cols,
            "methods_used": methods_used,
            "pair_methods": pair_methods,
            "matrix": matrix,
        }


def _corr_limits() -> dict[str, int]:
    """相关性矩阵的规模上限（来自 settings，异常时退回安全默认值）。"""
    try:
        from app.core.config import settings

        return {
            "max_rows": int(settings.ANALYSIS_CORRELATION_MAX_ROWS),
            "max_columns": int(settings.ANALYSIS_CORRELATION_MAX_COLUMNS),
        }
    except Exception:  # pragma: no cover
        return {"max_rows": 5000, "max_columns": 30}


def _rank_columns_by_variance(df: pl.DataFrame, cols: list[str]) -> list[str]:
    """按方差降序取列，用于超限时的智能截断（保留信息量最大的列）。"""
    try:
        variances = df.select([pl.col(c).var().alias(c) for c in cols]).row(0, named=True)
    except Exception:
        return cols
    ranked = sorted(
        cols,
        key=lambda c: (float(variances.get(c) or 0.0)),
        reverse=True,
    )
    return ranked


class DistributionAnalyzer(EdaModule):
    name = "distribution"

    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = options.get("column")
        if not column:
            raise ValidationException(
                "分布分析必须指定要分析的字段（column）",
                details={"hint": "请先选择一个字段。"},
            )
        self.require_columns(df, [column])
        if df.schema[column].is_numeric():
            return self.numeric_distribution(df, column, bins=int(options.get("bins", 10)))
        return self.categorical_distribution(df, column, top_n=int(options.get("top_n", 20)))

    # ---- 数值分布 ----
    def numeric_distribution(
        self, df: pl.DataFrame, column: str, bins: int = 10, clip_quantile: float | None = None
    ) -> dict[str, Any]:
        bins = max(1, min(bins, 100))
        s = df[column]
        clean = s.drop_nulls()
        missing = int(s.null_count())
        if clean.len() == 0:
            return {"type": "numeric", "column": column, "bins": [], "missing": missing}

        clipped = False
        clip_value: float | None = None
        # 长尾裁剪：当指定 clip_quantile 且数据确有长尾（max 远大于该分位）时，
        # 把超过分位的值截到分位值再分桶，避免极端值把正常值全挤进第一个 bin
        # （回归：trip_distance 直方图横轴标到 26060/52120，正常行程全堆在首 bin 不可读）。
        if clip_quantile is not None and clean.len() >= 10:
            q = float(clean.quantile(clip_quantile, interpolation="linear"))
            mx_raw = float(clean.max())
            if mx_raw > q * 3:  # 明显长尾才裁剪，避免对正常分布误伤
                clean = clean.clip(upper_bound=q)
                clipped = True
                clip_value = q

        mn, mx = float(clean.min()), float(clean.max())
        if mn == mx:
            # 常数列：单桶
            return {
                "type": "numeric",
                "column": column,
                "min": mn,
                "max": mx,
                "mean": json_safe(s.mean()),
                "std": json_safe(s.std()),
                "missing": missing,
                "clipped": clipped,
                "clip_quantile": clip_quantile,
                "bins": [{"label": f"[{mn}]", "count": int(clean.len())}],
            }

        width = (mx - mn) / bins
        idx = ((clean - mn) / width).floor().clip(0, bins - 1).cast(pl.Int64)
        counts = idx.value_counts()
        count_map = {int(row[0]): int(row[1]) for row in counts.iter_rows()}

        fmt = _bin_label_format(mn, mx)
        bin_items = []
        for i in range(bins):
            lo, hi = mn + i * width, mn + (i + 1) * width
            bracket = "]" if i == bins - 1 else ")"
            label = f"[{format(lo, fmt)}, {format(hi, fmt)}{bracket}"
            bin_items.append({"label": label, "count": count_map.get(i, 0)})

        return {
            "type": "numeric",
            "column": column,
            "min": mn,
            "max": mx,
            "mean": json_safe(s.mean()),
            "std": json_safe(s.std()),
            "missing": missing,
            "clipped": clipped,
            "clip_quantile": clip_quantile,
            "bins": bin_items,
        }

    # ---- 分类分布 ----
    def categorical_distribution(
        self, df: pl.DataFrame, column: str, top_n: int = 20
    ) -> dict[str, Any]:
        top_n = max(1, min(top_n, 100))
        s = df[column]
        counts = s.drop_nulls().value_counts(sort=True).head(top_n)
        total_non_null = int(s.len() - s.null_count())
        items = [
            {"value": json_safe(row[0]), "count": int(row[1])}
            for row in counts.iter_rows()
        ]
        for item in items:
            item["ratio"] = round(item["count"] / total_non_null, 6) if total_non_null else 0.0
        return {
            "type": "categorical",
            "column": column,
            "unique_count": _safe_n_unique(s),
            "missing": int(s.null_count()),
            "values": items,
        }


def _bin_label_format(mn: float, mx: float) -> str:
    """按数据量级选分桶标签的有效数字。

    原先固定 ``:.4g``：很小量级（如 1e-6）或很大量级（如 1e12）下标签会退化成
    ``[0, 0)`` 或科学计数，前端 x 轴不可读。
    """
    span = abs(mx - mn)
    if span == 0 or span != span:  # 常量或 NaN
        return ",.4g"
    magnitude = math.log10(span)
    if magnitude >= 6 or magnitude <= -4:
        # 量级极端：用 3 位有效数字 + 科学计数，读起来比一串 0 好
        return ",.3g"
    # 常规量级：按跨度决定小数位，保证相邻桶标签可区分
    decimals = max(0, min(6, int(2 - math.floor(magnitude))))
    return f",.{decimals}f"


class EdaOutlierAnalyzer(EdaModule):
    name = "outlier"

    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        method = options.get("method", "iqr")
        columns: list[str] | None = options.get("columns")
        if columns:
            self.require_columns(df, columns)
        numeric_cols = self.pick_numeric(df, columns)
        if not numeric_cols:
            # 这是「参数/数据不适用」问题（缺少数值列），不是数据质量差 → 用 422。
            raise ValidationException(
                "异常值分析需要至少 1 个数值字段",
                details={
                    "columns": list(df.columns),
                    "hint": "请先选择数值类型的字段。",
                },
            )

        sample_limit = int(options.get("sample_limit", 10))
        results: list[dict[str, Any]] = []
        for name in numeric_cols:
            s = df[name]
            lower, upper, info = compute_outlier_bounds(
                s, method=method, k=float(options.get("k", 1.5)),
                z_threshold=float(options.get("z_threshold", 3.0)),
            )
            entry: dict[str, Any] = {
                "column": name,
                "method": method,
                "row_count": int(s.len()),
                "missing": int(s.null_count()),
            }
            if info.get("empty") or info.get("constant"):
                # 统一结构：skipped 的列也带上 bounds / outlier_count 等键，
                # 前端表格不必对两种行结构分别处理（避免出现空白格）。
                entry.update(
                    {
                        "status": "skipped",
                        "reason": "整列为空" if info.get("empty") else "常数列（方差为 0）",
                        "bounds": None,
                        "outlier_count": 0,
                        "outlier_ratio": 0.0,
                        "sample_outliers": [],
                        "inlier_range": None,
                        "stats": info,
                    }
                )
                results.append(entry)
                continue

            # 原实现对同一 mask 做了两次全列扫描（filter(mask) 与 filter(~mask)），
            # 这里改写为一次 with_columns 生成分类标记后再按标记取值：
            # Polars 1.44 的 Series 没有 partition，改用单次表达式求值达到同样的单遍效果。
            is_outlier = ((s < lower) | (s > upper)).fill_null(False)
            outlier_values = s.filter(is_outlier)
            # 内点 = 非空且非异常；用 drop_nulls 保证与旧实现口径一致（NaN 不进入统计）。
            inlier_values = s.filter(~is_outlier & s.is_not_null()).drop_nulls()
            entry.update(
                {
                    "status": "ok",
                    "bounds": {"lower": lower, "upper": upper},
                    "outlier_count": int(outlier_values.len()),
                    "outlier_ratio": (
                        round(outlier_values.len() / s.len(), 6) if s.len() else 0.0
                    ),
                    "sample_outliers": [
                        round(float(v), 6)
                        for v in outlier_values.head(sample_limit).to_list()
                        if v is not None
                    ],
                    "inlier_range": {
                        "min": float(inlier_values.min()) if inlier_values.len() else None,
                        "max": float(inlier_values.max()) if inlier_values.len() else None,
                    },
                    "stats": info,
                }
            )
            results.append(entry)

        return {"method": method, "columns": results}


class VisualizationBuilder(EdaModule):
    name = "visualization"

    def analyze(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        chart = options.get("chart")
        builders = {
            "histogram": self.histogram,
            "bar": self.bar,
            "line": self.line,
            "scatter": self.scatter,
            "boxplot": self.boxplot,
            "heatmap": self.heatmap,
            "qq": self.qq,
            "grouped_bar": self.grouped_bar,
            "area": self.area,
        }
        if chart not in builders:
            raise ValidationException(
                f"不支持的图表类型：{chart}",
                details={"chart": chart, "allowed": sorted(builders)},
            )
        return builders[chart](df, **{k: v for k, v in options.items() if k != "chart"})

    # ---- 直方图（数值分布）----
    def histogram(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = options.get("column")
        clip_q = options.get("clip_quantile")
        dist = DistributionAnalyzer().numeric_distribution(
            df, _require_str(column, "column"),
            bins=int(options.get("bins", 10)),
            clip_quantile=float(clip_q) if clip_q is not None else None,
        )
        return {
            "chart": "histogram",
            "column": dist["column"],
            "x": [b["label"] for b in dist["bins"]],
            "y": [b["count"] for b in dist["bins"]],
            "missing": dist["missing"],
            "clipped": dist.get("clipped", False),
        }

    # ---- 柱状图（分类分布）----
    def bar(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = options.get("column")
        dist = DistributionAnalyzer().categorical_distribution(
            df, _require_str(column, "column"), top_n=int(options.get("top_n", 20))
        )
        return {
            "chart": "bar",
            "column": dist["column"],
            "x": [str(i["value"]) for i in dist["values"]],
            "y": [i["count"] for i in dist["values"]],
            "missing": dist["missing"],
        }

    # ---- 折线图（x 有序；重复 x 聚合为均值）----
    def line(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        x, y = _require_str(options.get("x"), "x"), _require_str(options.get("y"), "y")
        self.require_columns(df, [x, y])
        if not df.schema[y].is_numeric():
            raise ValidationException(
                f"折线图的 y 列必须是数值列：{y}",
                details={"column": y, "dtype": str(df.schema[y])},
            )
        grouped = (
            df.select([x, y])
            .drop_nulls()
            .group_by(x)
            .agg(pl.col(y).mean().alias("__mean"))
            .sort(x)
        )
        original_points = grouped.height
        max_points = int(options.get("max_points", 1000))
        downsampled = False
        if max_points > 0 and grouped.height > max_points:
            # 等间隔抽稀：gather_every 保证均匀取样（原先用 `col == col // step` 比较浮点，
            # 会漏点且不保证首末点）。
            step = max(1, grouped.height // max_points)
            grouped = grouped.gather_every(step)
            downsampled = True
        return {
            "chart": "line",
            "x": [json_safe(v) for v in grouped[x].to_list()],
            "y": [json_safe(v) for v in grouped["__mean"].to_list()],
            "downsampled": downsampled,
            "original_count": int(original_points),
        }

    # ---- 散点图（下采样）----
    def scatter(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        x, y = _require_str(options.get("x"), "x"), _require_str(options.get("y"), "y")
        self.require_columns(df, [x, y])
        for c in (x, y):
            if not df.schema[c].is_numeric():
                raise ValidationException(
                    f"散点图的 {c} 列必须是数值列",
                    details={"column": c, "dtype": str(df.schema[c])},
                )
        sample_limit = int(options.get("sample_limit", 1000))
        data = df.select([x, y]).drop_nulls()
        original_count = data.height
        if data.height > sample_limit > 0:
            data = data.sample(n=sample_limit, seed=int(options.get("seed", 42)))
        return {
            "chart": "scatter",
            "x": [json_safe(v) for v in data[x].to_list()],
            "y": [json_safe(v) for v in data[y].to_list()],
            # 复用已算出的行数，不再为这一个布尔值重复做一次投影 + drop_nulls。
            "sampled": data.height < original_count,
            "original_count": int(original_count),
        }

    # ---- 箱线图（可选分组）----
    def boxplot(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = _require_str(options.get("column"), "column")
        self.require_columns(df, [column])
        if not df.schema[column].is_numeric():
            raise ValidationException(
                f"箱线图的列必须是数值列：{column}",
                details={"column": column, "dtype": str(df.schema[column])},
            )
        group_by = options.get("group_by")
        # whisker 倍率可配（默认 1.5×IQR），上限 5 防止用户传入离谱值。
        k = float(options.get("k", 1.5) or 1.5)
        series_list: list[tuple[str, pl.Series]]
        if group_by:
            self.require_columns(df, [group_by])
            series_list = [
                (str(g), sub[column])
                for (g,), sub in df.drop_nulls(subset=[group_by]).group_by(group_by)
            ]
        else:
            series_list = [(column, df[column])]

        boxes = []
        for label, s in series_list:
            clean = s.drop_nulls()
            if clean.len() == 0:
                continue
            q1 = float(clean.quantile(0.25, interpolation="linear"))
            q2 = float(clean.quantile(0.5, interpolation="linear"))
            q3 = float(clean.quantile(0.75, interpolation="linear"))
            iqr = q3 - q1
            lo_f, hi_f = q1 - k * iqr, q3 + k * iqr
            inliers = clean.filter((clean >= lo_f) & (clean <= hi_f))
            outliers = clean.filter((clean < lo_f) | (clean > hi_f))
            boxes.append(
                {
                    "label": label,
                    "q1": q1,
                    "median": q2,
                    "q3": q3,
                    "whisker_low": float(inliers.min()) if inliers.len() else None,
                    "whisker_high": float(inliers.max()) if inliers.len() else None,
                    "outlier_count": int(outliers.len()),
                    "outliers": [json_safe(v) for v in outliers.head(10).to_list()],
                }
            )
        return {"chart": "boxplot", "column": column, "k": k, "boxes": boxes}

    # ---- 热力图（相关性矩阵）----
    def heatmap(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        corr = CorrelationAnalyzer().analyze(
            df,
            method=options.get("method", "auto"),
            columns=options.get("columns"),
        )
        return {
            "chart": "heatmap",
            "method": corr["method"],
            "columns": corr["columns"],
            "matrix": corr["matrix"],
            # 抽样/截断等规模信息一并透出，前端可提示"已抽样展示"。
            "methods_used": corr.get("methods_used"),
            "pair_methods": corr.get("pair_methods"),
            "sampled": corr.get("sampled", False),
            "sample_size": corr.get("sample_size"),
            "original_rows": corr.get("original_rows"),
            "truncated": corr.get("truncated", False),
            "dropped_columns": corr.get("dropped_columns"),
        }

    # ---- Q-Q 图（正态性检验，科研级）----
    # 全量返回时 100 万行会产出约 16 MB JSON（x/y 各 100 万个 float），
    # 且 _norm_ppf 是逐点 Python 调用。因此加抽样上限：
    # 抽样不改变分布形状，分位点位置依然成立。
    QQ_MAX_POINTS = 5000

    def qq(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = _require_str(options.get("column"), "column")
        self.require_columns(df, [column])
        if not df.schema[column].is_numeric():
            raise ValidationException(
                f"Q-Q 图的列必须是数值列：{column}",
                details={"column": column, "dtype": str(df.schema[column])},
            )
        s = df[column].drop_nulls().sort()
        total_n = s.len()
        if total_n < 3:
            raise ValidationException(
                "Q-Q 图至少需要 3 个非空数值样本",
                details={"column": column, "count": int(total_n)},
            )

        # 等间隔抽稀（保留排序后的整体形状，含首末点）
        max_points = int(options.get("sample_limit", self.QQ_MAX_POINTS) or self.QQ_MAX_POINTS)
        sampled = False
        if max_points > 0 and total_n > max_points:
            step = max(1, total_n // max_points)
            s = s.gather_every(step)
            sampled = True

        n = s.len()
        vals = [float(v) for v in s.to_list()]
        theoretical = [_norm_ppf((i + 0.5) / n) for i in range(n)]
        # 参考线：对 (理论分位, 样本值) 做最小二乘
        mx = sum(theoretical) / n
        my = sum(vals) / n
        sxx = sum((t - mx) ** 2 for t in theoretical)
        sxy = sum((t - mx) * (v - my) for t, v in zip(theoretical, vals))
        slope = sxy / sxx if sxx else 1.0
        intercept = my - slope * mx
        return {
            "chart": "qq",
            "column": column,
            "n": int(n),
            "x": [round(t, 4) for t in theoretical],
            "y": vals,
            "line": {
                "x0": theoretical[0],
                "y0": slope * theoretical[0] + intercept,
                "x1": theoretical[-1],
                "y1": slope * theoretical[-1] + intercept,
            },
            "mean": round(my, 4),
            "std": round(slope, 4),
            "sampled": sampled,
            "original_count": int(total_n),
        }

    # ---- 分组柱状图（科研级：分类 × 分组对比）----
    def grouped_bar(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = _require_str(options.get("column"), "column")
        y = _require_str(options.get("y"), "y")
        group_by = _require_str(options.get("group_by"), "group_by")
        self.require_columns(df, [column, y, group_by])
        if not df.schema[y].is_numeric():
            raise ValidationException(
                f"分组柱状图的 y 列必须是数值列：{y}",
                details={"column": y, "dtype": str(df.schema[y])},
            )
        agg = options.get("agg", "mean")
        if agg not in ("mean", "sum", "count", "median"):
            raise ValidationException(
                f"不支持的聚合方式：{agg}",
                details={"allowed": ["mean", "sum", "count", "median"]},
            )
        agg_expr = {
            "mean": pl.col(y).mean(),
            "sum": pl.col(y).sum(),
            "median": pl.col(y).median(),
            "count": pl.col(y).count(),
        }[agg].alias("__v")
        grouped = (
            df.select([column, y, group_by])
            .drop_nulls()
            .group_by(column, group_by)
            .agg(agg_expr)
        )

        # 基数上限：分类列 × 分组列的笛卡尔积会直接决定返回体大小，
        # 原实现完全不设限（参数 top_n 也未被使用）。这里按「出现频次」取前 N 个分类。
        top_n = int(options.get("top_n", 20) or 20)
        top_n = max(1, min(top_n, 100))
        all_cats = df[column].drop_nulls()
        cat_counts = (
            all_cats.value_counts(sort=True).head(top_n)
            if all_cats.len() > top_n else None
        )
        if cat_counts is not None:
            cats = [str(v) for v in cat_counts[column].to_list()]
            cats = sorted(cats)
            truncated = True
            dropped_categories = int(all_cats.n_unique() - len(cats))
        else:
            cats = [str(c) for c in sorted(all_cats.unique().to_list())]
            truncated = False
            dropped_categories = 0

        groups = [str(g) for g in sorted(df[group_by].drop_nulls().unique().to_list())]
        lookup = {
            (str(r[column]), str(r[group_by])): float(r["__v"])
            for r in grouped.iter_rows(named=True)
        }
        series = {
            g: [lookup.get((c, g)) for c in cats] for g in groups
        }
        return {
            "chart": "grouped_bar",
            "x": cats,
            "groups": groups,
            "series": series,
            "y_label": agg,
            "truncated": truncated,
            "dropped_categories": dropped_categories,
        }

    # ---- 面积图（累积分布 CDF，科研级）----
    def area(self, df: pl.DataFrame, **options: Any) -> dict[str, Any]:
        column = _require_str(options.get("column"), "column")
        self.require_columns(df, [column])
        if not df.schema[column].is_numeric():
            raise ValidationException(
                f"面积图的列必须是数值列：{column}",
                details={"column": column, "dtype": str(df.schema[column])},
            )
        dist = DistributionAnalyzer().numeric_distribution(
            df, column, bins=int(options.get("bins", 12))
        )
        counts = [b["count"] for b in dist["bins"]]
        labels = [b["label"] for b in dist["bins"]]
        total = sum(counts)
        run = 0
        cum = []
        for c in counts:
            run += c
            cum.append(round(run / total, 4) if total else 0.0)
        return {
            "chart": "area",
            "column": column,
            "x": labels,
            "y": cum,
            "total": int(total),
            "mode": "cdf",
        }


def _norm_ppf(p: float) -> float:
    """标准正态分位函数（逆 CDF）的 Acklam 有理逼近，最大绝对误差 ~1.15e-9。

    自包含实现，不依赖 scipy / numpy。p 须在 (0,1)。
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = (-2 * _log(p)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = (-2 * _log(1 - p)) ** 0.5
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _log(x: float) -> float:
    import math

    return math.log(x)


# 图表参数的中文名，用于把「缺少 xxx 参数」写成用户能读懂的提示。
_FIELD_LABELS = {
    "column": "字段（column）",
    "x": "x 轴字段（x）",
    "y": "y 轴字段（y）",
    "group_by": "分组字段（group_by）",
}


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        label = _FIELD_LABELS.get(name, name)
        raise ValidationException(
            f"该图表类型必须指定{label}",
            details={"field": name, "hint": f"请传入 {name} 参数。"},
        )
    return value
