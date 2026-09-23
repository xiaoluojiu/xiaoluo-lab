"""Quality 层 · 字段语义标准（第二层改造）。

为什么需要它
------------
通用方法（IQR / Z-score / one-hot / 相关系数）之所以会在真实数据上失效，
是因为**方法的选择没有依赖字段是什么**。IQR 在 ``DepDelay`` 上给出下界 -16.5、
在 ``Distance`` 上给出 -636.5 —— 方法本身没错，错的是把它用在了非负字段上；
而调用方（报告、Agent）拿到 ``[-16.5, 19.5]`` 无从判断这是不是胡说。

本模块把「字段是什么」标准化成一份可复用、可序列化、可审计的元数据：

    ColumnSemantics(role, domain, shape, cardinality, null_ratio, is_target, ...)

它是后续所有「方法适配」的唯一输入：

* :mod:`app.quality.outlier_strategy` —— 选异常值判定方法；
* :mod:`app.ml_engine.feature.operations` —— 选特征工程操作；
* :mod:`app.ml_engine.metric_advisor` —— 选评估指标。

设计约束
--------
* **判定必须带证据**：每一条 role / domain / shape 判定都在 ``evidence`` 里留下
  可读依据，Agent 与报告据此解释，不允许出现无法追溯的结论；
* **不确定就说不确定**：推不出来时给 ``UNKNOWN`` + 低 confidence，绝不静默给默认值；
* **复用成熟库**：统计量一律走 Polars 原生聚合（``skew`` / ``kurtosis`` / ``quantile``），
  批量 ``select`` 一次取回，不逐列全表扫描；
* **用户标注优先**：``annotations`` / ``business_rules`` 覆盖自动判定（业务口径 > 统计口径）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

import polars as pl

__all__ = [
    "BusinessRule",
    "ColumnRole",
    "ColumnSemantics",
    "DistributionShape",
    "ValueDomain",
    "column_tokens",
    "infer_semantics",
]

# --------------------------------------------------------------------------- #
# 列名分词（同时供 ml_engine.target_inference 复用，避免两份实现）
# --------------------------------------------------------------------------- #

_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def column_tokens(column: Any) -> list[str]:
    """把列名切成小写词元。

    ``DayofMonth`` → ``['dayof','month']``；``crs_dep_time`` → ``['crs','dep','time']``；
    ``DepDelay`` → ``['dep','delay']``；中文列名整块保留。
    """
    parts: list[str] = []
    for chunk in _SPLIT_RE.split(str(column or "")):
        if not chunk:
            continue
        parts.extend(p for p in _CAMEL_RE.split(chunk) if p)
    return [p.lower() for p in parts]


# --------------------------------------------------------------------------- #
# 枚举：角色 / 值域 / 分布形态
# --------------------------------------------------------------------------- #


class ColumnRole(StrEnum):
    """字段在分析链路里扮演的角色。"""

    CONTINUOUS = "continuous"      # 连续数值
    CATEGORICAL = "categorical"    # 低基数类别（含编码列）
    TEMPORAL = "temporal"          # 日期/时间（含 HHMM 伪数值）
    IDENTIFIER = "identifier"      # 主键/编号，不参与分析
    CONSTANT = "constant"          # 常数列，无信息量
    TEXT = "text"                  # 自由文本
    TARGET = "target"              # 目标列（由调用方指定）
    UNKNOWN = "unknown"


class ValueDomain(StrEnum):
    """取值域约束——决定「统计口径」能否直接使用。"""

    NON_NEGATIVE = "non_negative"  # 非负（距离、时长、计数）：IQR 下界为负即不适用
    POSITIVE = "positive"          # 严格正
    SIGNED = "signed"              # 有正负（温度、盈亏）：IQR 可用
    UNIT = "unit"                  # 落在 [0,1]
    UNKNOWN = "unknown"


class DistributionShape(StrEnum):
    """目标/连续列的分布形态——决定指标与异常值口径。"""

    NORMAL = "normal"
    LONG_TAIL = "long_tail"        # 长尾：不判异常，只标注
    ZERO_INFLATED = "zero_inflated"  # 零膨胀：中位数为 0 但方差极大
    BIMODAL = "bimodal"
    UNIFORM = "uniform"
    UNKNOWN = "unknown"


_ROLE_LABELS = {
    ColumnRole.CONTINUOUS: "连续数值",
    ColumnRole.CATEGORICAL: "类别",
    ColumnRole.TEMPORAL: "时间",
    ColumnRole.IDENTIFIER: "标识/编号",
    ColumnRole.CONSTANT: "常量",
    ColumnRole.TEXT: "文本",
    ColumnRole.TARGET: "目标列",
    ColumnRole.UNKNOWN: "未知",
}

_DOMAIN_LABELS = {
    ValueDomain.NON_NEGATIVE: "非负",
    ValueDomain.POSITIVE: "严格正",
    ValueDomain.SIGNED: "有正负",
    ValueDomain.UNIT: "[0,1]",
    ValueDomain.UNKNOWN: "未知",
}

_SHAPE_LABELS = {
    DistributionShape.NORMAL: "近似正态",
    DistributionShape.LONG_TAIL: "长尾",
    DistributionShape.ZERO_INFLATED: "零膨胀",
    DistributionShape.BIMODAL: "双峰",
    DistributionShape.UNIFORM: "均匀",
    DistributionShape.UNKNOWN: "未知",
}


# --------------------------------------------------------------------------- #
# 业务口径
# --------------------------------------------------------------------------- #


@dataclass
class BusinessRule:
    """「什么算异常」的业务口径。

    业务口径优先于统计口径：用户说「Distance ≤ 0 才是问题」时，IQR 算出来的
    -636.5 就不该被当成下界。表达式刻意**不支持**自由字符串求值（避免 eval），
    只支持区间与枚举两类可序列化、可校验的约束。
    """

    column: str
    lower: float | None = None
    upper: float | None = None
    #: True ⇒ 边界本身不算违规（默认 ``<=`` / ``>=`` 为合规）
    strict_lower: bool = False
    strict_upper: bool = False
    allowed_values: list[Any] | None = None
    note: str = ""

    @property
    def active(self) -> bool:
        return any(
            x is not None
            for x in (self.lower, self.upper, self.allowed_values)
        )

    def describe(self) -> str:
        bits: list[str] = []
        if self.lower is not None:
            bits.append(f"{'>' if self.strict_lower else '>='}{self.lower:g}")
        if self.upper is not None:
            bits.append(f"{'<' if self.strict_upper else '<='}{self.upper:g}")
        if self.allowed_values is not None:
            bits.append(f"取值限于 {self.allowed_values}")
        text = " 且 ".join(bits) or "（空规则）"
        return f"{self.column}: {text}" + (f"（{self.note}）" if self.note else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "lower": self.lower,
            "upper": self.upper,
            "strict_lower": self.strict_lower,
            "strict_upper": self.strict_upper,
            "allowed_values": self.allowed_values,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# 字段语义
# --------------------------------------------------------------------------- #


@dataclass
class ColumnSemantics:
    """一个字段的完整语义画像。"""

    name: str
    dtype: str
    role: ColumnRole = ColumnRole.UNKNOWN
    domain: ValueDomain = ValueDomain.UNKNOWN
    shape: DistributionShape = DistributionShape.UNKNOWN
    cardinality: int = -1
    null_ratio: float = 0.0
    is_target: bool = False
    #: 伪时间编码格式（如 ``HHMM``）：值是整数但存在 2359→0000 的跳变
    pseudo_time_format: str | None = None
    #: 业务含义标签（日历 / 金额 / 时长 / 编码 …），来自命名 + 用户标注
    tags: list[str] = field(default_factory=list)
    #: 判定依据（人可读），供 Agent 解释与报告脚注使用
    evidence: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    # ---- 供下游做方法适配的快捷判据 -------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.role in (
            ColumnRole.CONTINUOUS,
            ColumnRole.CATEGORICAL,
            ColumnRole.TEMPORAL,
            ColumnRole.TARGET,
        ) and self.stats.get("numeric", False)

    @property
    def is_non_negative(self) -> bool:
        return self.domain in (ValueDomain.NON_NEGATIVE, ValueDomain.POSITIVE, ValueDomain.UNIT)

    @property
    def is_long_tail_or_zero_inflated(self) -> bool:
        return self.shape in (DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED)

    @property
    def is_high_cardinality(self) -> bool:
        return self.role == ColumnRole.CATEGORICAL and self.cardinality > 50

    @property
    def label(self) -> str:
        return f"{self.name}（{_ROLE_LABELS[self.role]}／{_SHAPE_LABELS[self.shape]}）"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": str(self.role),
            "role_label": _ROLE_LABELS[self.role],
            "domain": str(self.domain),
            "domain_label": _DOMAIN_LABELS[self.domain],
            "shape": str(self.shape),
            "shape_label": _SHAPE_LABELS[self.shape],
            "cardinality": self.cardinality,
            "null_ratio": round(self.null_ratio, 6),
            "is_target": self.is_target,
            "pseudo_time_format": self.pseudo_time_format,
            "tags": list(self.tags),
            "evidence": list(self.evidence),
            "stats": {k: v for k, v in self.stats.items()},
            "confidence": round(self.confidence, 3),
        }


# --------------------------------------------------------------------------- #
# 命名启发式
# --------------------------------------------------------------------------- #

_ID_TOKENS = frozenset(
    {"id", "ids", "index", "idx", "rowid", "row", "key", "uuid", "guid",
     "code", "seq", "serial", "pk", "fk"}
)
_CALENDAR_TOKENS = frozenset(
    {"month", "months", "day", "days", "date", "year", "years", "week", "weeks",
     "hour", "hours", "minute", "minutes", "second", "seconds", "time", "times",
     "timestamp", "quarter", "weekday", "weekend", "dow", "dom", "doy", "moy"}
)
_TIME_OF_DAY_TOKENS = frozenset({"deptime", "arrtime", "crsdeptime", "crsarrtime"})
_MONEY_TOKENS = frozenset({"price", "fare", "amount", "cost", "fee", "revenue", "sales", "金额", "价格", "票价"})
_DURATION_TOKENS = frozenset({"duration", "elapsed", "delay", "interval", "时长", "耗时", "延误"})


def _tags_of(name: str) -> list[str]:
    tokens = set(column_tokens(name))
    tags: list[str] = []
    if tokens & _ID_TOKENS:
        tags.append("identifier")
    if tokens & _CALENDAR_TOKENS:
        tags.append("calendar")
    if tokens & _TIME_OF_DAY_TOKENS:
        tags.append("time_of_day")
    if tokens & _MONEY_TOKENS:
        tags.append("money")
    if tokens & _DURATION_TOKENS:
        tags.append("duration")
    return tags


def _detect_pseudo_time_format(
    series: "pl.Series", stats: dict[str, Any], name: str
) -> tuple[str | None, str]:
    """识别 HHMM 这类「伪数值时间」。

    判据（**四条同时成立**才算，缺一条就放弃，宁可漏判不可误判）：

    ① 值全部为整数；
    ② 落在 ``[100, 2359]``（上界排除 Month=12 / DayofMonth=31 这类日历列）；
    ③ 分钟部分 ``v % 100`` 恒 < 60；
    ④ 存在 ``…59 → …00`` 的跳变（间隔 41）——这是 HHMM 的独有指纹：
       分钟走到 59 后不是 60 而是下一小时的 00。

    第 ④ 条是关键：只靠 ①②③ 会把 ``Year``(1990..2024)、``Month``(1..12)
    这类整值日历列误判成时间编码。

    ``CRSDepTime`` = 2359 时下一分钟是 0000，直接当数值用会出现 2359→0 的断崖，
    是典型的「必须做特征工程而不是直接入模」的字段。
    """
    if not stats.get("integral", False):
        return None, ""
    low, high = stats.get("min"), stats.get("max")
    if low is None or high is None:
        return None, ""
    if not (0 <= low and 100 <= high <= 2359):
        return None, ""
    if (stats.get("minute_part_max") or 99) >= 60:
        return None, ""
    try:
        uniq = [int(v) for v in series.unique().sort().to_list()]
    except Exception:  # noqa: BLE001 - 取不到唯一值就放弃判定
        return None, ""
    jump = any(
        b - a == 41 and a % 100 == 59 and b % 100 == 0
        for a, b in zip(uniq, uniq[1:])
    )
    if not jump:
        return None, ""
    return (
        "HHMM",
        f"整数值落在 [{low:.0f}, {high:.0f}]、分钟部分恒 <60 且存在 59→00 跳变，"
        f"判定为 HHMM 伪数值时间（{name} 存在 2359→0000 的断崖，需做循环/分解处理）",
    )


# --------------------------------------------------------------------------- #
# 分布形态
# --------------------------------------------------------------------------- #


def _shape_of(stats: dict[str, Any], rows: int) -> tuple[DistributionShape, str]:
    """由统计量判分布形态（样本太少时 honest 地返回 UNKNOWN）。"""
    if rows < 30 or not stats.get("numeric"):
        return DistributionShape.UNKNOWN, "样本量或类型不足以判定分布形态"

    zero_ratio = float(stats.get("zero_ratio") or 0.0)
    mean = stats.get("mean")
    std = stats.get("std")
    skew = stats.get("skew")
    kurt = stats.get("kurtosis")

    if zero_ratio >= 0.5 and std:
        return (
            DistributionShape.ZERO_INFLATED,
            f"零值占比 {zero_ratio:.1%} ≥ 50%，判定为零膨胀分布",
        )
    if skew is not None and abs(float(skew)) >= 2.0:
        return (
            DistributionShape.LONG_TAIL,
            f"偏度 |skew|={abs(float(skew)):.2f} ≥ 2，判定为长尾分布",
        )
    if mean and std and abs(float(std) / float(mean)) >= 1.5:
        return (
            DistributionShape.LONG_TAIL,
            f"变异系数 std/|mean|={abs(float(std) / float(mean)):.2f} ≥ 1.5，判定为长尾分布",
        )
    if kurt is not None and float(kurt) < -1.0:
        return (
            DistributionShape.UNIFORM,
            f"峰度 kurtosis={float(kurt):.2f} < -1，判定为平坦/均匀型分布",
        )
    if skew is not None and abs(float(skew)) <= 0.75:
        return (
            DistributionShape.NORMAL,
            f"偏度 |skew|={abs(float(skew)):.2f} ≤ 0.75 且非零膨胀，近似正态",
        )
    return DistributionShape.UNKNOWN, "统计量不足以给出明确形态判定"


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def _numeric_stats(df: pl.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    """一次 select 取回所有数值列的统计量（避免逐列全表扫描）。"""
    out: dict[str, dict[str, Any]] = {}
    if not columns:
        return out
    exprs: list[pl.Expr] = []
    for c in columns:
        s = pl.col(c)
        exprs += [
            s.min().alias(f"{c}::min"),
            s.max().alias(f"{c}::max"),
            s.mean().alias(f"{c}::mean"),
            s.std().alias(f"{c}::std"),
            s.median().alias(f"{c}::median"),
            s.quantile(0.25).alias(f"{c}::q1"),
            s.quantile(0.75).alias(f"{c}::q3"),
            s.quantile(0.99).alias(f"{c}::p99"),
            s.null_count().alias(f"{c}::nulls"),
            (s == 0).sum().alias(f"{c}::zeros"),
            (s < 0).sum().alias(f"{c}::negatives"),
            s.drop_nulls().n_unique().alias(f"{c}::nuniq"),
        ]
    try:
        row = df.select(exprs).row(0, named=True)
    except Exception:  # noqa: BLE001 - 批量失败退化为逐列，宁可慢也不报错
        row = {}
    for c in columns:
        def _v(key: str) -> Any:
            return row.get(f"{c}::{key}") if isinstance(row, dict) else None

        out[c] = {
            "numeric": True,
            "min": _v("min"), "max": _v("max"), "mean": _v("mean"),
            "std": _v("std"), "median": _v("median"), "q1": _v("q1"),
            "q3": _v("q3"), "p99": _v("p99"),
            "null_count": int(_v("nulls") or 0),
            "zero_count": int(_v("zeros") or 0),
            "negative_count": int(_v("negatives") or 0),
            "n_unique": int(_v("nuniq") or 0),
        }
    return out


def infer_semantics(
    df: pl.DataFrame,
    *,
    target: str | None = None,
    annotations: dict[str, dict[str, Any]] | None = None,
    business_rules: Iterable[BusinessRule] | None = None,
    columns: list[str] | None = None,
) -> dict[str, ColumnSemantics]:
    """推断全部字段的语义画像。

    Args:
        df: 数据（只读 schema + 聚合统计，不物化到 Python）。
        target: 目标列名（业务标记，优先级高于自动判定）。
        annotations: 用户/调用方标注，如 ``{"DepDelay": {"tags": ["duration"], "role": "continuous"}}``。
        business_rules: 业务口径（只登记，判定交由各策略模块消费）。
        columns: 只推断这些列（默认全部）。
    """
    cols = [c for c in (columns or df.columns) if c in df.columns]
    rows = int(df.height)
    annotations = annotations or {}
    result: dict[str, ColumnSemantics] = {}

    # 一次批量取分类性判定（复用 analysis.classify_columns，避免第二份实现）
    from app.analysis import classify_columns

    categorical_like = classify_columns(df, cols)
    numeric_cols = [c for c in cols if df.schema[c].is_numeric()]
    stats_map = _numeric_stats(df, numeric_cols)

    for c in cols:
        dtype = df.schema[c]
        ann = annotations.get(c) or {}
        sem = ColumnSemantics(name=c, dtype=str(dtype))
        sem.tags = list(ann.get("tags") or _tags_of(c))
        sem.pseudo_time_format = ann.get("pseudo_time_format")

        if c == target:
            sem.is_target = True

        if dtype.is_numeric():
            st = stats_map.get(c, {})
            series = None
            # 逐列补算的统计量（skew / kurtosis / 整值性）
            try:
                series = df[c].drop_nulls()
                st["skew"] = float(series.skew()) if series.len() else None
                st["kurtosis"] = float(series.kurtosis()) if series.len() else None
                st["integral"] = False
                if series.len():
                    diff = (series - series.round(0)).abs().max()
                    st["integral"] = bool(diff is not None and float(diff) <= 1e-9)
                st["minute_part_max"] = (
                    int((series % 100).max()) if st.get("integral") else None
                )
            except Exception:  # noqa: BLE001 - 退化：形态判定降级为 UNKNOWN
                st.setdefault("skew", None)
                st.setdefault("kurtosis", None)
                st["integral"] = False
            non_null = max(rows - int(st.get("null_count") or 0), 0)
            st["zero_ratio"] = (int(st.get("zero_count") or 0) / non_null) if non_null else 0.0
            st["negative_ratio"] = (int(st.get("negative_count") or 0) / non_null) if non_null else 0.0
            sem.stats = st
            sem.cardinality = int(st.get("n_unique") or 0)
            sem.null_ratio = (int(st.get("null_count") or 0) / rows) if rows else 0.0

            # ---- role ----
            if sem.cardinality <= 1:
                sem.role, sem.confidence = ColumnRole.CONSTANT, 0.95
                sem.evidence.append("唯一值数 ≤ 1，判定为常数列（无信息量）")
            elif dtype in (pl.Date, pl.Datetime):
                sem.role, sem.confidence = ColumnRole.TEMPORAL, 0.95
                sem.evidence.append("dtype 为日期/时间类型")
            elif rows > 20 and sem.cardinality == rows:
                sem.role, sem.confidence = ColumnRole.IDENTIFIER, 0.9
                sem.evidence.append("每行唯一，判定为主键/标识列")
            elif "identifier" in sem.tags and sem.cardinality > rows * 0.9:
                sem.role, sem.confidence = ColumnRole.IDENTIFIER, 0.8
                sem.evidence.append("列名含标识语义且基数接近行数")
            else:
                fmt, why = _detect_pseudo_time_format(
                    series if series is not None else df[c].drop_nulls(), st, c
                )
                if fmt:
                    sem.role, sem.confidence = ColumnRole.TEMPORAL, 0.85
                    sem.pseudo_time_format = fmt
                    sem.evidence.append(why)
                elif categorical_like.get(c, False):
                    sem.role, sem.confidence = ColumnRole.CATEGORICAL, 0.8
                    sem.evidence.append(
                        f"唯一值数 {sem.cardinality} ≤ 50 且远小于行数，判定为类别/编码列"
                    )
                else:
                    sem.role, sem.confidence = ColumnRole.CONTINUOUS, 0.8
                    sem.evidence.append("数值列且不满足类别/标识/时间判据，判定为连续变量")

            # ---- domain ----
            low, high = st.get("min"), st.get("max")
            neg_ratio = float(st.get("negative_ratio") or 0.0)
            if low is not None and high is not None:
                if low >= 0 and high <= 1:
                    sem.domain = ValueDomain.UNIT
                    sem.evidence.append(f"取值域 [{low:g}, {high:g}] 落在 [0,1]")
                elif low >= 0 and neg_ratio == 0.0:
                    sem.domain = ValueDomain.NON_NEGATIVE
                    sem.evidence.append(f"最小值 {low:g} ≥ 0 且无负值，判定为非负字段")
                elif low > 0:
                    sem.domain = ValueDomain.POSITIVE
                    sem.evidence.append(f"最小值 {low:g} > 0，判定为严格正字段")
                else:
                    sem.domain = ValueDomain.SIGNED
                    sem.evidence.append(f"最小值 {low:g} < 0，存在负值，判定为有符号字段")

            # ---- shape ----
            sem.shape, why = _shape_of(st, rows)
            if why:
                sem.evidence.append(why)
        else:
            # 非数值：类别 / 文本 / 时间
            try:
                sem.cardinality = int(df[c].n_unique())
            except Exception:  # noqa: BLE001
                sem.cardinality = -1
            sem.null_ratio = (int(df[c].null_count()) / rows) if rows else 0.0
            if dtype in (pl.Date, pl.Datetime):
                sem.role, sem.confidence = ColumnRole.TEMPORAL, 0.95
                sem.evidence.append("dtype 为日期/时间类型")
            elif sem.cardinality == 1:
                sem.role, sem.confidence = ColumnRole.CONSTANT, 0.95
                sem.evidence.append("唯一值数 = 1，判定为常数列")
            elif sem.cardinality > 100 and sem.cardinality > rows * 0.5 and rows > 20:
                sem.role, sem.confidence = ColumnRole.TEXT, 0.7
                sem.evidence.append(f"高基数文本列（{sem.cardinality} 个唯一值）")
            else:
                sem.role, sem.confidence = ColumnRole.CATEGORICAL, 0.85
                sem.evidence.append(f"非数值且唯一值数 {sem.cardinality} 可控，判定为类别列")

        # ---- 用户标注覆盖（业务口径 > 自动判定） -------------------------
        ann_role = ann.get("role")
        if ann_role:
            try:
                sem.role = ColumnRole(ann_role)
                sem.confidence = 1.0
                sem.evidence.append(f"由调用方标注覆盖角色为 {_ROLE_LABELS[sem.role]}")
            except ValueError:
                sem.evidence.append(f"忽略无法识别的标注角色 {ann_role!r}")
        if ann.get("domain"):
            try:
                sem.domain = ValueDomain(ann["domain"])
            except ValueError:
                pass
        if ann.get("is_target"):
            sem.is_target = True
        if sem.is_target and sem.role != ColumnRole.CONSTANT:
            # 目标列只描述、不做清洗建议（由下游消费 is_target 判定）
            sem.tags.append("target")

        result[c] = sem

    # 业务规则只做登记：把约束写回对应列的 stats，供策略模块直接取用
    for rule in business_rules or []:
        sem = result.get(rule.column)
        if sem is None or not rule.active:
            continue
        sem.stats["business_rule"] = rule.to_dict()
        sem.tags.append("business_rule")
        sem.evidence.append(f"已登记业务口径：{rule.describe()}")

    return result
