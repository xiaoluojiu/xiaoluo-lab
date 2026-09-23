"""Quality 层 · 异常值判定方法适配（第二层改造）。

真实事故
--------
``DepDelay`` 的 IQR 边界是 ``[-16.5, 19.5]``，``Distance`` 的是 ``[-636.5, 1896]``。
下界为负——**方法本身不适用于这两个字段**，但平台照常输出，报告还据此
建议「清洗异常值」。

根因不是 IQR 算错了，而是**方法的适用性从未被检查**：

    compute_outlier_bounds(s, method="iqr")  →  (lower, upper, info)

调用方拿到的三元组里没有任何「这个方法能不能用在这里」的信息。

本模块把它标准化为「先选方法、再算边界」两步，且每一步都返回
:class:`app.core.contracts.Decision`（带 source / confidence / reasons / findings）：

    select_outlier_method(sem)   → Decision[str]            选方法（含 method_mismatch 警告）
    compute_bounds(sem, s, ...)  → Decision[OutlierBounds]  算边界（含裁剪与不适用说明）

方法表 :data:`METHOD_SPECS` 是**声明式**的：每个方法自己声明适用条件
（``guard``）与优先级，新增方法只加一条记录，不改调度逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from app.core.contracts import Decision, Finding, Severity, finding
from app.quality.semantics import BusinessRule, ColumnRole, ColumnSemantics, DistributionShape

__all__ = [
    "METHOD_SPECS",
    "MethodSpec",
    "OutlierBounds",
    "assess_column",
    "compute_bounds",
    "select_outlier_method",
    "should_suggest_cleaning",
]


@dataclass
class OutlierBounds:
    """一组异常值边界及其口径说明。"""

    method: str
    lower: float | None
    upper: float | None
    #: 边界是否被适配改写（如非负字段把负下界裁到 0）
    adjusted: bool = False
    notes: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.lower is not None or self.upper is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lower": self.lower,
            "upper": self.upper,
            "adjusted": self.adjusted,
            "notes": list(self.notes),
            "info": dict(self.info),
        }


# --------------------------------------------------------------------------- #
# 声明式方法表
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MethodSpec:
    """一个异常值判定方法的适用声明。

    ``guard`` 返回 ``None`` 表示适用，返回字符串表示**不适用及原因**。
    这种「返回原因而不是布尔」的写法是为了让不适用也有可解释性——
    Agent 与报告需要告诉用户「为什么没用 IQR」，而不只是「没用 IQR」。
    """

    code: str
    label: str
    guard: Callable[[ColumnSemantics], str | None]
    priority: int = 0
    note: str = ""

    def check(self, sem: ColumnSemantics) -> str | None:
        return self.guard(sem)


def _guard_iqr(sem: ColumnSemantics) -> str | None:
    if sem.role == ColumnRole.CATEGORICAL:
        return f"{sem.name} 是类别/编码列，数值型异常判定无意义（应做低频类别检测）"
    if sem.role in (ColumnRole.CONSTANT, ColumnRole.IDENTIFIER, ColumnRole.TEMPORAL, ColumnRole.TEXT):
        return f"{sem.label} 不是普通连续字段，IQR 无意义"
    if sem.is_long_tail_or_zero_inflated:
        return (
            f"{sem.name} 为{_shape_label(sem)}分布，IQR 会把大量正常的大值判成异常，"
            "应改用鲁棒方法或只标注长尾"
        )
    if sem.is_non_negative:
        # 非负字段上 IQR 的下界常常为负 ⇒ 下界失效（只保留上界意义）
        return (
            f"{sem.name} 为非负字段，IQR 下界易为负（如 DepDelay 的 -16.5），"
            "需改用分位数口径或业务口径"
        )
    return None


def _guard_zscore(sem: ColumnSemantics) -> str | None:
    if sem.role == ColumnRole.CATEGORICAL:
        return f"{sem.name} 是类别/编码列，Z-score 无意义"
    if sem.role in (ColumnRole.CONSTANT, ColumnRole.IDENTIFIER, ColumnRole.TEMPORAL, ColumnRole.TEXT):
        return f"{sem.label} 不是普通连续字段，Z-score 无意义"
    if sem.shape in (DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED):
        return f"{sem.name} 为{_shape_label(sem)}分布，均值与标准差被极端值拉偏，Z-score 不可靠"
    if sem.shape != DistributionShape.NORMAL:
        return f"{sem.name} 分布形态为{_shape_label(sem)}，Z-score 只在近似正态下有意义"
    return None


def _guard_mad(sem: ColumnSemantics) -> str | None:
    """中位数绝对偏差：对长尾/零膨胀鲁棒（IQR/Z-score 都失效时的兜底）。"""
    if sem.role == ColumnRole.CATEGORICAL:
        return f"{sem.name} 是类别/编码列，MAD 无意义"
    if sem.role in (ColumnRole.CONSTANT, ColumnRole.IDENTIFIER, ColumnRole.TEXT):
        return f"{sem.label} 无散布，MAD 无意义"
    if sem.role == ColumnRole.TEMPORAL:
        return f"{sem.name} 是时间字段，MAD 不适用"
    return None


def _guard_quantile(sem: ColumnSemantics) -> str | None:
    """分位数口径：不假设分布形态，天然适配非负/长尾字段。"""
    if sem.role == ColumnRole.CATEGORICAL:
        return f"{sem.name} 是类别/编码列，分位数口径不适用（应做低频类别检测）"
    if sem.role in (ColumnRole.CONSTANT, ColumnRole.IDENTIFIER, ColumnRole.TEXT):
        return f"{sem.label} 无散布，分位数口径无意义"
    if sem.role == ColumnRole.TEMPORAL and not sem.pseudo_time_format:
        return f"{sem.name} 是时间字段，分位数口径不适用"
    return None


def _guard_low_frequency(sem: ColumnSemantics) -> str | None:
    """低频类别检测：类别字段的「异常」是长尾取值，不是数值越界。"""
    if sem.role != ColumnRole.CATEGORICAL:
        return f"{sem.name} 不是类别字段，低频类别检测不适用"
    return None


def _guard_none(sem: ColumnSemantics) -> str | None:
    return None


def _shape_label(sem: ColumnSemantics) -> str:
    return {
        DistributionShape.LONG_TAIL: "长尾",
        DistributionShape.ZERO_INFLATED: "零膨胀",
        DistributionShape.BIMODAL: "双峰",
        DistributionShape.NORMAL: "近似正态",
        DistributionShape.UNIFORM: "均匀",
        DistributionShape.UNKNOWN: "未知形态",
    }[sem.shape]


#: 方法表（按 priority 降序作为自动选择顺序）。
#: 新增方法：在表里加一条 MethodSpec，不改任何调度代码。
METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        code="business_rule", label="业务口径", guard=_guard_none, priority=100,
        note="用户/业务定义的「什么算异常」，优先于一切统计口径",
    ),
    MethodSpec(
        code="low_frequency", label="低频类别检测", guard=_guard_low_frequency, priority=90,
        note="类别字段的异常表现为低频取值，而非数值越界",
    ),
    MethodSpec(
        code="quantile", label="分位数口径", guard=_guard_quantile, priority=60,
        note="不假设分布形态，天然适配非负与长尾字段",
    ),
    MethodSpec(
        code="mad", label="中位数绝对偏差(MAD)", guard=_guard_mad, priority=50,
        note="对极端值鲁棒，长尾/零膨胀字段的兜底方法",
    ),
    MethodSpec(
        code="iqr", label="IQR 四分位距", guard=_guard_iqr, priority=40,
        note="经典方法，仅在有符号且非长尾的连续字段上成立",
    ),
    MethodSpec(
        code="zscore", label="Z-score", guard=_guard_zscore, priority=30,
        note="要求近似正态分布",
    ),
    MethodSpec(
        code="none", label="不判定（仅描述）", guard=_guard_none, priority=0,
        note="目标列、常量列、标识列等：只描述不做清洗建议",
    ),
)

_SPEC_BY_CODE = {spec.code: spec for spec in METHOD_SPECS}


# --------------------------------------------------------------------------- #
# 第一步：选方法
# --------------------------------------------------------------------------- #


def select_outlier_method(
    sem: ColumnSemantics,
    *,
    requested: str = "iqr",
    business_rule: BusinessRule | None = None,
) -> Decision[str]:
    """为某个字段选择异常值判定方法。

    返回 :class:`Decision`，其中：
    * ``value`` 为最终方法 code；
    * ``source`` 为 ``business_rule`` / ``requested`` / ``auto_adapted`` / ``target_protected``；
    * 请求的方法不适用时，产出一条 ``method_mismatch`` 软警告并说明改用了什么、为什么。
    """
    # ① 业务口径最优先（业务口径 > 统计口径）
    rule = business_rule or (sem.stats.get("business_rule") if isinstance(sem.stats.get("business_rule"), dict) else None)
    if isinstance(rule, dict):
        rule = BusinessRule(
            column=rule.get("column", sem.name),
            lower=rule.get("lower"), upper=rule.get("upper"),
            strict_lower=bool(rule.get("strict_lower")),
            strict_upper=bool(rule.get("strict_upper")),
            allowed_values=rule.get("allowed_values"),
            note=str(rule.get("note") or ""),
        )
    if isinstance(rule, BusinessRule) and rule.active:
        return Decision.of(
            "business_rule", source="business_rule", confidence=1.0,
            reason=f"已登记业务口径，优先于统计口径：{rule.describe()}",
        )

    # ② 目标列保护：目标列只描述、不做清洗建议
    if sem.is_target:
        return Decision.of(
            "none", source="target_protected", confidence=1.0,
            reason=f"{sem.name} 是目标列，按约定只做描述、不产出异常值清洗建议",
        )

    # ③ 请求的方法可用 ⇒ 直接用
    spec = _SPEC_BY_CODE.get(requested)
    if spec is not None:
        reason = spec.check(sem)
        if reason is None:
            return Decision.of(
                requested, source="requested", confidence=0.9,
                reason=f"请求的方法 {spec.label} 适用于{sem.label}",
            )
        mismatch = finding(
            "quality.method_mismatch", Severity.WARN,
            f"请求的异常值方法「{spec.label}」不适用于 {sem.name}：{reason}",
            target=sem.name,
            evidence={"requested": requested, "role": str(sem.role),
                      "domain": str(sem.domain), "shape": str(sem.shape)},
            suggestion="已自动改用适配方法；若确需用该方法，请显式声明业务口径",
        )
        adapted = _auto_select(sem)
        decision = Decision.of(
            adapted, source="auto_adapted", confidence=0.8,
            reason=f"「{spec.label}」不适用（{reason}），自动改用「{_SPEC_BY_CODE[adapted].label}」",
            findings=[mismatch],
        )
        return decision

    # ④ 未知方法名 ⇒ 自动选择并警告
    return Decision.of(
        _auto_select(sem), source="auto_adapted", confidence=0.7,
        reason=f"未知的方法名 {requested!r}，已按字段语义自动选择",
        findings=[finding(
            "quality.unknown_method", Severity.WARN,
            f"未知的异常值方法 {requested!r}", target=sem.name,
            suggestion=f"可用方法：{', '.join(_SPEC_BY_CODE)}",
        )],
    )


def _auto_select(sem: ColumnSemantics) -> str:
    """按方法表优先级选出第一个适用方法（``none`` 保证兜底）。"""
    for spec in sorted(METHOD_SPECS, key=lambda s: -s.priority):
        if spec.code == "business_rule":
            continue
        if spec.check(sem) is None:
            return spec.code
    return "none"


# --------------------------------------------------------------------------- #
# 第二步：算边界
# --------------------------------------------------------------------------- #


def compute_bounds(
    sem: ColumnSemantics,
    series: pl.Series,
    *,
    method: str,
    k: float = 1.5,
    z_threshold: float = 3.0,
    low_frequency_ratio: float = 0.01,
    business_rule: BusinessRule | None = None,
) -> Decision[OutlierBounds]:
    """按选定方法计算边界。

    与旧实现的差别：边界**不是裸三元组**，而是带 ``adjusted`` / ``notes`` /
    ``findings`` 的 Decision —— 非负字段把负下界裁到 0 这件事会被明确记录，
    不会静默发生。
    """
    clean = series.drop_nulls()
    if clean.len() == 0:
        return Decision.of(
            None, source="empty", confidence=1.0, reason="该列无非空取值，无法判定",
        )

    if method == "none":
        return Decision.of(
            OutlierBounds(method="none", lower=None, upper=None,
                          notes=["按约定不做异常判定，仅描述"]),
            source="method_none", confidence=1.0,
            reason=f"{sem.name} 不适用异常值判定（目标列/常量列/标识列）",
        )

    if method == "business_rule":
        rule = business_rule or _rule_from_stats(sem)
        if rule is None or not rule.active:
            return Decision.of(
                None, source="business_rule_inactive", confidence=0.0,
                reason="选择了业务口径但未登记有效规则",
                findings=[finding("quality.business_rule_missing", Severity.CLARIFY,
                                  f"{sem.name} 未登记有效业务口径", target=sem.name,
                                  suggestion="请提供 lower/upper/allowed_values 之一")],
            )
        return Decision.of(
            OutlierBounds(
                method="business_rule",
                lower=rule.lower,
                upper=rule.upper,
                notes=[f"业务口径：{rule.describe()}"],
                info={"rule": rule.to_dict()},
            ),
            source="business_rule", confidence=1.0,
            reason=f"按业务口径判定：{rule.describe()}",
        )

    if method == "low_frequency":
        return _low_frequency_bounds(sem, clean, low_frequency_ratio)

    if method in ("iqr", "zscore"):
        # 复用 analysis 里已验证的实现，不重复造轮子（延迟导入避免循环依赖）
        from app.analysis import compute_outlier_bounds as _legacy_bounds

        lower, upper, info = _legacy_bounds(
            series, method=method, k=k, z_threshold=z_threshold
        )
        if info.get("empty") or info.get("constant"):
            return Decision.of(
                None, source="constant", confidence=1.0,
                reason=f"{sem.name} 为常数列或无有效取值，异常判定无意义",
            )
        return Decision.of(
            OutlierBounds(method=method, lower=float(lower), upper=float(upper), info=info),
            source=f"method_{method}", confidence=0.9,
            reason=f"按 {method} 计算边界 [{lower:.4g}, {upper:.4g}]",
        )

    if method == "mad":
        median = float(clean.median())
        mad = float((clean - median).abs().median())
        scale = 1.4826 * mad
        if scale == 0:
            return Decision.of(
                None, source="constant", confidence=1.0,
                reason=f"{sem.name} 的 MAD 为 0（取值高度集中），异常判定无意义",
            )
        lower, upper = median - k * scale, median + k * scale
        return Decision.of(
            OutlierBounds(method="mad", lower=lower, upper=upper,
                          info={"median": median, "mad": mad, "k": k}),
            source="method_mad", confidence=0.85,
            reason=f"按 MAD 计算边界 [{lower:.4g}, {upper:.4g}]（对长尾鲁棒）",
        )

    if method == "quantile":
        lo_q, hi_q = 0.001, 0.999
        lower = float(clean.quantile(lo_q))
        upper = float(clean.quantile(hi_q))
        notes: list[str] = []
        adjusted = False
        if sem.is_non_negative and lower < 0:
            lower = 0.0
            adjusted = True
            notes.append("字段为非负，下界裁剪至 0")
        if lower == upper:
            return Decision.of(
                None, source="constant", confidence=1.0,
                reason=f"{sem.name} 的分位数边界退化为单点，异常判定无意义",
            )
        return Decision.of(
            OutlierBounds(method="quantile", lower=lower, upper=upper,
                          adjusted=adjusted, notes=notes,
                          info={"lower_quantile": lo_q, "upper_quantile": hi_q}),
            source="method_quantile", confidence=0.8,
            reason=f"按分位数口径计算边界 [{lower:.4g}, {upper:.4g}]",
        )

    return Decision.of(
        None, source="unknown_method", confidence=0.0,
        reason=f"未实现的方法 {method!r}",
        findings=[finding("quality.unknown_method", Severity.WARN,
                          f"未实现的方法 {method!r}", target=sem.name)],
    )


def _rule_from_stats(sem: ColumnSemantics) -> BusinessRule | None:
    raw = sem.stats.get("business_rule")
    if not isinstance(raw, dict):
        return None
    return BusinessRule(
        column=raw.get("column", sem.name),
        lower=raw.get("lower"), upper=raw.get("upper"),
        strict_lower=bool(raw.get("strict_lower")),
        strict_upper=bool(raw.get("strict_upper")),
        allowed_values=raw.get("allowed_values"),
        note=str(raw.get("note") or ""),
    )


def _low_frequency_bounds(
    sem: ColumnSemantics, series: pl.Series, ratio: float
) -> Decision[OutlierBounds]:
    """低频类别：返回「占比低于阈值的取值」清单，而不是数值边界。"""
    try:
        vc = series.value_counts(sort=True)
        total = max(int(series.len()), 1)
        name_col = vc.columns[0]
        count_col = vc.columns[-1]
        rare = vc.filter(pl.col(count_col) / total < ratio)
        values = rare[name_col].to_list()
    except Exception:  # noqa: BLE001 - 退化：给不出明细也要给出方法结论
        values = []
    return Decision.of(
        OutlierBounds(
            method="low_frequency", lower=None, upper=None,
            info={"threshold_ratio": ratio, "rare_values": values[:20],
                  "rare_value_count": len(values)},
        ),
        source="method_low_frequency", confidence=0.8,
        reason=f"低频类别检测：{len(values)} 个取值占比低于 {ratio:.2%}",
    )


# --------------------------------------------------------------------------- #
# 组合入口
# --------------------------------------------------------------------------- #


def assess_column(
    sem: ColumnSemantics,
    series: pl.Series,
    *,
    requested: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    business_rule: BusinessRule | None = None,
) -> Decision[OutlierBounds]:
    """「选方法 + 算边界」一步完成，返回统一 Decision。"""
    choice = select_outlier_method(sem, requested=requested, business_rule=business_rule)
    if not choice.resolved or choice.value == "none":
        return Decision.of(
            OutlierBounds(method="none", lower=None, upper=None,
                          notes=list(choice.reasons)),
            source=choice.source, confidence=choice.confidence,
            reason="；".join(choice.reasons),
            findings=list(choice.findings),
        )
    bounds = compute_bounds(
        sem, series, method=choice.value, k=k,
        z_threshold=z_threshold, business_rule=business_rule,
    )
    # 方法选择阶段的警告（如 method_mismatch）必须跟着边界一起回传，
    # 否则「改用了别的方法」这件事在下游就消失了。
    bounds.findings = list(choice.findings) + list(bounds.findings)
    bounds.reasons = list(choice.reasons) + list(bounds.reasons)
    return bounds


def should_suggest_cleaning(sem: ColumnSemantics) -> bool:
    """该字段是否应进入「清洗建议」。

    目标列永远返回 False——目标列的极端值是要预测的对象，不是脏数据。
    """
    if sem.is_target:
        return False
    if sem.role in (ColumnRole.CONSTANT, ColumnRole.IDENTIFIER):
        return False
    if sem.role == ColumnRole.TEMPORAL and not sem.pseudo_time_format:
        return False
    return True
