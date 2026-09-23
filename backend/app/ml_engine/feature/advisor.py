"""ML Engine · 特征工程建议器（第三层改造）。

产出**建议列表**而不是自动执行 —— 特征工程会改变数据语义（目标编码引入泄漏风险、
共线性处置会删列），这类决定必须可见、可审计、可撤销。

建议的优先级三档，与报告层「必须做 / 建议做 / 可选做」同一套词表：

* ``必须`` —— 不做会出错（HHMM 伪数值直接入模、高基数列 one-hot 撑爆内存）
* ``建议`` —— 做了解释性/效果更好（循环编码、共线性处置）
* ``可选`` —— 锦上添花（交叉特征、分组统计）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.contracts import Decision, Severity, finding
from app.ml_engine.feature.registry import FEATURE_OP_REGISTRY
from app.quality.semantics import ColumnRole, ColumnSemantics, DistributionShape

__all__ = ["FeatureAdvice", "PRIORITY_ORDER", "recommend_features"]

#: 优先级词表（报告层共用）
PRIORITY_ORDER = ("必须", "建议", "可选")

_HIGH_CARDINALITY = 50
_COLLINEARITY_THRESHOLD = 0.75


@dataclass
class FeatureAdvice:
    """一条特征工程建议。"""

    op: str
    column: str
    priority: str = "建议"
    reason: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    #: 该操作是否会引入目标泄漏（需 out-of-fold）
    leaky: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "column": self.column,
            "priority": self.priority,
            "reason": self.reason,
            "params": dict(self.params),
            "leaky": self.leaky,
        }


def recommend_features(
    semantics: dict[str, ColumnSemantics],
    *,
    target: str | None = None,
    corr_pairs: list[dict[str, Any]] | None = None,
    threshold: float = _COLLINEARITY_THRESHOLD,
) -> Decision[list[FeatureAdvice]]:
    """基于字段语义推荐该做的特征工程。

    Args:
        semantics: ``infer_semantics`` 的产物。
        target: 目标列名（有它才建议目标编码 / 分组统计）。
        corr_pairs: 相关性结果 ``[{"a": str, "b": str, "r": float}]``。
        threshold: 共线性阈值。
    """
    advices: list[FeatureAdvice] = []
    reasons: list[str] = []
    target_sem = semantics.get(target) if target else None

    for name, sem in semantics.items():
        # 目标列本身不做特征工程
        if sem.is_target:
            continue

        # ① 伪数值时间：必须解析
        op = FEATURE_OP_REGISTRY.try_get("time.parse_hhmm")
        if op is not None and op.check(sem) is None:
            advices.append(FeatureAdvice(
                op="time.parse_hhmm", column=name, priority="必须",
                reason=(
                    f"{name} 是 HHMM 伪数值时间（{sem.pseudo_time_format}）："
                    "2359→0000 存在数值断崖，直接当连续量入模会引入虚假的负相关。"
                ),
            ))

        # ② 周期性字段：建议循环编码
        op = FEATURE_OP_REGISTRY.try_get("time.cyclic")
        if op is not None and op.check(sem) is None and sem.pseudo_time_format is None:
            advices.append(FeatureAdvice(
                op="time.cyclic", column=name, priority="建议",
                reason=f"{name} 为周期性字段（{sem.cardinality} 个取值），"
                       "循环编码可消除周期边界跳变",
            ))

        # ③ 高基数类别：必须换掉 one-hot
        op = FEATURE_OP_REGISTRY.try_get("encode.frequency")
        if (
            sem.role == ColumnRole.CATEGORICAL
            and sem.cardinality > _HIGH_CARDINALITY
            and op is not None
            and op.check(sem) is None
        ):
            advices.append(FeatureAdvice(
                op="encode.frequency", column=name, priority="必须",
                reason=(
                    f"{name} 有 {sem.cardinality} 个取值：one-hot 会把它展开成 "
                    f"{sem.cardinality} 列（本次 3 个高基数列曾把 10 列放大成 767 列、"
                    "触发 57.1 GiB 内存申请）。改用频次编码，列数只 +1。"
                ),
            ))
            if target_sem is not None:
                advices.append(FeatureAdvice(
                    op="encode.target", column=name, priority="建议", leaky=True,
                    reason=f"{name} 可用目标编码替代频次编码（信息量更大），"
                           "但必须 out-of-fold：统计量只能在训练折内拟合",
                    params={"target": target, "folds": "<切分后传入>"},
                ))

        # ④ 共线性（连续列与时间列都可能是共线的一方：CRSDepTime / CRSArrTime）
        if corr_pairs and sem.role in (ColumnRole.CONTINUOUS, ColumnRole.TEMPORAL):
            partners = [
                p for p in corr_pairs
                if (p.get("a") == name or p.get("b") == name)
                and p.get("r") is not None
                and abs(float(p["r"])) >= threshold
            ]
            if partners:
                detail = "；".join(
                    f"与 {p['b'] if p['a'] == name else p['a']} 的 r={float(p['r']):.2f}"
                    for p in partners
                )
                advices.append(FeatureAdvice(
                    op="collinearity.flag", column=name, priority="建议",
                    reason=f"{name} 存在共线性（阈值 {threshold:.2f}）：{detail}。"
                           "可选处置：保留其一 / 构造差值 / PCA",
                    params={"threshold": threshold, "pairs": partners[:5]},
                ))

    # ⑤ 长尾目标：提示对数变换/分位数损失（交给第四层指标顾问复核）
    if target_sem is not None and target_sem.shape == DistributionShape.LONG_TAIL:
        reasons.append(
            f"目标列 {target_sem.name} 为长尾分布：可考虑 log1p 变换，"
            "并用分位数损失替代 RMSE（详见指标顾问）"
        )

    advices.sort(key=lambda a: (PRIORITY_ORDER.index(a.priority), a.column))
    if not advices:
        return Decision.of(
            [], source="no_match", confidence=0.6,
            reason="按字段语义未发现需要处理的特征工程项",
        )
    must = [a for a in advices if a.priority == "必须"]
    findings = []
    if must:
        findings.append(finding(
            "feature.required_missing", Severity.WARN,
            f"有 {len(must)} 项特征工程属于「必须做」："
            + "、".join(f"{a.column}→{a.op}" for a in must[:5]),
            suggestion="不做会出错（伪数值时间误当连续量 / 高基数列撑爆内存）",
        ))
    return Decision.of(
        advices, source="semantics", confidence=0.85,
        reason="；".join(reasons) or f"按字段语义产出 {len(advices)} 条特征工程建议",
        findings=findings,
    )
