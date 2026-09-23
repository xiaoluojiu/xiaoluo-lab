"""ML Engine · 评估指标顾问（第四层改造）。

改造前的问题
------------
平台只报 ``mae / mse / rmse / r2``。对 ``DepDelay`` 这种**零膨胀长尾目标**
（中位数 0，标准差是均值的 3.6 倍）这套指标几乎不说明问题：

* r² = 0.02 会被读成「模型没用」，但它只说明**线性关系弱**，
  换树模型或换指标后可能完全够用；
* RMSE 被少数极端延误主导，对「大多数准点航班」的预测质量不敏感；
* 业务真正关心的「会不会延误超过 15 分钟」根本没被度量。

本模块把「用什么指标」也变成一次**带依据的判定**：

    recommend_metrics(task, target_sem) -> Decision[list[MetricAdvice]]

并补上三件原先没有的事：

* :func:`stratified_regression_report` —— 按目标分位数分段报误差，
  回答「在延误严重的航班上到底准不准」；
* :func:`business_metric_report` —— 业务指标（如「延误 > 15 分钟」命中率）
  与统计指标并列；
* :func:`explain_metric` —— 给 Agent 用的指标解释，避免把 r² 低读成模型无用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from app.core.contracts import Decision, Severity, finding
from app.quality.semantics import ColumnSemantics, DistributionShape

__all__ = [
    "METRIC_SPECS",
    "BusinessMetric",
    "MetricAdvice",
    "MetricSpec",
    "business_metric_report",
    "explain_metric",
    "recommend_metrics",
    "stratified_regression_report",
]


@dataclass
class MetricAdvice:
    code: str
    label: str
    priority: str = "建议"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "label": self.label,
                "priority": self.priority, "reason": self.reason}


@dataclass(frozen=True)
class MetricSpec:
    """一个指标的适用声明（写法与 Quality 层的方法表一致）。"""

    code: str
    label: str
    task: str
    #: (目标列语义, 类别分布) -> None 表示适用，str 表示**不适用及原因**
    applies: Callable[[ColumnSemantics | None, dict[str, Any] | None], str | None]
    priority: int = 0
    note: str = ""


def _regression_guard(sem: ColumnSemantics | None, ctx: dict[str, Any] | None) -> str | None:
    if sem is None:
        return None
    if sem.role is not None and sem.shape == DistributionShape.UNKNOWN:
        return None
    return None


METRIC_SPECS: tuple[MetricSpec, ...] = (
    # ---- 回归：长尾 / 零膨胀 -------------------------------------------
    MetricSpec(
        code="mae", label="MAE（平均绝对误差）", task="regression", priority=90,
        applies=lambda sem, ctx: None,
        note="对所有样本等权，不被极端值主导；长尾/零膨胀目标的主指标",
    ),
    MetricSpec(
        code="quantile_loss", label="分位数损失（pinball）", task="regression", priority=88,
        applies=lambda sem, ctx: (
            None if sem is None or sem.shape in (
                DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED,
                DistributionShape.UNKNOWN,
            ) else f"{sem.name} 分布形态为 {sem.shape}，分位数损失收益有限"
        ),
        note="直接度量分位数预测质量，是长尾目标最诚实的指标",
    ),
    MetricSpec(
        code="hit_rate", label="业务阈值命中率", task="regression", priority=85,
        applies=lambda sem, ctx: (
            None if (ctx or {}).get("business_threshold") is not None
            else "未定义业务阈值（如「延误 > 15 分钟」），跳过命中率"
        ),
        note="把回归结果换算成业务关心的二分类判定准确率",
    ),
    MetricSpec(
        code="stratified_mae", label="分层 MAE（按目标分位数分段）", task="regression", priority=80,
        applies=lambda sem, ctx: (
            None if sem is None or sem.shape in (
                DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED,
            ) else f"{sem.name} 非长尾目标，分层收益有限"
        ),
        note="回答「在取值大的那部分样本上到底准不准」——总量指标会掩盖它",
    ),
    MetricSpec(
        code="rmse", label="RMSE", task="regression", priority=60,
        applies=lambda sem, ctx: (
            None if sem is None or sem.shape in (
                DistributionShape.NORMAL, DistributionShape.UNKNOWN,
            ) else f"{sem.name} 为{sem.shape}分布，RMSE 被极端值主导，不宜作主指标"
        ),
        note="对大误差敏感；只在近似正态下适合作主指标",
    ),
    MetricSpec(
        code="r2", label="R²", task="regression", priority=40,
        applies=lambda sem, ctx: None,
        note="解释方差比例；**低 R² 不等于模型无用**，需与 MAE、业务指标并列解读",
    ),
    # ---- 分类 ------------------------------------------------------------
    MetricSpec(
        code="auc", label="AUC", task="classification", priority=90,
        applies=lambda sem, ctx: None,
        note="对阈值不敏感，类别不平衡下的主指标",
    ),
    MetricSpec(
        code="f1", label="F1", task="classification", priority=85,
        applies=lambda sem, ctx: None,
        note="精确率与召回率的调和平均",
    ),
    MetricSpec(
        code="recall", label="召回率", task="classification", priority=88,
        applies=lambda sem, ctx: (
            None if (ctx or {}).get("imbalanced") is not False
            else "类别均衡，召回率不是关键指标"
        ),
        note="不平衡场景下漏报代价高时优先看它",
    ),
    MetricSpec(
        code="macro_f1", label="宏平均 F1", task="classification", priority=70,
        applies=lambda sem, ctx: (
            None if int((ctx or {}).get("n_classes") or 0) > 2
            else "二分类场景，宏平均与 F1 等价"
        ),
        note="多分类下对每个类别等权",
    ),
    # ---- 聚类 ------------------------------------------------------------
    MetricSpec(
        code="silhouette", label="轮廓系数", task="clustering", priority=80,
        applies=lambda sem, ctx: None,
        note="无标签时的分群质量；计算 O(n²)，大样本需采样",
    ),
)


def recommend_metrics(
    task: str,
    target_sem: ColumnSemantics | None = None,
    *,
    business_threshold: float | None = None,
    class_counts: dict[str, int] | None = None,
) -> Decision[list[MetricAdvice]]:
    """按任务类型 + 目标分布推荐评估指标。"""
    ctx: dict[str, Any] = {"business_threshold": business_threshold}
    if class_counts:
        counts = sorted(int(v) for v in class_counts.values())
        ctx["n_classes"] = len(counts)
        ctx["imbalanced"] = bool(counts) and len(counts) > 1 and counts[0] / max(counts[-1], 1) < 0.2

    advices: list[MetricAdvice] = []
    warnings: list[Finding] = []
    for spec in sorted(METRIC_SPECS, key=lambda s: -s.priority):
        if spec.task != task:
            continue
        reason = spec.applies(target_sem, ctx)
        if reason is not None:
            continue
        priority = "必须" if spec.priority >= 88 else ("建议" if spec.priority >= 60 else "可选")
        advices.append(MetricAdvice(spec.code, spec.label, priority, spec.note))

    if task == "regression" and target_sem is not None and target_sem.shape in (
        DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED,
    ):
        warnings.append(finding(
            "metric.rmse_misleading", Severity.WARN,
            f"目标列 {target_sem.name} 为{target_sem.shape}分布：RMSE / R² 会被少数极端值主导，"
            "不应作为唯一结论依据。",
            target=target_sem.name,
            suggestion="以 MAE 为主，配合分位数损失与业务阈值命中率；必要时分段报告",
        ))

    return Decision.of(
        advices, source="distribution", confidence=0.85,
        reason=f"按 task={task} 与目标分布形态推荐 {len(advices)} 个指标",
        findings=list(warnings),
    )


def explain_metric(code: str, *, target_sem: ColumnSemantics | None = None) -> str:
    """给 Agent / 报告用的一句话指标解释。

    存在意义：把「r²=0.02 ⇒ 模型没用」这种误读挡住 ——
    低 R² 只说明线性解释力弱，业务可用性要看 MAE 与命中率。
    """
    base = {
        "r2": "R² 是被解释方差的比例：低值只说明**线性关系弱**，不等于模型不可用；"
              "需与 MAE、业务阈值命中率一起看。",
        "rmse": "RMSE 对大误差平方加权：长尾目标上会被少数极端值主导，"
                "不能单独用来判断模型好坏。",
        "mae": "MAE 对所有样本等权，是长尾/零膨胀目标上最稳健的主指标。",
        "quantile_loss": "分位数损失直接度量分位数预测质量，长尾目标的首选。",
        "hit_rate": "命中率把回归结果换算成业务判定（如是否超阈值），最贴近决策。",
        "stratified_mae": "分层 MAE 按目标取值分段报告，回答「大值区间准不准」。",
        "auc": "AUC 与阈值无关，类别不平衡时的主指标。",
        "f1": "F1 平衡精确率与召回率。",
        "recall": "召回率衡量漏报，代价不对称场景优先看它。",
        "silhouette": "轮廓系数衡量分群紧密度，无标签时的参考值。",
    }.get(code, "")
    if target_sem is not None and code in ("r2", "rmse") and target_sem.shape in (
        DistributionShape.LONG_TAIL, DistributionShape.ZERO_INFLATED,
    ):
        base += f" 当前目标 {target_sem.name} 为{target_sem.shape}分布，该指标尤其需要谨慎解读。"
    return base


# --------------------------------------------------------------------------- #
# 分层评估
# --------------------------------------------------------------------------- #


def stratified_regression_report(
    y_true: "pl.Series | list[float]",
    y_pred: "pl.Series | list[float]",
    *,
    segments: int = 5,
) -> list[dict[str, Any]]:
    """按目标值分位数分段报误差。

    总量指标会掩盖「模型在大值区间完全失效」：延误数据集里绝大多数航班准点，
    整体 MAE 漂亮，但真正被延误的那一小撮可能预测得一塌糊涂。
    """
    true = y_true if isinstance(y_true, pl.Series) else pl.Series("y_true", list(y_true), dtype=pl.Float64)
    pred = y_pred if isinstance(y_pred, pl.Series) else pl.Series("y_pred", list(y_pred), dtype=pl.Float64)
    if true.len() == 0 or true.len() != pred.len():
        return []
    df = pl.DataFrame({"y": true.cast(pl.Float64), "p": pred.cast(pl.Float64)}).drop_nulls()
    if df.height == 0:
        return []
    seg = min(max(int(segments), 2), df.height)
    edges = [float(df["y"].quantile(i / seg)) for i in range(seg + 1)]
    out: list[dict[str, Any]] = []
    for i in range(seg):
        lo, hi = edges[i], edges[i + 1]
        part = df.filter(
            (pl.col("y") >= lo) & (pl.col("y") <= hi if i == seg - 1 else pl.col("y") < hi)
        )
        if part.height == 0:
            continue
        err = (part["p"] - part["y"]).abs()
        out.append(
            {
                "segment": i + 1,
                "range": [round(lo, 6), round(hi, 6)],
                "count": int(part.height),
                "mae": round(float(err.mean()), 6),
                "rmse": round(float(((part["p"] - part["y"]) ** 2).mean() ** 0.5), 6),
                "bias": round(float((part["p"] - part["y"]).mean()), 6),
            }
        )
    return out


@dataclass
class BusinessMetric:
    """一条业务指标定义（如「延误 > 15 分钟」）。"""

    code: str
    label: str
    threshold: float
    direction: str = "greater"  # greater / less

    def mask(self, values: "pl.Series") -> "pl.Series":
        v = values.cast(pl.Float64)
        return v > self.threshold if self.direction == "greater" else v < self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "label": self.label,
                "threshold": self.threshold, "direction": self.direction}


def business_metric_report(
    y_true: "pl.Series | list[float]",
    y_pred: "pl.Series | list[float]",
    metrics: list[BusinessMetric],
) -> list[dict[str, Any]]:
    """业务指标与统计指标并列报告。"""
    true = y_true if isinstance(y_true, pl.Series) else pl.Series("y_true", list(y_true), dtype=pl.Float64)
    pred = y_pred if isinstance(y_pred, pl.Series) else pl.Series("y_pred", list(y_pred), dtype=pl.Float64)
    if true.len() == 0 or true.len() != pred.len() or not metrics:
        return []
    total = int(true.len())
    out: list[dict[str, Any]] = []
    for m in metrics:
        actual = m.mask(true)
        predicted = m.mask(pred)
        actual_n = int(actual.sum())
        predicted_n = int(predicted.sum())
        hits = int((actual & predicted).sum())
        out.append(
            {
                **m.to_dict(),
                "actual_count": actual_n,
                "actual_rate": round(actual_n / total, 6),
                "predicted_count": predicted_n,
                "predicted_rate": round(predicted_n / total, 6),
                "hit_count": hits,
                "recall": round(hits / actual_n, 6) if actual_n else None,
                "precision": round(hits / predicted_n, 6) if predicted_n else None,
            }
        )
    return out
