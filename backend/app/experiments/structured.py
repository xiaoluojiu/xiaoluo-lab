"""Experiments · 实验记录的结构化与解读（第五层改造）。

改造前的问题
------------
失败实验 #29 只留一个 ``failed`` 状态，没人说清**为什么失败**、
**该怎么改**；成功实验 #30（kmeans）定位模糊——它到底验证什么、结论是什么、
下一步该跑什么，全靠人从参数里猜。

本模块把实验记录标准化为一份可机读、可渲染的结构：

    ExperimentNarrative(hypothesis, result, conclusion, next_action, failure_reason)

三块能力
--------
1. :func:`classify_failure` —— 把一条报错文本归类成**稳定的失败码** +
   **可操作的修复建议**（失败码走注册表，新增一种失败只加一条记录）；
2. :func:`conclude` —— 从多个 Run 的对比里读出「哪个改动带来了提升」，
   而不是罗列一张指标表；
3. :func:`build_narrative` —— 把实验 + 它的 Run 组装成上面那条结构，
   供报告层直接消费。

设计约束
--------
* 失败原因**必须由报错文本推导**，不允许 LLM 自由发挥；
* 结论必须能追溯到具体 run_id 与指标差值（可复核）；
* DB 字段扩展（hypothesis / conclusion / next_action / failure_reason 落到
  ``experiments`` 表）属于后续事项，本模块先把**纯逻辑**标准化，
  落库时直接复用这里的字段名，避免届时再定一遍口径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.contracts import Decision

__all__ = [
    "FAILURE_PATTERNS",
    "ExperimentNarrative",
    "FailureDiagnosis",
    "REQUIRED_STRUCTURED_FIELDS",
    "build_narrative",
    "classify_failure",
    "conclude",
]


#: 后续落库时使用的字段名（先在这里定死口径，避免迁移时再定一遍）
REQUIRED_STRUCTURED_FIELDS: tuple[str, ...] = (
    "hypothesis",      # 想验证什么
    "conclusion",      # 结果说明什么
    "next_action",     # 下一步做什么
    "failure_reason",  # 失败的具体原因（含修复建议）
)


# --------------------------------------------------------------------------- #
# 失败归因
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FailurePattern:
    code: str
    title: str
    pattern: re.Pattern[str]
    fix: str


#: 失败模式表（**唯一真源**）：新增一类失败只加一条记录。
#: 全部来自本项目真实出现过的报错。
FAILURE_PATTERNS: tuple[FailurePattern, ...] = (
    FailurePattern(
        code="target_in_excluded",
        title="目标列被列入排除清单",
        pattern=re.compile(r"target[_\s-]?column.{0,20}(不能|不可).{0,10}excluded|excluded_columns.{0,30}target", re.I),
        fix="把目标列从 excluded_columns 里移除（平台已改为自动剔除并回写 preprocessing，"
            "若仍失败请检查调用方是否覆盖了自动收敛结果）",
    ),
    FailurePattern(
        code="missing_target",
        title="目标列缺失或无法推断",
        pattern=re.compile(r"needs_target|未能按命名约定确定目标列|缺少必要参数.{0,10}target|target.{0,10}不在数据中", re.I),
        fix="显式指定 target；或让数据集名称/用户诉求带上预测对象语义，由推断层自动解析",
    ),
    FailurePattern(
        code="model_param_mismatch",
        title="模型与参数不匹配",
        pattern=re.compile(r"unexpected keyword argument|__init__\(\) got", re.I),
        fix="换模型时旧参数会残留：按新模型的参数表裁剪（平台已加 _prune_params_for_model，"
            "被丢弃的参数记在 model_adjusted.dropped_params）",
    ),
    FailurePattern(
        code="model_not_registered",
        title="模型未注册",
        pattern=re.compile(r"模型.{0,10}未注册|not registered", re.I),
        fix="换用 MODEL_REGISTRY 中已注册的模型名",
    ),
    FailurePattern(
        code="memory_overflow",
        title="内存不足",
        pattern=re.compile(r"unable to allocate|memoryerror|GiB|MemoryError", re.I),
        fix="高基数列不要 one-hot（改频次/目标编码）；调小 ML_MAX_TRAIN_ROWS；"
            "或提高 ML_MAX_DENSE_BYTES。列数是关键线索：先确认是否被 one-hot 放大了几十倍",
    ),
    FailurePattern(
        code="stratify_on_regression",
        title="回归任务被按分类分层",
        pattern=re.compile(r"测试集样本.{0,10}不足以覆盖|类别.{0,10}stratif", re.I),
        fix="分层只在 classification 下启用；回归目标取值档位多不代表它是分类",
    ),
    FailurePattern(
        code="column_not_found",
        title="列不存在",
        pattern=re.compile(r"目标列不存在于数据版本中|not found|不在数据集中", re.I),
        fix="列名可能来自上一个数据版本：确认 dataset.schema 后再填，"
            "或用 dataset.inspect 拿当前版本的真实列名",
    ),
    FailurePattern(
        code="silently_clustered",
        title="监督任务被降级为聚类",
        pattern=re.compile(r"静默聚类|默认按聚类|退化成聚类", re.I),
        fix="这是默认值害人的典型：监督任务必须显式拿到 target，不能用聚类顶替",
    ),
)


@dataclass
class FailureDiagnosis:
    code: str
    title: str
    message: str
    fix: str
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "title": self.title,
            "message": self.message, "fix": self.fix, "matched": self.matched,
        }


def classify_failure(message: str) -> Decision[FailureDiagnosis]:
    """把一条报错文本归类成稳定失败码 + 修复建议。"""
    text = str(message or "").strip()
    if not text:
        return Decision.of(None, source="empty", confidence=0.0, reason="无报错文本")
    for item in FAILURE_PATTERNS:
        if item.pattern.search(text):
            return Decision.of(
                FailureDiagnosis(item.code, item.title, text, item.fix, matched=True),
                source="pattern", confidence=0.9,
                reason=f"命中失败模式 {item.code}（{item.title}）",
            )
    return Decision.of(
        FailureDiagnosis("unknown", "未分类失败", text, "查看 run.error 原始报错与 trace 日志定位", matched=False),
        source="fallback", confidence=0.3,
        reason="未命中任何已知失败模式",
    )


# --------------------------------------------------------------------------- #
# 结论
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentNarrative:
    """一次实验的完整叙述（结构化字段）。"""

    experiment_id: int | None = None
    task_type: str = ""
    target_column: str = ""
    #: 想验证什么
    hypothesis: str = ""
    #: 结果说明什么
    conclusion: str = ""
    #: 下一步做什么
    next_action: str = ""
    #: 失败的具体原因（成功时为空）
    failure_reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "hypothesis": self.hypothesis,
            "conclusion": self.conclusion,
            "next_action": self.next_action,
            "failure_reason": self.failure_reason,
            "evidence": dict(self.evidence),
        }


def _fmt_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "—"
    delta = b - a
    return f"{delta:+.4f}"


def conclude(compare: Any, *, primary_metric: str | None = None) -> dict[str, Any]:
    """从对比结果里读出「哪个改动带来了提升」。

    返回结构：
    ``{best_run_id, primary_metric, delta_vs_baseline, driver_changes, note}``
    其中 ``driver_changes`` 是**最优 run 相对基线唯一不同的参数**——
    这才是「哪个改动带来了提升」的答案，而不是把所有参数都列一遍。
    """
    entries = list(getattr(compare, "entries", []) or [])
    if not entries:
        return {"best_run_id": None, "primary_metric": None, "note": "没有可对比的 Run"}
    best_by_metric = dict(getattr(compare, "best", {}) or {})
    param_diff = dict(getattr(compare, "parameter_diff", {}) or {})

    metric = primary_metric
    if not metric:
        for candidate in ("r2", "accuracy", "f1", "auc", "silhouette"):
            if candidate in best_by_metric:
                metric = candidate
                break
    best_run_id = best_by_metric.get(metric) if metric else None

    baseline = next((e for e in entries if getattr(e, "status", "") == "success"), None)
    best_entry = next((e for e in entries if getattr(e, "run_id", None) == best_run_id), None)
    if best_entry is None or baseline is None or best_entry is baseline:
        return {
            "best_run_id": best_run_id, "primary_metric": metric,
            "delta_vs_baseline": None, "driver_changes": {},
            "note": "只有一个成功 Run，缺少对照，无法归因改动效果",
        }

    base_metrics = dict(getattr(baseline, "metrics", {}) or {})
    best_metrics = dict(getattr(best_entry, "metrics", {}) or {})
    delta = _fmt_delta(
        base_metrics.get(metric) if isinstance(base_metrics.get(metric), (int, float)) else None,
        best_metrics.get(metric) if isinstance(best_metrics.get(metric), (int, float)) else None,
    )
    drivers: dict[str, dict[str, Any]] = {}
    for key, values in param_diff.items():
        before = values.get(getattr(baseline, "run_id", None))
        after = values.get(getattr(best_entry, "run_id", None))
        if repr(before) != repr(after):
            drivers[key] = {"before": before, "after": after}
    return {
        "best_run_id": best_run_id,
        "primary_metric": metric,
        "delta_vs_baseline": delta,
        "driver_changes": drivers,
        "note": (
            f"最优 Run #{best_run_id} 在 {metric} 上相对基线 {delta}；"
            + (f"差异参数为 {', '.join(sorted(drivers))}" if drivers else "两次运行参数一致，差异来自数据/随机性")
        ),
    }


def build_narrative(experiment: Any, runs: list[Any] | None = None, *, compare: Any = None) -> ExperimentNarrative:
    """把实验 + 它的 Run 组装成结构化叙述。"""
    runs = list(runs or [])
    failed = [r for r in runs if str(getattr(r, "status", "")) != "success"]
    succeeded = [r for r in runs if str(getattr(r, "status", "")) == "success"]

    target = str(getattr(experiment, "target_column", "") or "")
    task_type = str(getattr(experiment, "task_type", "") or "")
    narrative = ExperimentNarrative(
        experiment_id=getattr(experiment, "id", None),
        task_type=task_type,
        target_column=target,
        hypothesis=_hypothesis_of(experiment, task_type, target),
    )

    if not succeeded and failed:
        diagnosis = classify_failure(str(getattr(failed[-1], "error", "") or ""))
        diag = diagnosis.value
        narrative.failure_reason = (
            f"[{diag.code}] {diag.title}：{diag.fix}" if diag else "未知失败原因"
        )
        narrative.conclusion = "本次实验未产出可用结果，需按修复建议调整后重跑。"
        narrative.next_action = diag.fix if diag else "查看原始报错定位"
        narrative.evidence = {"failed_run_id": getattr(failed[-1], "id", None),
                              "diagnosis": diag.to_dict() if diag else None}
        return narrative

    conclusion = conclude(compare) if compare is not None else {}
    narrative.conclusion = str(conclusion.get("note") or f"完成 {len(succeeded)} 次成功运行。")
    narrative.next_action = _next_action_of(experiment, task_type, target, conclusion)
    narrative.evidence = {
        "success_runs": [getattr(r, "id", None) for r in succeeded],
        "failed_runs": [getattr(r, "id", None) for r in failed],
        "conclusion": conclusion,
    }
    return narrative


def _hypothesis_of(experiment: Any, task_type: str, target: str) -> str:
    if not task_type:
        return "（未标注任务类型：实验命名规范要求每个实验都有明确的 task_type 与 target_column）"
    if task_type == "clustering":
        return f"验证该数据集是否能按现有特征分出有意义的簇（无目标列 {target or '—'}）"
    if not target:
        return f"验证 {task_type} 任务在当前特征下的可达效果（目标列未标注）"
    return f"验证「用现有特征预测 {target}」的可达效果"


def _next_action_of(experiment: Any, task_type: str, target: str, conclusion: dict[str, Any]) -> str:
    metric = conclusion.get("primary_metric")
    if task_type in ("regression", "classification") and target:
        return (
            f"按特征工程建议（时间字段解析 / 高基数编码）改造特征后重跑 {task_type}，"
            f"主指标看 {metric or 'MAE/AUC'}，并用分层误差复核大值区间"
        )
    if task_type == "clustering":
        return "先确认业务是否真的没有目标列；若有，应改为监督任务而不是分群"
    return "补齐 task_type 与 target_column 后重跑，保证实验可解释"
