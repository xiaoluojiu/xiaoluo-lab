"""Agent 层 · Pre-flight Check（第一层改造，最高优先级）。

改造前的问题
------------
「航空公司出发延误预测」数据集（连续目标 DepDelay）在用户未指定目标列时，
平台默认判为 **clustering** 并实跑 KMeans。这不是某一处写错，而是**整条链路
没有任何一个环节在动手之前问一句「你到底要预测什么」**：

    用户请求 → 路由 → 规划（看不到列名）→ ml.detect_task（只列候选）→ 训练
                                          ↑ 死锁在这里，最终静默退化成聚类

本模块在 Planner **之前**插入一道确定性检查，输出三种结果：

============  ==========================================================
PROCEED       信息齐全，继续规划
CLARIFY       缺信息，但已生成结构化反问，等用户回答后继续（不重跑已完成的步骤）
BLOCK         信息矛盾，继续必错，必须由用户裁定
============  ==========================================================

设计约束（与其他层对齐）
------------------------
* **只做无需读数据的判定**：输入是 schema 级元数据 + 列名 + 数据集名称，
  不做全表扫描（千万行表上 n_unique 都嫌贵）。需要分布才能判的检查留给工具层；
* **检查项可注册**：每个检查是一个 :class:`PreflightCheck`，注册进
  :data:`CHECK_REGISTRY`（复用 ``app.core.registry.Registry``），
  新增检查不改调度代码；
* **产出统一契约**：检查产出 :class:`~app.core.contracts.Finding`，
  反问由 ``app.agent.clarify.render`` 按 code 渲染 —— 两者只靠 code 耦合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from app.agent.clarify import Clarification, render
from app.agent.intent import is_modeling
from app.core.contracts import Decision, Finding, Severity, finding, merge_findings
from app.core.registry import Registry

__all__ = [
    "CHECK_REGISTRY",
    "PreflightCheck",
    "PreflightInput",
    "PreflightOutcome",
    "PreflightResult",
    "run_preflight",
]


class PreflightOutcome(StrEnum):
    PROCEED = "proceed"
    CLARIFY = "clarify"
    BLOCK = "block"


@dataclass
class PreflightInput:
    """Pre-flight 的输入（全部为低成本元数据）。"""

    user_request: str = ""
    intent: Decision[Any] | None = None
    dataset_ids: list[int] = field(default_factory=list)
    #: dataset_id -> {name, rows, columns}
    dataset_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: 列清单：[{name, dtype, n_unique?}]；拿不到时留空，相关检查自动跳过
    columns: list[dict[str, Any]] = field(default_factory=list)
    target: str | None = None
    task_type: str | None = None
    #: 已回答过的反问（code -> answer），用于避免重复追问
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def column_names(self) -> list[str]:
        return [str(c.get("name")) for c in self.columns if c.get("name")]

    def meta_of(self, dataset_id: int | None = None) -> dict[str, Any]:
        if dataset_id is None:
            return next(iter(self.dataset_meta.values()), {}) if self.dataset_meta else {}
        return self.dataset_meta.get(int(dataset_id), {})


@dataclass
class PreflightResult:
    outcome: PreflightOutcome = PreflightOutcome.PROCEED
    findings: list[Finding] = field(default_factory=list)
    clarifications: list[Clarification] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    #: 已确认的事实（如推断出的目标列），供规划器直接使用，避免重复推断
    resolved: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.BLOCK]

    @property
    def needs_user(self) -> bool:
        return self.outcome in (PreflightOutcome.BLOCK, PreflightOutcome.CLARIFY)

    def first_question(self) -> Clarification | None:
        return self.clarifications[0] if self.clarifications else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "checks_run": list(self.checks_run),
            "findings": [f.to_dict() for f in self.findings],
            "clarifications": [c.to_dict() for c in self.clarifications],
            "resolved": dict(self.resolved),
        }


@dataclass(frozen=True)
class PreflightCheck:
    """一个 Pre-flight 检查项。

    ``run`` 返回 ``(findings, resolved_updates)``：前者是问题清单，
    后者是「这次检查顺带确认下来的事实」（如推断出的目标列），
    会被合并进 :attr:`PreflightResult.resolved` 供规划器复用。
    """

    code: str
    title: str
    run: Callable[[PreflightInput], tuple[list[Finding], dict[str, Any]]]
    #: 该检查是否需要列信息（无列信息时跳过）
    requires_columns: bool = False
    #: 是否仅在建模类意图下运行
    modeling_only: bool = False


CHECK_REGISTRY: Registry[PreflightCheck] = Registry(label="preflight 检查项")


def check(code: str, *, title: str, requires_columns: bool = False, modeling_only: bool = False):
    def _wrap(fn: Callable[[PreflightInput], tuple[list[Finding], dict[str, Any]]]):
        CHECK_REGISTRY.register(
            code,
            PreflightCheck(
                code=code, title=title, run=fn,
                requires_columns=requires_columns, modeling_only=modeling_only,
            ),
        )
        return fn
    return _wrap


# --------------------------------------------------------------------------- #
# 检查项实现
# --------------------------------------------------------------------------- #


@check("dataset_required", title="数据类请求必须关联数据集")
def _c_dataset_required(inp: PreflightInput) -> tuple[list[Finding], dict[str, Any]]:
    if inp.intent is None or str(inp.intent.value) == "chat":
        return [], {}
    if inp.dataset_ids:
        return [], {"dataset_ids": list(inp.dataset_ids)}
    return (
        [finding(
            "preflight.dataset_required", Severity.BLOCK,
            "这个请求需要数据集才能执行，但当前会话没有关联任何数据集。",
            suggestion="请先在会话中关联一个数据集，或直接说明要分析哪个数据集",
        )],
        {},
    )


def _dtype_is_numeric(dtype: Any) -> bool:
    text = str(dtype or "").lower()
    return any(k in text for k in ("int", "float", "double", "number", "decimal"))


def _is_categorical_dtype(dtype: Any) -> bool:
    text = str(dtype or "").lower()
    return any(k in text for k in ("str", "cat", "bool", "object", "enum"))


@check("target_resolution", title="建模请求必须能确定目标列", modeling_only=True)
def _c_target_resolution(inp: PreflightInput) -> tuple[list[Finding], dict[str, Any]]:
    """目标列解析。

    优先级与 ``ml_engine.target_inference`` 保持一致，但**只用列名与 dtype**
    （不读数据）：
    ① 显式指定 → 校验存在性；② 命名约定唯一命中；③ 诉求/数据集名语义唯一命中；
    ④ 都不行 → 列出候选，反问用户。
    """
    columns = inp.columns
    if not columns:
        return (
            [finding(
                "preflight.target_unknown", Severity.CLARIFY,
                "建模请求未指定目标列，且 Pre-flight 阶段拿不到列清单，无法确认。",
                suggestion="请在请求里写明目标列，或先执行 dataset.schema 获取列信息",
            )],
            {},
        )

    names = inp.column_names
    # ② 命名约定
    from app.ml_engine.target_inference import _convention_targets  # noqa: PLC2701

    convention = _convention_targets(names)
    if inp.target:
        if inp.target not in names:
            return (
                [finding(
                    "preflight.target_unknown", Severity.BLOCK,
                    f"指定的目标列 {inp.target!r} 不在数据集中。",
                    target=inp.target,
                    evidence={"available_columns": names[:30]},
                    suggestion=f"可选列：{'、'.join(names[:20])}",
                )],
                {},
            )
        return [], {"target": inp.target, "target_source": "explicit"}
    if len(convention) == 1:
        return [], {"target": convention[0], "target_source": "naming_convention"}

    # ③ 语义匹配（用户诉求 + 数据集名称）
    from app.ml_engine.target_inference import goal_concepts  # noqa: PLC2701

    meta = inp.meta_of(inp.dataset_ids[0] if inp.dataset_ids else None)
    intent_text = f"{inp.user_request} {meta.get('name', '')}"
    concepts = goal_concepts(intent_text)
    if concepts:
        hits: list[str] = []
        for col in names:
            token_text = " ".join(_tokens(col))
            for concept in concepts:
                if _concept_hits(concept, col, token_text):
                    hits.append(col)
                    break
        if len(hits) == 1:
            return [], {"target": hits[0], "target_source": "goal_match"}
        if len(hits) > 1:
            return (
                [finding(
                    "preflight.target_ambiguous", Severity.BLOCK,
                    f"语义命中多个可能的列（{'、'.join(hits[:8])}），无法唯一确定目标列。",
                    evidence={"candidates": hits[:12], "concepts": sorted(concepts)},
                    suggestion="请指定要预测哪一列",
                )],
                {},
            )

    # ④ 列候选：日历/标识类列排在后面——它们几乎不可能是预测对象，
    #    但排在第一位会让「默认选项」变成错的默认值（这正是默认值害人的地方）。
    from app.ml_engine.target_inference import feature_like_reason  # noqa: PLC2701

    regression: list[str] = []
    classification: list[str] = []
    deferred: list[str] = []
    for c in columns:
        name = str(c.get("name"))
        bucket = (
            regression if _dtype_is_numeric(c.get("dtype"))
            else classification if _is_categorical_dtype(c.get("dtype"))
            else None
        )
        if bucket is None:
            continue
        (deferred if feature_like_reason(name) else bucket).append(name)
    regression += [c for c in deferred if _dtype_is_numeric(
        next((x.get("dtype") for x in columns if str(x.get("name")) == c), None))]
    classification += [c for c in deferred if c not in regression]
    return (
        [finding(
            "preflight.target_unknown", Severity.BLOCK,
            "建模请求未指定目标列，且无法按命名约定或诉求语义唯一确定。",
            evidence={
                "regression_candidates": regression[:12],
                "classification_candidates": classification[:12],
                "task_type": inp.task_type or "",
            },
            suggestion="请指定目标列（回归任务选连续列，分类任务选类别列）",
        )],
        {},
    )


def _tokens(col: str) -> list[str]:
    from app.quality.semantics import column_tokens

    return column_tokens(col)


def _concept_hits(concept: str, column: str, token_text: str) -> bool:
    """复用目标列推断的语义匹配规则（避免 Pre-flight 与推断层两套口径）。"""
    from app.ml_engine.target_inference import _GOAL_CONCEPTS  # noqa: PLC2701

    for name, aliases in _GOAL_CONCEPTS:
        if name != concept:
            continue
        return any(str(a).lower() in token_text or str(a) in str(column).lower() for a in aliases)
    return False


@check("task_type_conflict", title="任务类型与目标列类型不能矛盾", modeling_only=True)
def _c_task_conflict(inp: PreflightInput) -> tuple[list[Finding], dict[str, Any]]:
    if not inp.task_type or not inp.target:
        return [], {}
    dtype = next(
        (c.get("dtype") for c in inp.columns if str(c.get("name")) == inp.target), None
    )
    if dtype is None:
        return [], {}
    numeric = _dtype_is_numeric(dtype)
    if inp.task_type == "regression" and not numeric:
        return (
            [finding(
                "preflight.task_type_conflict", Severity.BLOCK,
                f"目标列 {inp.target} 的类型为 {dtype}，与「回归」任务不符。",
                target=inp.target,
                evidence={"task_type": inp.task_type, "dtype": str(dtype)},
                suggestion="改用 classification，或换一个连续列作为目标",
            )],
            {"task_type": "classification"},
        )
    if inp.task_type == "classification" and numeric:
        return (
            [finding(
                "preflight.task_type_conflict", Severity.BLOCK,
                f"目标列 {inp.target} 是连续数值（{dtype}），与「分类」任务不符。",
                target=inp.target,
                evidence={"task_type": inp.task_type, "dtype": str(dtype)},
                suggestion="改用 regression，或换一个类别列作为目标",
            )],
            {"task_type": "regression"},
        )
    return [], {}


@check("scale_sanity", title="样本量是否足以支撑建模", modeling_only=True)
def _c_scale(inp: PreflightInput) -> tuple[list[Finding], dict[str, Any]]:
    meta = inp.meta_of(inp.dataset_ids[0] if inp.dataset_ids else None)
    rows = int(meta.get("rows") or 0)
    if rows <= 0:
        return [], {}
    if rows < 50:
        return (
            [finding(
                "preflight.scale_too_small", Severity.CLARIFY,
                f"数据集仅 {rows} 行，训练结果可能不稳定。",
                evidence={"rows": rows},
                suggestion="继续建模会在结论里标注样本量风险；也可改为只做描述性分析",
            )],
            {"rows": rows},
        )
    return [], {"rows": rows}


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #


def run_preflight(inp: PreflightInput) -> PreflightResult:
    """按注册顺序执行全部适用的检查，汇总为 :class:`PreflightResult`。

    outcome 判定（**只看最高严重度**）：
    ``BLOCK`` 存在 → ``BLOCK``；否则 ``CLARIFY`` 存在 → ``CLARIFY``；否则 ``PROCEED``。
    """
    result = PreflightResult()
    findings: list[Finding] = []
    for code in CHECK_REGISTRY.keys():
        item = CHECK_REGISTRY.get(code)
        if item.modeling_only and not is_modeling(inp.intent):
            continue
        if item.requires_columns and not inp.columns:
            continue
        # 已经阻塞在「没有数据集」上时，建模类检查只会产出连环噪音
        # （没有数据集 ⇒ 必然没有目标列 ⇒ 必然没有任务类型）。
        if item.modeling_only and any(
            f.code == "preflight.dataset_required" and f.severity == Severity.BLOCK
            for f in findings
        ):
            continue
        try:
            got, resolved = item.run(inp)
        except Exception as exc:  # noqa: BLE001 - 单个检查失败不得阻断整条链路
            got = [finding(
                "preflight.check_failed", Severity.INFO,
                f"检查项 {code} 执行异常：{exc}", target=code,
            )]
            resolved = {}
        result.checks_run.append(code)
        findings.extend(got)
        if resolved:
            result.resolved.update(resolved)

    # 已回答过的反问不再重复追问
    pending = [f for f in findings if f.code not in inp.answers]
    for f in findings:
        if f.code in inp.answers:
            result.resolved.setdefault("answered", {})[f.code] = inp.answers[f.code]

    result.findings = merge_findings(pending)
    result.clarifications = [
        render(f, {**inp.meta_of(inp.dataset_ids[0] if inp.dataset_ids else None),
                   **result.resolved})
        for f in result.findings
        if f.severity in (Severity.BLOCK, Severity.CLARIFY)
    ]
    if result.blocking:
        result.outcome = PreflightOutcome.BLOCK
    elif result.clarifications:
        result.outcome = PreflightOutcome.CLARIFY
    else:
        result.outcome = PreflightOutcome.PROCEED
    return result
