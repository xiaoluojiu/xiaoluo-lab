"""Agent 层 · 反问协议（第一层改造的核心交付物）。

为什么需要它
------------
平台为了「能用」，给未指定参数设了默认值（目标列不明 → 默认聚类；方法不适用 →
照常输出）。默认值一旦与用户意图不符，就会**产出看似完成、实则答非所问的结果**，
而且比直接失败危险得多——用户看不出来。

本模块把「默认值」改成「触发澄清」：

* :class:`Clarification` —— 一次反问的标准结构（code / question / options / default / severity）；
* :data:`TEMPLATES` —— 按 code 组织的反问模板库，**LLM 只负责填槽，不负责决定问什么**；
* :class:`ClarificationRequired` —— 工具在计划执行中发起反问的标准通道；
* :class:`ClarifyTool` —— 注册进 ``ToolRegistry`` 的 ``agent.clarify`` 工具，
  是 LLM 在计划中反问的**唯一**合法途径（禁止自由生成问题文本）。

三条硬约束
----------
1. 反问必须**带选项**：只问「你想预测哪一列」而不给候选，等于把工作量甩给用户；
2. 反问必须**带默认值**（``default``）：用户可以直接回车，链路不会卡死；
3. 反问必须**可机读**（``code`` + ``options[].value``）：前端据此渲染选择器，
   而不是让 LLM 解析自然语言回答。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.contracts import Finding, Severity
from app.core.exceptions import AppException
from app.agent.permission.rules import RiskLevel
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

__all__ = [
    "Clarification",
    "ClarificationOption",
    "ClarificationRequired",
    "ClarifyTool",
    "TEMPLATES",
    "render",
]


@dataclass
class ClarificationOption:
    """一个可选项。``value`` 是回填进参数的值，``label`` 是给用户看的文案。"""

    value: str
    label: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label or self.value,
            "note": self.note,
        }


@dataclass
class Clarification:
    """一次反问。"""

    code: str
    question: str
    options: list[ClarificationOption] = field(default_factory=list)
    default: str | None = None
    severity: Severity = Severity.CLARIFY
    #: 填槽用的客观上下文（候选列、任务类型、样本量……）
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "question": self.question,
            "options": [o.to_dict() for o in self.options],
            "default": self.default,
            "severity": str(self.severity),
            "severity_label": self.severity.label,
            "context": dict(self.context),
        }


class ClarificationRequired(AppException):
    """工具在执行中发起反问。

    与 :class:`~app.tools.base.ToolConfirmationRequired` 同构（前者问「要不要做」，
    后者问「做哪个」），执行器按同样的方式把它转成
    ``WAITING_CLARIFICATION`` 等待态。
    """

    http_status = 202
    default_code = "AGENT_CLARIFICATION_REQUIRED"
    default_message = "Agent requires clarification"

    def __init__(self, clarification: Clarification) -> None:
        super().__init__(clarification.question, code=self.default_code)
        self.clarification = clarification


# --------------------------------------------------------------------------- #
# 模板库：code -> 渲染函数
# --------------------------------------------------------------------------- #

#: 渲染函数签名：``(finding, context) -> Clarification``。
#: 模板**不包含**业务判断，只把 finding 的证据翻译成一句人话 + 可选项。
TEMPLATES: dict[str, Callable[[Finding, dict[str, Any]], Clarification]] = {}


def template(code: str) -> Callable[
    [Callable[[Finding, dict[str, Any]], Clarification]],
    Callable[[Finding, dict[str, Any]], Clarification],
]:
    """注册一个反问模板（用法同路由注册装饰器）。"""

    def _wrap(fn: Callable[[Finding, dict[str, Any]], Clarification]) -> Callable[..., Clarification]:
        TEMPLATES[code] = fn
        return fn

    return _wrap


def _options(values: list[str], notes: dict[str, str] | None = None) -> list[ClarificationOption]:
    notes = notes or {}
    return [ClarificationOption(value=v, note=notes.get(v, "")) for v in values]


@template("preflight.target_unknown")
def _t_target_unknown(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    candidates = list(ctx.get("regression_candidates") or []) + list(
        ctx.get("classification_candidates") or []
    )
    task = ctx.get("task_type") or "监督"
    return Clarification(
        code=finding.code,
        question=(
            f"这次是{task}任务，但没有指定目标列（要预测/解释的那一列）。"
            "请选择一个，或直接在回答里写列名。"
        ),
        options=_options(candidates[:12]),
        default=candidates[0] if candidates else None,
        severity=Severity.BLOCK,
        context={"task_type": task, "candidates": candidates[:12]},
    )


@template("preflight.target_ambiguous")
def _t_target_ambiguous(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    candidates = list(ctx.get("candidates") or [])
    return Clarification(
        code=finding.code,
        question=f"有多个列都可能作为目标列（{'、'.join(candidates[:8])}）。请指定要预测哪一列。",
        options=_options(candidates[:12]),
        default=candidates[0] if candidates else None,
        severity=Severity.BLOCK,
        context={"candidates": candidates[:12]},
    )


@template("preflight.task_type_conflict")
def _t_task_conflict(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    detail = ctx.get("detail") or finding.message
    return Clarification(
        code=finding.code,
        question=f"{detail}。请确认要做的任务类型。",
        options=_options(
            list(ctx.get("task_options") or ["regression", "classification", "clustering"]),
            notes={"regression": "预测连续值", "classification": "预测离散类别",
                   "clustering": "无目标的分群"},
        ),
        default=ctx.get("suggested_task"),
        severity=Severity.BLOCK,
        context=dict(ctx),
    )


@template("preflight.dataset_required")
def _t_dataset_required(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    items = list(ctx.get("datasets") or [])
    return Clarification(
        code=finding.code,
        question="这个请求需要指定数据集，但当前会话没有关联任何数据集。请选择要分析的数据集。",
        options=[ClarificationOption(value=str(d.get("id")), label=str(d.get("name"))) for d in items[:20]],
        default=str(items[0].get("id")) if items else None,
        severity=Severity.BLOCK,
        context={"datasets": items[:20]},
    )


@template("preflight.scale_too_small")
def _t_scale(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    return Clarification(
        code=finding.code,
        question=(
            f"当前可用样本仅 {ctx.get('rows', 0)} 行，"
            "训练结果可能不稳定。仍要继续吗？"
        ),
        options=_options(
            ["继续", "改用描述性分析"],
            notes={"继续": "按当前样本量训练并标注风险",
                   "改用描述性分析": "只做统计与质量画像，不建模"},
        ),
        default="继续",
        severity=Severity.CLARIFY,
        context=dict(ctx),
    )


@template("quality.method_mismatch")
def _t_method_mismatch(finding: Finding, ctx: dict[str, Any]) -> Clarification:
    return Clarification(
        code=finding.code,
        question=(
            f"{finding.message}。是否改用建议的方法「{ctx.get('recommended_method', '')}」？"
        ),
        options=_options(
            ["采用建议方法", "维持原方法"],
            notes={"采用建议方法": "结果更可信",
                   "维持原方法": "口径不变，但结论需标注局限"},
        ),
        default="采用建议方法",
        severity=Severity.CLARIFY,
        context=dict(ctx),
    )


def render(finding: Finding, context: dict[str, Any] | None = None) -> Clarification:
    """按 code 渲染反问；没有模板时退化为「无选项的自由回答」（仍带 code，可机读）。"""
    ctx = dict(context or {})
    ctx.setdefault("target", finding.target)
    ctx.update(finding.evidence or {})
    fn = TEMPLATES.get(finding.code)
    if fn is not None:
        try:
            return fn(finding, ctx)
        except Exception:  # noqa: BLE001 - 模板渲染失败不得卡死链路
            pass
    return Clarification(
        code=finding.code,
        question=finding.message,
        options=[],
        default=None,
        severity=finding.severity,
        context=ctx,
    )


# --------------------------------------------------------------------------- #
# 工具：agent.clarify
# --------------------------------------------------------------------------- #

#: 已回答的反问在 ``ToolExecutionContext.extra`` 里的键。
ANSWERS_KEY = "clarification_answers"


class ClarifyTool(Tool):
    """``agent.clarify``：计划执行过程中反问用户的**唯一**合法通道。

    执行语义：

    * 同一 ``code`` 已被回答过 → 直接返回答案（``ToolResult.ok``，
      ``data.answer`` 可被后续步骤用 ``{{stepN.answer}}`` 引用）；
    * 未回答 → 抛 :class:`ClarificationRequired`，由执行器转成等待态。

    这样设计的好处：反问在计划里是一个**普通步骤**，用户回答后计划从原处继续，
    而不是从头重来（重来会重复消耗前面已经跑完的重型步骤）。
    """

    name = "agent.clarify"
    description = (
        "向用户提出一个结构化问题并等待回答。只能在确实缺少必要信息时使用"
        "（如目标列不明、任务类型矛盾）；必须提供候选项。回答后本步骤返回 answer，"
        "后续步骤可用 {{stepN.answer}} 引用。"
    )
    category = "agent"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "问题标识，如 preflight.target_unknown",
            },
            "question": {"type": "string", "description": "要问用户的问题（一句话）"},
            "options": {
                "type": "array",
                "description": "候选项；每项 {value, label, note}",
                "items": {"type": "object"},
            },
            "default": {"type": "string", "description": "默认选项的 value"},
            "severity": {
                "type": "string",
                "enum": ["block", "clarify", "warn"],
                "description": "阻断级别",
            },
            "context": {"type": "object", "description": "供渲染用的客观上下文"},
        },
        "required": ["code", "question"],
    }
    output_schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "answer": {"type": "string"}},
    }
    permission = "read_data"
    risk_level = RiskLevel.LOW

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        code = str(params.get("code") or "").strip()
        question = str(params.get("question") or "").strip()
        if not code or not question:
            raise AppException("agent.clarify 需要 code 与 question", code="CLARIFY_ARGS_MISSING")

        answers = dict(context.extra.get(ANSWERS_KEY) or {})
        answered = answers.get(code)
        if answered:
            return ToolResult.ok(
                {"code": code, "answer": str(answered), "already_answered": True},
                summary=f"反问 {code} 已回答：{answered}",
            )

        options = [
            ClarificationOption(
                value=str(o.get("value", "")),
                label=str(o.get("label") or o.get("value") or ""),
                note=str(o.get("note") or ""),
            )
            for o in (params.get("options") or [])
            if isinstance(o, dict) and o.get("value")
        ]
        raise ClarificationRequired(
            Clarification(
                code=code,
                question=question,
                options=options,
                default=(str(params.get("default")) if params.get("default") else None),
                severity=Severity(str(params.get("severity") or "clarify")),
                context=dict(params.get("context") or {}),
            )
        )

