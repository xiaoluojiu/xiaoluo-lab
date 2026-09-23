"""标准化内核 · 统一契约。

为什么要有这个文件
------------------
改造前，平台里「程序替用户做的判断」散落在各处、形态各异：

* ``ml_engine.target_inference.TargetGuess`` —— 有 ``confidence`` / ``reasons``
  / ``alternatives``，是**唯一**一个自带证据链的判定结果；
* ``analysis.compute_outlier_bounds`` —— 只返回 ``(lower, upper, info)``，
  边界为 ``[-16.5, 19.5]`` 这种「负下界」照样输出，调用方无从判断方法是否适用；
* ``AgentPlanner._rule_plan`` —— 直接给 ``model="auto"``，不说为什么；
* ``reports`` 的结论 —— 只有字符串，没有严重度与依据。

结果是：**Agent 拿到什么就复述什么**，因为它拿到的东西里根本不带
「这个结论有多可信 / 依据是什么 / 有没有更合适的做法」。

本模块把这类「程序判断」统一成一个契约，任何一层（任务理解 / 质量判定 /
特征工程 / 指标选择 / 实验解读 / 报告结论）产出的判定都必须满足它：

    Decision(value, source, confidence, reasons, alternatives, findings)

配套两个辅助类型：

* :class:`Severity` —— 判定里附带问题的严重度（决定「阻断 / 反问 / 警告 / 标注」四档）；
* :class:`Finding` —— 一条可被 Agent 直接消费、也可直接渲染给用户的问题描述。

设计约束（与其他层对齐）
------------------------
* 只做**描述**，不做**执行**：Decision 不触发任何副作用，是否阻断由调用方按
  ``Severity`` 决定；
* 可序列化：所有类型都能 ``to_dict()`` 后写进 ``ToolResult.data`` / 事件 payload；
* 不引入第三方依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

__all__ = [
    "Decision",
    "Finding",
    "Severity",
    "finding",
    "merge_findings",
]

T = TypeVar("T")


class Severity(StrEnum):
    """问题严重度四档。

    这四档直接对应 Prompt 里的「反问三档输出」，外加一档纯标注：

    ====================  ==========================================
    BLOCK    硬阻断        必须等回答，否则继续就是错的结果
                          （目标列不明 / 任务类型矛盾 / 目标列混进特征）
    CLARIFY  反问          有默认值可以继续执行，但结果可能不是用户要的
    WARN     软警告        继续执行，但必须在结论里标注方法/指标局限
    INFO     标注          仅记录，不影响执行
    ====================  ==========================================
    """

    BLOCK = "block"
    CLARIFY = "clarify"
    WARN = "warn"
    INFO = "info"

    @property
    def rank(self) -> int:
        """严重度序（越大越严重），用于排序与取最高档。"""
        return _SEVERITY_RANK[self]

    @property
    def label(self) -> str:
        return _SEVERITY_LABEL[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARN: 1,
    Severity.CLARIFY: 2,
    Severity.BLOCK: 3,
}

_SEVERITY_LABEL: dict[Severity, str] = {
    Severity.BLOCK: "硬阻断",
    Severity.CLARIFY: "需澄清",
    Severity.WARN: "软警告",
    Severity.INFO: "提示",
}


@dataclass
class Finding:
    """一条结构化问题。

    ``code`` 是稳定标识（如 ``target.unknown``），供：
    * 反问模板库按 code 取模板（``agent/clarify/templates.py``）；
    * 测试按 code 断言，而不是断言中文文案；
    * 前端按 code 决定展示样式。
    """

    code: str
    severity: Severity
    message: str
    #: 涉及的对象：列名 / 工具名 / 参数名。可空。
    target: str = ""
    #: 支撑该判断的客观事实（数值、边界、样本量……），禁止放原始数据。
    evidence: dict[str, Any] = field(default_factory=dict)
    #: 可操作的下一步（不是「请检查数据」，而是「请指定 target=DepDelay」）
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": str(self.severity),
            "severity_label": self.severity.label,
            "message": self.message,
            "target": self.target,
            "evidence": dict(self.evidence),
            "suggestion": self.suggestion,
        }


def finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    target: str = "",
    evidence: dict[str, Any] | None = None,
    suggestion: str = "",
) -> Finding:
    """构造 :class:`Finding`（severity 允许传字符串，便于调用方少 import）。"""
    return Finding(
        code=code,
        severity=Severity(severity) if not isinstance(severity, Severity) else severity,
        message=message,
        target=target,
        evidence=dict(evidence or {}),
        suggestion=suggestion,
    )


def merge_findings(*groups: list[Finding] | None) -> list[Finding]:
    """合并多组 Finding 并按严重度降序（同档保持原有顺序）。"""
    out: list[Finding] = []
    for group in groups:
        out.extend(group or [])
    return sorted(out, key=lambda f: -f.severity.rank)


@dataclass
class Decision(Generic[T]):
    """统一的程序判定结果。

    约定
    ----
    * ``value is None`` ⇒ 本次判定**没有结论**（不是「结论为空字符串」）；
      调用方必须查 :attr:`needs_clarification` 决定是否反问，禁止拿 None 硬往下走。
    * ``confidence`` 与 ``source`` 成对出现：source 说明**凭什么**这么判；
      confidence 说明**多可信**。没有 source 的 confidence 没有意义。
    * ``findings`` 里只要有一条 ``Severity.BLOCK``，:attr:`needs_clarification`
      即为 True —— 这是「默认值改为触发澄清」那条原则的落点。
    """

    value: T | None = None
    #: 判定来源 code（如 explicit / naming_convention / goal_match / business_rule）
    source: str = ""
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    alternatives: list[Any] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    #: 支撑该判定的客观事实（命中词、统计量、样本量……），供审计与解释
    evidence: dict[str, Any] = field(default_factory=dict)

    # ---- 判定状态 -------------------------------------------------------
    @property
    def resolved(self) -> bool:
        """是否给出了结论。"""
        return self.value is not None

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.BLOCK]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in (Severity.WARN, Severity.CLARIFY)]

    @property
    def needs_clarification(self) -> bool:
        """是否需要反问用户（有硬阻断，或有结论但证据不足）。"""
        if self.blocking:
            return True
        return self.resolved and any(f.severity == Severity.CLARIFY for f in self.findings)

    @property
    def severity(self) -> Severity:
        """本次判定的最高严重度（无问题则为 INFO）。"""
        if not self.findings:
            return Severity.INFO
        return max(self.findings, key=lambda f: f.severity.rank).severity

    # ---- 构造与合并 -----------------------------------------------------
    @classmethod
    def of(
        cls,
        value: T | None,
        *,
        source: str,
        confidence: float,
        reason: str = "",
        alternatives: list[Any] | None = None,
        findings: list[Finding] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> "Decision[T]":
        return cls(
            value=value,
            source=source,
            confidence=confidence,
            reasons=[reason] if reason else [],
            alternatives=list(alternatives or []),
            findings=list(findings or []),
            evidence=dict(evidence or {}),
        )

    def with_finding(self, item: Finding) -> "Decision[T]":
        self.findings.append(item)
        return self

    def degrade(self, item: Finding) -> "Decision[T]":
        """追加一条「方法可能不适用」类的软警告（保留原结论，只做标注）。"""
        return self.with_finding(item)

    # ---- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "resolved": self.resolved,
            "needs_clarification": self.needs_clarification,
            "severity": str(self.severity),
            "reasons": list(self.reasons),
            "alternatives": list(self.alternatives),
            "findings": [f.to_dict() for f in self.findings],
            "evidence": dict(self.evidence),
        }

    def explain(self) -> str:
        """一行人类可读解释（写进日志 / 工具 summary / 报告脚注）。"""
        head = f"{self.value!r}" if self.resolved else "无结论"
        bits = [f"{head}（依据 {self.source or '未知'}，置信度 {self.confidence:.2f}）"]
        if self.reasons:
            bits.append("；".join(self.reasons))
        for item in self.findings:
            bits.append(f"[{item.severity.label}] {item.code}: {item.message}")
        return "；".join(bits)
