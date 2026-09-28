"""Agent 运行期数据模型。

契约警告
--------
字段名与前端 ``src/types/agent.ts`` **逐字对齐**。
改这里的字段名必须同步改前端类型，否则时间线、授权面板、Token 面板会
**静默失效**（前端大量使用 ``?? `` 兜底，缺字段不会报错，只会显示空白）。

几个最容易踩的字段（前端无兜底，缺失即白屏）：
- ``AgentRun.elapsed_seconds`` —— ``RunInspector`` 直接 ``.toFixed(1)``
- ``AgentRun.tool_calls`` —— 必须是数组
- ``AgentEvent.run_id`` —— 授权/拒绝/取消/澄清的目标 run 由它决定

关于被删除的旧事件
------------------
``preflight`` 与 ``replanning`` 是旧架构的产物（开工前检查 + 失败后重规划循环）。
新架构无重规划、无前置检查阶段，因此**不再产出这两种事件**；
枚举里也不再保留，避免出现「定义了却永远不会发」的死值。
前端对未知/未出现的事件类型无副作用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _now() -> float:
    """当前 unix 时间戳（秒）。前端把 created_at 当秒处理。"""
    return time.time()


class RunStatus(str, Enum):
    """运行状态。取值与前端 ``AgentRunStatus`` 一致。"""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_CLARIFICATION = "waiting_clarification"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value

    @property
    def in_flight(self) -> bool:
        """是否仍在推进（前端据此决定是否继续轮询）。"""
        return self in (RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RUNNING)

    @property
    def waiting(self) -> bool:
        """是否挂在人工输入上（确认 / 补充信息）。"""
        return self in (RunStatus.WAITING_CONFIRMATION, RunStatus.WAITING_CLARIFICATION)

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED)


class EventType(str, Enum):
    """SSE 事件类型。取值与前端 ``AgentEventType`` 一致。"""

    ROUTE = "route"
    CHAT = "chat"
    PLANNING = "planning"
    PERMISSION = "permission"
    CLARIFICATION = "clarification"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    VALIDATION = "validation"
    USAGE = "usage"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DENIED = "denied"

    def __str__(self) -> str:
        return self.value


@dataclass
class PermissionRequest:
    """待用户确认的高风险调用。前端 ``PermissionRequest`` 同构。"""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    step_index: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "step_index": self.step_index,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PermissionRequest | None:
        if not data:
            return None
        return cls(
            tool=str(data.get("tool") or ""),
            arguments=data.get("arguments") or {},
            step_index=int(data.get("step_index") or 0),
            reason=str(data.get("reason") or ""),
        )


@dataclass
class ClarificationOption:
    value: str
    label: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label or self.value, "note": self.note}


@dataclass
class ClarificationRequest:
    """待用户补充的信息。前端 ``ClarificationRequest`` 同构。

    ``outcome`` 为 ``"answered"`` 时前端撤掉面板。
    """

    question: str = ""
    code: str = ""
    options: list[ClarificationOption] = field(default_factory=list)
    default: str | None = None
    outcome: str | None = None
    step_index: int | None = None
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "question": self.question,
            "options": [o.to_dict() for o in self.options],
            "default": self.default,
            "outcome": self.outcome,
            "step_index": self.step_index,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClarificationRequest | None:
        if not data:
            return None
        options = [
            ClarificationOption(value=str(o.get("value") or ""), label=str(o.get("label") or ""), note=str(o.get("note") or ""))
            for o in (data.get("options") or [])
            if isinstance(o, dict)
        ]
        return cls(
            question=str(data.get("question") or ""),
            code=str(data.get("code") or ""),
            options=options,
            default=data.get("default"),
            outcome=data.get("outcome"),
            step_index=data.get("step_index"),
            tool=data.get("tool"),
        )


@dataclass
class ToolCall:
    """一次工具调用记录。"""

    step_index: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: dict[str, Any] | None = None
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool": self.tool,
            "arguments": self.arguments,
            "attempt": self.attempt,
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            step_index=int(data.get("step_index") or 0),
            tool=str(data.get("tool") or ""),
            arguments=data.get("arguments") or {},
            attempt=int(data.get("attempt") or 1),
            status=ToolCallStatus(data.get("status") or ToolCallStatus.PENDING.value),
            result=data.get("result"),
            error=str(data.get("error") or ""),
            elapsed_ms=int(data.get("elapsed_ms") or 0),
        )


@dataclass
class TokenUsage:
    """Token 用量。

    ``actual`` 是厂商回传的真实用量；``optimization`` 是**本架构节省量**的估算。
    两者的口径不同，绝不能相加或互相覆盖。
    """

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    note: str = ""
    max_llm_calls: int = 0
    max_total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        total = self.input_tokens + self.output_tokens
        return {
            "llm_calls": self.llm_calls,
            "actual": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": total,
            },
            # 新架构不做 plan cache / 结果压缩开关，节省量来自「不调用」本身：
            # 规则路由 0 次调用、工具结果只喂摘要。这里如实回填 0，不虚报。
            "optimization": {
                "estimated_context_saved_tokens": 0,
                "estimated_result_saved_tokens": 0,
                "estimated_saved_tokens": 0,
                "avoided_planner_calls": 0,
                "plan_cache_hits": 0,
            },
            "budget": {
                "max_llm_calls": self.max_llm_calls,
                "max_total_tokens": self.max_total_tokens,
                "remaining_llm_calls": max(0, self.max_llm_calls - self.llm_calls),
                "remaining_total_tokens": max(0, self.max_total_tokens - total),
            },
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TokenUsage:
        if not data:
            return cls()
        actual = data.get("actual") or {}
        budget = data.get("budget") or {}
        return cls(
            llm_calls=int(data.get("llm_calls") or 0),
            input_tokens=int(actual.get("input_tokens") or 0),
            output_tokens=int(actual.get("output_tokens") or 0),
            note=str(data.get("note") or ""),
            max_llm_calls=int(budget.get("max_llm_calls") or 0),
            max_total_tokens=int(budget.get("max_total_tokens") or 0),
        )


@dataclass
class AnswerSource:
    """答案来源。前端据此渲染「由远程大模型回答 / 由平台内置规则回答」。

    ``by_llm`` 必须真实：规则生成的答案不能标成大模型回答。
    """

    source: str
    label: str
    detail: str = ""
    by_llm: bool = False
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "detail": self.detail,
            "by_llm": self.by_llm,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnswerSource | None:
        if not data:
            return None
        return cls(
            source=str(data.get("source") or ""),
            label=str(data.get("label") or ""),
            detail=str(data.get("detail") or ""),
            by_llm=bool(data.get("by_llm")),
            model=str(data.get("model") or ""),
        )


#: 预置的答案来源。旧架构有七种（含 llm_error_fallback 等），
#: 新架构的降级是「有或无」，不是「降级到半吊子」，因此只保留
#: 「平台内置规则」与本文件 :func:`answer_source_llm`（远程大模型）两条路径。
ANSWER_SOURCE_RULE = AnswerSource(
    source="platform_rules", label="平台内置规则", detail="由平台规则与工具结果直接生成，未调用大模型", by_llm=False
)


def answer_source_llm(model: str = "") -> AnswerSource:
    return AnswerSource(
        source="remote_llm",
        label="远程大模型",
        detail="由远程大模型组织最终回答",
        by_llm=True,
        model=model,
    )


@dataclass
class AgentEvent:
    """一条事件（SSE 一帧的 data 就是它的 dict）。"""

    seq: int
    run_id: str
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "type": str(self.type),
            "payload": self.payload,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEvent:
        return cls(
            seq=int(data.get("seq") or 0),
            run_id=str(data.get("run_id") or ""),
            type=EventType(data.get("type") or EventType.CHAT.value),
            payload=data.get("payload") or {},
            created_at=float(data.get("created_at") or _now()),
        )


@dataclass
class AgentRun:
    """一次运行。"""

    id: str
    session_id: str
    user_request: str = ""
    status: RunStatus = RunStatus.PENDING
    final_answer: str = ""
    error: str = ""
    plan: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    pending_confirmation: PermissionRequest | None = None
    pending_clarification: ClarificationRequest | None = None
    cancel_requested: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    answer_source: AnswerSource | None = None
    events: list[AgentEvent] = field(default_factory=list)
    created_at: float = field(default_factory=_now)
    started_at: float = field(default_factory=_now)
    finished_at: float | None = None
    # 挂起恢复点：从哪一步继续（confirm / clarify 之后）
    resume_step: int = 0
    # 已授权的一次性凭据：只对「被确认的那一个 (step_index, tool)」生效
    authorized_key: str | None = None
    # 用户补充的答案（clarify 提交后写入，恢复时注入工具参数）
    clarification_answers: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else _now()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_request": self.user_request,
            "status": str(self.status),
            "final_answer": self.final_answer,
            "error": self.error,
            "plan": self.plan,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_call_count": self.tool_call_count,
            "pending_confirmation": self.pending_confirmation.to_dict() if self.pending_confirmation else None,
            "pending_clarification": self.pending_clarification.to_dict() if self.pending_clarification else None,
            "cancel_requested": self.cancel_requested,
            "token_usage": self.token_usage.to_dict(),
            "answer_source": self.answer_source.to_dict() if self.answer_source else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "events": [e.to_dict() for e in self.events],
            "created_at": self.created_at,
            "started_at": self.started_at,
            # 恢复点与授权凭据必须持久化：漏掉它们，重启后挂起的确认/澄清运行
            # 会丢 resume_step（从第 0 步重跑）、丢 authorized_key（已确认失效）、
            # 丢 clarification_answers（用户补充的答案消失）。finished_at 丢失则
            # elapsed_seconds 会退化成 now - started_at。
            "finished_at": self.finished_at,
            "resume_step": self.resume_step,
            "authorized_key": self.authorized_key,
            "clarification_answers": self.clarification_answers,
        }


@dataclass
class AgentSession:
    """一个会话。"""

    id: str
    user_id: str = "anonymous"
    title: str = "新会话"
    dataset_ids: list[int] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=_now)
    archived: bool = False
    # 会话的**数据基线版本**：{dataset_id: version}。
    #
    # 为什么需要它：写类工具（data.aggregate / data.merge）会把产出存成数据集的
    # 新版本，而新版本即 latest。「按 purpose 分组求平均」之后 latest 就变成一张
    # 7 行 × 2 列的聚合小表 —— 用户接着问「看看 age 的分布」，引擎默认读 latest，
    # 于是所有后续分析都跑在这张废表上（相关性报「至少需要 2 个数值字段」、
    # 质量检查说「7 行 2 列」、点名的列根本不存在）。
    #
    # 基线就是「用户当前认为的数据状态」：
    #   - 只读分析（eda/quality/ml/report/preview）一律基于基线；
    #   - 改写数据状态的写操作（clean/filter/transform）完成后推进基线；
    #   - 只产出结果、不改变数据状态的写操作（aggregate/merge）**不推进**基线，
    #     结果另存新版本，原数据不动。
    baseline_versions: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "dataset_ids": list(self.dataset_ids),
            "history": [dict(m) for m in self.history],
            "run_ids": list(self.run_ids),
            "created_at": self.created_at,
            "archived": self.archived,
            # JSON 的键只能是字符串，读回时转回 int（见 store._session_from_dict）。
            "baseline_versions": {str(k): int(v) for k, v in self.baseline_versions.items()},
        }
