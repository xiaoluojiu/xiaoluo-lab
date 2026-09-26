"""统一 TaskState：一个持续分析任务在多次 Run 之间的唯一状态真源。

设计原则
--------
* 会话历史（原始对话）仍由 ``AgentSession.history`` 持有，这里只引用不复制 ——
  状态是「任务」，不是第二份聊天记录。
* 工具结果只以**紧凑视图**进入状态（facts/findings/last_result），完整
  ``ToolResult.data`` 留在 Run 的 tool_calls 里；发给远程的载荷经
  :meth:`TaskState.compact_for_remote` 再裁剪一道。
* 状态可 JSON 序列化并随会话落盘，字段 ``to_dict / from_dict`` 严格对称。
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

#: 紧凑视图的硬预算（字符），任何进入状态的工具数据都不得超过。
FACT_VALUE_MAX_CHARS = 600
MAX_FACT_TOOLS = 8
MAX_FINDINGS = 12
MAX_REASONS = 5
MAX_COMPLETED = 30

#: 工具结果中被视为「可复用产物」的字段（供后续步骤/远程决策引用）。
_ARTIFACT_KEYS = ("run_id", "experiment_id", "workflow_id", "model_path", "report_path")


class Phase(StrEnum):
    """任务当前阶段（粗粒度，供按需召回工具与渲染使用）。"""

    INTAKE = "intake"
    ANALYSIS = "analysis"
    TRANSFORM = "transform"
    MODELING = "modeling"
    WRAPUP = "wrapup"
    DONE = "done"


class PendingAction(BaseModel):
    """待执行的一个动作。字段与 executor/step_resolution 消费的 step 形状兼容。"""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    permission: str = ""
    #: 决策来源：playbook / local_router / remote / user。
    source: str = ""
    rationale: str = ""

    def key(self) -> str:
        """去重键：同工具同参数视为同一动作（参数顺序无关）。"""
        import json

        return f"{self.tool}:{json.dumps(self.arguments, ensure_ascii=False, sort_keys=True, default=str)}"


class TaskState(BaseModel):
    """一个持续任务的统一状态。"""

    phase: str = Phase.INTAKE
    goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    #: 数据集上下文（dataset_id(str) -> 元数据 brief），仅元数据。
    dataset_context: dict[str, dict[str, Any]] = Field(default_factory=dict)

    #: 结构化事实：tool -> 紧凑 data 视图（每工具只留最新一版）。
    facts: dict[str, Any] = Field(default_factory=dict)
    #: 关键发现（信号与摘要提炼出的短句，新的在前）。
    findings: list[str] = Field(default_factory=list)
    #: 可复用产物（run_id / workflow_id / 新版本等）。
    artifacts: list[dict[str, Any]] = Field(default_factory=list)

    completed_actions: list[dict[str, Any]] = Field(default_factory=list)
    pending_actions: list[PendingAction] = Field(default_factory=list)
    pending_questions: list[dict[str, Any]] = Field(default_factory=list)

    last_result: dict[str, Any] | None = None
    signals: list[str] = Field(default_factory=list)

    confidence: float = 0.0
    uncertainty: float = 0.0
    uncertainty_reasons: list[str] = Field(default_factory=list)
    #: tool -> 连续失败次数。
    failures: dict[str, int] = Field(default_factory=dict)
    error_history: list[str] = Field(default_factory=list)

    #: 模型/资源用量摘要（每轮 Run 结束由 runtime 回填）。
    model_usage: dict[str, int] = Field(default_factory=dict)

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    closed: bool = False

    # ---- 构造与队列 -------------------------------------------------------
    @classmethod
    def initial(
        cls,
        goal: str,
        *,
        dataset_context: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
    ) -> "TaskState":
        return cls(
            goal=(goal or "").strip(),
            constraints=list(constraints or []),
            dataset_context={str(k): dict(v) for k, v in (dataset_context or {}).items()},
            confidence=0.5,
            uncertainty=0.3 if goal else 0.8,
        )

    def queue(self, actions: list[PendingAction] | PendingAction) -> list[PendingAction]:
        """入队待办，按 tool+参数 对 pending/completed 去重；返回实际新入队的动作。"""
        if isinstance(actions, PendingAction):
            actions = [actions]
        known = {a.key() for a in self.pending_actions}
        known |= {
            f"{c.get('tool')}:" + str(c.get("args_key", ""))
            for c in self.completed_actions
        }
        added: list[PendingAction] = []
        for action in actions:
            if action.key() in known:
                continue
            known.add(action.key())
            self.pending_actions.append(action)
            added.append(action)
        self._touch()
        return added

    def pop_next(self) -> PendingAction | None:
        """取出队首动作（从待办移除；失败重试由调用方重新 queue）。"""
        if not self.pending_actions:
            return None
        action = self.pending_actions.pop(0)
        self._touch()
        return action

    def has_pending(self) -> bool:
        return bool(self.pending_actions)

    # ---- 结果吸收 ---------------------------------------------------------
    def ingest_result(self, action: PendingAction, record: Any) -> None:
        """把一次工具执行的真实结果紧凑地吸收进状态。"""
        result = getattr(record, "result", None)
        status = str(getattr(record, "status", "unknown"))
        summary = str(getattr(result, "summary", "") or getattr(record, "error", "") or "")
        tool_signals = list(getattr(result, "signals", []) or []) if result is not None else []

        self.last_result = {
            "tool": action.tool,
            "status": status,
            "summary": summary[:300],
            "signals": tool_signals,
        }
        if tool_signals:
            for sig in tool_signals:
                if sig not in self.signals:
                    self.signals.append(sig)

        compact = None
        if result is not None:
            view = result.for_llm(max_items=8, max_chars=FACT_VALUE_MAX_CHARS)
            compact = (view or {}).get("data")
            self._collect_artifacts(action.tool, compact)

        if status == "ok":
            if compact is not None:
                self.facts[action.tool] = compact
            if summary:
                self._add_finding(f"[{action.tool}] {summary}")
            entry = {
                "tool": action.tool,
                "args_key": action.key().split(":", 1)[1],
                "summary": summary[:200],
                "signals": tool_signals,
            }
            self.completed_actions.append(entry)
            if len(self.completed_actions) > MAX_COMPLETED:
                self.completed_actions = self.completed_actions[-MAX_COMPLETED:]
            # 成功降低不确定性（有真实事实落地），但不低于 0.1。
            self.uncertainty = max(0.1, self.uncertainty - 0.15)
            self.failures.pop(action.tool, None)
        else:
            self.record_failure(action.tool, [summary or "工具执行失败"])
        self._touch()

    def record_failure(self, tool: str, errors: list[str] | None = None) -> None:
        """记录一次失败并动态抬高不确定性（连续失败 → 更可能需要升级）。"""
        count = self.failures.get(tool, 0) + 1
        self.failures[tool] = count
        for err in (errors or [])[:1]:
            self.error_history.append(f"{tool}: {str(err)[:160]}")
        self.error_history = self.error_history[-10:]
        self.uncertainty = min(0.95, self.uncertainty + 0.25)
        reason = f"{tool} 连续失败 {count} 次"
        if reason not in self.uncertainty_reasons:
            self.uncertainty_reasons.append(reason)
            self.uncertainty_reasons = self.uncertainty_reasons[-MAX_REASONS:]
        self._touch()

    def note_low_confidence(self, source: str, confidence: float) -> None:
        """本地决策低置信时抬高不确定性（动态复杂度的输入之一）。"""
        if confidence < 0.5:
            self.uncertainty = min(0.9, max(self.uncertainty, 0.6))
            reason = f"{source} 置信度不足({confidence:.2f})"
            if reason not in self.uncertainty_reasons:
                self.uncertainty_reasons.append(reason)
                self.uncertainty_reasons = self.uncertainty_reasons[-MAX_REASONS:]

    def _collect_artifacts(self, tool: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        for key in _ARTIFACT_KEYS:
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            artifact = {"tool": tool, "field": key, "value": value}
            if artifact not in self.artifacts:
                self.artifacts.append(artifact)

    def _add_finding(self, finding: str) -> None:
        finding = finding.strip()
        if finding and finding not in self.findings:
            self.findings.insert(0, finding)
            self.findings = self.findings[:MAX_FINDINGS]

    # ---- 多轮接续 ---------------------------------------------------------
    def follow_up_kind(self, text: str) -> str:
        """判定一条新消息相对当前任务的关系：state_question/constraint_change/new_task。"""
        t = (text or "").strip()
        # 新任务：显式重置信号（保守，宁可当作改要求也不轻易丢任务）。
        if any(marker in t for marker in ("重新开始", "新任务", "换个任务", "不做这个了", "算了")):
            return "new_task"
        # 改要求：否定旧动作 + 指定新做法，或明确的「改成/换成」。
        forbid = any(w in t for w in ("不要", "别", "不许", "不准", "不用"))
        redirect = any(w in t for w in ("改成", "换成", "改为", "改用", "替换成"))
        if (forbid and redirect) or (forbid and any(w in t for w in ("填充", "补上", "均值", "中位数", "众数"))):
            return "constraint_change"
        if redirect:
            return "constraint_change"
        return "state_question"

    def apply_constraint_change(self, text: str) -> bool:
        """把「不要删列，改成填充」类指令真实改写待办参数与约束。

        目前覆盖清洗策略（删除 ↔ 填充）；其余改要求仅沉淀为约束，
        由 Loop 下一轮 Decide 重新选动作。
        """
        t = (text or "").strip()
        changed = False
        fill_words = {"均值": "mean", "平均值": "mean", "mean": "mean",
                      "中位数": "median", "median": "median",
                      "众数": "mode", "mode": "mode",
                      "固定值": "constant", "填充": "mean", "补上": "mean"}
        forbid_drop = any(w in t for w in ("不要删", "别删", "不删除", "不要删除", "不准删"))
        wants_fill = any(w in t for w in fill_words) or ("填充" in t)
        strategy = next((v for k, v in fill_words.items() if k in t), "mean")

        if (forbid_drop or wants_fill) and any(a.tool == "data.clean" for a in self.pending_actions):
            for action in self.pending_actions:
                if action.tool != "data.clean":
                    continue
                missing = dict(action.arguments.get("missing") or {})
                if wants_fill or forbid_drop:
                    missing["strategy"] = strategy
                action.arguments["missing"] = missing
                changed = True
        constraint = f"用户修改要求：{t[:120]}"
        if constraint not in self.constraints:
            self.constraints.append(constraint)
        if changed:
            self._touch()
        return changed

    def completed_tools(self) -> set[str]:
        return {str(c.get("tool")) for c in self.completed_actions}

    # ---- 远程载荷 ---------------------------------------------------------
    def compact_for_remote(self, *, max_chars: int = 1600) -> dict[str, Any]:
        """发给远程 LLM 的唯一状态视图：紧凑、无完整 ToolResult、无完整历史。"""
        payload: dict[str, Any] = {
            "phase": self.phase,
            "goal": self.goal[:300],
            "constraints": self.constraints[-3:],
            "datasets": [
                {"id": k, "name": v.get("name"), "rows": v.get("rows"), "columns": v.get("columns")}
                for k, v in list(self.dataset_context.items())[:3]
            ],
            "findings": self.findings[:6],
            "artifacts": self.artifacts[:6],
            "signals": self.signals[-8:],
            "facts": {},
            "completed_tools": list(self.completed_tools()),
            "next_actions": [a.tool for a in self.pending_actions[:4]],
            "uncertainty": round(self.uncertainty, 2),
        }
        budget = max_chars - 400  # 留给其余固定字段
        per_tool = max(120, budget // max(len(self.facts), 1))
        for tool, data in list(self.facts.items())[-MAX_FACT_TOOLS:]:
            import json

            blob = json.dumps(data, ensure_ascii=False, default=str)
            payload["facts"][tool] = blob[:per_tool]
        return payload

    # ---- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskState | None":
        if not data:
            return None
        return cls.model_validate(dict(data))

    def _touch(self) -> None:
        self.updated_at = time.time()
