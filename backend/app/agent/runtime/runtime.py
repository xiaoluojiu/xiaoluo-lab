"""Agent Runtime：会话 / Run 生命周期与统一 Stateful Loop 的宿主。

重构后（统一 Agent Loop）：本模块不再承担任何「决策 / 规划 / 执行」——
那些全部收敛到 :class:`app.agent.loop.AgentLoop`。这里只保留：

* 会话 / Run 生命周期与 ``AgentStore`` 交互（原子领取、重启恢复）；
* 多轮接续：TaskState 的 hydrate（接续）/ reset（新任务）/ 回写；
* 等待态的原子领取与 resume / answer_clarification / deny / cancel；
* SSE 事件、Token 账本快照、DecisionTrace 的挂载与落盘。

外部契约（HTTP/SSE/存储）保持不变；``run()`` 不再按 mode 切两条流程，
统一委托 ``AgentLoop.turn()``——CHAT 只是 Loop 内的一种动作。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from app.agent.answer_source import NO_ANSWER, describe
from app.agent.loop import AgentLoop
from app.agent.llm.base import LLMException, LLMProvider
from app.agent.runtime.models import (
    AgentEvent,
    AgentRun,
    AgentSession,
    AgentStore,
    RunStatus,
)
from app.agent.state import TaskState
from app.core.config import settings
from app.core.exceptions import AgentException, ValidationException
from app.data_engine.service import DataEngineService
from app.tools.registry import TOOL_REGISTRY, ToolRegistry

EventCallback = Callable[[AgentEvent], None]
logger = logging.getLogger("xiaoluo.agent.runtime")

#: 单次运行允许的事件数硬上限（终态事件不计）；与 Loop 自持常量同值。
MAX_EVENTS_PER_RUN = 2000
_TERMINAL_EVENTS = frozenset({"completed", "failed"})


class AgentRuntime:
    """统一 Loop 的宿主：只管理生命周期、状态持久化与事件/Trace 挂载。"""

    def __init__(
        self,
        data_engine: DataEngineService,
        *,
        experiment_service: Any | None = None,
        db: Any | None = None,
        llm: LLMProvider | None = None,
        store: AgentStore | None = None,
        registry: ToolRegistry | None = None,
        role: str = "analyst",
    ) -> None:
        self.data_engine = data_engine
        self.experiment_service = experiment_service
        self.db = db
        self.llm = llm
        self.store = store or AgentStore()
        from app.tools.builtin import register_builtin_tools

        register_builtin_tools()
        self.registry = registry or TOOL_REGISTRY
        self.role = role
        # 唯一的控制流：决策 / 执行 / 收尾全部在 Loop 内完成。
        self.loop = AgentLoop(
            data_engine,
            experiment_service=experiment_service,
            db=db,
            llm=llm,
            registry=self.registry,
            role=role,
        )
        # DecisionTrace 运行期承载（run_id -> DecisionTrace），运行结束统一落盘。
        self._decision_traces: dict[str, Any] = {}

    # ------------------------------------------------------------------ 生命周期
    def create_session(self, *, user_id: str = "anonymous", title: str = "", dataset_ids: list[int] | None = None) -> AgentSession:
        ids = list(dict.fromkeys(int(x) for x in (dataset_ids or []) if int(x) > 0))
        return self.store.add_session(
            AgentSession(id=self.store.next_session_id(), user_id=user_id, title=title or "新会话", dataset_ids=ids)
        )

    def list_sessions(self, user_id: str | None = None, *, include_archived: bool = True) -> list[AgentSession]:
        return self.store.list_sessions(user_id, include_archived=include_archived)

    def get_session(self, session_id: str) -> AgentSession:
        return self.store.get_session(session_id)

    def set_session_archived(self, session_id: str, archived: bool) -> AgentSession:
        return self.store.set_session_archived(session_id, archived)

    def delete_session(self, session_id: str) -> AgentSession:
        return self.store.delete_session(session_id)

    def get_run(self, run_id: str) -> AgentRun:
        return self.store.get_run(run_id)

    def _usage_scope(self, run: AgentRun):
        """把 Loop 内所有 LLM 调用的真实 usage 转发进本 Run 账本。"""
        if self.llm is None:
            return nullcontext()
        return self.llm.capture_usage(run.token_ledger.record_usage, budget_checker=run.token_ledger.check_budget)

    # ------------------------------------------------------------------ 入口
    def run(self, session: AgentSession, user_request: str, *, confirmed: bool = False, on_event: EventCallback | None = None) -> AgentRun:
        """一轮新消息：hydrate 任务状态后统一交给 Loop，结束后回写并落盘。"""
        if not user_request or not user_request.strip():
            raise ValidationException("用户请求不能为空")
        run = AgentRun(id=self.store.next_run_id(), session_id=session.id, user_id=session.user_id, user_request=user_request.strip())
        self.store.add_run(run)
        if run.id not in session.run_ids:
            session.run_ids.append(run.id)
        session.history.append({"role": "user", "content": run.user_request})
        run.started_at = time.time()

        state = self._hydrate_state(session, run.user_request)
        self._begin_trace(run)
        try:
            with self._usage_scope(run):
                run.status = RunStatus.PLANNING
                self._trace_route(run, session)
                self.loop.turn(
                    run, session, state,
                    on_event=on_event,
                    persist=lambda: self.store.persist(force=True),
                    run_preflight_check=True,
                )
                self._snapshot_plan(run, state)
        except LLMException as exc:
            self._fail(run, exc.message, on_event)
        except AgentException as exc:
            self._fail(run, exc.message, on_event)
        except Exception as exc:  # noqa: BLE001 - 边界异常统一失败，绝不静默吞成「已完成」
            self._fail(run, f"Agent 运行异常：{exc}", on_event)
        finally:
            run.finished_at = run.finished_at or time.time()
            session.task_state = state.to_dict()
            self._trace_outcome(run, session)
            self._end_trace(run, state)
            self.store.persist(force=True)
        return run

    def resume(self, session: AgentSession, run_id: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """高风险操作授权后继续：进入**同一个** Loop 续跑点（confirmation resume）。"""
        run, pending = self.store.claim_confirmation(run_id)
        if pending is None:
            raise ValidationException("该运行不在等待确认状态（授权已被领取或运行已结束）")
        step_index = int(pending.get("step_index", 0))
        self._emit(run, "permission", {"stage": "confirmed", "tool": pending.get("call").tool, "step_index": step_index}, on_event)
        state = TaskState.from_dict(session.task_state) or TaskState.initial(run.user_request)
        try:
            with self._usage_scope(run):
                self.loop.turn(
                    run, session, state,
                    on_event=on_event,
                    persist=lambda: self.store.persist(force=True),
                    resume={"kind": "confirmation", "payload": pending},
                    confirmed=True,
                    run_preflight_check=True,
                )
                self._snapshot_plan(run, state)
        except LLMException as exc:
            self._fail(run, exc.message, on_event)
        except AgentException as exc:
            self._fail(run, exc.message, on_event)
        except Exception as exc:  # noqa: BLE001
            self._fail(run, f"Agent 运行异常：{exc}", on_event)
        finally:
            run.finished_at = run.finished_at or time.time()
            session.task_state = state.to_dict()
            self._trace_outcome(run, session)
            self._end_trace(run, state)
            self.store.persist(force=True)
        return run

    def answer_clarification(self, run_id: str, answer: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """回答反问：Pre-flight 反问重新检查、工具槽位反问从原步骤继续，都在同一 Loop。"""
        if not str(answer or "").strip():
            raise ValidationException("澄清回答不能为空")
        run, pending, code = self.store.claim_clarification(run_id, str(answer))
        if pending is None:
            raise ValidationException("该运行不在等待澄清状态（回答已被受理或运行已结束）")
        session = self.store.get_session(run.session_id)
        self._emit(run, "clarification", {"stage": "answered", "code": code, "answer": str(answer)}, on_event)
        state = TaskState.from_dict(session.task_state) or TaskState.initial(run.user_request)
        preflight = bool(pending.get("preflight"))
        try:
            with self._usage_scope(run):
                run.status = RunStatus.RUNNING
                self.loop.turn(
                    run, session, state,
                    on_event=on_event,
                    persist=lambda: self.store.persist(force=True),
                    resume={"kind": "clarification", "preflight": preflight, "payload": pending},
                    run_preflight_check=True,
                )
                self._snapshot_plan(run, state)
        except LLMException as exc:
            self._fail(run, exc.message, on_event)
        except AgentException as exc:
            self._fail(run, exc.message, on_event)
        except Exception as exc:  # noqa: BLE001
            self._fail(run, f"Agent 运行异常：{exc}", on_event)
        finally:
            run.finished_at = run.finished_at or time.time()
            session.task_state = state.to_dict()
            self._trace_outcome(run, session)
            self._end_trace(run, state)
            self.store.persist(force=True)
        return run

    #: 两个等待态各自的终止文案（不可共用，语义不同）。
    _DENY_MESSAGES = {
        RunStatus.WAITING_CONFIRMATION: "用户拒绝授权，运行终止",
        RunStatus.WAITING_CLARIFICATION: "用户未回答澄清问题，运行终止",
    }

    def deny(self, run_id: str, *, on_event: EventCallback | None = None) -> AgentRun:
        run, kind = self.store.claim_wait(run_id)
        if not kind:
            return run
        self._fail(run, self._DENY_MESSAGES[run.status], on_event)
        self.store.persist(force=True)
        return run

    def cancel(self, run_id: str) -> AgentRun:
        run = self.store.get_run(run_id)
        if run.status in (RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RUNNING):
            run.cancel_requested = True
            self.store.persist(force=True)
        elif run.status in (RunStatus.WAITING_CONFIRMATION, RunStatus.WAITING_CLARIFICATION):
            run = self.deny(run_id)
        return run

    # ------------------------------------------------------------------ 多轮接续
    def _hydrate_state(self, session: AgentSession, user_request: str) -> TaskState:
        """接续 / 重置会话级 TaskState；「改要求 / 追问」由 Loop 就地处理，这里只管 new_task。"""
        state = TaskState.from_dict(session.task_state)
        if state is None or state.closed:
            return TaskState.initial(user_request)
        if state.follow_up_kind(user_request) == "new_task":
            return TaskState.initial(user_request)
        return state

    def _snapshot_plan(self, run: AgentRun, state: TaskState) -> None:
        """把本轮实际规划 + 执行的动作写成 run.plan 快照（goal/steps 键不变，只读消费）。"""
        steps: list[dict[str, Any]] = []
        for record in run.tool_calls:
            steps.append({"tool": record.tool, "arguments": dict(record.arguments), "expected_output": ""})
        for action in state.pending_actions:
            steps.append(action.model_dump(mode="json"))
        run.plan = {"goal": state.goal or run.user_request, "steps": steps}

    # ------------------------------------------------------------------ DecisionTrace
    def _begin_trace(self, run: AgentRun) -> None:
        try:
            from app.agent.trace.decision_trace import DecisionTrace

            self._decision_traces[run.id] = DecisionTrace(trace_id=run.id, request=run.user_request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DecisionTrace 初始化失败：%s", exc)

    def _end_trace(self, run: AgentRun, state: TaskState | None = None) -> None:
        """补全 DecisionTrace 并落盘；决策跳由 Loop 发过的 planning 事件重建。"""
        try:
            trace = self._decision_traces.pop(run.id, None)
            if trace is None:
                return
            trace.final_answer = run.final_answer or ""
            trace.answer_source = run.answer_source or ""
            trace.success = str(run.status) == "completed"
            trace.remote_calls = int(run.token_ledger.llm_calls)
            trace.remote_escalations = int(run.token_ledger.remote_escalations)
            trace.local_calls = int(run.tool_call_count)
            trace.latency_ms = int(run.elapsed() * 1000)
            trace.tool_calls = [c.to_dict() for c in run.tool_calls]
            trace.tool_results_summary = [
                {"tool": c.tool, "status": c.status, "summary": c.result.summary if c.result else "",
                 "signals": list(c.result.signals) if c.result else []}
                for c in run.tool_calls
            ]
            # task_spec 字段改由 TaskState 快照填充（键名保留）。
            if state is not None:
                trace.task_spec = state.compact_for_remote()
            for event in run.events:
                if event.type != "planning":
                    continue
                payload = event.payload or {}
                stage = payload.get("stage")
                if stage not in ("task_understood", "local_direct", "remote_escalated", "dynamic_stop"):
                    continue
                trace.record_decision({
                    "step": "decide",
                    "stage": stage,
                    "tool": payload.get("tool"),
                    "source": payload.get("source"),
                    "confidence": payload.get("confidence"),
                    "rationale": str(payload.get("rationale") or "")[:200],
                })
            from app.agent.trace.decision_trace import write_trace

            write_trace(trace)
        except Exception as exc:  # noqa: BLE001 - 埋点失败不得影响运行
            logger.warning("DecisionTrace 写入失败：%s", exc)

    def _trace_route(self, run: AgentRun, session: AgentSession) -> None:
        """本地 Router shadow 埋点（只记录，不改变行为）。"""
        try:
            from app.local_router.contract import RouterRequest
            from app.local_router.trace import shadow_route

            shadow_route(
                run_id=run.id,
                session_id=session.id,
                request=RouterRequest(
                    utterance=run.user_request,
                    bound_dataset_id=(session.dataset_ids[0] if session.dataset_ids else None),
                ),
                rules_mode="agent",
                rules_reason="unified_stateful_loop",
            )
        except Exception:  # noqa: BLE001
            pass

    def _trace_outcome(self, run: AgentRun, session: AgentSession) -> None:
        try:
            from app.local_router.trace import record_outcome

            record_outcome(
                run_id=run.id,
                session_id=session.id,
                status=str(run.status),
                planned_tools=[str(step.get("tool")) for step in ((run.plan or {}).get("steps") or [])],
                executed_tools=[str(record.tool) for record in run.tool_calls],
                error=run.error or "",
                elapsed=round(run.elapsed(), 3),
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 事件与终态
    def _emit(self, run: AgentRun, event_type: str, payload: dict[str, Any], on_event: EventCallback | None) -> None:
        # 事件数硬上限：终态事件永远放行，否则连失败原因都发不出去。
        if event_type not in _TERMINAL_EVENTS and len(run.events) >= MAX_EVENTS_PER_RUN:
            raise AgentException(f"单次运行的事件数达到上限 {MAX_EVENTS_PER_RUN}，判定为异常循环，已终止")
        event = AgentEvent(seq=len(run.events) + 1, run_id=run.id, type=event_type, payload=payload)
        run.events.append(event)
        if on_event:
            on_event(event)

    def _fail(self, run: AgentRun, message: str, on_event: EventCallback | None) -> None:
        if not run.answer_source:
            run.answer_source = NO_ANSWER
        run.error = message
        run.status = RunStatus.FAILED
        run.finished_at = run.finished_at or time.time()
        self._emit_usage(run, on_event)
        self._emit(run, "failed", {"error": message, "answer_source": describe(run.answer_source)}, on_event)

    def _emit_usage(self, run: AgentRun, on_event: EventCallback | None) -> None:
        try:
            self._emit(run, "usage", {"token_usage": run.token_ledger.to_dict()}, on_event)
        except Exception:  # noqa: BLE001 - 账本上报失败不影响主流程
            pass
