"""统一 Stateful Agent Loop：Observe → Decide → Execute → Update → Continue/Finish。

重构后 Agent 的**唯一**控制流。旧体系（intent 硬分流 / 三套规划入口 / 两条执行
引擎 / 四层 DecisionProvider）全部收敛到这里：

* 决策只有一个分层序列（取消 → 槽位/preflight → 待办队首 → 客观信号 →
  本地 Router → 确定性 playbook → Remote 升级 → 对话），每跳带来源/置信度/理由；
* 状态只有一个 :class:`~app.agent.state.TaskState`，跨 Run 持续；
* 三层资源调度：确定性工具不调模型；Qwen/词法 Router 只做单步低成本决策；
  Remote LLM 仅在低置信/高不确定/开放式复杂任务时介入**一次**，且只产出
  结构化决策写进待办队列——永远不直接执行工具；
* 增量决策：每跳只看当前 TaskState 的紧凑视图，不做一次性大规划。

硬约束（TR-4.2）：本模块不得 import app.agent.decision / app.agent.planner /
app.agent.task_spec*。失败分类与熔断是从旧 Replanner 迁入的本地实现。
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent import playbooks as pb
from app.agent.answer_source import (
    LLM_ERROR_FALLBACK,
    NO_ANSWER,
    PLATFORM_RULES_CHAT,
    PLATFORM_RULES_NOTICE,
    PLATFORM_RULES_SUMMARY,
    REMOTE_LLM_CHAT,
    REMOTE_LLM_SUMMARY,
    describe,
)
from app.agent.clarify import ANSWERS_KEY
from app.agent.context.builder import ContextBuilder
from app.agent.context.models import AgentContext
from app.agent.executor.executor import AgentExecutor
from app.agent.intent import classify
from app.local_router.contract import Intent
from app.agent.llm.base import LLMException, LLMMessage, fallback_allowed
from app.agent.local_chat import local_reply, local_result_summary, no_llm_notice
from app.agent.permission.models import ROLE_PERMISSIONS
from app.agent.preflight import PreflightInput, run_preflight
from app.agent.runtime.models import AgentRun, AgentSession, RunStatus
from app.agent.runtime.step_resolution import resolve_arguments
from app.agent.state import PendingAction, Phase, TaskState, MAX_FACT_TOOLS
from app.agent.validator.validator import AgentResultValidator
from app.core.config import settings
from app.core.exceptions import AgentException, ValidationException
from app.tools.context import ToolExecutionContext

logger = logging.getLogger("xiaoluo.agent.loop")

#: 本地 Router 采纳阈值（与旧 _local_direct_plan 同口径；低于此值转 playbook/升级）。
ROUTER_MIN_CONFIDENCE = 0.45
#: 确定性长链优先于 Router 的 playbook（旧 _rule_plan 的高优先级分支）。
CHAIN_PRIORITY = frozenset({"merge", "workflow", "modeling", "transform", "comprehensive"})
#: 开放式/方案比较类措辞：词法 Router 无法可靠处理，直接升级 Remote 一次。
OPEN_ENDED_MARKERS = (
    "你看着办", "你来决定", "帮我决定", "多比较", "几种思路", "多种方案",
    "怎么做好", "怎么做比较好", "自由发挥", "综合判断", "哪个更好",
)
#: 知识性提问标志（命中后不再硬选工具）。
_KNOWLEDGE_MARKERS = ("什么是", "是什么", "为什么", "怎么办", "啥是", "如何理解", "讲讲", "介绍一下")
#: 但出现这些动作词时仍是任务请求（如「分析一下为什么缺失多」）。
_TASK_MARKERS = ("检查", "看看", "分析", "统计", "多少", "分布", "画图", "生成",
                 "训练", "清洗", "删除", "填充", "合并", "预测", "建模", "概览", "质量")

#: 单步最多尝试次数（旧 Replanner.MAX_ATTEMPTS_PER_STEP 迁入）。
MAX_ATTEMPTS_PER_STEP = 2
#: 重规划/重试总闸（旧 Replanner.assert_replan_budget 迁入）。
MAX_LOOP_REPLANS = 24
#: 单轮事件数硬上限（旧 runtime.MAX_EVENTS_PER_RUN 同值；Loop 自持常量不反向依赖 runtime）。
MAX_EVENTS_PER_RUN = 2000
#: 对话发给 Remote 时最多附带的历史条数（不重复发送完整历史）。
CHAT_HISTORY_LIMIT = 8


class LoopLimitExceeded(AgentException):
    """Loop 熔断：步数 / 工具调用数 / 时长 / Token / 重试次数超限。"""

    http_status = 400
    default_code = "AGENT_LOOP_LIMIT_EXCEEDED"
    default_message = "Agent loop limit exceeded"


# ---------------------------------------------------------------------- Remote schema
class RemoteDecision(BaseModel):
    """Remote LLM 唯一被允许的输出形状：一次战略决策，不是一份 20 步计划。"""

    action: Literal["execute_tool", "ask_user", "chat", "stop"] = "execute_tool"
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: 后续工具名（≤4），只允许当前候选集内的工具；由 Loop 入队本地执行。
    next_steps: list[str] = Field(default_factory=list, max_length=4)
    question: str = ""
    options: list[str] = Field(default_factory=list, max_length=8)
    answer: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------- 运行期上下文
@dataclass
class _Turn:
    run: AgentRun
    session: AgentSession
    state: TaskState
    on_event: Any
    context: AgentContext
    candidate_tools: set[str]
    primed: bool = False
    escalated: bool = False
    #: 下一个新动作的绝对步号（Run 内跨批次累计，0 基）。
    cursor: int = 0
    index_by_key: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    replans: int = 0
    #: Remote/对话层直接给出的最终回答（文本, answer_source）。
    direct_answer: tuple[str, str] | None = None
    any_ok: bool = False


# --------------------------------------------------------------------------- 主类
class AgentLoop:
    """一个 Agent Loop。无状态地持有基础设施，任务状态全部在 TaskState/Run 里。"""

    def __init__(
        self,
        data_engine: Any,
        *,
        experiment_service: Any | None = None,
        db: Any | None = None,
        llm: Any | None = None,
        registry: Any | None = None,
        executor: AgentExecutor | None = None,
        validator: AgentResultValidator | None = None,
        role: str = "analyst",
    ) -> None:
        self.data_engine = data_engine
        self.experiment_service = experiment_service
        self.db = db
        self.llm = llm
        if registry is None:
            from app.tools.registry import TOOL_REGISTRY

            registry = TOOL_REGISTRY
        self.registry = registry
        self.executor = executor or AgentExecutor(self.registry)
        self.validator = validator or AgentResultValidator()
        self.role = role

    # ============================================================ 入口（恢复点=同一循环）
    def turn(
        self,
        run: AgentRun,
        session: AgentSession,
        state: TaskState,
        *,
        on_event: Any = None,
        persist: Callable[[], None] | None = None,
        resume: dict[str, Any] | None = None,
        confirmed: bool = False,
        run_preflight_check: bool = True,
    ) -> str:
        """执行一轮「用户消息 → 终态/等待态」。返回 RunStatus 字符串。

        ``resume``：None=新消息轮；{"kind":"confirmation"}=高风险授权后继续；
        {"kind":"clarification","preflight":bool}=反问回答后继续。
        """
        ctx = _Turn(
            run=run, session=session, state=state, on_event=on_event,
            context=ContextBuilder(self.data_engine).build(
                run.user_request, dataset_ids=list(session.dataset_ids),
                role=self.role, history=session.history[:-1],
            ),
            candidate_tools=set(),
        )
        ctx.cursor = max((c.step_index for c in run.tool_calls), default=-1) + 1
        if resume:
            # 恢复路径的首轮决策在挂起前已经做完：只消费队列，不再对原文重放分层。
            ctx.primed = True
        self._emit(ctx, "route", {"mode": "agent", "reason": "unified_stateful_loop"})
        self._emit_usage(ctx)

        # ① 取消/停止熔断（显式取消等待中的运行由 runtime.cancel 处理；这里拦新消息）。
        if pb.is_cancel_command(run.user_request):
            return self._finish_notice(ctx, "好的，已停止当前操作。")

        # 纯问候/寒暄：本地模板直接应答（Remote 次数为 0），不进 preflight/工具链。
        # 「你好，请帮我分析数据」这类含任务的句子不命中（local_reply 只接纯寒暄）。
        if resume is None:
            greeting = local_reply(run.user_request)
            if greeting is not None and not self._looks_task_like(run.user_request):
                return self._complete(ctx, greeting, PLATFORM_RULES_CHAT)

        # 恢复路径：把等待中的动作重新放回队首（一次性授权凭据只对它生效）。
        authorized_key: str | None = None
        if resume and resume.get("kind") == "confirmation":
            action = self._requeue_waiting_action(ctx, resume)
            if action is not None:
                authorized_key = action.key()
                confirmed = True
        elif resume and resume.get("kind") == "clarification" and not resume.get("preflight"):
            self._requeue_waiting_action(ctx, resume)

        # Observe：元数据上下文 + 空数据版本提示。
        self._sync_dataset_context(ctx)
        self._emit(ctx, "planning", {"stage": "context_ready", "data_access": "metadata_only"})
        empty = self._datasets_without_version(ctx.context)
        if empty:
            return self._finish_no_data(ctx, empty)

        # ② Pre-flight / 必填槽位（新消息轮 与 preflight 反问恢复时执行）。
        do_preflight = run_preflight_check and settings.AGENT_PREFLIGHT_ENABLED and (
            resume is None or (resume.get("kind") == "clarification" and resume.get("preflight"))
        )
        if do_preflight:
            status = self._run_preflight(ctx)
            if status is not None:
                self._persist(persist)
                return status

        # 候选工具集：检索 + CORE 封顶（不把全部 schema 发给 Remote）。
        all_tools = self.registry.list()
        ContextBuilder(self.data_engine).with_tools(ctx.context, all_tools, registry=self.registry)
        ctx.candidate_tools = {
            str(t.get("name")) for t in (ctx.context.tool_context or {}).get("tools") or [] if t.get("name")
        }
        retrieval = (ctx.context.tool_context or {}).get("retrieval_scores") or {}
        self._emit(ctx, "planning", {
            "stage": "tools_retrieved", "count": len(ctx.candidate_tools),
            "tools": sorted(ctx.candidate_tools), "retrieval": retrieval,
        })

        # ③ 主循环。
        try:
            while True:
                if run.cancel_requested:
                    return self._fail(ctx, "运行已被用户取消")
                self._assert_circuit(ctx)

                action = state.pop_next()
                if action is None:
                    # ④ 客观信号 → 白名单工具（仅在首轮决策已铺开后，避免对老信号重复起链）。
                    action = self._signal_action(ctx) if ctx.primed else None
                    if action is None:
                        if not ctx.primed:
                            # ⑤ 首轮决策分层（确定性 → Router → playbook → Remote → 对话）。
                            keep_going = self._prime(ctx)
                            ctx.primed = True
                            if not keep_going:
                                break
                            continue
                        # 队列与信号都空了：收尾。
                        break
                # Execute（含等待态挂起；恢复点就是这个 while）。
                status, authorized_key = self._execute(ctx, action, authorized_key=authorized_key,
                                                       confirmed=confirmed)
                if status is not None:
                    self._persist(persist)
                    return status
        except LoopLimitExceeded:
            raise
        except AgentException:
            raise
        except LLMException:
            raise
        except Exception as exc:  # noqa: BLE001 — 边界异常统一失败，绝不静默吞成「已完成」
            logger.exception("Agent Loop 未捕获异常")
            return self._fail(ctx, f"Agent 运行异常：{exc}")

        # 反问/授权等待态在 break 路径上同样要原样返回（不能被收尾逻辑覆盖成 completed）。
        if str(run.status).startswith("waiting"):
            return str(run.status)
        if ctx.direct_answer is not None:
            text, source = ctx.direct_answer
            return self._complete(ctx, text, source)
        return self._finish_summary(ctx)

    # ============================================================ Decide：首轮分层
    def _prime(self, ctx: _Turn) -> bool:
        """首轮决策。返回 True=已有动作入队继续循环；False=进入收尾/对话终态。"""
        text = ctx.run.user_request
        state, ds = ctx.state, ctx.session.dataset_ids

        # 进行中任务的追问：读 TaskState 就地回答，零模型、不起工具链。
        # ★ 只有「真正的追问」（疑问/知识问句）才读状态作答；带任务动作词的祈使句
        #   （如「分析一下分布」）是**新的分析诉求**，应继续走确定性决策，否则会被
        #   follow_up_kind 的默认 state_question 吞掉、永远不执行工具。
        if state.completed_actions:
            kind = state.follow_up_kind(text)
            if kind == "state_question" and not self._looks_task_like(text):
                ctx.direct_answer = (self._state_grounded_answer(ctx), PLATFORM_RULES_SUMMARY)
                return False
            if kind == "constraint_change":
                state.apply_constraint_change(text)

        selection = pb.select_playbook(text, state, list(ds), base_index=ctx.cursor)

        # (a) 确定性长链优先（建模/工作流/合并/变换/综合分析），命中不调任何模型。
        if selection.name in CHAIN_PRIORITY and selection.actions:
            added = state.queue(selection.actions)
            if added:
                self._emit(ctx, "planning", {
                    "stage": "task_understood", "playbook": selection.name,
                    "steps": len(added), "source": "playbook",
                    "confidence": 1.0, "rationale": selection.reason or f"确定性 playbook: {selection.name}",
                })
                return True

        # (b) 本地 Router（唯一入口）：首分句、零成本词法/Qwen，异常诚实降级。
        outcome, router_error = self._route_local(ctx, pb.split_first_clause(text))

        # 升级触发条件先算出来：一旦存在明确升级信号，短链兜底不得抢在 Remote 前面。
        # ★ 本地 Router 不可用**不是**升级信号：FR-5 要求它诚实降级到词法/确定性层，
        # 否则 TF-IDF off / Qwen 缺失时，简单任务（行数/分布/异常值）也会被错误烧一次远程。
        escalate_reason = ""
        if outcome is not None and outcome.decision.escalate:
            escalate_reason = f"本地 Router 升级：{outcome.decision.escalate_reason or 'ambiguous'}"
        elif self._is_open_ended(text):
            escalate_reason = "开放式/多方案比较任务"
        elif self._is_knowledge_question(text) and not state.completed_actions:
            escalate_reason = "开放式知识问题"
        elif state.uncertainty >= 0.7:
            escalate_reason = f"任务不确定性 {state.uncertainty:.2f}"

        # (c) Router 高置信单步工具直接执行（知识问句除外）；缺槽位则反问。
        # ★ 两个让位条件：①升级信号已存在（开放式/知识问题/高不确定）→ 让位 Remote；
        #   ②Router 选中了需要 column 等**无法自动补全**必填参数的工具（如 eda.distribution）
        #     → 降级确定性 playbook（信号层「不收录需要 column 的工具」的同一原则）。
        if outcome is not None and not outcome.decision.escalate and not escalate_reason:
            decision = outcome.decision
            if decision.missing:
                self._ask_missing_slots(ctx, decision.tool or "", list(decision.missing))
                return False
            if decision.tool and decision.confidence >= ROUTER_MIN_CONFIDENCE and not self._is_knowledge_question(text):
                if decision.tool in ctx.candidate_tools:
                    if self._router_tool_needs_unfillable(ctx, decision.tool, dict(decision.params or {})):
                        state.note_low_confidence("local_router", min(decision.confidence, 0.4))
                    else:
                        action = PendingAction(
                            tool=decision.tool,
                            arguments=dict(decision.params or {}),
                            source="local_router",
                            rationale=f"本地 Router 选中 {decision.tool}（置信度 {decision.confidence:.2f}）",
                        )
                        if state.queue(action):
                            self._emit(ctx, "planning", {
                                "stage": "local_direct", "tool": decision.tool,
                                "source": outcome.source, "confidence": round(decision.confidence, 3),
                            })
                            return True
                else:
                    state.note_low_confidence("local_router", min(decision.confidence, 0.4))

        # (d) 确定性短链兜底（质量/EDA/数据接纳）——升级信号已存在时让位于 Remote。
        if not escalate_reason and selection.actions and selection.name in {"quality", "eda", "intake"}:
            added = state.queue(selection.actions)
            if added:
                self._emit(ctx, "planning", {
                    "stage": "task_understood", "playbook": selection.name,
                    "steps": len(added), "source": "playbook", "confidence": 0.9,
                    "rationale": selection.reason or f"确定性 playbook: {selection.name}",
                })
                return True

        # (e) Remote 升级（一轮最多一次战略决策；中途失败后的再升级在失败处理里另有闸门）。
        if escalate_reason and self.llm is not None:
            queued = self._remote_decide(ctx, escalate_reason)
            if queued is not None:
                return queued if queued else False
            # Remote 调用失败且允许降级：落到本地对话/收尾（下面）。

        # (f) 无数据集 / 纯对话 / 模型均不可用 → 对话终态。
        local = local_reply(text)
        if local is not None and not self._looks_task_like(text):
            ctx.direct_answer = (local, PLATFORM_RULES_CHAT)
            return False
        # 纯对话（非任务请求）→ 走远程/本地对话，**不**走 need_dataset 提示：
        # 「讲个笑话」不该被反问「请先选数据集」。
        if not self._looks_task_like(text):
            self._chat(ctx, knowledge=self._is_knowledge_question(text))
            return False
        if selection.name == "need_dataset" and not state.completed_actions:
            ctx.direct_answer = (
                no_llm_notice("需要先选择数据集：请在左侧选择一个数据集后再让我分析。"),
                PLATFORM_RULES_NOTICE,
            )
            return False
        self._chat(ctx, knowledge=self._is_knowledge_question(text))
        return False

    # ============================================================ Decide：本地 Router
    def _route_local(self, ctx: _Turn, utterance: str):
        """调用本地 Router 一次/轮；任何异常都诚实降级（Qwen 缺陷不得炸 Loop）。"""
        try:
            from app.local_router.router import route_request_detailed

            req = {
                "utterance": utterance,
                "bound_dataset_id": ctx.session.dataset_ids[0] if ctx.session.dataset_ids else None,
                "bound_dataset_name": self._bound_dataset_name(ctx),
                "available_datasets": self._available_datasets(ctx),
                "available_columns": self._available_columns(ctx),
                "recent_tools": list(ctx.state.completed_tools())[-8:],
            }
            outcome = route_request_detailed(req)
            if outcome.source == "qwen":
                ctx.run.token_ledger.record_qwen_call()
            if outcome.decision.confidence < ROUTER_MIN_CONFIDENCE and outcome.decision.tool:
                ctx.state.note_low_confidence("local_router", float(outcome.decision.confidence))
            return outcome, None
        except Exception as exc:  # noqa: BLE001 — 本地决策层永远是可缺省层
            logger.warning("本地 Router 降级：%s", exc)
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"

    def _router_tool_needs_unfillable(self, ctx: _Turn, tool: str, params: dict[str, Any]) -> bool:
        """本地 Router 选中的工具是否有「无法自动补全」的必填参数（如 column）。

        dataset_id 可由会话上下文自动补全，不算；column / target 等需要真实数据列名，
        词法 Router 给不出 ⇒ 返回 True，调用方应降级确定性 playbook 而非硬执行。
        """
        try:
            schema = self.registry.get(tool).input_schema or {}
        except Exception:  # noqa: BLE001
            return False
        required = list(schema.get("required") or [])
        ids = ctx.session.dataset_ids
        for name in required:
            if name == "dataset_id":
                if len(ids) == 1:
                    continue  # 单数据集可自动补全
                if params.get("dataset_id") is not None:
                    continue
                return True  # 多数据集 / 无数据集时 dataset_id 缺失
            if params.get(name) in (None, ""):
                return True
        return False

    # ============================================================ Decide：信号层
    def _signal_action(self, ctx: _Turn) -> PendingAction | None:
        ds_id = ctx.session.dataset_ids[0] if len(ctx.session.dataset_ids) == 1 else None
        return pb.action_for_signal(
            list(ctx.state.signals),
            dataset_id=ds_id,
            last_tool=(ctx.state.last_result or {}).get("tool"),
            done_tools=ctx.state.completed_tools(),
            pending_tools={a.tool for a in ctx.state.pending_actions},
            missing_strategy=pb.missing_fill_strategy(ctx.run.user_request, ctx.state),
        )

    # ============================================================ Decide：Remote 升级
    def _remote_decide(self, ctx: _Turn, reason: str) -> bool | None:
        """一次结构化战略决策 → 入队本地执行。Remote 永不直接执行工具。

        返回 True=已入队继续；False=已给终态（chat/stop）；None=Remote 不可用/失败。
        """
        compact = ctx.state.compact_for_remote()
        tool_lines = [
            f"- {name}: {self._tool_brief(name)}"
            for name in sorted(ctx.candidate_tools)
        ]
        system = (
            "你是数据分析 Agent 的战略决策层。本地执行循环已经获得了一些真实数据事实，"
            "你只决定下一步，不执行操作、不写长计划。可用工具仅限给出的候选集。"
            "输出一个结构化决策：execute_tool（选一个工具+参数，可附最多 4 个后续工具名）、"
            "ask_user（缺少必要信息，必须带可选项）、chat（直接回答开放/知识问题）或 stop。"
            "参数中需要数据集时用 dataset_id；不确定的列名/目标不要臆造，改为 ask_user。"
        )
        user_payload = json.dumps(
            {
                "用户请求": ctx.run.user_request,
                "升级原因": reason,
                "当前任务状态": compact,
                "候选工具": tool_lines,
            },
            ensure_ascii=False, default=str,
        )
        try:
            data = self.llm.structured_output(
                [LLMMessage(role="system", content=system),
                 LLMMessage(role="user", content=user_payload)],
                RemoteDecision,
            )
            decision = RemoteDecision.model_validate(data or {})
        except LLMException as exc:
            if not fallback_allowed(exc):
                raise
            logger.warning("Remote 升级失败（%s），本地继续：%s", type(exc).__name__, exc)
            return None
        except Exception as exc:  # noqa: BLE001 — 结构化解析失败等同本次远程不可用
            logger.warning("Remote 决策解析失败：%s", exc)
            return None

        ctx.escalated = True
        ctx.run.token_ledger.record_escalation()
        self._emit(ctx, "planning", {
            "stage": "remote_escalated", "reason": reason,
            "action": decision.action, "tool": decision.tool,
            "rationale": decision.rationale[:200], "confidence": 1.0, "source": "remote_llm",
        })

        if decision.action == "chat":
            ctx.direct_answer = ((decision.answer or "").strip() or "（远程模型未给出回答）", REMOTE_LLM_CHAT)
            return False
        if decision.action == "ask_user":
            self._ask_remote_question(ctx, decision)
            return False
        if decision.action == "stop":
            return False

        queued: list[PendingAction] = []
        if decision.tool and decision.tool in ctx.candidate_tools:
            queued.append(PendingAction(
                tool=decision.tool, arguments=dict(decision.arguments or {}),
                source="remote", rationale=decision.rationale[:200],
            ))
        for name in (decision.next_steps or [])[:4]:
            if name in ctx.candidate_tools and name != decision.tool:
                queued.append(PendingAction(tool=name, source="remote",
                                            rationale="Remote 战略决策的后续步骤"))
        added = ctx.state.queue(queued)
        if not added:
            # Remote 没给出任何本地可执行动作：不再发起第二次远程，直接收尾。
            return False
        return True

    # ============================================================ Execute
    def _execute(self, ctx: _Turn, action: PendingAction, *, authorized_key: str | None,
                 confirmed: bool) -> tuple[str | None, str | None]:
        """执行单个动作；返回 (终态status|None, 剩余授权key)。等待态/终态时 status 非空。"""
        run, session, state = ctx.run, ctx.session, ctx.state
        key = action.key()
        idx = ctx.index_by_key.get(key)
        if idx is None:
            idx = ctx.cursor
            ctx.cursor += 1
            ctx.index_by_key[key] = idx
        attempt = ctx.attempts.get(key, 0) + 1
        ctx.attempts[key] = attempt

        tool_ctx = self._tool_context(session, user_request=run.user_request)
        try:
            schema = self.registry.get(action.tool).input_schema or {}
            resolved = resolve_arguments(action, run, session, ctx.context, schema)
            resolved = self._fill_required(action, resolved, schema, ctx)
        except ValidationException as exc:
            return self._handle_failure(ctx, action, idx, attempt, [getattr(exc, "message", str(exc))],
                                        resolved=None, authorized_key=authorized_key)

        # 一次性授权凭据：只对被确认的那一个动作生效，用掉即失效。
        # ★ 绝不能写成 ``authorized_key is None or authorized_key == key``：
        #   授权用掉后 authorized_key 置 None，那会让后续每个高风险动作都命中
        #   「authorized_key is None」而被整轮放行 —— 一次确认 = 放行整条高风险链。
        step_confirmed = confirmed and authorized_key == key
        if step_confirmed:
            authorized_key = None

        self._emit(ctx, "tool_call", {
            "step_index": idx, "tool": resolved.tool, "arguments": resolved.arguments,
            "source": action.source, "attempt": attempt,
        })
        run.token_ledger.record_tool_call()
        # 本轮已回答的结构化反问注入工具上下文（agent.clarify 幂等读它）。
        tool_ctx.extra[ANSWERS_KEY] = dict(run.clarification_answers or {})
        record = self.executor.execute_step(
            resolved, tool_ctx, self._services(),
            confirmed=step_confirmed, attempt=attempt, step_index=idx,
        )
        run.tool_calls.append(record)

        if record.status == "needs_confirmation":
            run.status = RunStatus.WAITING_CONFIRMATION
            run.pending_confirmation = {"call": record, "step_index": idx, "action": resolved.model_dump(mode="json")}
            self._emit(ctx, "permission", {
                "stage": "confirmation_required", "tool": resolved.tool,
                "reason": record.error, "step_index": idx,
            })
            return str(RunStatus.WAITING_CONFIRMATION), authorized_key
        if record.status == "needs_clarification":
            clarification = record.clarification.to_dict() if hasattr(record.clarification, "to_dict") else {}
            run.status = RunStatus.WAITING_CLARIFICATION
            run.pending_clarification = {
                **clarification, "step_index": idx, "tool": resolved.tool,
                "action": resolved.model_dump(mode="json"),
            }
            self._emit(ctx, "clarification", {
                **clarification, "stage": "clarification_required",
                "tool": resolved.tool, "step_index": idx,
            })
            return str(RunStatus.WAITING_CLARIFICATION), authorized_key
        if record.status == "denied":
            return self._fail(ctx, f"第 {idx} 步被拒绝：{record.error}"), authorized_key

        compact = self._result_view(ctx, record)
        self._emit(ctx, "tool_result", {
            "step_index": idx, "tool": resolved.tool, "status": record.status,
            "summary": record.result.summary if record.result else record.error,
            "llm_result": compact,
        })
        self._emit_usage(ctx)

        errors: list[str] = []
        if record.status == "ok":
            validation = self.validator.validate(resolved, record.result, self._output_schema(resolved.tool))
            self._emit(ctx, "validation", {
                "step_index": idx, "tool": resolved.tool, "valid": validation.valid,
                "errors": validation.errors, "warnings": validation.warnings,
            })
            if validation.valid:
                state.ingest_result(resolved, record)
                ctx.any_ok = True
                run.token_ledger.record_step()
                return None, authorized_key
            errors = list(validation.errors)
        else:
            errors = [record.error or "工具执行失败"]
        return self._handle_failure(ctx, resolved, idx, attempt, errors, resolved=resolved,
                                    authorized_key=authorized_key)

    def _handle_failure(self, ctx: _Turn, action: PendingAction, idx: int, attempt: int,
                        errors: list[str], *, resolved: PendingAction | None,
                        authorized_key: str | None) -> tuple[str | None, str | None]:
        """失败三策略（旧 Replanner 规则迁入）：依赖断裂即止 / 参数错误不重试 / 瞬时≤2 次。"""
        kind = _failure_kind(errors)
        ctx.state.record_failure(action.tool, errors)
        ctx.replans += 1
        self._assert_circuit(ctx)

        retry = False
        notes = ""
        if kind == "dependency":
            notes = f"第 {idx} 步失败且后续步骤依赖其输出，为避免级联错误停止该支路。"
            self._drop_dependents(ctx, idx)
        elif kind == "parameter":
            notes = f"第 {idx} 步参数校验失败，不重复执行相同参数：{'；'.join(errors)[:160]}"
        elif attempt < MAX_ATTEMPTS_PER_STEP:
            retry = True
            notes = f"第 {idx} 步失败（可能瞬时），第 {attempt + 1} 次尝试。"
        else:
            notes = f"第 {idx} 步重试 {attempt} 次后仍失败，继续后续动作。"
        self._emit(ctx, "replanning", {
            "step_index": idx, "notes": notes, "retry": retry,
            "attempt": attempt, "remaining": len(ctx.state.pending_actions),
        })
        if retry and resolved is not None:
            ctx.state.pending_actions.insert(0, resolved)
            return None, authorized_key

        # 动态复杂度：失败把不确定性顶上来后，允许中途升级 Remote 一次。
        if ctx.state.uncertainty >= 0.7 and not ctx.escalated and self.llm is not None:
            queued = self._remote_decide(ctx, f"执行 {action.tool} 失败后不确定性升高")
            if queued:
                return None, authorized_key

        if not ctx.state.pending_actions and not ctx.any_ok:
            return self._fail(ctx, f"第 {idx} 步失败：{'；'.join(errors)[:200]}"), authorized_key
        return None, authorized_key

    # ============================================================ Finish
    def _finish_summary(self, ctx: _Turn) -> str:
        """收尾：本地结构化汇总优先；仅升级过/明确要求综合时允许一次远程总结。"""
        records = [c for c in ctx.run.tool_calls if c.status == "ok"]
        if not records:
            if ctx.escalated:
                # Remote 明确 stop/未排动作：不是失败，给一句诚实的状态说明收尾。
                return self._complete(ctx, "当前没有需要继续执行的工具操作；你可以告诉我下一步想分析什么。",
                                      PLATFORM_RULES_NOTICE)
            return self._fail(ctx, ctx.state.error_history[-1] if ctx.state.error_history else "任务未能完成")
        fallback = local_result_summary(ctx.run.tool_calls)
        # 只在用户**明确要求**综合/总结时才花一次远程汇总；升级过的复杂任务同样优先
        # 本地结构化汇总（Remote 参与的是决策，不默认再参与一次总结）。
        wants_wrapup = any(
            w in ctx.run.user_request for w in ("总结", "结论", "综合报告", "给个报告", "解读一下")
        )
        if self.llm is None or not wants_wrapup:
            return self._complete(ctx, fallback, PLATFORM_RULES_SUMMARY)
        views = []
        for call in records[-8:]:
            view = self._result_view(ctx, call)
            if view is not None:
                views.append({"tool": call.tool, "result": view})
        system = (
            "你是数据分析助手。基于真实工具结果回答用户请求：先给 3-5 条结论式关键发现，"
            "再列数据质量问题与风险，最后给可操作建议。只引用有工具记录支撑的关键数值，"
            "400 字以内；不得声称做过没有记录的操作。"
        )
        try:
            response = self.llm.chat([
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=(
                    f"用户请求：{ctx.run.user_request}\n\n工具结果（紧凑视图）：\n"
                    f"{json.dumps(views, ensure_ascii=False, default=str) or '（无）'}")),
            ])
            text = (response.content or "").strip()
            if not text:
                return self._complete(ctx, fallback, PLATFORM_RULES_SUMMARY)
            return self._complete(ctx, text, REMOTE_LLM_SUMMARY)
        except Exception as exc:  # noqa: BLE001 — 总结失败的降级口径与旧链路一致
            if not fallback_allowed(exc):
                raise
            logger.warning("远程汇总失败，降级本地汇总：%s", exc)
            return self._complete(ctx, fallback, LLM_ERROR_FALLBACK)

    def _chat(self, ctx: _Turn, *, knowledge: bool) -> None:
        text = ctx.run.user_request
        local = local_reply(text)
        if self.llm is None:
            # 本地确定性应答优先；答不上给状态说明（来源如实区分 CHAT / NOTICE）。
            if local is not None:
                ctx.direct_answer = (local, PLATFORM_RULES_CHAT)
            else:
                ctx.direct_answer = (no_llm_notice("远程模型未启用" if knowledge else ""),
                                     PLATFORM_RULES_NOTICE)
            return
        history = [
            LLMMessage(role=str(m.get("role")), content=str(m.get("content")))
            for m in ctx.session.history[-CHAT_HISTORY_LIMIT - 1:-1]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        system = "你是小洛实验室的数据分析助手，简洁友好地回答；涉及数据分析的问题可说明平台能直接调用工具完成。"
        try:
            response = self.llm.chat([LLMMessage(role="system", content=system), *history,
                                      LLMMessage(role="user", content=text)])
            answer = (response.content or "").strip()
            if answer:
                ctx.direct_answer = (answer, REMOTE_LLM_CHAT)
            elif local is not None:
                ctx.direct_answer = (local, PLATFORM_RULES_CHAT)
            else:
                ctx.direct_answer = (no_llm_notice("远程模型返回为空"), PLATFORM_RULES_CHAT)
        except Exception as exc:  # noqa: BLE001
            if not fallback_allowed(exc):
                raise
            logger.warning("远程对话失败，降级本地应答：%s", exc)
            # ★ 来源不许撒谎：本地答得上 → LLM_ERROR_FALLBACK（真降级）；答不上 → NO_ANSWER。
            if local is not None:
                ctx.direct_answer = (local, LLM_ERROR_FALLBACK)
            else:
                ctx.direct_answer = (no_llm_notice(reason=str(exc)), NO_ANSWER)

    def _state_grounded_answer(self, ctx: _Turn) -> str:
        """分析过程中的追问：只用 TaskState 里的真实事实回答，零模型、零工具。"""
        state = ctx.state
        lines: list[str] = []
        last = state.last_result or {}
        if last.get("summary"):
            lines.append(f"最新进展（{last.get('tool')}）：{last['summary']}")
        for finding in state.findings[:4]:
            lines.append(f"· {finding}")
        # ★ 追问要能引用真实事实而非笼统的「N 个问题」：从 facts 里提取质量/异常等
        #   issue 的 message（如「字段 age 存在 5 个缺失值 (6.25%)」）。
        for tool, data in list(state.facts.items())[-MAX_FACT_TOOLS:]:
            if not isinstance(data, dict):
                continue
            for issue in (data.get("issues") or [])[:3]:
                if isinstance(issue, dict) and issue.get("message"):
                    lines.append(f"· {issue['message']}")
        if not lines:
            return "当前分析还没有落地的结论，我可以继续执行原任务；你也可以告诉我下一步想看什么。"
        lines.append("需要的话我可以继续未完成的分析步骤。")
        return "\n".join(lines)

    # ============================================================ 等待态反问
    def _ask_missing_slots(self, ctx: _Turn, tool: str, missing: list[str]) -> None:
        payload = {
            "code": "agent.missing_slot",
            "question": f"执行 {tool} 还缺少必要信息：{', '.join(missing)}。请补充后我再继续。",
            "options": [], "default": None, "missing": list(missing), "tool": tool,
        }
        run = ctx.run
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = {**payload, "stage": "clarification_required"}
        self._emit(ctx, "clarification", {**payload, "stage": "clarification_required"})

    def _ask_remote_question(self, ctx: _Turn, decision: RemoteDecision) -> None:
        payload = {
            "code": "agent.remote_need_info",
            "question": decision.question or "继续这个分析前需要你补充一点信息。",
            "options": [{"value": str(o), "label": str(o)} for o in (decision.options or [])[:8]],
            "default": str(decision.options[0]) if decision.options else None,
            "rationale": decision.rationale[:200],
        }
        run = ctx.run
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = {**payload, "stage": "clarification_required"}
        self._emit(ctx, "clarification", {**payload, "stage": "clarification_required"})

    # ============================================================ Pre-flight
    def _run_preflight(self, ctx: _Turn) -> str | None:
        run, session = ctx.run, ctx.session
        columns: list[dict[str, Any]] = []
        if settings.AGENT_PREFLIGHT_READ_SCHEMA and len(session.dataset_ids) == 1:
            try:
                columns = list(self.data_engine.column_schema(session.dataset_ids[0]) or [])
            except Exception:  # noqa: BLE001 — 读不到列结构就跳过该项检查
                columns = []
        meta: dict[int, dict[str, Any]] = {}
        for key, value in (ctx.context.dataset_context or {}).items():
            try:
                meta[int(key)] = dict(value)
            except (TypeError, ValueError):
                continue
        intent = classify(run.user_request, has_datasets=bool(session.dataset_ids))
        # ★ 纯对话不进 Pre-flight：dataset_required 等检查只对数据任务有意义，
        #   否则「讲个笑话」这类闲聊也会被反问「请关联数据集」。
        if intent.value == Intent.CHAT:
            return None
        pre = run_preflight(PreflightInput(
            user_request=run.user_request, intent=intent,
            dataset_ids=list(session.dataset_ids), dataset_meta=meta, columns=columns,
            answers=dict(run.clarification_answers),
        ))
        run.preflight = pre.to_dict()
        self._emit(ctx, "preflight", {
            "outcome": str(pre.outcome), "checks_run": pre.checks_run,
            "findings": [f.to_dict() for f in pre.findings],
            "clarifications": [c.to_dict() for c in pre.clarifications],
            "resolved": pre.resolved,
        })
        if not pre.needs_user:
            return None
        question = pre.first_question()
        payload = question.to_dict() if question else {
            "code": "preflight.unknown", "question": "需要补充信息才能继续", "options": [],
        }
        payload.update({"outcome": str(pre.outcome), "checks_run": pre.checks_run, "preflight": True})
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = payload
        self._emit(ctx, "clarification", {**payload, "stage": "clarification_required"})
        return str(RunStatus.WAITING_CLARIFICATION)

    # ============================================================ 辅助
    def _requeue_waiting_action(self, ctx: _Turn, resume: dict[str, Any]) -> PendingAction | None:
        """从等待载荷/历史记录还原未完成动作，插回队首；preflight 反问无动作返回 None。"""
        pending = resume.get("payload") or {}
        raw = pending.get("action")
        if isinstance(raw, dict) and raw.get("tool"):
            action = PendingAction.model_validate(raw)
        else:
            step_index = pending.get("step_index")
            record = next(
                (c for c in reversed(ctx.run.tool_calls)
                 if step_index is None or c.step_index == step_index),
                None,
            )
            if record is None or record.status not in ("needs_confirmation", "needs_clarification"):
                return None
            action = PendingAction(tool=record.tool, arguments=dict(record.arguments), source="resume")
        # 恢复的动作必须在队首，且不受 completed 去重影响（它从未成功过）。
        ctx.state.pending_actions.insert(0, action)
        idx = (resume.get("payload") or {}).get("step_index")
        if isinstance(idx, int):
            ctx.index_by_key[action.key()] = idx
        return action

    def _fill_required(self, original: PendingAction, resolved: PendingAction,
                       schema: dict[str, Any], ctx: _Turn) -> PendingAction:
        """dataset_id 必填补全 / 多集报错 / 越权校验 / 必填检查（旧 _resolve_step_arguments 尾部迁入）。"""
        args = dict(resolved.arguments or {})
        required = list(schema.get("required") or [])
        ids = ctx.session.dataset_ids
        if "dataset_id" in required:
            if args.get("dataset_id") is None:
                if len(ids) == 1:
                    args["dataset_id"] = ids[0]
                elif not ids:
                    raise ValidationException(f"工具 {resolved.tool} 需要 dataset_id，但当前会话没有关联数据集。")
                else:
                    raise ValidationException(f"工具 {resolved.tool} 需要明确的 dataset_id，但当前会话关联了多个数据集。")
            try:
                args["dataset_id"] = int(args["dataset_id"])
            except (TypeError, ValueError) as exc:
                raise ValidationException(f"工具 {resolved.tool} 的 dataset_id 必须是整数。") from exc
            if ids and args["dataset_id"] not in ids:
                raise ValidationException(f"工具 {resolved.tool} 指定的数据集不在当前会话关联范围内。")
        missing = [name for name in required if args.get(name) in (None, "")]
        if missing:
            raise ValidationException(f"工具 {resolved.tool} 缺少必要参数：{', '.join(missing)}")
        return PendingAction(tool=resolved.tool, arguments=args, expected_output=resolved.expected_output,
                             permission=resolved.permission, source=original.source,
                             rationale=original.rationale)

    def _drop_dependents(self, ctx: _Turn, failed_idx: int) -> None:
        needle = f"step{failed_idx + 1}."
        kept = [
            a for a in ctx.state.pending_actions
            if needle not in json.dumps(a.arguments, ensure_ascii=False, default=str)
        ]
        ctx.state.pending_actions = kept

    def _assert_circuit(self, ctx: _Turn) -> None:
        if ctx.replans >= MAX_LOOP_REPLANS:
            raise LoopLimitExceeded(f"循环重试次数达到上限 {MAX_LOOP_REPLANS}，判定为异常循环")
        calls = ctx.run.tool_call_count
        if calls > settings.AGENT_MAX_STEPS * 2:
            raise LoopLimitExceeded(f"工具调用次数 {calls} 超过循环上限")
        if ctx.run.elapsed() > 180.0:
            raise LoopLimitExceeded("运行时长超过 180s 上限")
        if ctx.run.token_ledger.actual_total_tokens >= settings.AGENT_LLM_MAX_TOTAL_TOKENS:
            raise LoopLimitExceeded("Token 用量达到本次任务预算上限")

    def _services(self):
        from app.connectors.service import ConnectorService
        from app.tools.base import ToolServices

        return ToolServices(
            dataset_service=self.data_engine.dataset_service,
            data_engine_service=self.data_engine,
            experiment_service=self.experiment_service,
            connector_service=ConnectorService(self.db, self.data_engine.dataset_service),
            db=self.db,
        )

    def _tool_context(self, session: AgentSession, *, user_request: str) -> ToolExecutionContext:
        # extra["user_request"]=工具的语义兜底输入（ml.detect_task 等读它）；
        # extra[ANSWERS_KEY]=本轮已回答的反问，执行前由调用方按 run 账本刷新（必须赋值不能 setdefault）。
        permissions = ROLE_PERMISSIONS.get(self.role, ROLE_PERMISSIONS["viewer"])
        return ToolExecutionContext(
            user_id=session.user_id, session_id=session.id,
            dataset_ids=set(session.dataset_ids), permissions=set(permissions),
            extra={"user_request": user_request, ANSWERS_KEY: {}},
        )

    def _result_view(self, ctx: _Turn, record: Any) -> dict[str, Any] | None:
        if record.result is None:
            return None
        if settings.AGENT_ENABLE_RESULT_COMPRESSION:
            view = record.result.for_llm(max_items=8, max_chars=1200)
        else:
            view = record.result.to_dict()
        token_meta = view.get("_token") if isinstance(view, dict) else None
        if isinstance(token_meta, dict):
            ctx.run.token_ledger.record_result_saving(int(token_meta.get("estimated_saved", 0) or 0))
        if isinstance(view, dict):
            view.pop("_token", None)
        return view

    def _output_schema(self, tool_name: str) -> dict[str, Any] | None:
        try:
            return self.registry.get(tool_name).output_schema
        except Exception:  # noqa: BLE001
            return None

    def _tool_brief(self, name: str) -> str:
        try:
            desc = self.registry.get(name).describe()
            return str(desc.get("description") or "")[:100]
        except Exception:  # noqa: BLE001
            return ""

    def _sync_dataset_context(self, ctx: _Turn) -> None:
        for key, brief in (ctx.context.dataset_context or {}).items():
            if isinstance(brief, dict):
                ctx.state.dataset_context[str(key)] = dict(brief)

    def _datasets_without_version(self, context: AgentContext) -> list[str]:
        names = []
        for brief in (context.dataset_context or {}).values():
            if isinstance(brief, dict) and not brief.get("has_version", True):
                names.append(str(brief.get("name") or f"数据集 {brief.get('dataset_id', '?')}"))
        return names

    def _bound_dataset_name(self, ctx: _Turn) -> str | None:
        if not ctx.session.dataset_ids:
            return None
        bound = ctx.session.dataset_ids[0]
        meta = (ctx.context.dataset_context or {}).get(str(bound)) or {}
        return str(meta.get("name") or "") or None

    def _available_datasets(self, ctx: _Turn) -> list[dict[str, Any]]:
        try:
            items, _ = self.data_engine.dataset_service.list(page=1, page_size=30)
            return [{"name": str(i.name), "id": int(i.id)} for i in items]
        except Exception:  # noqa: BLE001
            return []

    def _available_columns(self, ctx: _Turn) -> list[str]:
        if len(ctx.session.dataset_ids) != 1:
            return []
        try:
            return [str(c.get("name", "")) for c in (self.data_engine.column_schema(ctx.session.dataset_ids[0]) or [])]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _is_knowledge_question(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if any(m in t for m in _TASK_MARKERS):
            return False
        return ("？" in t or "?" in t or any(m in t for m in _KNOWLEDGE_MARKERS))

    @staticmethod
    def _looks_task_like(text: str) -> bool:
        """寒暄模板命中时的二次确认：句子里只要带任务动作词，就按任务走不按闲聊走。"""
        return any(m in (text or "") for m in _TASK_MARKERS)

    @staticmethod
    def _is_open_ended(text: str) -> bool:
        return any(m in (text or "") for m in OPEN_ENDED_MARKERS)

    # ============================================================ 终态
    def _complete(self, ctx: _Turn, answer: str, source: str) -> str:
        run, session = ctx.run, ctx.session
        run.final_answer = answer
        run.answer_source = source
        run.status = RunStatus.COMPLETED
        run.finished_at = time.time()
        session.history.append({"role": "assistant", "content": answer})
        if ctx.state.phase not in (Phase.DONE, Phase.WRAPUP):
            ctx.state.phase = Phase.WRAPUP if run.tool_calls else ctx.state.phase
        self._emit_usage(ctx)
        self._emit(ctx, "completed", {
            "final_answer": answer, "mode": "agent", "answer_source": describe(source),
        })
        return str(RunStatus.COMPLETED)

    def _finish_notice(self, ctx: _Turn, text: str) -> str:
        return self._complete(ctx, text, PLATFORM_RULES_NOTICE)

    def _finish_no_data(self, ctx: _Turn, datasets: list[str]) -> str:
        answer = (
            f"当前数据集（{'、'.join(datasets)}）还没有数据版本，暂时无法分析。\n"
            "请先导入数据：在「数据集」页上传 CSV / Excel，或用数据库连接器抽取，"
            "导入成功后再到这里让我分析。"
        )
        return self._complete(ctx, answer, PLATFORM_RULES_NOTICE)

    def _fail(self, ctx: _Turn, message: str) -> str:
        run = ctx.run
        if not run.answer_source:
            run.answer_source = NO_ANSWER
        run.error = message
        run.status = RunStatus.FAILED
        run.finished_at = time.time()
        self._emit_usage(ctx)
        self._emit(ctx, "failed", {"error": message, "answer_source": describe(run.answer_source)})
        return str(RunStatus.FAILED)

    def _emit(self, ctx: _Turn, event_type: str, payload: dict[str, Any]) -> None:
        if event_type not in {"completed", "failed"} and len(ctx.run.events) >= MAX_EVENTS_PER_RUN:
            raise LoopLimitExceeded(f"单次运行事件数达到上限 {MAX_EVENTS_PER_RUN}，判定为异常循环")
        from app.agent.runtime.models import AgentEvent

        event = AgentEvent(seq=len(ctx.run.events) + 1, run_id=ctx.run.id,
                           type=event_type, payload=payload)
        ctx.run.events.append(event)
        if ctx.on_event:
            ctx.on_event(event)

    def _emit_usage(self, ctx: _Turn) -> None:
        try:
            self._emit(ctx, "usage", {"token_usage": ctx.run.token_ledger.to_dict()})
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _persist(persist: Callable[[], None] | None) -> None:
        if persist is not None:
            persist()


def _failure_kind(errors: list[str]) -> str:
    """失败归因（旧 Replanner.analyze_failure 迁入；Loop 不再依赖 planner 包）。"""
    text = " ".join(errors)
    lower = text.lower()
    if "依赖" in text or ("步输出" in text):
        return "dependency"
    # 参数类错误不重试：包括「缺参数」与「策略/参数对当前列不适用」两类——
    # 「strategy 'mean' only applies to numeric columns」这类重试一万次也不会成功。
    markers = (
        "dataset_id", "参数", "缺少必需字段", "缺少必要参数", "parameter", "required", "预检失败",
        "only applies to", "only applies", "仅适用于", "不适用", "strategy", "dtype",
    )
    if any(m in lower or m in text for m in markers):
        return "parameter"
    return "transient"
