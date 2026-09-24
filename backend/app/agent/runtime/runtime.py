"""Agent Runtime：对话路由 -> 元数据上下文 -> 工具检索 -> 规划 -> 工具执行 -> 结果汇总。"""
from __future__ import annotations
import json
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
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
from app.agent.executor.executor import AgentExecutor, ToolCallRecord
from app.agent.intent import GREETINGS, INTENT_KEYWORDS, classify, explain as explain_intent
from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, fallback_allowed
from app.agent.permission.models import ROLE_PERMISSIONS
from app.local_router.contract import Intent
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner, PlanInvalidError
from app.agent.planner.replanner import AgentLimitExceeded, ReplanLimits, Replanner
from app.agent.preflight import PreflightInput, run_preflight
from app.agent.runtime.models import AgentEvent, AgentRun, AgentSession, AgentStore, RunStatus
from app.agent.validator.validator import AgentResultValidator
from app.core.config import settings
from app.core.exceptions import AgentException, ValidationException
from app.data_engine.service import DataEngineService
from app.tools.base import ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.registry import TOOL_REGISTRY, ToolRegistry

EventCallback = Callable[[AgentEvent], None]
logger = logging.getLogger("xiaoluo.agent.runtime")

# 一次运行允许的事件数上限（终态事件不计），见 `_emit` 里的说明。
MAX_EVENTS_PER_RUN = 2000
_TERMINAL_EVENTS = frozenset({"completed", "failed"})

class AgentRuntime:
    def __init__(self, data_engine: DataEngineService, *, experiment_service: Any | None = None, db: Any | None = None, llm: LLMProvider | None = None, store: AgentStore | None = None, registry: ToolRegistry | None = None, planner: AgentPlanner | None = None, validator: AgentResultValidator | None = None, replanner: Replanner | None = None, limits: ReplanLimits | None = None) -> None:
        self.data_engine = data_engine; self.experiment_service = experiment_service; self.db = db; self.llm = llm; self.store = store or AgentStore()
        from app.tools.builtin import register_builtin_tools
        register_builtin_tools()
        # 未显式传入 planner 时必须沿用配置的步数上限：AgentPlanner 默认 max_steps=6，
        # 直接构造会把「建模 + 评估 + 生成报告」这类长链路计划从尾部截断（计划里只剩到
        # ml.train，评估与报告步骤被静默丢弃），而 deps.py 走的是配置值 12，
        # 于是「服务端跑正常、脚本/测试里跑被截断」这类不一致很难定位。
        default_planner = AgentPlanner(llm, max_steps=settings.AGENT_MAX_STEPS)
        self.registry = registry or TOOL_REGISTRY; self.planner = planner or default_planner; self.validator = validator or AgentResultValidator(); self.replanner = replanner or Replanner(limits); self.limits = limits or ReplanLimits(); self.executor = AgentExecutor(self.registry)
        # ★ 统一决策入口（Phase 1/2/4）：Rule → LocalModel → Remote 的合成决策器。
        # 与 planner 分工：planner 产「整份计划」，decision_router 产「这一步怎么走」。
        # 远程升级（Phase 4）经它触达 —— 只有本地确实不足时才发起**一次**战略指导。
        from app.agent.decision.router import DecisionRouter
        self.decision_router = DecisionRouter(llm=llm)
        # DecisionTrace 运行期承载：run_id -> DecisionTrace。不写进 AgentRun 序列化
        # （那是 UI 运行记录），只用于运行结束后把完整决策链落成训练数据（Phase 5）。
        self._decision_traces: dict[str, Any] = {}

    def _usage_scope(self, run: AgentRun):
        if self.llm is None: return nullcontext()
        return self.llm.capture_usage(run.token_ledger.record_usage, budget_checker=run.token_ledger.check_budget)

    def create_session(self, *, user_id: str = "anonymous", title: str = "", dataset_ids: list[int] | None = None) -> AgentSession:
        ids = list(dict.fromkeys(int(x) for x in (dataset_ids or []) if int(x) > 0))
        return self.store.add_session(AgentSession(id=self.store.next_session_id(), user_id=user_id, title=title or "新会话", dataset_ids=ids))
    def list_sessions(self, user_id: str | None = None, *, include_archived: bool = True) -> list[AgentSession]: return self.store.list_sessions(user_id, include_archived=include_archived)
    def get_session(self, session_id: str) -> AgentSession: return self.store.get_session(session_id)
    def set_session_archived(self, session_id: str, archived: bool) -> AgentSession: return self.store.set_session_archived(session_id, archived)
    def delete_session(self, session_id: str) -> AgentSession: return self.store.delete_session(session_id)
    def get_run(self, run_id: str) -> AgentRun: return self.store.get_run(run_id)

    def run(self, session: AgentSession, user_request: str, *, role: str = "analyst", confirmed: bool = False, plan_override: AgentPlan | None = None, on_event: EventCallback | None = None) -> AgentRun:
        if not user_request or not user_request.strip(): raise ValidationException("用户请求不能为空")
        run = AgentRun(id=self.store.next_run_id(), session_id=session.id, user_id=session.user_id, user_request=user_request.strip())
        self.store.add_run(run)  # add_run 内部已把 run.id 幂等挂到 session.run_ids
        if run.id not in session.run_ids: session.run_ids.append(run.id)
        session.history.append({"role":"user","content":run.user_request}); run.started_at=time.time()
        # Phase 5：为本轮运行建立 DecisionTrace（训练数据闭环的数据出口）。
        self._begin_trace(run)
        try:
            with self._usage_scope(run):
                self._set_status(run, RunStatus.PLANNING)
                mode, route_reason, intent = self._route(
                    run.user_request, plan_override=plan_override, has_datasets=bool(session.dataset_ids)
                )
                self._emit(run,"route",{"mode":mode,"reason":route_reason,"intent":explain_intent(intent)},on_event)
                # 开局就发一次账本快照：预算上限是「这次会不会被截断」的关键信息，
                # 只在运行结束后才给出的话，用户无法在过程中判断还能不能继续问。
                self._emit_usage(run,on_event)
                if plan_override is None: self._trace_route(run, session, mode, route_reason)
                if plan_override is None and mode == "chat":
                    self._emit(run,"chat",{"stage":"direct_chat"},on_event); self._direct_chat(run,session,on_event); return run
                self._agent_turn(run, session, intent, confirmed=confirmed, plan_override=plan_override, on_event=on_event)
        except AgentLimitExceeded as exc: self._fail(run,str(exc),on_event)
        except AgentException as exc: self._fail(run,exc.message,on_event)
        # LLM 失败是「远程不可用」，不是「Agent 内部异常」：错误信息要原样给出去，
        # 不能被包成一句 `Agent 运行异常：...` 之后再让用户猜到底是哪里坏了。
        except LLMException as exc: self._fail(run,exc.message,on_event)
        except Exception as exc: self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally: run.finished_at=time.time(); self._trace_outcome(run, session); self._end_trace(run)
        return run

    def _agent_turn(self, run: AgentRun, session: AgentSession, intent, *, confirmed: bool, plan_override: AgentPlan | None, on_event: EventCallback | None) -> None:
        """路由之后、给出结果之前的完整一段（上下文 → Pre-flight → 规划 → 执行）。

        单独抽出来的原因：用户回答反问后要**从 Pre-flight 处继续**（重新规划），
        而不是从头再走一次路由；抽出来才能让「首次运行」与「回答后继续」
        共用同一段代码，避免两处逻辑漂移。
        """
        context=self._build_context(run,session); self._emit(run,"planning",{"stage":"context_ready","data_access":"metadata_only"},on_event)
        # ★ 本地理解（Phase 1 架构接通）：用 TaskSpecBuilder 产出 TaskSpec，挂到
        #   context.task_context，让本地 Router 模型真正参与「理解」而非仅 shadow。
        #   不改变下面的 chat/agent 分支逻辑（那是既有的稳定路径），只把理解结果
        #   结构化地交给规划器与 DecisionTrace。
        if plan_override is None:
            self._understand(run, session, context)
            # ★ Phase 4：本地理解判「复杂」时，发起一次远程战略指导（local-first →
            #   escalate → remote guidance → local execution）。简单任务不会触发，
            #   保证「简单任务零远程调用」。结果挂到 context.task_context 供规划器参考。
            self._escalate(run, session, context)
        # ★ 空数据集是**正常业务状态**：建了数据集还没导入数据时，给一句明确提示就结束，
        # 不进 Pre-flight、不规划、更不把它当成异常把整次运行打成 failed。
        if plan_override is None:
            empty = self._datasets_without_version(context)
            if empty:
                self._no_data_notice(run, session, empty, on_event)
                return
        if plan_override is None and settings.AGENT_PREFLIGHT_ENABLED:
            pre=self._preflight(run,session,intent,context)
            run.preflight=pre.to_dict()
            self._emit(run,"preflight",{"outcome":str(pre.outcome),"checks_run":pre.checks_run,"findings":[f.to_dict() for f in pre.findings],"clarifications":[c.to_dict() for c in pre.clarifications],"resolved":pre.resolved},on_event)
            if pre.needs_user:
                self._await_clarification(run,pre,on_event)
                return
            # Pre-flight 顺带确认下来的事实（如推断出的目标列）直接交给规划器，
            # 避免规划阶段再猜一遍。
            if pre.resolved:
                context.task_context = {**(context.task_context or {}), "preflight": dict(pre.resolved)}
        all_tools=self.registry.list(); self._fill_tools(context,all_tools); candidate_tools=self._candidate_tools(context,all_tools)
        retrieval=context.tool_context.get("retrieval_scores") or {}
        self._emit(run,"planning",{"stage":"tools_retrieved","count":len(candidate_tools),"tools":[t.get("name") for t in candidate_tools],"retrieval":retrieval},on_event)
        # ★★ 控制权归位（Phase B/C）：本地 Router 已经高置信度识别出的任务，直接由
        #    本地决策驱动执行（Observe→Decide→Act），不再交给旧 Planner 重新猜测。
        #    ``_run_dynamic`` 第一步执行本地 Router 选中的工具，之后若产出 signals
        #    会据其动态推进下一步 —— 简单任务（无 signals）等价于单步执行，
        #    多步任务（如「先看分布再分析异常」）则自然进入动态循环。
        #    旧 Planner 只在本地没把握 / 复杂任务 / 显式 plan_override 时兜底。
        if plan_override is None:
            direct_plan = self._local_direct_plan(run, session, context, candidate_tools)
            if direct_plan is not None:
                run.plan = direct_plan.to_dict()
                self._emit(run, "planning", {"stage": "local_direct", "tool": direct_plan.steps[0].tool if direct_plan.steps else None, "source": "local_router"}, on_event)
                self._emit_usage(run, on_event)
                if not direct_plan.steps:
                    self._direct_chat(run, session, on_event)
                    return
                run.status = RunStatus.RUNNING
                # 动态循环：本地决策驱动，signals 驱动下一步（复用 AgentContext + DecisionContext）。
                # 传入首步工具：多步任务（medium）的原文会再次命中 multi_step 升级，
                # 不能靠 _run_dynamic 自己重新路由出首步，必须把 _local_direct_plan
                # 已识别的首步工具作为循环的起点。
                first_tool = direct_plan.steps[0].tool if direct_plan.steps else None
                # simple 任务只跑首步即收尾（不进入信号驱动的循环，避免「分布↔异常」
                # 交替空转）；medium（多步编排）才允许信号驱动后续步骤。
                task_spec = (context.task_context or {}).get("task_spec") or {}
                is_medium = isinstance(task_spec, dict) and task_spec.get("complexity_hint") == "medium"
                self._run_dynamic(run, session, context, on_event=on_event,
                                  first_tool=first_tool, allow_loop=is_medium)
                return

        plan=self._build_plan(run, context, candidate_tools, all_tools, plan_override)
        run.plan=plan.to_dict(); self._emit(run,"planning",{"stage":"plan_ready","goal":plan.goal,"steps":len(plan.steps),"cache_hit":bool(getattr(plan,"cache_hit",False)),"planner_fallback":bool(getattr(plan,"planner_fallback",False))},on_event)
        # 规划器是本次运行的第一笔真实开销（除非命中 Plan Cache），这里立刻反映到账本上。
        self._emit_usage(run,on_event)
        if not plan.steps: self._direct_chat(run,session,on_event); return
        run.status=RunStatus.RUNNING; self._run_plan(run,session,context,self._role_of(run),plan,offset=0,attempts={},confirmed=confirmed,on_event=on_event)

    def _build_plan(self, run: AgentRun, context: AgentContext, candidate_tools: list[dict[str, Any]], all_tools: list[dict[str, Any]], plan_override: AgentPlan | None) -> AgentPlan:
        if plan_override is not None: return plan_override
        # ★ Phase D：复杂任务在 `_escalate` 已发起过一次远程战略指导（见 `_escalate`，
        #   结果挂在 context.task_context["remote_guidance"]）。若这里再走旧 Planner 的
        #   `build_plan_resilient`，会对复杂任务再调一次远程 LLM（`_llm_plan` →
        #   `llm.structured_output`）—— 这就成了「Remote Decision + Remote Planner」双重远程，
        #   正是 Phase D 要消除的反模式。已有远程指导时，改用**本地规则规划**并把指导
        #   并入计划 goal，保证复杂任务全程**最多一次远程调用**（那次已在 _escalate 完成）。
        guidance = (context.task_context or {}).get("remote_guidance")
        if guidance:
            plan = self.planner.rule_plan(run.user_request, context)
            plan.goal = f"{plan.goal}\n\n（远程战略指导：{str(guidance)[:300]}）"
            return plan
        return self.planner.build_plan_resilient(run.user_request, context, candidate_tools, all_tools=all_tools)

    def _local_direct_plan(self, run: AgentRun, session: AgentSession, context: AgentContext, candidate_tools: list[dict[str, Any]]) -> AgentPlan | None:
        """本地 Router 已高置信度识别的简单任务，直接构造单步 Plan 执行（控制权归位）。

        这是本次外科手术的核心：让「Local Router 判断正确」变成「Runtime 真正按它执行」，
        而不是让旧 Rule Planner 用关键词再猜一遍把语义结果覆盖掉（如「看看这批数据的分布」
        被规则规划器改成 dataset.inspect + eda.describe）。

        触发条件（**全部满足才直连**，任一不满足就退回旧 Planner）：
        1. ``_understand`` 已产出 TaskSpec，且 ``entities["tool"]`` 是本地 Router 的选择；
        2. 复杂度 hint ∈ {``simple``, ``medium``}（本地有把握；medium 是「先X再Y」这类
           本地多步编排，由 ``_run_dynamic`` 循环推进，同样不升级远程 / 不退回关键词规划）；
        3. 来源是 ``local_router`` / ``local_model``（不是规则 / 远程强塞）；
        4. 该工具在当前候选工具集里（不越权引用未注入工具）；
        5. 工具已绑定的数据集场景下能解析出必填参数（dataset_id 单一可确定）。

        返回 None 表示「本地不足以直连，交回旧 Planner 兜底」。
        """
        task_spec = (context.task_context or {}).get("task_spec") or {}
        tool = task_spec.get("entities", {}).get("tool") if isinstance(task_spec, dict) else None
        if not tool:
            return None
        complexity = task_spec.get("complexity_hint") if isinstance(task_spec, dict) else None
        source = task_spec.get("source") if isinstance(task_spec, dict) else ""
        # simple 与 medium 都走本地直连（medium = 本地多步，交给 _run_dynamic 循环推进）。
        if complexity not in ("simple", "medium"):
            return None
        if source not in ("local_router", "local_model"):
            return None
        # 工具必须在候选集里（与 _validate 同口径，不越权）。
        candidate_names = {t.get("name") for t in candidate_tools}
        if tool not in candidate_names:
            return None
        # 解析必填参数：仅 dataset_id（本地 Router 反问规则只认这个，且规划阶段拿不到列名）。
        tool_def = self.registry.get(tool)
        arguments: dict[str, Any] = {}
        if tool_def is not None:
            required = (tool_def.input_schema or {}).get("required") or []
            ids = context.dataset_ids()
            if "dataset_id" in required:
                if len(ids) == 1:
                    arguments["dataset_id"] = ids[0]
                elif not ids:
                    # 未绑定数据集但工具需要：本地不足以补齐，交回 Planner（会触发反问/提示）。
                    return None
                else:
                    return None
        # 记录到 DecisionTrace：这一步由本地模型决定执行该工具（真实 decision source）。
        trace = self._decision_traces.get(run.id)
        if trace is not None:
            trace.decision_made = {"action": "execute_tool", "tool": tool, "source": source}
            trace.decision_source = source
            trace.record_decision({"step": "act", "action": "execute_tool", "tool": tool,
                                   "source": source, "confidence": task_spec.get("confidence")})
        return AgentPlan(goal=task_spec.get("goal") or run.user_request,
                         steps=[PlanStep(tool=tool, arguments=arguments, expected_output=task_spec.get("sub_goal") or "")])


    def _datasets_without_version(self, context: AgentContext) -> list[str]:
        """会话里「还没有任何数据版本」的数据集名称。

        ``has_version`` 缺失时按「有数据」处理：老版本持久化的上下文没有这个键，
        默认成「没数据」会把正常会话全部拦下来。
        """
        names: list[str] = []
        for brief in (context.dataset_context or {}).values():
            if not isinstance(brief, dict) or brief.get("has_version", True):
                continue
            name = str(brief.get("name") or "").strip()
            names.append(name or f"数据集 {brief.get('dataset_id', '?')}")
        return names

    def _no_data_notice(self, run: AgentRun, session: AgentSession, datasets: list[str], on_event: EventCallback | None) -> None:
        """明确告诉用户「这份数据集还没有数据」，并给出下一步该做什么。

        错误文案要回答三件事：发生了什么 / 为什么 / 怎么办。一句
        ``dataset has no versions`` 三件都没说清。
        """
        joined = "、".join(datasets)
        answer = (
            f"当前数据集（{joined}）还没有数据版本，暂时无法分析。\n"
            "请先导入数据：在「数据集」页上传 CSV / Excel，或用数据库连接器抽取，"
            "导入成功后版本时间线会出现 v1，再到这里让我分析。"
        )
        run.answer_source = PLATFORM_RULES_NOTICE
        run.final_answer = answer
        run.status = RunStatus.COMPLETED
        run.finished_at = run.finished_at or time.time()
        session.history.append({"role": "assistant", "content": answer})
        self._emit(
            run, "completed",
            {"final_answer": answer, "mode": "notice", "reason": "dataset_has_no_version",
             "datasets": datasets, "answer_source": describe(run.answer_source)},
            on_event,
        )

    # ---- Pre-flight（第一层改造） ----------------------------------------
    def _preflight(self, run: AgentRun, session: AgentSession, intent, context: AgentContext):
        """开工前的确定性检查。拿不到列信息的检查会自动跳过（宁可不问，不乱问）。"""
        columns: list[dict[str, Any]] = []
        if settings.AGENT_PREFLIGHT_READ_SCHEMA and len(session.dataset_ids) == 1:
            columns = self._column_schema(session.dataset_ids[0])
        meta: dict[int, dict[str, Any]] = {}
        for key, value in (context.dataset_context or {}).items():
            try: meta[int(key)] = dict(value)
            except (TypeError, ValueError): continue
        return run_preflight(
            PreflightInput(
                user_request=run.user_request,
                intent=intent,
                dataset_ids=list(session.dataset_ids),
                dataset_meta=meta,
                columns=columns,
                answers=dict(run.clarification_answers),
            )
        )

    def _column_schema(self, dataset_id: int) -> list[dict[str, Any]]:
        """只读列结构（列名 + 类型），**不加载任何数据行**。

        Pre-flight 里刻意不用 `dataset.schema`（它要算每列唯一值，千万行表上
        是几十秒级开销），只读 Parquet 的 schema——列名与 dtype 已足够完成
        「目标列是否明确 / 任务类型是否矛盾」这两项判定。
        """
        try:
            return list(self.data_engine.column_schema(dataset_id) or [])
        except Exception:  # noqa: BLE001 - 拿不到列信息就跳过依赖它的检查
            return []

    def _await_clarification(self, run: AgentRun, pre, on_event: EventCallback | None) -> None:
        """把 Pre-flight 的反问挂到运行上，转入等待态。"""
        question = pre.first_question()
        payload = question.to_dict() if question else {
            "code": "preflight.unknown", "question": "需要补充信息才能继续", "options": [],
        }
        payload["outcome"] = str(pre.outcome)
        payload["checks_run"] = pre.checks_run
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = payload
        self._emit(run, "clarification", {**payload, "stage": "clarification_required"}, on_event)
        self.store.persist(force=True)

    def answer_clarification(self, run_id: str, answer: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """用户回答反问：把答案记进账本，然后从 Pre-flight / 原步骤继续。

        与 :meth:`resume` 的分工：``resume`` 处理「高风险操作授权」，
        这里处理「信息补全」。两者都会让运行脱离等待态，但语义不同。
        """
        # 空回答不是「接受默认值」，而是没答：直接往下走会用一个空串去填参数，
        # 得到看似成功实则答非所问的结果。要默认值就显式把 default 发回来。
        if not str(answer or "").strip():
            raise ValidationException("澄清回答不能为空")
        # ★ 与 confirm 同口径：原子领取，同一条反问只被消费一次。
        run, pending, code = self.store.claim_clarification(run_id, str(answer))
        if pending is None:
            raise ValidationException("该运行不在等待澄清状态（回答已被受理或运行已结束）")
        step_index = pending.get("step_index")
        session = self.store.get_session(run.session_id)
        intent = classify(run.user_request, has_datasets=bool(session.dataset_ids))
        self._emit(run, "clarification", {"stage": "answered", "code": code, "answer": str(answer)}, on_event)
        # 计划内反问（agent.clarify）→ 从原步骤继续；Pre-flight 反问 → 重新规划
        continue_from = int(step_index) if step_index is not None else None
        try:
            with self._usage_scope(run):
                run.status = RunStatus.RUNNING
                if continue_from is not None and run.plan:
                    continue_plan = AgentPlan(
                        goal=(run.plan or {}).get("goal", "继续执行"),
                        steps=self._plan_from_dict(run.plan).steps[continue_from:],
                    )
                    context = self._build_context(run, session)
                    self._run_plan(run, session, context, self._role_of(run), continue_plan,
                                   offset=continue_from, attempts={continue_from: 1},
                                   confirmed=False, on_event=on_event)
                else:
                    run.status = RunStatus.PLANNING
                    self._agent_turn(run, session, intent, confirmed=False, plan_override=None, on_event=on_event)
        except AgentLimitExceeded as exc: self._fail(run, str(exc), on_event)
        except AgentException as exc: self._fail(run, exc.message, on_event)
        except Exception as exc: self._fail(run, f"Agent 运行异常：{exc}", on_event)
        finally: run.finished_at = time.time()
        return run

    # 关键词表已迁到 `app.agent.intent`（唯一真源）：
    # 路由、候选工具注入、规则规划三处共用同一份，避免「改了两处漏了一处」
    # 造成的「工具已注册但 Agent 不调用」。这里保留别名仅为向后兼容。
    GREETINGS = GREETINGS
    DATA_TERMS = tuple(k for kws in INTENT_KEYWORDS.values() for k in kws)

    def _route(self, text: str, *, plan_override: AgentPlan | None = None, has_datasets: bool = False) -> tuple[str, str, Any]:
        """对话/工具路由。返回 ``(mode, reason, intent_decision)``。

        判定本身完全交给 :func:`app.agent.intent.classify`；这里只做
        「Intent → 走聊天还是走工具」的映射，不再维护第二套关键词。
        """
        if plan_override is not None:
            return "agent", "服务端指定计划（plan_override），直接进入工具流程", classify(text, has_datasets=has_datasets)
        decision = classify(text, has_datasets=has_datasets)
        if decision.value == Intent.CHAT:
            return "chat", decision.reasons[0] if decision.reasons else "按普通对话处理", decision
        return "agent", decision.reasons[0] if decision.reasons else "进入工具流程", decision

    def _needs_data_tools(self, text: str) -> bool:
        return self._route(text)[0] == "agent"

    # ---- 本地 Router shadow 埋点 -------------------------------------------------
    # 默认关闭（settings.LOCAL_ROUTER_MODE="off"），打开后**只记录、不改变行为**：
    # 即使模型缺失、推理抛异常、写盘失败，本次运行的路径与结果都完全不受影响。
    # 实现与数据形状见 app/local_router/trace.py。
    def _trace_route(self, run: AgentRun, session: AgentSession, mode: str, reason: str) -> None:
        try:
            from app.local_router.contract import RouterRequest
            from app.local_router.trace import shadow_route

            shadow_route(
                run_id=run.id, session_id=session.id,
                # 只喂路由发生时可**零成本**拿到的信号：available_columns 需要解析
                # Parquet 才能得到，shadow 阶段不为此付出代价（离线口径差异见 trace.py 文档）。
                request=RouterRequest(
                    utterance=run.user_request,
                    bound_dataset_id=(session.dataset_ids[0] if session.dataset_ids else None),
                ),
                rules_mode=mode, rules_reason=reason,
            )
        except Exception:
            pass

    def _trace_outcome(self, run: AgentRun, session: AgentSession) -> None:
        try:
            from app.local_router.trace import record_outcome

            record_outcome(
                run_id=run.id, session_id=session.id, status=str(run.status),
                planned_tools=[str(step.get("tool")) for step in ((run.plan or {}).get("steps") or [])],
                executed_tools=[str(record.tool) for record in run.tool_calls],
                error=run.error or "", elapsed=round(run.elapsed(), 3),
            )
        except Exception:
            pass

    def _direct_chat(self,run:AgentRun,session:AgentSession,on_event:EventCallback|None)->None:
        # 无远程大模型时，先试**平台自带能力的确定性应答**（问候 / 自述 / 能力清单 / 用法），
        # 拿不准才退回说明性文案。这样闲聊链路与数据链路（`_compose_answer` 的兜底）对称，
        # 「关掉远程」不再等于闲聊熄火 —— 见 app/agent/local_chat.py 的设计边界。
        if self.llm is None:
            from app.agent.local_chat import local_reply,no_llm_notice
            reply=local_reply(run.user_request)
            # ★ 区分「规则真的答了」与「规则答不了、只给了段状态说明」——
            # 两者对用户的含义完全不同：前者是答案，后者是「这里需要开远程」。
            run.answer_source=PLATFORM_RULES_CHAT if reply is not None else PLATFORM_RULES_NOTICE
            answer=reply or no_llm_notice()
        else:
            history=session.history[:-1][-settings.AGENT_CONTEXT_HISTORY_MESSAGES:]
            messages=[LLMMessage(role="system",content="你是小洛实验室的 AI 助手。当前是普通对话，不调用数据工具，不声称执行过任何数据处理。自然回答用户；只有明确的数据处理/分析需求才进入 Agent 工具流程。")]
            for item in history:
                if item.get("role") in {"user","assistant","system"} and item.get("content"): messages.append(LLMMessage(role=item["role"],content=str(item["content"])))
            messages.append(LLMMessage(role="user",content=run.user_request))
            try:
                answer=self.llm.chat(messages).content or "我在。有什么可以帮你？"
                run.answer_source=REMOTE_LLM_CHAT
            except Exception as exc:  # noqa: BLE001 — 只在 LLM 调用边界内捕获
                # ★ 「远程没走通」有三种完全不同的结局，不能一律塞一句「暂时无法完成对话请求」：
                #   ① 允许降级且内置规则答得上 ⇒ 真的给出本地回答，标 `LLM_ERROR_FALLBACK`；
                #   ② 允许降级但规则答不上 ⇒ 如实给一段状态说明，标 `NO_ANSWER`（这次没答上问题）；
                #   ③ 不允许降级（开关关着 / 这次错误不该兜底）⇒ 原样抛出，让这次运行如实失败。
                # 历史缺陷：三种一律走成「固定错误文案 + LLM_ERROR_FALLBACK」，
                # 既没真的兜底，又把一个纯报错标成了「已降级到规则」。
                if not fallback_allowed(exc):
                    raise
                from app.agent.local_chat import local_reply,no_llm_notice
                reply=local_reply(run.user_request,degraded=True)
                if reply is not None:
                    answer=reply
                    run.answer_source=LLM_ERROR_FALLBACK
                else:
                    answer=no_llm_notice(reason=str(exc))
                    run.answer_source=NO_ANSWER
        run.final_answer=answer; run.status=RunStatus.COMPLETED; session.history.append({"role":"assistant","content":answer}); self._emit_usage(run,on_event); self._emit(run,"completed",{"final_answer":answer,"mode":"chat","answer_source":describe(run.answer_source)},on_event)

    def _candidate_tools(self,context:AgentContext,all_tools:list[dict[str,Any]])->list[dict[str,Any]]:
        names={x.get("name") for x in context.tool_context.get("tools",[])}; return [t for t in all_tools if t.get("name") in names]

    def resume(self,session:AgentSession,run_id:str,*,on_event:EventCallback|None=None)->AgentRun:
        # ★ 原子领取：并发的两次 confirm 只有一个能拿到 pending，另一个被明确拒绝。
        # 不能依赖前端的 `confirming` 标记 —— 它只在单个标签页内有效，
        # 挡不住两个标签 / 网络重放 / 客户端重试。
        run, pending = self.store.claim_confirmation(run_id)
        if pending is None: raise ValidationException("该运行不在等待确认状态（授权已被领取或运行已结束）")
        step_index=int(pending["step_index"]); self._emit(run,"permission",{"stage":"confirmed","tool":pending["call"].tool,"step_index":step_index},on_event)
        plan=self._plan_from_dict(run.plan or {}); continue_plan=AgentPlan(goal=plan.goal,steps=plan.steps[step_index:])
        try:
            with self._usage_scope(run):
                # ★ 授权范围 = 用户刚才批准的那一步（``confirmed_step_index``）。
                # 用户确认的是弹窗里那一条「工具 + 参数」，不是整份计划；把 confirmed
                # 一路传给后续步骤等于一次确认放行整条链路，第二个高风险工具会被静默执行。
                context=self._build_context(run,session); self._run_plan(run,session,context,self._role_of(run),continue_plan,offset=step_index,attempts={step_index:1},confirmed=True,confirmed_step_index=step_index,on_event=on_event)
        except AgentLimitExceeded as exc:self._fail(run,str(exc),on_event)
        except AgentException as exc:self._fail(run,exc.message,on_event)
        except Exception as exc:self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally:run.finished_at=time.time()
        return run

    #: 两个等待态各自的终止文案。共用一条 deny 路径，但**不能共用一句文案**：
    #: 「拒绝授权」和「不回答澄清」对用户的含义不同，混在一起会让界面显示错误原因。
    _DENY_MESSAGES = {
        RunStatus.WAITING_CONFIRMATION: "用户拒绝授权，运行终止",
        RunStatus.WAITING_CLARIFICATION: "用户未回答澄清问题，运行终止",
    }

    def deny(self, run_id: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """放弃等待：终止运行并释放会话（否则 run 永远停在等待态，会话被 409 锁死）。

        覆盖两种等待态，且**只清自己那一份载荷**：

            WAITING_CONFIRMATION  -> pending_confirmation（问「这个高风险操作做不做」）
            WAITING_CLARIFICATION -> pending_clarification（问「信息不全，请补一个参数」）

        早先的实现用 ``pending_confirmation is None`` 做统一门禁，于是澄清态
        （载荷在 pending_clarification 里）被判定为「无需处理」直接返回——
        用户点取消/拒绝后运行毫无反应，会话永久锁死。
        """
        # ★ 同样走原子领取：并发的两次 deny 只有一个真正终止运行，
        # 另一个拿到 kind="" 直接返回（幂等），不会把已失败的运行再写一遍。
        run, kind = self.store.claim_wait(run_id)
        if not kind:
            return run

        self._fail(run, self._DENY_MESSAGES[run.status], on_event)
        self.store.persist(force=True)
        return run

    def cancel(self, run_id: str) -> AgentRun:
        """请求取消运行：设置标记，_run_plan 在当前步骤结束后停止。

        等待态下取消等价于放弃等待（确认 → 拒绝授权，澄清 → 不回答），
        必须真的终止，否则会话一直被 ACTIVE_STATUSES 锁住。
        """
        run = self.store.get_run(run_id)
        if run.status in (RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RUNNING):
            run.cancel_requested = True
            self.store.persist(force=True)
        elif run.status in (RunStatus.WAITING_CONFIRMATION, RunStatus.WAITING_CLARIFICATION):
            run = self.deny(run_id)
        return run

    def _run_plan(self,run:AgentRun,session:AgentSession,context:AgentContext,role:str,plan:AgentPlan,*,offset:int,attempts:dict[int,int],confirmed:bool=False,confirmed_step_index:int|None=None,on_event:EventCallback|None)->None:
        """执行计划；失败即交给 `Replanner` 决定「重试 / 跳过 / 终止」。

        ★★ 授权范围（P0）：``confirmed`` 只在 ``confirmed_step_index`` 指定的那一步生效。

        历史缺陷：``resume()`` 把 ``confirmed=True`` 传进 ``_run_plan`` 后，这个布尔值会
        随着循环一路传给**每一个后续步骤**。于是「Step1 data.clean(HIGH) → 用户确认」之后，
        Step2 ml.train(HIGH)、Step3 workflow.run(HIGH) 全部被静默放行 —— 一次确认 = 放行整条
        高风险链路，而用户以为自己只批准了弹窗里那一个工具。

        现在：一次确认最多授权**一次**对应的高风险调用；后续再次遇到 HIGH / CRITICAL 会重新进入
        WAITING_CONFIRMATION。``confirmed_step_index is None`` 表示调用方显式授权整轮执行，
        仅用于服务端/测试直接调用（HTTP API 不接受客户端传 confirmed）。

        ★★ 授权是**一次性凭据**，不是状态开关（2026-09-24 P0）
        ``authorized_step`` 在本轮执行里只存在到「那一步真正被执行一次」为止，用完即置空。
        否则会出现：确认 → 高风险工具执行失败 → Replanner 判「瞬时故障、重试同一步」→
        同一步以 ``abs_idx == confirmed_step_index`` 再次命中授权 → 未经用户同意又执行一次。
        用户点的是弹窗里那一次操作，失败后重试属于**新的一次授权请求**，必须重新弹窗。
        （此前侥幸没出事，只是因为 ``resume()`` 预置 ``attempts={step_index:1}`` 让重试在第
        一次失败后就被跳过 —— 那是记账口径的巧合，不是授权语义的保证。）


        ★★ 死循环防线（2026-09-22 r-22 P0，实测过同一条路径产生 199987 条 replanning 事件）
        原实现里有三处叠加的漏洞，任意一处都足以让运行永不结束：

        1. **「参数解析失败」这条分支不记账** —— 只把 `attempts.get(abs_idx,0)+1` 传进
           replanner，从不写回 `attempts`。于是 `MAX_ATTEMPTS_PER_STEP` 永远看到 attempt=1，
           永远判定「还能再试一次」。
        2. 同一条分支**无条件 `offset=abs_idx+1`**，即使 replanner 返回的是「重试同一步」
           （`retry=True`）。步号一路 +1、计划却原封不动 ⇒ 同一个失败步骤以**全新下标**
           被反复重试，第 1 条的计数自然永远归零。
        3. `assert_limits`（工具调用数 / 超时 / token）**只在执行工具前调用**，解析失败的
           迭代完全不过熔断 ⇒ 连 180s 超时都救不了。

        现在的三层保护：**尝试数写回** + **重试不前进下标** + **每次迭代过熔断与重规划总闸**。
        """
        tool_ctx=self._tool_context(session,role,user_request=run.user_request)
        # 本轮已回答的反问注入工具上下文（保持 `_tool_context` 原签名不变：
        # 它是既有测试替身的重写点，改签名会连带打断它们）。
        # ★ 必须**赋值**而不是 ``setdefault``：``_tool_context`` 总是预置一个空 dict，
        # setdefault 因此永远不生效，工具拿到的 answers 恒为空 —— 用户回答后
        # agent.clarify 会原样再问一遍，形成「回答 → 继续 → 再问」的死循环。
        extra = getattr(tool_ctx, "extra", None)
        if isinstance(extra, dict):
            extra[ANSWERS_KEY] = dict(run.clarification_answers or {})
        services=self._services(); current=plan; permanent_failure=""; total_steps=max(len(plan.steps),1); replans=0
        # 一次性授权凭据：不是「本轮已确认」这个状态，而是「还剩这一次没用掉」。
        # None 表示调用方授权整轮（服务端/测试直调），不参与消费。
        authorized_step: int | None = confirmed_step_index
        while current.steps:
            # ① 重规划总闸：与单步尝试上限互补，任何原因的循环到这里都会被截断。
            #    阈值只在 ``Replanner.assert_replan_budget`` 定义一处——早先这里内联了
            #    一份同样的判断，两边一旦漂移就会出现「测试过的闸不是线上那个」。
            try:
                self.replanner.assert_replan_budget(replans)
            except AgentLimitExceeded as exc:
                permanent_failure = str(exc); break
            replanned=False
            for i,raw_step in enumerate(current.steps):
                if run.cancel_requested:
                    self._fail(run,"运行已被用户取消",on_event); return
                abs_idx=offset+i
                # ② 熔断：**每次迭代**都过，不只是执行工具前那一次
                self.replanner.assert_limits(tool_calls=run.tool_call_count+1,elapsed_seconds=run.elapsed(),total_tokens=run.token_ledger.actual_total_tokens,max_total_tokens=settings.AGENT_LLM_MAX_TOTAL_TOKENS)
                try: step=self._resolve_step_arguments(raw_step,run,session,context)
                except (ValidationException,PlanInvalidError) as exc:
                    errors=[getattr(exc,"message",str(exc))]
                    # ③ 解析失败同样算一次尝试并**写回**，否则单步上限形同虚设
                    attempt_no=attempts.get(abs_idx,0)+1; attempts[abs_idx]=attempt_no
                    replan=self.replanner.replan(current,failed_step_index=i,errors=errors,attempts=attempt_no,base_index=offset); replans+=1
                    self._emit(run,"replanning",{"step_index":abs_idx,"notes":replan.notes,"remaining":len(replan.steps),"retry":bool(getattr(replan,"retry",False)),"attempt":attempt_no},on_event)
                    if not replan.steps: permanent_failure=f"第 {abs_idx} 步失败：{'；'.join(errors)}"; break
                    # ④ 与执行失败分支同口径：要重试就留在原地，要跳过才前进
                    retrying=bool(getattr(replan,"retry",False)); offset=abs_idx if retrying else abs_idx+1; current=replan; replanned=True; break
                attempt_no=attempts.get(abs_idx,0)+1; attempts[abs_idx]=attempt_no
                self._emit(run,"tool_call",{"step_index":abs_idx,"tool":step.tool,"arguments":step.arguments,"progress":min(90,max(5,round((run.tool_call_count/max(total_steps,1))*90)))},on_event)
                # 授权只落到被用户确认的那一步；其余步骤按 PermissionManager 的裁决走
                # （HIGH / CRITICAL 会再次进入 WAITING_CONFIRMATION）。
                step_confirmed=confirmed and (confirmed_step_index is None or abs_idx==authorized_step)
                if step_confirmed and confirmed_step_index is not None:
                    # 凭据用掉即失效：同一高风险步骤再次执行（重试 / 重规划）必须重新确认。
                    authorized_step=None
                record=self.executor.execute_step(step,tool_ctx,services,confirmed=step_confirmed,attempt=attempt_no,step_index=abs_idx); run.tool_calls.append(record)
                if record.status=="needs_confirmation":
                    run.status=RunStatus.WAITING_CONFIRMATION; run.pending_confirmation={"call":record,"step_index":abs_idx}; self._emit(run,"permission",{"stage":"confirmation_required","tool":step.tool,"reason":record.error,"step_index":abs_idx},on_event); return
                if record.status=="needs_clarification":
                    # 计划执行中的结构化反问（agent.clarify）：记下步号，
                    # 用户回答后从这一步继续，前面的重型步骤不重跑。
                    #
                    # ★ 载荷必须落在 ``pending_clarification``。曾经错写成
                    # ``pending_confirmation``，而 ``answer_clarification`` 只读
                    # ``pending_clarification`` —— 结果是计划内反问**永远无法被回答**，
                    # 运行卡在 WAITING_CLARIFICATION 直到进程重启（会话期间一直 409）。
                    clarification=record.clarification.to_dict() if hasattr(record.clarification,"to_dict") else {}
                    run.status=RunStatus.WAITING_CLARIFICATION
                    run.pending_clarification={**clarification,"step_index":abs_idx,"tool":step.tool}
                    self._emit(run,"clarification",{**clarification,"stage":"clarification_required","tool":step.tool,"step_index":abs_idx},on_event)
                    self.store.persist(force=True); return
                if record.status=="denied": permanent_failure=f"第 {abs_idx} 步被拒绝：{record.error}"; break
                result_payload=self._result_for_llm(run,record); self._emit(run,"tool_result",{"step_index":abs_idx,"tool":step.tool,"status":record.status,"summary":record.result.summary if record.result else record.error,"llm_result":result_payload,"progress":min(95,max(5,round(((abs_idx+1)/max(total_steps,1))*90)))},on_event)
                # `_result_for_llm` 刚把「结果压缩省下多少 Token」记进账本，这里立刻同步给界面。
                self._emit_usage(run,on_event)
                if record.status=="ok":
                    validation=self.validator.validate(step,record.result,self._tool_output_schema(step.tool)); self._emit(run,"validation",{"step_index":abs_idx,"tool":step.tool,"valid":validation.valid,"errors":validation.errors,"warnings":validation.warnings},on_event)
                    if validation.valid: continue
                    errors=validation.errors
                else:
                    errors=[record.error] if record.error else ["工具执行失败"]; self._emit(run,"validation",{"step_index":abs_idx,"tool":step.tool,"valid":False,"errors":errors},on_event)
                replan=self.replanner.replan(current,failed_step_index=i,errors=errors,attempts=attempt_no,base_index=offset); replans+=1; self._emit(run,"replanning",{"step_index":abs_idx,"notes":replan.notes,"remaining":len(replan.steps),"retry":bool(getattr(replan,"retry",False)),"attempt":attempt_no},on_event)
                if not replan.steps: permanent_failure=f"第 {abs_idx} 步失败且无剩余步骤：{'；'.join(errors)}"; break
                retrying=bool(getattr(replan,"retry",False)); offset=abs_idx if retrying else abs_idx+1; current=replan; replanned=True; break
            if permanent_failure or not replanned: break
        if permanent_failure:self._fail(run,permanent_failure,on_event)
        else:self._complete(run,session,on_event)

    def _run_dynamic(self, run: AgentRun, session: AgentSession, context: AgentContext, *, on_event: EventCallback | None, first_tool: str | None = None, allow_loop: bool = False) -> None:
        """Observe → Decide → Act 动态循环（任务 5.5 / 6）。

        与 ``_run_plan``（静态 for 循环）的区别：这里每一步执行完，都把
        ``ToolResult.signals`` + ``last_tool`` 反馈进下一轮 ``DecisionContext``，
        由 ``DecisionRouter``（Signal → LocalModel）动态决定下一步 —— 不靠预先写死的
        计划链条。复用既有 ``AgentContext`` + ``DecisionContext`` + ``executor``，
        **不新建 State/Context 系统**。

        适用场景：本地 Router 已判定为「多步本地决策」的任务（如「先看分布，再根据
        结果分析异常」）。简单单步任务仍走 ``_local_direct_plan``（更快），复杂任务
        走远程战略规划，这里只承载「本地能自己循环推进」的中间地带。

        终止条件（任一满足即停，防死循环）：
        1. DecisionRouter 给出 STOP / CHAT / ASK_USER / ESCALATE；
        2. 下一步工具与上一步重复（本地 Router 只会单轮判断，重复=没有新信息）；
        3. 达到 ``AGENT_MAX_STEPS`` 上限；
        4. 工具执行失败（交给既有 Replanner 口径，这里简化为终止并如实失败）。
        """
        from app.agent.decision.provider import DecisionAction, DecisionContext

        tool_ctx = self._tool_context(session, self._role_of(run), user_request=run.user_request)
        services = self._services()
        bound_dataset_id = session.dataset_ids[0] if session.dataset_ids else None
        executed: list[str] = []
        signals: dict[str, Any] = {}
        task_spec = (context.task_context or {}).get("task_spec") or {}
        # 可见列名：从真实 schema 取（与 _preflight 同口径，只读列结构不读数据行）。
        # 拿不到就留空 —— 本地 Router 的 request_text 只编码「可见列数」，空列表是
        # 诚实表达「规划阶段看不到列名」，不臆造。
        available_columns = [c.get("name", "") for c in (self._column_schema(bound_dataset_id) if bound_dataset_id else [])]

        for step_no in range(settings.AGENT_MAX_STEPS):
            if run.cancel_requested:
                self._fail(run, "运行已被用户取消", on_event)
                return
            # Observe → Decide：用当前 signals + 已执行工具 构造决策上下文。
            # 首步：若调用方给了 first_tool（多步任务拆解出的首步），直接用，
            # 不再对原文重新路由（原文会命中 multi_step 升级，导致首步无法确定）。
            if step_no == 0 and first_tool and not executed:
                decision = self._decision_for_first_step(run, context, task_spec, first_tool, bound_dataset_id)
            else:
                decision = self.decision_router.route(
                    DecisionContext(
                        user_request=run.user_request,
                        task_spec=task_spec,
                        bound_dataset_id=bound_dataset_id,
                        available_columns=available_columns,
                        recent_tools=list(executed),
                        signals=dict(signals),
                        attempts=step_no,
                    ),
                    allow_remote=False,  # 动态循环只在本地推进，升级留给复杂任务入口
                )
            if decision.action != DecisionAction.EXECUTE_TOOL or decision.tool is None:
                # STOP / CHAT / ASK_USER / ESCALATE：本地循环结束，交给常规路径收尾。
                break
            tool = decision.tool
            if executed and tool == executed[-1]:
                # 下一步与上一步相同：本地 Router 无新信息，终止循环避免空转。
                break

            # Act：执行工具（复用 executor + 必填参数补全口径，与 _run_plan 一致）。
            step = PlanStep(tool=tool, arguments={}, expected_output="")
            try:
                step = self._resolve_step_arguments(step, run, session, context)
            except (ValidationException, PlanInvalidError) as exc:
                # 本地循环里「下一步缺参数」是正常终止条件：本地拿不到的信息（如单列工具
                # 需要的 column），不再强行推进，用已执行步骤的真实结果收尾，而不是把
                # 整次运行打成 failed。首步失败才如实失败（下面 tool_calls 为空时兜底）。
                if not executed:
                    self._fail(run, f"第 1 步参数解析失败：{getattr(exc, 'message', str(exc))}", on_event)
                    return
                self._emit(run, "planning", {"stage": "dynamic_stop", "reason": f"下一步 {tool} 缺参数，本地无法继续", "detail": getattr(exc, "message", str(exc))}, on_event)
                break
            abs_idx = step_no
            self._emit(run, "tool_call", {"step_index": abs_idx, "tool": tool, "arguments": step.arguments, "source": decision.source, "confidence": decision.confidence}, on_event)
            record = self.executor.execute_step(step, tool_ctx, services, confirmed=False, attempt=1, step_index=abs_idx)
            run.tool_calls.append(record)
            if record.status == "needs_confirmation":
                run.status = RunStatus.WAITING_CONFIRMATION
                run.pending_confirmation = {"call": record, "step_index": abs_idx}
                self._emit(run, "permission", {"stage": "confirmation_required", "tool": tool, "reason": record.error, "step_index": abs_idx}, on_event)
                return
            if record.status == "needs_clarification":
                clarification = record.clarification.to_dict() if hasattr(record.clarification, "to_dict") else {}
                run.status = RunStatus.WAITING_CLARIFICATION
                run.pending_clarification = {**clarification, "step_index": abs_idx, "tool": tool}
                self._emit(run, "clarification", {**clarification, "stage": "clarification_required", "tool": tool, "step_index": abs_idx}, on_event)
                self.store.persist(force=True)
                return
            if record.status != "ok":
                self._fail(run, f"第 {step_no + 1} 步工具 {tool} 执行失败：{record.error or '未知错误'}", on_event)
                return

            executed.append(tool)
            # Observe：提取真实 signals 反馈进下一轮（数据通路真正打通）。
            result_signals = list(getattr(record.result, "signals", []) or [])
            signals = {s: True for s in result_signals}
            self._emit(run, "tool_result", {"step_index": abs_idx, "tool": tool, "status": record.status, "summary": record.result.summary if record.result else "", "signals": result_signals}, on_event)
            # 记录到 DecisionTrace：这一跳由谁决定执行（真实 decision source）。
            trace = self._decision_traces.get(run.id)
            if trace is not None:
                trace.record_decision({"step": "act", "action": "execute_tool", "tool": tool,
                                       "source": decision.source, "confidence": decision.confidence,
                                       "signals": result_signals})
            # simple 任务只跑首步：执行完即收尾，不进入信号驱动的后续循环
            # （否则「分布→异常→分布…」会交替空转）。只有 medium（多步编排）
            # 才允许信号驱动下一步。
            if not allow_loop:
                break

        self._complete(run, session, on_event)

    def _decision_for_first_step(self, run: AgentRun, context: AgentContext, task_spec: Any, first_tool: str, bound_dataset_id: int | None):
        """动态循环首步的决策：直接用 `_local_direct_plan` 已识别的首步工具。

        多步任务（medium，如「先看分布再分析异常」）的**原文**会再次命中
        local_router 的 multi_step 升级（L0 规则先于 L1），若对原文重新路由
        会拿不到首步工具、循环在第一轮就空转终止。首步工具是 `_understand`
        阶段从第一个分句可靠识别出来的，这里把它固化为首步决策，后续步骤
        仍走 signals 驱动的正常路由。
        """
        from app.agent.decision.provider import AgentDecision, DecisionSource

        # 从 TaskSpec 拿置信度与来源（与 _local_direct_plan 同口径）。
        conf = float(task_spec.get("confidence") or 0.0) if isinstance(task_spec, dict) else 0.0
        source = task_spec.get("source") if isinstance(task_spec, dict) else DecisionSource.LOCAL_ROUTER
        return AgentDecision(
            action="execute_tool",
            tool=first_tool,
            source=source or DecisionSource.LOCAL_ROUTER,
            confidence=conf,
            reasons=[f"本地 Router 拆解多步任务，首步工具 {first_tool}"],
            evidence={"first_step": True},
        )

    def _resolve_step_arguments(self,step:PlanStep,run:AgentRun,session:AgentSession,context:AgentContext)->PlanStep:
        tool=self.registry.get(step.tool); schema=tool.input_schema or {}
        # 先解析 {{stepN.field}} 结构化依赖（后续工具消费前一步输出），再做必填补全
        from app.agent.runtime.step_resolution import resolve_arguments
        step=resolve_arguments(step,run,session,context,schema)
        args=dict(step.arguments or {}); required=list(schema.get("required") or [])
        if "dataset_id" in required:
            if args.get("dataset_id") is None:
                ids=context.dataset_ids()
                if len(ids)==1: args["dataset_id"]=ids[0]
                elif not ids: raise ValidationException(f"工具 {step.tool} 需要 dataset_id，但当前会话没有关联数据集。")
                else: raise ValidationException(f"工具 {step.tool} 需要明确的 dataset_id，但当前会话关联了多个数据集。")
            try: args["dataset_id"]=int(args["dataset_id"])
            except (TypeError,ValueError) as exc: raise ValidationException(f"工具 {step.tool} 的 dataset_id 必须是整数。") from exc
            if session.dataset_ids and args["dataset_id"] not in session.dataset_ids: raise ValidationException(f"工具 {step.tool} 指定的数据集不在当前会话关联范围内。")
        missing=[name for name in required if args.get(name) in (None,"")]
        if missing: raise ValidationException(f"工具 {step.tool} 缺少必要参数：{', '.join(missing)}")
        return PlanStep(tool=step.tool,arguments=args,expected_output=step.expected_output,permission=step.permission)

    def _result_for_llm(self,run:AgentRun,record:ToolCallRecord,*,compact:bool=False)->dict[str,Any]|None:
        if record.result is None:return None
        if settings.AGENT_ENABLE_RESULT_COMPRESSION:
            view=record.result.for_llm(max_items=10 if compact else 16,max_chars=1600 if compact else 2600)
        else:view=record.result.to_dict()
        token_meta=view.get("_token") if isinstance(view,dict) else None
        if isinstance(token_meta,dict): run.token_ledger.record_result_saving(int(token_meta.get("estimated_saved",0) or 0))
        if isinstance(view,dict): view.pop("_token",None)
        return view

    def _build_context(self,run:AgentRun,session:AgentSession)->AgentContext: return ContextBuilder(self.data_engine).build(run.user_request,dataset_ids=session.dataset_ids,role=self._role_of(run),history=session.history[:-1])

    # ---- DecisionTrace（Phase 5 数据闭环）---------------------------------
    def _begin_trace(self, run: AgentRun) -> None:
        """为本轮运行建立 DecisionTrace（不写盘，运行结束统一落）。"""
        try:
            from app.agent.trace.decision_trace import DecisionTrace
            self._decision_traces[run.id] = DecisionTrace(trace_id=run.id, request=run.user_request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DecisionTrace 初始化失败：%s", exc)

    def _end_trace(self, run: AgentRun) -> None:
        """运行结束：把 DecisionTrace 补全（最终回答/成功/用量）并落盘。"""
        try:
            trace = self._decision_traces.pop(run.id, None)
            if trace is None:
                return
            trace.final_answer = run.final_answer or ""
            trace.answer_source = run.answer_source or ""
            trace.success = run.status.value == "completed" if hasattr(run.status, "value") else str(run.status) == "completed"
            trace.remote_calls = int(run.token_ledger.llm_calls)
            trace.remote_escalations = int(run.token_ledger.remote_escalations)
            trace.local_calls = int(getattr(run, "tool_call_count", 0))
            trace.latency_ms = int(run.elapsed() * 1000)
            trace.tool_calls = [c.to_dict() for c in run.tool_calls]
            trace.tool_results_summary = [
                {"tool": c.tool, "status": c.status, "summary": c.result.summary if c.result else "",
                 "signals": list(c.result.signals) if c.result else []}
                for c in run.tool_calls
            ]
            from app.agent.trace.decision_trace import write_trace
            write_trace(trace)
        except Exception as exc:  # noqa: BLE001 — 埋点失败不得影响运行
            logger.warning("DecisionTrace 写入失败：%s", exc)

    def _understand(self, run: AgentRun, session: AgentSession, context: AgentContext) -> None:
        """本地理解：把用户语言转成 TaskSpec，挂到 context.task_context。

        Phase 1 的架构接通点。用 TaskSpecBuilder（本地 Router 模型 + intent 分类）
        产出 TaskSpec，写入 ``context.task_context["task_spec"]``，并记录一条
        ``understanding`` 事件供 UI / Trace 观测。

        不改变既有 chat/agent 分支（那是稳定路径），只补上「本地模型真正参与理解」
        这一环 —— 之前本地 Router 只在 shadow 模式记录，从未进入运行链路。
        """
        try:
            from app.agent.task_spec_builder import TaskSpecBuilder

            # 复用运行期 decision_router（携带 llm），使本地理解阶段在本地 Router 判
            # 「本地不足」时也能**真正触发一次远程升级**（Phase 4），而非只产出空壳。
            builder = TaskSpecBuilder(router=self.decision_router)
            understanding = builder.understand(
                run.user_request,
                bound_dataset_id=(session.dataset_ids[0] if session.dataset_ids else None),
                intent_decision=classify(run.user_request, has_datasets=bool(session.dataset_ids)),
            )
            # TaskSpec 挂到 task_context（AgentContext 的可变状态里最贴切的位置）。
            context.task_context = {
                **(context.task_context or {}),
                "task_spec": understanding.spec.to_dict(),
                "understanding": understanding.to_dict(),
            }
            # 记录进 DecisionTrace：task_spec + router 候选（训练数据出口）。
            trace = self._decision_traces.get(run.id)
            if trace is not None:
                trace.task_spec = understanding.spec.to_dict()
                evidence = understanding.evidence or {}
                router_decision = evidence.get("router_decision") or {}
                trace.router_candidates = [{
                    "candidate": router_decision.get("tool"),
                    "score": router_decision.get("confidence"),
                    "source": router_decision.get("source"),
                }]
                trace.decision_source = understanding.source or ""
                trace.decision_made = {"action": "understand", "tool": understanding.spec.entities.get("tool"),
                                       "source": understanding.source}
                trace.record_decision({"step": "understand", "action": "understand",
                                       "tool": understanding.spec.entities.get("tool"),
                                       "source": understanding.source,
                                       "confidence": understanding.spec.confidence})
            # 记录一次「本地理解」事件（不进落盘白名单，避免写放大）。
            self._emit(run, "planning", {
                "stage": "task_understood",
                "task_spec": understanding.spec.to_dict(),
                "source": understanding.source,
                "needs_clarification": understanding.needs_clarification,
            }, on_event=None)
        except Exception as exc:  # noqa: BLE001 — 本地理解失败不得拖垮主流程
            logger.warning("本地理解失败（%s）：%s", type(exc).__name__, exc)

    def _escalate(self, run: AgentRun, session: AgentSession, context: AgentContext) -> dict[str, Any]:
        """Phase 4 远程升级：本地不足时发起**一次**远程战略指导，本地仍负责执行。

        触发条件（由本地理解 / 本地 Router 判定，而非本方法自行猜）：
        ``context.task_context["task_spec"]["complexity_hint"] == "complex"``，
        即本地 Router 明确给出「升级」判定。简单 / 一般任务**不会**走到这里
        （PoC「简单任务零远程调用」的关键保证）。

        职责边界（任务 10）：只做一次远程调用，拿战略级方向（下一步最该做什么），
        **不控制工具执行循环**。产出写进 ``context.task_context["remote_guidance"]``
        供规划器参考，并如实记录进 DecisionTrace 与账本（remote_escalations）。
        远程失败不吞错：走 ``fallback_allowed`` 同口径 —— 允许降级就保留本地结论，
        不允许就抛出，让这次运行如实失败。
        """
        if self.llm is None:
            return {"escalated": False, "reason": "no_llm"}
        task_spec = (context.task_context or {}).get("task_spec") or {}
        if str(task_spec.get("complexity_hint")) != "complex":
            return {"escalated": False, "reason": "not_complex"}
        from app.agent.decision.provider import DecisionContext
        from app.agent.llm.base import LLMException, LLMMessage, fallback_allowed

        try:
            decision = self.decision_router.route(
                DecisionContext(
                    user_request=run.user_request,
                    task_spec=task_spec,
                    bound_dataset_id=(session.dataset_ids[0] if session.dataset_ids else None),
                ),
                allow_remote=True,
            )
            guidance = (decision.evidence or {}).get("remote_guidance")
            if not guidance:
                return {"escalated": False, "reason": "no_guidance"}
            run.token_ledger.record_escalation()
            context.task_context = {**(context.task_context or {}), "remote_guidance": guidance}
            self._emit(run, "planning", {"stage": "remote_escalated", "guidance": str(guidance)[:200]}, on_event=None)
            trace = self._decision_traces.get(run.id)
            if trace is not None:
                trace.record_decision({"step": "escalate", "action": "escalate",
                                       "source": decision.source, "guidance": str(guidance)[:200]})
            return {"escalated": True, "source": decision.source, "guidance": str(guidance)[:200]}
        except LLMException as exc:
            if not fallback_allowed(exc):
                raise
            # 允许降级：远程这次不可用，保留本地结论继续（不静默吞成「已升级」）。
            logger.warning("远程升级失败（%s），按本地结论继续：%s", type(exc).__name__, exc)
            return {"escalated": False, "reason": f"llm_failed:{type(exc).__name__}"}

    def _fill_tools(self,context:AgentContext,tools:list[dict[str,Any]])->None: ContextBuilder(self.data_engine).with_tools(context,tools,registry=self.registry)
    def _role_of(self,run:AgentRun)->str:return "analyst"
    def _tool_context(self, session: AgentSession, role: str, *, user_request: str = "") -> ToolExecutionContext:
        # `extra["user_request"]`：工具的「语义兜底输入」。
        # 没有它，ml.detect_task 这类需要理解用户诉求的工具就只能指望 LLM 记得把
        # 诉求原文抄进参数里 —— 那是把正确性押在提示词遵从度上（且规划器本来
        # 也拿不到列名）。工具读这个字段即可获得稳定的意图来源。
        # `extra[ANSWERS_KEY]`：本轮已回答的结构化反问（第一层），
        # agent.clarify 靠它判断「这个问题是否已经答过」。
        permissions=ROLE_PERMISSIONS.get(role,ROLE_PERMISSIONS["viewer"])
        return ToolExecutionContext(
            user_id=session.user_id,
            session_id=session.id,
            # 会话关联了哪些数据集就是哪些：空列表 ⇒ set()（一个都不许访问），
            # 而不是 None（不限制）。Agent 场景下不存在「无限制访问全部数据集」的语义，
            # 用户没选数据集时应由 Pre-flight / 反问来处理，而不是静默读任意数据。
            dataset_ids=set(session.dataset_ids),
            permissions=set(permissions),
            extra={
                "user_request": str(user_request or ""),
                ANSWERS_KEY: {},
            },
        )
    def _services(self)->ToolServices:
        # connector_service 按需构造（依赖同一个 Session 与 DatasetService）：
        # 连接器导入必须复用 DatasetService 的版本链路，不能自建一套存储。
        from app.connectors.service import ConnectorService
        return ToolServices(dataset_service=self.data_engine.dataset_service,data_engine_service=self.data_engine,experiment_service=self.experiment_service,connector_service=ConnectorService(self.db,self.data_engine.dataset_service),db=self.db)
    def _tool_output_schema(self,tool_name:str)->dict[str,Any]|None:
        try:return self.registry.get(tool_name).output_schema
        except Exception:return None
    def _plan_from_dict(self,data:dict[str,Any])->AgentPlan: return AgentPlan(goal=data.get("goal","继续执行"),steps=[PlanStep(tool=s["tool"],arguments=s.get("arguments") or {},expected_output=s.get("expected_output") or "",permission=s.get("permission") or "") for s in data.get("steps",[])])
    def _set_status(self,run:AgentRun,status:RunStatus)->None:run.status=status
    def _emit(self,run:AgentRun,event_type:str,payload:dict[str,Any],on_event:EventCallback|None)->None:
        # ★ 事件数硬上限：r-3 一次死循环往同一次运行里写了 **199987 条事件**（store 撑到 51MB，
        # 且每次持久化都要整体重写一遍）。重规划总闸理论上已经拦住这一类循环，但这条是
        # 「不管什么原因」的最后护栏 —— 宁可让这一次运行失败，也不能把进程和磁盘拖死。
        # 终态事件（completed / failed）永远放行，否则连失败原因都发不出去。
        if event_type not in _TERMINAL_EVENTS and len(run.events)>=MAX_EVENTS_PER_RUN:
            raise AgentLimitExceeded(f"单次运行的事件数达到上限 {MAX_EVENTS_PER_RUN}，判定为异常循环，已终止")
        event=AgentEvent(seq=len(run.events)+1,run_id=run.id,type=event_type,payload=payload); run.events.append(event)
        if on_event:on_event(event)
    def _fail(self,run:AgentRun,message:str,on_event:EventCallback|None)->None:
        # 失败/取消/拒绝都不产生回答，来源如实标成「未产出回答」而不是留空让前端猜。
        if not run.answer_source: run.answer_source=NO_ANSWER
        run.error=message;run.status=RunStatus.FAILED;run.finished_at=run.finished_at or time.time();self._emit_usage(run,on_event);self._emit(run,"failed",{"error":message,"answer_source":describe(run.answer_source)},on_event)
    def _complete(self,run:AgentRun,session:AgentSession,on_event:EventCallback|None)->None:run.final_answer=self._compose_answer(run);run.status=RunStatus.COMPLETED;run.finished_at=run.finished_at or time.time();session.history.append({"role":"assistant","content":run.final_answer});self._emit_usage(run,on_event);self._emit(run,"completed",{"final_answer":run.final_answer,"mode":"agent","answer_source":describe(run.answer_source)},on_event)
    def _emit_usage(self,run:AgentRun,on_event:EventCallback|None)->None:
        """发一帧 Token 账本快照（`usage` 事件）。

        为什么单独发一类事件而不是塞进每个事件里：账本在**运行过程中**就在变
        （规划器调用、结果压缩节省、Plan Cache 命中），而界面原先只在结束后从
        `GET /runs/{id}` 拿一次总数 —— 长任务（或像 r-22 那种卡住的任务）期间
        用户完全看不到「已经花了多少 / 省了多少」。
        这里只发快照、不改任何行为；写盘节流由上层 persist 控制，
        故意**不**加进 `agent_runtime.py` 的落盘白名单，避免多出的写放大。
        """
        try: self._emit(run,"usage",{"token_usage":run.token_ledger.to_dict()},on_event)
        except Exception: pass  # noqa: BLE001 — 账本上报失败绝不能影响主流程
    def _compose_answer(self,run:AgentRun)->str:
        views=[]
        for call in run.tool_calls:
            if call.status!="ok" or call.result is None:continue
            view=self._result_for_llm(run,call,compact=True)
            if view is not None:views.append({"tool":call.tool,"result":view})
        # ★ 无远程模型时，基于**真实 ToolResult.data** 生成本地结果，而不是一句
        #   「已完成 N 个步骤」。数据本身是真实工具跑出来的，这里只做统一的结构化渲染。
        def _local_fallback():
            from app.agent.local_chat import local_result_summary
            return local_result_summary(run.tool_calls)
        fallback=_local_fallback()
        # 无远程模型 ⇒ 汇总也是规则拼的，如实标注（数据本身仍是真实工具跑出来的）。
        if self.llm is None:
            run.answer_source=PLATFORM_RULES_SUMMARY
            return fallback
        system=("你是小洛实验室的数据分析助手。基于真实工具结果回答用户的原始请求。输出结构：先给 3-5 条结论式关键发现，再列数据质量问题与风险，最后给可操作建议。只引用对结论有支撑的关键数值，禁止逐项复述工具结果里的全部数字；总长度控制在 400 字以内。不得声称做过没有工具记录的操作。")
        try:
            response=self.llm.chat([LLMMessage(role="system",content=system),LLMMessage(role="user",content=f"用户原始请求：{run.user_request}\n\n工具结果（已压缩）：\n{json.dumps(views,ensure_ascii=False,default=str) or '（无有效工具结果）'}")])
        except Exception as exc:  # noqa: BLE001 — 只在 LLM 调用边界内捕获
            # ★ 与闲聊链路同口径：走了 LLM 但没走通 ⇒ 标降级，不算「远程生成」。
            # 但降级必须是被允许的（总闸 × 错误分类），否则如实失败——
            # 「工具都跑成功了，只是汇总时模型挂了」仍然是一次失败的运行，
            # 不该被静默包装成「已降级」蒙混过去。
            if not fallback_allowed(exc):
                raise
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        if not (response.content or "").strip():
            # 模型返回空串也算「这次没拿到模型输出」，不能标成远程生成。
            if not settings.AGENT_ALLOW_MODEL_FALLBACK:
                raise LLMException("远程大模型返回空内容")
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        run.answer_source=REMOTE_LLM_SUMMARY
        return response.content
