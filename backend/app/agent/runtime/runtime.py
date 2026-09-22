"""Agent Runtime：对话路由 -> 元数据上下文 -> 工具检索 -> 规划 -> 工具执行 -> 结果汇总。"""
from __future__ import annotations
import json
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
from app.agent.context.builder import ContextBuilder
from app.agent.context.models import AgentContext
from app.agent.executor.executor import AgentExecutor, ToolCallRecord
from app.agent.llm.base import LLMMessage, LLMProvider
from app.agent.permission.models import ROLE_PERMISSIONS
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner, PlanInvalidError
from app.agent.planner.replanner import AgentLimitExceeded, ReplanLimits, Replanner
from app.agent.runtime.models import AgentEvent, AgentRun, AgentSession, AgentStore, RunStatus
from app.agent.validator.validator import AgentResultValidator
from app.core.config import settings
from app.core.exceptions import AgentException, ValidationException
from app.data_engine.service import DataEngineService
from app.tools.base import ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.registry import TOOL_REGISTRY, ToolRegistry

EventCallback = Callable[[AgentEvent], None]

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
        try:
            with self._usage_scope(run):
                self._set_status(run, RunStatus.PLANNING)
                mode, route_reason = self._route(
                    run.user_request, plan_override=plan_override, has_datasets=bool(session.dataset_ids)
                )
                self._emit(run,"route",{"mode":mode,"reason":route_reason},on_event)
                # 开局就发一次账本快照：预算上限是「这次会不会被截断」的关键信息，
                # 只在运行结束后才给出的话，用户无法在过程中判断还能不能继续问。
                self._emit_usage(run,on_event)
                if plan_override is None: self._trace_route(run, session, mode, route_reason)
                if plan_override is None and mode == "chat":
                    self._emit(run,"chat",{"stage":"direct_chat"},on_event); self._direct_chat(run,session,on_event); return run
                context=self._build_context(run,session); self._emit(run,"planning",{"stage":"context_ready","data_access":"metadata_only"},on_event)
                all_tools=self.registry.list(); self._fill_tools(context,all_tools); candidate_tools=self._candidate_tools(context,all_tools)
                retrieval=context.tool_context.get("retrieval_scores") or {}
                self._emit(run,"planning",{"stage":"tools_retrieved","count":len(candidate_tools),"tools":[t.get("name") for t in candidate_tools],"retrieval":retrieval},on_event)
                plan=plan_override or self.planner.build_plan_resilient(run.user_request, context, candidate_tools, all_tools=all_tools); run.plan=plan.to_dict(); self._emit(run,"planning",{"stage":"plan_ready","goal":plan.goal,"steps":len(plan.steps),"cache_hit":bool(getattr(plan,"cache_hit",False))},on_event)
                # 规划器是本次运行的第一笔真实开销（除非命中 Plan Cache），这里立刻反映到账本上。
                self._emit_usage(run,on_event)
                if not plan.steps: self._direct_chat(run,session,on_event); return run
                run.status=RunStatus.RUNNING; self._run_plan(run,session,context,role,plan,offset=0,attempts={},confirmed=confirmed,on_event=on_event)
        except AgentLimitExceeded as exc: self._fail(run,str(exc),on_event)
        except AgentException as exc: self._fail(run,exc.message,on_event)
        except Exception as exc: self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally: run.finished_at=time.time(); self._trace_outcome(run, session)
        return run

    GREETINGS={"你好","您好","嗨","哈喽","hello","hi","hey","早上好","下午好","晚上好","谢谢","感谢","在吗","你是谁","你叫什么","再见","拜拜"}
    # 说明：这里漏词会直接表现为「明明是数据任务却走普通对话」——用户说「关联 / 拼接 / 宽表」
    # 这类常见说法时不含「合并」二字，请求会被当闲聊，只回一句固定的「尚未配置大模型」。
    DATA_TERMS=("数据","数据集","csv","excel","xlsx","表格","字段","列","行","分析","处理","清洗","缺失","重复","异常","质量","统计","描述","分布","相关","相关性","可视化","图表","eda","训练","模型","预测","分类","回归","机器学习","特征","目标列","读取","导入","导出","转换","合并","筛选","聚合","排序","去重","运行","实验","检测","画像","profile","inspect","python","pandas","workflow","工作流","流程","报告","report","pdf",
                # 多表关联类说法
                "关联","join","merge","拼接","宽表","连接","外键",
                # 建模评估类说法
                "建模","评估","准确率","auc","特征工程","聚类")

    def _route(self, text: str, *, plan_override: AgentPlan | None = None, has_datasets: bool = False) -> tuple[str, str]:
        """对话/工具路由。返回 (mode, reason)；reason 用于 route 事件向用户解释路由依据。"""
        if plan_override is not None:
            return "agent", "服务端指定计划（plan_override），直接进入工具流程"
        t = text.strip().lower()
        if t in self.GREETINGS:
            return "chat", "命中问候语，无需数据工具"
        matched = [term for term in self.DATA_TERMS if term in t]
        if matched:
            return "agent", f"命中数据任务关键词：{'、'.join(matched[:5])}{'…' if len(matched) > 5 else ''}"
        # 会话已绑定数据集时，需求几乎一定围绕这批数据展开；关键词漏召回
        # （如「关联一下」「看看分布」）不应退化成一句「尚未配置大模型」。
        if has_datasets and len(t) >= 4:
            return "agent", "会话已绑定数据集，默认进入工具流程（关键词未命中时的兜底）"
        return "chat", "未命中数据任务关键词，走普通对话"

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
            except Exception as exc:
                # ★ 开关开着 ≠ 这次真的是模型在答。请求失败后下面这段是**固定文案**，
                # 必须标成降级，否则界面会把「规则兜底」显示成「远程模型生成」。
                answer=f"暂时无法完成对话请求：{exc}"
                run.answer_source=LLM_ERROR_FALLBACK
        run.final_answer=answer; run.status=RunStatus.COMPLETED; session.history.append({"role":"assistant","content":answer}); self._emit_usage(run,on_event); self._emit(run,"completed",{"final_answer":answer,"mode":"chat","answer_source":describe(run.answer_source)},on_event)

    def _candidate_tools(self,context:AgentContext,all_tools:list[dict[str,Any]])->list[dict[str,Any]]:
        names={x.get("name") for x in context.tool_context.get("tools",[])}; return [t for t in all_tools if t.get("name") in names]

    def resume(self,session:AgentSession,run_id:str,*,on_event:EventCallback|None=None)->AgentRun:
        run=self.store.get_run(run_id); pending=run.pending_confirmation
        if run.status!=RunStatus.WAITING_CONFIRMATION or pending is None: raise ValidationException("该运行不在等待确认状态")
        run.pending_confirmation=None; run.status=RunStatus.RUNNING; step_index=int(pending["step_index"]); self._emit(run,"permission",{"stage":"confirmed","tool":pending["call"].tool,"step_index":step_index},on_event)
        plan=self._plan_from_dict(run.plan or {}); continue_plan=AgentPlan(goal=plan.goal,steps=plan.steps[step_index:])
        try:
            with self._usage_scope(run):
                context=self._build_context(run,session); self._run_plan(run,session,context,self._role_of(run),continue_plan,offset=step_index,attempts={step_index:1},confirmed=True,on_event=on_event)
        except AgentLimitExceeded as exc:self._fail(run,str(exc),on_event)
        except AgentException as exc:self._fail(run,exc.message,on_event)
        except Exception as exc:self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally:run.finished_at=time.time()
        return run

    def deny(self, run_id: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """用户拒绝高风险操作：终止该运行并释放会话（否则 run 永远停在 WAITING_CONFIRMATION，会话被 409 锁死）。"""
        run = self.store.get_run(run_id)
        if run.status != RunStatus.WAITING_CONFIRMATION or run.pending_confirmation is None:
            return run
        run.pending_confirmation = None
        self._fail(run, "用户拒绝授权，运行终止", on_event)
        self.store.persist(force=True)
        return run

    def cancel(self, run_id: str) -> AgentRun:
        """请求取消运行：设置标记，_run_plan 在当前步骤结束后停止。"""
        run = self.store.get_run(run_id)
        if run.status in (RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RUNNING):
            run.cancel_requested = True
            self.store.persist(force=True)
        elif run.status == RunStatus.WAITING_CONFIRMATION:
            # 等待授权时取消等价于拒绝授权
            run = self.deny(run_id)
        return run

    def _run_plan(self,run:AgentRun,session:AgentSession,context:AgentContext,role:str,plan:AgentPlan,*,offset:int,attempts:dict[int,int],confirmed:bool,on_event:EventCallback|None)->None:
        """执行计划；失败即交给 `Replanner` 决定「重试 / 跳过 / 终止」。

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
        tool_ctx=self._tool_context(session,role); services=self._services(); current=plan; permanent_failure=""; total_steps=max(len(plan.steps),1); replans=0
        while current.steps:
            # ① 重规划总闸：与单步尝试上限互补，任何原因的循环到这里都会被截断
            if replans>=self.replanner.limits.max_replans:
                permanent_failure=f"重规划已达上限 {self.replanner.limits.max_replans} 次，判定为不可恢复的循环，已终止。"; break
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
                record=self.executor.execute_step(step,tool_ctx,services,confirmed=confirmed,attempt=attempt_no,step_index=abs_idx); run.tool_calls.append(record)
                if record.status=="needs_confirmation":
                    run.status=RunStatus.WAITING_CONFIRMATION; run.pending_confirmation={"call":record,"step_index":abs_idx}; self._emit(run,"permission",{"stage":"confirmation_required","tool":step.tool,"reason":record.error,"step_index":abs_idx},on_event); return
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
    def _fill_tools(self,context:AgentContext,tools:list[dict[str,Any]])->None: ContextBuilder(self.data_engine).with_tools(context,tools,registry=self.registry)
    def _role_of(self,run:AgentRun)->str:return "analyst"
    def _tool_context(self,session:AgentSession,role:str)->ToolExecutionContext:
        permissions=ROLE_PERMISSIONS.get(role,ROLE_PERMISSIONS["viewer"]); return ToolExecutionContext(user_id=session.user_id,session_id=session.id,dataset_ids=set(session.dataset_ids),permissions=set(permissions))
    def _services(self)->ToolServices: return ToolServices(dataset_service=self.data_engine.dataset_service,data_engine_service=self.data_engine,experiment_service=self.experiment_service,db=self.db)
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
        views=[];summaries=[]
        for call in run.tool_calls:
            if call.status!="ok" or call.result is None:continue
            summaries.append(f"{call.tool}：{call.result.summary}");view=self._result_for_llm(run,call,compact=True)
            if view is not None:views.append({"tool":call.tool,"result":view})
        fallback=f"已完成 {len(summaries)} 个真实数据工具步骤。\n"+"\n".join(f"{i}. {s}" for i,s in enumerate(summaries,1)) if summaries else "任务没有产生有效的数据处理结果。"
        # 无远程模型 ⇒ 汇总也是规则拼的，如实标注（数据本身仍是真实工具跑出来的）。
        if self.llm is None:
            run.answer_source=PLATFORM_RULES_SUMMARY
            return fallback
        system=("你是小洛实验室的数据分析助手。基于真实工具结果回答用户的原始请求。输出结构：先给 3-5 条结论式关键发现，再列数据质量问题与风险，最后给可操作建议。只引用对结论有支撑的关键数值，禁止逐项复述工具结果里的全部数字；总长度控制在 400 字以内。不得声称做过没有工具记录的操作。")
        try:
            response=self.llm.chat([LLMMessage(role="system",content=system),LLMMessage(role="user",content=f"用户原始请求：{run.user_request}\n\n工具结果（已压缩）：\n{json.dumps(views,ensure_ascii=False,default=str) or '（无有效工具结果）'}")])
        except Exception:
            # ★ 与闲聊链路同口径：走了 LLM 但没走通 ⇒ 标降级，不算「远程生成」。
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        if not (response.content or "").strip():
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        run.answer_source=REMOTE_LLM_SUMMARY
        return response.content
