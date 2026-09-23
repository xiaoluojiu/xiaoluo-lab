"""Agent Runtime models with lightweight JSON persistence."""
from __future__ import annotations
import json, logging, threading, time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from app.agent.answer_source import describe
from app.agent.executor.executor import ToolCallRecord
from app.core.config import settings
from app.core.exceptions import NotFoundException
from app.tools.result import ToolResult

logger = logging.getLogger("xiaoluo.agent.store")

# `usage` 是 Token 账本的增量快照事件：运行过程中就要能看到「花了多少 / 省了多少」，
# 不能只在结束后算总账（同一批改动的动机见 app/agent/answer_source.py）。
#: 事件类型（唯一真源）。``clarification`` = Pre-flight / agent.clarify 发起的结构化反问。
#: 新增事件类型必须同步这里与前端渲染，避免「事件发了但界面不认」。
EVENT_TYPES = (
    "route", "planning", "permission", "tool_call", "tool_result", "validation",
    "replanning", "completed", "failed", "chat", "usage",
    "preflight", "clarification",
)

@dataclass
class AgentTokenLedger:
    llm_calls:int=0; actual_input_tokens:int=0; actual_output_tokens:int=0; actual_total_tokens:int=0; estimated_context_saved_tokens:int=0; estimated_result_saved_tokens:int=0; avoided_planner_calls:int=0; cache_hits:int=0
    def check_budget(self):
        if self.llm_calls>=settings.AGENT_LLM_MAX_CALLS: raise RuntimeError(f"Agent LLM 调用次数已达到本次任务上限 {settings.AGENT_LLM_MAX_CALLS}")
        if self.actual_total_tokens>=settings.AGENT_LLM_MAX_TOTAL_TOKENS: raise RuntimeError(f"Agent Token 预算已达到本次任务上限 {settings.AGENT_LLM_MAX_TOTAL_TOKENS}")
    def record_usage(self,usage):
        usage=usage or {};self.llm_calls+=1;inp=int(usage.get("input_tokens",usage.get("prompt_tokens",0)) or 0);out=int(usage.get("output_tokens",usage.get("completion_tokens",0)) or 0);total=usage.get("total_tokens");self.actual_input_tokens+=inp;self.actual_output_tokens+=out;self.actual_total_tokens+=int(total if total is not None else inp+out)
    def record_context_saving(self,tokens:int):self.estimated_context_saved_tokens+=max(int(tokens or 0),0)
    def record_result_saving(self,tokens:int):self.estimated_result_saved_tokens+=max(int(tokens or 0),0)
    def record_cache_hit(self):self.cache_hits+=1;self.avoided_planner_calls+=1
    def to_dict(self):
        return {
            "llm_calls": self.llm_calls,
            "actual": {"input_tokens": self.actual_input_tokens, "output_tokens": self.actual_output_tokens, "total_tokens": self.actual_total_tokens},
            "budget": {
                "max_llm_calls": int(settings.AGENT_LLM_MAX_CALLS),
                "max_total_tokens": int(settings.AGENT_LLM_MAX_TOTAL_TOKENS),
                "remaining_llm_calls": max(int(settings.AGENT_LLM_MAX_CALLS) - self.llm_calls, 0),
                "remaining_total_tokens": max(int(settings.AGENT_LLM_MAX_TOTAL_TOKENS) - self.actual_total_tokens, 0),
            },
            "optimization": {"estimated_context_saved_tokens": self.estimated_context_saved_tokens, "estimated_result_saved_tokens": self.estimated_result_saved_tokens, "estimated_saved_tokens": self.estimated_context_saved_tokens + self.estimated_result_saved_tokens, "avoided_planner_calls": self.avoided_planner_calls, "plan_cache_hits": self.cache_hits},
            "note": "estimated_saved_tokens 是本地优化层估算值，不等同于 Provider 实际 usage。",
        }

@dataclass
class AgentEvent:
    seq:int;run_id:str;type:str;payload:dict[str,Any]=field(default_factory=dict);created_at:float=field(default_factory=time.time)
    def to_sse(self):return f"event: {self.type}\ndata: {json.dumps(self.to_dict(),ensure_ascii=False,default=str)}\n\n"
    def to_dict(self):return {"seq":self.seq,"run_id":self.run_id,"type":self.type,"payload":self.payload,"created_at":self.created_at}

class RunStatus(StrEnum):
    PENDING="pending";PLANNING="planning";RUNNING="running";WAITING_CONFIRMATION="waiting_confirmation";COMPLETED="completed";FAILED="failed"
    #: Pre-flight / agent.clarify 反问等待态（第一层改造）。
    #: 与 WAITING_CONFIRMATION 的区别：后者问「这个高风险操作要不要做」，
    #: 前者问「信息不完整，请补一个参数」。两者都是「等用户提供输入才能继续」。
    WAITING_CLARIFICATION="waiting_clarification"

@dataclass
class AgentRun:
    # answer_source：这条回答由谁产出（远程大模型 / 平台内置规则 / 调用失败降级）。
    # 取值与语义见 app/agent/answer_source.py；空串表示「尚未产生回答」。
    id:str;session_id:str;user_id:str;user_request:str;status:RunStatus=RunStatus.PENDING;plan:dict[str,Any]|None=None;tool_calls:list[ToolCallRecord]=field(default_factory=list);events:list[AgentEvent]=field(default_factory=list);final_answer:str="";error:str="";pending_confirmation:dict[str,Any]|None=None;token_ledger:AgentTokenLedger=field(default_factory=AgentTokenLedger);cancel_requested:bool=False;answer_source:str="";created_at:float=field(default_factory=time.time);started_at:float|None=None;finished_at:float|None=None
    #: Pre-flight 结果快照（第一层）：本次运行开工前判定了什么、问了什么。
    preflight:dict[str,Any]|None=None
    #: 用户对本轮反问的回答（code -> answer）。回答后重新规划，据此跳过已解决项。
    clarification_answers:dict[str,str]=field(default_factory=dict)
    @property
    def tool_call_count(self):return len(self.tool_calls)
    def elapsed(self):return max((self.finished_at or time.time())-(self.started_at or self.created_at),0.0)
    def summary(self):return {"id":self.id,"session_id":self.session_id,"user_request":self.user_request,"status":str(self.status),"final_answer":self.final_answer,"error":self.error,"plan":self.plan,"tool_calls":[c.to_dict() for c in self.tool_calls],"tool_call_count":self.tool_call_count,"cancel_requested":self.cancel_requested,"pending_confirmation":({"tool":self.pending_confirmation["call"].tool,"arguments":self.pending_confirmation["call"].arguments,"step_index":self.pending_confirmation["step_index"],"reason":self.pending_confirmation["call"].error} if self.pending_confirmation else None),"token_usage":self.token_ledger.to_dict(),"answer_source":describe(self.answer_source),"elapsed_seconds":round(self.elapsed(),3),"preflight":self.preflight,"clarification_answers":dict(self.clarification_answers)}
    def full(self):
        d=self.summary();d["events"]=[e.to_dict() for e in self.events];return d

@dataclass
class AgentSession:
    id:str;user_id:str;title:str="";dataset_ids:list[int]=field(default_factory=list);history:list[dict[str,str]]=field(default_factory=list);run_ids:list[str]=field(default_factory=list);created_at:float=field(default_factory=time.time);archived:bool=False
    def set_dataset_ids(self, dataset_ids:list[int]) -> list[int]:
        self.dataset_ids=list(dict.fromkeys(int(x) for x in dataset_ids if int(x)>0));return self.dataset_ids
    def summary(self):return {"id":self.id,"user_id":self.user_id,"title":self.title,"dataset_ids":self.dataset_ids,"history":self.history,"run_ids":self.run_ids,"created_at":self.created_at,"archived":self.archived}

class AgentStore:
    """线程安全的轻量 JSON 持久化；一个 Session 同一时间只允许一个活动 Turn。"""
    ACTIVE_STATUSES={RunStatus.PENDING,RunStatus.PLANNING,RunStatus.RUNNING,RunStatus.WAITING_CONFIRMATION,RunStatus.WAITING_CLARIFICATION}
    # 事件级 persist 的最小间隔（秒）。_persist_locked 每次都会把**全部**会话与运行（含每条事件）
    # 序列化重写一遍，而一次运行会产生几十个事件，逐事件落盘是明显的写放大。
    _PERSIST_MIN_INTERVAL=0.5
    def __init__(self,path=None):
        self._lock=threading.RLock();self._sessions={};self._runs={};self._session_seq=0;self._run_seq=0;self._last_persist=0.0;self._path=Path(path or (Path(__file__).resolve().parents[3]/"data"/"agent_store.json"));self._load()
    def next_session_id(self):
        with self._lock:self._session_seq+=1;return f"s-{self._session_seq}"
    def next_run_id(self):
        with self._lock:self._run_seq+=1;return f"r-{self._run_seq}"
    def add_session(self,s):
        with self._lock:self._sessions[s.id]=s;self._persist_locked()
        return s
    def get_session(self,i):
        with self._lock:x=self._sessions.get(i)
        if x is None:raise NotFoundException("Agent 会话不存在",details={"session_id":i})
        return x
    def active_run(self,session_id:str):
        with self._lock:
            s=self.get_session(session_id)
            for run_id in reversed(s.run_ids):
                run=self._runs.get(run_id)
                if run and run.status in self.ACTIVE_STATUSES:return run
        return None
    def update_session_context(self,i,dataset_ids:list[int]):
        with self._lock:
            active=self.active_run(i)
            if active is not None:
                raise ValueError(f"会话存在活动运行 {active.id}，暂不能修改数据集上下文；请等待当前 Turn 完成。")
            x=self.get_session(i);x.set_dataset_ids(dataset_ids);self._persist_locked();return x
    def list_sessions(self,user_id=None,*,include_archived:bool=True):
        # 必须持锁取快照：无锁遍历 dict.values() 时若另一线程正在 add/delete 会话，
        # 会抛 "dictionary changed size during iteration"（500）。
        with self._lock:snapshot=list(self._sessions.values())
        items=[s for s in snapshot if (not user_id or s.user_id==user_id) and (include_archived or not s.archived)]
        # 归档会话沉底，其余按创建时间倒序；保证「未归档的最新会话」始终在最前。
        return sorted(items,key=lambda x:(1 if x.archived else 0,-x.created_at))
    def set_session_archived(self,i,archived:bool):
        # 归档不销毁数据，只改变列表可见性，因此不要求会话空闲；删除才要求空闲。
        with self._lock:
            x=self.get_session(i);x.archived=bool(archived);self._persist_locked();return x
    def delete_session(self,i):
        """删除会话及其全部运行记录。存在活动运行（含等待确认）时拒绝，避免 worker 线程回写已删除对象。"""
        with self._lock:
            x=self.get_session(i)
            active=self.active_run(i)
            if active is not None:
                raise ValueError(f"会话存在活动运行 {active.id}，请先停止当前 Turn 再删除。")
            for run_id in list(x.run_ids):self._runs.pop(run_id,None)
            self._sessions.pop(i,None);self._persist_locked();return x
    def add_run(self,r):
        with self._lock:
            s=self.get_session(r.session_id)
            active=self.active_run(r.session_id)
            if active is not None: raise ValueError(f"会话已有活动运行 {active.id}，同一会话一次只执行一个 Turn。")
            # run_ids 由 store 维护（幂等），active_run 依赖它判断活动状态
            if r.id not in s.run_ids: s.run_ids.append(r.id)
            self._runs[r.id]=r;self._persist_locked()
        return r
    def get_run(self,i):
        with self._lock:x=self._runs.get(i)
        if x is None:raise NotFoundException("Agent 运行不存在",details={"run_id":i})
        return x
    def persist(self,*,force:bool=False):
        """落盘；事件级调用（force=False）按最小间隔节流，终态与显式调用 force=True 立即写。

        节流只影响「运行中」的中间态：completed/failed 与 cancel/deny/会话变更均已强制落盘，
        因此刷新页面或重启后不会丢最终状态（重启本来也会把未终态的运行标记为中断）。
        """
        with self._lock:
            now=time.time()
            if not force and (now-self._last_persist)<self._PERSIST_MIN_INTERVAL:return
            self._last_persist=now;self._persist_locked()
    def _persist_locked(self):
        try:
            self._path.parent.mkdir(parents=True,exist_ok=True);data={"session_seq":self._session_seq,"run_seq":self._run_seq,"sessions":[x.summary() for x in self._sessions.values()],"runs":[x.full() for x in self._runs.values()]};tmp=self._path.with_suffix(".tmp");tmp.write_text(json.dumps(data,ensure_ascii=False,default=str),encoding="utf-8");tmp.replace(self._path)
        except Exception as exc:  # noqa: BLE001 - 持久化失败不能打断运行，但必须可观测
            logger.warning("AgentStore 持久化失败（%s）：%s", self._path, exc)
    def _load(self):
        try:
            if not self._path.exists():return
            data=json.loads(self._path.read_text(encoding="utf-8"));self._session_seq=int(data.get("session_seq",0));self._run_seq=int(data.get("run_seq",0))
            for x in data.get("sessions",[]):self._sessions[x["id"]]=AgentSession(id=x["id"],user_id=x.get("user_id","anonymous"),title=x.get("title",""),dataset_ids=list(x.get("dataset_ids",[])),history=list(x.get("history",[])),run_ids=list(x.get("run_ids",[])),created_at=float(x.get("created_at",time.time())),archived=bool(x.get("archived",False)))
            for x in data.get("runs",[]):
                status=RunStatus(x.get("status","pending"))
                # WAITING_CONFIRMATION 同样标记为中断：重启后 pending_confirmation
                # 未完整序列化、无法恢复授权流程，保留该状态只会让会话永久 409。
                interrupted=status in {RunStatus.PENDING,RunStatus.PLANNING,RunStatus.RUNNING,RunStatus.WAITING_CONFIRMATION,RunStatus.WAITING_CLARIFICATION}
                if interrupted: status=RunStatus.FAILED
                r=AgentRun(id=x["id"],session_id=x.get("session_id",""),user_id=x.get("user_id","anonymous"),user_request=x.get("user_request",""),status=status,plan=x.get("plan"),final_answer=x.get("final_answer",""),error=(x.get("error","") or "") if not interrupted else "后端进程重启，上一轮 Agent Turn 被中断；原执行过程已保留，可重新发起任务。",created_at=float(x.get("created_at",time.time())),started_at=x.get("started_at"),finished_at=x.get("finished_at") or (time.time() if interrupted else None));u=x.get("token_usage",{});a=u.get("actual",{});o=u.get("optimization",{});r.token_ledger=AgentTokenLedger(llm_calls=int(u.get("llm_calls",0)),actual_input_tokens=int(a.get("input_tokens",0)),actual_output_tokens=int(a.get("output_tokens",0)),actual_total_tokens=int(a.get("total_tokens",0)),estimated_context_saved_tokens=int(o.get("estimated_context_saved_tokens",0)),estimated_result_saved_tokens=int(o.get("estimated_result_saved_tokens",0)),avoided_planner_calls=int(o.get("avoided_planner_calls",0)),cache_hits=int(o.get("plan_cache_hits",0)))
                for e in x.get("events",[]):r.events.append(AgentEvent(seq=int(e.get("seq",len(r.events)+1)),run_id=r.id,type=e.get("type","planning"),payload=dict(e.get("payload") or {}),created_at=float(e.get("created_at",time.time()))))
                for c in x.get("tool_calls",[]):
                    q=ToolCallRecord(step_index=int(c.get("step_index",0)),tool=str(c.get("tool","")),arguments=dict(c.get("arguments") or {}),attempt=int(c.get("attempt",1)));q.status=str(c.get("status","ok"));q.error=str(c.get("error",""));z=c.get("result")
                    if z is not None:q.result=ToolResult(success=bool(z.get("success")),data=z.get("data"),summary=str(z.get("summary","")),warnings=list(z.get("warnings") or []),errors=list(z.get("errors") or []),metadata=dict(z.get("metadata") or {}));q.finished_at=time.time()
                    r.tool_calls.append(q)
                self._runs[r.id]=r
            # 旧运行若被恢复为中断状态，立即写回磁盘，避免每次启动重复标记。
            self._persist_locked()
        except Exception:self._sessions.clear();self._runs.clear()
