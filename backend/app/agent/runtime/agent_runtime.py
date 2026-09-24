"""Public AgentRuntime facade.

基类已内置 Turn 参数解析（{{stepN.field}} 结构化依赖）与逐事件增量持久化挂钩，
facade 只保留两件事：
1. 关键事件发生时增量落盘（浏览器刷新/断线重连后运行过程不丢）。

授权范围不再由本层定义：一次确认只放行被确认的那一步，判定统一在
``runtime.AgentRuntime._run_plan`` 里按 ``confirmed_step_index`` 完成。
此前这里靠覆写 ``_run_plan`` 把授权放大到「整份计划」，理由是「confirm 是同步返回、
原始 SSE 流已关闭，前端感知不到第二次确认」；该前提现已不成立 —— 前端在 confirm
之后会继续轮询 run，第二次 WAITING_CONFIRMATION 能被正常取回。
"""
from __future__ import annotations

from app.agent.runtime.models import AgentRun
from app.agent.runtime.runtime import AgentRuntime as _AgentRuntime


class AgentRuntime(_AgentRuntime):
    """AgentRuntime 增强层：事件增量持久化（授权范围由基类统一判定）。"""

    def _emit(self, run: AgentRun, event_type: str, payload: dict, on_event=None) -> None:
        """事件先进入统一事件流，再增量落盘。

        这样浏览器离开 AI Lab、重新进入或刷新页面时，运行过程不会只存在于当前
        SSE 连接内；同时只在关键 Agent 事件发生时写盘，避免每个内部对象变更都触发 I/O。
        """
        super()._emit(run, event_type, payload, on_event)
        if event_type in {"route", "planning", "permission", "tool_call", "tool_result", "validation", "replanning", "completed", "failed", "chat"}:
            # 终态必须立即落盘；中间事件走节流，避免每个事件都全量重写 store。
            self.store.persist(force=event_type in {"completed", "failed"})

    # 曾在这里覆写 `_run_plan` 把「一次确认」放大成「放行整份计划」，
    # 现已删除：授权范围收敛到单步（见本模块头部说明），基类实现即正确口径。
