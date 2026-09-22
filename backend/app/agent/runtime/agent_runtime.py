"""Public AgentRuntime facade.

基类已内置 Turn 参数解析（{{stepN.field}} 结构化依赖）与逐事件增量持久化挂钩，
facade 只保留两件事：
1. 关键事件发生时增量落盘（浏览器刷新/断线重连后运行过程不丢）；
2. 把 confirm 的授权范围定义为「用户已批准的计划本身」——一次确认放行本次计划内
   其余受控工具，避免每遇到一个高风险工具就重新弹确认。
"""
from __future__ import annotations

from app.agent.runtime.models import AgentRun
from app.agent.runtime.runtime import AgentRuntime as _AgentRuntime


class AgentRuntime(_AgentRuntime):
    """AgentRuntime 增强层：事件增量持久化 + 单工具授权范围控制。"""

    def _emit(self, run: AgentRun, event_type: str, payload: dict, on_event=None) -> None:
        """事件先进入统一事件流，再增量落盘。

        这样浏览器离开 AI Lab、重新进入或刷新页面时，运行过程不会只存在于当前
        SSE 连接内；同时只在关键 Agent 事件发生时写盘，避免每个内部对象变更都触发 I/O。
        """
        super()._emit(run, event_type, payload, on_event)
        if event_type in {"route", "planning", "permission", "tool_call", "tool_result", "validation", "replanning", "completed", "failed", "chat"}:
            # 终态必须立即落盘；中间事件走节流，避免每个事件都全量重写 store。
            self.store.persist(force=event_type in {"completed", "failed"})

    def _run_plan(
        self,
        run: AgentRun,
        session,
        context,
        role: str,
        plan,
        *,
        offset: int,
        attempts: dict[int, int],
        confirmed: bool,
        on_event=None,
    ) -> None:
        """确认后的继续执行：一次确认放行本次计划内所有受控工具。

        历史实现把授权范围收敛到「当前 pending 的单个工具」，第二个受控工具会再次
        进入 WAITING_CONFIRMATION。但 confirm API 是同步返回、且原始 SSE 流已关闭，
        前端既拿不到新事件也感知不到「又需要确认」，表现就是点击确认后界面卡死。

        用户点确认的对象是他看到的**整份执行计划**（UI 会展示计划步骤），因此授权语义
        应当是「批准这份计划」：计划内后续工具不再重复拦截。真正的越权保护由
        PermissionManager 的权限判定与deny / cancel 承担，不依赖逐步二次确认。
        """
        if not confirmed:
            return super()._run_plan(run, session, context, role, plan, offset=offset, attempts=attempts, confirmed=False, on_event=on_event)

        return super()._run_plan(run, session, context, role, plan, offset=offset, attempts=attempts, confirmed=True, on_event=on_event)
