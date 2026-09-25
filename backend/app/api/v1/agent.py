"""Prompt 131-132：Agent API。"""

from __future__ import annotations

import json
import queue
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.api.deps import get_agent_runtime
from app.core.config import settings
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentSessionCreate(BaseModel):
    user_id: str = "anonymous"
    title: str = ""
    dataset_ids: list[int] = Field(default_factory=list)


class AgentSessionContextUpdate(BaseModel):
    dataset_ids: list[int] = Field(default_factory=list)


class AgentSessionArchiveUpdate(BaseModel):
    archived: bool = True


class AgentMessageRequest(BaseModel):
    content: str
    stream: bool = False
    dataset_ids: list[int] | None = None


class AgentClarifyRequest(BaseModel):
    """回答 Agent 的结构化反问（第一层：Pre-flight / agent.clarify）。

    ``answer`` 是**机读值**：取反问里的 ``options[].value``，
    不是自由文本。前端应渲染成选择器而不是输入框。
    """

    answer: str = Field(..., min_length=1, max_length=500)
    # 安全约束（S-1/S-2）：
    # - 不接受客户端传入 confirmed —— 高风险确认只能走 POST /agent/runs/{id}/confirm
    # - 不接受客户端传入 role —— 角色由服务端决定（当前固定 analyst），防止提权到 admin


def _ensure_idle(runtime: AgentRuntime, session_id: str) -> None:
    active = runtime.store.active_run(session_id)
    if active is not None:
        raise HTTPException(status_code=409, detail={"code": "agent_turn_active", "message": f"当前会话正在执行运行 {active.id}，请等待当前 Turn 完成。", "run_id": active.id})


# S-6：_ensure_idle 检查与 worker 线程 store.add_run 之间存在竞态窗口，
# 用进程内会话预留集合堵住：请求线程先占位，worker 结束后释放。
_RESERVE_LOCK = threading.Lock()
_RESERVED_SESSIONS: set[str] = set()


def _try_reserve(session_id: str) -> bool:
    with _RESERVE_LOCK:
        if session_id in _RESERVED_SESSIONS:
            return False
        _RESERVED_SESSIONS.add(session_id)
        return True


def _release_reservation(session_id: str) -> None:
    with _RESERVE_LOCK:
        _RESERVED_SESSIONS.discard(session_id)


#: SSE tail 循环的退出条件：落到这些状态就收流。
#
# 注意 WAITING_CLARIFICATION **必须**在这里：澄清态的恢复走独立的
# ``POST /runs/{id}/clarify``（与 /confirm 一样是同步跑完的另一个 HTTP 请求），
# 而前端在 SSE 未结束时 ``busy`` / ``sendingRef`` 都不会复位——界面停在「发送中」、
# 用户输入被静默吞掉，用户根本没有机会去回答那个问题。
# 早期这里只有 completed/failed（那时还没有澄清态），于是澄清态被当成「还在跑」，
# tail 一直挂到 AGENT_SSE_CONFIRM_WAIT_SECONDS（默认 900s）—— 这就是「8% 永久不动」。
# 收流后前端走 refreshRun + 轮询拉全量事件，恢复阶段的事件通过 /clarify 的响应拿到。
#
# WAITING_CONFIRMATION **不**在这里：授权闭环依赖同一条流把 resume 之后的事件
# 继续推给浏览器（见 _sse_live_run 的 docstring），这条路径已验证，保持原样。
_TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.WAITING_CLARIFICATION}


def _sse_from_events(events: list) -> StreamingResponse:
    def generate():
        for event in events:
            yield event.to_sse()
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


def _sse_live_run(runtime: AgentRuntime, session, content: str) -> StreamingResponse:
    """跑到运行结束（含「等待确认 → 用户确认 → 继续执行」的完整闭环）。

    过去 SSE 在 runtime.run() 返回时就关闭，而此时运行可能只是停在
    WAITING_CONFIRMATION：
      - 用户点确认后 resume 在另一个 HTTP 请求里同步执行，事件只写进 run.events，
        没有任何通道推给浏览器 → 界面从「已确认」开始就再无进展，看起来卡死；
      - 若计划里还有第二个受控工具，UI 甚至看不到新的确认提示。
    这里让 SSE 在等待确认期间保持打开，并持续 tail run.events，把 resume 之后
    产生的事件继续推给同一条流，前端无需改动即可看到完整过程。
    """
    # 调用方已完成 _ensure_idle + 会话预留
    events: queue.Queue = queue.Queue()
    # 客户端断开标记：由 generate() 在流被关闭时置位，让 tail 线程立即退出，
    # 而不是一直挂到 AGENT_SSE_CONFIRM_WAIT_SECONDS（默认 900s）。
    tail_stop = threading.Event()

    def on_event(event) -> None:
        events.put(event)

    def emit_new(seen_seq: int) -> int:
        """把 run 上新增的事件补发到流里，返回最新 seq。"""
        try:
            current = runtime.get_run(run_holder[0].id)
        except Exception:  # noqa: BLE001 - run 被删除时直接结束流
            return seen_seq
        for event in current.events:
            if getattr(event, "seq", 0) > seen_seq:
                seen_seq = event.seq
                events.put(event)
        return seen_seq

    run_holder: list = []
    original_run = runtime.run

    def worker() -> None:
        try:
            run = original_run(session, content, on_event=on_event)
            run_holder.append(run)
        except Exception as exc:
            # ★ exc 必须在 except 块内**立刻取值**。
            # 之前把 str(exc) 写进了下面那个 lambda 的 f-string 里，而 lambda 是
            # 在 SSE 取元素时才被调用的 —— 那时 except 块早已结束，Python 已经
            # 隐式 `del exc`，于是本该报告「为什么失败」的这行代码自己抛
            # NameError，SSE 流被掐断，前端只看到"连接中断"而没有任何原因。
            error_payload = json.dumps(str(exc), ensure_ascii=False)
            # 运行对象尚未建立时也要把失败原因送回 UI，避免 SSE 静默结束。
            events.put(type("BootstrapEvent", (), {"to_sse": lambda self: f'event: failed\ndata: {{"payload": {{"error": {error_payload}}}}}\n\n'})())
            _release_reservation(session.id)
            events.put(None)
            return

        seen = len(run_holder[0].events)
        deadline = time.time() + max(float(settings.AGENT_SSE_CONFIRM_WAIT_SECONDS), 1.0)
        try:
            # 进入等待确认：保持连接，等用户确认后继续 tail 事件，直到运行终态
            while time.time() < deadline and not tail_stop.is_set():
                status = run_holder[0].status
                seen = emit_new(seen)
                if status in _TERMINAL_STATUSES:
                    break
                # resume 是同步跑完的，留出一点余量确保事件都已落盘再收尾
                time.sleep(0.3 if status == RunStatus.WAITING_CONFIRMATION else 0.15)
        except Exception:  # noqa: BLE001 - tail 失败不应影响已产出结果
            pass
        finally:
            runtime.store.persist(force=True)
            _release_reservation(session.id)
            events.put(None)

    threading.Thread(target=worker, daemon=True, name="agent-run").start()

    def generate():
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event.to_sse()
            yield "event: done\ndata: {}\n\n"
        except GeneratorExit:
            tail_stop.set()
            raise
        except Exception:  # noqa: BLE001 - 客户端断开后 ASGI 会在下一次 send 抛错
            tail_stop.set()
            raise

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/sessions", response_model=ApiResponse[dict])
def create_session(body: AgentSessionCreate, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    session = runtime.create_session(user_id=body.user_id, title=body.title, dataset_ids=body.dataset_ids)
    return ApiResponse[dict](data=session.summary())


@router.get("/sessions", response_model=ApiResponse[list])
def list_sessions(
    user_id: str | None = Query(None),
    include_archived: bool = Query(True, description="是否包含已归档会话；归档会话始终排在最后"),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ApiResponse[list]:
    sessions = runtime.list_sessions(user_id, include_archived=include_archived)
    return ApiResponse[list](data=[s.summary() for s in sessions])


@router.patch("/sessions/{session_id}", response_model=ApiResponse[dict])
def update_session(session_id: str, body: AgentSessionArchiveUpdate, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    """归档 / 取消归档。归档只改变列表可见性，不删除历史与运行记录。"""
    session = runtime.set_session_archived(session_id, body.archived)
    return ApiResponse[dict](data=session.summary())


@router.delete("/sessions/{session_id}", response_model=ApiResponse[dict])
def delete_session(session_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    try:
        session = runtime.delete_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "agent_turn_active", "message": str(exc)}) from exc
    _release_reservation(session_id)
    return ApiResponse[dict](data={"id": session.id, "deleted": True})


@router.patch("/sessions/{session_id}/context", response_model=ApiResponse[dict])
def update_session_context(session_id: str, body: AgentSessionContextUpdate, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    try:
        session = runtime.store.update_session_context(session_id, body.dataset_ids)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "agent_turn_active", "message": str(exc)}) from exc
    return ApiResponse[dict](data=session.summary())


@router.post("/sessions/{session_id}/messages")
def post_message(session_id: str, body: AgentMessageRequest, runtime: AgentRuntime = Depends(get_agent_runtime)):
    session = runtime.get_session(session_id)
    if body.dataset_ids is not None:
        try:
            session = runtime.store.update_session_context(session_id, body.dataset_ids)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "agent_turn_active", "message": str(exc)}) from exc
    _ensure_idle(runtime, session_id)
    if not _try_reserve(session_id):
        raise HTTPException(status_code=409, detail={"code": "agent_turn_active", "message": "当前会话已有进行中的请求，请稍后再试。"})
    if body.stream:
        return _sse_live_run(runtime, session, body.content)
    try:
        run = runtime.run(session, body.content)
    finally:
        _release_reservation(session_id)
    runtime.store.persist(force=True)
    data = run.summary()
    if run.status == RunStatus.WAITING_CONFIRMATION:
        data["hint"] = "存在高风险操作待确认：POST /agent/runs/{id}/confirm 继续执行"
    return ApiResponse[dict](data=data)


@router.get("/runs/{run_id}", response_model=ApiResponse[dict])
def get_run(run_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    return ApiResponse[dict](data=runtime.get_run(run_id).full())


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)):
    return _sse_from_events(runtime.get_run(run_id).events)


@router.post("/runs/{run_id}/confirm", response_model=ApiResponse[dict])
def confirm_run(run_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    run = runtime.get_run(run_id)
    if run.status != RunStatus.WAITING_CONFIRMATION or run.pending_confirmation is None:
        return ApiResponse[dict](data={**run.summary(), "hint": "该运行不在等待确认状态，已返回当前进度"})
    session = runtime.get_session(run.session_id)
    resumed = runtime.resume(session, run_id)
    runtime.store.persist(force=True)
    return ApiResponse[dict](data=resumed.summary())


@router.post("/runs/{run_id}/clarify", response_model=ApiResponse[dict])
def clarify_run(run_id: str, body: AgentClarifyRequest, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    """回答 Agent 的反问（第一层改造）。

    与 ``/confirm`` 的区别：``/confirm`` 是授权（做不做），``/clarify`` 是补信息
    （做哪个）。回答后运行从 Pre-flight 处重新规划，或从发起反问的那一步继续，
    前面的重型步骤不重跑。
    """
    run = runtime.get_run(run_id)
    if run.status != RunStatus.WAITING_CLARIFICATION or run.pending_clarification is None:
        return ApiResponse[dict](data={**run.summary(), "hint": "该运行不在等待澄清状态，已返回当前进度"})
    resumed = runtime.answer_clarification(run_id, body.answer)
    runtime.store.persist(force=True)
    return ApiResponse[dict](data=resumed.summary())


@router.post("/runs/{run_id}/deny", response_model=ApiResponse[dict])
def deny_run(run_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    """S-3：拒绝授权必须落库终止运行，否则会话被 WAITING_CONFIRMATION 永久锁死。"""
    run = runtime.deny(run_id)
    return ApiResponse[dict](data=run.summary())


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse[dict])
def cancel_run(run_id: str, runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    """T-10：请求取消运行；等待授权状态下取消等价于拒绝授权。"""
    run = runtime.cancel(run_id)
    data = run.summary()
    if run.cancel_requested:
        data["hint"] = "已请求取消，运行将在当前步骤结束后停止"
    return ApiResponse[dict](data=data)


def _trace_markdown(run) -> str:
    """把一次 Agent 运行导出为可读的 Markdown 执行轨迹。"""
    status = str(run.status)
    lines = [
        "# Agent 执行轨迹",
        "",
        f"- 运行 ID：`{run.id}`（会话 `{run.session_id}`）",
        f"- 用户请求：{run.user_request}",
        f"- 状态：{status} · 耗时 {run.elapsed():.1f}s · 工具调用 {run.tool_call_count} 次",
        f"- Token：{run.token_ledger.actual_total_tokens}（LLM 调用 {run.token_ledger.llm_calls} 次）",
        "",
    ]
    if run.error:
        lines += ["## 失败原因", "", run.error, ""]
    if run.plan:
        lines += ["## 执行计划", "", f"目标：{run.plan.get('goal', '')}", ""]
        for i, step in enumerate(run.plan.get("steps", []), 1):
            if isinstance(step, dict):
                lines.append(f"{i}. `{step.get('tool')}` · 期望：{step.get('expected_output', '-') or '-'}")
        lines.append("")
    if run.tool_calls:
        lines += ["## 工具调用", ""]
        for call in run.tool_calls:
            args = json.dumps(call.arguments, ensure_ascii=False, default=str)
            summary = call.result.summary if call.result else ""
            lines.append(f"### 第 {call.step_index + 1} 步 `{call.tool}`（{call.status}，第 {call.attempt} 次尝试，{call.elapsed_ms:.0f} ms）")
            lines.append(f"- 参数：`{args}`")
            if summary:
                lines.append(f"- 摘要：{summary}")
            if call.error:
                lines.append(f"- 错误：{call.error}")
            if call.result and call.result.warnings:
                lines.append(f"- 警告：{'；'.join(str(w) for w in call.result.warnings)}")
            lines.append("")
    if run.events:
        lines += ["## 事件时间线", "", "| 时间 | 事件 | 内容 |", "| --- | --- | --- |"]
        for ev in run.events:
            payload = json.dumps(ev.payload, ensure_ascii=False, default=str)
            if len(payload) > 160:
                payload = payload[:160] + "…"
            lines.append(f"| {ev.created_at:.1f} | {ev.type} | {payload.replace('|', '\\|')} |")
        lines.append("")
    if run.final_answer:
        lines += ["## 最终回答", "", run.final_answer, ""]
    return "\n".join(lines)


@router.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str, format: str = Query("md", pattern="^(json|md)$"), runtime: AgentRuntime = Depends(get_agent_runtime)):
    """T-7：导出执行轨迹（Markdown 供人读，JSON 供复盘/分享）。"""
    run = runtime.get_run(run_id)
    if format == "json":
        return ApiResponse[dict](data=run.full())
    return PlainTextResponse(_trace_markdown(run), media_type="text/markdown; charset=utf-8")


@router.get("/tools", response_model=ApiResponse[list])
def list_tools(runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[list]:
    return ApiResponse[list](data=runtime.registry.list())


@router.get("/capabilities", response_model=ApiResponse[dict])
def get_capabilities(runtime: AgentRuntime = Depends(get_agent_runtime)) -> ApiResponse[dict]:
    """暴露 Agent 能力参数与预算上限，供前端在 UI 中展示预算、校验和策略提示。"""
    return ApiResponse[dict](data={
        "agent": settings.agent_context_summary(),
        "llm": settings.llm_model_summary(),
        "tools": {"count": len(runtime.registry.list()), "names": runtime.registry.names()},
    })
