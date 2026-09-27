"""Agent API（数据分析 Agent）。

端点与事件名**照旧**，前端零改动；内部实现已整体替换为
「规则路由 → 固定 playbook → 有界执行 → 模板/LLM 渲染」。

关于 SSE
--------
``POST /sessions/{id}/messages`` 且 ``stream=true`` 时返回 ``text/event-stream``，
一帧一个完整事件对象（前端按 ``\\n\\n`` 切帧，读 ``event:`` 与 ``data:``）。

三条不能违反的约定：

1. 每条事件都要带 ``run_id`` —— 前端用它作为 confirm/deny/cancel/clarify 的目标
2. 结束帧必须是 ``event: done`` —— 收到它前端才停止读取
3. **挂起时不关闭流**：等待确认/补充信息期间流保持打开，
   恢复后的 ``completed`` 必须经同一条流补发，那是答案进入聊天区的唯一出口
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.channels import CHANNELS, EMPTY, is_terminal_event, sse_frame, wait_deadline
from app.agent.engine import MAX_LLM_CALLS, MAX_STEPS
from app.agent.intents import is_abandonment
from app.agent.models import AgentRun, EventType, RunStatus
from app.agent.store import AgentStore
from app.api.deps import build_agent_engine, get_agent_store
from app.schemas.common import ApiResponse
from app.tools.builtin import TOOL_REGISTRY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

#: SSE 在挂起状态下的最长等待时间（秒）。
#: 前端有两条超时：空闲 120s（收到事件就重置）与硬上限 620s（对齐引擎的
#: ``WALL_CLOCK_SECONDS``），到点会 abort 并转轮询；这里放宽到 15 分钟，
#: 是为了让「用户稍后才点确认」仍能走同一条流拿到答案。
SSE_WAIT_SECONDS = 900.0


# ----------------------------------------------------------------------
# 请求体
# ----------------------------------------------------------------------
class SessionCreate(BaseModel):
    user_id: str = "anonymous"
    title: str = "新会话"
    dataset_ids: list[int] = Field(default_factory=list)


class SessionPatch(BaseModel):
    archived: bool = False


class SessionContextPatch(BaseModel):
    dataset_ids: list[int] = Field(default_factory=list)


class MessageRequest(BaseModel):
    content: str
    stream: bool = True
    dataset_ids: list[int] = Field(default_factory=list)


class ClarifyRequest(BaseModel):
    answer: str


# ----------------------------------------------------------------------
# 会话
# ----------------------------------------------------------------------
@router.post("/sessions", response_model=ApiResponse[dict])
def create_session(body: SessionCreate, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    session = store.create_session(user_id=body.user_id, title=body.title, dataset_ids=body.dataset_ids)
    return ApiResponse[dict](data=session.to_dict())


@router.get("/sessions", response_model=ApiResponse[list])
def list_sessions(
    user_id: str = "anonymous",
    include_archived: bool = False,
    store: AgentStore = Depends(get_agent_store),
) -> ApiResponse[list]:
    sessions = store.list_sessions(user_id=user_id, include_archived=include_archived)
    return ApiResponse[list](data=[s.to_dict() for s in sessions])


@router.patch("/sessions/{session_id}", response_model=ApiResponse[dict])
def patch_session(
    session_id: str,
    body: SessionPatch,
    store: AgentStore = Depends(get_agent_store),
) -> ApiResponse[dict]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    session.archived = bool(body.archived)
    store.update_session(session)
    return ApiResponse[dict](data=session.to_dict())


@router.delete("/sessions/{session_id}", response_model=ApiResponse[dict])
def delete_session(session_id: str, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    """删除会话。

    删之前必须先把**挂起态**运行收成终态，否则永远删不掉（历史缺陷）：

    挂起（等待确认 / 等待补充信息）时引擎线程已在挂起点退出，没有任何人会再推进
    这条 run。而 store 的守卫只认「terminal」，会把 waiting 也算成「未结束」→ 一律
    409。前端那个「停止」按钮只对**当前正在查看**的 run 有效，从列表 / 归档视图够
    不着，用户就卡在「既删不掉、也停不了」的死局里。

    收成终态还有个副作用是必要的：挂起期间 SSE 流仍开着，``_finalize_cancelled``
    会经 store 发一条 completed 并关通道，流才能干净收尾（否则要干等 900s 超时）。

    对**真正在推进**的运行（pending / planning / running）线程还活着，强删会把 run
    变成没人引用的孤儿，仍按 409 拒绝，前端据此提示「请先停止运行」。
    """
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    for run_id in list(session.run_ids):
        run = store.get_run(run_id)
        if run is not None and run.status.waiting:
            _finalize_cancelled(run, store)

    ok, message = store.delete_session(session_id)
    if not ok:
        status = 409 if "未结束" in message else 404
        raise HTTPException(status_code=status, detail=message or "删除失败")
    return ApiResponse[dict](data={"id": session_id, "deleted": True})


@router.patch("/sessions/{session_id}/context", response_model=ApiResponse[dict])
def patch_session_context(
    session_id: str,
    body: SessionContextPatch,
    store: AgentStore = Depends(get_agent_store),
) -> ApiResponse[dict]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    session.dataset_ids = list(body.dataset_ids)
    store.update_session(session)
    return ApiResponse[dict](data=session.to_dict())


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------
@router.get("/runs/{run_id}", response_model=ApiResponse[dict])
def get_run(run_id: str, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    return ApiResponse[dict](data=run.to_dict())


@router.post("/runs/{run_id}/confirm", response_model=ApiResponse[dict])
async def confirm_run(run_id: str, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    """放行当前待确认的高风险步骤。

    一次性凭据只对「被确认的那一个 (step_index, tool)」生效 —— 写成
    ``authorized_key is None 也算通过`` 会让一次确认放行整条高风险链。
    """
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    if run.status is not RunStatus.WAITING_CONFIRMATION or run.pending_confirmation is None:
        # 幂等：重复确认不改变状态，直接回当前运行
        return ApiResponse[dict](data=run.to_dict())

    pending = run.pending_confirmation
    run.authorized_key = f"{pending.step_index}:{pending.tool}"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    store.update_run(run)

    _resume(run_id, store)
    return ApiResponse[dict](data=run.to_dict())


@router.post("/runs/{run_id}/deny", response_model=ApiResponse[dict])
def deny_run(run_id: str, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    if run.pending_confirmation is None:
        return ApiResponse[dict](data=run.to_dict())

    tool = run.pending_confirmation.tool
    index = run.pending_confirmation.step_index
    run.pending_confirmation = None
    run.authorized_key = None
    run.status = RunStatus.COMPLETED
    run.final_answer = f"已拒绝「{tool}」的授权，该步骤未执行。"
    run.finished_at = time.time()
    store.add_event(
        run,
        EventType.COMPLETED,
        {"final_answer": run.final_answer, "answer_source": None, "denied_step": index},
    )
    store.update_run(run)
    channel = CHANNELS.get(run_id)
    if channel is not None:
        channel.close()
    return ApiResponse[dict](data=run.to_dict())


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse[dict])
def cancel_run(run_id: str, store: AgentStore = Depends(get_agent_store)) -> ApiResponse[dict]:
    """请求停止。分两种情形，处理方式不同：

    - **运行中**（pending / planning / running）：只打标记，引擎在**步骤边界**
      检查它，不会强杀正在跑的工具（长训练/报告要跑完当前步骤才停）。
    - **挂起中**（等待确认 / 等待补充信息）：引擎线程**已经退出**，没人会再推进
      这条 run，标记只会被写进磁盘然后石沉大海。这里必须当场收成终态，
      否则界面永远停在「等待确认」—— 既不能继续，也不能结束，只能刷新页面。
    """
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    if run.status.terminal:
        # 幂等：已经结束的运行再点停止不改变任何东西
        return ApiResponse[dict](data=run.to_dict())

    if run.status.waiting:
        _finalize_cancelled(run, store)
        return ApiResponse[dict](data=run.to_dict())

    run.cancel_requested = True
    store.update_run(run)
    return ApiResponse[dict](data=run.to_dict())


def _finalize_cancelled(run: AgentRun, store: AgentStore) -> None:
    """把挂起中的运行收成「已停止」终态。

    ``completed`` 事件必须经 store 发出：等待期间 SSE 流还开着，那是这条运行
    唯一能把最终话术送进聊天区的出口；直接只改状态不改事件，前端会有状态无文案。
    """
    if run.pending_confirmation is not None:
        what = f"「{run.pending_confirmation.tool}」"
        run.final_answer = f"已停止本次运行，{what}未执行，数据未做任何改动。"
    elif run.pending_clarification is not None:
        tool = run.pending_clarification.tool or "当前步骤"
        run.final_answer = f"已停止本次运行，未继续执行「{tool}」。"
    else:
        run.final_answer = "已停止本次运行。"

    run.pending_confirmation = None
    run.pending_clarification = None
    run.authorized_key = None
    run.cancel_requested = True
    run.status = RunStatus.COMPLETED
    run.finished_at = time.time()
    store.add_event(
        run,
        EventType.COMPLETED,
        {"final_answer": run.final_answer, "answer_source": None, "cancelled": True},
    )
    store.update_run(run)
    # 关闭通道让挂着的 SSE 流收尾（先发事件再关，队列是 FIFO，事件不会丢）
    channel = CHANNELS.get(run.id)
    if channel is not None:
        channel.close()


@router.post("/runs/{run_id}/clarify", response_model=ApiResponse[dict])
async def clarify_run(
    run_id: str,
    body: ClarifyRequest,
    store: AgentStore = Depends(get_agent_store),
) -> ApiResponse[dict]:
    """回答待补充的信息，并从挂起点继续。"""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    answer = (body.answer or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="回答不能为空")
    if run.status is not RunStatus.WAITING_CLARIFICATION or run.pending_clarification is None:
        return ApiResponse[dict](data=run.to_dict())

    request = run.pending_clarification
    # code 形如 slot.column —— 去掉前缀即工具参数名
    slot = request.code.split(".", 1)[1] if request.code.startswith("slot.") else request.code
    value: Any = answer
    if slot in ("run_id", "right_dataset_id", "dataset_id"):
        try:
            value = int(answer)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{slot} 需要是数字") from None

    run.clarification_answers[slot] = value
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    store.append_history(run.session_id, "user", answer)
    store.add_event(
        run,
        EventType.CLARIFICATION,
        {"stage": "answered", "answer": answer, "code": request.code, "step_index": request.step_index},
    )
    store.update_run(run)

    _resume(run_id, store)
    return ApiResponse[dict](data=run.to_dict())


@router.get("/runs/{run_id}/trace")
def get_trace(run_id: str, format: str = "md", store: AgentStore = Depends(get_agent_store)) -> PlainTextResponse:
    """导出运行轨迹（前端用 <a download> 直连，不走信封）。"""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    if format == "json":
        payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str)
        return PlainTextResponse(payload, media_type="application/json")
    return PlainTextResponse(_trace_markdown(run), media_type="text/markdown; charset=utf-8")


# ----------------------------------------------------------------------
# 元信息
# ----------------------------------------------------------------------
@router.get("/tools", response_model=ApiResponse[list])
def list_tools() -> ApiResponse[list]:
    return ApiResponse[list](data=TOOL_REGISTRY.list())


@router.get("/capabilities", response_model=ApiResponse[dict])
def capabilities() -> ApiResponse[dict]:
    """能力摘要。结构与前端 CapabilitiesPanel 一一对应（字段缺一即显示「—」）。"""
    from app.core.config import settings

    tools = TOOL_REGISTRY.list()
    return ApiResponse[dict](
        data={
            "agent": {
                "context_max_chars": settings.AGENT_CONTEXT_MAX_CHARS,
                "context_sections": {
                    "user_request": settings.AGENT_CONTEXT_USER_REQUEST_CHARS,
                    "dataset": settings.AGENT_CONTEXT_DATASET_CHARS,
                    "permissions": settings.AGENT_CONTEXT_PERMISSION_CHARS,
                    "tools": settings.AGENT_CONTEXT_TOOL_CHARS,
                    "history": settings.AGENT_CONTEXT_HISTORY_CHARS,
                },
                "history_messages": settings.AGENT_CONTEXT_HISTORY_MESSAGES,
                # 新架构不缓存 DataFrame：每个工具只读自己需要的列
                "dataset_cache": {"enabled": False, "max_items": 0},
                "max_steps": MAX_STEPS,
                "llm_budget": {
                    "max_calls": MAX_LLM_CALLS,
                    "max_input_tokens": settings.AGENT_LLM_MAX_INPUT_TOKENS,
                    "max_output_tokens": settings.AGENT_LLM_MAX_OUTPUT_TOKENS,
                    "max_total_tokens": settings.AGENT_LLM_MAX_TOTAL_TOKENS,
                },
                # 旧架构的「开关」在这里全部固化：路由是确定的，压缩是默认行为，
                # 没有 plan cache，也没有重规划。如实回填，避免前端显示假开关。
                "agent_policy": {
                    "allow_model_fallback": True,
                    "enable_tool_retrieval": False,
                    "tool_retrieval_top_k": 0,
                    "tool_retrieval_min_score": 0.0,
                    "enable_result_compression": True,
                    "enable_plan_cache": False,
                    "plan_cache_max_items": 0,
                },
            },
            "llm": settings.llm_model_summary(),
            "tools": {"count": len(tools), "names": [t["name"] for t in tools]},
        }
    )


# ----------------------------------------------------------------------
# 发消息（SSE）
# ----------------------------------------------------------------------
def _pending_run(store: AgentStore, session: Any) -> AgentRun | None:
    """会话里还挂着的那条运行（等待确认 / 等待补充信息）。"""
    for run_id in reversed(list(getattr(session, "run_ids", ()) or ())):
        run = store.get_run(run_id)
        if run is not None and run.status.waiting:
            return run
    return None


def _finish_with_answer(store: AgentStore, session_id: str, run: AgentRun, answer: str) -> None:
    """把一条新建的运行直接收成终态（用于「改口」这类不需要跑工具的回合）。

    completed 事件必须发出去：SSE 流是答案进入聊天区的唯一出口。
    """
    run.status = RunStatus.COMPLETED
    run.final_answer = answer
    run.finished_at = time.time()
    store.add_event(
        run,
        EventType.COMPLETED,
        {"final_answer": answer, "answer_source": None, "abandoned": True},
    )
    store.update_run(run)
    store.append_history(session_id, "assistant", answer)
    channel = CHANNELS.get(run.id)
    if channel is not None:
        channel.close()


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    body: MessageRequest,
    store: AgentStore = Depends(get_agent_store),
) -> Any:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    if body.dataset_ids:
        session.dataset_ids = list(body.dataset_ids)
        store.update_session(session)
    store.append_history(session_id, "user", content)

    # 上一条还挂着（等确认 / 等补充信息），用户却又发了新消息 ——
    # 那条挂起的运行再也没人推进，界面会永远停在「等待确认」。
    # 新指令本身就代表放弃旧的，这里直接把它收成终态。
    pending = _pending_run(store, session)
    abandoned = pending is not None and is_abandonment(content)
    if pending is not None:
        _finalize_cancelled(pending, store)

    run = store.create_run(session_id, content, dataset_ids=session.dataset_ids)

    if abandoned:
        # 用户说的是「算了 / 不用了」，不是新指令。系统此前会把它当成新诉求
        # 再跑一遍高风险操作 —— 等确认清洗时说「不用清洗了」，结果真的洗了一遍。
        _finish_with_answer(store, session_id, run, "好的，这一步不做了，数据没有改动。")
        if not body.stream:
            return ApiResponse[dict](data=run.to_dict())
        channel = CHANNELS.get_or_create(run.id, start_seq=0)
        return StreamingResponse(
            _event_stream(run.id, session_id, channel, store),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if not body.stream:
        # 非流式：在线程里跑完再返回（前端当前不使用这条分支，保留以兼容）
        engine = build_agent_engine()
        await asyncio.to_thread(engine.run, run, session)
        return ApiResponse[dict](data=run.to_dict())

    channel = CHANNELS.get_or_create(run.id, start_seq=0)
    return StreamingResponse(
        _event_stream(run.id, session_id, channel, store),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 反向代理必须关掉缓冲，否则 SSE 会被攒成一坨再吐
            "X-Accel-Buffering": "no",
        },
    )


async def _event_stream(run_id: str, session_id: str, channel: RunChannel, store: AgentStore) -> Any:
    """产出 SSE 帧。

    顺序很重要：**先注册监听者，再补发历史，最后消费队列**。
    反过来的话，补发期间新产生的事件会先进队列，造成重复推送。
    """
    run = store.get_run(run_id)
    if run is None:
        return

    def _listener(event: Any) -> None:
        channel.publish(event)

    store.add_listener(run_id, _listener)
    session = store.get_session(session_id)
    if session is None:
        store.remove_listener(run_id, _listener)
        return
    engine = build_agent_engine()
    task = asyncio.create_task(asyncio.to_thread(engine.run, run, session))

    deadline = wait_deadline(SSE_WAIT_SECONDS)
    try:
        # 补发已有事件（重连 / 切会话重建时间线靠它）
        for event in list(run.events):
            yield _frame(event)
            channel.mark_started(event.seq)

        while time.time() < deadline:
            item = await asyncio.to_thread(channel.poll, 1.0)
            if item is None:  # 通道已关闭
                break
            if item is EMPTY:
                current = store.get_run(run_id)
                if current is None:
                    break
                # 挂起中：继续等（用户还没点确认/补充信息）
                if current.status.waiting:
                    continue
                if current.status.terminal and task.done():
                    break
                continue
            yield _frame(item)
            if is_terminal_event(item):
                channel.close()
                break
    finally:
        store.remove_listener(run_id, _listener)
        if not task.done():
            task.cancel()
        CHANNELS.discard(run_id)
    # 控制帧：前端收到即停止读取
    yield sse_frame("done", "{}")


async def _noop() -> None:
    return None


def _frame(event: Any) -> str:
    return sse_frame(str(event.type), json.dumps(event.to_dict(), ensure_ascii=False, default=str))


def _trace_markdown(run: Any) -> str:
    lines: list[str] = [
        f"# Agent 运行轨迹 {run.id}",
        "",
        f"- 会话：{run.session_id}",
        f"- 状态：{run.status}",
        f"- 请求：{run.user_request}",
        "",
        "## 事件",
        "",
    ]
    for event in run.events:
        lines.append(f"{event.seq}. [{event.type}] {json.dumps(event.payload, ensure_ascii=False, default=str)}")
    lines += ["", "## 工具调用", ""]
    for call in run.tool_calls:
        lines.append(f"- 步骤 {call.step_index} · {call.tool} · {call.status}")
        if call.error:
            lines.append(f"  - 错误：{call.error}")
    if run.final_answer:
        lines += ["", "## 最终回答", "", run.final_answer]
    return "\n".join(lines)


def _resume(run_id: str, store: AgentStore) -> None:
    """从挂起点继续执行（后台线程）。

    刻意**不复用**请求级依赖：线程里自建引擎与会话，避免请求结束后
    session 被关闭导致的线程间误用。
    """
    run = store.get_run(run_id)
    if run is None:
        return
    session = store.get_session(run.session_id)
    if session is None:
        return
    engine = build_agent_engine()

    def _task() -> None:
        try:
            engine.run(run, session)
        except Exception:  # noqa: BLE001
            logger.exception("恢复运行失败（run=%s）", run_id)

    asyncio.get_event_loop().create_task(asyncio.to_thread(_task))
