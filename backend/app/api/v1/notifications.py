"""通知 API。

- GET  /notifications           通知列表 + 未读数 + 版本号
- GET  /notifications/stream    SSE：仅在版本号变化时推送（替代客户端轮询）
- POST /notifications/read      标记单条已读
- POST /notifications/read-all  全部已读
- GET  /notifications/prefs     通知偏好（邮箱 + 类型开关）
- PUT  /notifications/prefs     更新通知偏好

推送 vs 轮询
------------
列表接口返回的每一项都含完整正文，浏览器每 15 秒拉一次等于在后台持续做
「序列化 200 条通知 + 反序列化」的无效功。这里改用版本号：SSE 只推
``unread`` 计数与 ``version``（几十字节），前端发现版本变了才去拉一次的列表。
**:func:`stream_notifications` 与列表接口因此必须共用同一份版本号**，
否则「推了变化但列表读不到」和「半天不推」会交替出现。
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.notifications import NOTIFICATION_STORE
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/notifications", tags=["notifications"])

# SSE 心跳间隔（秒）：中间层（nginx / 云 LB）常会掐掉长时间无字节的连接。
_HEARTBEAT_SECONDS = 20.0


class MarkReadRequest(BaseModel):
    notification_id: str


class PrefsUpdateRequest(BaseModel):
    email: str = ""
    notify_training: bool = True
    notify_report: bool = True
    notify_workflow: bool = True
    notify_permission: bool = False
    notify_system: bool = True


def _sse(event: str, payload: dict) -> str:
    """构造一条 SSE 帧；data 必须是单行 JSON（多行会破坏 SSE 帧边界）。"""
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {body}\n\n"


@router.get("", response_model=ApiResponse[dict])
def list_notifications() -> ApiResponse[dict]:
    # snapshot() 一次加锁取"列表+未读数+版本"，避免两次调用之间又插进来一条通知，
    # 导致前端「未读数对了但列表是旧的」。
    return ApiResponse[dict](data=NOTIFICATION_STORE.snapshot())


@router.get("/stream")
async def stream_notifications(request: Request) -> StreamingResponse:
    """通知变更推送（SSE）。

    刻意**只推版本号与未读数**，不推完整列表：
    1. 列表在收到变更事件后再由客户端拉一次，语义单一、易解释；
    2. 服务端不需要在每次变更时序列化 200 条通知。

    连接到达 ``NOTIFICATION_SSE_MAX_SECONDS`` 后主动结束，浏览器会自动重连，
    避免服务重启后残留一堆僵尸任务。
    """
    interval = max(float(settings.NOTIFICATION_SSE_INTERVAL_SECONDS), 0.5)
    max_seconds = max(float(settings.NOTIFICATION_SSE_MAX_SECONDS), 30.0)

    async def generator():
        last = NOTIFICATION_STORE.version()
        yield _sse("snapshot", NOTIFICATION_STORE.snapshot())
        deadline = time.time() + max_seconds
        since_beat = 0.0
        while True:
            await asyncio.sleep(interval)
            since_beat += interval
            if await request.is_disconnected():
                return
            current = NOTIFICATION_STORE.version()
            if current != last:
                last = current
                yield _sse(
                    "changed",
                    {"version": current, "unread": NOTIFICATION_STORE.unread_count()},
                )
                since_beat = 0.0
            elif since_beat >= _HEARTBEAT_SECONDS:
                # 注释行（`:` 开头）：SSE 合法的保活帧，不会被代理服务器当成空连接。
                yield ": keep-alive\n\n"
                since_beat = 0.0
            if time.time() > deadline:
                return

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 关掉 Nginx 的响应缓冲，否则推送会被攒着一起下发。
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/read", response_model=ApiResponse[dict])
def mark_read(body: MarkReadRequest) -> ApiResponse[dict]:
    NOTIFICATION_STORE.mark_read(body.notification_id)
    return ApiResponse[dict](data={"unread": NOTIFICATION_STORE.unread_count()})


@router.post("/read-all", response_model=ApiResponse[dict])
def mark_all_read() -> ApiResponse[dict]:
    NOTIFICATION_STORE.mark_all_read()
    return ApiResponse[dict](data={"unread": 0})


@router.get("/prefs", response_model=ApiResponse[dict])
def get_prefs() -> ApiResponse[dict]:
    return ApiResponse[dict](data=NOTIFICATION_STORE.get_prefs())


@router.put("/prefs", response_model=ApiResponse[dict])
def update_prefs(body: PrefsUpdateRequest) -> ApiResponse[dict]:
    return ApiResponse[dict](data=NOTIFICATION_STORE.update_prefs(body.model_dump()))
