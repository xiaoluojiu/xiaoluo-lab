"""通知 API。

- GET  /notifications           通知列表 + 未读数
- POST /notifications/read      标记单条已读
- POST /notifications/read-all  全部已读
- GET  /notifications/prefs     通知偏好（邮箱 + 类型开关）
- PUT  /notifications/prefs     更新通知偏好
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.notifications import NOTIFICATION_STORE
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/notifications", tags=["notifications"])


class MarkReadRequest(BaseModel):
    notification_id: str


class PrefsUpdateRequest(BaseModel):
    email: str = ""
    notify_training: bool = True
    notify_report: bool = True
    notify_workflow: bool = True
    notify_permission: bool = False
    notify_system: bool = True


@router.get("", response_model=ApiResponse[dict])
def list_notifications() -> ApiResponse[dict]:
    items = NOTIFICATION_STORE.list()
    return ApiResponse[dict](data={"items": items, "unread": NOTIFICATION_STORE.unread_count()})


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
    prefs = NOTIFICATION_STORE.update_prefs(body.model_dump())
    return ApiResponse[dict](data=prefs)
