"""Workflow API（Prompt 201）。

- POST   /workflows                      创建（校验 DAG）
- GET    /workflows                      列表
- GET    /workflows/{id}                 详情
- PUT    /workflows/{id}                 更新
- DELETE /workflows/{id}                 删除
- POST   /workflows/{id}/run             同步执行
- GET    /workflows/runs/{run_id}        运行状态/日志
- POST   /workflows/runs/{run_id}/cancel 请求取消
- POST   /workflows/{id}/clone           克隆
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_data_engine_service, get_dataset_service, get_workflow_service
from app.notifications import notify
from app.schemas.common import ApiResponse
from app.workflow.service import WorkflowRunHandle, WorkflowService

router = APIRouter(prefix="/workflows", tags=["workflows"])


class WorkflowBody(BaseModel):
    name: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunRequest(BaseModel):
    context: dict[str, Any] = Field(default_factory=dict)


def _workflow_dict(workflow) -> dict[str, Any]:
    data = workflow.to_dict()
    data["id"] = workflow.metadata.get("id")
    return data


def _run_dict(handle: WorkflowRunHandle) -> dict[str, Any]:
    result = None
    if handle.result is not None:
        from app.workflow.runners import sanitize_output

        data = handle.result.to_dict()
        data["outputs"] = {k: sanitize_output(v) for k, v in data["outputs"].items()}
        result = data
    return {
        "run_id": handle.run_id,
        "workflow_id": handle.workflow_id,
        "status": str(handle.status.value),
        "result": result,
    }


@router.post("", response_model=ApiResponse[dict])
def create_workflow(
    body: WorkflowBody,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    workflow = service.create(
        name=body.name, nodes=body.nodes, edges=body.edges, metadata=body.metadata
    )
    return ApiResponse[dict](data=_workflow_dict(workflow))


@router.get("", response_model=ApiResponse[list])
def list_workflows(
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[list]:
    return ApiResponse[list](data=service.list())


@router.get("/runs/{run_id}", response_model=ApiResponse[dict])
def get_run(
    run_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=_run_dict(service.get_run(run_id)))


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse[dict])
def cancel_run(
    run_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data={"cancelled": service.cancel(run_id)})


@router.get("/{workflow_id}", response_model=ApiResponse[dict])
def get_workflow(
    workflow_id: int,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=_workflow_dict(service.get(workflow_id)))


@router.put("/{workflow_id}", response_model=ApiResponse[dict])
def update_workflow(
    workflow_id: int,
    body: WorkflowBody,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    workflow = service.update(
        workflow_id, name=body.name, nodes=body.nodes, edges=body.edges,
        metadata=body.metadata,
    )
    return ApiResponse[dict](data=_workflow_dict(workflow))


@router.delete("/{workflow_id}", response_model=ApiResponse[dict])
def delete_workflow(
    workflow_id: int,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    service.delete(workflow_id)
    return ApiResponse[dict](data={"deleted": True, "workflow_id": workflow_id})


@router.post("/{workflow_id}/run", response_model=ApiResponse[dict])
def run_workflow(
    workflow_id: int,
    body: WorkflowRunRequest,
    service: WorkflowService = Depends(get_workflow_service),
    dataset_service=Depends(get_dataset_service),
    data_engine_service=Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """同步执行；数据类节点经 context 获得请求级服务。"""
    context = {
        **(body.context or {}),
        "dataset_service": dataset_service,
        "data_engine_service": data_engine_service,
    }
    handle = service.run(workflow_id, context=context)
    _notify_workflow_done(handle, service)
    return ApiResponse[dict](data=_run_dict(handle))


@router.post("/{workflow_id}/clone", response_model=ApiResponse[dict])
def clone_workflow(
    workflow_id: int,
    service: WorkflowService = Depends(get_workflow_service),
) -> ApiResponse[dict]:
    workflow = service.clone(workflow_id)
    return ApiResponse[dict](data=_workflow_dict(workflow))


def _notify_workflow_done(handle: WorkflowRunHandle, service: WorkflowService) -> None:
    """工作流运行结束后写一条通知（成功/失败/取消）。"""
    status = str(handle.status.value)
    try:
        name = service.get(handle.workflow_id).name
    except Exception:  # noqa: BLE001
        name = f"工作流 #{handle.workflow_id}"
    if status == "success":
        notify(
            "workflow",
            "工作流运行完成",
            f"「{name}」执行成功，可查看各节点输出。",
            link="/workflow",
        )
    elif status == "failed":
        notify(
            "workflow",
            "工作流运行失败",
            f"「{name}」执行失败，请查看运行日志。",
            link="/workflow",
        )
    # cancelled 不发通知：用户主动取消，无需提醒。
