"""Prompt 087：WorkflowService。

create / get / list / update / run / cancel / clone。
第一版为进程内持久化（内存仓库）；run 为同步执行，
cancel 通过线程安全的事件标记在下一个节点边界生效。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import NotFoundException, ValidationException
from app.workflow.executor import NodeRunner, WorkflowExecutor, WorkflowRunResult
from app.workflow.models import Workflow
from app.workflow.state import NodeStatus
from app.workflow.validator import validate_workflow


@dataclass
class WorkflowRunHandle:
    """一次工作流运行的句柄（保存状态与取消标记）。"""

    run_id: str
    workflow_id: int
    status: NodeStatus = NodeStatus.PENDING
    result: WorkflowRunResult | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class WorkflowService:
    """工作流全生命周期管理（进程内仓库）。"""

    # 运行句柄保留上限。handle.result 会持有节点输出，其中包含 _df/_model/_pipeline
    # 这些运行时对象（完整数据集与训练好的模型），不设上限会让长驻进程把所有历史
    # 运行的内存一直占着。超限时按插入顺序淘汰最旧的「已结束」句柄。
    _MAX_RUNS = 200

    def __init__(
        self,
        node_runners: dict[str, NodeRunner] | None = None,
        required_config_keys: dict[str, set[str]] | None = None,
    ) -> None:
        self._workflows: dict[int, Workflow] = {}
        self._next_id = 1
        self._runs: dict[str, WorkflowRunHandle] = {}
        # run_id 序号必须独立于 _runs 的长度：用 len(_runs) 推导会在淘汰/清理后
        # 重号，新句柄直接覆盖掉同名活句柄（cancel 随即失效）。
        self._run_seq = 0
        self._run_lock = threading.Lock()
        # 工作流仓库锁：_next_id 自增与 _workflows 写入此前完全无锁（_run_lock 只保护 _runs），
        # 并发 create 会产生重复 id 并互相覆盖。共享单例下必须串行化。
        self._wf_lock = threading.RLock()
        self.node_runners: dict[str, NodeRunner] = dict(node_runners or {})
        # 必需参数规格（来自 runners.NODE_REQUIRED_CONFIG）。只在 run() 前的预检里用，
        # **刻意不参与 create/update 校验**：画布允许「先搭图、后填参数」的草稿，
        # 前端「流程体检」也把缺参数判为 warning 而非 error。若在创建时就拦截，
        # 用户将无法保存半成品流程。
        self.required_config_keys: dict[str, set[str]] = dict(required_config_keys or {})
        self._executor = WorkflowExecutor()

    # ------------------------------------------------------------------
    # create / get / list / update
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        name: str,
        nodes: list[dict[str, Any]] | list | None = None,
        edges: list[dict[str, Any]] | list | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Workflow:
        workflow = self._build_workflow(name, nodes, edges, metadata)
        self._ensure_valid(workflow)
        with self._wf_lock:
            workflow_id = self._next_id
            self._next_id += 1
            # id 通过 metadata 回传（保持 Workflow 为纯数据结构）。写入必须和入库
            # 处于同一个临界区：否则仓库里会短暂存在一个「没有 id」的工作流，而
            # 并发的 get()/to_dict() 恰好读到它时就会漏掉 id 字段。
            workflow.metadata["id"] = workflow_id
            self._workflows[workflow_id] = workflow
        return workflow

    def get(self, workflow_id: int) -> Workflow:
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise NotFoundException(
                "workflow not found", details={"workflow_id": workflow_id}
            )
        return workflow

    def list(self) -> list[dict[str, Any]]:
        with self._wf_lock:
            items = list(self._workflows.items())
        return [
            {"id": wid, "name": w.name, "nodes": len(w.nodes), "edges": len(w.edges)}
            for wid, w in sorted(items)
        ]

    def update(
        self,
        workflow_id: int,
        *,
        name: str | None = None,
        nodes: list | None = None,
        edges: list | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Workflow:
        workflow = self.get(workflow_id)
        new_name = name if name is not None else workflow.name
        new_nodes = (
            nodes
            if nodes is not None
            else [n.to_dict() for n in workflow.nodes]
        )
        new_edges = (
            edges if edges is not None else [e.to_dict() for e in workflow.edges]
        )
        new_metadata = dict(metadata if metadata is not None else workflow.metadata)
        updated = self._build_workflow(new_name, new_nodes, new_edges, new_metadata)
        self._ensure_valid(updated)
        updated.metadata["id"] = workflow_id
        with self._wf_lock:
            self._workflows[workflow_id] = updated
        return updated

    def delete(self, workflow_id: int) -> None:
        self.get(workflow_id)
        with self._wf_lock:
            self._workflows.pop(workflow_id, None)

    # ------------------------------------------------------------------
    # run / cancel / clone
    # ------------------------------------------------------------------
    def run(
        self,
        workflow_id: int,
        *,
        context: dict[str, Any] | None = None,
        node_runners: dict[str, NodeRunner] | None = None,
    ) -> WorkflowRunHandle:
        """同步执行工作流；节点边界响应取消请求。"""
        workflow = self.get(workflow_id)
        # 运行前预检：把「必需参数缺失」一次性全部报出来。
        # 不这样做的话，缺 dataset_id 的节点要等前面节点都跑完才报错——对 ml.train
        # 这类节点意味着白白先做一遍昂贵的数据加工。此检查只挡 run，不挡 create/update。
        self._ensure_runnable(workflow)
        runners = {**self.node_runners, **(node_runners or {})}
        # run_id 与登记动作必须在同一临界区：并发 run 若各自推导序号会互相覆盖句柄。
        with self._run_lock:
            self._run_seq += 1
            handle = WorkflowRunHandle(
                run_id=f"wf_{workflow_id}_{self._run_seq}",
                workflow_id=workflow_id,
            )
            self._runs[handle.run_id] = handle
            self._evict_runs_locked()
        handle.status = NodeStatus.RUNNING
        try:
            handle.result = self._executor.execute(
                workflow,
                runners,
                context=context,
                is_cancelled=handle.cancel_event.is_set,
            )
            handle.status = handle.result.status
        except Exception:
            handle.status = NodeStatus.FAILED
            raise
        return handle

    def cancel(self, run_id: str) -> bool:
        """请求取消运行（下一个节点边界生效）。"""
        handle = self._runs.get(run_id)
        if handle is None:
            raise NotFoundException("workflow run not found", details={"run_id": run_id})
        if handle.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
            handle.cancel_event.set()
            return True
        return False

    def get_run(self, run_id: str) -> WorkflowRunHandle:
        handle = self._runs.get(run_id)
        if handle is None:
            raise NotFoundException("workflow run not found", details={"run_id": run_id})
        return handle

    def _evict_runs_locked(self) -> None:
        """淘汰最旧的已结束运行句柄（调用方必须已持有 _run_lock）。

        只淘汰终态句柄：仍在 PENDING/RUNNING 的运行一旦被移除，cancel(run_id)
        就会报 404，用户将永远无法取消它。
        """
        if len(self._runs) <= self._MAX_RUNS:
            return
        # dict 保持插入顺序 ⇒ 从最旧的一条开始尝试淘汰。
        for run_id in list(self._runs):
            if len(self._runs) <= self._MAX_RUNS:
                break
            handle = self._runs.get(run_id)
            if handle is None:
                continue
            if handle.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
                continue
            self._runs.pop(run_id, None)

    def clone(self, workflow_id: int, *, new_name: str | None = None) -> Workflow:
        """深拷贝工作流为新工作流（名称缺省加 (copy) 后缀）。"""
        source = self.get(workflow_id)
        copy = Workflow.from_dict(source.to_dict())
        copy.name = new_name or f"{source.name} (copy)"
        copy.metadata = dict(copy.metadata)
        copy.metadata.pop("id", None)
        self._ensure_valid(copy)
        with self._wf_lock:
            workflow_id_new = self._next_id
            self._next_id += 1
            copy.metadata["id"] = workflow_id_new
            self._workflows[workflow_id_new] = copy
        return copy

    # ------------------------------------------------------------------
    @staticmethod
    def _build_workflow(
        name: str,
        nodes: list | None,
        edges: list | None,
        metadata: dict[str, Any] | None,
    ) -> Workflow:
        if not name:
            raise ValidationException("workflow name 不能为空")
        return Workflow.from_dict(
            {
                "name": name,
                "nodes": nodes or [],
                "edges": edges or [],
                "metadata": metadata or {},
            }
        )

    def _ensure_valid(self, workflow: Workflow) -> None:
        validation = validate_workflow(workflow, known_types=set(self.node_runners))
        if not validation.ok:
            raise ValidationException(
                "工作流校验失败",
                details={"errors": validation.errors, "warnings": validation.warnings},
            )

    def _ensure_runnable(self, workflow: Workflow) -> None:
        """运行前预检：结构 + 必需参数。

        与 _ensure_valid 分开是有意的：创建/更新只做结构校验（允许保存草稿），
        运行才要求参数齐全。未配置规格时退化为「不检查」，保持旧行为。
        """
        if not self.required_config_keys:
            return
        validation = validate_workflow(
            workflow,
            known_types=set(self.node_runners),
            required_config_keys=self.required_config_keys,
        )
        if not validation.ok:
            raise ValidationException(
                "工作流无法执行：配置不完整",
                details={"errors": validation.errors, "warnings": validation.warnings},
            )
