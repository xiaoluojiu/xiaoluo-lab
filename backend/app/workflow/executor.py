"""Prompt 085：Workflow Executor（DAG 执行器）。

- 拓扑排序后逐节点执行，节点输入 = 所有直接前驱的输出
- 状态流转：PENDING -> RUNNING -> SUCCESS/FAILED/SKIPPED（取消为 CANCELLED）
- 日志：每个节点记录开始 / 结束时间、状态、错误、耗时
  不变式：logs 与节点一一对应（每个节点至多一条日志），skip 由信息最精确的
  来源写出，因此 len(logs) 恒等于节点数 —— 调用方可以放心用它做分母。
- 失败：节点异常 -> FAILED，其后继（含传递后继）-> SKIPPED；其余分支继续
- 取消：执行前检查取消标记，命中后剩余节点全部 CANCELLED
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import WorkflowException
from app.workflow.models import Workflow
from app.workflow.state import NodeStatus
from app.workflow.validator import validate_workflow

# 节点执行器：node, upstream_outputs, context -> 该节点输出（任意 JSON 可序列化值）
NodeRunner = Callable[[Any, dict[str, Any], dict[str, Any]], Any]
# 取消探测器：返回 True 表示请求取消
CancelProbe = Callable[[], bool]


@dataclass
class NodeLog:
    node_id: str
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class WorkflowRunResult:
    status: NodeStatus
    node_states: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    logs: list[NodeLog] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == NodeStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "node_states": dict(self.node_states),
            "outputs": dict(self.outputs),
            "logs": [log.to_dict() for log in self.logs],
        }


class WorkflowExecutor:
    """DAG 工作流执行器。"""

    def execute(
        self,
        workflow: Workflow,
        node_runners: dict[str, NodeRunner],
        *,
        context: dict[str, Any] | None = None,
        is_cancelled: CancelProbe | None = None,
    ) -> WorkflowRunResult:
        """执行工作流。

        node_runners: {节点类型: 执行函数}；类型未注册在执行前报错。
        context: 工作流级共享上下文（可被所有节点读写）。
        """
        validation = validate_workflow(workflow, known_types=set(node_runners))
        if not validation.ok:
            raise WorkflowException(
                "工作流校验失败", details={"errors": validation.errors}
            )

        order = self._topo_order(workflow)
        result = WorkflowRunResult(status=NodeStatus.RUNNING)
        ctx = context if context is not None else {}
        successors = self._successor_map(workflow)
        # 节点表与前驱表一次性取出。原先在循环里反复调用 workflow.node_map()
        # （每次重建整张字典）与 workflow.predecessors()（每次扫全量边），
        # 节点一多就是 O(N²)/O(N·E) 的白工，与节点数无关的准备工作没必要重复做。
        node_map = workflow.node_map()
        pred_map = self._predecessor_map(workflow)

        for node_id in order:
            # 取消：剩余节点全部 CANCELLED
            if is_cancelled and is_cancelled():
                self._mark_remaining(order, result, NodeStatus.CANCELLED)
                result.status = NodeStatus.CANCELLED
                return result

            preds = pred_map[node_id]
            pred_states = [result.node_states.get(p) for p in preds]
            # 依赖失败/跳过/取消 -> 本节点跳过
            if any(s != NodeStatus.SUCCESS.value for s in pred_states):
                # 失败节点的传递后继可能已被 _skip_descendants 预先标记过。
                # 那条日志带有「是哪个上游失败」的精确定位信息，不能被这里的
                # 泛化描述覆盖：否则每条日志后来居上，Agent 最终读到的反而是
                # 更模糊的失败原因，而且 logs 条数与节点数不再相等。
                already_skipped = (
                    result.node_states.get(node_id) == NodeStatus.SKIPPED.value
                )
                result.node_states[node_id] = NodeStatus.SKIPPED.value
                if not already_skipped:
                    result.logs.append(
                        NodeLog(
                            node_id=node_id,
                            status=NodeStatus.SKIPPED.value,
                            error="上游节点未全部成功",
                        )
                    )
                continue

            node = node_map[node_id]
            runner = node_runners[node.type]
            upstream = {p: result.outputs.get(p) for p in preds}
            log = NodeLog(node_id=node_id, status=NodeStatus.RUNNING.value)
            result.logs.append(log)
            log.started_at = time.time()
            result.node_states[node_id] = NodeStatus.RUNNING.value
            try:
                output = runner(node, upstream, ctx)
                result.outputs[node_id] = output
                result.node_states[node_id] = NodeStatus.SUCCESS.value
                log.status = NodeStatus.SUCCESS.value
            except Exception as exc:  # noqa: BLE001 - 节点失败不中断其它分支
                result.node_states[node_id] = NodeStatus.FAILED.value
                log.status = NodeStatus.FAILED.value
                log.error = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                # 传递后继全部 SKIPPED
                self._skip_descendants(node_id, successors, result)
            finally:
                log.finished_at = time.time()
                log.duration_ms = round(
                    (log.finished_at - log.started_at) * 1000, 3
                )

        failed = any(s == NodeStatus.FAILED.value for s in result.node_states.values())
        result.status = NodeStatus.FAILED if failed else NodeStatus.SUCCESS
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _topo_order(workflow: Workflow) -> list[str]:
        """Kahn 拓扑排序（校验已保证无环）。

        用 deque 而非 list：list 的 pop(0) 是 O(n) 搬移，节点多时排序本身就成 O(N²)。
        """
        indegree = {n.id: 0 for n in workflow.nodes}
        adjacency: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for nxt in sorted(adjacency[current]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    @staticmethod
    def _predecessor_map(workflow: Workflow) -> dict[str, list[str]]:
        """直接前驱表，按边声明顺序 —— 与 Workflow.predecessors 同口径。"""
        preds: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.target in preds:
                preds[edge.target].append(edge.source)
        return preds

    @staticmethod
    def _successor_map(workflow: Workflow) -> dict[str, set[str]]:
        successors: dict[str, set[str]] = {n.id: set() for n in workflow.nodes}
        for edge in workflow.edges:
            successors[edge.source].add(edge.target)
        return successors

    @staticmethod
    def _skip_descendants(
        node_id: str,
        successors: dict[str, set[str]],
        result: WorkflowRunResult,
    ) -> None:
        stack = sorted(successors.get(node_id, set()))
        while stack:
            current = stack.pop()
            if result.node_states.get(current) in (
                NodeStatus.SKIPPED.value,
                NodeStatus.FAILED.value,
            ):
                continue
            result.node_states[current] = NodeStatus.SKIPPED.value
            result.logs.append(
                NodeLog(
                    node_id=current,
                    status=NodeStatus.SKIPPED.value,
                    error=f"上游节点 {node_id!r} 失败",
                )
            )
            stack.extend(sorted(successors.get(current, set())))

    @staticmethod
    def _mark_remaining(
        order: list[str],
        result: WorkflowRunResult,
        status: NodeStatus,
    ) -> None:
        for node_id in order:
            if node_id not in result.node_states:
                result.node_states[node_id] = status.value
                result.logs.append(NodeLog(node_id=node_id, status=status.value))
