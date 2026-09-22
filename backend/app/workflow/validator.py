"""Prompt 084：Workflow Validator。

检查：node 存在 / edge 合法 / 无循环 / 参数合法 / 依赖完整（节点类型已注册）。
输出结构化 WorkflowValidationResult（ok + errors + warnings）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.workflow.models import Workflow, flatten_config


@dataclass
class WorkflowValidationResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _has_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> list[str] | None:
    """Kahn 拓扑排序；返回成环的节点列表（无环返回 None）。

    引用了不存在节点的边不参与环检测（该错误已由边校验报告）。
    """
    node_set = set(nodes)
    indegree = {n: 0 for n in nodes}
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src not in node_set or dst not in node_set:
            continue
        adjacency[src].append(dst)
        indegree[dst] += 1
    queue = [n for n, d in indegree.items() if d == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited == len(nodes):
        return None
    return [n for n, d in indegree.items() if d > 0]


def validate_workflow(
    workflow: Workflow,
    *,
    known_types: set[str] | None = None,
    required_config_keys: dict[str, set[str]] | None = None,
) -> WorkflowValidationResult:
    """校验工作流结构。

    known_types: 已注册的节点类型集合；提供时未知类型视为错误（依赖完整）。
    required_config_keys: {type: 必需 config 键集合}；提供时做参数合法性检查。
    """
    result = WorkflowValidationResult()

    # 1. 节点本身合法
    node_ids: list[str] = []
    for node in workflow.nodes:
        if not node.id:
            result.error("存在 id 为空的节点")
            continue
        if node.id in node_ids:
            result.error(f"节点 id 重复：{node.id!r}")
            continue
        node_ids.append(node.id)
        if not node.type:
            result.error(f"节点 {node.id!r} 缺少 type")
        if not isinstance(node.config, dict):
            result.error(f"节点 {node.id!r} 的 config 必须是对象")

    # 2. 边合法：端点必须存在；自环报错；重复边警告
    seen_edges: set[tuple[str, str]] = set()
    edge_pairs: list[tuple[str, str]] = []
    id_set = set(node_ids)
    for edge in workflow.edges:
        for endpoint, role in ((edge.source, "source"), (edge.target, "target")):
            if endpoint not in id_set:
                result.error(f"边的 {role} 引用了不存在的节点：{endpoint!r}")
        if edge.source == edge.target and edge.source in id_set:
            result.error(f"存在自环边：{edge.source!r}")
        pair = (edge.source, edge.target)
        if pair in seen_edges:
            result.warn(f"重复边：{edge.source!r} -> {edge.target!r}")
        seen_edges.add(pair)
        edge_pairs.append(pair)

    # 3. 无循环
    if node_ids:
        cycle = _has_cycle(node_ids, edge_pairs)
        if cycle:
            result.error(f"工作流存在循环依赖，涉及节点：{sorted(cycle)}")

    # 4. 参数合法：按类型检查必需 config 键
    #    必须走 flatten_config：config 可能是 {params: {...}} 嵌套写法，
    #    用顶层键直接查会把合法配置误报为缺失。显式 null 与键缺失同样算缺失
    #    （与前端「流程体检」的 `config?.dataset_id == null` 口径一致）。
    specs = required_config_keys or {}
    for node in workflow.nodes:
        required = specs.get(node.type)
        if not required:
            continue
        view = flatten_config(node.config)
        missing = sorted(k for k in required if view.get(k) is None)
        if missing:
            result.error(
                f"节点 {node.id!r}（type={node.type!r}）缺少必需参数：{missing}"
            )

    # 5. 依赖完整：节点类型必须已注册
    if known_types is not None:
        for node in workflow.nodes:
            if node.type and node.type not in known_types:
                result.error(
                    f"节点 {node.id!r} 使用了未注册的节点类型：{node.type!r}"
                )

    return result
