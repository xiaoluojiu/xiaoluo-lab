"""Prompt 083：Workflow 数据结构。

Node(id/type/config)、Edge(source/target)、Workflow(nodes/edges/metadata)。
纯数据结构（dataclass），支持 to_dict / from_dict 序列化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """工作流节点：id 唯一，type 决定执行器，config 为该节点参数。"""

    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "config": dict(self.config)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Node:
        return Node(
            id=str(data["id"]),
            type=str(data["type"]),
            config=dict(data.get("config") or {}),
        )


@dataclass
class Edge:
    """有向边：source -> target（输出从 source 流入 target 的输入）。"""

    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Edge:
        return Edge(source=str(data["source"]), target=str(data["target"]))


def flatten_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """把节点 config 归一化成「平铺参数视图」。

    平台里 config 有并存的两套写法：平铺（``{"dataset_id": 1}``，前端画布的参数
    JSON 面板、``data.load`` 走这套）与嵌套（``{"params": {...}}``，``ml.*`` 与
    ``data.*`` 操作节点走这套）。做「必需参数是否齐全」这类结构判断时必须两种都认，
    否则会把写着 ``params`` 的合法配置误报为缺失；``__ui`` 只是画布坐标，一并剔除。
    """
    if not isinstance(config, dict):
        return {}
    params = config.get("params")
    if isinstance(params, dict):
        merged = {
            k: v for k, v in config.items() if k not in {"params", "__ui"}
        }
        merged.update(params)
        return merged
    return {k: v for k, v in config.items() if k != "__ui"}


@dataclass
class Workflow:
    """DAG 工作流：节点 + 边 + 元信息。"""

    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def predecessors(self, node_id: str) -> list[str]:
        """直接前驱节点（按边声明顺序）。"""
        return [e.source for e in self.edges if e.target == node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Workflow:
        return Workflow(
            name=str(data["name"]),
            nodes=[Node.from_dict(n) for n in data.get("nodes") or []],
            edges=[Edge.from_dict(e) for e in data.get("edges") or []],
            metadata=dict(data.get("metadata") or {}),
        )
