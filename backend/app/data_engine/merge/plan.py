"""MergePlan（Prompt 057）。

描述一次合并的完整计划：数据集、映射、Key、连接类型、冲突与警告。
计划与执行分离：MergePlan 只描述意图，由 Validator / Executor 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JOIN_TYPES = ("inner", "left", "right", "outer")


@dataclass
class ColumnMapping:
    """右表列 -> 输出列的映射（重命名引入）。"""

    right_column: str
    output_column: str


@dataclass
class JoinKey:
    """一对 Join Key。"""

    left: str
    right: str


@dataclass
class MergePlan:
    """合并计划。"""

    # 数据集描述（{"dataset_id": 1, "version": 2} 或 {"name": "left"}）
    left: dict[str, Any] = field(default_factory=dict)
    right: dict[str, Any] = field(default_factory=dict)
    keys: list[JoinKey] = field(default_factory=list)
    mapping: list[ColumnMapping] = field(default_factory=list)
    join_type: str = "inner"
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # 冲突列后缀（除 Key 外同名列的处理方式）
    left_suffix: str = "_left"
    right_suffix: str = "_right"
    # 只引入右表这些列（None = 全部非 Key 列）
    right_columns: list[str] | None = None

    def left_keys(self) -> list[str]:
        return [k.left for k in self.keys]

    def right_keys(self) -> list[str]:
        return [k.right for k in self.keys]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "keys": [{"left": k.left, "right": k.right} for k in self.keys],
            "mapping": [
                {"right_column": m.right_column, "output_column": m.output_column}
                for m in self.mapping
            ],
            "join_type": self.join_type,
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
            "left_suffix": self.left_suffix,
            "right_suffix": self.right_suffix,
            "right_columns": self.right_columns,
        }
