"""MergeReport（Prompt 060）。

至少记录：input rows / output rows / matched rows / unmatched rows /
duplicate keys / conflicts / warnings。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MergeReport:
    """合并执行报告。"""

    join_type: str = "inner"
    input_rows_left: int = 0
    input_rows_right: int = 0
    output_rows: int = 0
    output_columns: int = 0
    matched_rows: int = 0
    unmatched_rows_left: int = 0
    unmatched_rows_right: int = 0
    duplicate_keys_left: int = 0
    duplicate_keys_right: int = 0
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_type": self.join_type,
            "input_rows_left": self.input_rows_left,
            "input_rows_right": self.input_rows_right,
            "output_rows": self.output_rows,
            "output_columns": self.output_columns,
            "matched_rows": self.matched_rows,
            "unmatched_rows_left": self.unmatched_rows_left,
            "unmatched_rows_right": self.unmatched_rows_right,
            "duplicate_keys_left": self.duplicate_keys_left,
            "duplicate_keys_right": self.duplicate_keys_right,
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
        }
