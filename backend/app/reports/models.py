"""Prompt 134：Report 模型。

结构化报告：title / dataset / sections / experiments / charts /
conclusions / metadata。sections 为有序小节（标题 + 文本 + 表格），
导出器（Markdown / HTML / PDF）只负责渲染，不改动内容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSection:
    """报告小节。"""

    heading: str
    content: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)  # {"title","headers","rows"}
    charts: list[dict[str, Any]] = field(default_factory=list)  # {"type","title","data"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "content": self.content,
            "tables": self.tables,
            "charts": self.charts,
        }


@dataclass
class Report:
    """完整报告。"""

    title: str
    dataset: dict[str, Any] = field(default_factory=dict)
    sections: list[ReportSection] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "dataset": self.dataset,
            "sections": [s.to_dict() for s in self.sections],
            "experiments": self.experiments,
            "charts": self.charts,
            "conclusions": self.conclusions,
            "metadata": self.metadata,
        }
