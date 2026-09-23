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


def describe_experiment(exp: dict[str, Any]) -> str:
    """一行描述一次实验（三个导出器共用，转义交给各渲染器）。

    **状态与失败原因必须写出来**：只报 id/task/model 会让读者以为
    「列了关联实验 = 建模成功」，而真实情况常常是几次失败 + 一次成功 ——
    事故里 dataset 9 的 5 次运行只有 1 次是回归成功，另外 3 次是
    「目标列不能出现在排除列中」的即时失败、1 次是 57.1 GiB 崩溃。
    """
    line = (
        f"实验 #{exp.get('experiment_id', '-')}"
        f"（{exp.get('task', '-')}/{exp.get('model', '-')}）"
    )
    if exp.get("target_column"):
        line += f"，目标列 {exp['target_column']}"
    if exp.get("status"):
        line += f"，状态 {exp['status']}"
    if exp.get("error"):
        line += f"：{str(exp['error'])[:200]}"
    return line
