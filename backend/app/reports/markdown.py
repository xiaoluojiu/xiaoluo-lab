"""Prompt 136：Markdown 导出。"""

from __future__ import annotations
from urllib.parse import quote

from app.reports.models import Report, describe_experiment
from app.reports.numbering import chapter_heading, missing_chapters


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _missing_block(report: Report) -> list[str]:
    """未生成章节的显式说明。

    章节号按固定计划表排定，某章缺失时序号不会顺延 —— 但必须告诉读者
    「这一章为什么没有」，否则「一 / 二 / 三 / 六」会被当成编号错误。
    """
    missing = missing_chapters(report)
    if not missing:
        return []
    out = ["---", "", "**未生成章节说明**", ""]
    for item in missing:
        out.append(f"- **{item['heading']}**：{item['reason']}")
    out.append("")
    return out


def export_markdown(report: Report) -> str:
    """渲染为 Markdown 文本。"""
    parts: list[str] = [f"# {report.title}", ""]
    if report.dataset:
        ds = report.dataset
        parts.append(f"> 数据集：{ds.get('name', ds.get('dataset_id', '-'))}")
        parts.append("")
    for section in report.sections:
        parts.append(f"## {section.heading}")
        parts.append("")
        if section.content:
            parts.append(section.content)
            parts.append("")
        for table in section.tables:
            if table.get("title"):
                parts.append(f"**{table['title']}**")
                parts.append("")
            parts.append(_table(table.get("headers", []), table.get("rows", [])))
            parts.append("")
        for chart in section.charts:
            svg = chart.get("svg") or ""
            uri = "data:image/svg+xml;utf8," + quote(svg)
            title = chart.get("title", "")
            parts.append(f"![{title}]({uri})")
            parts.append(f"*图：{title}*")
            parts.append("")
    parts.extend(_missing_block(report))
    if report.experiments:
        parts.append(f"## {chapter_heading('experiments')}")
        parts.append("")
        for exp in report.experiments:
            parts.append(f"- {describe_experiment(exp)}")
        parts.append("")
    if report.conclusions:
        parts.append(f"## {chapter_heading('conclusions')}")
        parts.append("")
        for c in report.conclusions:
            parts.append(f"- {c}")
        parts.append("")
    meta = report.metadata or {}
    if meta:
        parts.append("---")
        generated = meta.get("generated_at", "-")
        generator = meta.get("generator", "xiaoluo-lab")
        parts.append(f"*生成时间：{generated} · 生成器：{generator}*")
    return "\n".join(parts).rstrip() + "\n"
