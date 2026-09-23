"""Prompt 137：HTML 导出。

独立单文件 HTML（内联样式），布局适合正式实验报告，
<meta charset=utf-8> 保证中文显示。
"""

from __future__ import annotations

import html as _html

from app.reports.models import Report, describe_experiment
from app.reports.numbering import chapter_heading, missing_chapters

_CSS = """
body { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
       max-width: 860px; margin: 24px auto; padding: 0 24px; color: #222; line-height: 1.7; }
h1 { border-bottom: 3px solid #2c5aa0; padding-bottom: 8px; }
h2 { color: #2c5aa0; margin-top: 28px; border-left: 5px solid #2c5aa0; padding-left: 10px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #eef3fb; }
blockquote { background: #f5f7fa; border-left: 4px solid #2c5aa0; margin: 0; padding: 8px 14px; }
footer { margin-top: 32px; color: #888; font-size: 0.9em;
         border-top: 1px solid #ddd; padding-top: 10px; }
.conclusion li { margin: 4px 0; }
.chart-figure { margin: 18px 0; text-align: center; }
.chart-figure figcaption { font-size: 0.92em; color: #475569; margin-bottom: 6px; }
.chart-figure svg { max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }
.missing-note { background: #fff8e6; border-left: 4px solid #d9a406; padding: 8px 14px;
                margin: 18px 0; font-size: 0.94em; }
.missing-note ul { margin: 6px 0 0; padding-left: 20px; }
"""


def _esc(value: object) -> str:
    return _html.escape(str(value))


def _render_table(table: dict) -> str:
    out = []
    if table.get("title"):
        out.append(f"<p><strong>{_esc(table['title'])}</strong></p>")
    out.append("<table><thead><tr>")
    for h in table.get("headers", []):
        out.append(f"<th>{_esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    for row in table.get("rows", []):
        out.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def export_html(report: Report) -> str:
    """渲染为完整 HTML 文档。"""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{_esc(report.title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_esc(report.title)}</h1>",
    ]
    if report.dataset:
        ds = report.dataset
        parts.append(
            f"<blockquote>数据集：{_esc(ds.get('name', ds.get('dataset_id', '-')))}"
            f"（{_esc(ds.get('rows', '?'))} 行 × {_esc(ds.get('columns', '?'))} 列）</blockquote>"
        )
    for section in report.sections:
        parts.append(f"<h2>{_esc(section.heading)}</h2>")
        if section.content:
            parts.append(
                "".join(f"<p>{_esc(line)}</p>" for line in section.content.split("\n") if line)
            )
        for table in section.tables:
            parts.append(_render_table(table))
        for chart in section.charts:
            title = _esc(chart.get("title", ""))
            svg = chart.get("svg") or ""
            parts.append(
                f'<figure class="chart-figure"><figcaption>{title}</figcaption>{svg}</figure>'
            )
    missing = missing_chapters(report)
    if missing:
        parts.append('<div class="missing-note"><strong>未生成章节说明</strong><ul>')
        for item in missing:
            parts.append(f"<li><strong>{_esc(item['heading'])}</strong>：{_esc(item['reason'])}</li>")
        parts.append("</ul></div>")
    if report.experiments:
        parts.append(f"<h2>{_esc(chapter_heading('experiments'))}</h2><ul>")
        for exp in report.experiments:
            parts.append(f"<li>{_esc(describe_experiment(exp))}</li>")
        parts.append("</ul>")
    if report.conclusions:
        parts.append(f'<h2>{_esc(chapter_heading("conclusions"))}</h2><ul class="conclusion">')
        for c in report.conclusions:
            parts.append(f"<li>{_esc(c)}</li>")
        parts.append("</ul>")
    meta = report.metadata or {}
    parts.append(
        f"<footer>生成时间：{_esc(meta.get('generated_at', '-'))} · "
        f"生成器：{_esc(meta.get('generator', 'xiaoluo-lab'))}</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)
