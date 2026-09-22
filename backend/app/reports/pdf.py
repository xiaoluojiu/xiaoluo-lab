"""Prompt 138：PDF 导出（fpdf2）。

- 中文必须正常显示：注册系统中文字体（SimHei / 等线 / 仿宋 / 楷体），
  可通过配置项 PDF_FONT_PATH、环境变量 XIAOLUO_PDF_FONT 或参数 font_path 显式指定 TTF
- 布局面向毕业设计与正式实验报告：标题、分节、表格、结论、页码
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import AppException
from app.reports.models import Report

# 常见中文字体候选（按优先级；TTC 集合部分版本不支持，优先 TTF）
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\Deng.ttf",  # 等线
    r"C:\Windows\Fonts\simfang.ttf",  # 仿宋
    r"C:\Windows\Fonts\simkai.ttf",  # 楷体
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


class PdfExportError(AppException):
    """PDF 导出失败（缺库 / 无中文字体）。"""

    http_status = 500
    default_code = "PDF_EXPORT_ERROR"
    default_message = "PDF export failed"


def find_chinese_font(font_path: str | None = None) -> str:
    """定位可用的中文字体文件。"""
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    if settings.PDF_FONT_PATH:
        candidates.append(settings.PDF_FONT_PATH)
    env_font = os.environ.get("XIAOLUO_PDF_FONT")
    if env_font:
        candidates.append(env_font)
    candidates.extend(FONT_CANDIDATES)
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    raise PdfExportError(
        "未找到可用中文字体，无法导出 PDF；"
        "请设置配置项 PDF_FONT_PATH 或环境变量 XIAOLUO_PDF_FONT 指向一个 .ttf 中文字体文件",
        code="PDF_FONT_NOT_FOUND",
    )


def _mc(pdf: Any, h: float, text: str, align: str | None = None) -> None:
    """multi_cell 封装：每次从左边距开始，渲染后回到左边距。

    fpdf2 新版 multi_cell 默认 new_x=RIGHT，w=0 时会把 x 停在页面右缘，
    导致下一个 multi_cell 可用宽度为 0（"Not enough horizontal space"）。
    """
    from fpdf.enums import XPos, YPos

    pdf.set_x(pdf.l_margin)
    if align:
        pdf.multi_cell(0, h, text, align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _new_pdf_class(font_file: str) -> Any:
    from fpdf import FPDF

    class ReportPDF(FPDF):  # type: ignore[misc,valid-type]
        title_text: str = ""

        def header(self) -> None:  # noqa: D102
            if self.page_no() == 1:
                return
            self.set_font("cjk", size=9)
            self.set_text_color(130, 130, 130)
            self.set_y(12)
            self.cell(0, 8, self.title_text, align="L")
            self.set_y(28)

        def footer(self) -> None:  # noqa: D102
            self.set_y(-15)
            self.set_font("cjk", size=9)
            self.set_text_color(130, 130, 130)
            self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    pdf = ReportPDF()
    pdf.add_font("cjk", "", font_file)
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=22)
    return pdf


def _fit(pdf: Any, text: str, width: float) -> str:
    """按像素宽度截断文本（用于表格单元格）。"""
    if pdf.get_string_width(text) <= width - 4:
        return text
    out = text
    while out and pdf.get_string_width(out + "…") > width - 4:
        out = out[:-1]
    return out + "…"


def _add_table(pdf: Any, table: dict[str, Any]) -> None:
    headers: list[str] = [str(h) for h in table.get("headers", [])]
    rows: list[list[str]] = [[str(c) for c in row] for row in table.get("rows", [])]
    if not headers:
        return
    if table.get("title"):
        pdf.set_font("cjk", size=10)
        _mc(pdf, 6, str(table["title"]))
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable / len(headers)
    pdf.set_font("cjk", size=9)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_fill_color(238, 243, 251)
    # 表头
    for h in headers:
        pdf.cell(col_w, 8, _fit(pdf, h, col_w), border=1, fill=True)
    pdf.ln(8)
    # 数据行
    for row in rows:
        cells = [_fit(pdf, c, col_w) for c in (list(row) + [""] * len(headers))[: len(headers)]]
        for c in cells:
            pdf.cell(col_w, 8, c, border=1)
        pdf.ln(8)
    pdf.ln(2)


def export_pdf(
    report: Report, path: str | None = None, *, font_path: str | None = None
) -> bytes | str:
    """导出 PDF。给定 path 写入文件并返回路径；否则返回字节流。"""
    try:
        import fpdf  # noqa: F401 - 探测依赖
    except ImportError as exc:
        raise PdfExportError("缺少 PDF 导出依赖 fpdf2，请先安装：pip install fpdf2") from exc

    font_file = find_chinese_font(font_path)
    pdf = _new_pdf_class(font_file)
    pdf.add_page()

    # 封面式标题区
    pdf.set_font("cjk", size=20)
    pdf.set_text_color(0, 0, 0)
    _mc(pdf, 12, report.title, align="C")
    pdf.title_text = report.title
    ds = report.dataset or {}
    pdf.set_font("cjk", size=11)
    if ds:
        _mc(
            pdf,
            8,
            f"数据集：{ds.get('name', ds.get('dataset_id', '-'))}"
            f"（{ds.get('rows', '?')} 行 × {ds.get('columns', '?')} 列）",
            align="C",
        )
    meta = report.metadata or {}
    if meta.get("generated_at"):
        _mc(pdf, 8, f"生成时间：{meta['generated_at']}", align="C")
    pdf.ln(4)
    pdf.set_draw_color(44, 90, 160)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # 章节
    for section in report.sections:
        pdf.set_font("cjk", size=14)
        _mc(pdf, 9, section.heading)
        pdf.ln(1)
        pdf.set_font("cjk", size=11)
        for line in section.content.split("\n"):
            if line.strip():
                _mc(pdf, 7, line.strip())
                pdf.ln(1)
        for table in section.tables:
            _add_table(pdf, table)
        for chart in section.charts:
            _mc(pdf, 7, f"【图表】{chart.get('title', '')}（矢量图详见 HTML / Markdown 导出）")
            pdf.ln(1)
        pdf.ln(3)

    # 关联实验
    if report.experiments:
        pdf.set_font("cjk", size=14)
        _mc(pdf, 9, "五、关联实验")
        pdf.set_font("cjk", size=11)
        for exp in report.experiments:
            exp_id = exp.get("experiment_id", "-")
            _mc(
                pdf,
                7,
                f"实验 #{exp_id}（{exp.get('task', '-')}/{exp.get('model', '-')}）",
            )
        pdf.ln(3)

    # 结论
    if report.conclusions:
        pdf.set_font("cjk", size=14)
        _mc(pdf, 9, "六、结论")
        pdf.set_font("cjk", size=11)
        for c in report.conclusions:
            _mc(pdf, 7, f"· {c}")
            pdf.ln(1)

    output = pdf.output()
    if path:
        Path(path).write_bytes(bytes(output))
        return path
    return bytes(output)
