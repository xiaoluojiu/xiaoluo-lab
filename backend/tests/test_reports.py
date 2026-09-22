"""Phase 7（Prompt 134-138）：Report Engine 测试。"""

import polars as pl
import pytest
from app.core.exceptions import AppException
from app.reports.generator import ReportGenerator
from app.reports.html import export_html
from app.reports.markdown import export_markdown
from app.reports.models import Report, ReportSection
from app.reports.pdf import export_pdf, find_chinese_font
from app.reports.report_charts import build_report_charts


@pytest.fixture()
def report() -> Report:
    generator = ReportGenerator()
    return generator.generate(
        title="鸢尾花分类实验报告",
        dataset_info={"dataset_id": 1, "name": "iris", "rows": 150, "columns": 5, "version": 1},
        quality={
            "score": 88,
            "issues": [{"column": "sepal_len", "type": "missing", "message": "存在 2 个缺失值"}],
        },
        eda={
            "describe": {
                "stats": [
                    {"column": "sepal_len", "mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}
                ]
            },
            "correlation": {"top_pairs": [{"a": "x1", "b": "x2", "correlation": 0.91}]},
        },
        ml={
            "task": "classification",
            "model": "logistic_regression",
            "metrics": {"accuracy": 0.93, "f1": 0.92},
        },
        experiments=[
            {"experiment_id": 7, "task": "classification", "model": "logistic_regression"}
        ],
        conclusions=["逻辑回归在该任务上可作为基线。"],
    )


class TestReportGenerator:
    def test_structure(self, report):
        assert report.title == "鸢尾花分类实验报告"
        headings = [s.heading for s in report.sections]
        assert any("数据概览" in h for h in headings)
        assert any("数据质量" in h for h in headings)
        assert any("探索性分析" in h for h in headings)
        assert any("建模与评估" in h for h in headings)
        assert report.experiments[0]["experiment_id"] == 7
        assert report.conclusions  # 自动结论已追加
        assert report.metadata["generator"]

    def test_empty_report(self):
        empty = ReportGenerator().generate()
        assert empty.sections and empty.conclusions


class TestMarkdownExport:
    def test_export(self, report):
        text = export_markdown(report)
        assert text.startswith("# 鸢尾花分类实验报告")
        assert "## 一、数据概览" in text
        assert "| 指标 | 值 |" in text
        assert "accuracy" in text
        assert "## 六、结论" in text


class TestHtmlExport:
    def test_export(self, report):
        html = export_html(report)
        assert '<meta charset="utf-8">' in html
        assert "鸢尾花分类实验报告" in html
        assert "<table>" in html
        assert "结论" in html


class TestPdfExport:
    def test_chinese_pdf(self, report, tmp_path):
        try:
            font = find_chinese_font(None)
        except AppException:
            pytest.skip("本机没有可用的中文字体 TTF")
        out_path = str(tmp_path / "report.pdf")
        result = export_pdf(report, out_path, font_path=font)
        assert result == out_path
        data = open(out_path, "rb").read()
        assert data[:4] == b"%PDF"
        # 多页（含页码 footer）
        assert b"/Page" in data

    def test_missing_font_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XIAOLUO_PDF_FONT", str(tmp_path / "nope.ttf"))
        from app.reports import pdf as pdf_mod

        original = pdf_mod.FONT_CANDIDATES
        pdf_mod.FONT_CANDIDATES = ()
        try:
            with pytest.raises(AppException):
                find_chinese_font(None)
        finally:
            pdf_mod.FONT_CANDIDATES = original

    def test_report_section_model(self):
        section = ReportSection(heading="h", content="c")
        assert section.to_dict()["heading"] == "h"


class TestReportChartsFeatureAware:
    """图表策划必须随数据集特征变化，不再千篇一律（回归）。

    历史缺陷：build_report_charts 写死「直方图×2 + 热力图 + 散点 + 柱状 + Q-Q + CDF」，
    时间序列数据拿不到折线图、有分组列的数据拿不到分组柱状图，line/boxplot/grouped_bar
    三个已实现的类型形同虚设。
    """

    def _time_series_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 31), interval="1d", eager=True),
            "sales": [float(i % 7 + 3) for i in range(31)],
            "customers": [float(i % 5 + 2) for i in range(31)],
            "region": (["华东", "华南", "华北"] * 11)[:31],
            "channel": (["线上", "线下"] * 16)[:31],
        })

    def _numeric_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "a": [float(i % 11) for i in range(100)],
            "b": [float(i % 13) for i in range(100)],
            "c": [float(i % 7) for i in range(100)],
        })

    def test_time_series_gets_line_and_grouped_bar(self):
        charts = build_report_charts(self._time_series_df())
        types = [c["type"] for c in charts]
        # 必保三件套始终存在
        for req in ("histogram", "heatmap", "scatter"):
            assert req in types
        # 时间列 → 折线图；两个低基数分类列 + 数值列 → 分组柱状图
        assert "line" in types
        assert "grouped_bar" in types
        assert "boxplot" in types

    def test_pure_numeric_skips_line_and_grouped_bar(self):
        charts = build_report_charts(self._numeric_df())
        types = [c["type"] for c in charts]
        assert "line" not in types
        assert "grouped_bar" not in types
        assert "histogram" in types and "heatmap" in types and "scatter" in types

    def test_charts_differ_between_datasets(self):
        a = [c["type"] for c in build_report_charts(self._time_series_df())]
        b = [c["type"] for c in build_report_charts(self._numeric_df())]
        assert a != b
