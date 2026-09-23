"""报告章节编号 / 未生成章节说明 / 质量章渲染的回归测试。

对应真实缺陷（来自航空公司延误数据集的 Agent 报告）：
1. 章节序号在生成器与三个导出器里各写一处，某章缺失时报告出现「一/二/三/六」
   的跳号且毫无说明；
2. 质量问题表读的是不存在的 `type` 键，「类型」列全 "-"，`column=None` 被渲染成
   字符串 "None"，严重度根本不展示，`statistics` 被「排除 dict/list」的兜底逻辑丢光；
3. 长尾列（延误分钟）被 IQR 判出 13.48% 的「异常值」，报告按 severity=high 呈现，
   读者会以为数据脏，而它其实是待预测现象本身。
"""

from __future__ import annotations

import pytest

from app.reports.generator import ReportGenerator
from app.reports.html import export_html
from app.reports.markdown import export_markdown
from app.reports.models import Report, ReportSection
from app.reports.numbering import (
    CHAPTERS,
    chapter_heading,
    chapter_status,
    cn_number,
    missing_chapters,
    refresh_chapter_status,
    strip_section_number,
)

DS = {"dataset_id": 9, "name": "航空公司出发延误预测（回归）", "rows": 10_000_000, "columns": 10, "version": 1}
EDA = {"describe": {"stats": [{"column": "DepDelay", "mean": 8.2, "std": 29.4, "min": -1197, "max": 2119}]}}


class TestNumbering:
    def test_chinese_numbers(self):
        assert [cn_number(i) for i in range(1, 7)] == ["一", "二", "三", "四", "五", "六"]
        assert cn_number(10) == "十"
        assert cn_number(11) == "十一"
        assert cn_number(21) == "二十一"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("一、数据概览", "数据概览"),
            ("1. Overview", "Overview"),
            ("（一）方法", "方法"),
            ("第一章 绪论", "绪论"),
            ("一般性说明", "一般性说明"),  # 无分隔符，不能误切
            ("2024年度报告", "2024年度报告"),
        ],
    )
    def test_strip_section_number(self, raw, expected):
        assert strip_section_number(raw) == expected

    def test_plan_is_contiguous(self):
        assert [chapter_heading(k) for k, _ in CHAPTERS] == [
            "一、数据概览",
            "二、数据质量",
            "三、探索性分析",
            "四、建模与评估",
            "五、关联实验",
            "六、结论",
        ]


class TestMissingChapterNote:
    def _report_without_ml(self) -> Report:
        return ReportGenerator().generate(
            dataset_info=DS,
            quality={"issues": [], "statistics": {"row_count": 10}},
            eda=EDA,
        )

    def test_status_marks_missing_chapters(self):
        report = self._report_without_ml()
        status = {c["key"]: c for c in chapter_status(report)}
        assert status["ml"]["generated"] is False
        assert status["experiments"]["generated"] is False
        # 原因必须可操作，不能是「无数据」这类废话
        assert "ml.train" in status["ml"]["reason"]
        assert status["ml"]["heading"] == "四、建模与评估"

    def test_markdown_explains_the_gap(self):
        md = export_markdown(self._report_without_ml())
        # 序号仍然按固定计划表排（结论永远是「六、」），但缺的章必须有说明
        assert "## 一、数据概览" in md
        assert "## 三、探索性分析" in md
        assert "## 六、结论" in md
        assert "未生成章节说明" in md
        assert "**四、建模与评估**" in md
        assert "**五、关联实验**" in md

    def test_html_and_pdf_explain_the_gap(self):
        report = self._report_without_ml()
        html = export_html(report)
        assert 'class="missing-note"' in html
        assert "四、建模与评估" in html

        from app.reports.pdf import export_pdf, find_chinese_font

        try:
            font = find_chinese_font(None)
        except Exception:  # noqa: BLE001 - 本机无中文字体时跳过
            pytest.skip("本机没有可用的中文字体 TTF")
        assert export_pdf(report, font_path=font)[:4] == b"%PDF"

    def test_no_note_when_everything_generated(self):
        report = ReportGenerator().generate(
            dataset_info=DS,
            quality={"issues": []},
            eda=EDA,
            ml={"task": "regression", "model": "linear_regression", "target_column": "DepDelay", "metrics": {"r2": 0.71}},
            experiments=[{"experiment_id": 1, "run_id": 2, "task": "regression", "model": "linear_regression"}],
        )
        assert missing_chapters(report) == []
        md = export_markdown(report)
        assert "未生成章节说明" not in md
        assert "## 四、建模与评估" in md
        assert "## 五、关联实验" in md

    def test_narration_filling_a_chapter_syncs_status(self):
        """LLM 叙述补齐了原本缺失的章节后，「该章未生成」说明必须消失（否则自相矛盾）。"""
        from app.reports.narrator import apply_narration

        payload = self._report_without_ml().to_dict()
        apply_narration(
            payload,
            {
                "executive_summary": "综述。",
                "sections": {"四、建模与评估": "模型补写的建模正文。"},
                "conclusions": ["c1", "c2", "c3", "c4"],
                "recommendations": ["r1", "r2", "r3"],
            },
        )
        assert "四、建模与评估" in [s["heading"] for s in payload["sections"]]
        assert "四、建模与评估" not in [c["heading"] for c in missing_chapters(payload)]
        report = Report(
            title=payload["title"],
            dataset=payload["dataset"],
            sections=[ReportSection(**s) for s in payload["sections"]],
            experiments=payload.get("experiments") or [],
            conclusions=payload.get("conclusions") or [],
            metadata=payload.get("metadata") or {},
        )
        md = export_markdown(report)
        assert "**四、建模与评估**" not in md
        assert "**五、关联实验**" in md

    def test_refresh_never_downgrades_generated(self):
        payload = {"sections": [], "metadata": {"chapters": [{"key": "eda", "generated": True}]}}
        plan = {c["key"]: c for c in refresh_chapter_status(payload)}
        assert plan["eda"]["generated"] is True


class TestQualitySection:
    def _quality(self) -> dict:
        return {
            "issues": [
                {"check": "duplicate", "severity": "low", "message": "按 整行 检测到 3799 行重复 (0.04%)", "column": None, "details": {}},
                {
                    "check": "outlier",
                    "severity": "high",
                    "message": "字段 'DepDelay' 检测到 1347955 个异常值 (13.48%)，方法=iqr，边界=[-16.5, 19.5]",
                    "column": "DepDelay",
                    "details": {"outlier_ratio": 0.134795, "lower": -16.5, "upper": 19.5},
                },
            ],
            "severity": {"info": 0, "low": 1, "medium": 0, "high": 1, "critical": 0},
            "statistics": {
                "row_count": 10_000_000,
                "column_count": 10,
                "checked_by": ["missing", "duplicate", "outlier"],
                "missing_cells": 0,
                "duplicate_rows": 3799,
                "issue_count": 2,
            },
            "recommendations": ["存在重复数据，可使用 duplicate 操作处理。"],
            "has_errors": True,
        }

    def _section(self, target_column=None):
        report = ReportGenerator().generate(
            dataset_info=DS,
            quality=self._quality(),
            eda=EDA,
            ml=(
                {"task": "regression", "model": "linear_regression", "target_column": target_column, "metrics": {"r2": 0.7}}
                if target_column
                else None
            ),
        )
        return next(s for s in report.sections if s.heading.endswith("数据质量"))

    def test_issue_table_columns_align(self):
        section = self._section()
        table = next(t for t in section.tables if t["title"] == "质量问题明细")
        assert table["headers"] == ["位置", "类型", "严重度", "说明"]
        assert table["rows"][0][0] == "全表"  # column=None 不能渲染成 "None"
        assert table["rows"][0][1] == "重复行"  # 读的是 check，不是不存在的 type
        assert table["rows"][0][2] == "低"
        assert table["rows"][1][2] == "高"

    def test_statistics_are_not_dropped(self):
        section = self._section()
        table = next(t for t in section.tables if t["title"] == "质量统计口径")
        labels = [r[0] for r in table["rows"]]
        assert "总行数" in labels and "缺失单元格" in labels and "已执行检查" in labels
        values = dict(table["rows"])
        assert values["已执行检查"] == "missing、duplicate、outlier"
        assert values["其中需要建模前处理的问题数"].startswith("0")
        # 内部标志位不再混进「其他质量指标」
        assert all("其他质量指标" != t["title"] for t in section.tables)

    def test_skewed_outlier_is_explained_not_alarmist(self):
        section = self._section()
        assert "IQR 边界在此不适用" in section.content
        assert "不等于录入错误" in section.content

    def test_target_column_outlier_gets_extra_warning(self):
        section = self._section(target_column="DepDelay")
        assert "正是本次建模的目标列（DepDelay）" in section.content
        assert "不应作为异常值删除" in section.content

    def test_auto_conclusion_does_not_demand_cleaning(self):
        report = ReportGenerator().generate(dataset_info=DS, quality=self._quality(), eda=EDA)
        joined = " ".join(report.conclusions)
        assert "均为长尾列的 IQR 离群判定" in joined
        assert "不要删除本身就是待预测现象的尾部记录" in joined


class TestExperimentLine:
    """关联实验必须写出状态与失败原因。

    只列 id/task/model 会让读者以为「有关联实验 = 建模成功」，
    而事故里 dataset 9 的 5 次运行只有 1 次回归成功。
    """

    def test_failed_run_shows_status_and_reason(self):
        report = ReportGenerator().generate(
            dataset_info=DS,
            quality={"issues": []},
            eda=EDA,
            experiments=[
                {
                    "experiment_id": 29,
                    "run_id": 29,
                    "task": "regression",
                    "model": "linear_regression",
                    "target_column": "DepDelay",
                    "status": "failed",
                    "error": "target_column 不能出现在 excluded_columns 中",
                },
                {
                    "experiment_id": 31,
                    "run_id": 31,
                    "task": "regression",
                    "model": "linear_regression",
                    "target_column": "DepDelay",
                    "status": "success",
                },
            ],
        )
        md = export_markdown(report)
        assert "状态 failed" in md
        assert "target_column 不能出现在 excluded_columns 中" in md
        assert "目标列 DepDelay" in md
        html = export_html(report)
        assert "状态 failed" in html


class TestChartPointCompaction:
    """点集必须压成单条 `<path>`，且点数超限时抽样并如实标注。

    事故报告里两张 Q-Q 图各 5000 点、逐点一个 `<circle>`（约 74 字符/点），
    光点集 716 KB，占全部 SVG 的 87%，整份 Markdown 被撑到 1.36 MB。
    """

    @staticmethod
    def _series(n: int):
        return [float(i) for i in range(n)], [float((i * 7919) % 1000) for i in range(n)]

    def test_scatter_is_one_path_without_circles(self):
        from app.reports.chart_svg import to_svg

        x, y = self._series(5000)
        svg = to_svg({"chart": "scatter", "x": x, "y": y})
        assert "<circle" not in svg
        assert svg.count("<path") == 1

    def test_qq_is_one_path_without_circles(self):
        from app.reports.chart_svg import to_svg

        x, y = self._series(5000)
        svg = to_svg({"chart": "qq", "x": x, "y": y, "line": {"x0": 0, "y0": 0, "x1": 4999, "y1": 999}})
        assert "<circle" not in svg
        assert svg.count("<path") == 1

    def test_downsampling_is_disclosed(self):
        from app.reports.chart_svg import _MAX_POINTS, to_svg

        x, y = self._series(_MAX_POINTS * 2)
        svg = to_svg({"chart": "scatter", "x": x, "y": y})
        assert "等距抽样显示" in svg  # 抽样必须写在图上，不能静默丢点

    def test_small_point_set_is_kept_whole(self):
        from app.reports.chart_svg import to_svg

        x, y = self._series(100)
        svg = to_svg({"chart": "scatter", "x": x, "y": y})
        assert "等距抽样显示" not in svg
        assert len(svg) < 20_000
