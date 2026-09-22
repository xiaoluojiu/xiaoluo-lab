import polars as pl
from app.analysis import VisualizationBuilder, DescriptiveAnalyzer, CorrelationAnalyzer
from app.reports.chart_svg import to_svg
from app.reports.report_charts import build_report_charts
from app.reports.generator import ReportGenerator

df = pl.DataFrame({
    "age": [22, 25, 27, 30, 33, 35, 38, 41, 44, 47, 50, 53, 28, 31, 36],
    "score": [10, 12, 11, 15, 18, 17, 20, 22, 21, 25, 24, 28, 13, 16, 19],
    "group": ["A", "B"] * 7 + ["A"],
    "city": ["BJ", "SH", "BJ", "SH", "BJ", "SH", "BJ", "SH", "BJ", "SH", "BJ", "SH", "GZ", "GZ", "GZ"],
})

vb = VisualizationBuilder()
for t, kw in [
    ("histogram", {"column": "age", "bins": 8}),
    ("bar", {"column": "city", "top_n": 8}),
    ("line", {"x": "age", "y": "score"}),
    ("scatter", {"x": "age", "y": "score"}),
    ("boxplot", {"column": "score"}),
    ("heatmap", {"columns": ["age", "score"]}),
    ("qq", {"column": "age"}),
    ("grouped_bar", {"column": "city", "y": "score", "group_by": "group"}),
    ("area", {"column": "age", "bins": 8}),
]:
    out = vb.analyze(df, chart=t, **kw)
    svg = to_svg(out)
    assert svg.startswith("<svg"), f"{t} svg broken"
    assert "NaN" not in svg and "None" not in svg, f"{t} svg has NaN/None"
    print(f"OK {t:12s} svg_len={len(svg)}")

charts = build_report_charts(df)
print("report charts:", [(c["type"], c["title"]) for c in charts])
assert len(charts) >= 5
assert all(c["svg"].startswith("<svg") for c in charts)

# 生成报告并确认图表进入 EDA 小节
rep = ReportGenerator().generate(
    title="验证报告",
    dataset_info={"dataset_id": 1, "name": "demo", "rows": df.height, "columns": df.width, "version": 1},
    quality=None,
    eda={"describe": {"stats": DescriptiveAnalyzer().analyze(df)["columns"]},
         "correlation": {"pairs": []}},
    charts=charts,
)
eda_section = rep.sections[1]
print("eda section charts:", len(eda_section.charts))
assert len(eda_section.charts) == len(charts)
print("ALL GOOD")
