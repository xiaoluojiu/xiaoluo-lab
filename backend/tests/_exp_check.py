import polars as pl
from app.analysis import DescriptiveAnalyzer
from app.reports.generator import ReportGenerator
from app.reports.html import export_html
from app.reports.markdown import export_markdown
from app.reports.report_charts import build_report_charts

df = pl.DataFrame({
    "age": [22, 25, 27, 30, 33, 35, 38, 41, 44, 47],
    "score": [10, 12, 11, 15, 18, 17, 20, 22, 21, 25],
    "city": ["BJ", "SH"] * 5,
})

charts = build_report_charts(df)
rep = ReportGenerator().generate(
    title="嵌入验证报告",
    dataset_info={"dataset_id": 1, "name": "demo", "rows": df.height, "columns": df.width, "version": 1},
    eda={"describe": {"stats": DescriptiveAnalyzer().analyze(df)["columns"]}, "correlation": {"pairs": []}},
    charts=charts,
)

html = export_html(rep)
md = export_markdown(rep)

print("charts built:", len(charts))
print("html contains figure count:", html.count('class="chart-figure"'))
print("html contains svg count:", html.count("<svg"))
print("md image count:", md.count("!["))
assert html.count('class="chart-figure"') == len(charts), "HTML 未嵌入全部图表"
assert html.count("<svg") >= len(charts)
assert md.count("![") == len(charts), "Markdown 未嵌入全部图表"
assert "chart-figure" in html
print("EXPORT OK")
