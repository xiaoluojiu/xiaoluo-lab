import type { Report, ReportChart } from "../../types/report";
import "./ReportPreview.css";

// Prompt 193：报告预览 —— 标题 / 章节 / 表格 / 图表 / 结论。
//
// 历史缺陷：报告里的图表一直不显示。后端其实已经把图表渲染成内联 SVG 存进报告，
// 但这里只渲染了 content / tables，把 sections[].charts 整段丢掉了。
// 现在按「章节内图表优先、report.charts 兜底」的方式渲染 SVG。
function ChartFigure({ chart, index }: { chart: ReportChart; index: number }) {
  const svg = typeof chart?.svg === "string" ? chart.svg.trim() : "";
  if (!svg) return null;
  return (
    <figure className="report-chart-figure" key={index}>
      <figcaption>{chart.title || "图表"}</figcaption>
      <div
        className="report-chart-svg"
        // SVG 由后端 chart_svg.py 渲染并转义，作为静态矢量图插入
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </figure>
  );
}

export function ReportPreview({ report }: { report: Report | null }) {
  if (!report) return <div className="muted">生成报告后在此预览</div>;
  const globalCharts = (report.charts || []).filter((c) => c && typeof c.svg === "string" && c.svg.trim());
  // 只要没有任何章节挂上图表，就在末尾兜底渲染一份图表集：
  // 宁可多一个图集区块，也不能让 Agent 产出的图「算生成了但页面上找不到」。
  let anyChartRendered = false;
  const body = (report.sections || []).map((sec, i) => {
    const sectionCharts = (sec.charts || []).filter((c) => c && typeof c.svg === "string" && c.svg.trim());
    // 章节没有自带图表时，用报告级 charts 兜底（挂在「探索性分析」等分析章节）
    const fallback =
      sectionCharts.length === 0 && /探索|分析|EDA|分布|图表/i.test(sec.heading || "") ? globalCharts : [];
    const charts = sectionCharts.length > 0 ? sectionCharts : fallback;
    if (charts.length > 0) anyChartRendered = true;
    return (
      <div key={i} className="mt">
        <h3>{sec.heading}</h3>
        {sec.content && <div style={{ whiteSpace: "pre-wrap", fontSize: 14 }}>{sec.content}</div>}
        {charts.map((c, j) => (
          <ChartFigure chart={c} index={j} key={`${i}-${j}`} />
        ))}
        {sec.tables?.map((t, j) => (
          <div key={j} className="mt">
            <h4>{t.title}</h4>
            <table className="data-table">
              <thead>
                <tr>
                  {t.headers.map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {t.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c}>{cell == null ? "-" : String(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    );
  });

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>{report.title}</h2>
      {body}
      {globalCharts.length > 0 && !anyChartRendered && (
        <div className="mt">
          <h3>图表集</h3>
          {globalCharts.map((c, j) => (
            <ChartFigure chart={c} index={j} key={`gallery-${j}`} />
          ))}
        </div>
      )}
      {report.experiments?.length > 0 && (
        <div className="mt">
          <h3>相关实验</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>实验</th>
                <th>模型</th>
                <th>指标</th>
              </tr>
            </thead>
            <tbody>
              {report.experiments.map((e, i) => (
                <tr key={i}>
                  <td>{String(e.name ?? e.id ?? "-")}</td>
                  <td>{String(e.model ?? "-")}</td>
                  <td>{e.metrics != null ? JSON.stringify(e.metrics) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {report.conclusions?.length > 0 && (
        <div className="mt">
          <h3>结论</h3>
          <ul>
            {report.conclusions.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
