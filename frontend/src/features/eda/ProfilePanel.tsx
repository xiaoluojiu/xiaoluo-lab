import { CHART_SERIES, CHART_GRID } from "../../lib/chartColors";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface DescriptiveData {
  row_count: number;
  columns: {
    column: string;
    dtype: string;
    count: number;
    missing_count: number;
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
    quantiles?: Record<string, number>;
  }[];
}

// Prompt 167 + 增强：描述性统计面板 —— 缺失值条形图 + 数值分布摘要 + 原表格。
export function ProfilePanel({ data }: { data: DescriptiveData | null }) {
  if (!data) return <div className="muted">请先运行描述性统计</div>;

  const missingChart = data.columns.map((c) => ({
    name: c.column,
    缺失数: c.missing_count,
  }));

  const numericCols = data.columns.filter(
    (c) => typeof c.mean === "number" && typeof c.std === "number",
  );

  const numericSummaryChart = numericCols.map((c) => ({
    name: c.column,
    均值: round2(c.mean),
    标准差: round2(c.std),
  }));

  return (
    <div>
      <p className="muted">共 {data.row_count} 行 · {data.columns.length} 列</p>

      {/* 缺失值条形图 */}
      <details open style={{ marginBottom: "var(--space-4)" }}>
        <summary style={{ cursor: "pointer", fontWeight: 500 }}>
          缺失值统计（{missingChart.filter((r) => r["缺失数"] > 0).length} 列有缺失）
        </summary>
        <div style={{ marginTop: "var(--space-2)" }}>
          {missingChart.some((r) => r["缺失数"] > 0) ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={missingChart}
                margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis dataKey="name" fontSize={11} interval={0} angle={-25} height={60} textAnchor="end" />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="缺失数" fill={CHART_SERIES[3]} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="muted">所有列均无缺失值</div>
          )}
        </div>
      </details>

      {/* 数值列分布摘要 */}
      {numericCols.length > 0 && (
        <details open style={{ marginBottom: "var(--space-4)" }}>
          <summary style={{ cursor: "pointer", fontWeight: 500 }}>
            数值列分布摘要（均值 ± 标准差）
          </summary>
          <div style={{ marginTop: "var(--space-2)" }}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={numericSummaryChart}
                margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis dataKey="name" fontSize={11} interval={0} angle={-25} height={60} textAnchor="end" />
                <YAxis fontSize={11} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="均值" fill={CHART_SERIES[0]} radius={[3, 3, 0, 0]} />
                <Bar dataKey="标准差" fill={CHART_SERIES[2]} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </details>
      )}

      {/* 原表格 */}
      <details>
        <summary style={{ cursor: "pointer", fontWeight: 500 }}>详细统计表</summary>
        <div style={{ marginTop: "var(--space-2)", overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>列</th>
                <th>类型</th>
                <th>缺失</th>
                <th>均值</th>
                <th>标准差</th>
                <th>最小</th>
                <th>最大</th>
              </tr>
            </thead>
            <tbody>
              {data.columns.map((c) => (
                <tr key={c.column}>
                  <td>{c.column}</td>
                  <td>{c.dtype}</td>
                  <td>{c.missing_count}</td>
                  <td>{fmt(c.mean)}</td>
                  <td>{fmt(c.std)}</td>
                  <td>{fmt(c.min)}</td>
                  <td>{fmt(c.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "-";
  return typeof v === "number" ? String(Math.round(v * 1000) / 1000) : String(v);
}

function round2(v: number | undefined): number {
  if (v === undefined) return 0;
  return Math.round(v * 100) / 100;
}
