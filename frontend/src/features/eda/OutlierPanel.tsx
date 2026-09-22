import { CHART_SERIES, CHART_GRID } from "../../lib/chartColors";
import {
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface OutlierData {
  method: string;
  columns: {
    column: string;
    status: string;
    reason?: string;
    outlier_count?: number;
    outlier_ratio?: number;
    sample_outliers?: number[];
    bounds?: { lower: number; upper: number };
  }[];
}

// Prompt 170 + 增强：异常值面板 —— 表格 + IQR 范围/散点图。
export function OutlierPanel({ data }: { data: OutlierData | null }) {
  if (!data) return <div className="muted">请先运行异常值分析</div>;

  const hasOutliers = data.columns.some(
    (c) => (c.outlier_count ?? 0) > 0,
  );

  const outlierCols = data.columns.filter((c) => (c.outlier_count ?? 0) > 0);

  return (
    <div>
      {hasOutliers && (
        <details open style={{ marginBottom: "var(--space-4)" }}>
          <summary style={{ cursor: "pointer", fontWeight: 500 }}>
            异常值可视化（{outlierCols.length} 列检出异常）
          </summary>
          <div style={{ marginTop: "var(--space-2)", display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            {outlierCols.map((col) => {
              if (!col.bounds) {
                return (
                  <div key={col.column} className="muted">
                    {col.column}：有 {col.outlier_count} 个异常，但无边界数据
                  </div>
                );
              }
              const { lower, upper } = col.bounds;
              return (
                <OutlierBoxChart
                  key={col.column}
                  colName={col.column}
                  lower={lower}
                  upper={upper}
                  outliers={col.sample_outliers ?? []}
                  outlierCount={col.outlier_count ?? 0}
                />
              );
            })}
          </div>
        </details>
      )}

      <table className="data-table">
        <thead>
          <tr>
            <th>列</th>
            <th>状态</th>
            <th>边界</th>
            <th>异常数</th>
            <th>占比</th>
            <th>样例</th>
          </tr>
        </thead>
        <tbody>
          {data.columns.map((c) => (
            <tr key={c.column}>
              <td>{c.column}</td>
              <td>
                {c.status === "ok" ? (
                  <span className="badge success">OK</span>
                ) : (
                  <span className="badge warning">{c.reason ?? "跳过"}</span>
                )}
              </td>
              <td>
                {c.bounds ? `[${c.bounds.lower.toFixed(2)}, ${c.bounds.upper.toFixed(2)}]` : "-"}
              </td>
              <td>
                {(c.outlier_count ?? 0) > 0 ? (
                  <span className="badge failed">{c.outlier_count}</span>
                ) : (
                  "0"
                )}
              </td>
              <td>{c.outlier_ratio !== undefined ? `${(c.outlier_ratio * 100).toFixed(2)}%` : "-"}</td>
              <td className="muted">{(c.sample_outliers ?? []).slice(0, 5).join(", ") || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** 模拟箱线图：x 轴是单列名，y 轴画 [lower, upper] 的区间条（一个 Bar，base=lower，高度=upper-lower），叠加 outlier 散点和边界 ReferenceLine。 */
function OutlierBoxChart({
  colName,
  lower,
  upper,
  outliers,
  outlierCount,
}: {
  colName: string;
  lower: number;
  upper: number;
  outliers: number[];
  outlierCount: number;
}) {
  const iqrHeight = upper - lower;
  void iqrHeight;
  // Scatter 点：每个异常值是一个点，x=colName（会被映射成 0 号 category），y=实际值
  const scatterData = outliers.map((v, i) => ({ cat: colName, y: v, i }));

  return (
    <div>
      <p className="muted" style={{ margin: "4px 0" }}>
        <b>{colName}</b> · 边界 [{lower.toFixed(2)}, {upper.toFixed(2)}] · 异常 {outlierCount} 个
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
          <XAxis
            type="category"
            dataKey="cat"
            name=""
            fontSize={11}
            allowDuplicatedCategory={true}
          />
          <YAxis type="number" fontSize={11} domain={["auto", "auto"]} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {/* IQR 范围参考线（边界） */}
          <ReferenceLine
            y={lower}
            stroke={CHART_SERIES[1]}
            strokeDasharray="3 3"
            label={{ value: `下限 ${lower.toFixed(2)}`, position: "right", fontSize: 10, fill: CHART_SERIES[1] }}
          />
          <ReferenceLine
            y={upper}
            stroke={CHART_SERIES[3]}
            strokeDasharray="3 3"
            label={{ value: `上限 ${upper.toFixed(2)}`, position: "right", fontSize: 10, fill: CHART_SERIES[3] }}
          />
          {/* 异常点 */}
          {scatterData.length > 0 && (
            <Scatter
              name="异常值"
              data={scatterData}
              dataKey="y"
              fill={CHART_SERIES[3]}
              r={5}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

