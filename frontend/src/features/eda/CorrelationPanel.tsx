import { CHART_SERIES, CHART_GRID, withAlpha } from "../../lib/chartColors";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface CorrelationData {
  method: string;
  columns: string[];
  matrix: Record<string, Record<string, number | null>>;
}

interface Props {
  data: CorrelationData | null;
  /** 可选：原始样本行，用于绘制散点图（从 preview 来） */
  scatterPoints?: Record<string, unknown>[];
}

// Prompt 168 + 增强：相关性热力表 + top 2 相关特征对的散点图。
export function CorrelationPanel({ data, scatterPoints }: Props) {
  if (!data || !data.columns.length) return <div className="muted">请先运行相关性分析</div>;

  // 找出相关性绝对值最大的前 2 对（排除对角线）
  const topPairs: { a: string; b: string; v: number }[] = [];
  for (let i = 0; i < data.columns.length; i++) {
    for (let j = i + 1; j < data.columns.length; j++) {
      const a = data.columns[i];
      const b = data.columns[j];
      const v = data.matrix[a]?.[b];
      if (typeof v === "number") topPairs.push({ a, b, v });
    }
  }
  topPairs.sort((x, y) => Math.abs(y.v) - Math.abs(x.v));
  const top2 = topPairs.slice(0, 2);

  return (
    <div style={{ overflowX: "auto" }}>
      <p className="muted">方法：{data.method}（值域 -1 ~ 1）</p>
      <table className="corr-table">
        <thead>
          <tr>
            <th></th>
            {data.columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.columns.map((r) => (
            <tr key={r}>
              <th>{r}</th>
              {data.columns.map((c) => {
                const v = data.matrix[r]?.[c];
                return (
                  <td key={c} style={{ background: cellColor(v) }}>
                    {v === null || v === undefined ? "-" : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* 散点图：top 相关对 */}
      {top2.length > 0 && (
        <details style={{ marginTop: "var(--space-4)" }}>
          <summary style={{ cursor: "pointer", fontWeight: 500 }}>
            Top 相关特征对散点图（共 {top2.length} 对）
          </summary>
          <div style={{ marginTop: "var(--space-2)", display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
            {top2.map((pair, idx) => {
              const points = buildScatterPoints(scatterPoints, pair.a, pair.b);
              return (
                <div key={idx}>
                  <p className="muted" style={{ margin: "4px 0" }}>
                    #{idx + 1}  <b>{pair.a}</b> × <b>{pair.b}</b> · 相关系数 {pair.v.toFixed(3)}
                  </p>
                  {points ? (
                    <ResponsiveContainer width="100%" height={220}>
                      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                        <XAxis type="number" dataKey="x" name={pair.a} fontSize={11} />
                        <YAxis type="number" dataKey="y" name={pair.b} fontSize={11} />
                        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                        <Scatter data={points} fill={CHART_SERIES[0]} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="muted">暂无散点图数据（需先预览数据）</div>
                  )}
                </div>
              );
            })}
          </div>
        </details>
      )}
    </div>
  );
}

function cellColor(v: number | null | undefined): string {
  if (v === null || v === undefined) return "transparent";
  const alpha = Math.min(Math.abs(v), 1) * 0.75;
  return v >= 0 ? withAlpha(CHART_SERIES[0], alpha) : withAlpha(CHART_SERIES[3], alpha);
}

/** 从原始样本里提取两列数值点，供散点图用。 */
function buildScatterPoints(
  rows: Record<string, unknown>[] | undefined,
  colA: string,
  colB: string,
): { x: number; y: number }[] | null {
  if (!rows || !rows.length) return null;
  const points: { x: number; y: number }[] = [];
  for (const row of rows) {
    const ax = Number(row[colA]);
    const by = Number(row[colB]);
    if (!Number.isNaN(ax) && !Number.isNaN(by)) {
      points.push({ x: ax, y: by });
    }
    if (points.length >= 200) break; // 限制点数量
  }
  return points.length > 0 ? points : null;
}
