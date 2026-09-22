import { CHART_SERIES, CHART_GRID } from "../../lib/chartColors";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import type { TrainResult, ConfusionMatrixData } from "../../types/ml";
import type { SchemaColumn } from "../../types/dataset";
import { modelLabel } from "./modelMeta";
import { recommendCharts, CHART_LABELS, type ChartType } from "./modelMeta";

type Importance = { feature: string; importance: number };
type ResidualStats = {
  count: number;
  mean: number | null;
  std: number | null;
  max_abs: number | null;
  p50: number | null;
  p95: number | null;
  histogram: { range: string; count: number }[];
  note?: string;
};

const COLORS = CHART_SERIES;

function isNumericColumn(c: SchemaColumn): boolean {
  const t = c.dtype.toLowerCase();
  return t.includes("int") || t.includes("float");
}

function toFiniteNumber(v: unknown): number | null {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** 将一列数值样本等宽分桶，返回柱状图数据。 */
function buildBins(values: number[], binCount = 10): { name: string; count: number }[] {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [{ name: String(Number(min.toFixed(2))), count: values.length }];
  const width = (max - min) / binCount;
  const counts = new Array(binCount).fill(0);
  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.floor((v - min) / width));
    counts[idx] += 1;
  }
  return counts.map((count, i) => ({
    name: `${(min + i * width).toFixed(1)}~${(min + (i + 1) * width).toFixed(1)}`,
    count,
  }));
}

/** 目标列分布：分类任务按类别计数；回归任务对数值分桶。 */
function buildTargetDistribution(
  rows: Record<string, unknown>[],
  target: string,
  columns: SchemaColumn[],
): { name: string; count: number }[] {
  const col = columns.find((c) => c.column === target);
  const numeric = col ? isNumericColumn(col) : false;
  const values = rows.map((r) => r[target]);
  if (numeric) {
    const nums = values.map(toFiniteNumber).filter((v): v is number => v !== null);
    return buildBins(nums);
  }
  const counts = new Map<string, number>();
  for (const v of values) {
    const key = v === null || v === undefined ? "(空)" : String(v);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20);
}

interface Props {
  result: TrainResult | null;
  columns?: SchemaColumn[];
  sampleRows?: Record<string, unknown>[];
  target?: string | null;
}

// Prompt 174：训练结果展示 —— 运行状态 / 指标 / 耗时 / 错误 / 图表。
export function ExperimentResult({ result, columns = [], sampleRows = [], target = null }: Props) {
  const [chartType, setChartType] = useState<ChartType | null>(null);

  if (!result) return <div className="muted">尚未训练</div>;
  const { experiment, run } = result;
  const statusClass =
    run.status === "success" ? "success" : run.status === "failed" ? "failed" : "warning";

  const availableCharts =
    run.status === "success"
      ? recommendCharts(experiment.task, columns, target, experiment.model)
      : [];
  const activeChart =
    chartType && availableCharts.includes(chartType)
      ? chartType
      : (availableCharts[0] ?? null);

  // 数据图表所需字段
  const numericFeatures = columns
    .filter((c) => isNumericColumn(c) && c.column !== target)
    .map((c) => c.column);
  const scatterName = activeChart === "feature_scatter" ? `${numericFeatures[0]} / ${numericFeatures[1]}` : "";
  const scatterData =
    activeChart === "feature_scatter" && numericFeatures.length >= 2
      ? sampleRows
          .map((row) => ({
            x: toFiniteNumber(row[numericFeatures[0]]),
            y: toFiniteNumber(row[numericFeatures[1]]),
          }))
          .filter((p): p is { x: number; y: number } => p.x !== null && p.y !== null)
      : [];
  const histogramData =
    activeChart === "feature_histogram" && numericFeatures.length >= 1
      ? buildBins(
          sampleRows
            .map((row) => toFiniteNumber(row[numericFeatures[0]]))
            .filter((v): v is number => v !== null),
        )
      : [];
  const targetData =
    activeChart === "target_distribution" && target
      ? buildTargetDistribution(sampleRows, target, columns)
      : [];

  // 准备图表数据
  const metricEntries = Object.entries(run.metrics ?? {}).filter(
    ([k, v]) => typeof v === "number" && !k.includes("_note"),
  );
  const barData = metricEntries.map(([k, v]) => ({
    name: k,
    value: typeof v === "number" ? Number(v.toFixed(4)) : 0,
  }));
  const radarData = barData;

  const clusterCount = (run.metrics ?? {}).cluster_count as number | undefined;

  // 训练产物：特征重要性 / 混淆矩阵 / 残差统计
  const artifacts = (run.artifacts ?? {}) as Record<string, unknown>;
  const importance = (artifacts.feature_importance as { importances?: Importance[] } | null)
    ?.importances ?? [];
  const confusion = artifacts.confusion_matrix as ConfusionMatrixData | undefined;
  const residuals = artifacts.residual_stats as ResidualStats | undefined;
  const maxImportance = importance.reduce((m, item) => Math.max(m, item.importance), 0) || 1;

  return (
    <div>
      <div className="flex-between">
        <span>
          实验 #{experiment.id} · {modelLabel(experiment.model)}
          <span className={`badge ${statusClass}`} style={{ marginLeft: "var(--space-2)" }}>
            {run.status}
          </span>
        </span>
        <span className="muted">
          {run.runtime != null ? `耗时 ${run.runtime.toFixed(3)}s` : ""}
        </span>
      </div>

      <div className="mt" style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap" }}>
        {metricEntries.map(([k, v]) => (
          <div key={k} className="kv-item" style={{ minWidth: 140 }}>
            <div className="k">{k}</div>
            <div className="v" style={{ fontSize: 18, fontWeight: 600 }}>
              {typeof v === "number" ? v.toFixed(4) : String(v)}
            </div>
          </div>
        ))}
      </div>

      {run.error && (
        <p style={{ color: "var(--danger)" }}>
          训练失败：{run.error}
        </p>
      )}

      {/* 特征重要性（训练产物，模型不支持时后端返回 null） */}
      {run.status === "success" && importance.length > 0 && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>特征重要性</h4>
          <div>
            {importance.slice(0, 12).map((item) => (
              <div key={item.feature} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                <span style={{ width: 160, fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {item.feature}
                </span>
                <span
                  style={{
                    flex: 1,
                    height: 12,
                    background: "var(--surface-muted)",
                    borderRadius: 6,
                    overflow: "hidden",
                  }}
                >
                  <span
                    style={{
                      display: "block",
                      width: `${Math.max(2, (item.importance / maxImportance) * 100)}%`,
                      height: "100%",
                      background: CHART_SERIES[0],
                    }}
                  />
                </span>
                <span className="muted" style={{ width: 70, textAlign: "right", fontSize: 12 }}>
                  {item.importance.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 混淆矩阵：行=真实类别，列=预测类别 */}
      {run.status === "success" && confusion && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>混淆矩阵</h4>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>真实 \ 预测</th>
                  {confusion.labels.map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusion.matrix.map((row, i) => (
                  <tr key={confusion.labels[i] ?? i}>
                    <td>{confusion.labels[i] ?? i}</td>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        style={{
                          background: i === j ? "var(--success-weak)" : undefined,
                          fontWeight: i === j ? 600 : undefined,
                        }}
                      >
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 回归残差统计 */}
      {run.status === "success" && residuals && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>残差统计（真实 − 预测）</h4>
          {residuals.note ? (
            <p className="muted">{residuals.note}</p>
          ) : (
            <>
              <div className="kv-grid">
                <div className="kv-item">
                  <div className="k">样本数</div>
                  <div className="v">{residuals.count}</div>
                </div>
                <div className="kv-item">
                  <div className="k">均值</div>
                  <div className="v">{residuals.mean?.toFixed(4) ?? "-"}</div>
                </div>
                <div className="kv-item">
                  <div className="k">标准差</div>
                  <div className="v">{residuals.std?.toFixed(4) ?? "-"}</div>
                </div>
                <div className="kv-item">
                  <div className="k">最大绝对误差</div>
                  <div className="v">{residuals.max_abs?.toFixed(4) ?? "-"}</div>
                </div>
                <div className="kv-item">
                  <div className="k">P95</div>
                  <div className="v">{residuals.p95?.toFixed(4) ?? "-"}</div>
                </div>
              </div>
              {residuals.histogram.length > 0 && (
                <div className="chart-frame mt">
                  <ResponsiveContainer width="100%" aspect={1.8}>
                    <BarChart data={residuals.histogram} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                      <XAxis dataKey="range" fontSize={10} interval={0} angle={-25} height={56} textAnchor="end" />
                      <YAxis fontSize={11} allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill={CHART_SERIES[1]} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* 图表选择 + 可视化 */}
      {run.status === "success" && availableCharts.length > 0 && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <div className="flex-between" style={{ marginBottom: "var(--space-2)" }}>
            <h4 style={{ margin: 0 }}>可视化图表</h4>
            <div style={{ display: "flex", gap: 6 }}>
              {availableCharts.map((ct) => (
                <button
                  key={ct}
                  className={`btn ${activeChart === ct ? "primary" : ""}`}
                  style={{ padding: "4px 10px", fontSize: 12 }}
                  onClick={() => setChartType(ct)}
                >
                  {CHART_LABELS[ct]}
                </button>
              ))}
            </div>
          </div>

          {activeChart === "metrics_bar" && (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={barData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis dataKey="name" fontSize={11} interval={0} angle={-20} height={52} textAnchor="end" />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                  {barData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}

          {activeChart === "metrics_radar" && (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" fontSize={11} />
                <PolarRadiusAxis fontSize={10} />
                <Radar
                  name="指标值"
                  dataKey="value"
                  stroke={CHART_SERIES[0]}
                  fill={CHART_SERIES[0]}
                  fillOpacity={0.4}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          )}

          {activeChart === "cluster_bar" && clusterCount != null && (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={[{ name: "簇数量", count: clusterCount }]}
                margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill={CHART_SERIES[4]} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {activeChart === "feature_scatter" && (
            numericFeatures.length < 2 ? (
              <div className="muted">至少需要两个数值特征字段</div>
            ) : scatterData.length === 0 ? (
              <div className="muted">样本中没有可绘制的数值点（请先加载列信息）</div>
            ) : (
              <div>
                <p className="muted">
                  {scatterName} · 共 {scatterData.length} 个样本点
                </p>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis
                      type="number"
                      dataKey="x"
                      fontSize={11}
                      name={numericFeatures[0]}
                      label={{ value: numericFeatures[0], position: "insideBottom", offset: -2, fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      fontSize={11}
                      name={numericFeatures[1]}
                      label={{ value: numericFeatures[1], angle: -90, position: "insideLeft", fontSize: 11 }}
                    />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={scatterData} fill={CHART_SERIES[0]} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )
          )}

          {activeChart === "feature_histogram" &&
            (numericFeatures.length < 1 ? (
              <div className="muted">至少需要一个数值特征字段</div>
            ) : histogramData.length === 0 ? (
              <div className="muted">样本中没有可绘制的数值（请先加载列信息）</div>
            ) : (
              <div>
                <p className="muted">{numericFeatures[0]} 分布</p>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={histogramData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="name" fontSize={10} interval={0} angle={-25} height={56} textAnchor="end" />
                    <YAxis fontSize={11} allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill={CHART_SERIES[1]} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}

          {activeChart === "target_distribution" &&
            (!target ? (
              <div className="muted">未选择目标字段</div>
            ) : targetData.length === 0 ? (
              <div className="muted">样本中没有可绘制的目标值（请先加载列信息）</div>
            ) : (
              <div>
                <p className="muted">{target} 分布</p>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={targetData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="name" fontSize={11} interval={0} angle={-20} height={52} textAnchor="end" />
                    <YAxis fontSize={11} allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {targetData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
