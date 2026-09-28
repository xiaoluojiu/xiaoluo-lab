import { CHART_SERIES, CHART_GRID } from "../../lib/chartColors";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
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
import type {
  CalibrationData,
  CrossValidationData,
  LearningCurveData,
  MetricInterval,
  TrainArtifacts,
  TrainResult,
  ConfusionMatrixData,
} from "../../types/ml";
import type { SchemaColumn } from "../../types/dataset";
import { exportRunReport, type RunReportFormat } from "../../api/ml";
import { modelLabel } from "./modelMeta";
import { recommendCharts, CHART_LABELS, formatMetric, type ChartType } from "./modelMeta";

/**
 * 值域在 [0,1] 的指标（准确率类 + r2 + 轮廓系数）。
 *
 * 柱状图与雷达图只画这一组：roc_auc 虽也在 [0,1]，但它是「排序质量」而不是
 * 「命中率」，和 accuracy 混在同一根坐标轴上会诱导人直接比高低。
 */
const RATIO_METRIC_RE = /^(accuracy|precision|recall|f1|r2|silhouette)$/;

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
  // 散点 / 直方图的显式字段选择：缺省用前两个数值列，用户可自行切换。
  const [xField, setXField] = useState<string | null>(null);
  const [yField, setYField] = useState<string | null>(null);
  const [histField, setHistField] = useState<string | null>(null);
  const [reportFormat, setReportFormat] = useState<RunReportFormat>("html");
  const [exporting, setExporting] = useState(false);

  /** 导出本次运行的详细报告：报告体与渲染都在后端，前端只负责触发下载。 */
  async function downloadRunReport() {
    if (!result) return;
    setExporting(true);
    try {
      const blob = await exportRunReport(run.id, reportFormat);
      const ext = reportFormat === "pdf" ? "pdf" : reportFormat === "markdown" ? "md" : "html";
      const mime = reportFormat === "pdf" ? "application/pdf"
        : reportFormat === "markdown" ? "text/markdown;charset=utf-8" : "text/html;charset=utf-8";
      const url = URL.createObjectURL(new Blob([blob], { type: mime }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `ML-Run${run.id}.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      window.alert(e instanceof Error ? e.message : "导出失败，请重试");
    } finally {
      setExporting(false);
    }
  }

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
  // 用户选过的字段若已不在当前列清单里（换数据集 / 换目标列），自动回落到缺省
  const effX = xField && numericFeatures.includes(xField) ? xField : numericFeatures[0];
  const effY = yField && numericFeatures.includes(yField) ? yField : numericFeatures[1];
  const effHist = histField && numericFeatures.includes(histField) ? histField : numericFeatures[0];
  const scatterName = activeChart === "feature_scatter" ? `${effX} / ${effY}` : "";
  const scatterData =
    activeChart === "feature_scatter" && numericFeatures.length >= 2
      ? sampleRows
          .map((row) => ({
            x: toFiniteNumber(row[effX]),
            y: toFiniteNumber(row[effY]),
          }))
          .filter((p): p is { x: number; y: number } => p.x !== null && p.y !== null)
      : [];
  const histogramData =
    activeChart === "feature_histogram" && numericFeatures.length >= 1
      ? buildBins(
          sampleRows
            .map((row) => toFiniteNumber(row[effHist]))
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
  // 柱状图 / 雷达图只画 [0,1] 口径组；roc_auc、mae、rmse 这类量纲不同的指标
  // 拆出来单独列在图下 —— 混在一根坐标轴上 0.39 与 0.98 会被误读成「rmse 远差于 r2」。
  const ratioEntries = metricEntries.filter(([k]) => RATIO_METRIC_RE.test(k));
  const otherEntries = metricEntries.filter(([k]) => !RATIO_METRIC_RE.test(k));
  const barData = ratioEntries.map(([k, v]) => ({
    name: k,
    value: typeof v === "number" ? Number(v.toFixed(4)) : 0,
  }));
  const radarData = barData;
  const otherNote = otherEntries.some(([k]) => k === "roc_auc")
    ? "roc_auc 与准确率类指标口径不同，不放在同一坐标比较。"
    : "其余指标量纲与 [0,1] 口径不同，不放在同一坐标比较。";

  const clusterCount = (run.metrics ?? {}).cluster_count as number | undefined;

  // 训练产物：特征重要性 / 混淆矩阵 / 残差统计
  const artifacts = (run.artifacts ?? {}) as Record<string, unknown>;
  const importance = (artifacts.feature_importance as { importances?: Importance[] } | null)
    ?.importances ?? [];
  const confusion = artifacts.confusion_matrix as ConfusionMatrixData | undefined;
  const residuals = artifacts.residual_stats as ResidualStats | undefined;
  const calibration = artifacts.calibration as CalibrationData | undefined;
  const learningCurve = artifacts.learning_curve as LearningCurveData | undefined;
  const maxImportance = importance.reduce((m, item) => Math.max(m, item.importance), 0) || 1;

  // 指标置信区间：点估计看不出「稳不稳」，区间才说明 0.83 与 0.85 有没有差别。
  // metric_ci 里混着 n_bootstrap / basis 这类元信息，取值前按对象判定。
  const metricCi = (artifacts.metric_ci ?? null) as Record<
    string,
    MetricInterval | number | string | null
  > | null;
  const ciOf = (name: string): MetricInterval | null => {
    const v = metricCi?.[name];
    return v && typeof v === "object" ? v : null;
  };
  const cv = (artifacts.cv ?? null) as CrossValidationData | null;

  // 训练数据预算与抽样口径：抽没抽、抽了多少、上限是谁定的，必须写在结果上。
  // 抽样过的指标只代表那份子集，不标出来就会被当成全量数据的结论。
  const sampling = (artifacts.sampling ?? null) as TrainArtifacts["sampling"] | null;
  const samplingNote = sampling
    ? sampling.sampled
      ? `本次训练从 ${sampling.original_rows.toLocaleString()} 行中随机抽取 ${sampling.used_rows.toLocaleString()} 行（${(sampling.sample_rate * 100).toFixed(1)}%），上限来自${sampling.limit_source === "user" ? "你在「训练数据量」里指定的预算" : "系统默认上限"} —— 指标只代表这份抽样数据上的表现。`
      : `本次训练使用全部 ${sampling.used_rows.toLocaleString()} 行，未抽样 —— 指标可直接对应该数据集。`
    : null;

  return (
    <div>
      <div className="flex-between">
        <span>
          实验 #{experiment.id} · {modelLabel(experiment.model)}
          <span className={`badge ${statusClass}`} style={{ marginLeft: "var(--space-2)" }}>
            {run.status}
          </span>
          {sampling && (
            <span
              className={`badge ml-sampling-badge ${sampling.sampled ? "warning" : "success"}`}
              style={{ marginLeft: "var(--space-2)" }}
              title={samplingNote ?? undefined}
            >
              {sampling.sampled
                ? `抽样训练 ${sampling.used_rows.toLocaleString()} / ${sampling.original_rows.toLocaleString()} 行`
                : `全量 ${sampling.used_rows.toLocaleString()} 行`}
            </span>
          )}
        </span>
        <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
          {run.runtime != null && <span className="muted">耗时 {run.runtime.toFixed(3)}s</span>}
          <select
            aria-label="导出格式"
            value={reportFormat}
            onChange={(e) => setReportFormat(e.target.value as RunReportFormat)}
            style={{ padding: "4px 6px" }}
          >
            <option value="html">HTML</option>
            <option value="markdown">Markdown</option>
            <option value="pdf">PDF</option>
          </select>
          <button
            type="button"
            className="btn primary btn-sm"
            disabled={exporting || run.status !== "success"}
            onClick={() => void downloadRunReport()}
          >
            {exporting ? "导出中..." : "📥 一键导出详细报告"}
          </button>
        </span>
      </div>

      {/* 抽样口径：指标能不能代表全量数据，取决于这一行 */}
      {samplingNote && <p className="muted ml-sampling-note">{samplingNote}</p>}
      {/* 轮廓系数没有「越高越好到什么程度」的直觉，不给刻度就无法判断簇分得好不好 */}
      {run.status === "success" && String(artifacts.task ?? experiment.task) === "clustering" && (
        <p className="muted ml-sampling-note">
          轮廓系数取值 -1 ~ 1：&gt; 0.5 说明簇结构清晰，0.25 ~ 0.5 勉强可接受，
          &lt; 0.25 基本没有分离出簇，建议换特征或换簇数重跑。
        </p>
      )}

      <div className="mt" style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap" }}>
        {metricEntries.map(([k, v]) => {
          const ci = ciOf(k);
          return (
            <div key={k} className="kv-item" style={{ minWidth: 140 }}>
              <div className="k">{k}</div>
              <div className="v" style={{ fontSize: 18, fontWeight: 600 }}>
                {formatMetric(k, v)}
              </div>
              {ci && (
                <div className="muted" style={{ fontSize: 12 }}>
                  95% CI [{ci.lower.toFixed(3)}, {ci.upper.toFixed(3)}]
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 交叉验证：与上面的测试集指标是两个独立口径（换切分还稳不稳） */}
      {cv?.mean && Object.keys(cv.mean).length > 0 && (
        <div className="mt ml-cv">
          <div className="muted" style={{ fontWeight: 600 }}>
            {cv.folds} 折交叉验证（{cv.note ?? "预处理在每折训练集内拟合"}）
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>指标</th>
                <th>各折均值</th>
                <th>标准差</th>
                <th>每折</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(cv.mean).map(([m, mean]) => (
                <tr key={m}>
                  <td>{m}</td>
                  <td>{mean.toFixed(4)}</td>
                  <td>{(cv.std?.[m] ?? 0).toFixed(4)}</td>
                  <td className="muted">
                    {(cv.per_fold?.find((p) => p.metric === m)?.values ?? [])
                      .map((x) => x.toFixed(3))
                      .join(" / ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

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

      {/* 概率校准：可靠性曲线 + ECE（二分类，留出测试集口径） */}
      {run.status === "success" && calibration && calibration.points.length > 0 && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>概率校准（可靠性曲线）</h4>
          <p className="muted">
            依据留出测试集 · 正类 {calibration.positive_class ?? "—"} · 虚线对角线 = 完美校准
          </p>
          <div className="kv-grid">
            <div className="kv-item">
              <div className="k">ECE（期望校准误差）</div>
              <div className="v">{calibration.ece.toFixed(4)}</div>
            </div>
            <div className="kv-item">
              <div className="k">MCE（最大校准误差）</div>
              <div className="v">{calibration.mce.toFixed(4)}</div>
            </div>
          </div>
          <p className="muted">
            ECE ≤ 0.05 校准良好；0.05~0.10 一般；&gt; 0.10 概率偏差明显，不宜直接按概率做决策。
          </p>
          <div className="chart-frame mt">
            <ResponsiveContainer width="100%" aspect={1.8}>
              <LineChart margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[0, 1]}
                  fontSize={11}
                  label={{ value: "平均预测概率", position: "insideBottom", offset: -4, fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 1]}
                  fontSize={11}
                  label={{ value: "实际正类比例", angle: -90, position: "insideLeft", fontSize: 11 }}
                />
                <Tooltip />
                <Line
                  data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                  dataKey="y"
                  stroke="#9ca3af"
                  strokeDasharray="6 4"
                  dot={false}
                  isAnimationActive={false}
                  name="完美校准"
                />
                <Line
                  data={calibration.points.map((p) => ({
                    x: p.mean_predicted,
                    y: p.observed_frequency,
                    count: p.count,
                  }))}
                  dataKey="y"
                  stroke={CHART_SERIES[0]}
                  strokeWidth={2}
                  name="实际校准"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 学习曲线：测试集分数趋平=数据够了；持续上升=加数据仍有效 */}
      {run.status === "success" && learningCurve && learningCurve.points.length > 0 && (
        <div className="mt" style={{ borderTop: "1px solid var(--border)", paddingTop: "var(--space-3)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>
            学习曲线（{learningCurve.metric} vs 训练样本量）
          </h4>
          <p className="muted">
            测试集曲线趋于平缓 → 继续增加数据收益很小；仍明显上升 → 增加训练数据有效。
          </p>
          <div className="chart-frame mt">
            <ResponsiveContainer width="100%" aspect={1.8}>
              <LineChart
                data={learningCurve.points}
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis
                  type="number"
                  dataKey="rows"
                  fontSize={11}
                  label={{ value: "训练样本数", position: "insideBottom", offset: -4, fontSize: 11 }}
                />
                <YAxis domain={[0, "auto"]} fontSize={11} />
                <Tooltip />
                <Line
                  dataKey="train_score"
                  stroke={CHART_SERIES[1]}
                  strokeWidth={2}
                  name="训练集分数"
                />
                <Line
                  dataKey="test_score"
                  stroke={CHART_SERIES[0]}
                  strokeWidth={2}
                  name="测试集分数"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {learningCurve.note && <p className="muted">{learningCurve.note}</p>}
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
            <>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={barData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                  <XAxis dataKey="name" fontSize={11} interval={0} angle={-20} height={52} textAnchor="end" />
                  <YAxis fontSize={11} domain={[0, 1]} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {barData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              {otherEntries.length > 0 && (
                <>
                  <p className="muted" style={{ fontSize: 12, marginTop: "var(--space-1)" }}>
                    {otherEntries.map(([k, v]) => `${k} = ${formatMetric(k, v)}`).join("　")}
                  </p>
                  <p className="muted" style={{ fontSize: 12 }}>{otherNote}</p>
                </>
              )}
            </>
          )}

          {activeChart === "metrics_radar" && (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" fontSize={11} />
                <PolarRadiusAxis fontSize={10} domain={[0, 1]} />
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
                <div className="flex-between" style={{ marginBottom: "var(--space-1)" }}>
                  <p className="muted" style={{ margin: 0 }}>
                    {scatterName} · 共 {scatterData.length} 个样本点
                  </p>
                  <div style={{ display: "flex", gap: 6 }}>
                    <label className="field" style={{ marginBottom: 0, fontSize: 12 }}>
                      横轴
                      <select
                        className="input"
                        value={effX ?? ""}
                        onChange={(e) => setXField(e.target.value || null)}
                        style={{ width: 140 }}
                      >
                        {numericFeatures.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                    </label>
                    <label className="field" style={{ marginBottom: 0, fontSize: 12 }}>
                      纵轴
                      <select
                        className="input"
                        value={effY ?? ""}
                        onChange={(e) => setYField(e.target.value || null)}
                        style={{ width: 140 }}
                      >
                        {numericFeatures.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis
                      type="number"
                      dataKey="x"
                      fontSize={11}
                      name={effX}
                      label={{ value: effX, position: "insideBottom", offset: -2, fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      fontSize={11}
                      name={effY}
                      label={{ value: effY, angle: -90, position: "insideLeft", fontSize: 11 }}
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
                <div className="flex-between" style={{ marginBottom: "var(--space-1)" }}>
                  <p className="muted" style={{ margin: 0 }}>{effHist} 分布</p>
                  <label className="field" style={{ marginBottom: 0, fontSize: 12 }}>
                    字段
                    <select
                      className="input"
                      value={effHist ?? ""}
                      onChange={(e) => setHistField(e.target.value || null)}
                      style={{ width: 140 }}
                    >
                      {numericFeatures.map((f) => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                    </select>
                  </label>
                </div>
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
