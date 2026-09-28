import { CHART_PRIMARY, CHART_INDIGO_LIGHT, CHART_NULL, CHART_SERIES, CHART_PRIMARY_RGB } from "../../lib/chartColors";
/**
 * 统一可视化面板：直方图 / 柱状图 / 折线图 / 散点图 / 箱线图 / 热力图 /
 * Q-Q 图 / 分组柱状图 / 面积图（CDF）。
 * - 图表使用固定 aspect 比例（16:9 左右），避免容器宽度变化导致的畸形拉伸。
 * - 每张图支持「下载 PNG / 下载 CSV」。
 * 字段选择由上层 EDA 工作台统一管理，这里只负责图表类型、数据请求和展示。
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { edaVisualize, type VisualizeChart } from "../../api/analysis";
import { Link } from "react-router-dom";
import { formatError, type AnalysisError } from "../../lib/analysisError";
import { chartFieldAdvice, isContinuousNumeric, isNumericColumn, isTemporalColumn } from "../../lib/edaColumns";
import type { SchemaColumn } from "../../types/dataset";
import { Icon } from "../../components/icons/Icon";

/**
 * 列类型判定统一由 `lib/edaColumns` 提供（该文件是全仓 dtype 正则的唯一真源）。
 * 这里保留同名 re-export 只是为了不打断既有调用方（如 pages/Analysis）——
 * 此前本文件内联了一份与 lib 逐字相同的实现，属重复逻辑。
 */
export { isNumericColumn, isTemporalColumn };

const CHART_TYPES: Array<{ value: VisualizeChart; label: string }> = [
  { value: "histogram", label: "直方图" },
  { value: "bar", label: "柱状图" },
  { value: "line", label: "折线图" },
  { value: "scatter", label: "散点图" },
  { value: "boxplot", label: "箱线图" },
  { value: "heatmap", label: "热力图" },
  { value: "qq", label: "Q-Q 图" },
  { value: "grouped_bar", label: "分组柱状图" },
  { value: "area", label: "面积图(CDF)" },
];

const PRIMARY = CHART_PRIMARY;
// 统一比例：宽度 / 高度，防止宽屏下被拉扁（用户反馈的“自适应比例异常”）。
const ASPECT = 1.8;
// 失败后自动重试的次数上限：给「改字段/换数据集」留出自愈机会，又不至于空转。
const RETRY_LIMIT = 2;

interface Props {
  datasetId: number;
  columns: SchemaColumn[];
  selectedColumns: string[];
  chart: VisualizeChart;
  onChartChange: (chart: VisualizeChart) => void;
  /** /schema 返回的总行数：低基数判定必须与后端 continuous_columns 同口径。 */
  rowCount: number;
}

/** 图表字段位：每种图有哪些"槽位"（column / x / y / group_by）。 */
type ChartSlot = "column" | "x" | "y" | "group_by";

/** 一个字段位的完整描述：候选列 + 未手动指定时的自动值。UI 与取值共用同一份。 */
interface SlotSpec {
  slot: ChartSlot;
  label: string;
  auto: string | undefined;
  options: string[];
}

export function VisualizationPanel({
  datasetId,
  columns,
  selectedColumns,
  chart,
  onChartChange,
  rowCount,
}: Props) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<AnalysisError | null>(null);
  // 字段位覆盖：key 形如 `${chart}.${slot}`。用户在下拉里手动指定后优先于自动值。
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  // 用字符串做依赖，避免对象引用变化造成无谓的重跑判断。
  const overrideKey = useMemo(() => JSON.stringify(overrides), [overrides]);

  const colNames = useMemo(() => columns.map((c) => c.column), [columns]);
  const numericCols = useMemo(
    () => columns.filter(isNumericColumn).map((c) => c.column),
    [columns],
  );
  // 「取值连续的数值列」：排除 Month/DayofMonth/DayOfWeek 这类低基数编码列。
  // 后端 continuous_columns 会把它们按分类处理并拒绝作相关性输入，
  // 前端若直接按 dtype 当数值列提交，就会稳定拿到 422。
  const continuousNumericCols = useMemo(
    () => columns.filter((c) => isContinuousNumeric(c, rowCount)).map((c) => c.column),
    [columns, rowCount],
  );
  const temporalCols = useMemo(
    () => columns.filter(isTemporalColumn).map((c) => c.column),
    [columns],
  );
  const categoricalCols = useMemo(
    () =>
      columns
        .filter((c) => !isNumericColumn(c) && !isTemporalColumn(c))
        .map((c) => c.column),
    [columns],
  );
  const selectedSet = useMemo(() => new Set(selectedColumns), [selectedColumns]);
  const selectedNumeric = useMemo(
    () => continuousNumericCols.filter((name) => selectedSet.has(name)),
    [continuousNumericCols, selectedSet],
  );
  const selectedCategorical = useMemo(
    () => categoricalCols.filter((name) => selectedSet.has(name)),
    [categoricalCols, selectedSet],
  );
  const selectedTemporal = useMemo(
    () => temporalCols.filter((name) => selectedSet.has(name)),
    [temporalCols, selectedSet],
  );
  // 字段位下拉的候选列：勾选中的排前面，其余随后（去重保序）。
  // 这样「左侧勾选 → 立刻可选」与「没勾选也能直接挑」两种用法都成立。
  const continuousOptions = useMemo(
    () => [...selectedNumeric, ...continuousNumericCols.filter((name) => !selectedSet.has(name))],
    [selectedNumeric, continuousNumericCols, selectedSet],
  );
  const categoricalOptions = useMemo(
    () => [...selectedCategorical, ...categoricalCols.filter((name) => !selectedSet.has(name))],
    [selectedCategorical, categoricalCols, selectedSet],
  );
  // 折线图 x 轴候选：选中时间列 → 全部时间列 → 连续数值列（去重保序）。
  const lineXOptions = useMemo(
    () =>
      [...selectedTemporal, ...temporalCols, ...continuousNumericCols].filter(
        (name, index, arr) => arr.indexOf(name) === index,
      ),
    [selectedTemporal, temporalCols, continuousNumericCols],
  );
  // 热力图：优先用勾选的连续数值列；没有勾选时用数据集里全部连续数值列。
  // 绝不把分类列混进来 —— 这正是日志里 columns=DepDelay,Month 的来源。
  const effectiveNumeric = selectedNumeric.length ? selectedNumeric : continuousNumericCols;

  // 用户勾选的列里，哪些会被后端按分类编码排除（用于给出精确提示）。
  const rejectedByBackend = useMemo(() => {
    const picked = selectedColumns.filter(
      (name) => !continuousNumericCols.includes(name) && !categoricalCols.includes(name),
    );
    return picked.filter((name) => numericCols.includes(name));
  }, [selectedColumns, continuousNumericCols, categoricalCols, numericCols]);

  const advice = useMemo(
    () => chartFieldAdvice(chart, columns, rowCount),
    [chart, columns, rowCount],
  );

  useEffect(() => {
    setData(null);
    setError(null);
    // overrideKey 一并作为依赖：改字段位后必须作废旧图，否则会把上一参数的图当成新的。
  }, [datasetId, selectedColumns.join("\u0001"), overrideKey]);

  // 切换数据集清空字段位覆盖：旧数据集的列名在新数据集里可能不存在（或含义不同）。
  useEffect(() => {
    setOverrides({});
  }, [datasetId]);

  // 自动生成：失败后必须能重跑，否则用户改完字段仍停在报错页（回归：改字段后
  // 报错文案不消失、图表要手动点「生成图表」才恢复，看起来像功能坏了）。
  // retryKey 自增以重新触发请求；次数设上限，避免与后端校验死磕时反复打接口。
  const [retryKey, setRetryKey] = useState(0);
  const retryBudgetRef = useRef(RETRY_LIMIT);

  useEffect(() => {
    if (data || loading || !error) return undefined;
    if (retryBudgetRef.current <= 0) return undefined;
    const timer = setTimeout(() => {
      retryBudgetRef.current -= 1;
      setRetryKey((k) => k + 1);
    }, 400);
    return () => clearTimeout(timer);
  }, [error, data, loading, retryKey]);

  // 勾选字段 / 数据集 / 图表类型 / 字段位变化 → 重置重试预算并清空错误，重新自动生成。
  // 字段位变化走这里（bump retryKey）触发重新请求，不需要额外的触发机制。
  useEffect(() => {
    retryBudgetRef.current = RETRY_LIMIT;
    setError(null);
    setRetryKey((k) => k + 1);
  }, [datasetId, chart, selectedColumns.join("\u0001"), overrideKey]);

  useEffect(() => {
    void run();
    // run 依赖的 params 由 chart/columns/selection 派生，这里用显式依赖表达意图。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, chart, selectedColumns.join("\u0001"), retryKey]);

  const selectionNote = selectedColumns.length
    ? `已勾选 ${selectedColumns.length} 个字段（下方选择器可指定每个字段位）`
    : "未勾选字段，已按图表类型自动选择字段";

  /** 该字段位被手动指定的值（仍需存在于当前 dataset schema），否则 undefined。 */
  function overrideOf(type: VisualizeChart, slot: ChartSlot): string | undefined {
    const picked = overrides[`${type}.${slot}`];
    return picked && colNames.includes(picked) ? picked : undefined;
  }

  /** 按图表类型描述各字段位（heatmap 不产出字段位，沿用「勾选优先」的既有逻辑）。 */
  function slotSpecs(type: VisualizeChart): SlotSpec[] {
    const numeric = continuousOptions;
    const categorical = categoricalOptions;
    switch (type) {
      case "histogram":
      case "boxplot":
      case "qq":
      case "area":
        return [{ slot: "column", label: "字段", auto: numeric[0], options: numeric }];
      case "bar":
        return [{ slot: "column", label: "分类字段", auto: categorical[0], options: categorical }];
      case "scatter": {
        // y 的自动值跟随「实际生效的 x」：与改动前 `find(name => name !== x)` 的口径一致，
        // 否则只改 x 就会撞成同列、图表直接消失（x≠y 的校验是给手改两处时兜底的）。
        const effectiveX = overrideOf(type, "x") ?? numeric[0];
        return [
          { slot: "x", label: "X 轴", auto: numeric[0], options: numeric },
          { slot: "y", label: "Y 轴", auto: numeric.find((name) => name !== effectiveX), options: numeric },
        ];
      }
      case "line": {
        const defaultX = selectedTemporal[0] ?? temporalCols[0] ?? numeric[0];
        const effectiveX = overrideOf(type, "x") ?? defaultX;
        return [
          { slot: "x", label: "X 轴", auto: defaultX, options: lineXOptions },
          { slot: "y", label: "Y 轴", auto: numeric.find((name) => name !== effectiveX), options: numeric },
        ];
      }
      case "grouped_bar": {
        // 分组列的自动值同样跟随实际生效的分类列（两列相同时后端直接 422）。
        const effectiveColumn = overrideOf(type, "column") ?? categorical[0];
        return [
          { slot: "column", label: "分类列", auto: categorical[0], options: categorical },
          { slot: "y", label: "数值列", auto: numeric[0], options: numeric },
          {
            slot: "group_by",
            label: "分组列",
            auto: categorical.find((name) => name !== effectiveColumn),
            options: categorical,
          },
        ];
      }
      default:
        return [];
    }
  }

  /** 字段位最终生效值：手动指定且仍存在于当前 dataset schema 时用它，否则回退自动值。 */
  function resolveSlot(spec: SlotSpec, type: VisualizeChart): string | undefined {
    return overrideOf(type, spec.slot) ?? spec.auto;
  }

  function chartParams(): Parameters<typeof edaVisualize>[1] | null {
    if (chart === "heatmap") {
      if (effectiveNumeric.length < 2) return null;
      return { chart, columns: effectiveNumeric, sample_limit: 1000 };
    }
    const specs = slotSpecs(chart);
    const at = (slot: ChartSlot) => {
      const spec = specs.find((s) => s.slot === slot);
      return spec ? resolveSlot(spec, chart) : undefined;
    };
    if (chart === "scatter" || chart === "line") {
      const x = at("x");
      const y = at("y");
      // 必须取两个不同字段：后端对 x==y 会返回 422（同一列做不了散点/折线）。
      // 不足 2 列时返回 null 走空状态提示。
      if (!x || !y || x === y) return null;
      return { chart, x, y, sample_limit: 1000 };
    }
    if (chart === "grouped_bar") {
      const column = at("column");
      const y = at("y");
      const groupBy = at("group_by");
      // 三个字段两两互异（后端对重名列会返回 422）。
      if (!column || !y || !groupBy) return null;
      if (column === y || column === groupBy || y === groupBy) return null;
      return { chart, column, y, group_by: groupBy, agg: "mean" };
    }
    const column = at("column");
    if (!column) return null;
    if (chart === "bar") return { chart, column, top_n: 20, sample_limit: 1000 };
    if (chart === "histogram" || chart === "boxplot") {
      return { chart, column, top_n: 20, bins: 10, sample_limit: 1000 };
    }
    if (chart === "qq") return { chart, column };
    if (chart === "area") return { chart, column, bins: 12 };
    return null;
  }

  const params = chartParams();
  const ready = params !== null;
  const slots = slotSpecs(chart);

  async function run() {
    if (!params) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await edaVisualize(datasetId, params);
      setData(result);
      // 若后端补齐了空数据，清掉本地「重试」状态，避免无意义的自动重跑。
      setError(null);
    } catch (e) {
      // 展示后端业务原因（缺哪列 / 为什么这列不可用），并补全「影响 + 怎么解决」。
      setError(formatError(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="visualization-toolbar">
        {CHART_TYPES.map((t) => (
          <button
            key={t.value}
            type="button"
            className={`btn ${chart === t.value ? "primary" : ""}`}
            onClick={() => {
              onChartChange(t.value);
              setData(null);
              setError(null);
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* 字段位选择器：把 X/Y/分组等槽位显式交给用户，不再只能吃数组前两列。 */}
      {slots.length > 0 && (
        <div className="visualization-slots">
          {slots.map((spec) => (
            <label className="field" key={spec.slot}>
              {spec.label}
              <select
                value={resolveSlot(spec, chart) ?? ""}
                onChange={(event) =>
                  setOverrides((current) => ({
                    ...current,
                    [`${chart}.${spec.slot}`]: event.target.value,
                  }))
                }
              >
                {spec.options.length === 0 && <option value="">无可选字段</option>}
                {spec.options.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}

      <div className="visualization-selection-note">
        <span>{selectionNote}</span>
        {params && <span className="muted">· {describeParams(chart, params)}</span>}
      </div>

      {!ready && (
        <div className="analysis-empty visualization-empty">
          <span className="analysis-empty-illustration" aria-hidden="true">
            <Icon name="chart" size={28} />
          </span>
          <strong className="analysis-empty-title">
            {chart === "heatmap" || chart === "scatter" || chart === "grouped_bar" ? "还需要更多字段" : "当前字段不足以生成此图表"}
          </strong>
          <span className="analysis-empty-hint">{advice.message}</span>
        </div>
      )}

      {ready && (
        <button className="btn primary" type="button" disabled={loading} onClick={() => setRetryKey((k) => k + 1)}>
          {loading ? "生成中..." : "重新生成"}
        </button>
      )}

      {/* 失败原因必须说清「哪一列、为什么、怎么办」：
          后端已经返回了业务 message，这里再补上「依据当前数据集该怎么选」的建议。
          回归：原先只渲染一行 <p>，用户看不到是哪个字段的问题。 */}
      {error && (
        <div className="analysis-error" role="alert" style={{ marginTop: "var(--space-3)" }}>
          <span className="analysis-error-icon" aria-hidden="true">
            <Icon name="alert" size={18} />
          </span>
          <div className="analysis-error-body">
            <strong className="analysis-error-title">这张图暂时生成不了</strong>
            <span className="analysis-error-message">
              <b className="analysis-error-tag">发生了什么</b>
              {error.what}
            </span>
            {rejectedByBackend.length > 0 && (
              <span className="analysis-error-message">
                你勾选的 {rejectedByBackend.join("、")} 属于低基数编码列（取值种类过少），
                后端按分类字段处理，不能参与相关性计算。
              </span>
            )}
            <span className="analysis-error-hint">
              <b className="analysis-error-tag">影响</b>
              {error.impact}
            </span>
            <span className="analysis-error-hint">
              <b className="analysis-error-tag">怎么解决</b>
              {error.solution}
            </span>
            {advice.message && (
              <span className="analysis-error-hint">建议：{advice.message}</span>
            )}
            {advice.suggested.length > 0 && (
              <span className="analysis-error-hint">
                可直接使用：<strong>{advice.suggested.join("、")}</strong>
              </span>
            )}
            {error.actionLink && error.actionLabel && (
              <Link className="btn btn-primary analysis-error-action" to={error.actionLink}>
                {error.actionLabel}
                <Icon name="arrow-right" size={14} />
              </Link>
            )}
          </div>
        </div>
      )}
      {loading && <div className="muted" style={{ marginTop: "var(--space-3)" }}>正在生成...</div>}
      {data && <ChartRender chart={chart} data={data} />}
    </div>
  );
}

function describeParams(chart: VisualizeChart, params: Parameters<typeof edaVisualize>[1]) {
  if (chart === "heatmap") return `字段 ${params.columns?.join("、") ?? ""}`;
  if (chart === "scatter" || chart === "line") return `${params.x ?? ""} × ${params.y ?? ""}`;
  if (chart === "grouped_bar") return `${params.column} × ${params.y}（分组 ${params.group_by}）`;
  if (chart === "qq" || chart === "area") return `字段 ${params.column ?? ""}`;
  return `字段 ${params.column ?? ""}`;
}

function ChartRender({ chart, data }: { chart: VisualizeChart; data: Record<string, unknown> }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <div>
      <ChartActions chart={chart} data={data} containerRef={ref} />
      <div ref={ref} className="chart-frame">
        {renderChart(chart, data)}
      </div>
    </div>
  );
}

function renderChart(chart: VisualizeChart, data: Record<string, unknown>): ReactNode {
  if (chart === "histogram" || chart === "bar") {
    const x = (data.x as string[]) ?? [];
    const y = (data.y as number[]) ?? [];
    const rows = x.map((name, i) => ({ name, value: Number.isFinite(y[i]) ? y[i] : 0 }));
    return (
      <>
        <p className="muted">列「{String(data.column ?? "")}」· 缺失 {String(data.missing ?? 0)}</p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <BarChart data={rows} margin={{ top: 12, right: 20, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} angle={rows.length > 8 ? -20 : 0} height={rows.length > 8 ? 64 : 32} textAnchor={rows.length > 8 ? "end" : "middle"} interval={0} />
            <YAxis fontSize={12} allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill={PRIMARY} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "line") {
    const x = (data.x as (string | number)[]) ?? [];
    const y = (data.y as number[]) ?? [];
    const rows = x.map((name, i) => ({ name: String(name), value: Number.isFinite(y[i]) ? y[i] : 0 }));
    return (
      <>
        <p className="muted">
          {String(data.x_label ?? "")} × {String(data.y_label ?? "")} · 重复 x 按均值聚合
        </p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <LineChart data={rows} margin={{ top: 12, right: 20, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} interval="preserveStartEnd" />
            <YAxis fontSize={12} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke={PRIMARY} dot={rows.length <= 60} />
          </LineChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "scatter") {
    const x = (data.x as number[]) ?? [];
    const y = (data.y as number[]) ?? [];
    const rows = x.map((xv, i) => ({ x: Number(xv), y: Number(y[i]) })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const xLabel = String(data.x_label ?? "x");
    const yLabel = String(data.y_label ?? "y");
    return (
      <>
        <p className="muted">{xLabel} × {yLabel}</p>
        {Boolean(data.sampled) && <p className="muted">数据已下采样展示 {rows.length} 个点</p>}
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <ScatterChart margin={{ top: 12, right: 20, bottom: 16, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" fontSize={12} name={xLabel} />
            <YAxis type="number" dataKey="y" fontSize={12} name={yLabel} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={rows} fill={PRIMARY} />
          </ScatterChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "boxplot") {
    const boxes = (data.boxes as Array<Record<string, number | string | null>>) ?? [];
    if (!boxes.length) return <div className="muted">无可绘制的箱线数据</div>;
    const rows = boxes.map((b) => {
      const low = Number(b.whisker_low ?? b.q1);
      const high = Number(b.whisker_high ?? b.q3);
      const q1 = Number(b.q1);
      const median = Number(b.median);
      return {
        name: String(b.label),
        base: low,
        lowQ1: q1 - low,
        med: median - q1,
        q3High: high - median,
      };
    });
    return (
      <>
        <p className="muted">列「{String(data.column ?? "")}」· Q1 / 中位数 / Q3</p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <BarChart data={rows} margin={{ top: 12, right: 20, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="base" stackId="box" fill="transparent" />
            <Bar dataKey="lowQ1" stackId="box" fill={CHART_INDIGO_LIGHT} />
            <Bar dataKey="med" stackId="box" fill={PRIMARY} />
            <Bar dataKey="q3High" stackId="box" fill={CHART_INDIGO_LIGHT} />
          </BarChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "qq") {
    const xs = (data.x as number[]) ?? [];
    const ys = (data.y as number[]) ?? [];
    const rows = xs.map((xv, i) => ({ x: Number(xv), y: Number(ys[i]) }));
    const line = (data.line as { x0?: number; y0?: number; x1?: number; y1?: number }) ?? {};
    return (
      <>
        <p className="muted">列「{String(data.column ?? "")}」· n={String(data.n ?? 0)} · 红色虚线为理论正态参考线</p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <ScatterChart margin={{ top: 12, right: 20, bottom: 16, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" fontSize={12} name="理论分位" />
            <YAxis type="number" dataKey="y" fontSize={12} name="样本值" />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={rows} fill={PRIMARY} />
            {line.x0 !== undefined && (
              <ReferenceLine
                segment={[
                  { x: Number(line.x0), y: Number(line.y0) },
                  { x: Number(line.x1), y: Number(line.y1) },
                ]}
                stroke={CHART_SERIES[3]}
                strokeDasharray="6 4"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "grouped_bar") {
    const cats = (data.x as string[]) ?? [];
    const groups = (data.groups as string[]) ?? [];
    const series = (data.series as Record<string, (number | null)[]>) ?? {};
    const rows = cats.map((cat, i) => {
      const row: Record<string, number | string | null> = { name: cat };
      for (const g of groups) row[g] = series[g]?.[i] ?? null;
      return row;
    });
    return (
      <>
        <p className="muted">
          {String(data.column_label ?? "")} × {String(data.y_label ?? "")}
          （按 {String(data.group_label ?? "")} 分组）
        </p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <BarChart data={rows} margin={{ top: 12, right: 20, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} interval={0} angle={cats.length > 6 ? -20 : 0} height={cats.length > 6 ? 60 : 30} textAnchor={cats.length > 6 ? "end" : "middle"} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Legend />
            {groups.map((g, i) => (
              <Bar key={g} dataKey={g} fill={CHART_SERIES[i % CHART_SERIES.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </>
    );
  }

  if (chart === "area") {
    const x = (data.x as string[]) ?? [];
    const y = (data.y as number[]) ?? [];
    const rows = x.map((name, i) => ({ name, value: Number.isFinite(y[i]) ? y[i] : 0 }));
    return (
      <>
        <p className="muted">列「{String(data.column ?? "")}」· 累积分布(CDF) · 共 {String(data.total ?? 0)} 个样本</p>
        <ResponsiveContainer width="100%" aspect={ASPECT}>
          <AreaChart data={rows} margin={{ top: 12, right: 20, bottom: 16, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} interval={0} angle={x.length > 8 ? -20 : 0} height={x.length > 8 ? 60 : 30} textAnchor={x.length > 8 ? "end" : "middle"} />
            <YAxis fontSize={12} domain={[0, 1]} />
            <Tooltip />
            <Area type="monotone" dataKey="value" stroke={PRIMARY} fill={PRIMARY} fillOpacity={0.16} />
          </AreaChart>
        </ResponsiveContainer>
      </>
    );
  }

  const cols = Array.isArray(data.columns) ? data.columns.map(String) : [];
  const rawMatrix = data.matrix as Record<string, Record<string, number | null>> | number[][] | undefined;
  const matrix = normalizeMatrix(cols, rawMatrix);

  if (cols.length < 2 || !matrix.length) {
    return <div className="muted">暂无可绘制的相关性矩阵。</div>;
  }

  return (
    <div>
      <p className="muted">相关性热力图（{String(data.method ?? "")}）</p>
      <div className="correlation-heatmap">
        <div
          className="correlation-heatmap-grid"
          style={{ gridTemplateColumns: `minmax(120px, 1.3fr) repeat(${cols.length}, minmax(54px, 1fr))` }}
        >
          <HeatmapCell label />
          {cols.map((col) => <HeatmapCell key={`head-${col}`} label>{col}</HeatmapCell>)}
          {cols.map((rowName, rowIndex) => (
            <div key={rowName} className="correlation-heatmap-row" style={{ display: "contents" }}>
              <HeatmapCell label>{rowName}</HeatmapCell>
              {cols.map((_, colIndex) => {
                const value = matrix[rowIndex]?.[colIndex] ?? null;
                return <HeatmapCell key={`${rowName}-${colIndex}`} value={value} />;
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------
// 下载（PNG / CSV）
// ----------------------------------------------------------------------

function ChartActions({
  chart,
  data,
  containerRef,
}: {
  chart: VisualizeChart;
  data: Record<string, unknown>;
  containerRef: React.RefObject<HTMLDivElement>;
}) {
  function download(href: string, filename: string) {
    const a = document.createElement("a");
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function toPng() {
    const svg = containerRef.current?.querySelector("svg");
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const xml = new XMLSerializer().serializeToString(clone);
    const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
    const img = new Image();
    img.onload = () => {
      const scale = 2;
      const w = (svg.clientWidth || 800) * scale;
      const h = (svg.clientHeight || 450) * scale;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((blob) => {
        if (blob) download(URL.createObjectURL(blob), `${chart}-${Date.now()}.png`);
      });
    };
    img.src = url;
  }

  function toCsv() {
    const rows = toCsvRows(chart, data);
    const content = rows
      .map((r) => r.map((c) => String(c).replace(/"/g, '""')).join(","))
      .join("\n");
    const blob = new Blob(["\ufeff" + content], { type: "text/csv;charset=utf-8" });
    download(URL.createObjectURL(blob), `${chart}-${Date.now()}.csv`);
  }

  return (
    <div className="visualization-download">
      <button className="btn" type="button" onClick={toPng}>下载 PNG</button>
      <button className="btn" type="button" onClick={toCsv}>下载 CSV</button>
    </div>
  );
}

function toCsvRows(chart: VisualizeChart, data: Record<string, unknown>): string[][] {
  if (chart === "heatmap") {
    const cols = (data.columns as (string | number)[]) ?? [];
    const matrix = data.matrix as Record<string, Record<string, number | null>> | undefined;
    const head = ["", ...cols.map(String)];
    const body = cols.map((r) => [
      String(r),
      ...cols.map((c) => {
        const v = matrix?.[String(r)]?.[String(c)];
        return v === null || v === undefined ? "" : String(v);
      }),
    ]);
    return [head, ...body];
  }
  if (chart === "boxplot") {
    const boxes = (data.boxes as Array<Record<string, unknown>>) ?? [];
    const head = ["label", "q1", "median", "q3", "whisker_low", "whisker_high"];
    const body = boxes.map((b) => [
      String(b.label ?? ""),
      String(b.q1 ?? ""),
      String(b.median ?? ""),
      String(b.q3 ?? ""),
      String(b.whisker_low ?? ""),
      String(b.whisker_high ?? ""),
    ]);
    return [head, ...body];
  }
  if (chart === "qq") {
    const xs = (data.x as (number | string)[]) ?? [];
    const ys = (data.y as (number | string)[]) ?? [];
    return [["theoretical", "sample"], ...xs.map((x, i) => [String(x), String(ys[i])])];
  }
  if (chart === "grouped_bar") {
    const cats = (data.x as (string | number)[]) ?? [];
    const groups = (data.groups as string[]) ?? [];
    const series = (data.series as Record<string, (number | null)[]>) ?? {};
    const head = ["category", ...groups];
    const body = cats.map((cat, i) => [
      String(cat),
      ...groups.map((g) => {
        const v = series[g]?.[i];
        return v === null || v === undefined ? "" : String(v);
      }),
    ]);
    return [head, ...body];
  }
  if (chart === "area" || chart === "histogram" || chart === "bar") {
    const x = (data.x as (string | number)[]) ?? [];
    const y = (data.y as (number | string)[]) ?? [];
    const label = chart === "area" ? "bin" : "label";
    return [[label, "value"], ...x.map((name, i) => [String(name), String(y[i])])];
  }
  // line / scatter
  const x = (data.x as (number | string)[]) ?? [];
  const y = (data.y as (number | string)[]) ?? [];
  return [["x", "y"], ...x.map((name, i) => [String(name), String(y[i])])];
}

// ----------------------------------------------------------------------
// 热力图单元格
// ----------------------------------------------------------------------

function HeatmapCell({ children, value, label = false }: { children?: ReactNode; value?: number | null; label?: boolean }) {
  const background = label ? undefined : heatColor(value);
  return (
    <div className={`correlation-heatmap-cell${label ? " label" : ""}`} style={background ? { background } : undefined}>
      {label ? children : value === null || value === undefined ? "-" : value.toFixed(2)}
    </div>
  );
}

function normalizeMatrix(
  columns: string[],
  matrix: Record<string, Record<string, number | null>> | number[][] | undefined,
): Array<Array<number | null>> {
  if (!matrix) return [];
  if (Array.isArray(matrix)) {
    return columns.map((_, row) => columns.map((__, col) => {
      const value = matrix[row]?.[col];
      return typeof value === "number" && Number.isFinite(value) ? value : null;
    }));
  }
  return columns.map((row) => columns.map((col) => {
    const value = matrix[row]?.[col];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }));
}

function heatColor(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return CHART_NULL;
  const t = Math.min(1, Math.max(0, (v + 1) / 2));
  const [pr, pg, pb] = CHART_PRIMARY_RGB;
  const r = Math.round(255 + (pr - 255) * t);
  const g = Math.round(255 + (pg - 255) * t);
  const b = Math.round(255 + (pb - 255) * t);
  return `rgb(${r}, ${g}, ${b})`;
}
