import { useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../../components/PageHeader";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { edaCorrelation, edaDescriptive, edaDistribution, edaOutlier, type VisualizeChart } from "../../api/analysis";
import { getSchema, previewDataset } from "../../api/datasets";
import { formatError, withPrefix, type AnalysisError } from "../../lib/analysisError";
import type { SchemaColumn } from "../../types/dataset";
import { ProfilePanel } from "../../features/eda/ProfilePanel";
import { CorrelationPanel } from "../../features/eda/CorrelationPanel";
import { DistributionChart } from "../../features/eda/DistributionChart";
import { OutlierPanel } from "../../features/eda/OutlierPanel";
import { VisualizationPanel, isNumericColumn, isTemporalColumn } from "../../features/eda/VisualizationPanel";
import { isContinuousNumeric } from "../../lib/edaColumns";
import { PreviewTable } from "../../features/dataset/PreviewTable";
import { Skeleton } from "../../components/StateBlock";
import { ErrorNotice } from "../../components/ErrorNotice";
import { Icon, type IconName } from "../../components/icons/Icon";

/** 把一次子分析的失败包成「带上下文前缀」的标准错误对象。 */
function sectionError(error: unknown, label: string): AnalysisError {
  return withPrefix(formatError(error), `${label}：`);
}

/* 数据分析页的空状态：带插画图标 + 明确的引导文案与下一步动作，
   替代原先只有一句灰字的「暂无数据 / 至少需要 N 个字段」。 */
function AnalysisEmpty({ icon, title, hint, action }: { icon: IconName; title: string; hint: string; action?: React.ReactNode }) {
  return (
    <div className="analysis-empty">
      <span className="analysis-empty-illustration" aria-hidden="true">
        <Icon name={icon} size={28} />
      </span>
      <strong className="analysis-empty-title">{title}</strong>
      <span className="analysis-empty-hint">{hint}</span>
      {action ? <div className="analysis-empty-action">{action}</div> : null}
    </div>
  );
}

type AnalysisTab = "preview" | "profile" | "correlation" | "distribution" | "outlier" | "visualization";
const ANALYSIS_TABS: Array<{ key: AnalysisTab; label: string }> = [
  { key: "preview", label: "数据预览" }, { key: "profile", label: "描述统计" },
  { key: "correlation", label: "相关性" }, { key: "distribution", label: "分布" },
  { key: "outlier", label: "异常值" }, { key: "visualization", label: "可视化探索" },
];

export default function Analysis() {
  const [datasetIds, setDatasetIds] = useState<number[]>([]);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  // /schema 返回的行数：低基数判定（isContinuousNumeric）需要它才能与后端口径对齐。
  const [rowCount, setRowCount] = useState(0);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [distributionColumn, setDistributionColumn] = useState("");
  const [corrMethod, setCorrMethod] = useState<"pearson" | "spearman" | "auto">("auto");
  const [chart, setChart] = useState<VisualizeChart>("histogram");
  const [activeTab, setActiveTab] = useState<AnalysisTab>("preview");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<AnalysisError[]>([]);
  const [profile, setProfile] = useState<Parameters<typeof ProfilePanel>[0]["data"]>(null);
  const [corr, setCorr] = useState<Parameters<typeof CorrelationPanel>[0]["data"]>(null);
  const [dist, setDist] = useState<Parameters<typeof DistributionChart>[0]["data"]>(null);
  const [outlier, setOutlier] = useState<Parameters<typeof OutlierPanel>[0]["data"]>(null);
  const [sampleRows, setSampleRows] = useState<Record<string, unknown>[]>([]);

  // 当前 datasetId 的镜像：runAll 里 await 结束后用它判断响应是否已过期。
  const datasetIdRef = useRef<number | undefined>(undefined);
  datasetIdRef.current = datasetIds[0];

  const datasetId = datasetIds[0];
  const colNames = useMemo(() => columns.map((c) => c.column), [columns]);
  // numericNames：dtype 事实（指标卡「数值字段」展示用，含 Month 这类编码列）。
  const numericNames = useMemo(() => columns.filter(isNumericColumn).map((c) => c.column), [columns]);
  // continuousNames：与后端 continuous_columns 同口径的「连续数值列」，
  // 用于「能不能做相关性」这类需要与后端判据一致的场景。
  const continuousNames = useMemo(
    () => columns.filter((c) => isContinuousNumeric(c, rowCount)).map((c) => c.column),
    [columns, rowCount],
  );
  const temporalNames = useMemo(() => columns.filter(isTemporalColumn).map((c) => c.column), [columns]);
  const categoricalNames = useMemo(() => columns.filter((c) => !isNumericColumn(c) && !isTemporalColumn(c)).map((c) => c.column), [columns]);
  // 只保留仍存在于当前数据集 schema 中的已选列。
  // 切换数据集后 setSelectedColumns([]) 虽然是同步的，但 columns 要到 getSchema
  // 返回才更新；若此刻仍带着旧数据集的列名发请求，后端会回 422「指定的字段不存在」。
  // 这里以「当前 schema」为准做一次过滤，保证任何请求都不携带失效列名。
  const validSelectedColumns = useMemo(
    () => {
      const known = new Set(colNames);
      return selectedColumns.filter((name) => known.has(name));
    },
    [colNames, selectedColumns],
  );
  const selectedColsParam = validSelectedColumns.length ? validSelectedColumns : undefined;
  const selectedNumericCount = validSelectedColumns.length ? validSelectedColumns.filter((name) => continuousNames.includes(name)).length : continuousNames.length;

  useEffect(() => {
    if (!datasetId) {
      setColumns([]); setRowCount(0); setSelectedColumns([]); setDistributionColumn(""); setCorrMethod("auto"); setSampleRows([]); setActiveTab("preview");
      return;
    }
    setSelectedColumns([]); setError([]); setProfile(null); setCorr(null); setDist(null); setOutlier(null); setCorrMethod("auto"); setActiveTab("preview");
    getSchema(datasetId).then((schema) => {
      setColumns(schema.columns);
      setRowCount(schema.row_count);
      const numeric = schema.columns.filter(isNumericColumn).map((c) => c.column);
      const temporal = schema.columns.filter(isTemporalColumn).map((c) => c.column);
      const categorical = schema.columns.filter((c) => !isNumericColumn(c) && !isTemporalColumn(c)).map((c) => c.column);
      setDistributionColumn(numeric[0] ?? schema.columns[0]?.column ?? "");
      let recommended: VisualizeChart = "histogram";
      if (temporal.length && numeric.length) recommended = "line";
      else if (numeric.length > 2) recommended = "heatmap";
      else if (numeric.length === 2) recommended = "scatter";
      else if (categorical.length) recommended = "bar";
      setChart(recommended);
    }).catch(() => setColumns([]));
    previewDataset(datasetId, { page: 1, page_size: 50 }).then((data) => setSampleRows(data.items)).catch(() => setSampleRows([]));
  }, [datasetId]);

  // 切换数据集时，若旧的分布字段不在新 schema 中则清空，避免继续用失效列名请求。
  useEffect(() => {
    if (!columns.length) return;
    setDistributionColumn((current) => (current && !colNames.includes(current) ? "" : current));
  }, [colNames, columns.length]);

  function toggleColumn(name: string) { setSelectedColumns((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name]); }
  function selectAllColumns() { setSelectedColumns(colNames); }
  function clearColumns() { setSelectedColumns([]); }

  async function runAll() {
    if (!datasetId) return;
    setError([]); setProfile(null); setCorr(null); setDist(null); setOutlier(null);
    // 冻结本次运行的目标数据集：await 期间用户可能切换数据集，
    // 若不加这个守卫，切回时会用新 datasetId 去解释旧响应，或把旧错误写进新数据集的界面。
    const targetDatasetId = datasetId;
    const stale = () => targetDatasetId !== datasetIdRef.current;
    const errors: AnalysisError[] = [];
    const cols = selectedColsParam;
    const distColumn = distributionColumn.trim();
    // 与后端 continuous_columns 同口径：只有「连续数值列」才可能参与相关性，
    // 低基数编码列（Month/DayOfWeek 等）不算，勾选它们不会把相关性分析带起来。
    const count = validSelectedColumns.length ? validSelectedColumns.filter((name) => continuousNames.includes(name)).length : continuousNames.length;
    try { setBusy("描述性统计"); const data = await edaDescriptive(targetDatasetId, { columns: cols }); if (!stale()) setProfile(data as never); }
    catch (e) { if (!stale()) errors.push(sectionError(e, "描述性统计")); }
    // strict=true：相关性只用显式勾选的字段，后端不再静默补齐其它列；
    // 勾选里有效数值列不足 2 个时后端返回中文 422（说明哪列因低基数被剔除）。
    if (count >= 2) { try { setBusy("相关性分析"); const data = await edaCorrelation(targetDatasetId, { columns: cols, method: corrMethod, strict: true }); if (!stale()) setCorr(data as never); } catch (e) { if (!stale()) errors.push(sectionError(e, "相关性分析")); } }
    if (distColumn) { try { setBusy("分布分析"); const data = await edaDistribution(targetDatasetId, distColumn); if (!stale()) setDist(data as never); } catch (e) { if (!stale()) errors.push(sectionError(e, "分布分析")); } }
    if (count >= 1) { try { setBusy("异常值分析"); const data = await edaOutlier(targetDatasetId, { columns: cols }); if (!stale()) setOutlier(data as never); } catch (e) { if (!stale()) errors.push(sectionError(e, "异常值分析")); } }
    if (stale()) return;
    setBusy(null); if (errors.length) setError(errors); else if (activeTab === "preview") setActiveTab("profile");
  }

  function renderResult() {
    if (!datasetId) return null;
    switch (activeTab) {
      case "preview": return <><div className="analysis-result-header"><div><h3 style={{ margin: 0 }}>数据预览</h3><div className="muted">快速检查样本与字段，不修改数据版本。</div></div></div><div className="analysis-preview-wrap"><PreviewTable datasetId={datasetId} pageSize={20} columns={selectedColsParam} /></div></>;
      case "profile": return busy === "描述性统计" ? <Skeleton lines={3} /> : <ProfilePanel data={profile} />;
      case "correlation": return selectedNumericCount < 2 ? <AnalysisEmpty icon="chart" title="还差一个数值字段" hint="相关性分析需要至少 2 个数值字段。当前数据集里的数值字段不够，去左侧勾选更多字段，或换个数据更完整的数据集。" /> : busy === "相关性分析" ? <Skeleton lines={3} /> : <CorrelationPanel data={corr} scatterPoints={sampleRows} />;
      case "distribution": return busy === "分布分析" ? <Skeleton lines={3} /> : <DistributionChart data={dist} />;
      case "outlier": return busy === "异常值分析" ? <Skeleton lines={3} /> : <OutlierPanel data={outlier} />;
      case "visualization": return <VisualizationPanel datasetId={datasetId} columns={columns} selectedColumns={validSelectedColumns} chart={chart} onChartChange={setChart} rowCount={rowCount} />;
    }
  }

  return (
    <div>
      <PageHeader
        breadcrumbs={<>数据中心 / <b>数据分析</b></>}
        title="数据分析"
        description="勾选字段后运行分析；可视化页签随字段与图表类型自动刷新。"
      />

      <section className="card analysis-toolbar">
        <div className="analysis-toolbar-head">
          <strong>当前数据</strong>
        </div>
        <DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} bare />
      </section>

      {datasetId && (
        <section className="analysis-metrics">
          <Metric label="字段" value={String(columns.length)} />
          <Metric label="数值字段" value={String(numericNames.length)} />
          <Metric label="类别/其他" value={String(categoricalNames.length)} />
          <Metric label="时间字段" value={String(temporalNames.length)} />
        </section>
      )}

      <div className="analysis-workspace">
        <aside className="analysis-fields">
          <div className="analysis-fields-title">字段与范围</div>
          {columns.length ? <>
            <div className="analysis-field-actions"><button className="btn" type="button" onClick={selectAllColumns}>全选</button><button className="btn" type="button" onClick={clearColumns}>清空</button><span className="muted" style={{ marginLeft: "auto", fontSize: 12 }}>{selectedColumns.length ? `${selectedColumns.length} 已选` : "全部"}</span></div>
            <div className="analysis-field-list">{columns.map((item) => <label className="analysis-field-item" key={item.column}><input type="checkbox" checked={selectedColumns.includes(item.column)} onChange={() => toggleColumn(item.column)} /><span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{item.column}</span><span className="analysis-field-type">{item.dtype}</span></label>)}</div>
            <label className="field" style={{ marginTop: 14 }}>分布字段<select value={distributionColumn} onChange={(e) => setDistributionColumn(e.target.value)}><option value="">不运行分布</option>{colNames.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
            <label className="field" style={{ marginTop: 10 }}>相关性方法<select value={corrMethod} onChange={(e) => setCorrMethod(e.target.value as "pearson" | "spearman" | "auto")}><option value="auto">自动（按列类型）</option><option value="pearson">Pearson（线性）</option><option value="spearman">Spearman（秩相关）</option></select></label>
            <button className="btn primary analysis-run" type="button" disabled={!datasetId || busy !== null} onClick={() => void runAll()}>{busy ? `正在${busy}…` : "运行分析"}</button>
          </> : <div className="analysis-empty" style={{ minHeight: 180, padding: "var(--space-3)" }}>{datasetId ? "正在读取字段…" : "选择数据集后显示字段。"}</div>}
          {error.length > 0 && (
            <div style={{ display: "grid", gap: "var(--space-2)", marginTop: "var(--space-3)" }}>
              {error.map((item, index) => (
                <ErrorNotice key={`${item.what}-${index}`} error={item} title="分析没能全部完成" compact />
              ))}
            </div>
          )}
        </aside>

        <section className="analysis-results">
          <div className="analysis-tabs" role="tablist" aria-label="EDA 结果页面">{ANALYSIS_TABS.map((tab) => <button key={tab.key} type="button" role="tab" aria-selected={activeTab === tab.key} className={`analysis-tab${activeTab === tab.key ? " active" : ""}`} onClick={() => setActiveTab(tab.key)}>{tab.label}</button>)}</div>
          {activeTab === "preview" && !profile && !corr && !dist && !outlier && <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", padding: "12px 0" }}><button className="btn" type="button" onClick={() => setActiveTab("visualization")} disabled={!datasetId}>开始可视化</button></div>}
          <div className="analysis-result-header"><div><h3 style={{ margin: 0 }}>{ANALYSIS_TABS.find((tab) => tab.key === activeTab)?.label ?? "分析结果"}</h3></div>{selectedNumericCount >= 2 && <span className="badge">{selectedNumericCount} 个数值字段可用于相关性</span>}</div>
          {renderResult()}
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="card" style={{ padding: "12px 14px" }}><div className="muted" style={{ fontSize: 12 }}>{label}</div><strong style={{ display: "block", marginTop: "var(--space-1)", fontSize: 18 }}>{value}</strong></div>; }
