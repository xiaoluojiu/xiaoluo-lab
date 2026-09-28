import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { getDataset, getProfile, getQuality, getSchema, getVersionTimeline } from "../../api/datasets";
import type { Dataset, ProfileData, QualityData, SchemaColumn, VersionTimelineEntry } from "../../types/dataset";
import { ErrorState } from "../../components/Loading";
import { Skeleton } from "../../components/StateBlock";
import { PreviewTable } from "../../features/dataset/PreviewTable";
import { VersionTimeline } from "../../features/dataset/VersionTimeline";
import { PageHeader } from "../../components/PageHeader";
import { useGlobalContext } from "../../store/globalStore";

type Tab = "overview" | "preview" | "schema" | "profile" | "quality" | "versions";
/** 合法页签，供 `?tab=` 深链校验（形状与 Settings 的 section 白名单一致）。 */
const DETAIL_TABS: Tab[] = ["overview", "preview", "schema", "profile", "quality", "versions"];

/**
 * 按版本缓存的一份页签数据。
 *
 * 为什么要带 `version`：schema / profile / quality 都可以按版本查询，
 * 换了版本必须重取；但把它们直接塞进 effect 依赖又会「取一次 → setState → 依赖变化 → 再取」
 * 无限循环。把「这份数据属于哪个版本」一并存下来，就能用一个纯比较做缓存判断。
 */
interface VersionedData<T> {
  version: number | null;
  data: T | null;
}

export default function DatasetDetail() {
  const { id } = useParams();
  const datasetId = Number(id);
  // 支持 /datasets/:id?tab=versions 直达页签 —— 数据处理页执行成功后的
  // 「查看新版本」链接用的就是它。挂载时读一次即可：跨路由进入本页必然重新挂载。
  const [searchParams] = useSearchParams();
  const requestedTab = searchParams.get("tab");
  const [tab, setTab] = useState<Tab>(
    DETAIL_TABS.includes(requestedTab as Tab) ? (requestedTab as Tab) : "overview",
  );
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [versions, setVersions] = useState<VersionTimelineEntry[]>([]);
  const [schema, setSchema] = useState<VersionedData<SchemaColumn[]> | null>(null);
  const [profile, setProfile] = useState<VersionedData<ProfileData> | null>(null);
  const [quality, setQuality] = useState<VersionedData<QualityData> | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 跨页面数据上下文：本页既是「当前数据集」的声明处，也是「当前版本」的选择处。
  const selectDataset = useGlobalContext((s) => s.selectDataset);
  const selectVersion = useGlobalContext((s) => s.selectVersion);
  const contextDatasetId = useGlobalContext((s) => s.currentDatasetId);
  const contextVersionId = useGlobalContext((s) => s.currentVersionId);

  /**
   * 只认「属于当前数据集」的版本号。
   * 进入 B 的详情页时，store 里可能还残留着 A 的版本（下面的 effect 尚未执行），
   * 不加这层判断就会把 A 的版本号显示出来、甚至带到 B 上去。
   */
  const selectedVersion = contextDatasetId === datasetId ? contextVersionId : null;

  useEffect(() => {
    if (!Number.isInteger(datasetId) || datasetId <= 0) {
      setError("无效的数据集 ID");
      return;
    }
    let cancelled = false;
    setError(null);
    setDataset(null);
    setSchema(null);
    setProfile(null);
    setQuality(null);
    void getDataset(datasetId)
      .then((d) => {
        if (cancelled) return;
        setDataset(d);
        // 声明「当前数据集」到全局上下文。不传版本号 = 同一数据集保留已选版本、换数据集回到最新版本，
        // 这样从 EDA / ML 再跳回来时，用户之前选的版本不会被冲掉。
        selectDataset(datasetId);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "加载失败"));
    // 版本下拉的可选项来自既有的版本时间线接口（与「版本」页签同源，不另造一份数据）。
    void getVersionTimeline(datasetId)
      .then((t) => !cancelled && setVersions(t.versions))
      .catch(() => !cancelled && setVersions([]));
    return () => {
      cancelled = true;
    };
  }, [datasetId, selectDataset]);

  // schema / profile / quality 跟随所选版本：缓存里记录的版本与当前版本不一致时才重取。
  useEffect(() => {
    if (!dataset) return;
    const version = selectedVersion ?? null;
    const versionArg = selectedVersion ?? undefined;
    if (tab === "schema" && schema?.version !== version) {
      void getSchema(datasetId, versionArg).then((r) => setSchema({ version, data: r.columns })).catch(() => setSchema({ version, data: null }));
    }
    if (tab === "profile" && profile?.version !== version) {
      void getProfile(datasetId, versionArg).then((r) => setProfile({ version, data: r })).catch(() => setProfile({ version, data: null }));
    }
    if (tab === "quality" && quality?.version !== version) {
      void getQuality(datasetId, versionArg).then((r) => setQuality({ version, data: r })).catch(() => setQuality({ version, data: null }));
    }
  }, [tab, datasetId, dataset, selectedVersion, schema, profile, quality]);

  const latestVersion = dataset?.latest_version ?? null;
  const selectedEntry = versions.find((item) => item.version === selectedVersion) ?? null;
  // 规模 / 格式优先反映「选中的那个版本」，未选中具体版本时回落到最新版本。
  const shownRowCount = selectedEntry?.row_count ?? latestVersion?.row_count ?? 0;
  const shownColumnCount = selectedEntry?.column_count ?? latestVersion?.column_count ?? 0;
  const shownFormat = selectedEntry?.format ?? latestVersion?.format ?? "-";
  const versionLabel = selectedVersion !== null ? `v${selectedVersion}` : latestVersion ? `v${latestVersion.version}` : "暂无版本";

  // 深链同时保留版本号：既是可分享的 URL，也让刷新后能经由路由入口的同步恢复同一上下文。
  const contextQuery = selectedVersion !== null ? `?dataset=${datasetId}&version=${selectedVersion}` : `?dataset=${datasetId}`;

  /** 跳转前把上下文写进 globalStore（路由入口的 URL 同步只是兜底恢复，这里才是主路径）。 */
  function publishContext() {
    selectDataset(datasetId, selectedVersion);
  }

  const actions = useMemo(() => [
    { path: "/processing", label: "处理", description: "清洗、筛选、转换并创建新版本" },
    { path: "/analysis", label: "分析", description: "描述统计、相关性、分布与可视化" },
    { path: "/ml", label: "机器学习", description: "将当前数据用于模型实验" },
    { path: "/workflow", label: "Workflow", description: "把数据接入可复用流程" },
    { path: "/ai", label: "AI 实验室", description: "让 AI 基于当前数据进行分析" },
  ], []);

  if (error) return <ErrorState message={error} onRetry={() => location.reload()} />;
  if (!dataset) return <Skeleton lines={4} card />;

  const tabs: [Tab, string][] = [
    ["overview", "概览"], ["preview", "预览"], ["schema", "Schema"],
    ["profile", "Profile"], ["quality", "质量"], ["versions", "版本"],
  ];

  const qualityIssues = quality?.data?.statistics.issue_count ?? 0;

  return (
    <div className="dataset-workspace">
      <PageHeader
        title={dataset.name}
        description={dataset.description || "暂无描述。"}
        breadcrumbs={<Link to="/datasets" className="muted">← 数据中心</Link>}
        actions={<span className="badge">Dataset #{dataset.id}</span>}
      />

      <section className="dataset-context-bar card" style={{ padding: 14, marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
          <div className="field" style={{ marginBottom: 0 }}>
            <div className="muted" style={{ fontSize: 11 }}>当前版本（带到后续页面）</div>
            <select
              aria-label="选择当前数据版本"
              value={selectedVersion ?? ""}
              onChange={(event) => {
                const raw = event.target.value;
                selectVersion(raw === "" ? null : Number(raw));
              }}
              style={{ minHeight: 32, padding: "4px 8px" }}
            >
              <option value="">最新版本{latestVersion ? `（v${latestVersion.version}）` : ""}</option>
              {/* 兜底：版本时间线还没回来（或加载失败）时，已选版本也必须在选项里，
                  否则 select 的 value 找不到对应 option，界面会显示成「最新版本」。 */}
              {selectedVersion !== null && !versions.some((item) => item.version === selectedVersion) && (
                <option value={selectedVersion}>v{selectedVersion}</option>
              )}
              {versions.map((item) => (
                <option key={item.version} value={item.version}>v{item.version}</option>
              ))}
            </select>
          </div>
          <div><div className="muted" style={{ fontSize: 11 }}>规模</div><strong>{shownRowCount} × {shownColumnCount}</strong></div>
          <div><div className="muted" style={{ fontSize: 11 }}>格式</div><strong>{shownFormat}</strong></div>
          <div><div className="muted" style={{ fontSize: 11 }}>数据质量</div><strong>{qualityLabel(quality?.data ?? null)}</strong></div>
          <div style={{ marginLeft: "auto" }}><button className="btn btn-primary" type="button" onClick={() => setTab("preview")}>查看数据</button></div>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          当前上下文：数据集 #{datasetId} · {versionLabel}。前往分析 / 机器学习 / 实验 / 报告时会沿用这一选择。
        </div>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 10, marginBottom: "var(--space-4)" }}>
        {actions.map((action) => (
          <Link
            key={action.path}
            to={`${action.path}${contextQuery}`}
            className="card dataset-action-card"
            style={{ textDecoration: "none", color: "inherit", padding: 14 }}
            onClick={publishContext}
          >
            <strong>{action.label}</strong>
            <div className="muted" style={{ fontSize: 12, lineHeight: 1.5, marginTop: 6 }}>{action.description}</div>
          </Link>
        ))}
      </section>

      <div className="dataset-tabs" role="tablist" aria-label="数据集工作区">
        {tabs.map(([key, label]) => (
          <button key={key} className={`dataset-tab${tab === key ? " active" : ""}`} role="tab" aria-selected={tab === key} onClick={() => setTab(key)} type="button">{label}</button>
        ))}
      </div>

      {tab === "overview" && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.4fr) minmax(280px, .6fr)", gap: 14 }}>
          <section className="card">
            <h3 style={{ marginTop: 0 }}>数据预览</h3>
            {latestVersion ? <PreviewTable datasetId={datasetId} version={selectedVersion ?? undefined} pageSize={12} /> : <div className="muted">还没有可预览的数据版本。</div>}
          </section>
          <section className="card">
            <h3 style={{ marginTop: 0 }}>下一步</h3>
            <p className="muted">先理解数据，再决定是否处理。处理不会覆盖当前版本。</p>
            <div style={{ display: "grid", gap: "var(--space-2)" }}>
              <button className="btn" type="button" onClick={() => setTab("schema")}>查看字段结构</button>
              <button className="btn" type="button" onClick={() => setTab("quality")}>检查数据质量</button>
              <button className="btn" type="button" onClick={() => setTab("profile")}>查看数据概况</button>
            </div>
          </section>
        </div>
      )}

      {tab === "preview" && <div className="card"><PreviewTable datasetId={datasetId} version={selectedVersion ?? undefined} /></div>}

      {tab === "schema" && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}><div><h3 style={{ margin: 0 }}>字段结构</h3><div className="muted" style={{ marginTop: "var(--space-1)" }}>{schema?.data?.length ?? "-"} 个字段 · {versionLabel}</div></div></div>
          {schema?.data ? <table className="data-table"><thead><tr><th>列名</th><th>类型</th></tr></thead><tbody>{schema.data.map((c) => <tr key={c.column}><td>{c.column}</td><td>{c.dtype}</td></tr>)}</tbody></table> : <Skeleton lines={3} />}
        </div>
      )}

      {tab === "profile" && (
        <div className="card"><h3 style={{ marginTop: 0 }}>数据概况</h3><div className="muted" style={{ marginBottom: "var(--space-2)" }}>{versionLabel}</div>{profile?.data ? <pre style={{ fontSize: 12, overflow: "auto", maxHeight: 520, margin: 0 }}>{JSON.stringify(profile.data, null, 2)}</pre> : <Skeleton lines={3} />}</div>
      )}

      {tab === "quality" && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}><div><h3 style={{ margin: 0 }}>数据质量</h3><div className="muted" style={{ marginTop: "var(--space-1)" }}>发现的问题只影响当前检查，不会自动修改数据。</div></div><span className={`badge ${qualityIssues ? "failed" : "success"}`}>{quality?.data ? qualityLabel(quality.data) : "检查中"}</span></div>
          {quality?.data ? <>{quality.data.issues.length ? <table className="data-table"><thead><tr><th>类型</th><th>列</th><th>说明</th></tr></thead><tbody>{quality.data.issues.map((issue, i) => <tr key={i}><td>{String(issue.type ?? "-")}</td><td>{String(issue.column ?? "-")}</td><td>{String(issue.message ?? JSON.stringify(issue))}</td></tr>)}</tbody></table> : <div className="empty-state"><strong>没有发现质量问题</strong><div className="muted">当前检查范围内数据状态正常。</div></div>}</> : <Skeleton lines={3} />}
        </div>
      )}

      {tab === "versions" && (
        <div className="card">
          <VersionTimeline datasetId={datasetId} />
        </div>
      )}
    </div>
  );
}

function qualityLabel(quality: QualityData | null): string {
  if (!quality) return "未检查";
  const issues = quality.statistics.issue_count ?? 0;
  return issues === 0 ? "良好" : `${issues} 个问题`;
}
