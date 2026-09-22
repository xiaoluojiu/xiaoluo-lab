import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getDataset, getProfile, getQuality, getSchema } from "../../api/datasets";
import type { Dataset, ProfileData, QualityData, SchemaColumn } from "../../types/dataset";
import { ErrorState, Loading } from "../../components/Loading";
import { PreviewTable } from "../../features/dataset/PreviewTable";
import { PageHeader } from "../../components/PageHeader";

type Tab = "overview" | "preview" | "schema" | "profile" | "quality" | "versions";

export default function DatasetDetail() {
  const { id } = useParams();
  const datasetId = Number(id);
  const [tab, setTab] = useState<Tab>("overview");
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [schema, setSchema] = useState<SchemaColumn[] | null>(null);
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [quality, setQuality] = useState<QualityData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isInteger(datasetId) || datasetId <= 0) {
      setError("无效的数据集 ID");
      return;
    }
    setError(null);
    setDataset(null);
    void getDataset(datasetId)
      .then(setDataset)
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"));
  }, [datasetId]);

  useEffect(() => {
    if (!dataset) return;
    if (tab === "schema" && !schema) void getSchema(datasetId).then((r) => setSchema(r.columns)).catch(() => null);
    if (tab === "profile" && !profile) void getProfile(datasetId).then(setProfile).catch(() => null);
    if (tab === "quality" && !quality) void getQuality(datasetId).then(setQuality).catch(() => null);
  }, [tab, datasetId, dataset, schema, profile, quality]);

  const v = dataset?.latest_version;
  const qualityIssues = quality?.statistics.issue_count ?? 0;
  const qualityState = quality ? (qualityIssues === 0 ? "良好" : `${qualityIssues} 个问题`) : "未检查";

  const actions = useMemo(() => [
    { href: `/processing?dataset=${datasetId}`, label: "处理", description: "清洗、筛选、转换并创建新版本" },
    { href: `/analysis?dataset=${datasetId}`, label: "分析", description: "描述统计、相关性、分布与可视化" },
    { href: `/ml?dataset=${datasetId}`, label: "机器学习", description: "将当前数据用于模型实验" },
    { href: `/workflow?dataset=${datasetId}`, label: "Workflow", description: "把数据接入可复用流程" },
    { href: `/ai?dataset=${datasetId}`, label: "AI 实验室", description: "让 AI 基于当前数据进行分析" },
  ], [datasetId]);

  if (error) return <ErrorState message={error} onRetry={() => location.reload()} />;
  if (!dataset) return <Loading />;

  const tabs: [Tab, string][] = [
    ["overview", "概览"], ["preview", "预览"], ["schema", "Schema"],
    ["profile", "Profile"], ["quality", "质量"], ["versions", "版本"],
  ];

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
          <div><div className="muted" style={{ fontSize: 11 }}>当前版本</div><strong>{v ? `v${v.version}` : "暂无版本"}</strong></div>
          <div><div className="muted" style={{ fontSize: 11 }}>规模</div><strong>{v ? `${v.row_count} × ${v.column_count}` : "-"}</strong></div>
          <div><div className="muted" style={{ fontSize: 11 }}>格式</div><strong>{v?.format || "-"}</strong></div>
          <div><div className="muted" style={{ fontSize: 11 }}>数据质量</div><strong>{qualityState}</strong></div>
          <div style={{ marginLeft: "auto" }}><button className="btn btn-primary" type="button" onClick={() => setTab("preview")}>查看数据</button></div>
        </div>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 10, marginBottom: "var(--space-4)" }}>
        {actions.map((action) => (
          <Link key={action.href} to={action.href} className="card dataset-action-card" style={{ textDecoration: "none", color: "inherit", padding: 14 }}>
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
            {v ? <PreviewTable datasetId={datasetId} pageSize={12} /> : <div className="muted">还没有可预览的数据版本。</div>}
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

      {tab === "preview" && <div className="card"><PreviewTable datasetId={datasetId} /></div>}

      {tab === "schema" && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}><div><h3 style={{ margin: 0 }}>字段结构</h3><div className="muted" style={{ marginTop: "var(--space-1)" }}>{schema?.length ?? "-"} 个字段</div></div></div>
          {schema ? <table className="data-table"><thead><tr><th>列名</th><th>类型</th></tr></thead><tbody>{schema.map((c) => <tr key={c.column}><td>{c.column}</td><td>{c.dtype}</td></tr>)}</tbody></table> : <Loading />}
        </div>
      )}

      {tab === "profile" && (
        <div className="card"><h3 style={{ marginTop: 0 }}>数据概况</h3>{profile ? <pre style={{ fontSize: 12, overflow: "auto", maxHeight: 520, margin: 0 }}>{JSON.stringify(profile, null, 2)}</pre> : <Loading />}</div>
      )}

      {tab === "quality" && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}><div><h3 style={{ margin: 0 }}>数据质量</h3><div className="muted" style={{ marginTop: "var(--space-1)" }}>发现的问题只影响当前检查，不会自动修改数据。</div></div><span className={`badge ${qualityIssues ? "failed" : "success"}`}>{quality ? qualityState : "检查中"}</span></div>
          {quality ? <>{quality.issues.length ? <table className="data-table"><thead><tr><th>类型</th><th>列</th><th>说明</th></tr></thead><tbody>{quality.issues.map((issue, i) => <tr key={i}><td>{String(issue.type ?? "-")}</td><td>{String(issue.column ?? "-")}</td><td>{String(issue.message ?? JSON.stringify(issue))}</td></tr>)}</tbody></table> : <div className="empty-state"><strong>没有发现质量问题</strong><div className="muted">当前检查范围内数据状态正常。</div></div>}</> : <Loading />}
        </div>
      )}

      {tab === "versions" && (
        <div className="card"><h3 style={{ marginTop: 0 }}>当前版本</h3>{v ? <div className="kv-grid"><div className="kv-item"><div className="k">版本</div><div className="v">v{v.version}</div></div><div className="kv-item"><div className="k">行数</div><div className="v">{v.row_count}</div></div><div className="kv-item"><div className="k">列数</div><div className="v">{v.column_count}</div></div><div className="kv-item"><div className="k">格式</div><div className="v">{v.format}</div></div></div> : <div className="muted">暂无版本。</div>}</div>
      )}
    </div>
  );
}
