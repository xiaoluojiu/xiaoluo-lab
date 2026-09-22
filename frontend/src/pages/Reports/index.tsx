import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import "./reports.css";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { PageHeader } from "../../components/PageHeader";
import { deleteSavedReport, generateReport, listSavedReports, type SavedReportSummary } from "../../api/reports";
import type { Report } from "../../types/report";

/** URL 深链参数名：统一为 key（历史上 href 用 ?report=、window.open 用 ?key=，导致新窗口永远打不开报告）。 */
const KEY_PARAM = "key";

/** 报告详情独立路由：/reports/:key —— 正文不再内联展开在列表下方。 */
function reportPath(key: string) {
  return `/reports/${encodeURIComponent(key)}`;
}

export default function Reports() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [datasetIds, setDatasetIds] = useState<number[]>([]);
  const [includeQuality, setIncludeQuality] = useState(true);
  const [includeEda, setIncludeEda] = useState(true);
  const [includeMl, setIncludeMl] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [savedReports, setSavedReports] = useState<SavedReportSummary[]>([]);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<SavedReportSummary | null>(null);

  const datasetId = datasetIds[0];

  async function refreshSaved() {
    try {
      setSavedReports(await listSavedReports());
    } catch {
      setSavedReports([]);
    }
  }

  useEffect(() => {
    void refreshSaved();
    // 兼容旧分享链接（?key=xxx）：直接改跳到独立详情路由，刷新与分享行为一致。
    const key = searchParams.get(KEY_PARAM);
    if (key) navigate(reportPath(key), { replace: true });
  }, []);

  async function generate() {
    if (!datasetId) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const generated: Report = await generateReport({
        dataset_id: datasetId,
        include_quality: includeQuality,
        include_eda: includeEda,
        include_ml: includeMl,
      });
      await refreshSaved();
      const key = typeof generated.metadata?.report_key === "string" ? generated.metadata.report_key : "";
      // 生成后一步到位打开详情，不再让用户回到列表里再点一次「查看」。
      if (key) navigate(reportPath(key), { state: { report: generated } });
      else setNotice("报告已生成，但后端未返回可打开的报告编号，请刷新列表后重试。");
    } catch (e) {
      setError(e instanceof Error ? e.message : "生成报告失败");
    } finally {
      setBusy(false);
    }
  }

  async function removeSaved(key: string) {
    setPendingDelete(null);
    setDeletingKey(key);
    setError(null);
    try {
      await deleteSavedReport(key);
      await refreshSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除报告失败");
    } finally {
      setDeletingKey(null);
    }
  }

  return (
    <div className="reports-page">
      <PageHeader
        title="分析报告"
        description="基于数据集生成质量、EDA 与实验报告，生成后可导出 Markdown / HTML / PDF 或用链接分享。"
        breadcrumbs={<>交付 / <b>分析报告</b></>}
      />
      <DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} />

      <div className="card">
        <h3>生成选项</h3>
        <div className="control-row" style={{ alignItems: "center" }}>
          <label className="inline-check"><input type="checkbox" checked={includeQuality} onChange={(e) => setIncludeQuality(e.target.checked)} />数据质量</label>
          <label className="inline-check"><input type="checkbox" checked={includeEda} onChange={(e) => setIncludeEda(e.target.checked)} />探索分析（EDA）</label>
          <label className="inline-check"><input type="checkbox" checked={includeMl} onChange={(e) => setIncludeMl(e.target.checked)} />最近实验</label>
          <button className="btn primary" disabled={!datasetId || busy} onClick={() => void generate()}>{busy ? "生成中…" : "生成报告"}</button>
        </div>
        {notice && <p className="muted" style={{ marginTop: "var(--space-3)" }}>{notice}</p>}
        {error && (
          <div className="error-box" style={{ marginTop: "var(--space-3)" }}>
            <div className="error-title">操作未完成</div>
            <div className="error-detail">{error}</div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="flex-between">
          <h3 style={{ margin: 0 }}>已保存报告（{savedReports.length}）</h3>
          <button className="btn btn-sm" onClick={() => void refreshSaved()}>刷新</button>
        </div>
        {savedReports.length === 0 ? (
          <div className="empty" style={{ marginTop: "var(--space-3)" }}>
            <span className="empty-title">还没有保存的报告</span>
            选择上方数据集并点击「生成报告」，生成结果会自动归档到这里。
          </div>
        ) : (
          /* 报告多时表格会把页面拉到很长，这里给列表一个固定可视高度 + 内部滚动，
             表头 sticky 常驻，页面本身不再被撑开。 */
          <div className="report-saved-scroll">
            <table className="data-table report-saved-table">
              <thead><tr><th>标题</th><th>数据集</th><th>大小</th><th>保存时间</th><th>操作</th></tr></thead>
              <tbody>{savedReports.map((item) => (
                <tr key={item.key}>
                  <td>{item.title || "(未命名)"}</td>
                  <td>{String(item.dataset?.name ?? "")}{item.dataset?.rows != null ? ` · ${String(item.dataset.rows)} 行` : ""}</td>
                  <td>{(item.size / 1024).toFixed(1)} KB</td>
                  <td>{item.modified_at.replace("T", " ").slice(0, 19)}</td>
                  <td>
                    <div className="inline-actions">
                      {/* 查看改为独立路由跳转：正文不再内联展开在本页底部。 */}
                      <Link className="btn btn-sm" to={reportPath(item.key)}>查看</Link>
                      <button className="btn btn-sm danger" disabled={deletingKey === item.key} onClick={() => setPendingDelete(item)}>
                        {deletingKey === item.key ? "删除中…" : "删除"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除这份已保存报告？"
        message={<>《{pendingDelete?.title || "(未命名)"}》将被永久删除，删除后无法从报告中心恢复。</>}
        confirmText="删除"
        danger
        onConfirm={() => { if (pendingDelete) void removeSaved(pendingDelete.key); }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
