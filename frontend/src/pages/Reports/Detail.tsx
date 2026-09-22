import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import "./reports.css";
import { ReportPreview } from "../../features/report/ReportPreview";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { PageHeader } from "../../components/PageHeader";
import { Skeleton } from "../../components/StateBlock";
import { deleteSavedReport, exportReport, getSavedReport, type ExportFormat } from "../../api/reports";
import type { Report } from "../../types/report";

const FORMATS: { key: ExportFormat; label: string; mime: string; ext: string }[] = [
  { key: "markdown", label: "Markdown", mime: "text/markdown", ext: "md" },
  { key: "html", label: "HTML", mime: "text/html", ext: "html" },
  { key: "pdf", label: "PDF", mime: "application/pdf", ext: "pdf" },
];

/**
 * 报告详情（独立路由 /reports/:key）。
 *
 * 历史问题：报告中心把正文内联展开在列表页下方，长报告会把列表挤到页面最底部，
 * 返回列表要往上滚很久，链接分享也只会打开整个列表页。现在改为独立路由加载，
 * 头部固定提供「返回报告中心」入口与导出 / 删除操作。
 */
export default function ReportDetail() {
  const { key = "" } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [report, setReport] = useState<Report | null>(() => {
    // 生成后直接跳转过来时带上结果，省一次请求；刷新 / 分享链接时再回源读取。
    const carried = (location.state as { report?: Report } | null)?.report;
    return carried ?? null;
  });
  const [loading, setLoading] = useState(!report);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (report) return;
    let alive = true;
    setLoading(true);
    getSavedReport(key)
      .then((loaded) => { if (alive) { setReport(loaded); setError(null); } })
      .catch((e: unknown) => { if (alive) setError(e instanceof Error ? e.message : "报告不存在或已被删除"); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [key, report]);

  async function download(format: ExportFormat) {
    if (!report) return;
    try {
      const blob = await exportReport(report, format);
      const meta = FORMATS.find((f) => f.key === format);
      const url = URL.createObjectURL(new Blob([blob], { type: meta?.mime }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `${report.title || "report"}.${meta?.ext ?? "txt"}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "导出失败");
    }
  }

  async function remove() {
    setPendingDelete(false);
    setDeleting(true);
    try {
      await deleteSavedReport(key);
      navigate("/reports", { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除报告失败");
      setDeleting(false);
    }
  }

  return (
    <div className="report-detail-page">
      <PageHeader
        title={report?.title || "报告详情"}
        breadcrumbs={<>交付 / <Link to="/reports">报告中心</Link> / <b>报告详情</b></>}
        actions={
          <>
            <Link className="btn" to="/reports">← 返回报告中心</Link>
            {report && FORMATS.map((f) => (
              <button key={f.key} className="btn" type="button" onClick={() => void download(f.key)}>{f.label}</button>
            ))}
            {report && (
              <button className="btn danger" type="button" disabled={deleting} onClick={() => setPendingDelete(true)}>
                {deleting ? "删除中…" : "删除报告"}
              </button>
            )}
          </>
        }
      />

      {error && (
        <div className="card">
          <div className="error-box">
            <div className="error-title">报告加载失败</div>
            <div className="error-detail">{error}</div>
          </div>
          <div className="inline-actions" style={{ marginTop: "var(--space-3)" }}>
            <Link className="btn" to="/reports">← 返回报告中心</Link>
          </div>
        </div>
      )}

      {loading && !error && <div className="card"><Skeleton lines={4} /></div>}

      {report && !error && (
        <div className="card report-detail-card">
          <ReportPreview report={report} />
        </div>
      )}

      <ConfirmDialog
        open={pendingDelete}
        title="删除这份报告？"
        message={<>《{report?.title || "(未命名)"}》将被永久删除，删除后无法从报告中心恢复。</>}
        confirmText="删除"
        danger
        onConfirm={() => void remove()}
        onCancel={() => setPendingDelete(false)}
      />
    </div>
  );
}
