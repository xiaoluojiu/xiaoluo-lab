import { useMemo, useState } from "react";
import type { WorkflowRun } from "../../types/workflow";
import "./workflow-run.css";

function statusClass(status: string) {
  return status === "success" || status === "completed" ? "success" : status === "failed" || status === "error" ? "failed" : "warning";
}

function pretty(value: unknown) {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

export function WorkflowRunPanel({
  run,
  onCancel,
  onSelectNode,
}: {
  run: WorkflowRun | null;
  onCancel: () => void;
  onSelectNode?: (id: string) => void;
}) {
  const [tab, setTab] = useState<"overview" | "outputs" | "logs">("overview");
  const logs = run?.result?.logs ?? [];
  const outputs = run?.result?.outputs ?? {};
  const failedCount = useMemo(() => logs.filter((item) => item.status === "failed" || item.error).length, [logs]);

  if (!run) {
    return <div className="workflow-run-empty"><strong>还没有执行记录</strong></div>;
  }

  return <div className="workflow-run-panel">
    <div className="workflow-run-summary">
      <div><span className="workflow-run-label">RUN</span><strong>{run.run_id}</strong><span>{logs.length} 个节点 · {failedCount ? `${failedCount} 个问题` : "未发现节点错误"}</span></div>
      <div className="workflow-run-actions"><span className={`badge ${statusClass(run.status)}`}>{run.status}</span>{(run.status === "running" || run.status === "pending") && <button className="btn danger" onClick={onCancel}>取消运行</button>}</div>
    </div>
    <div className="workflow-run-tabs">
      <button type="button" className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}>节点</button>
      <button type="button" className={tab === "outputs" ? "active" : ""} onClick={() => setTab("outputs")}>输出 <span>{Object.keys(outputs).length}</span></button>
      <button type="button" className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}>日志</button>
    </div>
    {tab === "overview" && <div className="workflow-run-node-list">
      {logs.length === 0 ? <div className="muted">当前运行尚未返回节点日志。</div> : logs.map((log) => <button type="button" className="workflow-run-node" key={log.node_id} onClick={() => onSelectNode?.(log.node_id)}>
        <span className={`workflow-run-dot ${statusClass(log.status)}`} />
        <span className="workflow-run-node-main"><strong>{log.node_id}</strong><small>{log.error || (log.status === "success" ? "执行完成" : log.status)}</small></span>
        <span className="workflow-run-duration">{log.duration_ms != null ? `${log.duration_ms} ms` : "—"}</span><span>›</span>
      </button>)}
    </div>}
    {tab === "outputs" && <div className="workflow-run-output-list">
      {Object.keys(outputs).length === 0 ? <div className="muted">没有可展示的节点输出。</div> : Object.entries(outputs).map(([id, output]) => <details key={id} className="workflow-run-output" open={Object.keys(outputs).length === 1}><summary><strong>{id}</strong><span>查看输出</span></summary><pre>{pretty(output)}</pre></details>)}
    </div>}
    {tab === "logs" && <div className="workflow-run-log-list">
      {logs.map((log) => <div className="workflow-run-log" key={log.node_id}><div><strong>{log.node_id}</strong><span className={`badge ${statusClass(log.status)}`}>{log.status}</span></div><p>{log.error || "节点完成，无错误信息。"}</p><small>{log.duration_ms != null ? `耗时 ${log.duration_ms} ms` : "未提供耗时"}</small></div>)}
    </div>}
  </div>;
}
