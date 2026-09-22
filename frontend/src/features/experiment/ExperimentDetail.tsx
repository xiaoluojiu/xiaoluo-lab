import type { Experiment, ExperimentRun } from "../../types/ml";

// Prompt 187：实验详情 —— 配置 + 运行历史。
export function ExperimentDetail({
  experiment,
  runs,
  busy,
  onRun,
}: {
  experiment: Experiment | null;
  runs: ExperimentRun[];
  busy?: boolean;
  onRun: () => void;
}) {
  if (!experiment) return <div className="muted">从上方列表选择一个实验</div>;
  return (
    <div>
      <div className="flex-between">
        <div>
          <strong>实验 #{experiment.id}</strong>
          <span className="muted" style={{ marginLeft: "var(--space-2)" }}>
            {experiment.task} · {experiment.model}
            {experiment.target_column ? ` · target=${experiment.target_column}` : ""}
            {experiment.description ? ` · ${experiment.description}` : ""}
          </span>
        </div>
        <button className="btn primary" disabled={busy} onClick={onRun}>
          {busy ? "运行中..." : "再跑一次"}
        </button>
      </div>
      {Object.keys(experiment.parameters ?? {}).length > 0 && (
        <details className="mt" style={{ fontSize: 12 }}>
          <summary className="muted" style={{ cursor: "pointer" }}>参数</summary>
          <pre style={{ background: "var(--c-code-bg)", padding: "var(--space-2)", borderRadius: "var(--radius-sm)", overflow: "auto" }}>
            {JSON.stringify(experiment.parameters, null, 2)}
          </pre>
        </details>
      )}
      <h4 className="mt">运行记录</h4>
      {!runs.length && <div className="muted">暂无运行</div>}
      {runs.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>状态</th>
              <th>指标</th>
              <th>耗时(s)</th>
              <th>错误</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>#{r.id}</td>
                <td>
                  <span
                    className={`badge ${r.status === "success" ? "success" : r.status === "failed" ? "failed" : "warning"}`}
                  >
                    {r.status}
                  </span>
                </td>
                <td>
                  {Object.entries(r.metrics ?? {})
                    .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`)
                    .join("  ")}
                </td>
                <td>{r.runtime != null ? r.runtime.toFixed(3) : "-"}</td>
                <td style={{ color: "var(--danger)" }}>{r.error ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
