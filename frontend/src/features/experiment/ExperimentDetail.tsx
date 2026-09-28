import { useEffect, useState } from "react";
import type { Experiment, ExperimentRun } from "../../types/ml";
import { getExperimentNarrative, type ExperimentNarrative } from "../../api/experiments";

// Prompt 187：实验详情 —— 实验叙述 + 配置 + 运行历史。
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
  const [narrative, setNarrative] = useState<ExperimentNarrative | null>(null);

  // 叙述由后端按「实验 + 全部运行」现算，切换实验时必须重新拉取；
  // 请求失败（如旧运行无产物）只是少一块解读，不该让整个详情页报错。
  useEffect(() => {
    if (!experiment) {
      setNarrative(null);
      return;
    }
    let alive = true;
    getExperimentNarrative(experiment.id)
      .then((n) => { if (alive) setNarrative(n); })
      .catch(() => { if (alive) setNarrative(null); });
    return () => { alive = false; };
  }, [experiment?.id]);

  if (!experiment) return <div className="muted">从上方列表选择一个实验</div>;

  // 四个字段全空（例如还没跑过）时不渲染该区块 —— 空壳标题比没有更误导。
  const narrativeItems: { label: string; text: string }[] = [];
  if (narrative?.hypothesis) narrativeItems.push({ label: "想验证什么", text: narrative.hypothesis });
  if (narrative?.conclusion) narrativeItems.push({ label: "结果说明什么", text: narrative.conclusion });
  if (narrative?.failure_reason) narrativeItems.push({ label: "失败原因", text: narrative.failure_reason });
  if (narrative?.next_action) narrativeItems.push({ label: "下一步做什么", text: narrative.next_action });

  return (
    <div>
      {narrativeItems.length > 0 && (
        <div className="mt" style={{ borderBottom: "1px solid var(--border)", paddingBottom: "var(--space-2)" }}>
          <h4 style={{ margin: "0 0 var(--space-2)" }}>实验叙述</h4>
          {narrativeItems.map((item) => (
            <p key={item.label} style={{ margin: "0 0 var(--space-1)" }}>
              <span className="muted">{item.label}：</span>
              {item.text}
            </p>
          ))}
        </div>
      )}
      <div className="flex-between mt">
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
