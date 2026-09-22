import type { CompareResult } from "../../types/ml";

// 多 run 指标对比：兼容当前后端返回的 best 字段，也兼容旧版 best_by_metric。
export function ModelComparison({ result }: { result: CompareResult | null }) {
  if (!result) return <div className="muted">先在实验历史中勾选至少 2 个运行，再点击“对比选中运行”。</div>;

  const entries = result.entries ?? [];
  const best = result.best_by_metric ?? result.best ?? {};
  if (!entries.length) return <div className="muted">对比结果为空。</div>;

  const metricNames = [...new Set(entries.flatMap((e) => Object.keys(e.metrics ?? {})))];
  const successfulCount = entries.filter((e) => e.status === "success").length;

  return (
    <div>
      <div className="ml-compare-toolbar">
        <div className="muted">
          已比较 {entries.length} 个运行 · 成功 {successfulCount} 个
        </div>
        {result.runtime_ranking?.length ? (
          <span className="badge">耗时最快：Run #{result.runtime_ranking[0]}</span>
        ) : null}
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>状态</th>
            {metricNames.map((metric) => <th key={metric}>{metric}</th>)}
            <th>耗时(s)</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.run_id}>
              <td>#{entry.run_id}</td>
              <td>
                <span className={`badge ${entry.status === "success" ? "success" : entry.status === "failed" ? "failed" : "warning"}`}>
                  {entry.status}
                </span>
              </td>
              {metricNames.map((metric) => {
                const value = entry.metrics?.[metric];
                const isBest = best[metric] === entry.run_id;
                return (
                  <td key={metric}>
                    {value != null ? (typeof value === "number" ? value.toFixed(4) : String(value)) : "-"}
                    {isBest && <span className="badge success" style={{ marginLeft: 6 }}>最优</span>}
                  </td>
                );
              })}
              <td>{entry.runtime != null ? entry.runtime.toFixed(3) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {result.parameter_diff && Object.keys(result.parameter_diff).length > 0 && (
        <details className="mt">
          <summary className="muted" style={{ cursor: "pointer" }}>查看参数差异</summary>
          <pre style={{ background: "var(--surface-muted)", color: "var(--text)", border: "1px solid var(--border)", padding: 10, borderRadius: "var(--radius-sm)", fontSize: 13, overflow: "auto" }}>
            {JSON.stringify(result.parameter_diff, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
