import { useState } from "react";
import { compareRuns } from "../../api/ml";
import { ModelComparison } from "../ml/ModelComparison";
import type { CompareResult, ExperimentRun } from "../../types/ml";

// Prompt 188：实验对比 —— 勾选多个 run 后调用 /experiments/compare。
export function ExperimentCompare({ runs }: { runs: ExperimentRun[] }) {
  const [selected, setSelected] = useState<number[]>([]);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function toggle(id: number) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function doCompare() {
    if (selected.length < 2) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await compareRuns(selected));
    } catch (e) {
      setError(e instanceof Error ? e.message : "对比失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="flex-between">
        <span className="muted">勾选 2 个以上运行进行对比</span>
        <button className="btn primary" disabled={busy || selected.length < 2} onClick={() => void doCompare()}>
          对比（{selected.length}）
        </button>
      </div>
      {runs.length > 0 && (
        <div className="mt" style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)" }}>
          {runs.map((r) => (
            <label
              key={r.id}
              style={{
                display: "flex",
                gap: 6,
                alignItems: "center",
                border: `1px solid ${selected.includes(r.id) ? "var(--primary)" : "var(--border)"}`,
                borderRadius: "var(--radius-sm)",
                padding: "4px 8px",
                cursor: "pointer",
              }}
            >
              <input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggle(r.id)} />
              <span style={{ fontSize: 13 }}>
                Run #{r.id}
                <span className="muted" style={{ marginLeft: "var(--space-1)" }}>
                  {r.status === "success" ? "✓" : r.status}
                </span>
              </span>
            </label>
          ))}
        </div>
      )}
      {error && <p style={{ color: "var(--danger)" }}>{error}</p>}
      <div className="mt">
        <ModelComparison result={result} />
      </div>
    </div>
  );
}
