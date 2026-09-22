import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { listDatasets } from "../../api/datasets";
import type { Dataset } from "../../types/dataset";

export function DatasetSelector({
  value,
  onChange,
  multi = true,
  requireVersion = true,
  compact = false,
  showLabel = true,
  bare = false,
}: {
  value: number[];
  onChange: (ids: number[]) => void;
  multi?: boolean;
  requireVersion?: boolean;
  compact?: boolean;
  showLabel?: boolean;
  /** 嵌入到外部卡片（toolbar）内时置 true：不再渲染自带 .card 外壳，避免双层边框/阴影与错位。 */
  bare?: boolean;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchParams] = useSearchParams();

  useEffect(() => {
    let cancelled = false;
    listDatasets(1, 200)
      .then((r) => {
        if (!cancelled) setDatasets(requireVersion ? r.items.filter((d) => d.latest_version) : r.items);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "加载失败"))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [requireVersion]);

  // 从 Dataset Workspace 跳入 Processing / Analysis / ML / Workflow / AI 时保留数据上下文。
  useEffect(() => {
    if (!datasets.length) return;
    const raw = searchParams.get("dataset");
    const id = raw ? Number(raw) : NaN;
    if (!Number.isInteger(id) || id <= 0 || !datasets.some((d) => d.id === id) || value.includes(id)) return;
    onChange(multi ? [...value, id] : [id]);
  }, [datasets, multi, onChange, searchParams, value]);

  if (loading) return <span className="muted">加载数据集...</span>;
  if (error) return <span className="badge failed">{error}</span>;
  if (!datasets.length) return <span className="muted">暂无可用数据集</span>;

  function toggle(id: number) {
    if (multi) onChange(value.includes(id) ? value.filter((v) => v !== id) : [...value, id]);
    else onChange([id]);
  }

  if (compact) {
    return (
      <label className="dataset-selector-compact">
        {showLabel && <span>数据集</span>}
        <select
          multiple={multi}
          value={value.map(String)}
          size={1}
          onChange={(event) => {
            if (multi) onChange(Array.from(event.target.selectedOptions, (option) => Number(option.value)));
            else if (event.target.value) toggle(Number(event.target.value));
            else onChange([]);
          }}
        >
          {!multi && <option value="">选择数据集…</option>}
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}{d.latest_version ? ` · v${d.latest_version.version} · ${d.latest_version.row_count}行` : ""}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <div className={bare ? "dataset-selector-bare" : "card"}>
      {!bare && <h3>选择数据集{multi ? "（可多选）" : ""}</h3>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "var(--space-2)" }}>
        {datasets.map((d) => {
          const checked = value.includes(d.id);
          return (
            <label key={d.id} style={{ display: "flex", gap: "var(--space-2)", alignItems: "center", border: `1px solid ${checked ? "var(--primary)" : "var(--border)"}`, borderRadius: "var(--radius-sm)", padding: "8px 10px", cursor: "pointer" }}>
              <input type={multi ? "checkbox" : "radio"} checked={checked} onChange={() => toggle(d.id)} />
              <span>{d.name}<span className="muted" style={{ marginLeft: 6 }}>{d.latest_version ? `v${d.latest_version.version} · ${d.latest_version.row_count}行` : "无版本"}</span></span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
