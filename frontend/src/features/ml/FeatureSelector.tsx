import type { SchemaColumn } from "../../types/dataset";
import { recommendTargetColumn } from "./modelMeta";

// Prompt 172：特征 / 目标列选择器。
// target：目标列（聚类任务不需要）；excluded：不参与训练的列；
// suggestedExcluded：页面算好的「建议排除」高基数列（本组件不再自行推断，
// 避免与页面两处各写一套启发式、给出互相矛盾的徽章）。
export function FeatureSelector({
  columns,
  task,
  target,
  onTargetChange,
  excluded,
  onExcludedChange,
  suggestedExcluded = [],
}: {
  columns: SchemaColumn[];
  task?: string;
  target: string | null;
  onTargetChange: (col: string | null) => void;
  excluded: string[];
  onExcludedChange: (cols: string[]) => void;
  suggestedExcluded?: string[];
}) {
  function toggleExclude(col: string) {
    if (excluded.includes(col)) {
      onExcludedChange(excluded.filter((c) => c !== col));
    } else {
      onExcludedChange([...excluded, col]);
    }
  }

  // 目标列的推荐与提示来自唯一事实源（modelMeta.recommendTargetColumn），
  // 页面自动选中与此处文案因此必然一致 —— 这里只负责把结论说出来。
  const rec = recommendTargetColumn(columns, task ?? "classification");
  let targetHint = "";
  if (task === "clustering") {
    targetHint = `💡 ${rec.reason}`;
  } else {
    targetHint = rec.column
      ? `💡 已自动选中「${rec.column}」——${rec.reason}`
      : `💡 ${rec.reason}`;
  }

  const featureCount = columns.filter((c) => c.column !== target && !excluded.includes(c.column)).length;

  return (
    <div>
      <div className="form-row">
        <label className="field" style={{ width: 260 }}>
          目标列（聚类任务可留空）
          <select
            value={target ?? ""}
            onChange={(e) => onTargetChange(e.target.value || null)}
          >
            <option value="">（无目标列）</option>
            {columns.map((c) => (
              <option key={c.column} value={c.column}>
                {c.column}（{c.dtype}）
              </option>
            ))}
          </select>
        </label>
        <span className="muted" style={{ alignSelf: "end" }}>
          特征列：{featureCount} / {columns.length}
        </span>
      </div>
      {targetHint && (
        <div className="muted" style={{ fontSize: 12, marginTop: "var(--space-1)" }}>{targetHint}</div>
      )}
      {target && rec.column && target !== rec.column && (
        <div className="ml-tuner-warn" style={{ fontSize: 12, marginTop: "var(--space-1)" }}>
          系统推荐的目标是「{rec.column}」，当前选择不同，请再次确认。
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginTop: "var(--space-2)" }}>
        {columns.map((c) => {
          const isTarget = c.column === target;
          const isExcluded = excluded.includes(c.column);
          const disabled = isTarget;
          const isSuggestedExclude = suggestedExcluded.includes(c.column);
          return (
            <label
              key={c.column}
              style={{
                display: "flex",
                gap: 6,
                alignItems: "center",
                border: `1px solid ${isTarget ? "var(--primary)" : isExcluded ? "var(--border)" : isSuggestedExclude ? "var(--c-exclude)" : "var(--success)"}`,
                borderRadius: "var(--radius-sm)",
                padding: "4px 8px",
                cursor: disabled ? "not-allowed" : "pointer",
                opacity: disabled ? 0.6 : 1,
              }}
            >
              <input
                type="checkbox"
                checked={!isExcluded && !isTarget}
                disabled={disabled}
                onChange={() => toggleExclude(c.column)}
              />
              <span>
                {c.column}
                {isTarget && <span className="muted" style={{ marginLeft: "var(--space-1)" }}>· 目标</span>}
                {!isTarget && isSuggestedExclude && (
                  <span
                    style={{
                      marginLeft: 6,
                      fontSize: 11,
                      padding: "1px 5px",
                      background: "var(--c-exclude-weak)",
                      color: "var(--c-exclude-text)",
                      borderRadius: 4,
                    }}
                  >
                    💡 建议排除
                  </span>
                )}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
