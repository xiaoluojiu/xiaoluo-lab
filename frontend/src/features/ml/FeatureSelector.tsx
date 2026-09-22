import type { SchemaColumn } from "../../types/dataset";

// Prompt 172：特征 / 目标列选择器（增强版 — 带推荐徽章）。
// target：目标列（聚类任务不需要）；excluded：不参与训练的列。
export function FeatureSelector({
  columns,
  task,
  target,
  onTargetChange,
  excluded,
  onExcludedChange,
}: {
  columns: SchemaColumn[];
  task?: string;
  target: string | null;
  onTargetChange: (col: string | null) => void;
  excluded: string[];
  onExcludedChange: (cols: string[]) => void;
}) {
  function toggleExclude(col: string) {
    if (excluded.includes(col)) {
      onExcludedChange(excluded.filter((c) => c !== col));
    } else {
      onExcludedChange([...excluded, col]);
    }
  }

  // ===== 推荐逻辑 =====
  const numericCols = columns.filter(
    (c) =>
      c.dtype.toLowerCase().includes("int") || c.dtype.toLowerCase().includes("float"),
  );

  // 推荐的 target 列
  let recommendedTarget: SchemaColumn | undefined;
  if (task === "classification") {
    recommendedTarget = columns.find(
      (c) => c.unique_count <= 20 && !c.dtype.toLowerCase().includes("float"),
    );
  } else if (task === "regression") {
    recommendedTarget =
      columns.find(
        (c) =>
          c.dtype.toLowerCase().includes("float") &&
          /target|label|price|score|amount|y/i.test(c.column),
      ) ?? numericCols[0];
  }

  // 推荐排除：高基数（>100）的非数值列 + 非数值且非 target 的列（粗略启发式）
  const autoExcludedSet = new Set<string>();
  for (const c of columns) {
    const isHighCardinality = c.unique_count > 100 || c.unique_count === columns.length;
    const isNonNumeric = !c.dtype.toLowerCase().includes("int") && !c.dtype.toLowerCase().includes("float");
    if (isHighCardinality && isNonNumeric) {
      autoExcludedSet.add(c.column);
    }
  }

  // 推荐特征：数值列且不是 target、没被排除
  const recommendedFeatureSet = new Set<string>();
  for (const c of numericCols) {
    if (c.column !== target && !autoExcludedSet.has(c.column)) {
      recommendedFeatureSet.add(c.column);
    }
  }

  // target 推荐依据说明
  let targetHint = "";
  if (task === "classification") {
    targetHint = recommendedTarget
      ? `💡 已自动选中「${recommendedTarget.column}」—— 低基数（unique=${recommendedTarget.unique_count}）非浮点列，适合作为分类标签`
      : "💡 建议选择低基数（unique_count ≤ 20）的非浮点列作为 target";
  } else if (task === "regression") {
    targetHint = recommendedTarget
      ? `💡 已自动选中「${recommendedTarget.column}」—— 数值型且名称匹配常见 target 关键词`
      : "💡 建议选择数值型列作为 target，优先选名称含 target/label/price/score 的列";
  } else if (task === "clustering") {
    targetHint = "💡 聚类任务不需要 target 列";
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
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginTop: "var(--space-2)" }}>
        {columns.map((c) => {
          const isTarget = c.column === target;
          const isExcluded = excluded.includes(c.column);
          const disabled = isTarget;
          const isSuggestedExclude = autoExcludedSet.has(c.column);
          const isSuggestedFeature = recommendedFeatureSet.has(c.column) && !isExcluded;
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
                {!isTarget && !isExcluded && isSuggestedFeature && (
                  <span
                    style={{
                      marginLeft: 6,
                      fontSize: 11,
                      padding: "1px 5px",
                      background: "var(--c-green-weak)",
                      color: "var(--c-green)",
                      borderRadius: 4,
                    }}
                  >
                    ✓ 推荐特征
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
