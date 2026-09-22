import type { MlModel } from "../../types/ml";
import { modelLabel, modelDesc } from "./modelMeta";

// Prompt 173：模型选择器（按任务过滤后端注册表）。
export function ModelSelector({
  models,
  value,
  onChange,
}: {
  models: MlModel[];
  value: string;
  onChange: (model: string) => void;
}) {
  return (
    <div className="form-row" style={{ gap: "var(--space-2)", flexWrap: "wrap" }}>
      {models.map((m) => (
        <button
          key={m.name}
          className={`btn ${value === m.name ? "primary" : ""}`}
          onClick={() => onChange(m.name)}
          title={modelDesc(m.name) ?? m.desc ?? m.name}
        >
          {modelLabel(m.name)}
        </button>
      ))}
      {!models.length && <span className="muted">该任务暂无可用模型</span>}
    </div>
  );
}
