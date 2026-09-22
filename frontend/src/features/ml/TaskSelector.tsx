import type { MlModel } from "../../types/ml";

// Prompt 171：任务类型选择器（classification / regression / clustering）。
export function TaskSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (task: string) => void;
}) {
  const tasks = [
    { key: "classification", label: "分类" },
    { key: "regression", label: "回归" },
    { key: "clustering", label: "聚类" },
  ];
  return (
    <div className="form-row" style={{ gap: "var(--space-2)" }}>
      {tasks.map((t) => (
        <button
          key={t.key}
          className={`btn ${value === t.key ? "primary" : ""}`}
          onClick={() => onChange(t.key)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/** 从模型目录推导某任务的可用模型（后端 /ml/models）。 */
export function modelsForTask(models: MlModel[], task: string): MlModel[] {
  return models.filter((m) => m.task === task);
}
