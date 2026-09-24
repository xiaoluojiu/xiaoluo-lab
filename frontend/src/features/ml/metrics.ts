import type { ExperimentRun } from "../../types/ml";

/**
 * 「主指标」口径的**唯一定义处**。
 *
 * 历史问题：Home 页与实验中心各自写过一份「分类取 accuracy、回归取 r²」的挑选逻辑，
 * 两边口径一旦漂移，同一个实验在两个页面会显示不同数字。这里收敛成一处，
 * 任何页面要展示「代表性能的那一个数」都从这里取。
 *
 * 只读真实值，不虚构、不归一化、不做跨任务比较。
 */

/** 挑出最能代表性能的单一数值；取不到返回 null（调用方按「无指标」展示，不要补 0）。 */
export function primaryMetric(
  metrics: Record<string, number | string> | undefined | null,
): number | null {
  if (!metrics) return null;
  for (const key of ["accuracy", "r2", "silhouette", "f1"]) {
    const raw = metrics[key];
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  }
  return null;
}

/** 主指标的展示名（与后端评估口径对齐）。 */
export function metricLabel(
  metrics: Record<string, number | string> | undefined | null,
): string | null {
  if (!metrics) return null;
  if (typeof metrics.accuracy === "number") return "accuracy";
  if (typeof metrics.r2 === "number") return "r²";
  if (typeof metrics.silhouette === "number") return "轮廓系数";
  if (typeof metrics.f1 === "number") return "f1";
  return null;
}

/** 一行「指标名=值」的可读摘要；没有指标时返回 null，由调用方决定空态文案。 */
export function metricSummary(
  metrics: Record<string, number | string> | undefined | null,
): string | null {
  const value = primaryMetric(metrics);
  const label = metricLabel(metrics);
  if (value === null || label === null) return null;
  return `${label}=${value.toFixed(4)}`;
}

/** 取最近一次成功运行（失败的运行没有可比指标）。 */
export function latestSuccessfulRun(runs: ExperimentRun[]): ExperimentRun | null {
  return runs.find((r) => r.status === "success") ?? null;
}
