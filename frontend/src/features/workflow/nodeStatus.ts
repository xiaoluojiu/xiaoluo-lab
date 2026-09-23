/**
 * 节点状态 → 展示样式 / 文案的唯一映射。
 *
 * 对齐后端 `app/workflow/state.py::NodeStatus`：
 *   pending / running / success / failed / cancelled / skipped
 *
 * 原实现的问题（审查 #5）：`tone()` 只认 success/completed/error，
 * 未运行的节点一律落到 idle，于是 pending / skipped / cancelled 三种
 * 语义完全不同的状态在画布上长得一模一样；「等待配置」又和 pending 混淆。
 * 这里把「运行状态」与「配置是否完整」拆成两件事分别表达。
 */

export type NodeTone = "idle" | "running" | "success" | "danger" | "warning" | "muted";

const TONES: Record<string, NodeTone> = {
  pending: "idle",
  running: "running",
  success: "success",
  failed: "danger",
  cancelled: "muted",
  skipped: "warning",
};

const LABELS: Record<string, string> = {
  pending: "等待运行",
  running: "运行中",
  success: "已完成",
  failed: "已失败",
  cancelled: "已取消",
  skipped: "已跳过",
};

/** 状态 → 色调；未知状态（含尚未运行的 undefined）落到 idle。 */
export function statusTone(status?: string): NodeTone {
  return status ? (TONES[status] ?? "idle") : "idle";
}

/** 状态 → 中文文案。 */
export function statusLabel(status?: string): string {
  return status ? (LABELS[status] ?? status) : "尚未运行";
}

/** 是否终态（与后端 TERMINAL_STATUSES 一致，用于判断能否重跑）。 */
export function isTerminal(status?: string): boolean {
  return status === "success" || status === "failed" || status === "cancelled" || status === "skipped";
}

/**
 * 节点完整状态 = 运行状态 + 配置完整度。
 *
 * 「等待配置」（配置没填全）与「等待运行」（配置好了但还没跑）是两回事，
 * 原实现把它们都显示成灰色文字，用户无法区分"我还没配完"与"配好了可以跑"。
 */
export interface NodeDisplay {
  tone: NodeTone;
  label: string;
  /** 配置是否缺必填项（由 nodeSpecs 的 required 推导）。 */
  needsConfig: boolean;
}

export function nodeDisplay(status: string | undefined, needsConfig: boolean): NodeDisplay {
  if (needsConfig && !status) {
    return { tone: "warning", label: "等待配置", needsConfig: true };
  }
  return { tone: statusTone(status), label: statusLabel(status), needsConfig };
}

/* ------------------------------------------------------------------ */
/* 节点产出摘要（分层信息的数据来源）                                  */
/* ------------------------------------------------------------------ */

/** 单个节点的产出规模。`metricsFromOutputs` 的取值类型。 */
export interface NodeMetric {
  rows?: number;
  columns?: number;
}

/**
 * 从运行结果里提取每个节点的「产出多少行 × 多少列」。
 *
 * 为什么放在这里而不是各个页面里各写一遍：画布节点卡片、连线标注、
 * 血缘<｜hy_place▁holder▁no▁813｜>三处都要用同一份口径，`WorkflowHealthPanel.outputSummary`
 * 之前就实现过一次，属于重复逻辑。
 *
 * 后端 `WorkflowRunResult.outputs[nodeId]` 现在是 `Record<string, unknown>`，
 * 数据节点通常带 `row_count` / `column_count`；这两键是潜在多样 duck-typing
 * 入口，键名若变成本也有唯一一处可改。读不到就返回空对象，由调用方决定降级。
 */
export function metricsFromOutputs(
  outputs: Record<string, unknown> | null | undefined,
): Record<string, NodeMetric> {
  const metrics: Record<string, NodeMetric> = {};
  for (const [nodeId, value] of Object.entries(outputs ?? {})) {
    if (!value || typeof value !== "object") continue;
    const record = value as Record<string, unknown>;
    const metric: NodeMetric = {};
    if (typeof record.row_count === "number") metric.rows = record.row_count;
    if (typeof record.column_count === "number") metric.columns = record.column_count;
    if (metric.rows != null || metric.columns != null) metrics[nodeId] = metric;
  }
  return metrics;
}
