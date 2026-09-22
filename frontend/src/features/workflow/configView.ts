import type { WorkflowNode } from "../../types/workflow";

/**
 * 节点 config 的两种写法：平铺（`{"dataset_id": 1}`，画布参数面板与 data.load 常用）
 * 与嵌套（`{"params": {...}}`，ml.* 与 data.* 操作节点常用）。两者在后端都合法，
 * 前端任何「读某个参数」的地方都必须两种都认，否则同一份配置换个写法就显示成「没配」。
 *
 * 与后端 `app/workflow/models.py::flatten_config` 同口径；`__ui` 只是画布坐标，剔除。
 */
export function flattenConfig(config: Record<string, unknown> | undefined): Record<string, unknown> {
  const flat: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config ?? {})) {
    if (key !== "__ui") flat[key] = value;
  }
  const params = flat.params;
  if (params && typeof params === "object") {
    return { ...flat, ...(params as Record<string, unknown>) };
  }
  return flat;
}

/**
 * 缺少就必然跑不起来的必需参数（与后端 `runners.NODE_REQUIRED_CONFIG` 对应）。
 *
 * 这里只用于前端提示：**保存草稿不受限制**，只有运行时后端才会硬拦。
 * 因此这些问题在「流程体检」里是 warning 而非 error —— 用户本来就需要先搭图再填参数。
 */
export const NODE_REQUIRED_CONFIG: Record<string, { key: string; label: string }> = {
  "data.load": { key: "dataset_id", label: "数据集" },
  "dataset.read": { key: "dataset_id", label: "数据集" },
  "ml.train": { key: "target_column", label: "目标列" },
  "ml.evaluate": { key: "target_column", label: "目标列" },
};

/** 收集「必需参数缺失」的提示文案，供流程体检展示。 */
export function missingConfigWarnings(nodes: WorkflowNode[]): string[] {
  const messages: string[] = [];
  for (const node of nodes) {
    const spec = NODE_REQUIRED_CONFIG[node.type];
    if (!spec) continue;
    if (flattenConfig(node.config)[spec.key] == null) {
      messages.push(`节点 ${node.id}（${node.type}）缺少${spec.label}，运行前需要配置 ${spec.key}。`);
    }
  }
  return messages;
}
