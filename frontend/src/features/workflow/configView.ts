import type { WorkflowNode } from "../../types/workflow";
import { specOf } from "./nodeSpecs";

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
    const merged = { ...flat, ...(params as Record<string, unknown>) };
    // 与后端 `app/workflow/models.py::flatten_config` 同口径：params 本身也要剔除。
    // 否则同一份嵌套配置会被展开成「平铺键 + params」两份，写回节点 config 时
    // 又把嵌套结构原样塞回去，等于在同一个节点里并存两种写法。
    delete merged.params;
    return merged;
  }
  return flat;
}

/**
 * 缺少就必然跑不起来的必需参数。
 *
 * 现已改为由 `nodeSpecs.NODE_SPECS` 的 `required` 标记驱动（单一事实源），
 * 本表仅作为 nodeSpecs 尚未覆盖的类型（如后端已注册但前端没建模的
 * `dataset.read`）的兜底。
 *
 * 只用于前端提示：**保存草稿不受限制**，只有运行时后端才会硬拦。
 * 因此这些问题在「流程体检」里是 warning 而非 error —— 用户本来就需要先搭图再填参数。
 */
export const NODE_REQUIRED_CONFIG: Record<string, { key: string; label: string }> = {
  "dataset.read": { key: "dataset_id", label: "数据集" },
};

export interface MissingConfig {
  nodeId: string;
  nodeType: string;
  nodeLabel: string;
  paramKey: string;
  paramLabel: string;
}

/**
 * 收集所有「必填参数缺失」的节点，供流程体检与运行前校验共用。
 *
 * 判定来源优先级：nodeSpecs 的 required 标记 > NODE_REQUIRED_CONFIG 兜底表。
 */
export function collectMissingConfig(nodes: WorkflowNode[]): MissingConfig[] {
  const out: MissingConfig[] = [];
  for (const node of nodes) {
    const flat = flattenConfig(node.config);
    const spec = specOf(node.type);
    if (spec) {
      for (const param of spec.params) {
        if (!param.required) continue;
        // 条件依赖的参数未显示时不应算缺失（例如 fill_value 只在 strategy=fill 时要填）。
        if (param.visibleWhen && flat[param.visibleWhen.key] !== param.visibleWhen.equals) continue;
        if (flat[param.key] == null) {
          out.push({
            nodeId: node.id,
            nodeType: node.type,
            nodeLabel: spec.label,
            paramKey: param.key,
            paramLabel: param.label,
          });
        }
      }
      continue;
    }
    const fallback = NODE_REQUIRED_CONFIG[node.type];
    if (fallback && flat[fallback.key] == null) {
      out.push({
        nodeId: node.id,
        nodeType: node.type,
        nodeLabel: node.type,
        paramKey: fallback.key,
        paramLabel: fallback.label,
      });
    }
  }
  return out;
}

/** 收集「必需参数缺失」的提示文案，供流程体检展示。 */
export function missingConfigWarnings(nodes: WorkflowNode[]): string[] {
  return collectMissingConfig(nodes).map(
    (item) => `节点 ${item.nodeId}（${item.nodeLabel}）缺少${item.paramLabel}，运行前需要配置 ${item.paramKey}。`,
  );
}

/**
 * 运行前校验（对应审查 #8：运行按钮未闸住）。
 *
 * 返回 null 表示可以运行；否则给出阻止原因与可跳转的节点。
 */
export interface RunBlocker {
  reason: string;
  /** 建议用户先去处理的节点 id（前端用于选中该节点）。 */
  focusNodeId?: string;
}

export function runPreflight(name: string, nodes: WorkflowNode[]): RunBlocker | null {
  if (!name.trim()) {
    return { reason: "请先给工作流起一个名字。" };
  }
  if (!nodes.length) {
    return { reason: "画布上还没有节点，先添加一个「加载数据」节点。" };
  }
  const missing = collectMissingConfig(nodes);
  if (missing.length) {
    const first = missing[0];
    const more = missing.length > 1 ? `（共 ${missing.length} 处）` : "";
    return {
      reason: `节点「${first.nodeLabel}」还缺少${first.paramLabel}${more}。`,
      focusNodeId: first.nodeId,
    };
  }
  return null;
}
