/**
 * 本文件历史上是画布 + 节点注册表 + 模板库三合一。
 * 现在只剩下 re-export，避免改动既有调用点（pages/Workflow、沉浸式编辑器等）：
 * - 节点元数据的唯一事实源是 `nodeSpecs.ts`；
 * - 智能建议与预置模板搬到 `workflowTemplates.ts`（纯 TS，可被单测直接加载）。
 */

export { NODE_LABELS, NODE_TYPES } from "./nodeSpecs";
export {
  applyTemplate,
  generateSuggestedWorkflow,
  WORKFLOW_TEMPLATES,
  type WorkflowTemplate,
} from "./workflowTemplates";
