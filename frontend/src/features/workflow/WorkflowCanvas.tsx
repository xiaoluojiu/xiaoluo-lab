import type { WorkflowEdge, WorkflowNode } from "../../types/workflow";

/** Workflow Studio 的节点注册表；执行能力与 backend/app/workflow/runners.py 对齐。 */
export const NODE_TYPES = [
  "noop",
  "data.load",
  "data.clean",
  "data.duplicate",
  "data.cast",
  "data.string",
  "data.filter",
  "data.transform",
  "data.aggregate",
  "data.pivot",
  "data.melt",
  "ml.train",
  "ml.predict",
  "ml.evaluate",
  "ml.cluster",
  "ml.pca",
  "ai.analyze",
] as const;

export const NODE_LABELS: Record<string, string> = {
  noop: "空操作",
  "data.load": "加载数据",
  "data.clean": "清洗（缺失值）",
  "data.duplicate": "去重",
  "data.cast": "类型转换",
  "data.string": "字符串处理",
  "data.filter": "筛选行",
  "data.transform": "计算新字段",
  "data.aggregate": "聚合统计",
  "data.pivot": "透视表",
  "data.melt": "逆透视",
  "ml.train": "训练模型",
  "ml.predict": "生成预测",
  "ml.evaluate": "模型评估",
  "ml.cluster": "聚类分析",
  "ml.pca": "主成分分析",
  "ai.analyze": "AI 分析",
};

export function generateSuggestedWorkflow(columns: string[]): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } {
  const nodes: WorkflowNode[] = [
    { id: "load", type: "data.load", config: { params: {}, schema_columns: columns } },
    { id: "clean", type: "data.clean", config: { params: { strategy: "drop" } } },
  ];
  return { nodes, edges: [{ source: "load", target: "clean" }] };
}

/* =============================================================
   Workflow 预置模板库（2026-09-22）
   -------------------------------------------------------------
   目的：空画布不再是「从一个数据节点开始」的孤立提示，而是一组可按场景
   一键填充的起点。每个模板 = 名字 + 场景标签 + 一段人话描述 + 节点/边，
   节点类型与 config 全部与后端 runners 的口径对齐，填充后即可直接保存运行。
   数据集 ID 由页面层在应用模板时注入（data.load 的 dataset_id 不可凭空给）。
   ============================================================= */

export interface WorkflowTemplate {
  id: string;
  name: string;
  category: "数据清洗" | "机器学习" | "分析报告";
  description: string;
  nodes: Array<{ id: string; type: string; config: Record<string, unknown> }>;
  edges: Array<{ source: string; target: string }>;
}

export const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: "basic-clean",
    name: "基础清洗",
    category: "数据清洗",
    description: "加载数据 → 去重 → 缺失值处理，产出可直接分析的数据集。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "dedupe", type: "data.duplicate", config: { params: { keep: "first" } } },
      { id: "clean", type: "data.clean", config: { params: { strategy: "drop" } } },
    ],
    edges: [
      { source: "load", target: "dedupe" },
      { source: "dedupe", target: "clean" },
    ],
  },
  {
    id: "filter-aggregate",
    name: "筛选与聚合",
    category: "数据清洗",
    description: "按条件筛选行，再按分组字段做统计聚合，输出汇总表。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "filter", type: "data.filter", config: { params: { conditions: [], logic: "and" } } },
      { id: "aggregate", type: "data.aggregate", config: { params: { group_by: [], aggregations: {} } } },
    ],
    edges: [
      { source: "load", target: "filter" },
      { source: "filter", target: "aggregate" },
    ],
  },
  {
    id: "train-evaluate",
    name: "训练与评估",
    category: "机器学习",
    description: "加载数据 → 训练模型 → 评估指标，一条标准的监督学习链。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "train", type: "ml.train", config: { params: { model: "random_forest_regressor", target_column: null, test_size: 0.2 } } },
      { id: "evaluate", type: "ml.evaluate", config: { params: { target_column: null } } },
    ],
    edges: [
      { source: "load", target: "train" },
      { source: "train", target: "evaluate" },
    ],
  },
  {
    id: "cluster-segment",
    name: "聚类分群",
    category: "机器学习",
    description: "加载数据后直接聚类，为每条记录打上簇标签用于分群。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "cluster", type: "ml.cluster", config: { params: { model: "kmeans", output_column: "cluster" } } },
    ],
    edges: [{ source: "load", target: "cluster" }],
  },
  {
    id: "analyze-report",
    name: "AI 分析报告",
    category: "分析报告",
    description: "清洗 → 训练 → 让 AI 基于结果生成分析结论（需启用远程大模型）。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "clean", type: "data.clean", config: { params: { strategy: "drop" } } },
      { id: "train", type: "ml.train", config: { params: { model: "random_forest_regressor", target_column: null } } },
      { id: "analyze", type: "ai.analyze", config: { params: { prompt: "请总结这个流程的主要发现与下一步建议。" } } },
    ],
    edges: [
      { source: "load", target: "clean" },
      { source: "clean", target: "train" },
      { source: "train", target: "analyze" },
    ],
  },
  {
    id: "transform-pivot",
    name: "派生字段与透视",
    category: "数据清洗",
    description: "计算新字段，再把长表透视成宽表用于报表展示。",
    nodes: [
      { id: "load", type: "data.load", config: { dataset_id: null } },
      { id: "transform", type: "data.transform", config: { params: { name: "", expression: "" } } },
      { id: "pivot", type: "data.pivot", config: { params: { index: [], columns: [], values: [] } } },
    ],
    edges: [
      { source: "load", target: "transform" },
      { source: "transform", target: "pivot" },
    ],
  },
];

/** 把模板实例化成真正的节点/边，并注入 dataset_id（若提供）。 */
export function applyTemplate(
  template: WorkflowTemplate,
  datasetId: number | null,
): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } {
  const nodes: WorkflowNode[] = template.nodes.map((node) => {
    if (node.type === "data.load" && datasetId != null) {
      return { id: node.id, type: node.type, config: { ...node.config, dataset_id: datasetId } };
    }
    return { id: node.id, type: node.type, config: node.config };
  });
  return { nodes, edges: template.edges.map((edge) => ({ ...edge })) };
}
