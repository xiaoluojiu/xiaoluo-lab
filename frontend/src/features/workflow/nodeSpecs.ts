/**
 * 节点规格（Node Spec）单一事实源。
 *
 * 解决的问题（对应审查清单 #1/#2/#4/#5/#10）：
 * - 节点库此前是 `NODE_TYPES` + `NODE_LABELS` 两份手写清单，与后端 runner 的
 *   对齐关系只能靠注释；这里补上**后端 runner 是否真的存在**这一事实，
 *   并暴露差异（见 `runnerMismatch`）。
 * - 节点图标此前是 ◈ / ◇ / ✦ 三个字符（审查说"统一菱形"不准确，但也确实
 *   不是图标系统）；这里改成 Icon 组件可用的图标名。
 * - 每个节点参数的类型 / 枚举 / 默认值 / 必填 / 条件依赖，后端已有
 *   `GET /processing/operations` 提供 op_type / params / label / description，
 *   但**没有**类型与枚举。为了不改后端协议，这里在前端补一层参数规格，
 *   与后端 operation 的实际实现一一对应（来源见每条的 source 注释）。
 */

import type { IconName } from "../../components/icons/Icon";

export type NodeCategory = "数据" | "机器学习" | "AI" | "报告";

export type ParamKind = "string" | "number" | "boolean" | "enum" | "columns" | "json";

export interface ParamSpec {
  key: string;
  label: string;
  kind: ParamKind;
  /** enum 取值；同时用于「按可用值渲染下拉」。 */
  options?: Array<{ value: string; label: string }>;
  default?: unknown;
  required?: boolean;
  /** 仅在该参数满足条件时显示（对应审查 #4 的条件依赖）。 */
  visibleWhen?: { key: string; equals: unknown };
  /** columns 类型：按语义过滤可选列。 */
  columnFilter?: "numeric" | "categorical" | "temporal" | "any";
  multi?: boolean;
  placeholder?: string;
  hint?: string;
}

export interface NodeSpec {
  type: string;
  label: string;
  category: NodeCategory;
  icon: IconName;
  /** 一句话说明该节点做什么。 */
  description: string;
  /** 是否由后端 `build_default_runners()` 直接注册（false 表示经 make_op 动态注册）。 */
  registeredRunner: boolean;
  params: ParamSpec[];
}

/* ------------------------------------------------------------------ */
/* 参数规格                                                            */
/* ------------------------------------------------------------------ */

const MISSING_STRATEGIES: ParamSpec["options"] = [
  { value: "drop", label: "删除含缺失的行" },
  { value: "fill", label: "填充固定值" },
  { value: "mean", label: "用均值填充（仅数值列）" },
  { value: "median", label: "用中位数填充（仅数值列）" },
  { value: "mode", label: "用众数填充" },
  { value: "ffill", label: "用前一个有效值填充" },
  { value: "bfill", label: "用后一个有效值填充" },
];

const STRING_OPS: ParamSpec["options"] = [
  { value: "lower", label: "转小写" },
  { value: "upper", label: "转大写" },
  { value: "strip", label: "去除首尾空白" },
  { value: "replace", label: "替换子串" },
  { value: "contains", label: "包含匹配" },
  { value: "startswith", label: "前缀匹配" },
  { value: "endswith", label: "后缀匹配" },
  { value: "slice", label: "按位置截取" },
];

const AGG_FUNCS: ParamSpec["options"] = [
  { value: "mean", label: "均值" },
  { value: "sum", label: "求和" },
  { value: "count", label: "计数" },
  { value: "min", label: "最小值" },
  { value: "max", label: "最大值" },
  { value: "median", label: "中位数" },
  { value: "std", label: "标准差" },
  { value: "nunique", label: "去重计数" },
];

const ML_MODELS: ParamSpec["options"] = [
  { value: "linear_regression", label: "线性回归" },
  { value: "random_forest_regressor", label: "随机森林回归" },
  { value: "logistic_regression", label: "逻辑回归（分类）" },
  { value: "random_forest_classifier", label: "随机森林分类" },
  { value: "kmeans", label: "K-Means 聚类" },
  { value: "pca", label: "主成分分析" },
];

const AGG_OPTIONS: ParamSpec["options"] = [
  { value: "mean", label: "均值" },
  { value: "sum", label: "求和" },
  { value: "count", label: "计数" },
  { value: "min", label: "最小值" },
  { value: "max", label: "最大值" },
];

/**
 * 按节点类型给出参数规格。
 *
 * 参数名严格取自 `app/data_engine/service.py::OPERATION_REGISTRY` 的第二个元组，
 * 以及 `app/workflow/runners.py` 里 ml.* / report.* 的 config 读取键。
 * ⚠️ `data.clean` 的填充值是 `value`，不是 `fill_value`（审查文档此处写错了）。
 */
export const NODE_SPECS: Record<string, NodeSpec> = {
  noop: {
    type: "noop",
    label: "空操作",
    category: "数据",
    icon: "flow",
    description: "占位节点，把上游数据原样透传，常用于连接或暂存。",
    registeredRunner: true,
    params: [],
  },
  "data.load": {
    type: "data.load",
    label: "加载数据",
    category: "数据",
    icon: "database",
    description: "从数据集读取一张表，作为整条流程的起点。",
    registeredRunner: true,
    params: [
      {
        key: "dataset_id",
        label: "数据集",
        kind: "enum",
        required: true,
        hint: "选择要加载的数据集；下游节点的可选列由它决定。",
      },
      {
        key: "version",
        label: "版本",
        kind: "number",
        hint: "留空则使用最新版本。数据集采用不可变版本快照，指定版本可复现历史结果。",
      },
    ],
  },
  "data.clean": {
    type: "data.clean",
    label: "清洗（缺失值）",
    category: "数据",
    icon: "wand",
    description: "按指定策略处理缺失值。",
    registeredRunner: false,
    params: [
      {
        key: "strategy",
        label: "处理策略",
        kind: "enum",
        options: MISSING_STRATEGIES,
        default: "drop",
        required: true,
      },
      {
        key: "columns",
        label: "作用列",
        kind: "columns",
        multi: true,
        columnFilter: "any",
        hint: "留空表示对所有列生效。",
      },
      {
        key: "value",
        label: "填充值",
        kind: "string",
        // 条件依赖：只有选「填充固定值」才需要填。
        visibleWhen: { key: "strategy", equals: "fill" },
        placeholder: "例如 0 或 未知",
      },
    ],
  },
  "data.duplicate": {
    type: "data.duplicate",
    label: "去重",
    category: "数据",
    icon: "layers",
    description: "删除重复记录。",
    registeredRunner: false,
    params: [
      { key: "subset", label: "判断列", kind: "columns", multi: true, columnFilter: "any", hint: "留空表示按所有列判断是否重复。" },
      {
        key: "keep",
        label: "保留哪一条",
        kind: "enum",
        options: [
          { value: "first", label: "保留第一条" },
          { value: "last", label: "保留最后一条" },
          { value: "none", label: "全部删除" },
        ],
        default: "first",
      },
    ],
  },
  "data.cast": {
    type: "data.cast",
    label: "类型转换",
    category: "数据",
    icon: "sliders",
    description: "转换字段的数据类型。",
    registeredRunner: false,
    params: [
      { key: "types", label: "列 → 目标类型", kind: "json", required: true, placeholder: '{ "Month": "Int32" }', hint: "键为列名，值为目标 dtype。" },
      { key: "formats", label: "日期解析格式", kind: "json", placeholder: '{ "FlightDate": "%Y-%m-%d" }' },
    ],
  },
  "data.string": {
    type: "data.string",
    label: "字符串处理",
    category: "数据",
    icon: "clipboard",
    description: "对文本字段执行大小写、替换、截取等操作。",
    registeredRunner: false,
    params: [
      { key: "column", label: "目标列", kind: "columns", columnFilter: "any", required: true },
      { key: "op", label: "操作", kind: "enum", options: STRING_OPS, required: true },
      { key: "params", label: "操作参数", kind: "json", placeholder: '{ "old": " ", "new": "" }', hint: "随所选操作而变，例如 replace 需要 old / new。" },
    ],
  },
  "data.filter": {
    type: "data.filter",
    label: "筛选行",
    category: "数据",
    icon: "filter",
    description: "按条件过滤数据行。",
    registeredRunner: false,
    params: [
      { key: "conditions", label: "条件", kind: "json", required: true, placeholder: '[{ "column": "DepDelay", "op": ">", "value": 0 }]' },
      {
        key: "logic",
        label: "组合方式",
        kind: "enum",
        options: [
          { value: "and", label: "全部满足（且）" },
          { value: "or", label: "满足任一（或）" },
        ],
        default: "and",
      },
    ],
  },
  "data.transform": {
    type: "data.transform",
    label: "计算新字段",
    category: "数据",
    icon: "sparkles",
    description: "用一个表达式派生新字段。",
    registeredRunner: false,
    params: [
      { key: "name", label: "新字段名", kind: "string", required: true },
      { key: "expression", label: "表达式", kind: "string", required: true, placeholder: "DepDelay * 2" },
      { key: "overwrite", label: "允许覆盖同名字段", kind: "boolean", default: false },
    ],
  },
  "data.aggregate": {
    type: "data.aggregate",
    label: "聚合统计",
    category: "数据",
    icon: "chart",
    description: "按分组字段做统计聚合。",
    registeredRunner: false,
    params: [
      { key: "group_by", label: "分组列", kind: "columns", multi: true, columnFilter: "categorical" },
      { key: "aggregations", label: "聚合方式", kind: "json", required: true, placeholder: '{ "DepDelay": ["mean", "max"] }', hint: "键为列名，值为聚合函数数组。" },
    ],
  },
  "data.pivot": {
    type: "data.pivot",
    label: "透视表",
    category: "数据",
    icon: "grid",
    description: "把长表转成宽表。",
    registeredRunner: false,
    params: [
      { key: "index", label: "行索引列", kind: "columns", multi: true, columnFilter: "any", required: true },
      { key: "columns", label: "展开为列的字段", kind: "columns", multi: true, columnFilter: "categorical", required: true },
      { key: "values", label: "取值列", kind: "columns", multi: true, columnFilter: "numeric", required: true },
      { key: "aggregation", label: "聚合方式", kind: "enum", options: AGG_OPTIONS, default: "mean" },
    ],
  },
  "data.melt": {
    type: "data.melt",
    label: "逆透视",
    category: "数据",
    icon: "table",
    description: "把宽表转成长表。",
    registeredRunner: false,
    params: [
      { key: "id_vars", label: "保持不变的列", kind: "columns", multi: true, columnFilter: "any" },
      { key: "value_vars", label: "转为值的列", kind: "columns", multi: true, columnFilter: "any" },
      { key: "variable_name", label: "变量列名", kind: "string", default: "variable" },
      { key: "value_name", label: "值列名", kind: "string", default: "value" },
    ],
  },
  "ml.train": {
    type: "ml.train",
    label: "训练模型",
    category: "机器学习",
    icon: "beaker",
    description: "在上游数据上训练一个监督或无监督模型。",
    registeredRunner: true,
    params: [
      { key: "model", label: "模型", kind: "enum", options: ML_MODELS, default: "random_forest_regressor", required: true },
      { key: "target_column", label: "目标列", kind: "columns", columnFilter: "any", required: true, hint: "要预测的列；聚类任务可留空。" },
      { key: "feature_columns", label: "特征列", kind: "columns", multi: true, columnFilter: "any", hint: "留空则使用除目标列外的所有列。" },
      { key: "test_size", label: "测试集比例", kind: "number", default: 0.2 },
      { key: "random_state", label: "随机种子", kind: "number", default: 42, hint: "固定种子可复现结果。" },
    ],
  },
  "ml.predict": {
    type: "ml.predict",
    label: "生成预测",
    category: "机器学习",
    icon: "sparkles",
    description: "用已训练的模型对新数据打分。",
    registeredRunner: true,
    params: [
      { key: "target_column", label: "目标列", kind: "columns", columnFilter: "any", required: true },
      { key: "output_column", label: "预测结果列名", kind: "string", default: "prediction" },
    ],
  },
  "ml.evaluate": {
    type: "ml.evaluate",
    label: "模型评估",
    category: "机器学习",
    icon: "chart",
    description: "计算模型的评估指标。",
    registeredRunner: true,
    params: [
      { key: "target_column", label: "目标列", kind: "columns", columnFilter: "any", required: true },
      { key: "metrics", label: "指标", kind: "json", placeholder: '["r2", "rmse", "mae"]' },
    ],
  },
  "ml.cluster": {
    type: "ml.cluster",
    label: "聚类分析",
    category: "机器学习",
    icon: "grid",
    description: "无监督分群，为每行打上簇标签。",
    registeredRunner: true,
    params: [
      { key: "model", label: "算法", kind: "enum", options: [{ value: "kmeans", label: "K-Means" }, { value: "dbscan", label: "DBSCAN" }], default: "kmeans" },
      { key: "n_clusters", label: "簇数量", kind: "number", default: 3, visibleWhen: { key: "model", equals: "kmeans" } },
      { key: "feature_columns", label: "参与聚类的列", kind: "columns", multi: true, columnFilter: "numeric" },
      { key: "output_column", label: "簇标签列名", kind: "string", default: "cluster" },
    ],
  },
  "ml.pca": {
    type: "ml.pca",
    label: "主成分分析",
    category: "机器学习",
    icon: "layers",
    description: "降维并保留主要信息量。",
    registeredRunner: true,
    params: [
      { key: "n_components", label: "主成分数量", kind: "number", default: 2 },
      { key: "feature_columns", label: "参与降维的列", kind: "columns", multi: true, columnFilter: "numeric" },
    ],
  },
  "ai.analyze": {
    type: "ai.analyze",
    label: "AI 分析",
    category: "AI",
    icon: "sparkles",
    description: "让大模型基于上游结果生成结论文本。",
    registeredRunner: true,
    params: [{ key: "prompt", label: "分析要求", kind: "string", required: true, placeholder: "请总结这个流程的主要发现与下一步建议。" }],
  },
};

export const NODE_TYPES = Object.keys(NODE_SPECS);

export const NODE_LABELS: Record<string, string> = Object.fromEntries(
  Object.values(NODE_SPECS).map((spec) => [spec.type, spec.label]),
);

export function specOf(type: string): NodeSpec | undefined {
  return NODE_SPECS[type];
}

export function categoryOf(type: string): NodeCategory | "其他" {
  return NODE_SPECS[type]?.category ?? "其他";
}

/**
 * 由后端 `build_default_runners()` 显式注册的节点。
 *
 * 其余 data.* 节点由 `make_op` 按 OPERATION_REGISTRY 动态注册，同样可执行。
 */
export const EXPLICIT_RUNNER_TYPES = [
  "noop",
  "data.load",
  "ml.train",
  "ml.predict",
  "ml.evaluate",
  "ml.cluster",
  "ml.pca",
  "ai.analyze",
  "data.quality_check",
  "data.statistics",
  "report.summary",
];

/**
 * 后端显式注册、但前端节点库还没提供的节点。
 *
 * 这是审查 #2 的真实缺口方向：不是"前端没跟上后端"，而是这几项
 * 后端已可运行、节点库里却选不到。
 */
export const MISSING_FROM_PALETTE = EXPLICIT_RUNNER_TYPES.filter(
  (type) => !(type in NODE_SPECS),
);
