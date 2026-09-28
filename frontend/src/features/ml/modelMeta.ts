import type { SchemaColumn } from "../../types/dataset";

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "逻辑回归",
  knn_classifier: "K近邻分类",
  decision_tree_classifier: "决策树分类",
  random_forest_classifier: "随机森林分类",
  hist_gradient_boosting_classifier: "直方图梯度提升",
  linear_regression: "线性回归",
  knn_regressor: "K近邻回归",
  decision_tree_regressor: "决策树回归",
  random_forest_regressor: "随机森林回归",
  kmeans: "K-Means聚类",
  dbscan: "DBSCAN密度聚类",
  pca: "PCA主成分分析",
};

export const MODEL_DESC: Record<string, string> = {
  logistic_regression: "线性分类模型，适合二分类/多分类，速度快可解释性强",
  knn_classifier: "基于距离的分类器，适合小数据集",
  decision_tree_classifier: "树形决策规则，可解释性极强",
  random_forest_classifier: "集成多棵决策树，抗过拟合",
  hist_gradient_boosting_classifier: "先分箱再逐轮提升，表格数据上通常比随机森林更快且略准",
  linear_regression: "最简单的回归模型，假设线性关系",
  knn_regressor: "基于距离的回归，适合局部模式",
  decision_tree_regressor: "树形回归，捕捉非线性关系",
  random_forest_regressor: "集成回归，适合非线性关系",
  kmeans: "划分式聚类，需指定簇数，适合球形簇",
  dbscan: "密度聚类，自动发现簇数，可识别噪声点",
  pca: "主成分分析，降维+特征提取，可视化高维数据",
};

export function modelLabel(name: string): string {
  return MODEL_LABELS[name] ?? name;
}

export function modelDesc(name: string): string | undefined {
  return MODEL_DESC[name];
}

export function recommendParams(
  model: string,
  columns: SchemaColumn[],
  target?: string | null,
  rowCount = 0,
): Record<string, unknown> {
  const numericCols = columns.filter((c) => {
    const dtype = c.dtype.toLowerCase();
    return dtype.includes("int") || dtype.includes("float") || dtype.includes("decimal");
  });
  const featureCount = target
    ? numericCols.filter((c) => c.column !== target).length
    : numericCols.length;

  switch (model) {
    case "kmeans": {
      if (rowCount < 2 || featureCount < 1) return {};
      const upperByFeatures = Math.max(2, Math.min(10, Math.ceil(featureCount / 5)));
      const k = Math.min(upperByFeatures, rowCount);
      return { n_clusters: Math.max(2, k), n_init: 10 };
    }
    case "dbscan":
      return {
        eps: 0.3,
        min_samples: Math.max(3, Math.min(20, Math.floor(Math.max(featureCount, 2) / 2))),
      };
    case "random_forest_classifier":
      return {
        n_estimators: 100,
        max_depth: Math.max(3, Math.min(20, Math.ceil(Math.max(featureCount, 1) * 1.5))),
        min_samples_leaf: 1,
        min_samples_split: 2,
        max_features: "sqrt",
      };
    case "random_forest_regressor":
      return {
        n_estimators: 100,
        max_depth: Math.max(3, Math.min(20, Math.ceil(Math.max(featureCount, 1) * 1.5))),
        min_samples_leaf: 1,
        min_samples_split: 2,
      };
    case "logistic_regression":
      return { max_iter: 2000, C: 1.0, class_weight: "balanced" };
    case "hist_gradient_boosting_classifier":
      return {
        max_iter: 100,
        learning_rate: 0.1,
        // 上限 31：内部用 8 位存叶节点索引，超过会直接报错
        max_leaf_nodes: 31,
        min_samples_leaf: 20,
        l2_regularization: 0.0,
      };
    case "decision_tree_classifier":
      return {
        max_depth: Math.max(3, Math.min(15, Math.max(1, Math.ceil(featureCount)))),
        min_samples_leaf: 1,
        min_samples_split: 2,
        criterion: "gini",
      };
    case "decision_tree_regressor":
      return {
        max_depth: Math.max(3, Math.min(15, Math.max(1, Math.ceil(featureCount)))),
        min_samples_leaf: 1,
        min_samples_split: 2,
        criterion: "squared_error",
      };
    case "knn_classifier":
    case "knn_regressor": {
      const trainRows = rowCount > 0 ? Math.floor(rowCount * 0.8) : 0;
      if (trainRows < 2) return {};
      const upperBySize = trainRows - 1;
      const k = Math.min(15, Math.max(1, Math.ceil(Math.sqrt(Math.max(featureCount, 1)))), upperBySize);
      return { n_neighbors: Math.max(1, k), weights: "uniform" };
    }
    case "pca": {
      const upper = Math.min(rowCount, featureCount);
      if (upper < 1) return {};
      return { n_components: Math.min(Math.max(1, upper), 5) };
    }
    default:
      return {};
  }
}

export type ChartType =
  | "metrics_bar"
  | "metrics_radar"
  | "cluster_bar"
  | "feature_histogram"
  | "feature_scatter"
  | "target_distribution";

export const PARAM_LABELS: Record<string, string> = {
  n_estimators: "树数量",
  max_depth: "最大深度",
  n_neighbors: "邻居数",
  n_clusters: "聚类数量",
  eps: "邻域半径 eps",
  min_samples: "最小样本数",
  max_iter: "最大迭代次数",
  n_components: "主成分数量",
  C: "正则强度倒数 C",
  class_weight: "类别权重",
  weights: "距离权重",
  min_samples_leaf: "叶最小样本数",
  min_samples_split: "分裂最小样本数",
  criterion: "分裂准则",
  max_features: "每树随机特征数",
  n_init: "初始化次数",
  learning_rate: "学习率",
  max_leaf_nodes: "单树最大叶子数",
  l2_regularization: "L2 正则强度",
  max_bins: "分箱数量",
  early_stopping: "早停",
};

function isNumericColumn(c: SchemaColumn): boolean {
  const t = c.dtype.toLowerCase();
  return t.includes("int") || t.includes("float") || t.includes("decimal");
}

export function recommendCharts(
  task: string,
  columns?: SchemaColumn[],
  target?: string | null,
  _model?: string,
): ChartType[] {
  const numericCount = (columns ?? []).filter((c) => isNumericColumn(c) && c.column !== target).length;

  switch (task) {
    case "classification": {
      const charts: ChartType[] = ["metrics_bar", "metrics_radar"];
      if (numericCount >= 2) charts.push("feature_scatter");
      if (target) charts.push("target_distribution");
      return charts;
    }
    case "regression": {
      const charts: ChartType[] = ["metrics_bar"];
      if (numericCount >= 2) charts.push("feature_scatter");
      if (target) charts.push("target_distribution");
      return charts;
    }
    case "clustering": {
      const charts: ChartType[] = [];
      if (numericCount >= 2) charts.push("feature_scatter");
      charts.push("cluster_bar");
      if (numericCount >= 1) charts.push("feature_histogram");
      return charts;
    }
    default:
      return ["metrics_bar"];
  }
}

export const CHART_LABELS: Record<ChartType, string> = {
  metrics_bar: "指标柱状图",
  metrics_radar: "指标雷达图",
  cluster_bar: "簇分布图",
  feature_histogram: "特征直方图",
  feature_scatter: "特征散点图",
  target_distribution: "目标分布图",
};

// ===== 目标列推荐：页面自动选中与选择器提示的唯一事实源 =====
const FLOAT_RE = /float|double|decimal/i;
const NUMERIC_RE = /int|float|double|decimal/i;

export interface TargetRecommendation {
  column: string | null;
  reason: string;
  alternatives: string[]; // 同样像标签但未排第一的列
}

export function recommendTargetColumn(
  columns: SchemaColumn[],
  task: string,
): TargetRecommendation {
  if (task === "clustering") {
    return { column: null, reason: "聚类任务不需要目标列", alternatives: [] };
  }
  if (task === "regression") {
    const numeric = columns.filter((c) => NUMERIC_RE.test(c.dtype));
    if (!numeric.length) {
      return { column: null, reason: "未发现数值型列，回归任务无可用目标", alternatives: [] };
    }
    const named = numeric.find((c) =>
      /target|label|price|score|amount|(^|[_\s-])y([_\s-]|$)/i.test(c.column));
    const pick = named ?? numeric[0];
    return {
      column: pick.column,
      reason: named
        ? `数值型列且名称匹配常见回归目标关键词`
        : "未发现名称像目标的数值列，已选第一个数值列，请确认",
      alternatives: numeric.filter((c) => c.column !== pick.column).slice(0, 3).map((c) => c.column),
    };
  }
  // classification：二值标签 >> 三值 >> 多类低基数；列名信号加权
  const scored = columns
    .filter((c) => !FLOAT_RE.test(c.dtype) && c.unique_count >= 2)
    .map((c) => {
      let score = 0;
      if (c.unique_count === 2) score += 100;
      else if (c.unique_count === 3) score += 60;
      else if (c.unique_count <= 20) score += 20;
      if (/^(y|target|label|class|cls)$/i.test(c.column)) score += 80;
      else if (/target|label|class|标签|目标/i.test(c.column)) score += 40;
      return { c, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.c.column.localeCompare(b.c.column));
  if (!scored.length) {
    return { column: null, reason: "未发现低基数非浮点列，分类任务请手动指定目标", alternatives: [] };
  }
  const best = scored[0];
  return {
    column: best.c.column,
    reason: `unique=${best.c.unique_count}${best.score >= 150 ? "，列名与基数都符合标签特征" : "，按低基数启发式推荐"}——请确认这是你要预测的对象`,
    alternatives: scored.slice(1, 4).map((x) => x.c.column),
  };
}

/** 疑似标签泄漏警告：像标签的列被当成了普通特征。 */
export function labelLeakWarnings(
  columns: SchemaColumn[],
  target: string | null,
  excluded: string[],
  rowCount: number,
): string[] {
  return columns
    .filter((c) =>
      c.column !== target &&
      !excluded.includes(c.column) &&
      !FLOAT_RE.test(c.dtype) &&
      c.unique_count >= 2 &&
      c.unique_count <= 3 &&
      c.unique_count !== rowCount)
    .map((c) =>
      `「${c.column}」只有 ${c.unique_count} 个取值，很可能是真实标签。确认要作为特征使用吗？`);
}

/**
 * 指标展示格式：整数型指标（计数 / 种子 / 簇数）不显示四位小数。
 *
 * 单一实现，供结果页指标卡与 ML 页历史表共用 —— 两处各写一份迟早会出现
 * 「结果页显示 1534、历史表显示 1534.0000」这类不一致。
 */
export function formatMetric(key: string, value: unknown): string {
  if (value == null) return "-";
  if (typeof value !== "number") return String(value);
  if (/(_seed$|_count$|^n_[a-z]+$|^k$|^cluster_count$)/.test(key)) {
    return String(Math.round(value));
  }
  return value.toFixed(4);
}
