import type { SchemaColumn } from "../../types/dataset";

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "逻辑回归",
  knn_classifier: "K近邻分类",
  decision_tree_classifier: "决策树分类",
  random_forest_classifier: "随机森林分类",
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
