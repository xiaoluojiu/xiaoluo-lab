/** EDA API —— 只读分析（Prompt 199），端点：/datasets/{id}/eda/*。 */
import { client, unwrap } from "./client";

type EdaResult = Record<string, unknown>;

function compact(params: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined),
  );
}

function columnsParam(columns?: string[]): string | undefined {
  return columns?.length ? columns.join(",") : undefined;
}

/** GET /datasets/{id}/eda/descriptive */
export function edaDescriptive(
  datasetId: number,
  options?: { version?: number; columns?: string[] },
) {
  return unwrap<EdaResult>(
    client.get(`/datasets/${datasetId}/eda/descriptive`, {
      params: compact({
        version: options?.version,
        columns: columnsParam(options?.columns),
      }),
    }),
  );
}

/** GET /datasets/{id}/eda/correlation */
export function edaCorrelation(
  datasetId: number,
  options?: { version?: number; columns?: string[]; method?: "pearson" | "spearman" },
) {
  return unwrap<EdaResult>(
    client.get(`/datasets/${datasetId}/eda/correlation`, {
      params: compact({
        version: options?.version,
        columns: columnsParam(options?.columns),
        method: options?.method,
      }),
    }),
  );
}

/** GET /datasets/{id}/eda/distribution */
export function edaDistribution(
  datasetId: number,
  column: string,
  options?: { version?: number; bins?: number; top_n?: number },
) {
  return unwrap<EdaResult>(
    client.get(`/datasets/${datasetId}/eda/distribution`, {
      params: compact({
        column,
        version: options?.version,
        bins: options?.bins,
        top_n: options?.top_n,
      }),
    }),
  );
}

/** GET /datasets/{id}/eda/outlier */
export function edaOutlier(
  datasetId: number,
  options?: {
    version?: number;
    columns?: string[];
    method?: "iqr" | "zscore";
    k?: number;
    z_threshold?: number;
  },
) {
  return unwrap<EdaResult>(
    client.get(`/datasets/${datasetId}/eda/outlier`, {
      params: compact({
        version: options?.version,
        columns: columnsParam(options?.columns),
        method: options?.method,
        k: options?.k,
        z_threshold: options?.z_threshold,
      }),
    }),
  );
}

export type VisualizeChart =
  | "histogram"
  | "bar"
  | "line"
  | "scatter"
  | "boxplot"
  | "heatmap"
  | "qq"
  | "grouped_bar"
  | "area";

/** GET /datasets/{id}/eda/visualize */
export function edaVisualize(
  datasetId: number,
  options: {
    chart: VisualizeChart;
    version?: number;
    column?: string;
    x?: string;
    y?: string;
    columns?: string[];
    group_by?: string;
    /** grouped_bar 的聚合方式 */
    agg?: "mean" | "sum" | "count" | "median";
    bins?: number;
    top_n?: number;
    sample_limit?: number;
  },
) {
  return unwrap<EdaResult>(
    client.get(`/datasets/${datasetId}/eda/visualize`, {
      params: compact({
        chart: options.chart,
        version: options.version,
        column: options.column,
        x: options.x,
        y: options.y,
        columns: columnsParam(options.columns),
        group_by: options.group_by,
        bins: options.bins,
        top_n: options.top_n,
        sample_limit: options.sample_limit,
      }),
    }),
  );
}
