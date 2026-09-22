/** 数据处理 API（操作目录 / 预览 / 执行 / 历史）。 */
import { client, unwrap } from "./client";

export interface OperationMeta {
  op_type: string;
  params: string[];
  label: string;
  description: string;
  risk: string;
}

export interface OperationPreview {
  input_version: number;
  operation: string;
  before: { rows: number; columns: number };
  after: { rows: number; columns: number };
  row_delta: number;
  column_delta: number;
  preview: {
    columns: string[];
    items: Array<Record<string, unknown>>;
    total: number;
    total_pages: number;
  };
}

export interface OperationResult {
  version: {
    id: number;
    version: number;
    parent_version_id: number | null;
    row_count: number;
    column_count: number;
  };
  operation: {
    id: number;
    operation_type: string;
    status: string;
    input_version_id: number | null;
    output_version_id: number | null;
  };
}

export interface OperationHistoryItem {
  id: number;
  operation_type: string;
  input_version_id: number | null;
  output_version_id: number | null;
  status: string;
  error: string | null;
  created_at: string | null;
}

/** 前端 Tab 名 -> 后端 op_type。 */
export type ProcessingKind =
  | "clean"
  | "duplicate"
  | "cast"
  | "string"
  | "filter"
  | "transform"
  | "aggregate"
  | "pivot"
  | "melt";

/** 前端 Tab 名 -> 后端 op_type。全链路唯一的 kind 映射，页面不得再复制一份。 */
export const OPERATION_KIND_MAP: Record<ProcessingKind, string> = {
  clean: "missing",
  duplicate: "duplicate",
  cast: "cast",
  string: "string",
  filter: "filter",
  transform: "transform",
  aggregate: "aggregate",
  pivot: "pivot",
  melt: "melt",
};

const KIND_MAP: Record<ProcessingKind, string> = OPERATION_KIND_MAP;

export function listOperations() {
  return unwrap<OperationMeta[]>(client.get("/processing/operations"));
}

export function previewProcessing(
  datasetId: number,
  kind: ProcessingKind,
  params: Record<string, unknown>,
  inputVersion?: number,
  pageSize = 20,
) {
  return unwrap<OperationPreview>(
    client.post(
      `/processing/datasets/${datasetId}/preview`,
      { params, input_version: inputVersion ?? null, page_size: pageSize },
      { params: { op_type: KIND_MAP[kind] } },
    ),
  );
}

export function runProcessing(
  datasetId: number,
  kind: ProcessingKind,
  params: Record<string, unknown>,
  inputVersion?: number,
) {
  return unwrap<OperationResult>(
    client.post(`/processing/datasets/${datasetId}/operation/${KIND_MAP[kind]}`, {
      params,
      input_version: inputVersion ?? null,
    }),
  );
}

export function getProcessingHistory(datasetId: number) {
  return unwrap<OperationHistoryItem[]>(
    client.get(`/processing/datasets/${datasetId}/history`),
  );
}
