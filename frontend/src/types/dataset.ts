/** Dataset 相关类型（对齐后端 /datasets 端点与 analyze_schema / preview / quality 输出）。 */

export interface DatasetVersion {
  id: number;
  dataset_id: number;
  version: number;
  parent_version_id: number | null;
  storage_path: string;
  format: string;
  row_count: number;
  column_count: number;
  schema: Record<string, unknown>;
}

export interface Dataset {
  id: number;
  name: string;
  description: string;
  source_file_id: number | null;
  created_at: string;
  updated_at: string;
  latest_version: DatasetVersion | null;
  /** 兼容部分旧页面的摘要字段；完整统计优先使用 latest_version。 */
  row_count?: number;
  column_count?: number;
}

export interface DatasetListResult {
  items: Dataset[];
  page_info: { page: number; page_size: number; total: number };
  /** 兼容旧调用方；后端规范字段为 page_info.total。 */
  total?: number;
}

/** GET /datasets/{id}/schema —— 对齐 analyze_schema 输出。 */
export interface SchemaColumn {
  column: string;
  dtype: string;
  nullable: boolean;
  null_count: number;
  unique_count: number;
  sample_values: unknown[];
}

export interface SchemaData {
  row_count: number;
  column_count: number;
  columns: SchemaColumn[];
  version?: number;
}

/** GET /datasets/{id}/profile。 */
export interface ProfileData {
  row_count: number;
  column_count: number;
  columns?: Record<string, unknown>[];
  [key: string]: unknown;
}

/** GET /datasets/{id}/quality —— 对齐 QualityReport.to_dict()。 */
export interface QualityIssue {
  type?: string;
  column?: string;
  message?: string;
  [key: string]: unknown;
}

export interface QualityData {
  issues: QualityIssue[];
  severity?: Record<string, unknown>;
  statistics: {
    row_count: number;
    column_count: number;
    checked_by: string[];
    missing_cells: number;
    duplicate_rows: number;
    issue_count: number;
  };
  recommendations?: string[];
  version?: number;
}

/** GET /datasets/{id}/preview —— 服务端分页预览。 */
export interface PreviewData {
  columns: string[];
  items: Record<string, unknown>[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  version: number;
}
