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

/** GET /datasets/{id}/versions —— 版本时间线（复用 DatasetVersion + Operation）。 */
export interface VersionTimelineEntry {
  id: number;
  dataset_id: number;
  version: number;
  parent_version_id: number | null;
  storage_path: string;
  format: string;
  row_count: number;
  column_count: number;
  schema: Record<string, unknown>;
  created_at: string | null;
  /** 版本来源：import（导入）或产出它的 operation 类型（filter / duplicate / ...）。 */
  origin: string;
  /** 来源的中文展示名（来自后端 OPERATION_METADATA，前端不另抄一份）。 */
  origin_label: string;
  operation: OperationRecord | null;
  /** 父版本号；父版本不在本次返回范围内时为 row id（历史数据），无父版本为 null。 */
  parent_version: number | null;
  /** 相对父版本的变化；无父版本时为 null（不是 0 —— 0 表示「没变」）。 */
  delta_rows: number | null;
  delta_columns: number | null;
}

export interface OperationRecord {
  id: number;
  dataset_id: number;
  input_version_id: number | null;
  output_version_id: number | null;
  operation_type: string;
  parameters: Record<string, unknown>;
  status: string;
  error: string | null;
  created_at: string | null;
}

export interface VersionTimelineResult {
  dataset_id: number;
  total: number;
  latest_version: number;
  versions: VersionTimelineEntry[];
}

/** GET /datasets/{id}/versions/diff */
export interface SchemaChange {
  column: string;
  dtype?: string;
  from?: string;
  to?: string;
}

export interface QualitySummary {
  issue_count: number;
  missing_cells: number;
  duplicate_rows: number;
  severity?: Record<string, number>;
}

export interface VersionDiffResult {
  dataset_id: number;
  base: DatasetVersion;
  target: DatasetVersion;
  row_count: { base: number; target: number; delta: number };
  column_count: { base: number; target: number; delta: number };
  schema_diff: {
    added: SchemaChange[];
    removed: SchemaChange[];
    type_changed: SchemaChange[];
    unchanged_count: number;
  };
  /** include_quality=false 时为 null（大表可跳过，只保留结构差异）。 */
  quality: { base: QualitySummary; target: QualitySummary } | null;
  operations: OperationRecord[];
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
