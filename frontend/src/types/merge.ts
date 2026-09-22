/** Merge 相关类型（对齐后端 /merge 端点）。 */

/** MappingCandidate（schema_mapper.py 输出）。 */
export interface MappingCandidate {
  source_column: string; // 左表列
  target_column: string; // 右表列
  confidence: number;
  reason: string;
  evidence?: Record<string, unknown>;
}

/** KeyProfile（key_analyzer.py 输出）。 */
export interface KeyProfile {
  column: string;
  dtype: string;
  row_count: number;
  unique_count: number;
  null_count: number;
  duplicate_count: number;
  coverage: number;
  /** composite key 时额外字段 */
  columns?: string[];
}

export interface KeyAnalysisResult {
  left: KeyProfile;
  right: KeyProfile;
  cardinality: "one-to-one" | "one-to-many" | "many-to-one" | "many-to-many";
  left_key_coverage_in_right: number;
  right_key_coverage_in_left: number;
}

/** JoinKey（plan 内）。 */
export interface JoinKey {
  left: string;
  right: string;
}

/** ColumnMapping（执行阶段的右列→输出重命名）。 */
export interface ColumnMapping {
  right_column: string;
  output_column: string;
}

export interface MergePlan {
  keys: JoinKey[];
  mapping: ColumnMapping[];
  join_type: "inner" | "left" | "right" | "outer";
  left_suffix?: string;
  right_suffix?: string;
  right_columns?: string[] | null;
  warnings?: string[];
}

export interface MergePairRequest {
  left_dataset_id: number;
  right_dataset_id: number;
  left_version: number | null;
  right_version: number | null;
}

export interface ValidationResult {
  ok: boolean;
  errors: string[];
  warnings: string[];
  details: Record<string, unknown>;
}

export interface MergePreviewResult {
  left_version: number;
  right_version: number;
  input_rows_left: number;
  input_rows_right: number;
  input_columns_left: number;
  input_columns_right: number;
  output_rows: number;
  output_columns: number;
  matched_rows: number;
  unmatched_rows_left: number;
  unmatched_rows_right: number;
  warnings: string[];
  conflicts: string[];
  output_columns_list: string[];
  preview: {
    columns: string[];
    items: Record<string, unknown>[];
    total: number;
  };
}

export interface MergeReport {
  join_type: string;
  input_rows_left: number;
  input_rows_right: number;
  output_rows: number;
  output_columns: number;
  matched_rows: number;
  unmatched_rows_left: number;
  unmatched_rows_right: number;
  duplicate_keys_left: number;
  duplicate_keys_right: number;
  conflicts: string[];
  warnings: string[];
}

export interface MergeExecuteResult {
  version: {
    id: number;
    dataset_id: number;
    version: number;
    parent_version_id: number | null;
    storage_path: string;
    format: string;
    row_count: number;
    column_count: number;
    schema: Record<string, unknown>;
  };
  report: MergeReport;
}
