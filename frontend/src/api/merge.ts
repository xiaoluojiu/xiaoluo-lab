/** Merge API —— T0-M2：合并 API 封装（mapping/keys/preview/validate/execute）。 */
import { client, unwrap } from "./client";
import type {
  ColumnMapping,
  KeyAnalysisResult,
  MappingCandidate,
  MergeExecuteResult,
  MergePlan,
  MergePreviewResult,
  ValidationResult,
} from "../types/merge";

interface MergePairBase {
  left_dataset_id: number;
  right_dataset_id: number;
  left_version: number | null;
  right_version: number | null;
}

interface KeyAnalysisBody extends MergePairBase {
  /** 单 key（兼容） */
  left_key?: string | null;
  right_key?: string | null;
  /** composite key */
  left_keys?: string[] | null;
  right_keys?: string[] | null;
}

interface MergePlanBody extends MergePairBase {
  plan: MergePlan;
}

/**
 * 字段映射候选：返回 source_column/target_column/confidence/reason。
 * source_column=左表列，target_column=右表列（仅描述对应关系，非执行重命名）。
 */
export function suggestMapping(body: MergePairBase) {
  return unwrap<MappingCandidate[]>(client.post("/merge/mapping", body));
}

/** Key 分析：单 key 或 composite key。 */
export function analyzeKeys(body: KeyAnalysisBody) {
  return unwrap<KeyAnalysisResult>(client.post("/merge/keys", body));
}

/** T0-M3：合并预览（不创建版本/Operation/不改原数据）。 */
export function previewMerge(body: MergePlanBody) {
  return unwrap<MergePreviewResult>(client.post("/merge/preview", body));
}

/** 计划校验（不执行）。 */
export function validateMerge(body: MergePlanBody) {
  return unwrap<ValidationResult>(client.post("/merge/validate", body));
}

/** T0-M4：执行合并（必须显式提供 left_version/right_version）。 */
export function executeMerge(body: MergePlanBody) {
  return unwrap<MergeExecuteResult>(client.post("/merge/execute", body));
}

/** 工具：从 MappingCandidate 构造 ColumnMapping（默认 output_column = target_column）。 */
export function candidateToMapping(candidate: MappingCandidate): ColumnMapping {
  return {
    right_column: candidate.target_column,
    output_column: candidate.target_column,
  };
}

// ==================== 多文件合并（N 个数据集纵向整合）====================

export interface MultiMergeRef {
  dataset_id: number;
  version?: number | null;
  label?: string | null;
}

export interface MultiMergeField {
  column: string;
  present_in: number[];
  in_all: boolean;
  in_count: number;
}

export interface MultiMergeFile {
  index: number;
  row_count: number;
  column_count: number;
  columns: string[];
  only_in_this: string[];
}

export interface MultiMergeAnalyzeResult {
  file_count: number;
  all_columns: string[];
  common_columns: string[];
  unique_columns: string[];
  field_presence: MultiMergeField[];
  files: MultiMergeFile[];
}

export interface MultiMergePreviewResult {
  row_count: number;
  column_count: number;
  columns: string[];
  input_rows: number[];
  preview: {
    columns: string[];
    items: Record<string, unknown>[];
    total?: number;
    page?: number;
    page_size?: number;
  };
}

export interface MultiMergeExecuteResult {
  dataset_id: number;
  version: { version: number; row_count: number; column_count: number };
  is_new: boolean;
}

/** 分析多个数据集的共有字段 / 独有字段，供前端勾选。 */
export function analyzeMultiMerge(datasets: MultiMergeRef[]) {
  return unwrap<MultiMergeAnalyzeResult>(
    client.post("/merge/multi/analyze", { datasets }),
  );
}

/** 内存预览多文件合并结果（不创建版本）。 */
export function previewMultiMerge(body: {
  datasets: MultiMergeRef[];
  columns: string[];
  add_source: boolean;
  source_column: string;
}) {
  return unwrap<MultiMergePreviewResult>(
    client.post("/merge/multi/preview", body),
  );
}

/** 执行多文件合并：生成新数据集 或 写入指定已有数据集的新版本。 */
export function executeMultiMerge(body: {
  datasets: MultiMergeRef[];
  columns: string[];
  add_source: boolean;
  source_column: string;
  create_new: boolean;
  target_dataset_id?: number | null;
  name?: string | null;
}) {
  return unwrap<MultiMergeExecuteResult>(
    client.post("/merge/multi/execute", body),
  );
}
