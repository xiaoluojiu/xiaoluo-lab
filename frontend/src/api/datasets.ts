/** Dataset API —— CRUD（/datasets）+ 只读分析（schema/profile/quality/preview）。 */
import { LONG_OPERATION_TIMEOUT_MS, client, unwrap } from "./client";
import type {
  Dataset,
  DatasetListResult,
  PreviewData,
  ProfileData,
  QualityData,
  SchemaData,
  VersionDiffResult,
  VersionTimelineResult,
} from "../types/dataset";

export function listDatasets(page = 1, pageSize = 20) {
  return unwrap<DatasetListResult>(
    client.get("/datasets", { params: { page, page_size: pageSize } }),
  );
}

export function getDataset(datasetId: number) {
  return unwrap<Dataset>(client.get(`/datasets/${datasetId}`));
}

export function createDataset(name: string, description = "", sourceFileId?: number) {
  return unwrap<Dataset>(
    client.post(
      "/datasets",
      {
        name,
        description,
        source_file_id: sourceFileId ?? null,
      },
      // 带 sourceFileId 时会**在本次请求内完成入库**（解析 → 落 Parquet），
      // 耗时由文件体积决定：385 MB ARFF 约 6s，不可流式格式或更大文件更久。
      // 用 60s 兜底会把正常处理误报成失败。
      { timeout: LONG_OPERATION_TIMEOUT_MS },
    ),
  );
}

export function deleteDataset(datasetId: number) {
  return unwrap<Record<string, unknown>>(client.delete(`/datasets/${datasetId}`));
}

export function getSchema(datasetId: number, version?: number) {
  return unwrap<SchemaData>(
    client.get(`/datasets/${datasetId}/schema`, {
      params: { version: version ?? undefined },
    }),
  );
}

export function getProfile(datasetId: number, version?: number) {
  return unwrap<ProfileData>(
    client.get(`/datasets/${datasetId}/profile`, {
      params: { version: version ?? undefined },
    }),
  );
}

export function getQuality(datasetId: number, version?: number) {
  return unwrap<QualityData>(
    client.get(`/datasets/${datasetId}/quality`, {
      params: { version: version ?? undefined },
    }),
  );
}

export function getVersionTimeline(datasetId: number) {
  return unwrap<VersionTimelineResult>(client.get(`/datasets/${datasetId}/versions`));
}

export function getVersionDiff(datasetId: number, base: number, target: number, includeQuality = true) {
  return unwrap<VersionDiffResult>(
    client.get(`/datasets/${datasetId}/versions/diff`, {
      params: { base, target, include_quality: includeQuality },
    }),
  );
}

export interface PreviewParams {
  page?: number;
  page_size?: number;
  version?: number;
  sort_column?: string;
  sort_desc?: boolean;
}

export function previewDataset(datasetId: number, params: PreviewParams = {}) {
  return unwrap<PreviewData>(
    client.get(`/datasets/${datasetId}/preview`, { params }),
  );
}
