/** Dataset API —— CRUD（/datasets）+ 只读分析（schema/profile/quality/preview）。 */
import { client, unwrap } from "./client";
import type {
  Dataset,
  DatasetListResult,
  PreviewData,
  ProfileData,
  QualityData,
  SchemaData,
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
    client.post("/datasets", {
      name,
      description,
      source_file_id: sourceFileId ?? null,
    }),
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
