/** 文件 API。 */

import { client, unwrap } from "./client";
import type { Pagination } from "../types/common";
import type { LabFile } from "../types/report";

export const FILE_MAX_SIZE = 100 * 1024 * 1024;

export const FILE_ALLOWED_EXTENSIONS = [
  ".csv",
  ".json",
  ".xlsx",
  ".xls",
  ".parquet",
] as const;

export function listFiles(
  page = 1,
  pageSize = 20,
) {
  return unwrap<Pagination<LabFile>>(
    client.get("/files", {
      params: {
        page,
        page_size: pageSize,
      },
    }),
  );
}

export function uploadFile(
  file: File,
  onProgress?: (pct: number) => void,
) {
  const form = new FormData();

  form.append(
    "file",
    file,
  );

  return unwrap<LabFile>(
    client.post(
      "/files/upload",
      form,
      {
        // 不手动设置 multipart/form-data。
        // 浏览器需要自动生成带 boundary 的 Content-Type，否则某些环境下
        // FastAPI/Starlette 无法正确解析 multipart body。
        onUploadProgress: (event) => {
          if (
            onProgress &&
            event.total
          ) {
            const pct = Math.min(
              100,
              Math.round(
                (event.loaded /
                  event.total) *
                  100,
              ),
            );

            onProgress(pct);
          }
        },
      },
    ),
  );
}

export function deleteFile(
  id: number,
) {
  return unwrap<{
    deleted: boolean;
    file_id: number;
  }>(
    client.delete(
      `/files/${id}`,
    ),
  );
}

export function fileDownloadUrl(
  id: number,
) {
  const base =
    import.meta.env
      .VITE_API_BASE_URL ?? "";

  const normalized = String(base).trim().replace(/\/+$/, "");
  return normalized && /\/api\/v1$/i.test(normalized)
    ? `${normalized}/files/${id}/content`
    : `${normalized}/api/v1/files/${id}/content`;
}

export function formatFileSize(
  size: number,
): string {
  if (size < 1024) {
    return `${size} B`;
  }

  if (size < 1024 ** 2) {
    return `${(
      size / 1024
    ).toFixed(1)} KB`;
  }

  if (size < 1024 ** 3) {
    return `${(
      size /
      1024 ** 2
    ).toFixed(2)} MB`;
  }

  return `${(
    size /
    1024 ** 3
  ).toFixed(2)} GB`;
}

export function getFileFormatLabel(
  format: string,
): string {
  const labels: Record<
    string,
    string
  > = {
    csv: "CSV",
    json: "JSON",
    xlsx: "Excel",
    xls: "Excel",
    parquet: "Parquet",
  };

  return (
    labels[
      format.toLowerCase()
    ] ??
    format.toUpperCase()
  );
}
