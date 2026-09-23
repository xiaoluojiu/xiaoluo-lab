/** 文件 API。 */

import { LONG_OPERATION_TIMEOUT_MS, client, unwrap } from "./client";
import type { Pagination } from "../types/common";
import type { LabFile } from "../types/report";

export const FILE_ALLOWED_EXTENSIONS = [
  ".csv",
  ".json",
  ".xlsx",
  ".xls",
  ".parquet",
  ".arff",
] as const;

/**
 * 上传上限的**兜底值**。
 *
 * 真实上限由后端配置下发（`GET /files/limits`）——此前前端硬编码 100 MB，
 * 结果是「后端调大了、前端还在拦」或反之。请优先使用 `getUploadLimits()`。
 */
export const FILE_MAX_SIZE = 2 * 1024 * 1024 * 1024;

export interface UploadLimits {
  max_size_bytes: number;
  chunk_size_bytes: number;
  streaming_ingest: boolean;
  allowed_extensions: string[];
}

let cachedLimits: UploadLimits | null = null;

/** 拉取后端上传限额（进程内缓存，失败时退回兜底常量）。 */
export async function getUploadLimits(): Promise<UploadLimits> {
  if (cachedLimits) return cachedLimits;

  try {
    const limits = await unwrap<UploadLimits>(client.get("/files/limits"));
    if (limits?.max_size_bytes) {
      cachedLimits = limits;
      return limits;
    }
  } catch {
    // 后端暂不可用时不阻断页面渲染，用兜底值让用户至少能看到错误提示。
  }

  return {
    max_size_bytes: FILE_MAX_SIZE,
    chunk_size_bytes: 4 * 1024 * 1024,
    streaming_ingest: true,
    allowed_extensions: [...FILE_ALLOWED_EXTENSIONS],
  };
}

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
        // 大文件上传（上限 2 GB）在慢链路上远超 60s，必须用长超时；
        // 进度条另有 onUploadProgress 反馈，不属于「卡住」。
        timeout: LONG_OPERATION_TIMEOUT_MS,
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
    arff: "ARFF",
  };

  return (
    labels[
      format.toLowerCase()
    ] ??
    format.toUpperCase()
  );
}
