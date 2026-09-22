/** 统一 HTTP Client。
 *
 * - 开发环境默认走 Vite 同源代理 /api/v1，避免本地 API 地址配置导致请求悬挂。
 * - 生产环境可通过 VITE_API_BASE_URL 配置主机地址或完整 API 前缀。
 * - 统一超时 / 统一错误提取（ApiResponse 包装）
 * - 请求/响应拦截：附带 request_id 透传（后端返回头）
 */
import axios from "axios";
import type { ApiResponse } from "../types/common";

function normalizeBaseURL(raw: unknown): string {
  const value = String(raw ?? "").trim().replace(/\/+$/, "");

  if (!value) {
    return "/api/v1";
  }

  // 允许直接填写完整 API 前缀，例如 http://localhost:8000/api/v1。
  if (/\/api\/v1$/i.test(value)) {
    return value;
  }

  return `${value}/api/v1`;
}

// 本地开发时优先使用 Vite proxy。
// 这样即使本机 .env.local 中残留了旧的 API 地址，也不会导致页面一直 Pending。
const baseURL = import.meta.env.DEV
  ? "/api/v1"
  : normalizeBaseURL(import.meta.env.VITE_API_BASE_URL);

export const client = axios.create({
  baseURL,
  // 普通同步请求的兜底超时。真正的长耗时操作（Agent 对话、ML 训练）走流式接口
  // （fetch + ReadableStream，不受此超时限制）；这里只覆盖同步短请求，放宽到 60s，
  // 避免 MB 级数据集的推理 / 对比等被 15s 误杀。
  timeout: 60_000,
});

client.interceptors.response.use(
  (resp) => {
    const body = resp.data as ApiResponse<unknown>;
    if (body && typeof body === "object" && "success" in body) {
      if (!body.success) {
        const err = new Error(body.error?.message ?? "请求失败") as Error & {
          code?: string;
          details?: unknown;
        };
        err.code = body.error?.code;
        err.details = body.error?.details;
        return Promise.reject(err);
      }
      resp.data = body.data;
    }
    return resp;
  },
  (error) => {
    const body = error.response?.data as (ApiResponse<unknown> & { detail?: string }) | undefined;
    const status = error.response?.status;
    const message =
      body?.error?.message ??
      body?.detail ??
      (status ? `请求失败（HTTP ${status}）` : undefined) ??
      (axios.isCancel(error) ? "请求已取消" : undefined) ??
      error.message ??
      "网络错误";
    return Promise.reject(new Error(String(message)));
  },
);

export async function unwrap<T>(promise: Promise<{ data: T }>): Promise<T> {
  const resp = await promise;
  return resp.data;
}
