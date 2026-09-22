/** 与 Backend schemas/common.py 对齐的统一响应结构。 */

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: ApiError | null;
  request_id: string | null;
}

export interface PageInfo {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface Pagination<T> {
  items: T[];
  page_info: PageInfo;
}
