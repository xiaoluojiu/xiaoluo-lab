/** 数据库连接器（拓展功能）API。 */

import { LONG_OPERATION_TIMEOUT_MS, client, unwrap } from "./client";
import type { Pagination } from "../types/common";

/** 可用数据库类型（来自后端方言注册表，含驱动就绪状态）。 */
export interface DialectInfo {
  name: string;
  label: string;
  default_port: number | null;
  requires_host: boolean;
  driver_installed: boolean;
  driver_hint: string;
  enabled: boolean;
  server_side_cursor: boolean;
  notes: string;
}

export interface Connector {
  id: number;
  name: string;
  dialect: string;
  host: string | null;
  port: number | null;
  database: string;
  schema_name: string | null;
  username: string | null;
  options: Record<string, unknown>;
  /** 只表示「是否已配置口令」，口令原文永不下发。 */
  has_password: boolean;
  last_status: "unknown" | "ok" | "error";
  last_error: string;
  last_checked_at: string | null;
  dataset_id: number | null;
  last_import: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ConnectorPayload {
  name: string;
  dialect: string;
  host?: string | null;
  port?: number | null;
  database: string;
  schema_name?: string | null;
  username?: string | null;
  password?: string | null;
  options?: Record<string, unknown>;
  test_on_create?: boolean;
}

export interface ConnectorTestResult {
  ok: boolean;
  latency_ms: number;
  server_version: string | null;
  message: string;
  dialect: string;
  driver_installed: boolean;
}

export interface ConnectorTable {
  name: string;
  type: string;
  schema: string | null;
}

export interface ConnectorTableList {
  connector_id: number;
  dialect: string;
  schema: string | null;
  schemas: string[];
  tables: ConnectorTable[];
  total: number;
}

export interface ConnectorColumn {
  name: string;
  type: string;
  nullable: boolean;
  primary_key: boolean;
}

export interface ConnectorPreview {
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  limit: number;
  sql: string;
}

export interface ConnectorImportRequest {
  table: string;
  schema_name?: string | null;
  columns?: string[] | null;
  where?: string | null;
  order_by?: string | null;
  keyset_column?: string | null;
  batch_size?: number | null;
  max_rows?: number | null;
  dataset_id?: number | null;
  dataset_name?: string | null;
  dataset_description?: string;
}

export interface ConnectorImportResult {
  dataset_id: number;
  dataset_version: number;
  row_count: number;
  column_count: number;
  strategy: string;
  batches: number;
  elapsed_seconds: number;
  rows_per_second: number;
  truncated: boolean;
  warnings: string[];
}

export function getDialectCatalog() {
  return unwrap<DialectInfo[]>(client.get("/connectors/catalog"));
}

export function listConnectors(page = 1, pageSize = 50) {
  return unwrap<Pagination<Connector>>(
    client.get("/connectors", { params: { page, page_size: pageSize } }),
  );
}

export function createConnector(payload: ConnectorPayload) {
  return unwrap<Connector>(client.post("/connectors", payload));
}

export function updateConnector(id: number, payload: Partial<ConnectorPayload>) {
  return unwrap<Connector>(client.patch(`/connectors/${id}`, payload));
}

export function deleteConnector(id: number) {
  return unwrap<{ deleted: boolean; connector_id: number }>(
    client.delete(`/connectors/${id}`),
  );
}

/** 试连（可传未保存的配置）。 */
export function testConnector(payload: Partial<ConnectorPayload> & { dialect: string; database: string }) {
  return unwrap<ConnectorTestResult>(client.post("/connectors/test", payload));
}

/** 试连已保存的连接器（会回写 last_status）。 */
export function testSavedConnector(id: number) {
  return unwrap<ConnectorTestResult>(client.post(`/connectors/${id}/test`));
}

export function listConnectorTables(id: number, schema?: string) {
  return unwrap<ConnectorTableList>(
    client.get(`/connectors/${id}/tables`, { params: schema ? { schema } : {} }),
  );
}

export function describeConnectorTable(id: number, table: string, schema?: string) {
  return unwrap<{ connector_id: number; table: string; columns: ConnectorColumn[]; total: number }>(
    client.get(`/connectors/${id}/tables/${encodeURIComponent(table)}/columns`, {
      params: schema ? { schema } : {},
    }),
  );
}

export function previewConnectorTable(
  id: number,
  body: { table: string; schema_name?: string | null; columns?: string[] | null; where?: string | null; limit?: number },
) {
  return unwrap<ConnectorPreview>(client.post(`/connectors/${id}/preview`, body));
}

export function importConnectorTable(id: number, body: ConnectorImportRequest) {
  // 抽取整张表并在服务端落成 Parquet：耗时由表规模决定，可能分钟级。
  return unwrap<ConnectorImportResult>(
    client.post(`/connectors/${id}/import`, body, {
      timeout: LONG_OPERATION_TIMEOUT_MS,
    }),
  );
}

export function dialectLabel(dialect: string): string {
  const labels: Record<string, string> = {
    sqlite: "SQLite",
    postgresql: "PostgreSQL",
    mysql: "MySQL / MariaDB",
    mssql: "SQL Server",
    oracle: "Oracle",
  };
  return labels[dialect] ?? dialect;
}

export function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    ok: "连接正常",
    error: "连接失败",
    unknown: "未测试",
  };
  return labels[status] ?? status;
}
