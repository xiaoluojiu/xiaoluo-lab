/** Reports API。 */
import { client, unwrap } from "./client";
import type { Report } from "../types/report";

export function generateReport(body: {
  dataset_id: number;
  version?: number | null;
  title?: string;
  include_quality?: boolean;
  include_eda?: boolean;
  include_ml?: boolean;
  conclusions?: string[];
}) {
  return unwrap<Report>(client.post("/reports/generate", body));
}

export type ExportFormat = "markdown" | "html" | "pdf";

export async function exportReport(report: Report, format: ExportFormat): Promise<Blob> {
  const resp = await client.post("/reports/export", { report, format }, {
    responseType: "blob",
  });
  return resp.data as Blob;
}

export interface SavedReportSummary {
  key: string;
  title: string;
  dataset: Record<string, unknown>;
  metadata: Record<string, unknown>;
  size: number;
  modified_at: string;
}

export function listSavedReports() {
  return unwrap<SavedReportSummary[]>(client.get("/reports/saved"));
}

export function getSavedReport(key: string) {
  return unwrap<Report>(
    client.get("/reports/saved/detail", { params: { key } }),
  );
}

export function deleteSavedReport(key: string) {
  return unwrap<{ deleted: boolean; key: string }>(
    client.delete("/reports/saved", { params: { key } }),
  );
}
