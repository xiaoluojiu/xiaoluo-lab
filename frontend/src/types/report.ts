export interface ReportTable {
  title: string;
  headers: string[];
  rows: unknown[][];
}

export interface ReportChart {
  type: string;
  title: string;
  data?: unknown;
  /** 内联 SVG 字符串（后端渲染，直接注入即可显示） */
  svg?: string;
}

export interface ReportSection {
  heading: string;
  content: string;
  tables: ReportTable[];
  charts: ReportChart[];
}

export interface Report {
  title: string;
  dataset: Record<string, unknown>;
  sections: ReportSection[];
  experiments: Record<string, unknown>[];
  charts: ReportChart[];
  conclusions: string[];
  metadata: Record<string, unknown>;
}

export interface LabFile {
  id: number;
  name: string;
  original_name: string;
  path: string;
  size: number;
  format: string;
  checksum: string;
  created_at: string;
}
