import { useEffect, useState } from "react";
import { previewDataset } from "../../api/datasets";
import type { PreviewData } from "../../types/dataset";

interface Props {
  datasetId: number;
  version?: number;
  pageSize?: number;
}

// Prompt 160：分页预览表 —— 服务端分页，禁止一次加载全部数据。
export function PreviewTable({ datasetId, version, pageSize = 20 }: Props) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<string | undefined>();
  const [sortDesc, setSortDesc] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    previewDataset(datasetId, {
      page,
      page_size: pageSize,
      version,
      sort_column: sortColumn,
      sort_desc: sortDesc,
    })
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "预览失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, version, page, pageSize, sortColumn, sortDesc]);

  if (error) return <div className="badge failed">{error}</div>;
  if (loading && !data) return <div className="muted">加载中...</div>;
  if (!data || !data.items.length) return <div className="muted">暂无数据</div>;

  const columns = Object.keys(data.items[0]);

  function toggleSort(col: string) {
    if (sortColumn === col) setSortDesc(!sortDesc);
    else {
      setSortColumn(col);
      setSortDesc(false);
    }
    setPage(1);
  }

  return (
    <div>
      {loading && <div className="muted">刷新中...</div>}
      <div style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} className="sortable" onClick={() => toggleSort(c)}>
                  {c}
                  {sortColumn === c ? (sortDesc ? " ↓" : " ↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.items.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c}>{formatCell(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex-between mt">
        <span className="muted">
          共 {data.total} 行 · 版本 v{data.version} · 第 {page} 页（每页 {pageSize} 行）
        </span>
        <div style={{ display: "flex", gap: "var(--space-2)" }}>
          <button className="btn" disabled={page <= 1} onClick={() => setPage(page - 1)}>
            上一页
          </button>
          <button
            className="btn"
            disabled={page * pageSize >= data.total}
            onClick={() => setPage(page + 1)}
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}
