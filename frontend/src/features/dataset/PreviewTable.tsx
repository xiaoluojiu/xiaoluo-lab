import { useEffect, useState } from "react";
import { previewDataset } from "../../api/datasets";
import type { PreviewData } from "../../types/dataset";
import { Skeleton } from "../../components/StateBlock";

interface Props {
  datasetId: number;
  version?: number;
  pageSize?: number;
  /** 只看这些列（缺省=全部列）。用于让预览跟随左侧字段勾选。 */
  columns?: string[];
}

// Prompt 160：分页预览表 —— 服务端分页，禁止一次加载全部数据。
export function PreviewTable({ datasetId, version, pageSize = 20, columns }: Props) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<string | undefined>();
  const [sortDesc, setSortDesc] = useState(false);

  // 数组引用每次都变，用 join 后的字符串做依赖，避免无谓的重复请求。
  const columnsKey = columns?.join(",");

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
      columns,
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
    // columns 通过 columnsKey（join 后的字符串）参与依赖，见上。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, version, page, pageSize, sortColumn, sortDesc, columnsKey]);

  if (error) return <div className="badge failed">{error}</div>;
  // 首次加载用表骨架（高度已知，避免表格出现时整页跳一下）；换页只加一行「刷新中」。
  if (loading && !data) return <Skeleton lines={4} card />;
  if (!data || !data.items.length) return <div className="muted">暂无数据</div>;

  // 表头直接用返回行自身的键：请求已按 columns 过滤，键就是实际展示的列。
  const tableColumns = Object.keys(data.items[0]);

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
              {tableColumns.map((c) => (
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
                {tableColumns.map((c) => (
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
