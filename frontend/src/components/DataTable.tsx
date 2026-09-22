import { useState, useMemo } from "react";

export interface DataTableColumn<T> {
  key: string;
  title: string;
  render?: (row: T) => React.ReactNode;
  sortable?: boolean;
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  loading?: boolean;
  emptyText?: string;
  error?: string | null;
  onRowClick?: (row: T) => void;
  // 服务端排序（预览表）；不传则用前端内存排序
  onSortChange?: (column: string, desc: boolean) => void;
}

/** Prompt 152：通用数据表 —— 分页 / 排序 / 加载 / 空 / 错误状态。 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  loading,
  emptyText = "暂无数据",
  error,
  onRowClick,
  onSortChange,
}: DataTableProps<T>) {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(15);
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDesc, setSortDesc] = useState(false);

  const sorted = useMemo(() => {
    if (!sortKey || onSortChange) return rows;
    return [...rows].sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortKey];
      const bv = (b as Record<string, unknown>)[sortKey];
      if (av === bv) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv), "zh-CN");
      return sortDesc ? -cmp : cmp;
    });
  }, [rows, sortKey, sortDesc, onSortChange]);

  const total = sorted.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const current = sorted.slice((page - 1) * pageSize, page * pageSize);

  if (error) return <div className="badge failed">加载失败：{error}</div>;
  if (loading) return <div className="muted">加载中...</div>;
  if (!rows.length) return <div className="muted">{emptyText}</div>;

  return (
    <div>
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={col.sortable ? "sortable" : ""}
                onClick={() => {
                  if (!col.sortable) return;
                  if (onSortChange) {
                    const desc = sortKey === col.key ? !sortDesc : false;
                    setSortKey(col.key);
                    setSortDesc(desc);
                    onSortChange(col.key, desc);
                  } else {
                    setSortKey(col.key);
                    setSortDesc(sortKey === col.key ? !sortDesc : false);
                  }
                }}
              >
                {col.title}
                {sortKey === col.key ? (sortDesc ? " ↓" : " ↑") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {current.map((row) => (
            <tr
              key={rowKey(row)}
              className={onRowClick ? "clickable" : ""}
              onClick={() => onRowClick?.(row)}
            >
              {columns.map((col) => (
                <td key={col.key}>
                  {col.render
                    ? col.render(row)
                    : String((row as Record<string, unknown>)[col.key] ?? "-")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex-between mt">
        <span className="muted">
          共 {total} 条 · 第 {page}/{totalPages} 页
        </span>
        <div style={{ display: "flex", gap: "var(--space-2)" }}>
          <button className="btn" disabled={page <= 1} onClick={() => setPage(page - 1)}>
            上一页
          </button>
          <button
            className="btn"
            disabled={page >= totalPages}
            onClick={() => setPage(page + 1)}
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}
