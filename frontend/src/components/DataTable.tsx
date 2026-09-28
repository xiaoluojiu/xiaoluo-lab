import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

/** 列优先级：窄屏（<768px）只保留 high 列，medium / low 自动隐藏。 */
export type ColumnPriority = "high" | "medium" | "low";

export interface DataTableColumn<T> {
  key: string;
  title: string;
  render?: (row: T) => React.ReactNode;
  sortable?: boolean;
  /**
   * 窄屏优先级。**不写 = "medium"**（窄屏自动隐藏）；
   * 想让某列在窄屏常驻就显式写 "high"。整表都没有 high 时，
   * 组件兜底保留第一列，避免窄屏下表头变空壳。
   */
  priority?: ColumnPriority;
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  loading?: boolean;
  emptyText?: string;
  /**
   * 自定义空态（通常是带图标的 <EmptyState>）。
   * 传了就不再渲染 emptyText 那行灰字 —— 空列表是引导新用户的最佳位置，
   * 一行「暂无数据」给不出下一步。
   */
  empty?: React.ReactNode;
  error?: string | null;
  onRowClick?: (row: T) => void;
  // 服务端排序（预览表）；不传则用前端内存排序
  onSortChange?: (column: string, desc: boolean) => void;
  /** 行首展开箭头：展开后以「键 → 值」列出全部列（含窄屏隐藏的列）。默认开。 */
  expandable?: boolean;
  /** 表头「列显示」下拉：手动勾选要显示的列。默认开。 */
  columnToggle?: boolean;
}

/**
 * 窄屏断点。需求指定 <768px，故取 767.98px ——
 * 用小数是为了让 768 这个整数宽度落在「宽屏」一侧，不产生边界歧义。
 *
 * 为什么用 JS 判定而不是 CSS 媒体查询：一列最终显不显示，是
 * 「优先级规则」和「用户手动勾选」两次判定的叠加（手动优先）。
 * 媒体查询只能表达前者，一旦用户在窄屏手动勾上某个 medium 列，
 * 纯 CSS 方案就会出现「勾了但不生效」的死结。
 */
const NARROW_QUERY = "(max-width: 767.98px)";

/**
 * 订阅窄屏状态。写法与 MainLayout 的导航自动收起保持一致
 * （addEventListener + 可选调用兜底，兼容老 Safari 的 addListener）。
 */
function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState<boolean>(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(NARROW_QUERY).matches
      : false
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(NARROW_QUERY);
    // 首帧到 effect 之间窗口可能已经被缩放过，这里再对齐一次。
    setNarrow(media.matches);
    const onChange = (event: MediaQueryListEvent) => setNarrow(event.matches);
    media.addEventListener?.("change", onChange);
    return () => media.removeEventListener?.("change", onChange);
  }, []);

  return narrow;
}

/** Prompt 152：通用数据表 —— 分页 / 排序 / 加载 / 空 / 错误状态。
 *  Prompt 153 窄屏增强（<768px）：只保留 high 列 + 行首展开 + 表头「列显示」勾选。 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  loading,
  emptyText = "暂无数据",
  empty,
  error,
  onRowClick,
  onSortChange,
  expandable = true,
  columnToggle = true,
}: DataTableProps<T>) {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(15);
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDesc, setSortDesc] = useState(false);

  const narrow = useNarrowViewport();
  /** 用户手动勾选的结果：key → 是否显示。没记录的列一律走优先级规则（即「自动」）。 */
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(() => new Set());
  const [toggleOpen, setToggleOpen] = useState(false);
  const toggleRef = useRef<HTMLDivElement | null>(null);

  /**
   * 是否属于窄屏常驻列。未声明 priority 的列按 "medium" 处理 ——
   * 需求要的是「窄屏只留 high」，若把未声明的列当 high，页面上就永远看不出效果；
   * 反过来，标了 high 的列一定常驻，这是给调用方的确定性开关。
   */
  const isHighPriority = useCallback(
    (col: DataTableColumn<T>) => (col.priority ?? "medium") === "high",
    []
  );

  /** 自动规则：宽屏全显示；窄屏只留 high。手动勾选优先于它。 */
  const isVisible = useCallback(
    (col: DataTableColumn<T>) => overrides[col.key] ?? (!narrow || isHighPriority(col)),
    [overrides, narrow, isHighPriority]
  );

  const visibleColumns = useMemo(() => {
    const shown = columns.filter(isVisible);
    // 兜底：窄屏且整张表没有 high 列时至少留第一列 ——
    // 否则表头变成空壳，用户既看不出这行是什么，也没有可点的列头。
    return shown.length ? shown : columns.slice(0, 1);
  }, [columns, isVisible]);

  const visibleKeys = useMemo(() => new Set(visibleColumns.map((c) => c.key)), [visibleColumns]);

  /**
   * 窄屏下「当前看不见的列数」。按最终结果算（总列数 - 实际可见列数），
   * 而不是按规则算 —— 首列兜底、用户手动勾回来都会改变结果，
   * 否则提示里的数字会跟眼睛看到的不一致。
   */
  const autoHiddenCount = narrow ? columns.length - visibleColumns.length : 0;

  /** 排序生效的列被隐藏时给个提示，否则「点了排序却没反应」会像 bug。 */
  const hiddenSortLabel = useMemo(() => {
    if (!sortKey || visibleKeys.has(sortKey)) return "";
    const col = columns.find((c) => c.key === sortKey);
    return col ? `${col.title || col.key}${sortDesc ? " ↓" : " ↑"}` : "";
  }, [sortKey, sortDesc, columns, visibleKeys]);

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

  // 下拉面板：点击外部或按 Esc 关闭。放在早返回之前，保证 hook 调用顺序稳定。
  useEffect(() => {
    if (!toggleOpen) return;
    const onPointerDown = (event: MouseEvent) => {
      if (toggleRef.current && !toggleRef.current.contains(event.target as Node)) setToggleOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setToggleOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [toggleOpen]);

  const total = sorted.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const current = sorted.slice((page - 1) * pageSize, page * pageSize);

  /** 排序：原逻辑一字未改，只从内联回调提成具名函数（服务端 / 内存两路都保留）。 */
  const handleSort = (col: DataTableColumn<T>) => {
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
  };

  const toggleRow = (key: string | number) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleColumn = (col: DataTableColumn<T>) => {
    const visible = visibleKeys.has(col.key);
    // 至少留一列：全关掉表格会退化成空壳，而且用户再没有把它勾回来的入口。
    if (visible && visibleColumns.length <= 1) return;
    setOverrides((prev) => ({ ...prev, [col.key]: !visible }));
  };

  const cellNode = (col: DataTableColumn<T>, row: T) =>
    col.render ? col.render(row) : String((row as Record<string, unknown>)[col.key] ?? "-");

  // 表格骨架而不是一行「加载中...」：表格高度已知，骨架能避免内容跳一下。
  if (error) return <div className="badge failed">加载失败：{error}</div>;
  if (loading) return <TableSkeleton rows={Math.min(pageSize, 5)} cols={Math.max(columns.length, 3)} />;
  if (!rows.length) return <>{empty ?? <div className="muted">{emptyText}</div>}</>;

  const colSpan = visibleColumns.length + (expandable ? 1 : 0);
  // 工具条本身没有内容时不要占位（例如关掉「列显示」且在宽屏下）。
  const showToolbar = columnToggle || narrow || !!hiddenSortLabel;

  return (
    <div>
      {showToolbar && (
        <div style={toolbarStyle}>
          {narrow && (
            <span className="muted" style={toolbarNoteStyle}>
              {autoHiddenCount > 0
                ? `窄屏已自动隐藏 ${autoHiddenCount} 列，点行首箭头展开可看全部数据`
                : "点行首箭头展开可看全部数据"}
            </span>
          )}
          {hiddenSortLabel && (
            <span className="muted" style={toolbarNoteStyle}>
              当前按「{hiddenSortLabel}」排序
            </span>
          )}
          {columnToggle && (
            <div ref={toggleRef} style={toggleWrapStyle}>
              <button
                type="button"
                className="btn"
                style={toggleBtnStyle}
                aria-haspopup="true"
                aria-expanded={toggleOpen}
                onClick={() => setToggleOpen((open) => !open)}
              >
                列显示 {visibleColumns.length}/{columns.length}
                <span aria-hidden="true" style={{ marginLeft: 6 }}>▾</span>
              </button>
              {toggleOpen && (
                <div role="group" aria-label="选择要显示的列" style={panelStyle}>
                  <div style={panelHeadStyle}>
                    <span>勾选要显示的列</span>
                    {Object.keys(overrides).length > 0 && (
                      <button type="button" className="btn link" onClick={() => setOverrides({})}>
                        恢复自动
                      </button>
                    )}
                  </div>
                  {columns.map((col) => {
                    const checked = visibleKeys.has(col.key);
                    const forcedHidden = narrow && !isHighPriority(col) && !checked;
                    return (
                      <label key={col.key} className="inline-check" style={panelRowStyle}>
                        <input type="checkbox" checked={checked} onChange={() => toggleColumn(col)} />
                        <span style={{ minWidth: 0 }}>{col.title || col.key}</span>
                        {forcedHidden && <span style={tagStyle}>窄屏隐藏</span>}
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              {expandable && (
                <th scope="col" style={expandThStyle}>
                  <span className="sr-only">展开行</span>
                </th>
              )}
              {visibleColumns.map((col) => (
                <th
                  key={col.key}
                  scope="col"
                  className={col.sortable ? "sortable" : ""}
                  onClick={() => handleSort(col)}
                >
                  {col.title}
                  {sortKey === col.key ? (sortDesc ? " ↓" : " ↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {current.map((row) => {
              const key = rowKey(row);
              const expanded = expandedKeys.has(key);
              return (
                <Fragment key={key}>
                  <tr
                    className={[onRowClick ? "clickable" : "", expanded ? "row-active" : ""]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => onRowClick?.(row)}
                  >
                    {expandable && (
                      <td style={expandThStyle}>
                        <button
                          type="button"
                          style={expandBtnStyle}
                          aria-expanded={expanded}
                          title={expanded ? "收起此行" : "展开此行"}
                          onClick={(event) => {
                            // 行本身可能绑了 onRowClick（打开详情），展开不能顺带触发它。
                            event.stopPropagation();
                            toggleRow(key);
                          }}
                        >
                          <svg
                            viewBox="0 0 24 24"
                            width="14"
                            height="14"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.4"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                            style={{
                              transform: expanded ? "rotate(90deg)" : "none",
                              transition: "transform var(--transition-fast)",
                            }}
                          >
                            <path d="M9 6l6 6-6 6" />
                          </svg>
                          <span className="sr-only">{expanded ? "收起" : "展开"}</span>
                        </button>
                      </td>
                    )}
                    {visibleColumns.map((col) => (
                      <td key={col.key}>{cellNode(col, row)}</td>
                    ))}
                  </tr>
                  {expandable && expanded && (
                    <tr>
                      <td colSpan={colSpan} style={expandPanelStyle}>
                        {/* 键值对形态：列多时用自适应栅格铺开，比横向滚动好读 */}
                        <dl style={kvGridStyle}>
                          {columns.map((col) => (
                            <div key={col.key} style={{ minWidth: 0 }}>
                              <dt style={kvKeyStyle}>{col.title || col.key}</dt>
                              <dd style={kvValueStyle}>{cellNode(col, row)}</dd>
                            </div>
                          ))}
                        </dl>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

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

/* ---------------------------------------------------------------------------
   样式：全部走 tokens.css 变量，随主题（浅色 / 深色）自动切换。
   本组件按「只改 DataTable 一处」的约束做成自包含，因此用内联样式承载；
   若后续要落成 class，直接把这些对象搬进 components.css 的「表格增强」一节即可。
   --------------------------------------------------------------------------- */

const toolbarStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "var(--space-3)",
  flexWrap: "wrap",
  marginBottom: "var(--space-3)",
};

const toolbarNoteStyle: React.CSSProperties = { fontSize: 13 };
const toggleWrapStyle: React.CSSProperties = { position: "relative", marginLeft: "auto" };
const toggleBtnStyle: React.CSSProperties = { height: "var(--control-height-sm)", padding: "0 var(--space-3)" };

const panelStyle: React.CSSProperties = {
  position: "absolute",
  right: 0,
  top: "calc(100% + 6px)",
  zIndex: 40,
  minWidth: 220,
  maxHeight: 320,
  overflowY: "auto",
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  boxShadow: "var(--shadow-lg)",
  padding: "var(--space-3)",
  display: "flex",
  flexDirection: "column",
  gap: 6,
  textAlign: "left",
};

const panelHeadStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "var(--space-3)",
  fontSize: 12,
  color: "var(--muted)",
  marginBottom: 2,
};

const panelRowStyle: React.CSSProperties = { cursor: "pointer", whiteSpace: "nowrap" };

const tagStyle: React.CSSProperties = {
  fontSize: 11,
  color: "var(--muted)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-full)",
  padding: "0 6px",
};

const expandThStyle: React.CSSProperties = { width: 40, paddingRight: 0 };

const expandBtnStyle: React.CSSProperties = {
  width: 22,
  height: 22,
  padding: 0,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  background: "var(--surface)",
  color: "var(--muted)",
  cursor: "pointer",
};

const expandPanelStyle: React.CSSProperties = {
  background: "var(--surface-2)",
  paddingTop: "var(--space-4)",
  paddingBottom: "var(--space-4)",
};

const kvGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: "var(--space-3) var(--space-5)",
  margin: 0,
};

const kvKeyStyle: React.CSSProperties = { fontSize: 12, color: "var(--muted)", marginBottom: 3 };
const kvValueStyle: React.CSSProperties = {
  margin: 0,
  fontSize: "var(--fs-control)",
  color: "var(--text)",
  wordBreak: "break-word",
};

/** 表格骨架：行高与真实表格一致，加载完成后不会跳动。 */
function TableSkeleton({ rows, cols }: { rows: number; cols: number }) {
  const widths = ["55%", "70%", "45%", "62%"];
  return (
    <div aria-busy="true" style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            {Array.from({ length: cols }).map((_, i) => (
              <th key={i}>
                <span className="skeleton skeleton-text" style={{ width: "60%", display: "block", margin: 0 }} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, r) => (
            <tr key={r}>
              {Array.from({ length: cols }).map((_, c) => (
                <td key={c}>
                  <span
                    className="skeleton skeleton-text"
                    style={{ width: widths[(r + c) % widths.length], display: "block", margin: 0 }}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
