import type { ReactNode } from "react";

/**
 * 样例数据表 —— 「前 N 行 × 若干列」的只读预览表（.processing-table-wrap）。
 *
 * 背景：这段 markup 连同 `formatCell` 此前在三个地方各写了一遍，且逐字相同：
 *   1. features/dataset/OperationPreviewDialog   —— 操作后的样例
 *   2. features/merge/MultiMergePanel            —— 多文件合并预览
 *   3. pages/Processing（本页的本地 PreviewTable）-- 操作预览面板
 * 现收敛为唯一实现。三处唯一的历史差异是「结果为空」时的占位样式
 * （`muted` 一行灰字 vs `processing-empty` 空态块），故用 `emptyClassName` 显式保留，
 * 不改变任何一处现有的视觉表现。
 *
 * 注意：这里只做「渲染」，数据由调用方准备
 * （`features/dataset/PreviewTable` 是另一回事 —— 它自己按 datasetId 分页取数，
 * 两者职责不同，不要混用）。
 */

/** 单元格显示口径：空值 → "—"，对象 → JSON 字符串，其余 String。 */
export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

interface Props {
  rows: Array<Record<string, unknown>>;
  columns: string[];
  /** 没有可展示行时的占位文案。 */
  emptyText?: ReactNode;
  /** 占位容器的 class：`processing-empty`（空态块）或 `muted`（一行灰字）。 */
  emptyClassName?: string;
}

export function SampleTable({
  rows,
  columns,
  emptyText = "暂无预览数据",
  emptyClassName = "processing-empty",
}: Props) {
  if (!rows.length) return <div className={emptyClassName}>{emptyText}</div>;
  return (
    <div className="processing-table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
