import type { ReactNode } from "react";

/**
 * 空态：发生了什么 + 为什么 + 下一步怎么做。
 *
 * 2026-09-27：这里原先自己实现了一份（只有标题/说明/主操作，没有图标），
 * 与 components/viz/Blocks.tsx 的同名组件各写各的，改了这头漏那头。
 * 现在只保留一个实现（viz/Blocks 的插画化版本：图标 + 标题 + 说明 + 主操作 + 次操作），
 * 这里改成 re-export，历史入口 `components/StateBlock` 继续可用。
 */
export { EmptyState } from "./viz/Blocks";
export type { EmptyStateProps } from "./viz/Blocks";

/** 错误态：必须带重试动作，而不是只显示一行红字。 */
export function ErrorState({
  title = "操作未完成",
  message,
  onRetry,
  retryText = "重试",
}: {
  title?: string;
  message: ReactNode;
  onRetry?: () => void;
  retryText?: string;
}) {
  return (
    <div className="error-box">
      <div className="error-title">{title}</div>
      <div className="error-detail">{message}</div>
      {onRetry ? (
        <div className="error-action">
          <button className="btn btn-sm" type="button" onClick={onRetry}>
            {retryText}
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** 加载骨架：替代散落各页的「正在加载…」文字。 */
export function Skeleton({ lines = 3, card = false }: { lines?: number; card?: boolean }) {
  const widths = ["70%", "92%", "55%", "80%", "48%"];
  return (
    <div aria-busy="true">
      {card ? <div className="skeleton skeleton-card" style={{ marginBottom: "var(--space-3)" }} /> : null}
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeleton skeleton-text" style={{ width: widths[i % widths.length] }} />
      ))}
    </div>
  );
}
