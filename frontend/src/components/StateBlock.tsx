import type { ReactNode } from "react";

/** 空态：发生了什么 + 为什么 + 下一步怎么做。历史上 13 个页面里只有 5 个有空态。 */
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty-title">{title}</span>
      {description}
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  );
}

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
