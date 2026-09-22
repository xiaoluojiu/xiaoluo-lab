// Prompt 153：加载状态。
export function Loading({ text = "加载中..." }: { text?: string }) {
  return <div className="muted">{text}</div>;
}

// Prompt 154：错误状态（可重试）。
export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="card" style={{ textAlign: "center", color: "var(--danger)" }}>
      <p>出错了：{message}</p>
      {onRetry && (
        <button className="btn" onClick={onRetry}>
          重试
        </button>
      )}
    </div>
  );
}
