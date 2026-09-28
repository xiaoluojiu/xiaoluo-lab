// Prompt 153：加载状态。
export function Loading({ text = "加载中..." }: { text?: string }) {
  return <div className="muted">{text}</div>;
}

/**
 * 不确定进度的转圈指示器。
 *
 * 存在的理由：像「实验重跑」「工作流运行」这类**同步接口**，后端一次性算完才返回，
 * 前端拿不到任何中间进度。此时最诚实的表达是「在跑、但不知道到哪了」——
 * 而不是拿定时器凑一个从 0 爬到 90 的假百分比（用户会以为 90% 之后马上好）。
 */
export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label ? <span className="muted">{label}</span> : null}
    </span>
  );
}

/**
 * 不确定进度的流动条：高亮段来回移动，**不表示任何百分比**。
 *
 * 刻意不给 `aria-valuenow`：没有数值可报时省略该属性，屏幕阅读器才会读成
 * 「忙碌」而不是「0%」。
 */
export function IndeterminateBar() {
  return (
    <div className="task-progress-track is-running" role="progressbar" aria-valuetext="进行中">
      <div className="task-progress-fill" />
    </div>
  );
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
