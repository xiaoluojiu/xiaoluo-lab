/** 路由懒加载时的统一占位：骨架屏替代散落各页的「正在加载…」文字。 */
export function PageLoading({ label }: { label?: string }) {
  return (
    <div aria-busy="true" aria-live="polite" style={{ padding: "4px 0" }}>
      <div className="skeleton skeleton-text" style={{ width: "28%", height: 26 }} />
      <div className="skeleton skeleton-text" style={{ width: "62%" }} />
      <div className="grid g3">
        <div className="skeleton skeleton-card" />
        <div className="skeleton skeleton-card" />
        <div className="skeleton skeleton-card" />
      </div>
      <span className="sr-only">{label ?? "正在加载页面"}</span>
    </div>
  );
}
