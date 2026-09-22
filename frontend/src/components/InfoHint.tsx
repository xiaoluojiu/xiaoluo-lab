import type { ReactNode } from "react";

/**
 * 圈问号提示（InfoHint）——把「参考文案」折叠进一个可点/可悬停的圆圈里。
 *
 * 设计取舍（用户 2026-09-21 明确要求）：
 * 像「范围：正整数，常用 200 ~ 5000 / 默认：1000 / 影响：优化器最多迭代多少轮…」
 * 这类说明对**第一次使用的人有用**，但常驻在页面上会持续占位、造成视觉噪音。
 * 因此改为默认只显示一个 ⓘ 圆圈，用户把指针移上去（或键盘聚焦）才展开。
 *
 * 无障碍与触屏：
 * - 用原生 `title` 属性兜底，鼠标悬停由浏览器渲染，键鼠之外的场景也不会丢失信息；
 * - 同时是可聚焦的 `<button>`，键盘 Tab 能到达、`aria-label` 给出摘要；
 * - 触屏设备无 hover —— 因为 button 可聚焦，点一下即展开（:focus-within 生效）。
 *
 * 注意：不要用 `position: fixed`，浮层用绝对定位挂在触发器上，避免被卡片裁切。
 */
export function InfoHint({
  label,
  children,
  block = false,
}: {
  /** 无障碍名称与原生 tooltip 的标题，例如「max_iter 参数说明」。 */
  label: string;
  /** 展开后显示的内容。 */
  children: ReactNode;
  /** true = 触发器独占一行（用于字段下方）；false = 紧跟标题（用于标签旁）。 */
  block?: boolean;
}) {
  return (
    <span className={`info-hint${block ? " info-hint-block" : ""}`}>
      <button
        type="button"
        className="info-hint-trigger"
        aria-label={label}
        title={label}
      >
        i
      </button>
      <span className="info-hint-pop" role="tooltip">{children}</span>
    </span>
  );
}
