/**
 * chartColors.ts —— 图表配色唯一来源
 * -------------------------------------------------------------
 * recharts 的 fill / stroke 是 SVG presentation attribute，
 * 无法解析 CSS 变量（var(--x) 在属性值中不生效），
 * 因此图表的颜色必须集中在此处以「真实 hex」输出。
 * 组件内不再出现任何 #hex 字面量；如需调整配色只改这里。
 */

/** 多系列默认调色板（顺序即语义：信息/成功/警告/危险/紫/青） */
export const CHART_SERIES = [
  "#2563eb",
  "#16a34a",
  "#d97706",
  "#dc2626",
  "#7c3aed",
  "#0891b2",
];

/** 坐标轴/网格线颜色 */
export const CHART_GRID = "#eef2f7";

/** 主色（indigo-600），用于箱线图须/箱体等 */
export const CHART_PRIMARY = "#4f46e5";

/** 主色 RGB 三元组，供需要透明度的渐变/热力计算使用 */
export const CHART_PRIMARY_RGB: [number, number, number] = [79, 70, 229];

/** 浅 indigo（indigo-400），用于箱线图 quartile 区间 */
export const CHART_INDIGO_LIGHT = "#a5b4fc";

/** 空值/缺失数据的兜底色（用于热力图等） */
export const CHART_NULL = "rgba(148, 163, 184, 0.12)";

/** 备用色（当前未直接使用，预留以便后续统一引用） */
export const CHART_AMBER = "#f59e0b";
export const CHART_VIOLET = "#7c3aed";
export const CHART_CYAN = "#0891b2";

/** 给 hex 颜色叠加透明度，返回 rgba() 字符串（SVG / 内联 style 可用） */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
