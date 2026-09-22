/* =============================================================
   Icon —— 自研零依赖 SVG 图标集
   -------------------------------------------------------------
   为什么自建：
   - 不引入 lucide / react-icons 等图标库，避免为几十个图标背一个包。
   - 全站统一的 24×24 线性栅格 + 1.7 描边，视觉节奏与 V2 原型一致。
   - 原先这套图标内联在 layouts/MainLayout.tsx 的 ICONS 常量里，
     其它页面用不到；提取后成为全站单一图标入口。

   用法：<Icon name="database" size={18} />
   颜色：始终继承 currentColor，不要在外层传 color。
   ============================================================= */

import type { ReactElement, SVGProps } from "react";

export type IconName =
  // 导航（沿用 MainLayout 原有形状）
  | "grid"
  | "database"
  | "table"
  | "sliders"
  | "chart"
  | "chip"
  | "sparkles"
  | "flow"
  | "beaker"
  | "file"
  | "book"
  | "gear"
  | "puzzle"
  // 指标与状态
  | "trend-up"
  | "trend-down"
  | "check"
  | "alert"
  | "clock"
  | "layers"
  // 操作
  | "arrow-right"
  | "upload"
  | "download"
  | "search"
  | "refresh"
  | "filter"
  | "trash"
  | "plus"
  | "more"
  // 空状态 / 引导（2026-09-22 新增，供 EmptyState 插画使用）
  | "clipboard"
  | "cursor"
  | "wand"
  // 第三方 / 外链（唯一一处需要实心填充的图标：描边版 GitHub 猫形辨识度极差）
  | "github";

const PATHS: Record<IconName, ReactElement> = {
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="2.6" />
      <path d="M4 5v14c0 1.45 3.58 2.6 8 2.6s8-1.15 8-2.6V5" />
      <path d="M4 12c0 1.45 3.58 2.6 8 2.6s8-1.15 8-2.6" />
    </>
  ),
  table: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 10h18" />
      <path d="M9.5 10v10" />
    </>
  ),
  sliders: (
    <>
      <path d="M4 7h16" />
      <circle cx="9" cy="7" r="2" />
      <path d="M4 12h16" />
      <circle cx="15" cy="12" r="2" />
      <path d="M4 17h16" />
      <circle cx="7" cy="17" r="2" />
    </>
  ),
  chart: (
    <>
      <path d="M4 20V9" />
      <path d="M10 20V4" />
      <path d="M16 20v-7" />
      <path d="M3 20h18" />
    </>
  ),
  chip: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
    </>
  ),
  sparkles: (
    <>
      <path d="M12 3l1.9 4.9L19 9.8l-5.1 1.9L12 16.6l-1.9-4.9L5 9.8l5.1-1.9L12 3z" />
      <path d="M18.5 15.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2z" />
    </>
  ),
  flow: (
    <>
      <circle cx="5.5" cy="5.5" r="2.2" />
      <circle cx="18.5" cy="5.5" r="2.2" />
      <circle cx="12" cy="18.5" r="2.2" />
      <path d="M5.5 7.7v1.8a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7.7" />
      <path d="M12 11.5v4.8" />
    </>
  ),
  beaker: (
    <>
      <path d="M9 3h6" />
      <path d="M10 3v5.2L4.9 17.6A2 2 0 0 0 6.7 20.5h10.6a2 2 0 0 0 1.8-2.9L14 8.2V3" />
      <path d="M7.5 14.5h9" />
    </>
  ),
  file: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h6" />
    </>
  ),
  book: (
    <>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </>
  ),
  gear: (
    <>
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1" />
    </>
  ),
  puzzle: (
    <>
      <path d="M10.5 3.2a2 2 0 0 1 3 0 2 2 0 0 1 1.2 3.1l.3.3h2.8a1 1 0 0 1 1 1v2.8a2 2 0 0 0 0 3.6V17a1 1 0 0 1-1 1h-2.8a2 2 0 0 1-3.6 0 2 2 0 0 1-3.2-1.4V14a2 2 0 0 0-3-.2 2 2 0 0 1-.2-3.6V7.6a1 1 0 0 1 1-1h2.8l.3-.3a2 2 0 0 1 3.1-1.1z" />
    </>
  ),
  "trend-up": (
    <>
      <polyline points="3 17 9 11 13 15 21 7" />
      <polyline points="15 7 21 7 21 13" />
    </>
  ),
  "trend-down": (
    <>
      <polyline points="3 7 9 13 13 9 21 17" />
      <polyline points="15 17 21 17 21 11" />
    </>
  ),
  check: <polyline points="4 12.5 9 17.5 20 6.5" />,
  alert: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v5" />
      <path d="M12 16.5h.01" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <polyline points="12 7 12 12 16 14" />
    </>
  ),
  layers: (
    <>
      <polygon points="12 2.5 2.5 8 12 13.5 21.5 8 12 2.5" />
      <polyline points="2.5 16 12 21.5 21.5 16" />
      <polyline points="2.5 12 12 17.5 21.5 12" />
    </>
  ),
  "arrow-right": (
    <>
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4" />
      <polyline points="7 9 12 4 17 9" />
      <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </>
  ),
  download: (
    <>
      <path d="M12 4v12" />
      <polyline points="7 11 12 16 17 11" />
      <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-4.2-4.2" />
    </>
  ),
  refresh: (
    <>
      <polyline points="21 4 21 10 15 10" />
      <polyline points="3 20 3 14 9 14" />
      <path d="M4.2 10.2A8 8 0 0 1 18 6.2L21 10" />
      <path d="M19.8 13.8A8 8 0 0 1 6 17.8L3 14" />
    </>
  ),
  filter: <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />,
  trash: (
    <>
      <polyline points="3 6 21 6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14" />
    </>
  ),
  plus: (
    <>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </>
  ),
  more: (
    <>
      <circle cx="5" cy="12" r="1.4" />
      <circle cx="12" cy="12" r="1.4" />
      <circle cx="19" cy="12" r="1.4" />
    </>
  ),
  clipboard: (
    <>
      <rect x="5" y="4" width="14" height="17" rx="2" />
      <path d="M9 4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2" />
      <path d="M9 10h6M9 14h6M9 18h4" />
    </>
  ),
  cursor: (
    <>
      <path d="M5 3l7 17 2.5-6.5L21 11z" />
    </>
  ),
  wand: (
    <>
      <path d="M4 20L14 10" />
      <path d="M12 5l.9 2.1L15 8l-2.1.9L12 11l-.9-2.1L9 8l2.1-.9z" />
      <path d="M18 13l.7 1.6 1.6.7-1.6.7L18 17.6l-.7-1.6-1.6-.7 1.6-.7z" />
    </>
  ),
  github: (
    <path
      d="M12 2C6.48 2 2 6.58 2 12.28c0 4.53 2.87 8.37 6.84 9.73.5.09.68-.22.68-.49v-1.7c-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.5-1.11-1.5-.91-.63.07-.62.07-.62 1 .07 1.53 1.06 1.53 1.06.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.06 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.27 2.75 1.05a9.4 9.4 0 0 1 5.01 0c1.91-1.32 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.93-2.35 4.8-4.58 5.05.36.32.68.94.68 1.9v2.82c0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.28C22 6.58 17.52 2 12 2z"
      fill="currentColor"
      stroke="none"
    />
  ),
};

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, "children" | "name"> {
  name: IconName;
  size?: number;
  strokeWidth?: number;
}

export function Icon({ name, size = 18, strokeWidth = 1.7, ...rest }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {PATHS[name] ?? PATHS.grid}
    </svg>
  );
}
