/**
 * 路由表：**「哪个页面用哪种布局」的单一事实源**。
 *
 * 2026-09-27 统一侧栏形态：全站只保留两种外壳——标准布局与对话布局，
 * 且侧栏折叠状态统一来自全局 store，跨页跳转不再出现形态打架。
 * 曾有的第三种外壳 FullWidthLayout（无侧栏、顶栏按钮变命令面板浮窗）已退役，
 * 原挂载页（/datasets、/analysis）回归标准布局。
 *
 * 与 `config/nav.ts` 的分工：
 * - `nav.ts` 管**侧栏/命令面板展示什么**（label / icon / hint）；
 * - 本文件管**页面挂在哪套外壳下**。
 * 两者都新增页面时要一起补，但职责不重叠。
 *
 * 不在本表内的路由：
 * - `/workflow/editor/:id` —— 沉浸式编辑器，刻意**不套任何外壳**（无顶栏无侧栏），
 *   单独在 router/index.tsx 里声明；
 * - `*` 兜底 404 —— 固定挂标准布局，也在 router/index.tsx 里声明。
 */

import { lazy, type ComponentType } from "react";

/** 可用的外壳。新增布局时在这里加一项，并在 router/index.tsx 的 LAYOUTS 里登记。 */
export type LayoutKind = "standard" | "chat";

export interface AppRoute {
  path: string;
  /** 页面组件。一律 `lazy` —— 首次访问不再一次性下载整个应用（见 router 注释）。 */
  component: ComponentType;
  layout: LayoutKind;
}

export const APP_ROUTES: AppRoute[] = [
  /* ---------- 标准布局（侧栏 + 内容区）---------- */
  { path: "/", component: lazy(() => import("../pages/Home")), layout: "standard" },
  // 数据集详情仍走标准布局：它是有层级、需要随时回列表的阅读型页面。
  { path: "/datasets/:id", component: lazy(() => import("../pages/Datasets/Detail")), layout: "standard" },
  { path: "/processing", component: lazy(() => import("../pages/Processing")), layout: "standard" },
  { path: "/ml", component: lazy(() => import("../pages/ML")), layout: "standard" },
  { path: "/workflow", component: lazy(() => import("../pages/Workflow")), layout: "standard" },
  { path: "/experiments", component: lazy(() => import("../pages/Experiments")), layout: "standard" },
  { path: "/reports", component: lazy(() => import("../pages/Reports")), layout: "standard" },
  // 报告详情独立成页：正文不再内联展开在列表下方，链接可直接分享。
  { path: "/reports/:key", component: lazy(() => import("../pages/Reports/Detail")), layout: "standard" },
  { path: "/settings", component: lazy(() => import("../pages/Settings")), layout: "standard" },
  { path: "/extensions", component: lazy(() => import("../pages/Extensions")), layout: "standard" },
  { path: "/learning", component: lazy(() => import("../pages/Learning")), layout: "standard" },
  { path: "/learning/workspace", component: lazy(() => import("../pages/LearningWorkspace")), layout: "standard" },
  { path: "/learning/history", component: lazy(() => import("../pages/LearningHistory")), layout: "standard" },
  { path: "/learning/card/:cardId", component: lazy(() => import("../pages/LearningCard")), layout: "standard" },

  /* ---------- 全宽页面回归标准布局（2026-09-27）----------
     /datasets 与 /analysis 曾挂 FullWidthLayout（无侧栏），导致从侧栏进入时
     导航彻底消失、只能靠命令面板浮窗跳转。现在统一挂标准布局：
     宽表场景需要横向空间时，用户可手动半收起侧栏（全局偏好会被记住）。 */
  { path: "/datasets", component: lazy(() => import("../pages/Datasets")), layout: "standard" },
  { path: "/analysis", component: lazy(() => import("../pages/Analysis")), layout: "standard" },

  /* ---------- 对话布局（Agent 会话页）----------
     AI 实验室是会话式工作台，`layout-chat` 只做内容区留白适配；
     侧栏折叠状态与标准布局同源（全局持久化偏好），不再强制默认收起。
     仅换外层布局：pages/AI 与其 ai-lab.css 一行未改（见 ChatLayout 注释）。
     想退回标准布局只需把这一行的 layout 改成 "standard"。 */
  { path: "/ai", component: lazy(() => import("../pages/AI")), layout: "chat" },
];
