/**
 * 主导航 / 命令面板的单一事实源。
 *
 * 历史问题：MainLayout 的 MENU 与 CommandPalette 的 NAV 是两份手写列表，
 * 改路由时经常只改一处，导致「侧栏能进、命令面板跳转错」或标签文案不一致。
 * 现在两处都从这里取，新增页面只改这一个文件。
 *
 * 2026-09-21 改版：取消可折叠 NavGroup，改为「静态分组」。
 *   原先只有「数据中心」一个组能展开/收起，其余 8 个路由平铺，
 *   层级不统一且多一次点击；现在统一为五个 NavSection，
 *   分组标题只做视觉分隔（不可点击、不折叠），11 个路由一条不少。
 *   FLAT_NAV 与 isActivePath 行为不变，命令面板无需修改。
 */

import type { IconName } from "../components/icons/Icon";

export interface NavItem {
  path: string;
  label: string;
  icon: IconName;
  /** 命令面板里的补充说明（可选）。 */
  hint?: string;
}

/** 静态分组：标题只做分隔，不带交互。 */
export interface NavSection {
  id: string;
  label: string;
  items: NavItem[];
}

export const NAV: NavSection[] = [
  {
    id: "overview",
    label: "概览",
    items: [{ path: "/", label: "工作台", icon: "grid", hint: "数据资产与最近工作总览" }],
  },
  {
    id: "data",
    label: "数据中心",
    items: [
      { path: "/datasets", label: "数据集列表", icon: "table", hint: "浏览、预览与版本管理" },
      { path: "/processing", label: "数据处理", icon: "sliders", hint: "清洗 / 转换 / 合并算子" },
      { path: "/analysis", label: "数据分析", icon: "chart", hint: "描述统计、相关性与可视化" },
    ],
  },
  {
    id: "modeling",
    label: "建模与编排",
    items: [
      { path: "/ml", label: "机器学习", icon: "chip", hint: "训练与评估模型" },
      { path: "/workflow", label: "Workflow", icon: "flow", hint: "编排数据处理流程" },
      { path: "/experiments", label: "实验中心", icon: "beaker", hint: "记录与对比实验" },
    ],
  },
  {
    id: "delivery",
    label: "交付",
    items: [
      { path: "/ai", label: "AI 实验室", icon: "sparkles", hint: "会话式分析与工具授权" },
      { path: "/reports", label: "报告中心", icon: "file", hint: "生成报告并导出 HTML / MD / PDF" },
    ],
  },
  {
    id: "extensions",
    label: "扩展",
    items: [
      { path: "/learning", label: "学习中心", icon: "book", hint: "机器学习与手写大模型机制实验" },
      { path: "/extensions", label: "扩展中心", icon: "puzzle", hint: "扩展能力目录" },
      { path: "/settings", label: "设置", icon: "gear", hint: "外观、LLM、数据与工具授权" },
    ],
  },
];

/** 收平后的条目，供命令面板直接遍历跳转。 */
export const FLAT_NAV: NavItem[] = NAV.flatMap((section) => section.items);

/** /reports/:key 这类详情页要能让父级 /reports 保持高亮。 */
export function isActivePath(path: string, current: string): boolean {
  return path === "/" ? current === "/" : current === path || current.startsWith(`${path}/`);
}
