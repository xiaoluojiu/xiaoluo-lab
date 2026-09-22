import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useUiShell } from "../store/ui";
import { readUi, writeUi } from "../lib/uiSettings";
import { NotificationBell } from "./NotificationBell";

/** 路由 → 面包屑末级标题。 */
const ROUTE_LABELS: Record<string, string> = {
  "/": "工作台",
  "/datasets": "数据集",
  "/processing": "数据处理",
  "/analysis": "数据分析",
  "/ml": "机器学习",
  "/ai": "AI 实验室",
  "/workflow": "工作流",
  "/experiments": "实验中心",
  "/reports": "报告中心",
  "/extensions": "扩展中心",
  "/learning": "学习中心",
  "/settings": "设置",
};

function buildCrumbs(pathname: string): { label: string; to?: string }[] {
  if (pathname === "/") return [{ label: "工作台" }];
  if (pathname.startsWith("/datasets/")) {
    return [
      { label: "数据集", to: "/datasets" },
      { label: "详情" },
    ];
  }
  // 报告详情是独立路由，面包屑要能一步回到报告中心。
  if (pathname.startsWith("/reports/")) {
    return [
      { label: "报告中心", to: "/reports" },
      { label: "报告详情" },
    ];
  }
  // 学习中心的两个子路由此前都退化成「学习中心」一条，
  // 面包屑无法区分实验台与学习记录，也无法从子页一步回列表。
  if (pathname.startsWith("/learning/")) {
    const sub =
      pathname === "/learning/workspace"
        ? "实验台"
        : pathname === "/learning/history"
          ? "学习记录"
          : pathname.startsWith("/learning/card/")
            ? "我的卡片"
            : "详情";
    return [
      { label: "学习中心", to: "/learning" },
      { label: sub },
    ];
  }
  const label = ROUTE_LABELS[pathname] ?? ROUTE_LABELS[`/${pathname.split("/")[1]}`] ?? "页面";
  return [{ label }];
}

/**
 * 吸顶顶栏：折叠按钮 + 面包屑 + 全局搜索（唤起命令面板）+ 主题切换 + 通知占位。
 * 不改变整体布局（侧栏 + 内容列），仅在内容列顶部叠加一个 sticky 头部。
 */
export function Topbar() {
  const location = useLocation();
  const toggleNav = useUiShell((s) => s.toggleNav);
  const openCommand = useUiShell((s) => s.openCommand);
  const crumbs = buildCrumbs(location.pathname);
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    writeUi({ ...readUi(), theme: next });
  }

  const isDark = theme === "dark";

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          type="button"
          className="icon-btn"
          onClick={toggleNav}
          aria-label="折叠或展开导航"
          title="折叠或展开导航"
        >
          <svg
            viewBox="0 0 24 24"
            width="18"
            height="18"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <nav className="crumbs topbar-crumbs" aria-label="面包屑">
          {crumbs.map((c, i) => (
            <span key={i} className="crumb">
              {c.to ? <Link to={c.to}>{c.label}</Link> : <b>{c.label}</b>}
              {i < crumbs.length - 1 && <span className="crumb-sep">/</span>}
            </span>
          ))}
        </nav>
      </div>

      <div className="topbar-center">
        <button type="button" className="topbar-search" onClick={openCommand}>
          <svg
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" />
          </svg>
          <span>搜索或跳转…</span>
          <kbd className="topbar-kbd">Ctrl K</kbd>
        </button>
      </div>

      <div className="topbar-right">
        <button
          type="button"
          className="icon-btn"
          onClick={toggleTheme}
          aria-label={isDark ? "切换到浅色" : "切换到深色"}
          title={isDark ? "切换到浅色" : "切换到深色"}
        >
          {isDark ? (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4.5" />
              <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8" />
            </svg>
          ) : (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.6 6.6 0 0 0 9.8 9.8z" />
            </svg>
          )}
        </button>
        <NotificationBell />
      </div>
    </header>
  );
}
