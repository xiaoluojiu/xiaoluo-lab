import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useUiShell } from "../store/ui";
import { brand } from "../config/brand";
// 主导航与命令面板共用 config/nav.ts 的单一事实源，避免两处列表不同步。
// 2026-09-21：NavGroup 折叠交互退役，改为「概览 / 数据中心 / 建模与编排 / 交付 / 扩展」
// 五个静态分组，分组标题只做视觉分隔；路由一条不少，命令面板仍取 FLAT_NAV。
import { NAV as MENU, isActivePath } from "../config/nav";
import { Topbar } from "../components/Topbar";
import { CommandPalette } from "../components/CommandPalette";
import { DynamicBackground } from "../components/DynamicBackground";
import { Icon } from "../components/icons/Icon";

/* 761~1100px 自动收起为图标栏；≤760px 走全局的横向顶部导航方案（design-foundation.css）。 */
const NAV_AUTO_COLLAPSE_QUERY = "(max-width: 1100px) and (min-width: 761px)";

/** 底部仓库卡：配置了 brand.repoUrl 才变成真链接，否则渲染成「未配置」占位。 */
function RepoCard({ collapsed }: { collapsed: boolean }) {
  const label = brand.repoLabel ?? "查看源码与文档";
  const body = (
    <>
      <span className="sidebar-repo-mark" aria-hidden="true">
        <Icon name="github" size={18} />
      </span>
      <span className="sidebar-repo-text">
        <span className="sidebar-repo-title">GitHub 仓库</span>
        <span className="sidebar-repo-sub">{brand.repoUrl ? label : "未配置 · 后期接入"}</span>
      </span>
      <Icon className="sidebar-repo-arrow" name="arrow-right" size={14} />
    </>
  );

  if (!brand.repoUrl) {
    return (
      <div
        className="sidebar-repo is-disabled"
        aria-disabled="true"
        title={collapsed ? "GitHub 仓库（未配置）" : "仓库地址尚未配置：config/brand.ts 填入 repoUrl 后自动生效"}
      >
        {body}
      </div>
    );
  }

  return (
    <a
      className="sidebar-repo"
      href={brand.repoUrl}
      target="_blank"
      rel="noreferrer"
      title={collapsed ? "GitHub 仓库" : `${brand.name} · ${label}`}
    >
      {body}
    </a>
  );
}

export function MainLayout() {
  const location = useLocation();
  const navCollapsed = useUiShell((s) => s.navCollapsed);
  const toggleNav = useUiShell((s) => s.toggleNav);

  // 窄屏时强制收起（autoCollapsed，不落盘）；与用户手动折叠取或。
  const [autoCollapsed, setAutoCollapsed] = useState<boolean>(() => window.matchMedia(NAV_AUTO_COLLAPSE_QUERY).matches);

  useEffect(() => {
    const media = window.matchMedia(NAV_AUTO_COLLAPSE_QUERY);
    const onChange = (event: MediaQueryListEvent) => setAutoCollapsed(event.matches);
    media.addEventListener?.("change", onChange);
    return () => media.removeEventListener?.("change", onChange);
  }, []);

  const isCollapsed = navCollapsed || autoCollapsed;

  return (
    <div className={`layout${isCollapsed ? " nav-collapsed" : ""}`}>
      <DynamicBackground />
      <aside className="sidebar" aria-label="主导航">
        <div className="logo-row">
          <Link to="/" className="logo" title={brand.name} aria-label={brand.name}>
            {brand.logoSrc ? (
              <img className="logo-img" src={brand.logoSrc} alt={brand.name} />
            ) : (
              <span className="logo-mark" aria-hidden="true" />
            )}
            <span className="logo-text">
              <span className="logo-full">{brand.name}</span>
              <span className="logo-tag">数据分析平台</span>
            </span>
            <span className="logo-mini" aria-hidden="true">
              {brand.shortName}
            </span>
          </Link>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={toggleNav}
            aria-label={isCollapsed ? "展开导航" : "收起导航"}
            aria-expanded={!isCollapsed}
            title={isCollapsed ? "展开导航" : "收起导航"}
          >
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
              <path d="M15 6l-6 6 6 6" />
            </svg>
          </button>
        </div>
        <nav>
          {MENU.map((section) => (
            <div className="nav-section" key={section.id}>
              {/* 分组标题只做分隔，不可点击；收起态由 CSS 降级为一条细分隔线 */}
              <p className="nav-section-title">
                <span className="nav-section-label">{section.label}</span>
              </p>
              {section.items.map((item) => {
                const active = isActivePath(item.path, location.pathname);
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`nav-item${active ? " active" : ""}`}
                    aria-current={active ? "page" : undefined}
                    // 收起态看不到组名，用原生 title 补「组 · 项」（避免 CSS tooltip 被 nav 的 overflow 裁掉）
                    title={isCollapsed ? `${section.label} · ${item.label}` : item.label}
                  >
                    <Icon className="nav-icon" name={item.icon} size={20} strokeWidth={1.7} />
                    <span className="nav-label">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <RepoCard collapsed={isCollapsed} />
        </div>
      </aside>
      <main className="main">
        <Topbar />
        <div className="content">
          <Outlet />
        </div>
      </main>
      <CommandPalette />
    </div>
  );
}
