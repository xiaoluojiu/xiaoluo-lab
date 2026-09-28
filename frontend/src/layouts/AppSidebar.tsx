import { Link, useLocation } from "react-router-dom";
import { brand } from "../config/brand";
// 主导航与命令面板共用 config/nav.ts 的单一事实源，避免两处列表不同步。
// 2026-09-21：NavGroup 折叠交互退役，改为「概览 / 数据中心 / 建模与编排 / 交付 / 扩展」
// 五个静态分组，分组标题只做视觉分隔；路由一条不少，命令面板仍取 FLAT_NAV。
import { NAV as MENU, isActivePath } from "../config/nav";
import { Icon } from "../components/icons/Icon";

/**
 * 全局侧栏（外壳的一部分，被 MainLayout / ChatLayout 共用）。
 *
 * 2026-09-27 从 MainLayout 原样抽出，只为让「对话布局默认收起侧栏」能复用同一份实现，
 * **DOM 结构与 class 名逐字保留**：侧栏的全部样式都挂在
 * `.layout .sidebar` / `.layout.nav-collapsed .sidebar` 这些后代选择器上（见 visual.css），
 * 换一层组件不会改变命中，但改一个 class 就会掉样式。
 *
 * 侧栏本身不再持有状态：折叠与否由外层布局从全局 store（`store/ui.ts` 的
 * `isSidebarCollapsed`）读取并传入，这里只负责渲染与回调。
 */
interface AppSidebarProps {
  /** 是否处于收起（图标栏）形态。 */
  collapsed: boolean;
  /** 点击折叠按钮时触发，由外层布局切换自己的折叠状态。 */
  onToggle: () => void;
}

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

export function AppSidebar({ collapsed, onToggle }: AppSidebarProps) {
  const location = useLocation();

  return (
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
          onClick={onToggle}
          aria-label={collapsed ? "展开导航" : "收起导航"}
          aria-expanded={!collapsed}
          title={collapsed ? "展开导航" : "收起导航"}
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
                  title={collapsed ? `${section.label} · ${item.label}` : item.label}
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
        <RepoCard collapsed={collapsed} />
      </div>
    </aside>
  );
}
