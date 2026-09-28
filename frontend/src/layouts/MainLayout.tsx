import { Outlet } from "react-router-dom";
import { useUiShell } from "../store/ui";
import { AppSidebar } from "./AppSidebar";
import { Topbar } from "../components/Topbar";
import { Breadcrumb } from "../components/Breadcrumb";
import { CommandPalette } from "../components/CommandPalette";
import { DynamicBackground } from "../components/DynamicBackground";

/**
 * 标准布局：侧边栏 + 内容区。**这是默认外壳**，没有特殊需求的路由都挂在这里。
 *
 * 2026-09-27 统一侧栏形态：折叠状态**只**来自全局 store（`store/ui.ts` 的
 * `isSidebarCollapsed`，持久化），本组件不再叠加任何视口驱动的自动收起——
 * 桌面端完全尊重用户偏好，跨页跳转、刷新均保持一致。
 * 窄屏（<768px）的横向导航兜底由 CSS（design-foundation.css）承担，与此外壳无关。
 *
 * 需要「对话优先」视觉适配的页面挂 ChatLayout（同样读这份全局状态）。
 * 见 config/routes.ts。
 */
export function MainLayout() {
  const isSidebarCollapsed = useUiShell((s) => s.isSidebarCollapsed);
  const toggleSidebar = useUiShell((s) => s.toggleSidebar);

  return (
    <div className={`layout${isSidebarCollapsed ? " nav-collapsed" : ""}`}>
      <DynamicBackground />
      <AppSidebar collapsed={isSidebarCollapsed} onToggle={toggleSidebar} />
      <main className="main">
        <Topbar />
        {/* 顶栏之下的层级路径条：多层级页面（数据集 → 版本 → 分析 …）随时可回溯。
            只做展示，不改路由；AI 实验室由组件内部自行跳过。 */}
        <Breadcrumb />
        <div className="content">
          <Outlet />
        </div>
      </main>
      <CommandPalette />
    </div>
  );
}
