import { Outlet } from "react-router-dom";
import { useUiShell } from "../store/ui";
import { AppSidebar } from "./AppSidebar";
import { Topbar } from "../components/Topbar";
import { Breadcrumb } from "../components/Breadcrumb";
import { CommandPalette } from "../components/CommandPalette";
import { DynamicBackground } from "../components/DynamicBackground";

/**
 * 对话布局：为 Agent 会话页（/ai）准备的「对话优先」外壳。
 *
 * 2026-09-27 修复：此前这里用组件局部 `useState(false)` 强制默认收起侧栏，
 * 状态与全局脱节，是「不同页面侧栏形态打架」的根源之一。现在改为**只**读写
 * 全局 `store/ui.ts` 的 `isSidebarCollapsed`（持久化），与 MainLayout 完全同源
 * ——用户手动收起过，进 AI 实验室也保持收起；展开过也保持展开。
 *
 * 与标准布局的唯一差别只剩 `layout-chat` 这个 class（内容区留白收窄，见
 * styles/shell.css），这是 AI 实验室页面的视觉适配，不涉及任何状态干预。
 * **不改 pages/AI 内部组件与 ai-lab.css**：该页的高度/滚动自成一套
 * （`.ai-lab-page` 以 100dvh 为基准），外壳只负责把宽度让出来。
 */
export function ChatLayout() {
  const isSidebarCollapsed = useUiShell((s) => s.isSidebarCollapsed);
  const toggleSidebar = useUiShell((s) => s.toggleSidebar);

  return (
    <div className={`layout layout-chat${isSidebarCollapsed ? " nav-collapsed" : ""}`}>
      <DynamicBackground />
      <AppSidebar collapsed={isSidebarCollapsed} onToggle={toggleSidebar} />
      <main className="main">
        <Topbar />
        <Breadcrumb />
        <div className="content content-chat">
          <Outlet />
        </div>
      </main>
      <CommandPalette />
    </div>
  );
}
