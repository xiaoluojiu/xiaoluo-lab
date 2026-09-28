/**
 * 外壳层 UI 状态（zustand）——侧栏折叠的**唯一状态源**。
 *
 * 2026-09-27 修复「不同页面侧栏行为不一致」：
 * 此前 MainLayout / ChatLayout / FullWidthLayout 三套外壳各持一份侧栏状态
 * （全局持久化偏好 / 页面局部 useState / 干脆没有侧栏），跨页跳转时形态互相打架。
 * 现在全站只有两种形态——全展开（图标+文字）与半收起（仅图标），
 * 且所有布局都必须从这里读取 `isSidebarCollapsed`，禁止页面级另立状态。
 *
 * 持久化：localStorage（key 沿用旧的 `xllab.nav.collapsed`，兼容用户已有的折叠偏好，
 * 刷新与路由跳转均不丢失）。窄屏（<768px）的横向导航兜底由 CSS 承担，与本状态无关。
 */
import { create } from "zustand";

const NAV_COLLAPSED_KEY = "xllab.nav.collapsed";

interface UiShellState {
  /** 用户手动折叠侧栏的偏好（持久化）。全站唯一事实源，所有布局共用。 */
  isSidebarCollapsed: boolean;
  /** 命令面板是否打开 */
  commandOpen: boolean;
  setSidebarCollapsed: (v: boolean) => void;
  toggleSidebar: () => void;
  openCommand: () => void;
  closeCommand: () => void;
  toggleCommand: () => void;
}

function readSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem(NAV_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export const useUiShell = create<UiShellState>((set, get) => ({
  isSidebarCollapsed: readSidebarCollapsed(),
  commandOpen: false,
  setSidebarCollapsed: (v) => {
    try {
      localStorage.setItem(NAV_COLLAPSED_KEY, v ? "1" : "0");
    } catch {
      /* localStorage 不可用时静默降级 */
    }
    set({ isSidebarCollapsed: v });
  },
  toggleSidebar: () => get().setSidebarCollapsed(!get().isSidebarCollapsed),
  openCommand: () => set({ commandOpen: true }),
  closeCommand: () => set({ commandOpen: false }),
  toggleCommand: () => set({ commandOpen: !get().commandOpen }),
}));
