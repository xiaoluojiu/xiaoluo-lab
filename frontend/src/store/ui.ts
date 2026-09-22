/**
 * 外壳层 UI 状态（zustand）。
 *
 * 仅承载「外壳交互态」：侧栏折叠（持久化）、命令面板开关。
 * 外观主题（主题 / 字号 / 密度 / 动态背景 / 示例模式）仍统一走 lib/uiSettings，
 * 避免两套状态源。
 */
import { create } from "zustand";

const NAV_COLLAPSED_KEY = "xllab.nav.collapsed";

interface UiShellState {
  /** 用户手动折叠侧栏的偏好（持久化）。窄屏自动折叠为另一状态，二者取或。 */
  navCollapsed: boolean;
  /** 命令面板是否打开 */
  commandOpen: boolean;
  setNavCollapsed: (v: boolean) => void;
  toggleNav: () => void;
  openCommand: () => void;
  closeCommand: () => void;
  toggleCommand: () => void;
}

function readNavCollapsed(): boolean {
  try {
    return localStorage.getItem(NAV_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export const useUiShell = create<UiShellState>((set, get) => ({
  navCollapsed: readNavCollapsed(),
  commandOpen: false,
  setNavCollapsed: (v) => {
    try {
      localStorage.setItem(NAV_COLLAPSED_KEY, v ? "1" : "0");
    } catch {
      /* localStorage 不可用时静默降级 */
    }
    set({ navCollapsed: v });
  },
  toggleNav: () => get().setNavCollapsed(!get().navCollapsed),
  openCommand: () => set({ commandOpen: true }),
  closeCommand: () => set({ commandOpen: false }),
  toggleCommand: () => set({ commandOpen: !get().commandOpen }),
}));
