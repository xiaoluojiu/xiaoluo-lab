/**
 * 外观设置（主题 / 字号 / 密度）的唯一读写入口。
 *
 * 历史问题：data-theme / data-font-size 只在「设置页挂载时」才被写入 <html>，
 * 用户不先进一次设置页主题就是默认值，且写入前存在首屏闪烁。
 *
 * 现在：main.tsx 启动时先调用 initUiSettings()，全站任何页面进入都立刻生效。
 */

export type Theme = "light" | "dark" | "system";
export type FontSize = "small" | "standard" | "large" | "xlarge";
export type Density = "compact" | "standard" | "comfortable";

export interface UiSettings {
  theme: Theme;
  fontSize: FontSize;
  density: Density;
  /** 动态背景（光晕 / 渐变过渡）：默认关闭，由 shell.css 在 html[data-bg="dynamic"] 时显现。 */
  dynamicBg: boolean;
}

export const UI_STORAGE_KEY = "xllab.ui";

export const DEFAULT_UI: UiSettings = { theme: "system", fontSize: "standard", density: "standard", dynamicBg: false };

const FONT_SIZES: FontSize[] = ["small", "standard", "large", "xlarge"];
const DENSITIES: Density[] = ["compact", "standard", "comfortable"];

export function readUi(): UiSettings {
  try {
    const parsed = JSON.parse(localStorage.getItem(UI_STORAGE_KEY) || "{}") as Partial<UiSettings>;
    return {
      theme: parsed.theme === "dark" || parsed.theme === "light" ? parsed.theme : "system",
      fontSize: FONT_SIZES.includes(parsed.fontSize as FontSize) ? (parsed.fontSize as FontSize) : "standard",
      density: DENSITIES.includes(parsed.density as Density) ? (parsed.density as Density) : "standard",
      dynamicBg: typeof parsed.dynamicBg === "boolean" ? parsed.dynamicBg : false,
    };
  } catch {
    return { ...DEFAULT_UI };
  }
}

export function writeUi(settings: UiSettings): void {
  localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(settings));
  applyUi(settings);
}

function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return theme;
}

/** 把设置写到 <html data-*> 上，供 tokens.css 消费。 */
export function applyUi(settings: UiSettings = readUi()): void {
  const root = document.documentElement;
  root.dataset.theme = resolveTheme(settings.theme);
  root.dataset.fontSize = settings.fontSize;
  root.dataset.density = settings.density;
  root.dataset.bg = settings.dynamicBg ? "dynamic" : "static";
}

/**
 * 应用启动时调用一次，并订阅系统深色模式变化。
 *
 * 注意：这里**故意不返回清理函数**——它只在应用启动时调用一次，
 * 监听需要覆盖整个应用生命周期。若未来出现热更新 / 多实例挂载场景，
 * 需改为返回 cleanup 并让调用方保存。
 */
export function initUiSettings(): void {
  applyUi();
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const onChange = () => {
    const current = readUi();
    if (current.theme === "system") applyUi(current);
  };
  media.addEventListener?.("change", onChange);
}

/** 切换动态背景（光晕 / 渐变）。 */
export function setDynamicBg(enabled: boolean): void {
  writeUi({ ...readUi(), dynamicBg: enabled });
}
