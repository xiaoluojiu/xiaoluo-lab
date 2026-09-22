import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
// Design Token 的唯一入口，必须最先加载。
import "./styles/tokens.css";
// 跨页面复用的原子样式（比页面/业务样式先落地，避免被 !important 反压）。
import "./styles/primitives.css";
import "./index.css";
import "./theme.css";
import "./ui-overrides.css";
import "./design-foundation.css";
import "./pages/Settings/settings.css";
import "./processing-layout.css";
// 外壳层样式（顶栏 / 命令面板 / 动态背景 / Logo 标记 / 微打磨）最后加载，确保在全局四件套之后生效。
import "./styles/shell.css";
// 视觉语言层（KPI 卡 / 段落卡 / 分区标题 / 品牌带 / 插画空态）在共享样式之后、
// 集中式组件之前加载，保证它消费 Token 又不会反压外壳样式。
import "./styles/visual.css";
// 集中式通用组件层（按钮/表单/徽章/弹窗/侧栏等）最后加载，成为组件样式事实源。
import "./styles/components.css";
// 外观设置在渲染前先落到 <html data-*>，避免主题只在进过设置页后才生效、且首屏闪烁。
import { initUiSettings } from "./lib/uiSettings";

// 样式加载顺序：Token → 原子样式 → 基础样式 → 统一主题 → 项目收口层 → 共享设计基础 → 页面局部布局。
initUiSettings();
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
