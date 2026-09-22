import { DEFAULT_UI, type Density, type FontSize, type UiSettings } from "../../lib/uiSettings";
import { Choice } from "./Choice";

// 字号只留档位名：具体 px 基准属于实现细节，用户靠「界面预览」判断效果更快。
const FONT_SIZE_OPTIONS: Array<[FontSize, string]> = [
  ["small", "小"],
  ["standard", "标准"],
  ["large", "大"],
  ["xlarge", "特大"],
];

// 密度保留一句话：三档之间差别难以一眼看出，需要「适合什么场景」来支撑选择。
const DENSITY_OPTIONS: Array<[Density, string, string]> = [
  ["compact", "紧凑", "适合高信息密度页面"],
  ["standard", "标准", "平衡阅读与信息量"],
  ["comfortable", "宽松", "适合教学与长文本"],
];

/**
 * 外观与体验分区：主题、字号、密度、动态背景 + 界面预览。
 *
 * 外壳约定：本组件返回 **Fragment**，`.settings-stack` 由页面层统一提供。
 * 若这里再套一层 `<div className="settings-stack">`，两层 gap 会叠加成 32px，
 * 卡片之间出现明显错位（同类问题见 index.tsx 中 AI 分区的注释）。
 */
export function AppearanceSection({
  ui, updateUi,
}: {
  ui: UiSettings;
  updateUi: (partial: Partial<UiSettings>) => void;
}) {
  /** 恢复默认只重置外观三件套，保留用户已开启的动态背景等实验项。 */
  function resetAppearance() {
    updateUi({ theme: DEFAULT_UI.theme, fontSize: DEFAULT_UI.fontSize, density: DEFAULT_UI.density });
  }

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div><h3>主题</h3></div>
      </div>
      <div className="settings-choice-list">
        <Choice active={ui.theme === "system"} title="跟随系统" description="根据操作系统的深色模式自动切换" onClick={() => updateUi({ theme: "system" })} />
        <Choice active={ui.theme === "light"} title="浅色" description="适合长时间阅读数据和表格" onClick={() => updateUi({ theme: "light" })} />
        <Choice active={ui.theme === "dark"} title="深色" description="适合较暗环境和夜间使用" onClick={() => updateUi({ theme: "dark" })} />
      </div>
    </section>

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div><h3>字体大小</h3></div>
      </div>
      <div className="settings-choice-list compact">
        {FONT_SIZE_OPTIONS.map(([value, title]) => (
          <Choice key={value} active={ui.fontSize === value} title={title} onClick={() => updateUi({ fontSize: value })} />
        ))}
      </div>
    </section>

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div><h3>界面密度</h3></div>
      </div>
      <div className="settings-choice-list compact">
        {DENSITY_OPTIONS.map(([value, title, description]) => (
          <Choice key={value} active={ui.density === value} title={title} description={description} onClick={() => updateUi({ density: value })} />
        ))}
      </div>
    </section>

    {/* 只保留真正实现且可见效果的开关（showcase 是纯预留、无任何视觉实现，已移除）。 */}
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div><h3>视觉增强</h3></div>
      </div>
      <div className="settings-choice-list compact">
        <Choice active={ui.dynamicBg} title="动态背景" description="渐变光晕背景，在低性能设备上可能增加耗电" onClick={() => updateUi({ dynamicBg: !ui.dynamicBg })} />
      </div>
    </section>

    <section className="card settings-panel settings-preview">
      <div className="settings-panel-heading">
        <div><h3>界面预览</h3></div>
        <button className="btn btn-sm" type="button" onClick={resetAppearance}>恢复默认外观</button>
      </div>
      <div className="settings-preview-window">
        <div className="settings-preview-toolbar">
          <strong>数据集</strong><span className="muted">iris.csv · v3</span><span className="settings-preview-chip">150 × 5</span>
        </div>
        <div className="settings-preview-row"><span>sepal_length</span><span>5.1</span><span>4.9</span><span>4.7</span></div>
        <div className="settings-preview-row"><span>petal_length</span><span>1.4</span><span>1.4</span><span>1.3</span></div>
      </div>
    </section>
  </>;
}
