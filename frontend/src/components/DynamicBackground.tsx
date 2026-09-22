/**
 * 动态背景占位层。
 *
 * 设计意图（用户需求：为未来引入渐变过渡与动态背景预留结构/接口）：
 * - 默认完全隐藏（opacity:0），仅当 <html data-bg="dynamic"> 时由 shell.css 显现。
 * - 显隐与动画纯由 CSS 属性驱动，组件本身无 JS 状态，便于未来扩展而不动调用方。
 * - 光晕颜色 / 渐变来自 tokens.css 的 --bg-glow-* / --bg-gradient，改主题只改 Token。
 *
 * 未来扩展点：
 *   1. 渐变过渡：在 tokens.css 调大 --bg-glow-* 透明度或给 .bg-layer 设 --bg-gradient。
 *   2. 动态背景：在 shell.css 增加更多 .bg-glow 或 canvas/粒子层，组件结构已预留三枚光晕位。
 *
 * 注：曾有一个 html[data-showcase] 的「示例模式」预留开关，因为**没有任何视觉实现**
 * 而只是一个空开关，已作为死入口移除（与数据分析模块删掉「查看样本」同一治理思路）。
 */
export function DynamicBackground() {
  return (
    <div className="bg-layer" aria-hidden="true">
      <span className="bg-glow bg-glow-1" />
      <span className="bg-glow bg-glow-2" />
      <span className="bg-glow bg-glow-3" />
    </div>
  );
}
