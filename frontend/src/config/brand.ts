/**
 * 品牌配置 —— 自定义 Logo 与主题的统一入口。
 *
 * ============================ 换 Logo 的唯一入口 ============================
 * 所有位置（侧栏品牌区、index.html 的 favicon / apple-touch-icon）都从这里取值，
 * 不需要改任何组件或 HTML 代码。
 *
 * 资源统一放在 `src/assets/brand/`，命名固定：
 *   logo.png         512x512 主 logo（透明底），组件引用
 *   logo.svg         SVG 包装（内嵌同一张图，供需要 .svg 后缀的场景）
 *   logo-source.png  原始归档图（2048，仅用于重新切图，不参与运行时）
 * 站点图标放 `public/`：favicon.ico / favicon-32.png / favicon-16.png / apple-touch-icon.png
 *
 * 后续换 logo 的操作：
 *   1. 用新图覆盖 `src/assets/brand/logo-source.png`；
 *   2. 跑一次 `python _build_brand.py`（仓库根目录），自动重切全部尺寸并同步 favicon；
 *   3. 若要改文件名，同步改本文件的 logoSrc 与 index.html 的图标引用。
 *
 * 其它可调项：
 *   - 品牌名 / 简称：修改 `name` / `shortName`。
 *   - 品牌渐变（Logo 占位标记、主按钮、强调色）：修改 `accentFrom` / `accentTo`，
 *     由 tokens.css 的 `--brand-gradient` 消费（见 styles/tokens.css）。
 *
 * 当前为「小洛实验室」配置。
 */
import logoSrc from "../assets/brand/logo.png";

export interface BrandConfig {
  /** 完整品牌名 */
  name: string;
  /** 收起态 / 极小空间下的简称 */
  shortName: string;
  /** 自定义 Logo 图片地址；留空则渲染默认渐变标记（.logo-mark）。 */
  logoSrc?: string;
  /** 品牌渐变起色（预留，未来可由后台/配置文件注入）。 */
  accentFrom?: string;
  /** 品牌渐变止色（预留）。 */
  accentTo?: string;
  /**
   * 开源仓库地址（侧栏底部入口）。
   * 当前**故意留空**：底部卡片会渲染成「未配置」的不可点状态，
   * 填入地址后自动变成外链跳转，无需改任何组件代码。
   */
  repoUrl?: string;
  /** 侧栏底部卡的副标题。 */
  repoLabel?: string;
}

export const brand: BrandConfig = {
  name: "小洛实验室",
  shortName: "洛",
  logoSrc,
};
