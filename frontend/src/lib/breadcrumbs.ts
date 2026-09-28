/**
 * 面包屑构建 —— 纯函数，不依赖 React / 不做网络请求，可直接单测。
 *
 * ## 为什么单独成模块
 *
 * 「当前层级路径」这件事有两类信息：
 *   1. **路由结构**（`/datasets/:id`、`/reports/:key`）—— 路径里就能读到；
 *   2. **数据上下文**（当前数据集叫什么名字、当前是第几版）—— 路径里没有，
 *      只存在于 `store/globalStore.ts`。
 * 把两者拼成面包屑的逻辑写进组件就会和渲染缠在一起、也没法测；
 * 这里只接收「已经拿到的名字 / 版本号」，输出一份 Crumb[]，
 * 组件负责取数据、模块负责算路径。
 *
 * ## 层级口径
 *
 * `首页 / 数据集 / 数据集A / 版本 v2 / 数据分析` ——
 * 前四级来自「路径 + 数据上下文」，最后一级是当前页面。
 * 只有**会消费数据上下文的页面**（处理 / 分析 / 机器学习 / 流程 / 实验 / 报告）
 * 才把「数据集 / 版本」链拼进去：在数据集列表页或设置页挂一条
 * 「数据集A / 版本 v2」既没意义、点了也会丢掉上下文。
 */

/** 面包屑的一级。带 `to` = 可点击；末级不带 `to`（当前页）。 */
export interface Crumb {
  label: string;
  to?: string;
}

/** 根节点文案。用面包屑惯例的「首页」而非侧栏的「工作台」。 */
export const HOME_LABEL = "首页";

/**
 * 静态路由 → 标题。
 * 文案与 `config/nav.ts` 对齐（「数据集列表」在面包屑里略作「数据集」），
 * 新增页面时这里补一行即可。
 */
export const ROUTE_LABELS: Record<string, string> = {
  "/": HOME_LABEL,
  "/datasets": "数据集",
  "/processing": "数据处理",
  "/analysis": "数据分析",
  "/ml": "机器学习",
  "/ai": "AI 实验室",
  "/workflow": "工作流",
  "/experiments": "实验中心",
  "/reports": "报告中心",
  "/extensions": "扩展中心",
  "/learning": "学习中心",
  "/settings": "设置",
};

/**
 * 会消费「当前数据集 / 版本」上下文的页面。
 * 只有这些页面的面包屑里才插入 数据集 → 版本 两级。
 */
const CONTEXT_PAGES: Record<string, string> = {
  "/processing": ROUTE_LABELS["/processing"],
  "/analysis": ROUTE_LABELS["/analysis"],
  "/ml": ROUTE_LABELS["/ml"],
  "/workflow": ROUTE_LABELS["/workflow"],
  "/experiments": ROUTE_LABELS["/experiments"],
  "/reports": ROUTE_LABELS["/reports"],
};

/**
 * 不渲染全局面包屑的路径前缀。
 *
 * AI 实验室是会话式三栏工作台，自成一套信息架构，顶栏之下的路径条
 * 对它没有意义（且会挤压聊天列高度），因此整体排除，不触碰该模块。
 */
const HIDDEN_PREFIXES = ["/ai"];

/** 该路径是否应当隐藏面包屑。 */
export function isBreadcrumbHidden(pathname: string): boolean {
  return HIDDEN_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

/**
 * 从路径里取出数据集 id。`/datasets/7` → 7，其余（列表页 / 非数字段）→ null。
 * 非数字的 `:id`（如手改 URL）不当作有效 id，避免拼出 `/datasets/abc` 这种死链。
 */
export function datasetIdFromPath(pathname: string): number | null {
  const matched = /^\/datasets\/([^/]+)/.exec(pathname);
  if (!matched) return null;
  const id = Number(matched[1]);
  return Number.isInteger(id) && id > 0 ? id : null;
}

export interface BuildBreadcrumbsInput {
  /** 当前路径（`useLocation().pathname`）。 */
  pathname: string;
  /** 当前数据集 id；来自路径或 globalStore。 */
  datasetId?: number | null;
  /** 数据集展示名；未解析出来时传 null，会回退成「数据集 #id」。 */
  datasetName?: string | null;
  /** 当前版本号；null = 未指定（等价于最新版本），此时不显示版本这一级。 */
  version?: number | null;
}

/** 数据集那一级的文案：优先真名，取不到名字时退回 id，至少能定位。 */
function datasetLabel(datasetId: number, datasetName: string | null): string {
  const name = datasetName?.trim();
  return name || `数据集 #${datasetId}`;
}

/**
 * 生成面包屑。
 *
 * 约定：返回值最后一项不带 `to`（当前页不可点）；只有一个元素时（首页）
 * 该元素即为当前页。返回空数组表示「这条路径没有可展示的层级」。
 */
export function buildBreadcrumbs(input: BuildBreadcrumbsInput): Crumb[] {
  const { pathname, datasetId = null, datasetName = null, version = null } = input;

  // 首页自身：只有一级，且就是当前页。
  if (pathname === "/") return [{ label: HOME_LABEL }];

  const crumbs: Crumb[] = [{ label: HOME_LABEL, to: "/" }];

  // ---------- 详情页：动态段换成名称，父级可一步回列表 ----------
  if (pathname.startsWith("/datasets/")) {
    crumbs.push({ label: ROUTE_LABELS["/datasets"], to: "/datasets" });
    crumbs.push({
      label: datasetId !== null ? datasetLabel(datasetId, datasetName) : "数据集详情",
    });
    return crumbs;
  }

  if (pathname.startsWith("/reports/")) {
    crumbs.push({ label: ROUTE_LABELS["/reports"], to: "/reports" });
    crumbs.push({ label: "报告详情" });
    return crumbs;
  }

  if (pathname.startsWith("/learning/")) {
    crumbs.push({ label: ROUTE_LABELS["/learning"], to: "/learning" });
    crumbs.push({ label: learningSubLabel(pathname) });
    return crumbs;
  }

  // ---------- 消费数据上下文的页面：插入 数据集 → 版本 ----------
  if (CONTEXT_PAGES[pathname] && datasetId !== null) {
    crumbs.push({ label: ROUTE_LABELS["/datasets"], to: "/datasets" });
    crumbs.push({
      label: datasetLabel(datasetId, datasetName),
      to: `/datasets/${datasetId}`,
    });
    if (version !== null) {
      // 版本不是独立路由，对应页面是数据集详情的「版本」页签 —— 点它直接跳到版本时间线。
      crumbs.push({ label: `版本 v${version}`, to: `/datasets/${datasetId}?tab=versions` });
    }
  }

  crumbs.push({ label: pageLabel(pathname) });
  return crumbs;
}

/** 学习中心的两个子页要互相区分，且都能一步回列表。 */
function learningSubLabel(pathname: string): string {
  if (pathname === "/learning/workspace") return "实验台";
  if (pathname === "/learning/history") return "学习记录";
  if (pathname.startsWith("/learning/card/")) return "我的卡片";
  return "详情";
}

/** 静态路由文案；不在表里的一律降级为「页面」（如 404）。 */
function pageLabel(pathname: string): string {
  const first = `/${pathname.split("/").filter(Boolean)[0] ?? ""}`;
  return ROUTE_LABELS[pathname] ?? ROUTE_LABELS[first] ?? "页面";
}
