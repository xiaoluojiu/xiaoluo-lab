import { Suspense, lazy, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { MainLayout } from "../layouts/MainLayout";
import { ChatLayout } from "../layouts/ChatLayout";
import { PageLoading } from "../components/PageLoading";
import NotFound from "../pages/NotFound";
import { APP_ROUTES, type LayoutKind } from "../config/routes";

// 沉浸式编辑器单独走一条路由，挂在任何外壳之外：
// 编辑当前流程时不需要全局导航与顶栏，把它们收起来才是「全屏」的本意。
const WorkflowEditor = lazy(() => import("../pages/Workflow/Editor"));

function SuspensePage({ children }: { children: ReactNode }) {
  return <Suspense fallback={<PageLoading />}>{children}</Suspense>;
}

/** 布局种类 → 外壳元素。新增布局时这里补一行，并在 config/routes.ts 登记。 */
const LAYOUTS: Record<LayoutKind, ReactNode> = {
  standard: <MainLayout />,
  chat: <ChatLayout />,
};

/** 渲染顺序固定，避免不同布局组之间的注册顺序随配置表增删而漂移。 */
const LAYOUT_ORDER: LayoutKind[] = ["standard", "chat"];

/**
 * 路由表。
 *
 * 结构 = 「按布局分组，组内放页面」，页面的布局归属全部读 `config/routes.ts`，
 * 这里不再逐个手写 `<Route>`，避免「加了页面忘了选布局」——
 * 想改某页的外壳只改配置表里那一行的 `layout`。
 *
 * 两个特例（刻意不进配置表）：
 * - `/workflow/editor/:id`：不套外壳的沉浸式编辑器；
 * - `*`：404 兜底，固定在标准布局组（保证未知路径仍有侧栏可回导航）。
 */
export function AppRouter() {
  return (
    <Routes>
      <Route
        path="/workflow/editor/:id"
        element={<SuspensePage><WorkflowEditor /></SuspensePage>}
      />
      {LAYOUT_ORDER.map((kind) => (
        <Route key={kind} element={LAYOUTS[kind]}>
          {APP_ROUTES.filter((route) => route.layout === kind).map(({ path, component: Page }) => (
            <Route key={path} path={path} element={<SuspensePage><Page /></SuspensePage>} />
          ))}
          {/* 未知路由给出明确的 404，而不是静默跳回首页。 */}
          {kind === "standard" && <Route path="*" element={<NotFound />} />}
        </Route>
      ))}
    </Routes>
  );
}
