import { Suspense, lazy, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { MainLayout } from "../layouts/MainLayout";
import { PageLoading } from "../components/PageLoading";
import NotFound from "../pages/NotFound";

// 体积较大的工作区页面改为按需加载，避免首次访问就下载整个应用（此前单 chunk 856.83 kB）。
// MainLayout 与 NotFound 保持同步加载：前者是外壳，后者体积极小。
const Home = lazy(() => import("../pages/Home"));
const Datasets = lazy(() => import("../pages/Datasets"));
const DatasetDetail = lazy(() => import("../pages/Datasets/Detail"));
const Processing = lazy(() => import("../pages/Processing"));
const Analysis = lazy(() => import("../pages/Analysis"));
const ML = lazy(() => import("../pages/ML"));
const AI = lazy(() => import("../pages/AI"));
const Workflow = lazy(() => import("../pages/Workflow"));
const Experiments = lazy(() => import("../pages/Experiments"));
const Reports = lazy(() => import("../pages/Reports"));
const ReportDetail = lazy(() => import("../pages/Reports/Detail"));
const Settings = lazy(() => import("../pages/Settings"));
const Extensions = lazy(() => import("../pages/Extensions"));
const Learning = lazy(() => import("../pages/Learning"));
const LearningWorkspace = lazy(() => import("../pages/LearningWorkspace"));
const LearningHistory = lazy(() => import("../pages/LearningHistory"));
const LearningCard = lazy(() => import("../pages/LearningCard"));

function SuspensePage({ children }: { children: ReactNode }) {
  return <Suspense fallback={<PageLoading />}>{children}</Suspense>;
}

// 路由表：学习中心收缩为实验任务、实验工作台、实验记录三个入口。
export function AppRouter() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/" element={<SuspensePage><Home /></SuspensePage>} />
        <Route path="/datasets" element={<SuspensePage><Datasets /></SuspensePage>} />
        <Route path="/datasets/:id" element={<SuspensePage><DatasetDetail /></SuspensePage>} />
        <Route path="/processing" element={<SuspensePage><Processing /></SuspensePage>} />
        <Route path="/analysis" element={<SuspensePage><Analysis /></SuspensePage>} />
        <Route path="/ml" element={<SuspensePage><ML /></SuspensePage>} />
        <Route path="/ai" element={<SuspensePage><AI /></SuspensePage>} />
        <Route path="/workflow" element={<SuspensePage><Workflow /></SuspensePage>} />
        <Route path="/experiments" element={<SuspensePage><Experiments /></SuspensePage>} />
        <Route path="/reports" element={<SuspensePage><Reports /></SuspensePage>} />
        {/* 报告详情独立成页：正文不再内联展开在列表下方，链接可直接分享。 */}
        <Route path="/reports/:key" element={<SuspensePage><ReportDetail /></SuspensePage>} />
        <Route path="/settings" element={<SuspensePage><Settings /></SuspensePage>} />
        <Route path="/extensions" element={<SuspensePage><Extensions /></SuspensePage>} />
        <Route path="/learning" element={<SuspensePage><Learning /></SuspensePage>} />
        <Route path="/learning/workspace" element={<SuspensePage><LearningWorkspace /></SuspensePage>} />
        <Route path="/learning/history" element={<SuspensePage><LearningHistory /></SuspensePage>} />
        <Route path="/learning/card/:cardId" element={<SuspensePage><LearningCard /></SuspensePage>} />
        {/* 未知路由给出明确的 404，而不是静默跳回首页。 */}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
