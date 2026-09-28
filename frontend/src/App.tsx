import { useEffect } from "react";
import { BrowserRouter, useLocation } from "react-router-dom";
import { AppRouter } from "./router";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastProvider } from "./components/ToastProvider";
import { useGlobalContext } from "./store/globalStore";

/**
 * 路由入口的 URL → 全局上下文同步（只做一次，作用于所有页面）。
 *
 * 历史上 `?dataset=` 只在 DatasetSelector 内部被读取，于是每个页面各读一遍、
 * 读到的还只是页面自己的本地 state —— 离开带 query 的入口（刷新、点侧栏、命令面板）
 * 上下文就丢了。现在收敛到一处：URL 里若有 `?dataset=` / `?version=`，
 * 进路由时把它同步进 globalStore；**参数缺失不清空**，持久化的选择继续有效。
 */
function RouteContextSync() {
  const { search } = useLocation();
  useEffect(() => {
    useGlobalContext.getState().syncFromSearch(search);
  }, [search]);
  return null;
}

// 连接 Router；外层套错误边界（避免白屏）与全局通知中心（统一成功/失败反馈）。
// ToastProvider 放在 BrowserRouter 内侧：错误提示里的「去处理缺失值」等按钮
// 需要走 SPA 路由跳转（Link），必须在 Router 上下文里才能渲染。
export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ToastProvider>
          <RouteContextSync />
          <AppRouter />
        </ToastProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
