import { BrowserRouter } from "react-router-dom";
import { AppRouter } from "./router";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastProvider } from "./components/ToastProvider";

// 连接 Router；外层套错误边界（避免白屏）与全局通知中心（统一成功/失败反馈）。
export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <BrowserRouter>
          <AppRouter />
        </BrowserRouter>
      </ToastProvider>
    </ErrorBoundary>
  );
}
