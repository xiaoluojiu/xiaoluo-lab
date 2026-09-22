import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // 把体积大户与框架拆成独立 chunk，避免单包超过 ~850KB 的告警，
        // 也让图表库（仅分析 / ML / 报告用到）按需加载。
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("recharts") || id.includes("/d3-") || id.includes("victory")) return "vendor-charts";
          if (id.includes("react-router") || id.includes("@remix-run")) return "vendor-router";
          if (id.includes("react-dom") || id.includes("/react/") || id.includes("scheduler")) return "vendor-react";
          return "vendor";
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        // Windows 本地开发时显式使用 IPv4，避免 localhost 被解析到
        // ::1，而 Uvicorn 仅在 IPv4 监听导致代理请求悬挂/超时。
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // ⚠️ 不要给 SSE 长连接设短空闲超时：训练进度、Agent 对话是流式响应，
        // 中间（大表加载 / 预处理 / 评估）会有 >15s 的空闲间隔，15s 的
        // proxyTimeout 会把连接误杀，前端报 network error（后端其实仍在跑）。
        // 这里放宽到 10 分钟兜底（真悬挂仍会断开），不再误杀 30s 级的训练流。
        timeout: 600_000,
        proxyTimeout: 600_000,
      },
    },
  },
});
