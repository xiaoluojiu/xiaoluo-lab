/** AI 实验室全局状态（zustand）：当前会话 + 关联数据集 + 工具目录缓存。 */
import { create } from "zustand";
import { listTools, type AgentToolInfo } from "../api/agent";

interface AiLabState {
  sessionId: string | null;
  datasetIds: number[];
  runId: string | null;
  tools: AgentToolInfo[];
  setSession: (sessionId: string | null) => void;
  setDatasetIds: (ids: number[]) => void;
  setRunId: (id: string | null) => void;
  setTools: (tools: AgentToolInfo[]) => void;
  loadTools: (force?: boolean) => Promise<void>;
}

export const useAiLab = create<AiLabState>((set, get) => ({
  sessionId: null,
  datasetIds: [],
  runId: null,
  tools: [],
  setSession: (sessionId) => set({ sessionId }),
  setDatasetIds: (datasetIds) => set({ datasetIds }),
  setRunId: (runId) => set({ runId }),
  setTools: (tools) => set({ tools }),
  loadTools: async (force = false) => {
    if (!force && get().tools.length) return;
    try {
      const tools = await listTools();
      set({ tools });
    } catch {
      set({ tools: [] });
    }
  },
}));

/**
 * 工具名 → 中文友好名。
 * 优先用后端 description 的首句（按中英文句号/分号切分），回退到原工具名。
 * 纯函数，便于在组件外与时间线描述中复用。
 */
export function toolDisplayName(name: string, tools: AgentToolInfo[]): string {
  const found = tools.find((t) => t.name === name);
  if (found?.description) {
    const first = found.description.split(/[。；;\n]/)[0]?.trim();
    if (first) return first;
  }
  return name;
}
