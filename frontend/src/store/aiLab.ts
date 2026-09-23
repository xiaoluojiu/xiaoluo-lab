/** AI 实验室全局状态（zustand）：当前会话 + 关联数据集 + 工具目录缓存。 */
import { create } from "zustand";
import { listTools, type AgentToolInfo } from "../api/agent";
import { toolLabel } from "../lib/toolLabel";

/** 工具目录缓存的有效期。
 *
 * 工具清单只随**后端代码变更**而变（新建工具要改 builtin.py 再重启），
 * 页面来回切换时不该重复拉取；但也不能像原来那样「进程内一次加载、永不失效」：
 * 后端重启或热更之后，前端会一直拿着旧清单，表现为「界面上找不到新工具」。
 */
const TOOLS_TTL_MS = 5 * 60_000;

interface AiLabState {
  sessionId: string | null;
  datasetIds: number[];
  runId: string | null;
  tools: AgentToolInfo[];
  /** 上次成功拉取工具清单的时间戳（epoch ms），0 表示从未拉取。 */
  toolsLoadedAt: number;
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
  toolsLoadedAt: 0,
  setSession: (sessionId) => set({ sessionId }),
  setDatasetIds: (datasetIds) => set({ datasetIds }),
  setRunId: (runId) => set({ runId }),
  setTools: (tools) => set({ tools, toolsLoadedAt: Date.now() }),
  loadTools: async (force = false) => {
    const { tools, toolsLoadedAt } = get();
    const fresh = Date.now() - toolsLoadedAt < TOOLS_TTL_MS;
    // force=true 供「刷新工具目录」按钮使用：用户显式点了就该立刻重拉。
    if (!force && tools.length && fresh) return;
    try {
      const loaded = await listTools();
      set({ tools: loaded, toolsLoadedAt: Date.now() });
    } catch {
      // 拉取失败不清空已有清单：宁可展示（可能略微过期的）名称，
      // 也不要把整个界面退回原始工具名。
      if (force) set({ tools: [], toolsLoadedAt: 0 });
    }
  },
}));

/**
 * 工具名 → 中文友好名。
 * 取后端 description 的首句并在 30 字处收尾（超出加省略号），回退到原工具名。
 * 纯逻辑实现在 `lib/toolLabel.ts`（零依赖、可被单测），这里只做转发，
 * 避免组件层各自改写这套规则。
 */
export function toolDisplayName(name: string, tools: AgentToolInfo[]): string {
  return toolLabel(name, tools);
}
