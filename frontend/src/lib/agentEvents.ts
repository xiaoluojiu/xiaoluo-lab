/**
 * SSE 事件 → 界面状态增量的**纯翻译层**（零 React / 零 Store 依赖，可被 Node 单测）。
 *
 * 放在 lib/ 是为了能被测试直接导入：原先这段 if/else 埋在 300 行的 send() 里，
 * 进度回退、自动放行、校验提示这几处历史 bug 只能靠肉眼看界面回归。
 * 它只做翻译，不碰网络、不做异步；副作用（自动放行、起轮询）在 useAgentRun。
 */
import type { AgentEvent, ChatMessage, InspectorTab, PermissionRequest } from "../types/agent";

const STAGE_LABEL: Record<string, string> = {
  context_ready: "准备上下文",
  tools_retrieved: "检索工具",
  plan_ready: "执行计划",
  direct_chat: "普通对话",
};

/** 一条事件对界面的增量影响；未出现的字段表示「不改」。 */
export interface EventEffects {
  stage?: string;
  /** 定值，或「与当前值取较大者」——进度条只允许前进，回退只在重新规划时显式发生。 */
  progress?: number | ((prev: number) => number);
  notice?: string;
  tab?: InspectorTab;
  permission?: PermissionRequest;
  append?: ChatMessage;
}

/**
 * @param label 工具名 → 中文名。由调用方注入（组件层传 `(n) => toolDisplayName(n, tools)`），
 *              这样本文件不必依赖 Store / React。
 */
export function eventEffects(ev: AgentEvent, label: (tool: string) => string = (tool) => tool): EventEffects {
  const p = ev.payload ?? {};
  switch (ev.type) {
    case "route": {
      const chat = String(p.mode ?? "chat") === "chat";
      const reason = String(p.reason ?? "");
      return {
        stage: `${chat ? "对话模式" : "工具模式"} · ${reason}`,
        progress: (prev) => Math.max(prev, chat ? 40 : 8),
      };
    }
    case "planning": {
      const s = String(p.stage ?? "planning");
      const bump = s === "plan_ready" ? 25 : s === "tools_retrieved" ? 15 : 8;
      return { stage: STAGE_LABEL[s] ?? s, progress: (prev) => Math.max(prev, bump) };
    }
    case "tool_call":
      return {
        tab: "activity",
        stage: `执行工具：${label(String(p.tool ?? "tool"))}`,
        progress: (prev) => Math.max(prev, Math.min(90, prev + 8)),
      };
    case "tool_result":
      return { stage: `工具完成：${label(String(p.tool ?? "tool"))}` };
    case "validation": {
      if (p.valid !== false) return {};
      const errs = (p.errors as string[] | undefined)?.join("；") || "未知原因";
      return { notice: `第 ${Number(p.step_index ?? 0) + 1} 步校验失败：${errs}` };
    }
    case "replanning": {
      const notes = String(p.notes ?? "");
      // 重试时进度显式回退，让用户感知到「卡住后重来」，而不是停在同一个数字上。
      return {
        stage: p.retry ? `重试中 · ${notes}` : `重新规划 · ${notes}`,
        progress: (prev) => Math.max(5, prev - 15),
        notice: notes || undefined,
      };
    }
    case "permission": {
      if (!p.tool) return {};
      return {
        permission: {
          tool: String(p.tool),
          arguments: (p.arguments as Record<string, unknown>) ?? {},
          reason: String(p.reason ?? ""),
          step_index: Number(p.step_index ?? 0),
        },
      };
    }
    case "completed":
      return {
        progress: 100,
        stage: "任务完成",
        append:
          typeof p.final_answer === "string"
            ? {
                role: "assistant",
                content: p.final_answer,
                source: (p.answer_source as ChatMessage["source"]) ?? null,
              }
            : undefined,
      };
    case "failed":
      return {
        progress: 100,
        stage: "任务失败",
        append: { role: "assistant", content: `执行失败：${String(p.error ?? "未知错误")}` },
      };
    default:
      // usage 不在这里处理：Token 账本由 useAgentUsage 直接从事件列表派生。
      return {};
  }
}
