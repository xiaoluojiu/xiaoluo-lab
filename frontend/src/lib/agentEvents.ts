/**
 * SSE 事件 → 界面状态增量的**纯翻译层**（零 React / 零 Store 依赖，可被 Node 单测）。
 *
 * 放在 lib/ 是为了能被测试直接导入：原先这段 if/else 埋在 300 行的 send() 里，
 * 进度回退、自动放行、校验提示这几处历史 bug 只能靠肉眼看界面回归。
 * 它只做翻译，不碰网络、不做异步；副作用（自动放行、起轮询）在 useAgentRun。
 */
import type { AgentEvent, ChatMessage, ClarificationRequest, InspectorTab, PermissionRequest } from "../types/agent";

const STAGE_LABEL: Record<string, string> = {
  context_ready: "准备上下文",
  tools_retrieved: "检索工具",
  plan_ready: "执行计划",
  direct_chat: "普通对话",
  clarification_required: "等待补充信息",
  clarification_answered: "已补充信息",
  // 本地 Router（Qwen 神经 / 词法）已高置信识别出任务，直接构造单步计划执行。
  // 缺这一条时界面会把裸串 "local_direct" 直接显示给用户。
  local_direct: "本地路由直连",
  dynamic_stop: "动态步骤结束",
  remote_escalated: "已升级远程",
};

/**
 * planning 各 stage 的进度下限。
 *
 * 用查表而不是 if/else 链：这个分支历史上出过两次「新增 stage 忘了加分支」的
 * 事故（preflight、local_direct），症状都是**进度条永久卡在某个数不动**——
 * 因为未知 stage 会落到兜底值 8，而它前一步已经是 15 了。查表至少让兜底值
 * 是「继续前进」而不是「原地不动」。
 */
const PLANNING_PROGRESS: Record<string, number> = {
  context_ready: 10,
  tools_retrieved: 15,
  local_direct: 25,
  plan_ready: 25,
  remote_escalated: 30,
  dynamic_stop: 90,
};

/** 一条事件对界面的增量影响；未出现的字段表示「不改」。 */
export interface EventEffects {
  stage?: string;
  /** 定值，或「与当前值取较大者」——进度条只允许前进，回退只在重新规划时显式发生。 */
  progress?: number | ((prev: number) => number);
  notice?: string;
  tab?: InspectorTab;
  permission?: PermissionRequest;
  /**
   * 待澄清问题。`null` 表示「澄清已结束，清掉面板」；`undefined` 表示「不改」。
   * 三态是刻意的：`clarification` 事件既有发起（stage=clarification_required）
   * 也有回答（stage=answered）两种，用二态布尔无法表达「回答后要撤掉面板」。
   */
  clarification?: ClarificationRequest | null;
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
      const bump = PLANNING_PROGRESS[s] ?? 20;
      // 远程规划不可用、已降级到平台内置规则：这是「数据仍能跑出来、但计划不是大模型定的」
      // 的关键事实，必须让用户看见，而不是让他在结果里自己猜。
      const degraded = s === "plan_ready" && p.planner_fallback === true;
      return {
        stage: STAGE_LABEL[s] ?? s,
        progress: (prev) => Math.max(prev, bump),
        notice: degraded ? "远程规划不可用，已改用平台内置规则规划（执行结果仍是真实工具跑出来的）。" : undefined,
      };
    }
    case "preflight":
      // Pre-flight 是开工前的确定性检查（不是执行步骤）。它有可能直接转入反问，
      // 原先这里没有分支 ⇒ 事件落到 default，进度停在 route 给的 8% 一动不动。
      return {
        stage: "开工前检查",
        progress: (prev) => Math.max(prev, 15),
      };
    case "chat":
      // 纯对话分支。后端在 `_direct_chat` 前会发一条 `chat` 事件，
      // 原先没有 case ⇒ 落到 default，进度从 route 的 40% 直接跳到 completed 的 100%，
      // 中间毫无反馈；一旦远程调用慢，界面就是「一动不动的 40%」。
      return {
        stage: STAGE_LABEL[String(p.stage ?? "direct_chat")] ?? "普通对话",
        progress: (prev) => Math.max(prev, 70),
      };
    case "clarification": {
      // 后端在等待用户补充信息。这是「等待人」，不是「卡住」——
      // 必须把问题原文透出去（渲染成气泡 / 选择器由调用方负责），
      // 否则用户只看到一个不动的进度条，没有任何可操作的东西。
      const answered = String(p.stage ?? "") === "answered";
      return {
        stage: answered ? "已补充信息" : "等待补充信息",
        progress: (prev) => Math.max(prev, answered ? prev : 30),
        clarification: answered
          ? null
          : {
              code: String(p.code ?? ""),
              question: String(p.question ?? ""),
              options: Array.isArray(p.options)
                ? (p.options as { value: string; label?: string; note?: string; hint?: string }[]).map((o) => ({
                    value: String(o?.value ?? ""),
                    label: o?.label ? String(o.label) : String(o?.value ?? ""),
                    // 后端字段是 note，hint 只是历史别称：两个都认，避免改名后静默失效。
                    note: (o?.note ?? o?.hint) ? String(o?.note ?? o?.hint) : undefined,
                  }))
                : [],
              default: p.default == null ? null : String(p.default),
              outcome: p.outcome == null ? undefined : String(p.outcome),
              step_index: p.step_index == null ? null : Number(p.step_index),
              tool: p.tool == null ? null : String(p.tool),
            },
      };
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
