import { memo, useMemo } from "react";
import { useAiLab, toolDisplayName } from "../../store/aiLab";
import type { AgentEvent, AgentEventType } from "../../types/agent";
import type { AgentToolInfo } from "../../api/agent";

const EVENT_META: Record<AgentEventType, { label: string; cls: string }> = {
  route: { label: "路由", cls: "success" },
  chat: { label: "对话", cls: "success" },
  planning: { label: "规划", cls: "success" },
  permission: { label: "等待授权", cls: "warning" },
  tool_call: { label: "调用工具", cls: "success" },
  tool_result: { label: "工具结果", cls: "success" },
  validation: { label: "结果校验", cls: "success" },
  replanning: { label: "重新规划", cls: "warning" },
  completed: { label: "完成", cls: "success" },
  failed: { label: "失败", cls: "failed" },
  // Token 账本快照：不是「发生了一件事」，而是「当前花销/节省的读数」。
  usage: { label: "用量", cls: "success" },
};

function describe(ev: AgentEvent, tools: AgentToolInfo[]): string {
  const p = ev.payload ?? {};
  const tn = (name?: unknown) => (name ? toolDisplayName(String(name), tools) : "");
  switch (ev.type as AgentEventType) {
    case "route":
      return `${p.mode === "chat" ? "进入对话模式" : "进入工具模式"}：${String(p.reason ?? "")}`;
    case "planning":
      if (p.stage === "tools_retrieved") {
        const names = (p.tools as string[] | undefined) ?? [];
        const head = names.slice(0, 3).map((n) => toolDisplayName(n, tools)).join("、");
        return `检索到 ${p.count ?? names.length} 个候选工具：${head}${names.length > 3 ? " 等" : ""}`;
      }
      if (p.stage === "plan_ready") {
        return `${p.cache_hit ? "命中 Plan Cache" : "生成执行计划"}：${String(p.goal ?? "")}（${p.steps ?? 0} 步）`;
      }
      return String(p.stage ?? "上下文就绪");
    case "permission":
      return `${tn(p.tool)} ${p.reason ?? ""}`.trim();
    case "tool_call":
      return `${tn(p.tool)}（步骤 ${(p.step_index as number | undefined) ?? "-"}）`;
    case "tool_result":
      return `${tn(p.tool)} → ${p.status ?? ""}${p.summary != null ? `：${String(p.summary).slice(0, 80)}` : ""}`;
    case "validation":
      return p.valid ? "校验通过" : `校验失败：${((p.errors as string[] | undefined) ?? []).join("；")}`;
    case "replanning":
      return String(p.notes ?? "") + (p.remaining != null ? `（剩余 ${p.remaining} 步）` : "");
    // 历史缺陷：EVENT_META 里定义了 chat，但这里漏了分支，落到 default 返回空串，
    // 活动时间线上「对话」事件只有徽章没有内容，用户看不到 AI 说了什么。
    case "chat": {
      const answer = p.final_answer ?? p.content ?? p.answer ?? p.message;
      if (answer == null || answer === "") return "进入对话模式（直接回答，未调用工具）";
      return String(answer).slice(0, 160);
    }
    case "completed":
      return "任务完成";
    case "usage": {
      const u = p.token_usage as { actual?: { total_tokens?: number }; llm_calls?: number; optimization?: { estimated_saved_tokens?: number } } | undefined;
      if (!u) return "Token 账本已更新";
      const spent = (u.actual?.total_tokens ?? 0).toLocaleString();
      const saved = (u.optimization?.estimated_saved_tokens ?? 0).toLocaleString();
      return `已用 ${spent} Token · 估算省下 ${saved} Token · LLM 调用 ${u.llm_calls ?? 0} 次`;
    }
    case "failed":
      return String(p.error ?? "未知错误");
    default:
      return "";
  }
}

// Prompt 179：Agent 执行时间线（SSE 事件流可视化）。
// React.memo：父组件（AI 实验室 885 行巨型组件）在运行中会高频 setState
// （usage 事件、progress、stage），若不加 memo，时间线里每一条 li 都会随之重建。
// events 引用仅在 SSE 真正追加事件时才变化，memo 能把高频无关状态更新挡在门外。
export const AgentTimeline = memo(function AgentTimeline({ events }: { events: AgentEvent[] }) {
  const tools = useAiLab((s) => s.tools);
  // describe 依赖 tools 且每条事件都要算一遍，用 useMemo 避免每次渲染重复解析 payload。
  const rendered = useMemo(
    () =>
      events.map((ev, i) => {
        const meta = EVENT_META[ev.type as AgentEventType] ?? { label: ev.type, cls: "success" };
        const text = describe(ev, tools);
        return { ev, i, meta, text };
      }),
    [events, tools]
  );
  if (!events.length) return <div className="muted">暂无执行记录</div>;
  return (
    <ul className="timeline">
      {rendered.map(({ ev, i, meta, text }) => (
        // seq 在极端情况下可能重复（历史数据补发/重复加载），带上下标保证 key 唯一。
        <li key={`${ev.seq}-${i}`} className="timeline-item">
          <span className={`badge ${meta.cls}`}>{meta.label}</span>
          <span style={{ fontSize: 13, marginLeft: "var(--space-2)" }}>{text || "—"}</span>
          <span className="muted" style={{ fontSize: 11, marginLeft: "var(--space-2)" }}>
            {ev.created_at ? new Date(ev.created_at * 1000).toLocaleTimeString("zh-CN") : "—"}
          </span>
        </li>
      ))}
    </ul>
  );
});
