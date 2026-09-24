/**
 * Token 用量：运行中的实时账本 + 会话内多次运行的用量历史。
 *
 * 两处「减法」：
 * - 实时账本不再是一个独立 state。它本来只在 `usage` 事件里被 setLiveUsage 写入，
 *   现在直接从事件列表派生（取最后一条 usage 事件），少一个需要手工清空的 state
 *   （send / 切会话都要记得置 null，漏一处就会把上一次运行的账本带到这一次）。
 * - 用量历史不再由 7 个调用点手工 push。它们全部紧跟在 setRun 之后，
 *   所以改为「run 变化即记录」，同 run_id 覆盖、保留首次时间。
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type { AgentEvent, AgentRun, AgentTokenUsage } from "../../../types/agent";

/** 一次运行结束（或过程中）的用量快照，用于「每次对话的调用次数与调用量变化」。 */
export interface RunUsageSnapshot {
  run_id: string;
  at: number;
  status: string;
  llm_calls: number;
  total_tokens: number;
  tool_calls: number;
  elapsed_seconds: number;
}

function snapshotOf(source: AgentRun): Omit<RunUsageSnapshot, "at"> {
  return {
    run_id: source.id,
    status: source.status,
    llm_calls: source.token_usage?.llm_calls ?? 0,
    total_tokens: source.token_usage?.actual?.total_tokens ?? 0,
    tool_calls: source.tool_call_count ?? source.tool_calls?.length ?? 0,
    elapsed_seconds: source.elapsed_seconds ?? 0,
  };
}

export function useAgentUsage(events: AgentEvent[], run: AgentRun | null, busy: boolean) {
  const [history, setHistory] = useState<RunUsageSnapshot[]>([]);

  /** 运行中后端持续下发 `usage` 事件；结束后 authoritative 值是 run.token_usage。 */
  const liveUsage = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const ev = events[i];
      if (ev.type === "usage" && ev.payload?.token_usage) {
        return ev.payload.token_usage as AgentTokenUsage;
      }
    }
    return null;
  }, [events]);

  useEffect(() => {
    if (!run) return;
    const snap = snapshotOf(run);
    setHistory((prev) => {
      const idx = prev.findIndex((item) => item.run_id === snap.run_id);
      if (idx === -1) return [...prev, { ...snap, at: Date.now() / 1000 }];
      // 轮询每 2s 都会拿到新的 run 对象：同一次运行只更新时间，避免「较上次」抖动。
      const next = [...prev];
      next[idx] = { ...snap, at: prev[idx].at };
      return next;
    });
  }, [run]);

  const resetHistory = useCallback(() => setHistory([]), []);

  return {
    /** 面板展示用的账本：运行中优先实时值，否则用 run 上的最终值。 */
    usage: liveUsage ?? run?.token_usage ?? null,
    live: busy && liveUsage !== null,
    history,
    resetHistory,
  };
}
