/**
 * SSE 事件流的容器：持有事件列表，并把每条事件翻译成界面增量。
 *
 * 翻译规则在 `lib/agentEvents.ts`（纯函数、可单测），这里只负责「入列 + 转发」。
 */
import { useCallback, useState } from "react";
import { eventEffects, type EventEffects } from "../../../lib/agentEvents";
import { toolDisplayName } from "../../../store/aiLab";
import type { AgentToolInfo } from "../../../api/agent";
import type { AgentEvent } from "../../../types/agent";

export type { EventEffects };

export function useAgentEvents(tools: AgentToolInfo[]) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const clear = useCallback(() => setEvents([]), []);
  /** 入列一条事件，并返回它对界面的增量（同步返回，调用方立刻应用）。 */
  const push = useCallback(
    (ev: AgentEvent): EventEffects => {
      setEvents((prev) => [...prev, ev]);
      return eventEffects(ev, (name) => toolDisplayName(name, tools));
    },
    [tools],
  );
  return { events, setEvents, push, clear };
}
