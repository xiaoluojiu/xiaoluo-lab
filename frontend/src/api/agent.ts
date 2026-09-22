/** Agent API（会话 / 消息（SSE 流式）/ 运行 / 确认）。 */
import { client, unwrap } from "./client";
import { ensureLlmAppliedOnce } from "../lib/llmSync";
import { useAiLab } from "../store/aiLab";
import type { AgentCapabilities, AgentEvent, AgentRun, AgentSession } from "../types/agent";

/**
 * 浏览器保存的连接配置在每个页面生命周期最多同步一次，不产生 LLM Token。
 *
 * 同步策略统一收敛到 lib/llmSync（含「前后端是否已一致」的一致性判断与
 * 「失败也只尝试一次」的重试保护）；本函数只保留调用时机。
 */
async function ensureAgentLlmBootstrap() {
  await ensureLlmAppliedOnce();
}

export function createSession(datasetIds: number[] = [], title = "") {
  return unwrap<AgentSession>(client.post("/agent/sessions", { user_id: "anonymous", title, dataset_ids: datasetIds }));
}
export function listSessions(userId = "anonymous", includeArchived = true) {
  return unwrap<AgentSession[]>(client.get("/agent/sessions", { params: { user_id: userId, include_archived: includeArchived } }));
}
/** 归档 / 取消归档：只改变列表可见性，不删除历史。 */
export function archiveSession(sessionId: string, archived = true) {
  return unwrap<AgentSession>(client.patch(`/agent/sessions/${sessionId}`, { archived }));
}
/** 删除会话及其运行记录；会话存在活动 Turn 时后端返回 409。 */
export function deleteSession(sessionId: string) {
  return unwrap<{ id: string; deleted: boolean }>(client.delete(`/agent/sessions/${sessionId}`));
}
export function updateSessionContext(sessionId: string, datasetIds: number[]) {
  return unwrap<AgentSession>(client.patch(`/agent/sessions/${sessionId}/context`, { dataset_ids: datasetIds }));
}
export function getRun(runId: string) { return unwrap<AgentRun>(client.get(`/agent/runs/${runId}`)); }
export function confirmRun(runId: string) { return unwrap<AgentRun>(client.post(`/agent/runs/${runId}/confirm`)); }
export function denyRun(runId: string) { return unwrap<AgentRun>(client.post(`/agent/runs/${runId}/deny`)); }
export function cancelRun(runId: string) { return unwrap<AgentRun>(client.post(`/agent/runs/${runId}/cancel`)); }
export function runTraceUrl(runId: string, format: "md" | "json" = "md") {
  return `${import.meta.env.VITE_API_BASE_URL ?? ""}/api/v1/agent/runs/${runId}/trace?format=${format}`;
}

export interface AgentToolInfo {
  name: string;
  description: string;
  category: string;
  permission: string;
  risk_level: string;
  requires_confirmation: boolean;
  input_schema: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
}
export function listTools() { return unwrap<AgentToolInfo[]>(client.get("/agent/tools")); }

/**
 * GET /agent/capabilities —— Agent 能力声明（步数上限、Token 预算、策略、模型、工具数）。
 * 后端在 ai-lab-ui-optimization 阶段就提供了这个接口，此前前端从未调用，
 * 导致 UI 上完全看不到 Agent 的真实上限。这里补上。
 */
export function getCapabilities() { return unwrap<AgentCapabilities>(client.get("/agent/capabilities")); }

export interface SendMessageOptions {
  content: string;
  stream?: boolean;
  datasetIds?: number[];
  onEvent: (event: AgentEvent) => void;
  onDone?: () => void;
  /** 外部取消句柄：组件卸载 / 切会话时中断流，避免后端 tail 线程挂到 AGENT_SSE_CONFIRM_WAIT_SECONDS。 */
  signal?: AbortSignal;
}

export async function sendMessage(sessionId: string, options: SendMessageOptions): Promise<void> {
  const { content, stream = true, signal } = options;
  const datasetIds = options.datasetIds ?? useAiLab.getState().datasetIds;
  await ensureAgentLlmBootstrap();
  // 安全约束：confirmed / role 不再由前端传递——高风险确认只能走 /confirm 专用端点。
  const payload = { content, stream, dataset_ids: datasetIds };
  if (!stream) {
    const run = await unwrap<AgentRun>(client.post(`/agent/sessions/${sessionId}/messages`, payload));
    options.onEvent({ seq: 0, run_id: run.id, type: run.status === "failed" ? "failed" : "completed", payload: { final_answer: run.final_answer, mode: "agent", error: run.error, run_id: run.id }, created_at: Date.now() / 1000 });
    options.onDone?.();
    return;
  }

  const base = `${import.meta.env.VITE_API_BASE_URL ?? ""}/api/v1`;
  const resp = await fetch(`${base}/agent/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!resp.ok || !resp.body) throw new Error(`Agent 请求失败（${resp.status}）`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        sep = buffer.indexOf("\n\n");
        const event = parseSseFrame(frame);
        if (!event) continue;
        if (event.type === "done") { options.onDone?.(); return; }
        options.onEvent(event);
      }
    }
    options.onDone?.();
  } finally {
    // 主动释放读取锁：取消时网络流不会自己关闭，否则连接会保持到后端超时。
    try { reader.releaseLock(); } catch { /* 已释放 */ }
  }
}

function parseSseFrame(frame: string): AgentEvent | { type: "done" } | null {
  let type = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!type) return null;
  if (type === "done") return { type: "done" };
  try { return JSON.parse(data) as AgentEvent; } catch { return null; }
}
