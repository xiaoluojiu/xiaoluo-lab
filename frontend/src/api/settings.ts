/** 设置与系统状态 API。 */
import { client, unwrap } from "./client";

export interface LlmTestResult { ok: boolean; message: string; model?: string; capabilities?: Record<string, unknown> }
export interface LlmModelSettings { provider_type: string; base_url: string; model: string; context_window: number | null; max_output_tokens: number | null; api_key_set: boolean; remote_enabled: boolean }
export interface AgentSettings {
  context_max_chars: number;
  context_sections: { user_request: number; dataset: number; task: number; permissions: number; tools: number; history: number };
  history_messages: number;
  dataset_cache: { enabled: boolean; max_items: number };
  llm_budget: { max_calls: number; max_input_tokens: number; max_output_tokens: number; max_total_tokens: number };
  agent_policy: {
    allow_model_fallback: boolean;
    enable_tool_retrieval: boolean;
    tool_retrieval_top_k: number;
    tool_retrieval_min_score: number;
    enable_result_compression: boolean;
    enable_plan_cache: boolean;
    plan_cache_max_items: number;
  };
}
export interface StorageSummary { app: string; version: string; data_root: string; total_files: number; total_bytes: number; groups: Record<string, { files: number; bytes: number }> }

export function testLlmConnection(body: { base_url?: string; model?: string; api_key?: string }) { return unwrap<LlmTestResult>(client.post("/settings/llm/test", body)); }
export function getLlmSettings() { return unwrap<LlmModelSettings>(client.get("/settings/llm")); }
export function updateLlmSettings(body: { provider_type: string; base_url: string; model: string; api_key?: string }) { return unwrap<LlmModelSettings>(client.put("/settings/llm", body)); }
/**
 * 启用 / 停用「远程 API 大模型」总开关。
 *
 * 只切开关，不动凭据（base_url / model / api_key 原样保留）：关闭后 Agent 不再构造远程
 * Provider，改走平台自带的规则规划器；重新开启立即生效，不需要重填 Key。
 */
export function setRemoteLlm(enabled: boolean) { return unwrap<LlmModelSettings>(client.put("/settings/llm/remote", { enabled })); }
export function getAgentSettings() { return unwrap<AgentSettings>(client.get("/settings/agent")); }
export function updateAgentSettings(body: {
  context_max_chars: number;
  context_sections: AgentSettings["context_sections"];
  history_messages: number;
  dataset_cache_enabled: boolean;
  dataset_cache_max_items: number;
  max_calls: number;
  max_input_tokens: number;
  max_output_tokens: number;
  max_total_tokens: number;
  enable_tool_retrieval: boolean;
  tool_retrieval_top_k: number;
  tool_retrieval_min_score: number;
  enable_result_compression: boolean;
  enable_plan_cache: boolean;
  plan_cache_max_items: number;
  /**
   * 远程大模型调用失败后是否允许退回平台内置规则。
   *
   * ★ 这是**真开关**：后端 `runtime._direct_chat` / `_compose_answer` /
   * `planner._fallback_allowed` 都会读它。关掉意味着「远程挂了就如实失败」，
   * 不再是界面上那个仅供展示的假指示器。
   */
  allow_model_fallback?: boolean;
}) { return unwrap<AgentSettings>(client.put("/settings/agent", body)); }
export function getStorageSummary() { return unwrap<StorageSummary>(client.get("/settings/storage")); }
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
