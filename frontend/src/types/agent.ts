/** GET /agent/capabilities 的返回结构（与 backend/core/config.py 的 summary 对齐，不含 API Key）。 */
export interface AgentCapabilities {
  agent: {
    context_max_chars: number;
    context_sections: Record<string, number>;
    history_messages: number;
    dataset_cache: { enabled: boolean; max_items: number };
    max_steps: number;
    llm_budget: {
      max_calls: number;
      max_input_tokens: number;
      max_output_tokens: number;
      max_total_tokens: number;
    };
    agent_policy: {
      allow_model_fallback: boolean;
      enable_tool_retrieval: boolean;
      tool_retrieval_top_k: number;
      tool_retrieval_min_score: number;
      enable_result_compression: boolean;
      enable_plan_cache: boolean;
      plan_cache_max_items: number;
    };
  };
  llm: {
    provider_type: string;
    base_url: string | null;
    model: string;
    context_window: number;
    max_output_tokens: number;
    api_key_set: boolean;
  };
  tools: { count: number; names: string[] };
}

/** 与 Backend Agent Runtime 模型对齐。 */

/**
 * 与后端 `RunStatus`（backend/app/agent/runtime/models.py）**逐项对齐**。
 *
 * 曾经漏掉 `waiting_clarification`：后端进入澄清态后，前端既没有对应分支，
 * 轮询又会在第一次取到该状态时立即停止 ⇒ 运行永久卡住、界面停在 8% 且没有任何提示。
 * 改这个联合类型时务必同步 `useAgentRun.ts` 的 `STATUS_LABEL` 与轮询的终止判断。
 */
export type AgentRunStatus =
  | "pending"
  | "planning"
  | "running"
  | "waiting_confirmation"
  | "waiting_clarification"
  | "completed"
  | "failed";

/** 后端下发的待澄清问题（`run.pending_clarification`）。 */
export interface ClarificationRequest {
  /** 反问编码，如 `preflight.target_unknown`；回答时原样回传给 /clarify。 */
  code?: string;
  question?: string;
  /**
   * 可选项；有值时前端渲染成选择器，`value` 才是回传给后端的机读值。
   * 后端 `ClarificationOption.to_dict()` 给的是 `value / label / note`。
   */
  options?: { value: string; label?: string; note?: string; hint?: string }[];
  /** 默认值（用户直接回车时的取值）。 */
  default?: string | null;
  outcome?: string;
  step_index?: number | null;
  tool?: string | null;
}

export interface ToolCall {
  step_index: number;
  tool: string;
  arguments: Record<string, unknown>;
  attempt: number;
  status: "pending" | "ok" | "failed" | "needs_confirmation" | "denied";
  result: { success?: boolean; data?: unknown; summary?: unknown; warnings?: unknown[]; errors?: unknown[]; metadata?: Record<string, unknown> } | null;
  error: string;
  elapsed_ms: number;
}

export interface AgentTokenUsage {
  llm_calls: number;
  actual: { input_tokens: number; output_tokens: number; total_tokens: number };
  budget?: { max_llm_calls: number; max_total_tokens: number; remaining_llm_calls: number; remaining_total_tokens: number };
  optimization: { estimated_context_saved_tokens: number; estimated_result_saved_tokens: number; estimated_saved_tokens: number; avoided_planner_calls: number; plan_cache_hits: number };
  note?: string;
}

/**
 * 「这条回答是谁给的」—— 与后端 `app/agent/answer_source.py::describe()` 对齐。
 *
 * 之所以要后端下发而不是前端猜：判定依据是**这一次运行到底有没有走通 LLM**，
 * 而不是「设置页开关是否打开」。开关开着但调用失败时，真实来源仍是平台内置规则，
 * 前端无从判断。
 */
export interface AnswerSource {
  /** 机器可读取值，如 `remote_llm_chat` / `platform_rules_chat` / `llm_error_fallback`。 */
  source: string;
  /** 短标签，直接显示用。 */
  label: string;
  /** 一句话说明，放在提示里。 */
  detail: string;
  /** 是否真的由远程语言模型生成。false = 平台内置规则产出（含调用失败降级）。 */
  by_llm: boolean;
  /** 模型名；仅在 by_llm 为真时非空。 */
  model: string;
}

export interface PermissionRequest { tool: string; arguments: Record<string, unknown>; step_index: number; reason: string; }

/** 聊天消息。定义在 types 里而不是组件里：会话历史、事件流、Hook 都要读写它。 */
export interface ChatMessage {
  role: string;
  content: string;
  /** 回答来源（后端判定后下发，前端不猜）。 */
  source?: AnswerSource | null;
}

/** 运行面板页签。事件流会主动切页签（如出现工具调用时切到「活动」），故提升到类型层。 */
export type InspectorTab = "overview" | "activity" | "chain" | "token" | "budget";

export interface AgentSession {
  id: string;
  user_id: string;
  title: string;
  dataset_ids: number[];
  history: { role: string; content: string }[];
  run_ids: string[];
  created_at: number;
  archived?: boolean;
}

export interface AgentRun {
  id: string;
  session_id: string;
  user_request: string;
  status: AgentRunStatus;
  final_answer: string;
  error: string;
  plan: { goal?: string; steps?: unknown[]; notes?: string } | null;
  tool_calls: ToolCall[];
  tool_call_count: number;
  pending_confirmation: PermissionRequest | null;
  /** 待澄清问题；与 pending_confirmation 互斥（前者问「做哪个」，后者问「做不做」）。 */
  pending_clarification?: ClarificationRequest | null;
  cancel_requested?: boolean;
  token_usage: AgentTokenUsage;
  /** 回答来源（后端下发的判定结果，前端不做推测）。 */
  answer_source?: AnswerSource | null;
  elapsed_seconds: number;
  events?: AgentEvent[];
}

/** `usage` = Token 账本的增量快照，运行过程中持续下发（不只结束后算总账）。 */
/** 与后端 `EVENT_TYPES`（runtime/models.py）对齐；`preflight` / `clarification` 是后加的两种。 */
export type AgentEventType =
  | "route"
  | "chat"
  | "planning"
  | "preflight"
  | "permission"
  | "clarification"
  | "tool_call"
  | "tool_result"
  | "validation"
  | "replanning"
  | "completed"
  | "failed"
  | "usage";

export interface AgentEvent {
  seq: number;
  run_id: string;
  type: AgentEventType;
  payload: Record<string, unknown>;
  created_at: number;
}

/**
 * 工具名 → 风险等级（**前端启发式估算**，非后端权威声明）。
 *
 * 为何仍保留：授权弹窗需要在只拿到 tool 名称时也能给出风险提示，
 * 而后端 `AgentToolInfo.risk_level` 不一定随权限事件一起到达。
 *
 * ⚠️ 口径提醒：这是按名字前缀/关键词猜的，与后端 `risk_level` 可能不一致
 * （例如 `report.generate` 会写盘、`ml.train` 会长时间占用算力，这里判「高/中」的依据仅有关键词）。
 * UI 上已标注「前端估算」，勿据此做安全决策。
 */
export function riskLevelOf(tool: string): "低" | "中" | "高" {
  const name = tool.toLowerCase();
  // 写库 / 落盘 / 训练 / 合并：不可逆或有明显副作用
  if (/^data\.write/.test(name) || /(^|[._])(train|merge|delete|drop|export|generate|publish)([._]|$)/.test(name)) return "高";
  // 其余 data.* 与 ml.* 属于可重算但耗算力的操作
  if (name.startsWith("data.") || name.startsWith("ml.") || name.startsWith("workflow.")) return "中";
  return "低";
}
