/**
 * 浏览器 AI 配置与后端 Agent 生效配置的同步入口（唯一事实源）。
 *
 * 历史问题（本轮修复的核心）：
 * 1. 设置页「保存配置」只调 writeBrowserLlm()，**完全不碰后端**。而后端 Agent 的
 *    Provider 由 `settings.LLM_API_KEY` 单一决定（deps.py:59，为空则返回 None
 *    = 退化为规则规划器）。于是 UI 显示「当前模型：gpt-4o-mini」，
 *    实际一次 LLM 调用都没发出去——典型的静默失效。
 * 2. 有三条路径在抢同一份配置且互不知情：
 *    ① api/agent.ts 的 ensureAgentLlmBootstrap()
 *    ② AgentModelPanel 的进页面自动同步
 *    ③ AgentModelPanel 的手动同步按钮
 *    且都没有「前后端是否已经一致」的判断，导致用旧值反复覆盖后端。
 *
 * 现在：
 * - 所有「应用到后端」的动作都收敛到 applyLlmToBackend()，内部先做一致性判断；
 * - 一致性靠 lib/browserLlm 的 llmFingerprint（base_url + model，不含 Key）比较，
 *   无需新增后端接口；
 * - api/agent.ts 的隐式同步改为直接复用本模块，避免各写一套。
 */
import { getLlmSettings, updateLlmSettings, type LlmModelSettings } from "../api/settings";
import { llmFingerprint, llmMatchesBackend, readBrowserLlm, type BrowserLlmConfig } from "./browserLlm";

export type ApplyResult =
  | { status: "applied"; model: LlmModelSettings }
  | { status: "already-synced"; model: LlmModelSettings }
  | { status: "not-configured" }
  | { status: "failed"; message: string };

/** 浏览器配置是否三项齐全。 */
export function isBrowserLlmComplete(value: BrowserLlmConfig): boolean {
  return Boolean(value.base_url.trim() && value.model.trim() && value.api_key.trim());
}

/**
 * 把浏览器里的 AI 配置应用到后端 Agent。
 *
 * @param options.force     跳过「已一致」判断，强制写一次（手动按钮用）。
 * @param options.serverState 调用方已取到的后端摘要，避免重复请求。
 *
 * 注意：**不会**因为 api_key 未填就报错——后端 updateLlmSettings 对空 api_key
 * 采取「保留原值」语义（settings.py:106），所以只改模型名也是合法操作。
 */
export async function applyLlmToBackend(
  options: { force?: boolean; serverState?: LlmModelSettings | null } = {},
): Promise<ApplyResult> {
  const browser = readBrowserLlm();
  if (!isBrowserLlmComplete(browser)) return { status: "not-configured" };

  let server = options.serverState ?? null;
  if (!server) {
    try {
      server = await getLlmSettings();
    } catch {
      server = null; // 读不到就当作需要同步，由下面的写入给出真实错误
    }
  }

  // 一致性判断：指纹相同说明后端已在用这组 base_url + model。
  // 但 api_key_set 为 false 时必须继续写（否则用户填了 Key 却同步不进去）。
  if (!options.force && server && server.api_key_set && llmMatchesBackend(browser, server)) {
    return { status: "already-synced", model: server };
  }

  try {
    const saved = await updateLlmSettings({
      provider_type: "openai_compatible",
      base_url: browser.base_url,
      model: browser.model,
      api_key: browser.api_key,
    });
    return { status: "applied", model: saved };
  } catch (error) {
    return { status: "failed", message: error instanceof Error ? error.message : "应用到 Agent 失败" };
  }
}

/** 进程内一次性隐式同步标记：api/agent.ts 在发请求前调用，避免每次对话都写配置。 */
let implicitSyncDone = false;

/**
 * 隐式同步（供 Agent 请求前调用）：**只做一次**，且失败也要置位。
 *
 * 历史问题：旧实现 `browserLlmSynced` 仅在成功时置位，一次失败后
 * 每次进入 AI Lab 都会重试写全局凭据，形成重试风暴。
 */
export async function ensureLlmAppliedOnce(): Promise<void> {
  if (implicitSyncDone) return;
  implicitSyncDone = true; // 先置位：无论成败都只尝试一次
  await applyLlmToBackend();
}

/** 与 applyLlmToBackend 配套的指纹，供 UI 展示「是否已同步」。 */
export function browserModelFingerprint(): string {
  return llmFingerprint(readBrowserLlm());
}

export function backendModelFingerprint(model: LlmModelSettings | null): string {
  return model ? llmFingerprint(model) : "";
}
