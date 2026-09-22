/**
 * 浏览器侧的 LLM 连接配置（Base URL / 模型 / API Key）。
 *
 * 历史问题：
 * - 同一份 localStorage 读写逻辑被复制了三份：
 *   api/agent.ts、pages/Settings/index.tsx、pages/Settings/AgentModelPanel.tsx。
 * - API Key 用 btoa() 编码后以 api_key_enc 存进 localStorage，这只是「编码」不是加密，
 *   任何能打开控制台的人一行就能还原。
 *
 * 现在：
 * - 读写统一收敛到本模块（唯一事实源）。
 * - 新写入的 Key 只放 sessionStorage（关闭标签页即失效），
 *   同时仍兼容读取历史 localStorage 中的配置，避免老用户配置丢失。
 * - 真正的安全方案是让后端代持 Key、前端永不接触明文；此项作为后续改造项。
 */

export interface BrowserLlmConfig {
  base_url: string;
  model: string;
  api_key: string;
}

const STORAGE_KEY = "xllab.llm";
const EMPTY: BrowserLlmConfig = { base_url: "", model: "", api_key: "" };

function decodeLegacyKey(raw: string): string {
  try {
    return decodeURIComponent(atob(raw));
  } catch {
    return raw;
  }
}

function normalize(parsed: Record<string, unknown>): BrowserLlmConfig {
  const enc = typeof parsed.api_key_enc === "string" ? parsed.api_key_enc : "";
  const plain = typeof parsed.api_key === "string" ? parsed.api_key : "";
  return {
    base_url: typeof parsed.base_url === "string" ? parsed.base_url : "",
    model: typeof parsed.model === "string" ? parsed.model : "",
    api_key: plain || (enc ? decodeLegacyKey(enc) : ""),
  };
}

function readRaw(source: Storage): BrowserLlmConfig | null {
  const raw = source.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return normalize(JSON.parse(raw) as Record<string, unknown>);
  } catch {
    return null;
  }
}

/** 读取当前配置：优先本标签页（sessionStorage），回退到历史 localStorage。 */
export function readBrowserLlm(): BrowserLlmConfig {
  const value = readRaw(sessionStorage) ?? readRaw(localStorage);
  return value ?? { ...EMPTY };
}

/** 是否已配置完整（三项齐全才算可用）。 */
export function hasBrowserLlm(): boolean {
  const value = readBrowserLlm();
  return Boolean(value.base_url && value.model && value.api_key);
}

/**
 * 连接指纹：base_url + model 的稳定短哈希（不含 API Key）。
 *
 * 用途：后端 `GET /settings/llm` 只回 `api_key_set: bool`，前端无法判断
 * 「浏览器里这份配置」与「后端进程里生效的配置」是否同一份，导致
 * ①用户以为保存即生效；②自动同步无条件覆盖后端。
 * 有了指纹后即可做一致性比较（见 llmSync.ts），不需要新增后端接口。
 */
export function llmFingerprint(value: Pick<BrowserLlmConfig, "base_url" | "model">): string {
  const raw = `${value.base_url.trim().toLowerCase()}|${value.model.trim().toLowerCase()}`;
  if (!raw.replace("|", "")) return "";
  let hash = 0x811c9dc5; // FNV-1a 32bit，够用于判等，不用于安全
  for (let i = 0; i < raw.length; i += 1) {
    hash ^= raw.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/** 浏览器配置与后端摘要是否指向同一个 base_url + model。 */
export function llmMatchesBackend(
  browser: Pick<BrowserLlmConfig, "base_url" | "model">,
  backend: { base_url: string; model: string },
): boolean {
  const a = llmFingerprint(browser);
  return Boolean(a) && a === llmFingerprint(backend);
}

/** 写入配置：只写 sessionStorage，避免 Key 长期驻留磁盘。 */
export function writeBrowserLlm(config: BrowserLlmConfig): void {
  const payload = {
    base_url: config.base_url.trim(),
    model: config.model.trim(),
    api_key: config.api_key,
  };
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    // 历史版本把 Key 写进了 localStorage，这里顺手清掉，避免明文长期留存。
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // 某些隐私模式下 Storage 不可用：静默降级，配置仅在当前会话内有效。
  }
}

export function clearBrowserLlm(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* 忽略 */
  }
}
