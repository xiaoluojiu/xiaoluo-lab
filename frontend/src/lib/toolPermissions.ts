/**
 * 工具授权开关的唯一读写入口。
 *
 * 语义：
 * - 开关「开」= 该工具被 Agent 调用时自动放行，不再逐次弹授权确认框；
 * - 开关「关」= 每次调用前都弹出授权确认框，由用户当场决定。
 *
 * 后端注册表只有 requires_confirmation 声明，没有「按工具持久化授权」的接口，
 * 因此这里用前端覆盖层实现：显式设置优先，未设置时跟随后端的风险声明
 * （无需确认的工具默认放行，需确认的工具默认每次询问）。
 * 状态写在 localStorage，与外观设置（lib/uiSettings）保持同样的作用域口径。
 */

export const TOOL_PERMISSION_STORAGE_KEY = "xllab.tool.permissions";

export type ToolPermissionMap = Record<string, boolean>;

function read(): ToolPermissionMap {
  try {
    const parsed = JSON.parse(localStorage.getItem(TOOL_PERMISSION_STORAGE_KEY) || "{}") as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: ToolPermissionMap = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === "boolean") out[key] = value;
    }
    return out;
  } catch {
    return {};
  }
}

function write(map: ToolPermissionMap): ToolPermissionMap {
  try {
    localStorage.setItem(TOOL_PERMISSION_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // 隐私模式或存储配额不足：内存态仍可用，不阻断页面。
  }
  return map;
}

export function readToolPermissions(): ToolPermissionMap {
  return read();
}

/** 单个工具开关。返回写入后的完整映射，便于调用方直接 setState。 */
export function setToolPermission(name: string, enabled: boolean): ToolPermissionMap {
  return write({ ...read(), [name]: enabled });
}

/** 批量写入（全开 / 全关）。 */
export function setToolPermissions(patch: ToolPermissionMap): ToolPermissionMap {
  return write({ ...read(), ...patch });
}

/** 清空所有显式设置，恢复为「跟随后端声明」。 */
export function resetToolPermissions(): ToolPermissionMap {
  return write({});
}

/** 用户是否显式设置过某个工具（用于 UI 标注「已自定义」）。 */
export function hasExplicitPermission(name: string): boolean {
  return typeof read()[name] === "boolean";
}

/**
 * 生效值：显式设置优先，否则跟随后端风险声明。
 * @param requiresConfirmation 后端 AgentToolInfo.requires_confirmation，未知时传 undefined（按「需确认」处理）。
 */
export function effectiveAutoAllow(name: string, requiresConfirmation?: boolean): boolean {
  const explicit = read()[name];
  if (typeof explicit === "boolean") return explicit;
  return requiresConfirmation === false;
}

/**
 * 前端是否应当替用户自动确认这次工具调用。
 * 只对「生效值为放行」的工具返回 true；高风险工具默认 false，仍走确认弹窗。
 */
export function shouldAutoConfirm(name: string, tools: { name: string; requires_confirmation: boolean }[]): boolean {
  const found = tools.find((t) => t.name === name);
  return effectiveAutoAllow(name, found?.requires_confirmation);
}
