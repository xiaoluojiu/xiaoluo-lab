/**
 * 浏览器本地状态清单：设置页「清理浏览器数据」的唯一事实源。
 *
 * 历史问题：
 * 旧实现是 `Object.keys(localStorage).filter(k => k.startsWith("xllab.cache"))`，
 * 但全仓库从来没有写过 `xllab.cache*` 前缀——用户点确认框后实际什么都没删，
 * 是「看起来存在但完全无效」的功能。而真正该被清理的 `xllab.recent.datasets`
 * （命令面板「最近数据集」）反而不在范围内。
 *
 * 现在：
 * - 在唯一的清单处声明「有哪些本地状态 / 是否可清理 / 清理后有什么后果」，
 *   设置页只按分组勾选，不再靠猜前缀。
 * - 含敏感信息（API Key）的 `xllab.llm` 单独归到「凭据」分组，默认不勾选。
 *
 * 新增本地存储项时，同步在本清单登记，否则设置页无法展示也无法清理。
 */

export type LocalStateId = "recent" | "ui" | "permissions" | "nav" | "llm";

export interface LocalStateEntry {
  id: LocalStateId;
  key: string;
  /** 面向用户的名称。 */
  label: string;
  /** 清理后会发生什么，用于确认框逐条列出。 */
  consequence: string;
  /** 是否含敏感信息（凭据），UI 上单独提示。 */
  sensitive?: boolean;
  /** 默认是否勾选。 */
  defaultChecked: boolean;
}

/** 本地状态清单（key 与各 lib 模块中导出的 STORAGE_KEY 保持一致）。 */
export const LOCAL_STATE_ENTRIES: LocalStateEntry[] = [
  {
    id: "recent",
    key: "xllab.recent.datasets",
    label: "最近打开的数据集",
    consequence: "命令面板（Ctrl/Cmd + K）的「最近数据集」快捷入口会被清空",
    defaultChecked: true,
  },
  {
    id: "ui",
    key: "xllab.ui",
    label: "外观设置",
    consequence: "主题、字号、界面密度与动态背景会恢复为默认值",
    defaultChecked: true,
  },
  {
    id: "permissions",
    key: "xllab.tool.permissions",
    label: "工具自动放行开关",
    consequence: "全部工具回到「跟随后端风险声明」的默认策略",
    defaultChecked: true,
  },
  {
    id: "nav",
    key: "xllab.nav.collapsed",
    label: "侧边栏折叠状态",
    consequence: "侧边栏会展开为默认宽度",
    defaultChecked: false,
  },
  {
    id: "llm",
    key: "xllab.llm",
    label: "AI 连接配置（含 API Key）",
    consequence: "Base URL、模型名与 API Key 会被清除，需要重新填写",
    sensitive: true,
    defaultChecked: false,
  },
];

/** 某个条目当前是否真的存在（用于在确认框里只列出实际会被删的项）。 */
export function isStatePresent(entry: LocalStateEntry): boolean {
  try {
    return sessionStorage.getItem(entry.key) !== null || localStorage.getItem(entry.key) !== null;
  } catch {
    return false;
  }
}

/** 当前实际存在的本地状态（按清单顺序）。 */
export function presentStateEntries(): LocalStateEntry[] {
  return LOCAL_STATE_ENTRIES.filter(isStatePresent);
}

/**
 * 清理指定条目。两个 Storage 都删——浏览器配置历史上曾在 localStorage，
 * 只清一处会留下「幽灵配置」，下次读取又冒出来。
 * @returns 实际被删除的条目数
 */
export function clearLocalState(ids: LocalStateId[]): number {
  const targets = LOCAL_STATE_ENTRIES.filter((entry) => ids.includes(entry.id));
  let removed = 0;
  for (const entry of targets) {
    try {
      if (localStorage.getItem(entry.key) !== null) removed += 1;
      localStorage.removeItem(entry.key);
    } catch {
      /* 隐私模式下 Storage 不可用，跳过 */
    }
    try {
      if (sessionStorage.getItem(entry.key) !== null) removed += 1;
      sessionStorage.removeItem(entry.key);
    } catch {
      /* 同上 */
    }
  }
  return removed;
}
