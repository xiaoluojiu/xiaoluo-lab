/**
 * 跨页面数据上下文（zustand）：当前数据集 / 版本 / 实验 / 任务的**唯一事实源**。
 *
 * ## 为什么需要它
 *
 * 改动前，「我现在在看哪个数据集」这件事被三种互不相通的机制分别承载：
 * 1. URL query（`/analysis?dataset=7`）—— 由 Home、Datasets/Detail、VersionTimeline 拼链接写入，
 *    DatasetSelector 再把它读回组件本地 state；
 * 2. 各页面自己的 `useState<number[]>`（Processing / Analysis / ML / Reports / Workflow 各一份）；
 * 3. `store/aiLab.ts` 的 `datasetIds`（只服务于 AI 实验室会话）。
 *
 * 后果：一旦离开带 query 的入口（点侧栏、刷新、从命令面板跳转），上下文当场丢失；
 * 「版本」更是**从来没有跨页面存在过** —— 只有 Processing 页有一个本地的 inputVersion 输入框。
 * 于是 Dataset → EDA → ML → Experiment → Report 这条链上，每一步都要重新选一次数据集。
 *
 * ## 职责边界（刻意不合并）
 *
 * - `store/ui.ts`：外壳交互态（侧栏折叠、命令面板）—— 与数据上下文无关，保持原样。
 * - `store/aiLab.ts`：AI 实验室的**会话级**状态（sessionId / runId / 工具目录缓存）——
 *   会话里的数据集是多选且随会话走的，与「当前数据上下文」语义不同，保持原样。
 * - `lib/llmSync.ts`：配置同步流程，不是 store，保持原样。
 * - 本文件只承载**跨页面共享的数据上下文**这四个字段。
 *
 * ## 版本语义
 *
 * `currentVersionId === null` 表示「最新版本」，与 Processing 页「留空 = 最新版本」的口径一致。
 * 调用方无需先查最大版本号，`null` 直接透传给后端即可。
 */
import { create } from "zustand";

/** localStorage 键名。新增/变更时同步登记到 `lib/localState.ts` 的清单，否则设置页无法清理。 */
export const GLOBAL_CONTEXT_STORAGE_KEY = "xllab.context";

/** 跨页面共享的数据上下文字段（可作为快照整体读写）。 */
export interface GlobalContextSnapshot {
  /** 当前数据集 id；null = 未选择。 */
  currentDatasetId: number | null;
  /** 当前数据版本号；null = 最新版本。 */
  currentVersionId: number | null;
  /** 当前实验 id（ML 训练产出 / 实验中心选中的实验）；null = 未选择。 */
  currentExperimentId: number | null;
  /** 当前任务 id（Workflow / 学习中心的任务标识）；null = 未选择。 */
  currentTaskId: string | null;
}

interface GlobalContextState extends GlobalContextSnapshot {
  /**
   * 选择数据集。
   *
   * @param datasetId 数据集的**行 id**（不是版本号）。
   * @param versionId 可选的版本号：
   *   - 省略 → 同一个数据集保持已选版本不变（跨页面跳转不会把版本冲掉）；
   *     换成另一个数据集时版本重置为 null（= 最新版本），避免拿旧数据集的版本号去读新数据集。
   *   - 显式传值（含 null）→ 按传入值设置，null 表示「最新版本」。
   */
  selectDataset: (datasetId: number | null, versionId?: number | null) => void;
  /** 选择版本号；传 null = 最新版本。 */
  selectVersion: (versionId: number | null) => void;
  /** 选择实验；传 null = 清空。 */
  selectExperiment: (experimentId: number | null) => void;
  /** 选择任务；传 null = 清空。 */
  selectTask: (taskId: string | null) => void;
  /**
   * 从 URL query 同步，兼容历史上散落在链接里的 `?dataset=` / `?version=`。
   * 只在参数确实有效时覆盖，**参数缺失不清空** —— 否则「刷新页面」会把持久化状态抹掉。
   */
  syncFromSearch: (search: string) => void;
  /** 清空全部上下文（删除数据集、退出工作区等场景）。 */
  clearContext: () => void;
  /** 读取当前快照（供事件回调等非订阅场景同步取用）。 */
  getContext: () => GlobalContextSnapshot;
}

/** 只接受正整数，其余（""、NaN、0、负数、小数）一律当作「未提供」。 */
function positiveInt(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

/** 非空字符串之外的输入（空串、纯空白、非字符串）一律当作「未提供」。 */
function nonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function snapshotOf(state: GlobalContextState): GlobalContextSnapshot {
  return {
    currentDatasetId: state.currentDatasetId,
    currentVersionId: state.currentVersionId,
    currentExperimentId: state.currentExperimentId,
    currentTaskId: state.currentTaskId,
  };
}

const EMPTY_CONTEXT: GlobalContextSnapshot = {
  currentDatasetId: null,
  currentVersionId: null,
  currentExperimentId: null,
  currentTaskId: null,
};

function readPersisted(): GlobalContextSnapshot {
  try {
    const raw = localStorage.getItem(GLOBAL_CONTEXT_STORAGE_KEY);
    if (!raw) return { ...EMPTY_CONTEXT };
    const parsed = JSON.parse(raw) as Partial<GlobalContextSnapshot>;
    return {
      currentDatasetId: positiveInt(parsed.currentDatasetId),
      currentVersionId: positiveInt(parsed.currentVersionId),
      currentExperimentId: positiveInt(parsed.currentExperimentId),
      currentTaskId: nonEmptyString(parsed.currentTaskId),
    };
  } catch {
    // 隐私模式 / JSON 损坏 / 非浏览器环境：静默降级为「无上下文」，不影响页面可用性。
    return { ...EMPTY_CONTEXT };
  }
}

function writePersisted(snapshot: GlobalContextSnapshot): void {
  try {
    // 全部为空时直接移除键：避免留下一个「看起来有上下文」的空壳。
    if (
      snapshot.currentDatasetId === null &&
      snapshot.currentVersionId === null &&
      snapshot.currentExperimentId === null &&
      snapshot.currentTaskId === null
    ) {
      localStorage.removeItem(GLOBAL_CONTEXT_STORAGE_KEY);
      return;
    }
    localStorage.setItem(GLOBAL_CONTEXT_STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    /* localStorage 不可用时静默降级：状态仍在内存里，只是刷新后不恢复。 */
  }
}

export const useGlobalContext = create<GlobalContextState>((set, get) => {
  /** 统一出口：先算好快照再落 state 与 storage，保证两者不会分叉。 */
  const commit = (patch: Partial<GlobalContextSnapshot>) => {
    const next = { ...snapshotOf(get()), ...patch };
    set(next);
    writePersisted(next);
  };

  return {
    ...readPersisted(),

    selectDataset: (datasetId, versionId) => {
      const next = positiveInt(datasetId);
      const current = get().currentDatasetId;

      if (next === null) {
        // 传 null = 清空数据集上下文。版本一并清空，避免留下一个无主的版本号。
        commit({ currentDatasetId: null, currentVersionId: null });
        return;
      }
      if (next === current && versionId === undefined) return; // 幂等：重复声明同一数据集不扰动已选版本

      commit({
        currentDatasetId: next,
        // 换了数据集且没显式指定版本 → 回到「最新版本」；其余情况按传入值（null 即最新版本）。
        currentVersionId:
          next === current || versionId !== undefined ? positiveInt(versionId) : null,
      });
    },

    selectVersion: (versionId) => {
      commit({ currentVersionId: positiveInt(versionId) });
    },

    selectExperiment: (experimentId) => {
      commit({ currentExperimentId: positiveInt(experimentId) });
    },

    selectTask: (taskId) => {
      commit({ currentTaskId: nonEmptyString(taskId) });
    },

    syncFromSearch: (search) => {
      // 手写解析而不依赖 URLSearchParams：本模块要能在非浏览器环境（单测）里跑。
      const params = new Map<string, string>();
      for (const pair of search.replace(/^\?/, "").split("&")) {
        if (!pair) continue;
        const [key, value = ""] = pair.split("=");
        if (key) params.set(decodeURIComponent(key), decodeURIComponent(value));
      }
      const datasetId = params.has("dataset") ? positiveInt(params.get("dataset")) : null;
      const hasVersion = params.has("version");
      const versionId = hasVersion ? positiveInt(params.get("version")) : null;

      if (datasetId !== null) {
        // URL 显式给了数据集：以链接为准（可分享的深链优先于本地残留）。
        // 注意：链接**没写** version 时传 undefined 而不是 null —— 传 null 会被理解成
        // 「显式回到最新版本」，于是「选好 v3 → 点一个只带 dataset 的老链接」就会把版本冲掉。
        get().selectDataset(datasetId, hasVersion ? versionId : undefined);
      } else if (versionId !== null && get().currentDatasetId !== null) {
        get().selectVersion(versionId);
      }
    },

    clearContext: () => {
      commit({ ...EMPTY_CONTEXT });
    },

    getContext: () => snapshotOf(get()),
  };
});
