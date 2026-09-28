/**
 * 数据集 id → 名称 的解析缓存（面包屑用）。
 *
 * 面包屑要显示「数据集A」而不是「#7」，但 `:id` 路由和 globalStore 里
 * 只有 id，没有名字。三条约束决定了这里不能每次直接用 `getDataset`：
 *
 * 1. **面包屑是外壳组件**，挂在 MainLayout 上，每次路由变化都会重算；
 *    不缓存就会「切一次页、拉一次数据集」。
 * 2. **同一 id 可能被并发解析**（面包屑解析一次、页面自己也请求一次），
 *    in-flight 去重让第二个调用者复用同一个 Promise，不产生重复请求。
 * 3. **列表页已经拿到了全部名字**：`/datasets` 列表调 `listDatasets` 后
 *    用 `primeDatasetNames` 灌进来，从列表点进详情时面包屑**一个请求都不用发**。
 *
 * 失败不写缓存（下次仍可重试），失败时返回 null 由调用方回退成「数据集 #id」。
 */
import { getDataset } from "../api/datasets";

/** 命中过的名字：id → name。进程内有效，退出应用即清空。 */
const nameCache = new Map<number, string>();

/** 正在飞的请求：id → Promise。用于并发去重。 */
const inFlight = new Map<number, Promise<string | null>>();

/** 同步读缓存（未命中返回 null），供首帧直接出名字、避免先闪一下 id。 */
export function peekDatasetName(datasetId: number): string | null {
  return nameCache.get(datasetId) ?? null;
}

/** 写入缓存。`name` 为空串/纯空白时按「没拿到」处理，不污染缓存。 */
export function primeDatasetName(datasetId: number, name: string | null | undefined): void {
  const trimmed = name?.trim();
  if (Number.isInteger(datasetId) && datasetId > 0 && trimmed) {
    nameCache.set(datasetId, trimmed);
  }
}

/**
 * 批量预热：直接吃 `listDatasets()` 的返回，省掉后续逐条查询。
 * 传 `null`/`undefined` 安全跳过（列表可能来自本地缓存或旧结构）。
 */
export function primeDatasetNames(items: Array<{ id: number; name?: string | null }> | null | undefined): void {
  if (!Array.isArray(items)) return;
  for (const item of items) {
    if (!item) continue;
    primeDatasetName(item.id, item.name);
  }
}

/** 清掉某个 id 的缓存（删除数据集后调用，避免面包屑还显示已删除的名字）。 */
export function forgetDatasetName(datasetId: number): void {
  nameCache.delete(datasetId);
  inFlight.delete(datasetId);
}

/**
 * 取数据集名称：命中缓存立即返回；否则发一次请求（同一 id 的并发调用共享同一个 Promise）。
 * 失败返回 null —— 不抛异常，面包屑的降级是「显示 id」而不是让外壳挂掉。
 */
export function fetchDatasetName(datasetId: number): Promise<string | null> {
  if (!Number.isInteger(datasetId) || datasetId <= 0) return Promise.resolve(null);

  const cached = nameCache.get(datasetId);
  if (cached !== undefined) return Promise.resolve(cached);

  const pending = inFlight.get(datasetId);
  if (pending) return pending;

  const request = getDataset(datasetId)
    .then((dataset) => {
      primeDatasetName(datasetId, dataset?.name);
      return nameCache.get(datasetId) ?? null;
    })
    .catch(() => null)
    .finally(() => {
      inFlight.delete(datasetId);
    });

  inFlight.set(datasetId, request);
  return request;
}
