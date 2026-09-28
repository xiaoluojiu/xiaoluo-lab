/**
 * 跨页面数据上下文（store/globalStore.ts）的回归测试。
 *
 * 这组断言的来源是本次修复要解决的两个真实症状：
 * 1. **版本在跨页面时丢失** —— 历史上版本只存在于 Processing 页的本地 state，
 *    Dataset → EDA → ML 一路走下来版本号没人带。这里钉住「同一数据集重复声明不扰动已选版本」。
 * 2. **老链接把版本冲掉** —— 链接里只有 `?dataset=` 没有 `?version=` 时，
 *    若被理解成「显式回到最新版本」，用户刚选的 v3 会在一次跳转后消失。这里钉住这个边界。
 *
 * 运行：node --experimental-strip-types --import ./tests/bootstrap.mjs --test tests/*.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { installDom } from "./dom.mjs";

// 先备好 localStorage（store 在模块加载时就会读一次持久化数据，顺序不能颠倒）。
installDom();

// 动态导入：必须在 installDom() 之后求值。
const { useGlobalContext, GLOBAL_CONTEXT_STORAGE_KEY } = await import("../src/store/globalStore.ts");
const { LOCAL_STATE_ENTRIES } = await import("../src/lib/localState.ts");

const store = () => useGlobalContext.getState();

/** 每个用例前把上下文与持久化存储一起清空。 */
function reset(): void {
  try {
    localStorage.removeItem(GLOBAL_CONTEXT_STORAGE_KEY);
  } catch {
    /* 无 storage 环境：状态本来就没有持久化 */
  }
  store().clearContext();
}

function persisted(): Record<string, unknown> | null {
  const raw = localStorage.getItem(GLOBAL_CONTEXT_STORAGE_KEY);
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : null;
}

test("初始状态没有上下文", () => {
  reset();
  assert.equal(store().currentDatasetId, null);
  assert.equal(store().currentVersionId, null);
  assert.equal(store().currentExperimentId, null);
  assert.equal(store().currentTaskId, null);
});

test("只选数据集时版本为「最新版本」（null）", () => {
  reset();
  store().selectDataset(7);
  assert.equal(store().currentDatasetId, 7);
  assert.equal(store().currentVersionId, null);
});

test("★ 同一数据集重复声明不扰动已选版本（跨页面跳转的核心保证）", () => {
  reset();
  store().selectDataset(7);
  store().selectVersion(3);
  assert.equal(store().currentVersionId, 3);

  // 从 Dataset 页再跳到 EDA、再跳到 ML，每一页都会声明一次「当前数据集」。
  store().selectDataset(7);
  store().selectDataset(7, undefined);
  assert.equal(store().currentDatasetId, 7);
  assert.equal(store().currentVersionId, 3, "重复声明同一数据集不得把版本重置为最新");
});

test("换数据集时版本回到「最新版本」，避免用旧版本号读新数据集", () => {
  reset();
  store().selectDataset(7);
  store().selectVersion(3);
  store().selectDataset(8);
  assert.equal(store().currentDatasetId, 8);
  assert.equal(store().currentVersionId, null);
});

test("显式传 null 表示「显式回到最新版本」", () => {
  reset();
  store().selectDataset(7, 3);
  assert.equal(store().currentVersionId, 3);
  store().selectDataset(7, null);
  assert.equal(store().currentVersionId, null);
});

test("★ 老链接只带 ?dataset= 时不得冲掉已选版本", () => {
  reset();
  store().selectDataset(7);
  store().selectVersion(3);
  // 这类链接遍布 Home / VersionTimeline / Datasets 的动作卡，历史上都只带 dataset。
  store().syncFromSearch("?dataset=7");
  assert.equal(store().currentDatasetId, 7);
  assert.equal(store().currentVersionId, 3, "链接未声明 version 时应沿用已选版本");
});

test("链接同时带 ?dataset= 与 ?version= 时按链接走（可分享深链）", () => {
  reset();
  store().selectDataset(7);
  store().selectVersion(3);
  store().syncFromSearch("?dataset=9&version=2");
  assert.equal(store().currentDatasetId, 9);
  assert.equal(store().currentVersionId, 2);
});

test("链接里 dataset 换成另一个数据集时，未声明版本则回到最新版本", () => {
  reset();
  store().selectDataset(7);
  store().selectVersion(3);
  store().syncFromSearch("?dataset=9");
  assert.equal(store().currentDatasetId, 9);
  assert.equal(store().currentVersionId, null);
});

test("query 缺失时不清空上下文（刷新页面的前提）", () => {
  reset();
  store().selectDataset(7, 3);
  store().syncFromSearch("");
  store().syncFromSearch("?section=appearance");
  assert.equal(store().currentDatasetId, 7);
  assert.equal(store().currentVersionId, 3);
});

test("非法参数被忽略：dataset=abc / 0 / -1 不改变现状", () => {
  reset();
  store().selectDataset(7, 3);
  store().syncFromSearch("?dataset=abc");
  store().syncFromSearch("?dataset=0");
  store().syncFromSearch("?dataset=-1");
  assert.equal(store().currentDatasetId, 7);
  assert.equal(store().currentVersionId, 3);
});

test("非正整数的 setter 入参被当作「未提供」", () => {
  reset();
  store().selectDataset(7, 3);
  store().selectVersion(0);
  store().selectVersion(Number.NaN);
  store().selectVersion(-2);
  assert.equal(store().currentVersionId, null);

  store().selectDataset(0);
  assert.equal(store().currentDatasetId, null, "非正整数数据集 id 视为「未选择」");
});

test("实验与任务上下文字段独立可写", () => {
  reset();
  store().selectExperiment(12);
  store().selectTask("wf-42");
  assert.equal(store().currentExperimentId, 12);
  assert.equal(store().currentTaskId, "wf-42");

  store().selectTask("   ");
  assert.equal(store().currentTaskId, null, "纯空白任务 id 视为未提供");

  store().selectExperiment(null);
  assert.equal(store().currentExperimentId, null);
});

test("状态写入 localStorage（刷新后可恢复的来源）", () => {
  reset();
  store().selectDataset(7, 3);
  assert.deepEqual(persisted(), {
    currentDatasetId: 7,
    currentVersionId: 3,
    currentExperimentId: null,
    currentTaskId: null,
  });
});

test("★ 模拟刷新：新实例从 localStorage 恢复上下文", async () => {
  reset();
  store().selectDataset(7, 3);
  store().selectExperiment(12);
  store().selectTask("wf-42");

  // 换一个模块实例（等价于刷新页面后重新加载 store），
  // 它只应依赖 localStorage 里的那份快照。
  const fresh = await import("../src/store/globalStore.ts?fresh=1");
  const restored = fresh.useGlobalContext.getState();
  assert.equal(restored.currentDatasetId, 7);
  assert.equal(restored.currentVersionId, 3);
  assert.equal(restored.currentExperimentId, 12);
  assert.equal(restored.currentTaskId, "wf-42");
});

test("clearContext 清空全部字段并移除持久化键", () => {
  reset();
  store().selectDataset(7, 3);
  store().selectExperiment(12);
  store().clearContext();
  assert.equal(store().currentDatasetId, null);
  assert.equal(store().currentVersionId, null);
  assert.equal(store().currentExperimentId, null);
  assert.equal(localStorage.getItem(GLOBAL_CONTEXT_STORAGE_KEY), null);
});

test("getContext 返回与 state 一致的快照", () => {
  reset();
  store().selectDataset(7, 3);
  store().selectTask("wf-1");
  assert.deepEqual(store().getContext(), {
    currentDatasetId: 7,
    currentVersionId: 3,
    currentExperimentId: null,
    currentTaskId: "wf-1",
  });
});

test("★ 存储键已登记进本地状态清单（否则设置页清不掉）", () => {
  const entry = LOCAL_STATE_ENTRIES.find((item) => item.id === "context");
  assert.ok(entry, "localState 清单里缺少 context 条目");
  assert.equal(entry.key, GLOBAL_CONTEXT_STORAGE_KEY, "两处键名必须一致，否则清理会落到空处");
});
