/**
 * 通用分阶段任务进度（`lib/taskProgress.ts` + 各链路的阶段描述）单元测试。
 *
 * 为什么值得测：这段逻辑里埋的全是**只在界面上看得见**的边界条件 ——
 *   1. 阶段引用有下标 / id / 名称三种写法，调用方各写各的；
 *   2. 顺序任务中间丢一条事件，进度条**不允许倒退**（历史 bug：「进度倒着走」）；
 *   3. 同步接口拿不到进度时不能乱猜当前阶段，否则点亮的格子是假信息；
 *   4. 「工具完成：xxx」含「完成」二字，不能被误判成「已到收尾阶段」。
 *
 * 本文件不渲染组件（Node 的类型剥离不支持 JSX），只钉住纯模型层 +
 * 三个调用方注入的阶段描述 —— 组件层退化成纯渲染，出错面很小。
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  clampPercent,
  normalizeStages,
  resolveStageIndex,
  resolveTaskProgress,
} from "../src/lib/taskProgress.ts";
import { AGENT_TASK_STAGES, agentStageIndex } from "../src/lib/agentEvents.ts";
import { workflowStages } from "../src/features/workflow/nodeStatus.ts";
import { ML_TRAIN_ESTIMATE } from "../src/api/ml.ts";
import { EXPERIMENT_RUN_ESTIMATE, EXPERIMENT_RUN_STAGES } from "../src/api/experiments.ts";
import { WORKFLOW_RUN_ESTIMATE } from "../src/api/workflow.ts";

/** 需求里给的例子：ML 训练的四个阶段。 */
const ML_STAGES = ["数据加载", "特征工程", "模型训练", "评估"];

// ---------------------------------------------------------------------------
// 阶段归一与定位
// ---------------------------------------------------------------------------

test("stages 支持纯字符串写法：id 与 name 同名", () => {
  const stages = normalizeStages(ML_STAGES);
  assert.deepEqual(stages.map((s) => s.id), ML_STAGES);
  assert.deepEqual(stages.map((s) => s.name), ML_STAGES);
});

test("resolveStageIndex 接受下标 / id / 名称三种写法，解析不出返回 -1", () => {
  const stages = normalizeStages([
    { id: "load", name: "数据加载" },
    { id: "train", name: "模型训练" },
  ]);
  assert.equal(resolveStageIndex(stages, 0), 0);
  assert.equal(resolveStageIndex(stages, "train"), 1);
  assert.equal(resolveStageIndex(stages, "数据加载"), 0);
  // 越界下标与未知名称都必须给出 -1（不是 0）——否则会点亮错误的阶段
  assert.equal(resolveStageIndex(stages, 9), -1);
  assert.equal(resolveStageIndex(stages, -1), -1);
  assert.equal(resolveStageIndex(stages, "nope"), -1);
  assert.equal(resolveStageIndex(stages, null), -1);
});

test("clampPercent 收敛到 0-100 的整数，NaN 归 0", () => {
  assert.equal(clampPercent(150), 100);
  assert.equal(clampPercent(-5), 0);
  assert.equal(clampPercent(66.6), 67);
  assert.equal(clampPercent(Number.NaN), 0);
});

// ---------------------------------------------------------------------------
// 有真实进度：按阶段推进
// ---------------------------------------------------------------------------

test("按 SSE 事件推进：显式 progress 优先，当前阶段高亮", () => {
  // 后端 done 的语义是「已完成阶段数（含当前）」⇒ 当前阶段下标 = done
  const model = resolveTaskProgress({
    stages: ML_STAGES,
    currentStage: 2,
    doneStages: ["数据加载", "特征工程"],
    progress: 50,
  });
  assert.equal(model.activeIndex, 2);
  assert.deepEqual(model.doneIndexes, [0, 1]);
  assert.equal(model.percent, 50);
  // 有真实进度 ⇒ 不该再显示「预计需要 30 秒」那种估算文案
  assert.equal(model.estimated, false);
});

test("没有显式 progress 时按已完成阶段数推算，当前阶段之前自动补完成", () => {
  const model = resolveTaskProgress({ stages: ML_STAGES, currentStage: "评估" });
  assert.equal(model.activeIndex, 3);
  assert.deepEqual(model.doneIndexes, [0, 1, 2]);
  assert.equal(model.percent, 75);
  assert.equal(model.estimated, true);
});

test("当前阶段往后走时进度单调不减（中间丢事件也不倒退）", () => {
  const percents = ["数据加载", "特征工程", "模型训练", "评估"].map(
    (stage) => resolveTaskProgress({ stages: ML_STAGES, currentStage: stage }).percent,
  );
  // 0% → 25% → 50% → 75%，严格不减
  assert.deepEqual(percents, [0, 25, 50, 75]);
  for (let i = 1; i < percents.length; i += 1) assert.ok(percents[i] >= percents[i - 1]);
});

test("阶段全跑完但没有当前阶段可定位时，activeIndex = -1（不高亮任何一格）", () => {
  const model = resolveTaskProgress({
    stages: ML_STAGES,
    currentStage: ML_STAGES.length,
    doneStages: ML_STAGES,
    progress: 100,
  });
  assert.equal(model.activeIndex, -1);
  assert.equal(model.doneIndexes.length, ML_STAGES.length);
});

// ---------------------------------------------------------------------------
// 没有真实进度：不确定进度
// ---------------------------------------------------------------------------

test("不确定进度：不猜当前阶段，也不谎报百分比", () => {
  const model = resolveTaskProgress({ stages: EXPERIMENT_RUN_STAGES, indeterminate: true });
  assert.equal(model.activeIndex, -1, "不知道跑到哪一步，就不能点亮任何阶段");
  assert.deepEqual(model.doneIndexes, []);
  assert.equal(model.percent, 0);
  assert.equal(model.estimated, true, "百分比是推算的 ⇒ 允许显示「预计需要 30 秒」");
});

test("不确定进度下同样不推断当前阶段，但阶段清单照常输出", () => {
  const model = resolveTaskProgress({
    stages: ["加载数据", "训练模型"],
    currentStage: null,
    indeterminate: true,
  });
  assert.equal(model.stages.length, 2);
  assert.equal(model.activeIndex, -1);
});

// ---------------------------------------------------------------------------
// 终态
// ---------------------------------------------------------------------------

test("成功：全部阶段标为完成，进度 100", () => {
  const model = resolveTaskProgress({ stages: ML_STAGES, progress: 42, status: "success" });
  assert.equal(model.percent, 100);
  assert.deepEqual(model.doneIndexes, [0, 1, 2, 3]);
  assert.equal(model.activeIndex, -1);
  assert.equal(model.estimated, false);
});

test("失败：保留已到达的进度，不谎报 100；没给当前阶段则不高亮", () => {
  const model = resolveTaskProgress({ stages: ML_STAGES, progress: 50, status: "error" });
  assert.equal(model.percent, 50);
  assert.equal(model.activeIndex, -1);
  // 调用方（ML 页）会把最后一次的 currentStage 继续传进来，于是失败点仍然高亮
  const withStage = resolveTaskProgress({
    stages: ML_STAGES,
    currentStage: 2,
    progress: 50,
    status: "error",
  });
  assert.equal(withStage.activeIndex, 2);
});

// ---------------------------------------------------------------------------
// 各链路注入的阶段描述
// ---------------------------------------------------------------------------

test("Agent 阶段归并：4 个粗阶段，「工具完成」不得被误判成收尾阶段", () => {
  assert.equal(AGENT_TASK_STAGES.length, 4);
  assert.equal(agentStageIndex("理解任务"), 0);
  assert.equal(agentStageIndex("执行计划"), 1);
  assert.equal(agentStageIndex("执行工具：训练模型"), 2);
  // ★ 这是本映射最容易写错的一条：「工具完成：xxx」也含「完成」。
  assert.equal(agentStageIndex("工具完成：训练模型"), 2);
  assert.equal(agentStageIndex("任务完成"), 3);
  assert.equal(agentStageIndex("任务失败"), 3);
  // 归不到类时停在起点，而不是跳到末尾
  assert.equal(agentStageIndex(""), 0);
  assert.equal(agentStageIndex("某种没见过的新阶段"), 0);
});

test("Agent 每个粗阶段都能被真实 stage 文案命中（不会有永远点不亮的阶段）", () => {
  const hits = ["理解任务", "执行计划", "执行工具：训练模型", "任务完成"].map(agentStageIndex);
  assert.deepEqual(hits, [0, 1, 2, 3]);
});

test("Workflow 阶段取画布节点顺序与节点名，未知类型回退成原始 type", () => {
  const stages = workflowStages([
    { id: "n1", type: "data.load" },
    { id: "n2", type: "ml.train" },
    { id: "n3", type: "not.a.type" },
  ]);
  assert.deepEqual(stages.map((s) => s.id), ["n1", "n2", "n3"]);
  assert.deepEqual(stages.map((s) => s.name), ["加载数据", "训练模型", "not.a.type"]);
  // 节点 id 缺失时也要有稳定 key，否则 React 列表会串位
  assert.deepEqual(workflowStages([{ id: "", type: "noop" }])[0].id, "node-0");
});

test("同步接口的预计耗时文案集中定义，与需求口径一致", () => {
  assert.equal(ML_TRAIN_ESTIMATE, "模型训练中，预计需要 30 秒");
  assert.ok(EXPERIMENT_RUN_STAGES.length >= 3);
  assert.match(EXPERIMENT_RUN_ESTIMATE, /30 秒/);
  assert.match(WORKFLOW_RUN_ESTIMATE, /30 秒/);
});
