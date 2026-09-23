/** configView / WorkflowCanvas 单元测试。
 *
 * 覆盖审查 #8 的修复：运行按钮前置校验（runPreflight）。
 * 历史上「运行」按钮直接发请求，缺参数要靠后端报错才知道；
 * 现在前端先闸一层，这里验证闸条件与「指出该去改哪个节点」的返回值。
 *
 * 运行（本机 npm 不可用，直接调 node 入口）：
 *   node --experimental-strip-types --test frontend/tests/configView.test.ts
 */
import assert from "node:assert/strict";
import test from "node:test";
import type { WorkflowNode } from "../src/types/workflow.ts";
import {
  collectMissingConfig,
  flattenConfig,
  missingConfigWarnings,
  runPreflight,
} from "../src/features/workflow/configView.ts";
import { applyTemplate, WORKFLOW_TEMPLATES } from "../src/features/workflow/workflowTemplates.ts";

function node(id: string, type: string, config: Record<string, unknown> = {}): WorkflowNode {
  return { id, type, config };
}

/* ------------------------------------------------------------------ */
/* flattenConfig：平铺 / 嵌套两种写法都要能读                          */
/* ------------------------------------------------------------------ */

test("flattenConfig: 平铺写法原样返回，并剔除画布坐标 __ui", () => {
  const flat = flattenConfig({ dataset_id: 3, __ui: { x: 10, y: 20 } });
  assert.deepEqual(flat, { dataset_id: 3 });
});

test("flattenConfig: 嵌套 params 会被提升到顶层，且 params 本身被剔除", () => {
  // 与后端 flatten_config 对齐：结果里不应还留着 params 这个「影子键」，
  // 否则读 config 的代码会同时看到两份相同含义的数据。
  const flat = flattenConfig({ params: { target_column: "y", model: "rf" } });
  assert.deepEqual(flat, { target_column: "y", model: "rf" });
});

test("flattenConfig: 混合写法下 params 覆盖同名顶层键", () => {
  // 后端口径是 `{**flat_wo_params, **params}`，这里保持一致。
  const flat = flattenConfig({ target_column: "old", extra: 1, params: { target_column: "new" } });
  assert.equal(flat.target_column, "new");
  assert.equal(flat.extra, 1);
  assert.equal("params" in flat, false);
});

/* ------------------------------------------------------------------ */
/* collectMissingConfig                                                */
/* ------------------------------------------------------------------ */

test("ml.train 缺 target_column 会被判定为缺失", () => {
  const missing = collectMissingConfig([node("train", "ml.train", { params: { model: "rf" } })]);
  assert.equal(missing.length, 1);
  assert.equal(missing[0].paramKey, "target_column");
  assert.equal(missing[0].nodeId, "train");
  // 提示要落在 UI 上，必须带人类可读的参数名
  assert.equal(missing[0].paramLabel, "目标列");
});

test("data.load 缺 dataset_id 会被判定为缺失", () => {
  const missing = collectMissingConfig([node("load", "data.load", {})]);
  assert.equal(missing.length, 1);
  assert.equal(missing[0].paramKey, "dataset_id");
});

test("data.clean 缺 strategy 会被判定为缺失", () => {
  const missing = collectMissingConfig([node("clean", "data.clean", {})]);
  assert.equal(missing.length, 1);
  assert.equal(missing[0].paramKey, "strategy");
  assert.equal(missing[0].paramLabel, "处理策略");
});

test("条件隐藏的参数不计入缺失：data.clean 的 value 只在 strategy=fill 时出现", () => {
  // 注意：nodeSpecs 里 data.clean 的填充值键名是 `value`（不是 `fill_value`），
  // 且它不是必填项 —— 因此无论 strategy 取什么值，都不该报它缺失。
  const drop = collectMissingConfig([node("c1", "data.clean", { params: { strategy: "drop" } })]);
  assert.deepEqual(drop, []);
  const fill = collectMissingConfig([node("c2", "data.clean", { params: { strategy: "fill" } })]);
  assert.deepEqual(fill, []);
});

test("ml.cluster 的 n_clusters 只在 kmeans 下有效，但不参与缺失判定", () => {
  const kmeans = collectMissingConfig([node("k1", "ml.cluster", { params: { model: "kmeans" } })]);
  assert.deepEqual(kmeans, []);
  const dbscan = collectMissingConfig([node("k2", "ml.cluster", { params: { model: "dbscan" } })]);
  assert.deepEqual(dbscan, []);
});

test("未建模的类型走兜底表（dataset.read → dataset_id）", () => {
  const missing = collectMissingConfig([node("read", "dataset.read", {})]);
  assert.equal(missing.length, 1);
  assert.equal(missing[0].paramKey, "dataset_id");
});

test("missingConfigWarnings 输出给体检面板的可读文案（含节点、参数名、键名）", () => {
  const warnings = missingConfigWarnings([node("t", "ml.train", { params: { model: "rf" } })]);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /节点 t/);
  assert.match(warnings[0], /训练模型/);
  assert.match(warnings[0], /目标列/);
  // 末尾要给出机器可读的 key，用户才知道该在哪一格填
  assert.match(warnings[0], /target_column/);
});

/* ------------------------------------------------------------------ */
/* runPreflight：运行按钮的前置闸                                      */
/* ------------------------------------------------------------------ */

test("runPreflight: 空名字会被闸住", () => {
  const blocker = runPreflight("   ", [node("load", "data.load", { dataset_id: 1 })]);
  assert.ok(blocker);
  assert.match(blocker.reason, /名字/);
  // 这类问题没有「该去改哪个节点」，不应瞎指路
  assert.equal(blocker.focusNodeId, undefined);
});

test("runPreflight: 空画布会被闸住", () => {
  const blocker = runPreflight("我的流程", []);
  assert.ok(blocker);
  assert.match(blocker.reason, /还没有节点/);
});

test("runPreflight: 缺必填参数会被闸住，并指出该去改哪个节点", () => {
  const blocker = runPreflight("我的流程", [
    node("load", "data.load", { dataset_id: 1 }),
    node("train", "ml.train", { params: { model: "rf" } }),
  ]);
  assert.ok(blocker);
  assert.match(blocker.reason, /target_column|目标列/);
  assert.equal(blocker.focusNodeId, "train");
});

test("runPreflight: 配置齐全时放行（返回 null）", () => {
  const blocker = runPreflight("我的流程", [
    node("load", "data.load", { dataset_id: 1 }),
    node("train", "ml.train", { params: { model: "rf", target_column: "y" } }),
  ]);
  assert.equal(blocker, null);
});

test("runPreflight: 多处缺失时给出总数提示", () => {
  // ml.train 的必填项是 model + target_column，两个空节点共 4 处缺失。
  const blocker = runPreflight("我的流程", [
    node("t1", "ml.train", {}),
    node("t2", "ml.train", {}),
  ]);
  assert.ok(blocker);
  assert.match(blocker.reason, /共 4 处/);
  // 焦点给第一个出问题的节点
  assert.equal(blocker.focusNodeId, "t1");
});

/* ------------------------------------------------------------------ */
/* 模板：应用后应可直接通过前置校验（或至少不因模板本身的参数不完整而全红） */
/* ------------------------------------------------------------------ */

test("预置模板应用后，data.load 一律带上数据集 id", () => {
  for (const template of WORKFLOW_TEMPLATES) {
    const { nodes } = applyTemplate(template, 42);
    const load = nodes.find((item) => item.type === "data.load");
    if (!load) continue;
    const missing = collectMissingConfig([load]);
    assert.deepEqual(missing, [], `模板 ${template.id} 的载入节点缺少 dataset_id`);
  }
});

test("applyTemplate: 未指定数据集时载入节点留空（交由用户填写）", () => {
  const template = WORKFLOW_TEMPLATES[0];
  const { nodes } = applyTemplate(template, null);
  const load = nodes.find((item) => item.type === "data.load");
  if (load) {
    // 模板里写的是 `null` 而不是删键，断言用宽松相等同时接受 null / undefined。
    assert.equal(flattenConfig(load.config).dataset_id == null, true);
  }
});
