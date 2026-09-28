/** Workflow 节点规格与后端 runner 对齐的单元测试。
 *
 * 覆盖任务书改动 1–6 的验收：枚举集合、条件依赖、模型清单、语义节点注册、
 * 模板形状与运行前置校验。数据来源全部经 `specOf` 读取，避免另起一份清单漂移。
 *
 * 运行方式沿用 `npm test`（node --experimental-strip-types --import bootstrap.mjs --test）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import type { WorkflowNode } from "../src/types/workflow.ts";
import { runPreflight, flattenConfig } from "../src/features/workflow/configView.ts";
import { MISSING_FROM_PALETTE, specOf } from "../src/features/workflow/nodeSpecs.ts";
import { WORKFLOW_TEMPLATES } from "../src/features/workflow/workflowTemplates.ts";

function node(id: string, type: string, config: Record<string, unknown> = {}): WorkflowNode {
  return { id, type, config };
}

/** 取某节点某参数 spec（nodeSpecs 是单一事实源，测试不另造清单）。 */
function param(type: string, key: string) {
  const spec = specOf(type);
  assert.ok(spec, `节点 ${type} 应有 specOf 定义`);
  const p = spec.params.find((item) => item.key === key);
  assert.ok(p, `节点 ${type} 应有参数 ${key}`);
  return p;
}

/* ------------------------------------------------------------------ */
/* 改动 1：data.clean 策略枚举 + 条件依赖                              */
/* ------------------------------------------------------------------ */

test("data.clean 的 strategy 选项值集合与后端 STRATEGIES 一致", () => {
  const strategy = param("data.clean", "strategy");
  const values = strategy.options!.map((item) => item.value);
  assert.deepEqual(values, ["drop", "constant", "mean", "median", "mode"]);
});

test("data.clean 填充值参数仅在 strategy=constant 时显示", () => {
  const value = param("data.clean", "value");
  assert.equal(value.visibleWhen?.key, "strategy");
  assert.equal(value.visibleWhen?.equals, "constant");
  // 键名仍是 value（后端按 strategy=constant + value 处理）
  assert.equal(value.key, "value");
});

/* ------------------------------------------------------------------ */
/* 改动 2：data.string 操作枚举                                        */
/* ------------------------------------------------------------------ */

test("data.string 的 op 选项值集合与后端 STRING_OPERATIONS 一致", () => {
  const op = param("data.string", "op");
  assert.deepEqual(op.options!.map((item) => item.value), ["trim", "lower", "upper", "replace", "regex"]);
});

/* ------------------------------------------------------------------ */
/* 改动 3：data.duplicate 删除 keep=none                                */
/* ------------------------------------------------------------------ */

test("data.duplicate 的 keep 仅保留 first/last", () => {
  const keep = param("data.duplicate", "keep");
  assert.deepEqual(keep.options!.map((item) => item.value), ["first", "last"]);
});

/* ------------------------------------------------------------------ */
/* 改动 5：ml.train 模型清单（仅监督模型）                             */
/* ------------------------------------------------------------------ */

test("ML_MODELS 含直方图梯度提升与 KNN 分类，且不含 kmeans/pca", () => {
  const model = param("ml.train", "model");
  const values = model.options!.map((item) => item.value);
  assert.ok(values.includes("hist_gradient_boosting_classifier"));
  assert.ok(values.includes("knn_classifier"));
  assert.ok(!values.includes("kmeans"));
  assert.ok(!values.includes("pca"));
});

/* ------------------------------------------------------------------ */
/* 改动 6：语义节点纳入 + 报告分类 + 无遗漏                            */
/* ------------------------------------------------------------------ */

test("四个语义节点均有 specOf 定义，且 MISSING_FROM_PALETTE 为空", () => {
  for (const type of ["dataset.read", "data.quality_check", "data.statistics", "report.summary"]) {
    assert.ok(specOf(type), `${type} 应有 specOf 定义`);
  }
  assert.equal(MISSING_FROM_PALETTE.length, 0);
});

/* ------------------------------------------------------------------ */
/* 模板形状                                                             */
/* ------------------------------------------------------------------ */

test("每个模板的所有节点类型均有 specOf 定义", () => {
  for (const template of WORKFLOW_TEMPLATES) {
    for (const n of template.nodes) {
      assert.ok(specOf(n.type), `模板 ${template.id} 的节点 ${n.type} 应有 specOf 定义`);
    }
  }
});

test("模板中 data.load 节点不自带非空 dataset_id", () => {
  for (const template of WORKFLOW_TEMPLATES) {
    for (const n of template.nodes) {
      if (n.type !== "data.load") continue;
      const datasetId = flattenConfig(n.config).dataset_id;
      assert.equal(datasetId == null, true, `模板 ${template.id} 的 data.load 不应自带非空 dataset_id`);
    }
  }
});

test("filter-aggregate 模板的 aggregations 为数组且首项含 column/func", () => {
  const template = WORKFLOW_TEMPLATES.find((item) => item.id === "filter-aggregate");
  assert.ok(template);
  const agg = template!.nodes.find((item) => item.type === "data.aggregate");
  assert.ok(agg);
  const params = agg!.config.params as { group_by: unknown; aggregations: unknown };
  assert.ok(Array.isArray(params.aggregations));
  const first = (params.aggregations as Array<Record<string, unknown>>)[0];
  assert.ok("column" in first && "func" in first);
});

/* ------------------------------------------------------------------ */
/* 运行前置校验                                                         */
/* ------------------------------------------------------------------ */

test("runPreflight：缺名字 / 空节点 / 缺必填均返回非 null blocker", () => {
  assert.ok(runPreflight("   ", [node("load", "data.load", { dataset_id: 1 })]));
  assert.ok(runPreflight("我的流程", []));
  assert.ok(runPreflight("我的流程", [
    node("load", "data.load", { dataset_id: 1 }),
    node("train", "ml.train", { params: { model: "rf" } }),
  ]));
});

test("runPreflight：配置齐全（含默认值）时返回 null", () => {
  const blocker = runPreflight("我的流程", [
    node("load", "data.load", { dataset_id: 1 }),
    node("train", "ml.train", { params: { model: "rf", target_column: "y" } }),
  ]);
  assert.equal(blocker, null);
});
