/** Workflow 结构化参数构造器与运行错误解析的单元测试。
 *
 * 覆盖任务书改动 1–9 的验收：dataset/conditions/aggregations 三种新参数类型、
 * 语义节点的多选枚举、AGGREGATION_FUNCS 白名单、extractRunErrors 结构化解析、
 * filter-aggregate 模板的形状。数据来源全部经 `specOf` / 导出常量读取，避免另起清单漂移。
 *
 * 运行方式沿用 `npm test`（node --experimental-strip-types --import bootstrap.mjs --test）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { AGGREGATION_FUNCS, specOf } from "../src/features/workflow/nodeSpecs.ts";
import { extractRunErrors } from "../src/features/workflow/configView.ts";
import { WORKFLOW_TEMPLATES } from "../src/features/workflow/workflowTemplates.ts";

/** 取某节点某参数 spec（nodeSpecs 是单一事实源，测试不另造清单）。 */
function param(type: string, key: string) {
  const spec = specOf(type);
  assert.ok(spec, `节点 ${type} 应有 specOf 定义`);
  const p = spec.params.find((item) => item.key === key);
  assert.ok(p, `节点 ${type} 应有参数 ${key}`);
  return p;
}

/* ------------------------------------------------------------------ */
/* 改动 1：dataset 参数类型                                            */
/* ------------------------------------------------------------------ */

test("data.load 与 dataset.read 的 dataset_id 均为 dataset 类型", () => {
  assert.equal(param("data.load", "dataset_id").kind, "dataset");
  assert.equal(param("dataset.read", "dataset_id").kind, "dataset");
});

/* ------------------------------------------------------------------ */
/* 改动 2：filter / aggregate 结构化类型                               */
/* ------------------------------------------------------------------ */

test("data.filter 的 conditions 与 data.aggregate 的 aggregations 为结构化类型", () => {
  assert.equal(param("data.filter", "conditions").kind, "conditions");
  assert.equal(param("data.aggregate", "aggregations").kind, "aggregations");
});

/* ------------------------------------------------------------------ */
/* 改动 3：删除撒谎参数                                                */
/* ------------------------------------------------------------------ */

test("ml.predict 不含 target_column，ml.evaluate 不含 metrics", () => {
  assert.equal(param("ml.predict", "output_column").kind, "string");
  assert.equal(specOf("ml.predict")!.params.find((p) => p.key === "target_column"), undefined);
  assert.equal(specOf("ml.evaluate")!.params.find((p) => p.key === "metrics"), undefined);
});

/* ------------------------------------------------------------------ */
/* 改动 3：语义节点多选枚举                                            */
/* ------------------------------------------------------------------ */

test("data.quality_check / data.statistics / report.summary 的数组参数为 multi enum", () => {
  const checks = param("data.quality_check", "checks");
  assert.equal(checks.kind, "enum");
  assert.equal(checks.multi, true);
  assert.deepEqual(
    checks.options!.map((o) => o.value),
    ["row_count", "duplicate_rows", "missing_values", "unique_keys"],
  );

  const metrics = param("data.statistics", "metrics");
  assert.equal(metrics.kind, "enum");
  assert.equal(metrics.multi, true);
  assert.deepEqual(
    metrics.options!.map((o) => o.value),
    ["count", "mean", "median", "min", "max", "std", "value_counts"],
  );

  const sections = param("report.summary", "sections");
  assert.equal(sections.kind, "enum");
  assert.equal(sections.multi, true);
  assert.deepEqual(
    sections.options!.map((o) => o.value),
    ["data_quality", "key_statistics", "issues"],
  );
});

/* ------------------------------------------------------------------ */
/* 改动 1：AGGREGATION_FUNCS 白名单                                    */
/* ------------------------------------------------------------------ */

test("AGGREGATION_FUNCS 与后端 AGG_FUNCTIONS 一致，且不含 nunique", () => {
  const values = AGGREGATION_FUNCS.map((o) => o.value);
  assert.deepEqual(values, ["count", "sum", "mean", "median", "min", "max", "std"]);
  assert.ok(!values.includes("nunique"));
});

/* ------------------------------------------------------------------ */
/* 改动 6：extractRunErrors 结构化解析                                 */
/* ------------------------------------------------------------------ */

test("extractRunErrors：结构化 details.errors 逐条解析节点 id", () => {
  const items = extractRunErrors({
    details: { errors: ["节点 'train'（type='ml.train'）缺少必需参数：['target_column']"] },
  });
  assert.equal(items.length, 1);
  assert.equal(items[0].nodeId, "train");
  assert.match(items[0].message, /target_column/);
});

test("extractRunErrors：非结构化异常退化为单条 message 且 nodeId 为 undefined", () => {
  const items = extractRunErrors(new Error("别的失败原因"));
  assert.equal(items.length, 1);
  assert.equal(items[0].nodeId, undefined);
  assert.match(items[0].message, /别的失败原因/);
});

/* ------------------------------------------------------------------ */
/* 改动 9：filter-aggregate 模板形状                                   */
/* ------------------------------------------------------------------ */

test("filter-aggregate 模板：filter 带占位条件行，aggregate 带占位聚合行", () => {
  const template = WORKFLOW_TEMPLATES.find((item) => item.id === "filter-aggregate");
  assert.ok(template);

  const filter = template!.nodes.find((item) => item.type === "data.filter");
  assert.ok(filter);
  const conditions = (filter!.config.params as { conditions: Array<Record<string, unknown>> }).conditions;
  assert.equal(conditions.length, 1);
  assert.ok("column" in conditions[0] && "op" in conditions[0] && "value" in conditions[0]);

  const aggregate = template!.nodes.find((item) => item.type === "data.aggregate");
  assert.ok(aggregate);
  const aggregations = (aggregate!.config.params as { aggregations: Array<Record<string, unknown>> }).aggregations;
  assert.ok(Array.isArray(aggregations));
  assert.ok("column" in aggregations[0] && "func" in aggregations[0]);
});
