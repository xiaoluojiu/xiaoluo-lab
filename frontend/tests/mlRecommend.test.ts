/**
 * 回归：机器学习模块的「目标列推荐 / 标签泄漏警告 / 指标格式化 / 调参信号」
 * 四条启发式。它们都会被界面直接展示，判错就是误导，所以钉成测试。
 *
 * 其中目标列推荐此前在 ML 页与 FeatureSelector 里各写了一份：
 * 页面按「unique<=20 的非浮点列」挑，选择器按「高基数列」排除，
 * 在 Bank-Marketing 上会自动选中 job（12 个取值）而不是真正的标签 y。
 * 现在两处都走 modelMeta.recommendTargetColumn，本用例钉住它是唯一事实源。
 *
 * 运行：npm test（node --test，与 tests/ 下其余用例同一套骨架）
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatMetric,
  labelLeakWarnings,
  recommendTargetColumn,
} from "../src/features/ml/modelMeta.ts";
import { detectSignals } from "../src/features/ml/TuningAdvice.tsx";
import type { SchemaColumn } from "../src/types/dataset.ts";

function col(column: string, dtype: string, unique_count: number): SchemaColumn {
  return {
    column,
    dtype,
    nullable: false,
    null_count: 0,
    unique_count,
    sample_values: [],
  };
}

/** 与 Bank-Marketing 列画像同构：y 是二值标签，job 是 12 取值的类别列。 */
const BANK = [
  col("age", "Int64", 77),
  col("job", "Utf8", 12),
  col("marital", "Utf8", 3),
  col("education", "Utf8", 4),
  col("default", "Utf8", 2),
  col("balance", "Float64", 7168),
  col("duration", "Int64", 1543),
  col("y", "Utf8", 2),
];

const REGRESSION_COLS = [
  col("id", "Int64", 200),
  col("area", "Float64", 180),
  col("rooms", "Int64", 6),
  col("price", "Float64", 190),
];

test("分类任务推荐真正的二值标签 y，而不是第一个低基数列", () => {
  const rec = recommendTargetColumn(BANK, "classification");
  assert.equal(rec.column, "y");
  // y 恰好是二值 + 名字命中 ⇒ 两条信号同时命中
  assert.ok(rec.reason.includes("列名与基数都符合标签特征"), rec.reason);
  // 备选里应当出现其他像标签的列（default 也是二值）
  assert.ok(rec.alternatives.includes("default"), `备选应含 default：${rec.alternatives}`);
});

test("目标选了非推荐列时，备选里给出真正的标签供对照", () => {
  const rec = recommendTargetColumn(BANK, "classification");
  const warnings = labelLeakWarnings(BANK, "job", [], 45211);
  // y 没被选为目标、也没被排除 ⇒ 必须被点名
  assert.ok(
    warnings.some((w) => w.includes("「y」")),
    `应警告 y 疑似标签：${warnings}`,
  );
  // 目标列是 job ⇒ 推荐仍是 y，两者不同正是界面要提示的落差
  assert.notEqual(rec.column, "job");
});

test("目标就是 y 时，y 自身不再被当成泄漏列", () => {
  const warnings = labelLeakWarnings(BANK, "y", [], 45211);
  assert.ok(
    !warnings.some((w) => w.includes("「y」")),
    `目标列自身不应被告警：${warnings}`,
  );
  // 注意：marital/default 这类低基数列仍会提示 —— 那是「是否还有别的疑似标签」
  // 的提醒，不是误报。只有不存在其他低基数列时才应该完全安静（见下一条）。
});

test("除目标外没有别的低基数列时不产生告警", () => {
  const lean = [col("y", "Utf8", 2), col("balance", "Float64", 7168), col("duration", "Int64", 1543)];
  assert.deepEqual(labelLeakWarnings(lean, "y", [], 45211), []);
});

test("被排除的低基数列不再告警（已明确不当特征用）", () => {
  assert.deepEqual(labelLeakWarnings(BANK, "job", ["y", "default", "marital"], 45211), []);
});

test("回归任务选名称像目标的数值列", () => {
  const rec = recommendTargetColumn(REGRESSION_COLS, "regression");
  assert.equal(rec.column, "price");
  assert.ok(rec.reason.includes("关键词"), rec.reason);
});

test("回归任务没有名称命中时回落第一个数值列，并在理由里说明", () => {
  const rec = recommendTargetColumn([col("a", "Float64", 10), col("b", "Float64", 10)], "regression");
  assert.equal(rec.column, "a");
  assert.ok(rec.reason.includes("请确认"), rec.reason);
});

test("回归任务没有数值列时明确说无可用目标", () => {
  const rec = recommendTargetColumn([col("city", "Utf8", 3)], "regression");
  assert.equal(rec.column, null);
  assert.ok(rec.reason.includes("无可用目标"), rec.reason);
});

test("聚类任务不推荐目标列", () => {
  const rec = recommendTargetColumn(BANK, "clustering");
  assert.equal(rec.column, null);
  assert.ok(rec.reason.includes("不需要目标列"), rec.reason);
});

test("整数型指标不显示四位小数，比率型指标保留四位", () => {
  assert.equal(formatMetric("cluster_count", 1534), "1534");
  assert.equal(formatMetric("n_clusters", 3.0), "3");
  assert.equal(formatMetric("silhouette_sample_seed", 42), "42");
  assert.equal(formatMetric("accuracy", 0.334), "0.3340");
  assert.equal(formatMetric("rmse", 0.394002), "0.3940");
});

test("指标缺失或非数值时给出占位而不是 NaN", () => {
  assert.equal(formatMetric("accuracy", null), "-");
  assert.equal(formatMetric("accuracy", undefined), "-");
  assert.equal(formatMetric("accuracy", "n/a"), "n/a");
});

test("训练/测试差距大命中过拟合", () => {
  const signals = detectSignals(
    "classification",
    { accuracy: 0.8, f1: 0.79 },
    { train_metrics: { accuracy: 0.95, f1: 0.94 } } as never,
    1.2,
  );
  assert.ok(signals.has("overfit"), `应命中 overfit：${[...signals]}`);
});

test("测试集 0.9 且无训练/测试落差时不误报任何拟合类信号", () => {
  const signals = detectSignals(
    "classification",
    { accuracy: 0.9, f1: 0.89 },
    { train_metrics: { accuracy: 0.91, f1: 0.9 } } as never,
    1.2,
  );
  assert.ok(!signals.has("overfit"), `不应命中 overfit：${[...signals]}`);
  assert.ok(!signals.has("underfit"), `不应命中 underfit：${[...signals]}`);
  assert.ok(!signals.has("no_signal"), `不应命中 no_signal：${[...signals]}`);
});

test("阈值以后端 signal_rules 为准：收紧过拟合线后同一份指标不再命中", () => {
  const metrics: Record<string, number> = { accuracy: 0.8, f1: 0.79 };
  const artifacts = { train_metrics: { accuracy: 0.95, f1: 0.94 } } as never;
  // 默认 0.10：差距 0.15 → 命中
  assert.ok(detectSignals("classification", metrics, artifacts, 1.2).has("overfit"));
  // 后端把线放宽到 0.30 → 同一份指标不再算过拟合
  const loose = detectSignals("classification", metrics, artifacts, 1.2, {
    overfit: { classification_metric_gap_above: 0.3 },
  });
  assert.ok(!loose.has("overfit"), `不应命中 overfit：${[...loose]}`);
  // 收紧到 0.05 → 仍应命中，说明收紧方向同样生效（不是走死的兜底值）
  const tight = detectSignals("classification", metrics, artifacts, 1.2, {
    overfit: { classification_metric_gap_above: 0.05 },
  });
  assert.ok(tight.has("overfit"), `应命中 overfit：${[...tight]}`);
});

test("signal_rules 缺项时回落到内置默认值，不会把阈值变成 undefined", () => {
  const signals = detectSignals(
    "classification",
    { accuracy: 0.8, f1: 0.79 },
    { train_metrics: { accuracy: 0.95, f1: 0.94 } } as never,
    1.2,
    { overfit: {} },
  );
  assert.ok(signals.has("overfit"), `缺项应回落默认 0.1：${[...signals]}`);
});

test("测试集 0.9 但没有训练指标时不命中 underfit", () => {
  // underfit 的口径是「测试本身就差、且训练也没好到哪去」。
  // 缺了训练指标就没有差距可算，此时不该凭测试值单独下欠拟合结论。
  const noTrain = detectSignals("classification", { accuracy: 0.9, f1: 0.89 }, null, 1.2);
  assert.ok(!noTrain.has("underfit"), `无训练指标不应命中 underfit：${[...noTrain]}`);
  // 对照：同一份测试指标，补上一个同样高的训练指标，仍不应命中（差距小）
  const flat = detectSignals(
    "classification",
    { accuracy: 0.9, f1: 0.89 },
    { train_metrics: { accuracy: 0.91, f1: 0.9 } } as never,
    1.2,
  );
  assert.ok(!flat.has("underfit"), `差距小不应命中 underfit：${[...flat]}`);
  // 但测试集真的差（0.6）时，即使没有训练指标也要命中
  const weak = detectSignals("classification", { accuracy: 0.6, f1: 0.58 }, null, 1.2);
  assert.ok(weak.has("underfit"), `测试集 0.6 应命中 underfit：${[...weak]}`);
});

test("rules 为 null 时回落到内置默认阈值，而不是整段失效", () => {
  const clean = { accuracy: 0.9, f1: 0.89, roc_auc: 0.93 };
  // 一份什么都没触发的指标：rules=null 应得到空集
  const none = detectSignals("classification", clean, null, 1.2, null as never);
  assert.equal(none.size, 0, `应为空集：${[...none]}`);
  // 空集必须是因为阈值走的是默认值、这份指标确实干净 —— 换成过拟合指标仍要命中
  const overfit = detectSignals(
    "classification",
    { accuracy: 0.8, f1: 0.79 },
    { train_metrics: { accuracy: 0.95, f1: 0.94 } } as never,
    1.2,
    null as never,
  );
  assert.ok(overfit.has("overfit"), `rules=null 应回落默认阈值并命中：${[...overfit]}`);
});
