/**
 * 回归：前端不得把「低基数编码列」（Month / DayofMonth / DayOfWeek 等）
 * 当作可参与相关性的数值列提交，否则后端会稳定返回 422。
 *
 * 运行：node --experimental-strip-types --test tests/*.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  chartFieldAdvice,
  isContinuousNumeric,
  splitHeatmapColumns,
} from "../src/lib/edaColumns.ts";

function col(column: string, dtype: string, extra: Record<string, unknown> = {}) {
  return { column, dtype, ...extra } as never;
}

// 与日志里的真实数据集（dataset 9）同构：
// Month/DayofMonth/DayOfWeek 是低基数整数编码列；DepDelay/Distance 是连续量。
// CRSDepTime 取值数（1400）相对 2000 行不够低，按后端判据（n*2 < rows）仍算连续量，
// 所以它出现在 usable 里是正确的 —— 这里刻意保留以钉住「行数相关」的边界行为。
const FLIGHT = [
  col("Month", "Int32", { distinct: 12 }),
  col("DayofMonth", "Int32", { distinct: 31 }),
  col("DayOfWeek", "Int32", { distinct: 7 }),
  col("CRSDepTime", "Int32", { distinct: 1400 }),
  col("DepDelay", "Float64", { distinct: 180 }),
  col("Distance", "Float64", { distinct: 1500 }),
];

test("低基数整数编码列不算「可用数值列」", () => {
  const byName = Object.fromEntries(FLIGHT.map((c) => [c.column, c]));
  assert.equal(isContinuousNumeric(byName["Month"], 2000), false);
  assert.equal(isContinuousNumeric(byName["DayOfWeek"], 2000), false);
  assert.equal(isContinuousNumeric(byName["DepDelay"], 2000), true);
});

test("非数值类型永远不是可用数值列", () => {
  assert.equal(isContinuousNumeric(col("UniqueCarrier", "Utf8"), 2000), false);
});

test("基数未知时保守放行，把权威判定留给后端", () => {
  assert.equal(isContinuousNumeric(col("Mystery", "Float64"), 2000), true);
});

test("复现日志：勾了 DepDelay + Month，热力图不得把 Month 一起提交", () => {
  const split = splitHeatmapColumns(["DepDelay", "Month"], FLIGHT, 2000);
  assert.deepEqual(split.usable, ["DepDelay"]);
  assert.deepEqual(split.rejected, ["Month"]);
  // 关键断言：以前提交的是 ['DepDelay','Month']，正是 422 的来源。
  assert.ok(!split.usable.includes("Month"), "Month 不得进入热力图数值列");
});

test("勾了 4 个列的场景同样只保留真正可算的列", () => {
  const split = splitHeatmapColumns(
    ["DepDelay", "Month", "DayofMonth", "DayOfWeek"],
    FLIGHT,
    2000,
  );
  assert.deepEqual(split.usable, ["DepDelay"]);
  assert.deepEqual(split.rejected, ["Month", "DayofMonth", "DayOfWeek"]);
});

test("未勾选时用数据集全部连续数值列", () => {
  const split = splitHeatmapColumns([], FLIGHT, 2000);
  assert.deepEqual(split.usable, ["CRSDepTime", "DepDelay", "Distance"]);
  assert.deepEqual(split.rejected, ["Month", "DayofMonth", "DayOfWeek"]);
});

test("可用数值列不足 2 个时，建议里说明原因并指向替代图表", () => {
  const thin = [col("Month", "Int32", { distinct: 12 }), col("DepDelay", "Float64", { distinct: 180 })];
  const advice = chartFieldAdvice("heatmap", thin, 2000);
  assert.ok(advice.message.includes("取值连续"), advice.message);
  assert.ok(advice.message.includes("Month"), `应点名问题列：${advice.message}`);
  assert.ok(advice.message.includes("柱状图"), `应给出替代方案：${advice.message}`);
});

test("可用数值列充足时，建议直接给出字段名", () => {
  const advice = chartFieldAdvice("heatmap", FLIGHT, 2000);
  assert.ok(advice.message.includes("DepDelay"), advice.message);
  assert.deepEqual(advice.suggested, ["CRSDepTime", "DepDelay"]);
});

test("折线图建议时间字段作横轴", () => {
  const withTime = [col("FlightDate", "Date"), ...FLIGHT];
  const advice = chartFieldAdvice("line", withTime, 2000);
  assert.ok(advice.message.includes("FlightDate"), advice.message);
  assert.ok(advice.suggested.includes("CRSDepTime"), advice.message);
});

test("柱状图没有分类字段时明说不可用", () => {
  const advice = chartFieldAdvice("bar", FLIGHT, 2000);
  assert.ok(advice.message.includes("没有分类字段"), advice.message);
});

test("分组柱状图字段不足时给出可读说明", () => {
  const advice = chartFieldAdvice("grouped_bar", FLIGHT, 2000);
  assert.ok(advice.message.length > 0);
});

test("单字段图表指向一个具体数值列", () => {
  const advice = chartFieldAdvice("histogram", FLIGHT, 2000);
  assert.deepEqual(advice.suggested, ["CRSDepTime"]);
});
