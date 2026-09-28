/**
 * 回归：数据处理「影响评估」的口径。
 *
 * 用户诉求是「执行前先看到会删掉多少行、多少列，删列 / 删大量行必须二次确认」，
 * 这里钉住实现该诉求的三件事：
 *   1. 影响量的方向换算（delta -> removed/added）；
 *   2. 风险分级与「是否强制二次确认」；
 *   3. 展示文案（含千分位与百分比只舍入一次）。
 *
 * 运行：node --experimental-strip-types --test tests/*.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  HIGH_RISK_REMOVED_ROWS,
  HIGH_RISK_REMOVED_ROW_RATIO,
  assessImpact,
  assessOperationRisk,
  formatInt,
  formatPercent,
  impactSummary,
  shapeTransition,
} from "../src/lib/processingRisk.ts";

const shape = (rows: number, columns: number) => ({ rows, columns });

// ---------- 影响量换算 ----------

test("assessImpact：筛选删行时方向为「删除」", () => {
  const impact = assessImpact(shape(10000, 10), shape(3000, 10));
  assert.equal(impact.removedRows, 7000);
  assert.equal(impact.addedRows, 0);
  assert.equal(impact.removedColumns, 0);
  assert.equal(impact.addedColumns, 0);
  assert.equal(impact.emptiesTable, false);
  assert.equal(impact.removedRowRatio, 0.7);
  assert.equal(impact.sameShape, false);
});

test("assessImpact：逆透视 / 透视为「列减少 + 行增加」", () => {
  const impact = assessImpact(shape(1000, 21), shape(8000, 3));
  assert.equal(impact.removedColumns, 18);
  assert.equal(impact.addedColumns, 0);
  assert.equal(impact.addedRows, 7000);
  assert.equal(impact.removedRows, 0);
});

test("assessImpact：输入为空表时比例记 0，不产生 NaN", () => {
  const impact = assessImpact(shape(0, 0), shape(0, 0));
  assert.equal(impact.removedRowRatio, 0);
  assert.equal(impact.emptiesTable, false);
  assert.equal(impact.sameShape, true);
});

test("assessImpact：形状不变才算 sameShape（缺失值填充 / 类型转换）", () => {
  const impact = assessImpact(shape(1000, 10), shape(1000, 10));
  assert.equal(impact.sameShape, true);
  assert.equal(impact.removedRows, 0);
});

test("assessImpact：emptiesTable 只在原表非空、结果为空时成立", () => {
  assert.equal(assessImpact(shape(500, 5), shape(0, 5)).emptiesTable, true);
  assert.equal(assessImpact(shape(0, 5), shape(0, 5)).emptiesTable, false);
});

// ---------- 风险分级 ----------

test("少量删行（未达任一阈值）只算中风险，不需要二次确认", () => {
  // 100 / 1000 = 10%，且远小于 1000 行。
  const risk = assessOperationRisk(shape(1000, 5), shape(900, 5));
  assert.equal(risk.level, "medium");
  assert.equal(risk.requiresSecondConfirmation, false);
  assert.match(risk.reasons.join(" "), /删除 100 行/);
});

test("删除占比达到 30% 即判高风险（比例阈值边界）", () => {
  const risk = assessOperationRisk(shape(1000, 5), shape(700, 5));
  assert.equal(risk.level, "high");
  assert.equal(risk.requiresSecondConfirmation, true);
  assert.ok(HIGH_RISK_REMOVED_ROW_RATIO === 0.3);
});

test("占比未到阈值但行数达到 1000 也判高风险（行数阈值边界）", () => {
  const before = shape(HIGH_RISK_REMOVED_ROWS * 100, 5);
  const after = shape(before.rows - HIGH_RISK_REMOVED_ROWS, 5);
  const risk = assessOperationRisk(before, after);
  assert.equal(risk.level, "high");
  assert.equal(risk.requiresSecondConfirmation, true);
});

test("999 行 / 10 万行：两个阈值都差一点，保持中风险", () => {
  const risk = assessOperationRisk(shape(100000, 5), shape(99901, 5));
  assert.equal(risk.level, "medium");
  assert.equal(risk.requiresSecondConfirmation, false);
});

test("删除列必判高风险并要求二次确认", () => {
  const risk = assessOperationRisk(shape(1000, 21), shape(1000, 2));
  assert.equal(risk.level, "high");
  assert.equal(risk.requiresSecondConfirmation, true);
  assert.match(risk.reasons.join(" "), /删除列：19 列/);
});

test("结果变空表判高风险，且理由说明「原有 N 行会被全部删除」", () => {
  const risk = assessOperationRisk(shape(3035, 10), shape(0, 10));
  assert.equal(risk.level, "high");
  assert.equal(risk.requiresSecondConfirmation, true);
  assert.match(risk.reasons.join(" "), /结果将是空表：原有 3,035 行会被全部删除/);
});

test("只新增行 / 列不算高风险（不丢数据）", () => {
  const widened = assessOperationRisk(shape(1000, 5), shape(1000, 40));
  assert.equal(widened.level, "medium");
  assert.equal(widened.requiresSecondConfirmation, false);

  const lengthened = assessOperationRisk(shape(10, 5), shape(5000, 5));
  assert.equal(lengthened.level, "medium");
  assert.equal(lengthened.requiresSecondConfirmation, false);
});

test("形状不变判低风险，且不声称「数据未变」", () => {
  const risk = assessOperationRisk(shape(1000, 10), shape(1000, 10));
  assert.equal(risk.level, "low");
  assert.equal(risk.requiresSecondConfirmation, false);
  assert.deepEqual(risk.reasons, ["行数与列数都不变，仅可能修改单元格内容"]);
});

test("一次操作同时删行与删列时取更严重的等级", () => {
  const risk = assessOperationRisk(shape(1000, 10), shape(980, 8));
  assert.equal(risk.level, "high");
  assert.equal(risk.requiresSecondConfirmation, true);
  // 两条理由都要给出来，用户才知道哪一项触发了二次确认。
  assert.equal(risk.reasons.length, 2);
});

// ---------- 文案 ----------

test("impactSummary 与需求口径一致：「将影响 X 行，Y 列」", () => {
  assert.equal(impactSummary(shape(10000, 10), shape(3000, 10)), "将影响 7,000 行，0 列");
  assert.equal(impactSummary(shape(1000, 21), shape(8000, 3)), "将影响 7,000 行，18 列");
});

test("shapeTransition 同时给出前后形状", () => {
  assert.equal(shapeTransition(shape(2964624, 28), shape(118, 2)), "2,964,624 行 × 28 列 → 118 行 × 2 列");
});

test("formatInt 千分位分组（含负数）", () => {
  assert.equal(formatInt(0), "0");
  assert.equal(formatInt(999), "999");
  assert.equal(formatInt(1000), "1,000");
  assert.equal(formatInt(2964624), "2,964,624");
  assert.equal(formatInt(-1500), "-1,500");
});

test("formatPercent 只舍入一次（1/3 显示 33.3%）", () => {
  assert.equal(formatPercent(1 / 3), "33.3%");
  assert.equal(formatPercent(0.3), "30.0%");
  assert.equal(formatPercent(0), "0.0%");
});
