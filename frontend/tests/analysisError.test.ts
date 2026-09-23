/**
 * bug5 回归：前端必须展示后端返回的具体业务错误，而不是只给 HTTP 状态码。
 *
 * 运行：node --experimental-strip-types --test tests/*.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { describeAnalysisError, detailHints } from "../src/lib/analysisError.ts";

/** 模拟 client.ts 拦截器包装后的 Error（带 code/details/status）。 */
function backendError(
  message: string,
  status: number,
  extra: { code?: string; details?: unknown } = {},
): Error & { code?: string; details?: unknown; status?: number } {
  const err = new Error(message) as Error & {
    code?: string;
    details?: unknown;
    status?: number;
  };
  err.status = status;
  err.code = extra.code ?? "VALIDATION_ERROR";
  err.details = extra.details;
  return err;
}

test("展示后端业务 message，而不是 HTTP 状态码", () => {
  const err = backendError(
    "热力图：以下列虽为数值但取值种类过少（按分类编码处理），不参与相关性：Month、DayofMonth；当前可用数值列 0 个（需要至少 2 个）",
    422,
    {
      details: {
        rejected_categorical_numeric: ["Month", "DayofMonth"],
        available_numeric_columns: ["DepDelay", "Distance"],
        hint: "请选择 2 个以上取值连续的数值字段（如金额、时长、距离）。",
      },
    },
  );
  const text = describeAnalysisError(err);
  assert.ok(text.includes("Month"), `应指出具体列：${text}`);
  assert.ok(text.includes("取值种类过少"), `应说明原因：${text}`);
  assert.ok(!text.includes("HTTP 422"), `不得只给状态码：${text}`);
});

test("剥离「请求失败（HTTP 422）」外壳，取真实文案", () => {
  const err = backendError("请求失败（HTTP 422）", 422, {
    details: { missing: ["DepDelay", "Month"] },
  });
  const text = describeAnalysisError(err);
  assert.ok(!text.includes("HTTP 422"), `状态码外壳应被剥掉：${text}`);
  assert.ok(text.includes("DepDelay"), `应回退到 details 线索：${text}`);
  assert.ok(text.includes("请求参数无法处理"), `应给出可读的状态说明：${text}`);
});

test("缺列场景列出 missing 字段", () => {
  const err = backendError("指定的字段不存在：DepDelay、Month、DayofMonth", 422, {
    details: {
      missing: ["DepDelay", "Month", "DayofMonth"],
      available: ["ArrDelay", "CRSArrTime"],
      hint: "请从可用字段中重新选择。",
    },
  });
  const text = describeAnalysisError(err);
  assert.ok(text.includes("DepDelay"));
  assert.ok(text.includes("缺失字段"), text);
  assert.ok(text.includes("ArrDelay"), `应给出可用字段：${text}`);
});

test("x 与 y 同列给出明确原因", () => {
  const err = backendError("y 轴字段（y）与x 轴字段（x）不能是同一列：DepDelay", 422, {
    details: { column: "DepDelay", fields: ["x", "y"] },
  });
  const text = describeAnalysisError(err);
  assert.ok(text.includes("同一列"), text);
  assert.ok(!text.includes("HTTP 500"), text);
});

test("500 也要说人话，不暴露内部实现", () => {
  const err = backendError("Internal Server Error", 500);
  const text = describeAnalysisError(err);
  assert.ok(text.includes("服务端内部错误"), text);
  assert.ok(text.includes("稍后重试"), `应给出下一步：${text}`);
});

test("无详情时退回状态码文案，不抛异常", () => {
  assert.ok(describeAnalysisError(backendError("请求失败（HTTP 500）", 500)).length > 0);
  assert.ok(describeAnalysisError(new Error("网络错误")).includes("网络错误"));
  assert.ok(describeAnalysisError("纯字符串错误").includes("纯字符串错误"));
  assert.ok(describeAnalysisError(null).length > 0);
  assert.ok(describeAnalysisError(undefined).length > 0);
  assert.ok(describeAnalysisError({}).length > 0);
});

test("detailHints 容错各种畸形 details", () => {
  assert.deepEqual(detailHints(null), []);
  assert.deepEqual(detailHints(undefined), []);
  assert.deepEqual(detailHints("字符串"), []);
  assert.deepEqual(detailHints(42), []);
  assert.deepEqual(detailHints({}), []);
  // 非字符串元素被忽略，不产生 "undefined" 之类的脏文案
  assert.deepEqual(detailHints({ missing: [1, null, "Month"] }), ["缺失字段：Month"]);
  assert.deepEqual(detailHints({ missing: "不是数组" }), []);
});

test("可用字段过多时折叠，避免提示过长", () => {
  const many = Array.from({ length: 20 }, (_, i) => `col_${i}`);
  const hints = detailHints({ available: many });
  assert.equal(hints.length, 1);
  assert.ok(hints[0].includes("前 8 个"), hints[0]);
  assert.ok(!hints[0].includes("col_19"), "超出部分应折叠");
});

test("不重复拼接同一条线索", () => {
  const hint = "请选择 2 个以上取值连续的数值字段";
  const text = describeAnalysisError(
    backendError("相关性分析至少需要 2 个数值字段", 422, { details: { hint } }),
  );
  assert.equal(text.split(hint).length - 1, 1, `线索不应重复：${text}`);
});
