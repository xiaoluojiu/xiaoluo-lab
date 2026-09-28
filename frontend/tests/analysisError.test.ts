/**
 * bug5 回归：前端必须展示后端返回的具体业务错误，而不是只给 HTTP 状态码。
 *
 * 运行：node --experimental-strip-types --test tests/*.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { describeAnalysisError, detailHints, formatError, targetMissingValuesError, clientError, withPrefix } from "../src/lib/analysisError.ts";

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

/* =====================================================================
 * formatError：标准错误对象（what / impact / solution + 可选行动入口）
 * 覆盖需求里点名的四类：数据格式错误、目标列缺失、训练失败、API 超时。
 * ===================================================================== */

/** 所有错误都必须包含完整的三层信息，缺一层就算不合格。 */
function assertThreeLayers(err: ReturnType<typeof formatError>) {
  assert.ok(err.what && err.what.trim().length > 0, `what 不能为空：${JSON.stringify(err)}`);
  assert.ok(err.impact && err.impact.trim().length > 0, `impact 不能为空：${JSON.stringify(err)}`);
  assert.ok(err.solution && err.solution.trim().length > 0, `solution 不能为空：${JSON.stringify(err)}`);
}

test("数据格式错误：点出具体列名 + 给出重传建议与入口", () => {
  const err = formatError(
    backendError("could not parse 'abc123' as dtype 'Float64' in column 'Age'", 422, {
      code: "PARSE_ERROR",
    }),
  );
  assertThreeLayers(err);
  assert.ok(err.what.includes("Age"), `应点出具体列名：${err.what}`);
  assert.ok(err.solution.includes("重新上传") || err.solution.includes("修正"), err.solution);
  assert.equal(err.actionLink, "/datasets");
  assert.ok(err.actionLabel);
});

test("数据格式错误：拿不到列名时给通用的 CSV 排查建议", () => {
  const err = formatError("无法解析上传的文件：分隔符错误");
  assertThreeLayers(err);
  assert.ok(err.solution.includes("CSV") || err.solution.includes("编码"), err.solution);
});

test("目标列缺失值过多：命中需求指定文案并带跳转按钮", () => {
  const err = targetMissingValuesError({ target: "Price", missing: 40, total: 100 });
  assertThreeLayers(err);
  assert.ok(err.what.includes("目标列包含缺失值"), err.what);
  assert.ok(err.what.includes("Price"), err.what);
  assert.ok(err.solution.includes("建议先处理"), err.solution);
  assert.equal(err.actionLabel, "去处理缺失值");
  assert.equal(err.actionLink, "/processing");
});

test("目标列不存在：说明是哪一列并引导重选", () => {
  const err = formatError(backendError("目标列 'TotalPrice' 不存在", 422, { code: "TARGET_NOT_FOUND" }));
  assertThreeLayers(err);
  assert.ok(err.what.includes("TotalPrice"), err.what);
  assert.ok(err.what.includes("不在数据集中"), err.what);
  assert.equal(err.actionLink, "/ml");
});

test("训练失败（后端 preflight 文案）：翻译成可操作建议", () => {
  const emptyLabel = formatError("目标列全为空");
  assertThreeLayers(emptyLabel);
  assert.ok(emptyLabel.what.includes("目标列全部为空"), emptyLabel.what);
  assert.equal(emptyLabel.actionLink, "/processing");

  const noVariance = formatError("目标列无变化（所有取值相同），回归模型无法学习");
  assertThreeLayers(noVariance);
  assert.ok(noVariance.what.includes("取值都相同"), noVariance.what);
});

test("训练失败（ML_* 错误码）：保留原因并给重试建议", () => {
  const err = formatError(backendError("模型拟合过程中断", 500, { code: "ML_TRAIN_FAILED" }));
  assertThreeLayers(err);
  assert.ok(err.what.includes("模型拟合过程中断"), err.what);
  assert.equal(err.actionLink, "/ml");
  assert.ok(err.solution.includes("重试"), err.solution);
});

test("无错误码的训练失败（SSE error 原文）也按训练失败处理", () => {
  const err = formatError("模型拟合过程中断，请检查特征是否含非法值");
  assertThreeLayers(err);
  assert.equal(err.actionLink, "/ml");
  assert.ok(err.impact.includes("模型"), err.impact);
});

test("普通业务文案不会被误判成训练失败", () => {
  const err = formatError("数据集名称不能为空");
  assertThreeLayers(err);
  assert.equal(err.actionLink, undefined);
});

test("API 超时：提示不要立刻重试，并给确认入口", () => {
  const err = formatError(backendError("timeout of 60000ms exceeded", 0, { code: "ECONNABORTED" }));
  assertThreeLayers(err);
  assert.ok(err.what.includes("超时"), err.what);
  assert.ok(err.impact.includes("重复数据"), err.impact);
  assert.equal(err.actionLink, "/datasets");
});

test("网络错误与 5xx 也走三层结构", () => {
  const network = formatError(new Error("Network Error"));
  assertThreeLayers(network);
  assert.ok(network.what.includes("网络"), network.what);

  const server = formatError(backendError("Internal Server Error", 500));
  assertThreeLayers(server);
  assert.ok(server.what.includes("服务端内部错误"), server.what);
});

test("缺列（details.missing）：把列名补进提示", () => {
  const err = formatError(
    backendError("请求失败（HTTP 422）", 422, { details: { missing: ["DepDelay", "Month"] } }),
  );
  assertThreeLayers(err);
  assert.ok(err.what.includes("DepDelay") && err.what.includes("Month"), err.what);
});

test("formatError 对上任何畸形输入都返回完整三层，且不抛异常", () => {
  for (const input of [null, undefined, {}, "", 0, "纯字符串错误", { message: 123 }, { details: 42 }]) {
    assertThreeLayers(formatError(input));
  }
});

test("clientError / withPrefix 保持三层与上下文", () => {
  const base = clientError("还没有选择文件", "请先选择文件再上传。");
  assertThreeLayers(base);

  const prefixed = withPrefix(base, "相关性分析：");
  assert.ok(prefixed.what.startsWith("相关性分析："), prefixed.what);
  assert.equal(prefixed.impact, base.impact);
  assert.equal(prefixed.solution, base.solution);
});

