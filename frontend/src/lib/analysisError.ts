/**
 * EDA/分析请求的错误文案提取。
 *
 * 背景（bug 5）：分析失败时前端只显示「相关性：请求失败（HTTP 422）」这类通用
 * 文案，既不显示后端给出的具体业务原因，也不说明如何修正。后端其实返回了
 * 结构化错误：``{code, message, details:{missing|numeric_columns|...}}``
 * （见 backend/app/core/exceptions.py 与 middleware.py 的 _error_response）。
 *
 * 本模块把「HTTP 状态码」翻译成人话，并把后端 message 与 details 里的可用线索
 * 拼成一句可操作的中文提示；拿不到后端 message 时才退回状态码文案。
 */

export interface AnalysisErrorLike {
  message?: unknown;
  code?: unknown;
  details?: unknown;
  status?: unknown;
}

const STATUS_HINTS: Record<number, string> = {
  400: "请求参数有误",
  401: "未通过身份校验",
  403: "没有权限执行该操作",
  404: "数据集或版本不存在",
  408: "请求超时",
  413: "请求数据过大",
  422: "请求参数无法处理",
  429: "请求过于频繁",
  500: "服务端内部错误",
  502: "上游服务不可用",
  503: "服务暂时不可用",
};

/** 从字符串里剥掉 axios 包装出的「请求失败（HTTP 422）」外壳，取真实业务文案。 */
function stripTransportWrapper(raw: string): string {
  const matched = raw.match(/^请求失败（HTTP \d{3}）$/);
  return matched ? "" : raw;
}

/**
 * 5xx 的 message 通常是框架/代理抛出的英文内部串（Internal Server Error、
 * Internal Server Error 之类），对用户没有意义，直接换成人话文案。
 */
const OPAQUE_SERVER_MESSAGES = new Set([
  "internal server error",
  "internal error",
  "bad gateway",
  "service unavailable",
  "gateway timeout",
]);

function isOpaqueServerMessage(message: string): boolean {
  return OPAQUE_SERVER_MESSAGES.has(message.toLowerCase());
}

function readString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => readString(item)).filter(Boolean);
}

/** 从 details 里尽量提取「能帮用户修好这次请求」的具体线索。 */
export function detailHints(details: unknown): string[] {
  if (!details || typeof details !== "object") return [];
  const bag = details as Record<string, unknown>;
  const hints: string[] = [];

  const missing = readStringArray(bag.missing);
  if (missing.length) hints.push(`缺失字段：${missing.join("、")}`);

  const rejected = readStringArray(
    bag.rejected_categorical_numeric ?? bag.rejected_columns,
  );
  if (rejected.length) {
    hints.push(`以下字段取值种类过少、按分类处理，不参与相关性：${rejected.join("、")}`);
  }

  const nonNumeric = readStringArray(bag.rejected_non_numeric);
  if (nonNumeric.length) hints.push(`以下字段不是数值类型：${nonNumeric.join("、")}`);

  const available = readStringArray(bag.available ?? bag.available_numeric_columns);
  if (available.length) {
    const shown = available.slice(0, 8).join("、");
    hints.push(
      available.length > 8 ? `可用字段（前 8 个）：${shown} 等` : `可用字段：${shown}`,
    );
  }

  const hint = readString(bag.hint);
  if (hint) hints.push(hint);

  return hints;
}

/**
 * 把任意抛出的错误转成「带后端业务原因」的可读文案。
 *
 * 优先使用后端 message（如「热力图：以下列虽为数值但取值种类过少…」），
 * 再补 details 里的列名线索；两者都没有才退回状态码文案。
 */
export function describeAnalysisError(error: unknown): string {
  if (!error) return "分析失败，请稍后重试";
  if (typeof error === "string") return stripTransportWrapper(error) || "分析失败，请稍后重试";

  const bag = error as AnalysisErrorLike;
  const status = typeof bag.status === "number" ? bag.status : undefined;
  const rawMessage = readString(bag.message);
  const message = stripTransportWrapper(rawMessage);

  const hints = detailHints(bag.details);
  const parts: string[] = [];

  const serverHint = status && status >= 500 ? STATUS_HINTS[status] ?? "服务端内部错误" : "";
  const messageIsOpaque = !message || isOpaqueServerMessage(message);

  if (serverHint && messageIsOpaque) {
    // 5xx：不把内部实现吐给用户，给一句人话 + 下一步动作。
    parts.push(`${serverHint}，请稍后重试或联系管理员`);
    if (message) parts.push(message);
  } else if (message) {
    parts.push(message);
  } else if (status && STATUS_HINTS[status]) {
    parts.push(STATUS_HINTS[status] + (status >= 500 ? "，请稍后重试或联系管理员" : ""));
  } else if (rawMessage) {
    parts.push(rawMessage);
  } else {
    parts.push("分析失败，请稍后重试");
  }

  for (const hint of hints) {
    if (!parts.includes(hint)) parts.push(hint);
  }

  return parts.join("。");
}
