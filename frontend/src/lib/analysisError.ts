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

/* =====================================================================
 * 统一错误提示层（what / impact / solution）
 *
 * 背景：`describeAnalysisError` 只给一句可读文案，用户知道「哪里错了」，
 * 却不知道「这件事影响到什么」以及「下一步该点哪」。本模块把任意错误收敛
 * 成一个标准对象，交给 ErrorNotice / Toast 渲染成三层信息，并在能给出
 * 明确去处时附带一个可点击的行动按钮。
 *
 * 约束：不改后端错误码 —— 这里只做「读」：按 code / status / message 关键词
 * 把后端已有的错误翻译成人话，拿不到业务线索时退回状态码文案。
 * ===================================================================== */

/** 行动按钮的目标路由（应用内）。 */
export const ERROR_ROUTES = {
  datasets: "/datasets",
  processing: "/processing",
  analysis: "/analysis",
  ml: "/ml",
} as const;

/**
 * 标准错误对象：面向用户的错误一律收敛成这三层 + 可选行动入口。
 * - what     发生了什么（尽量带具体列名 / 数据集名）
 * - impact   影响（让用户判断严重程度：数据有没有被改动、还能不能继续）
 * - solution 怎么解决（可操作的下一步）
 */
export interface AnalysisError {
  what: string;
  impact: string;
  solution: string;
  actionLabel?: string;
  actionLink?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/** 客户端本地校验（未发起请求）也能给出三层信息，保持全局口径一致。 */
export function clientError(
  what: string,
  solution: string,
  impact = "本次操作已中止，数据没有发生变化。",
): AnalysisError {
  return { what, impact, solution };
}

/** 给已经结构化的错误补一个上下文前缀（如「描述性统计：」）。 */
export function withPrefix(error: AnalysisError, prefix: string): AnalysisError {
  return { ...error, what: `${prefix}${error.what}` };
}

/** 目标列缺失值偏多时的提示（前端预检，后端不会主动报这个错）。 */
export function targetMissingValuesError(options: {
  target: string;
  missing: number;
  total: number;
}): AnalysisError {
  const { target, missing, total } = options;
  const pct = total > 0 ? Math.round((missing / total) * 100) : 0;
  return {
    what: `目标列包含缺失值：目标列「${target}」有 ${missing} / ${total} 行（约 ${pct}%）为空`,
    impact: "这些行会在训练时被自动剔除，样本量可能骤减，模型容易学到偏斜的结果。",
    solution: "建议先处理：前往「数据处理」对目标列做缺失值填充或删除这些行，再回来重新训练。",
    actionLabel: "去处理缺失值",
    actionLink: ERROR_ROUTES.processing,
  };
}

/* ------------------------------ 归一化 ------------------------------ */

interface RawError {
  message: string;
  code: string;
  status?: number;
  details?: unknown;
}

function normalizeError(error: unknown): RawError {
  if (!error) return { message: "", code: "" };
  if (typeof error === "string") {
    return { message: stripTransportWrapper(error.trim()), code: "" };
  }

  if (typeof error === "object") {
    const bag = error as AnalysisErrorLike & { response?: { data?: unknown; status?: number } };
    const rawMessage = readString(bag.message);
    const responseData = bag.response?.data;
    const responseBody = isRecord(responseData)
      ? (responseData as { error?: { code?: unknown; details?: unknown } })
      : undefined;

    const status =
      typeof bag.status === "number"
        ? bag.status
        : typeof bag.response?.status === "number"
          ? bag.response.status
          : undefined;

    const details =
      bag.details ??
      responseBody?.error?.details ??
      (isRecord(responseData) ? (responseData as { details?: unknown }).details : undefined);

    const code = (readString(bag.code) || readString(responseBody?.error?.code)).toUpperCase();

    return { message: stripTransportWrapper(rawMessage), code, status, details };
  }

  return { message: String(error), code: "" };
}

/* ------------------------------ 分类判定 ------------------------------ */

const TIMEOUT_CODES = new Set([
  "ECONNABORTED",
  "ETIMEDOUT",
  "TIMEOUT",
  "REQUEST_TIMEOUT",
  "UPSTREAM_TIMEOUT",
]);
const NETWORK_CODES = new Set([
  "ERR_NETWORK",
  "ECONNREFUSED",
  "ECONNRESET",
  "ENETUNREACH",
  "EAI_AGAIN",
  "ERR_INTERNET_DISCONNECTED",
]);
const FORMAT_CODES = new Set([
  "PARSE_ERROR",
  "DATA_FORMAT_ERROR",
  "INVALID_FORMAT",
  "FILE_PARSE_ERROR",
  "CSV_PARSE_ERROR",
  "INGEST_PARSE_ERROR",
  "MALFORMED_INPUT",
  "SCHEMA_INFERENCE_ERROR",
]);

const TIMEOUT_MESSAGE = /超时|timeout|timed out/i;
const NETWORK_MESSAGE =
  /network error|err_network|econnrefused|econnreset|failed to fetch|load failed|网络|连接中断|无法连接|连接被重置/i;
const FORMAT_MESSAGE =
  /could not parse|could not read|parse error|parsing error|delimiter|invalid utf|utf-?8|encoding|decode|malformed|not a valid|failed to (read|parse|decode)|expected \d+ fields|no data|empty (file|csv)|schema ?inference|解析失败|无法解析|格式(错误|不正确|无法)|编码(错误|不支持)|分隔符|列数不一致|不是有效的(表格|文件)/i;
const TARGET_COLUMN_MESSAGE =
  /目标列\s*['"「]?([^'"」\s]+)['"」]?\s*不存在|目标列不存在|target column.*not (found|exist)|unknown target/i;
const MISSING_COLUMN_MESSAGE =
  /指定的字段不存在|字段不存在|缺失字段|missing column|column .*not found|columnnotfound|列不存在|未知字段/i;

const STATUS_SOLUTIONS: Record<number, string> = {
  400: "请检查所选的字段与参数后重试。",
  401: "请重新登录后再试。",
  403: "请确认当前账号是否有该数据集的操作权限。",
  404: "请刷新列表，重新选择一个仍然存在的数据集或版本。",
  413: "请缩小数据范围或分批处理后再试。",
  422: "请按提示调整所选的字段或参数后重试。",
  429: "请稍等片刻再试，避免连续点击。",
};

/** 从解析类报错里尽量抠出「是哪一列出的问题」。 */
function extractColumnFromParse(message: string, details?: unknown): string {
  // 只认「明确点名某一列」的写法：Polars 的报错形如
  // `could not parse 'x' as dtype 'Float64' at column 'Age'`。不把被解析失败的那个
  // **值**（如 'x'）当列名 —— 否则会显示成「列 x 有问题」，反而误导。
  const patterns = [
    /at column [`'"]([^`'"]+)[`'"]/i,
    /in column [`'"]([^`'"]+)[`'"]/i,
    /column [`'"]([^`'"]+)[`'"]/i,
    /字段\s*[「'"]([^」'"]+)[」'"]/,
    /列\s*[「'"]([^」'"]+)[」'"]/,
  ];
  for (const pattern of patterns) {
    const matched = message.match(pattern);
    if (matched?.[1]) return matched[1].trim();
  }

  if (isRecord(details)) {
    for (const key of ["column", "column_name", "field", "target"]) {
      const value = readString(details[key]);
      if (value) return value;
    }
    const columns = readStringArray(details.columns);
    if (columns.length === 1) return columns[0];
  }
  return "";
}

/** details 里的可用线索，拼到 solution 后面，告诉用户「能拿什么来修」。 */
function hintsAsSolution(hints: string[]): string {
  return hints.join("；");
}

/* ------------------------------ 各类错误构造 ------------------------------ */

function timeoutError(): AnalysisError {
  return {
    what: "请求超时：服务端在规定时间内没有返回结果",
    impact: "本次请求的结果没有拿到；但服务端可能仍在后台处理（大文件入库较慢），立即重试可能产生重复数据。",
    solution:
      "请稍后刷新列表确认结果，确认没有生成数据后再重试；若文件很大，可先切分再上传。",
    actionLabel: "去数据集列表确认",
    actionLink: ERROR_ROUTES.datasets,
  };
}

function networkError(raw: RawError): AnalysisError {
  return {
    what: "网络连接失败：浏览器没能连上服务端",
    impact:
      raw.message && raw.message !== "网络错误"
        ? `本次操作没有完成。（底层信息：${raw.message}）`
        : "本次操作没有完成，页面上的数据仍是上一次的结果。",
    solution: "请检查本机网络或后端服务是否在运行，恢复后重试。",
  };
}

function formatErrorOf(raw: RawError): AnalysisError {
  const column = extractColumnFromParse(raw.message, raw.details);
  return {
    what: column
      ? `文件格式无法解析：列「${column}」的内容与列类型不匹配`
      : "上传的文件格式无法解析，可能不是有效的表格文件",
    impact: "该文件没有被导入，数据集尚未创建；基于它的分析与训练都无法进行。",
    solution: column
      ? `请检查「${column}」列是否混入了文字或空行、数值列是否含非法字符，修正后重新上传。`
      : "请确认文件是标准 CSV/Excel：以逗号分隔、每行列数一致、编码为 UTF-8、没有多余空行或合并单元格。",
    actionLabel: "重新上传数据",
    actionLink: ERROR_ROUTES.datasets,
  };
}

function targetColumnMissingError(column: string): AnalysisError {
  return {
    what: column ? `目标列「${column}」不在数据集中` : "所选的目标列不在数据集中",
    impact: "训练无法开始，因为模型不知道该预测哪一列。",
    solution: "请在「目标列」下拉中重新选择一个真实存在的列，或使用系统推荐的目标列。",
    actionLabel: "去选择目标列",
    actionLink: ERROR_ROUTES.ml,
  };
}

function missingColumnsError(raw: RawError): AnalysisError {
  const missing = isRecord(raw.details) ? readStringArray(raw.details.missing) : [];
  const hints = hintsAsSolution(detailHints(raw.details));
  // 优先用后端文案；但若它没点名缺哪几列，就把 details.missing 补上 —— 用户最需要的就是「缺哪列」。
  const base = raw.message || "请求的字段在当前数据集中不存在";
  const listsNames = missing.length > 0 && missing.every((name) => base.includes(name));
  const what = missing.length && !listsNames ? `${base}（缺失字段：${missing.join("、")}）` : base;
  return {
    what,
    impact: "本次分析没有完成，结果面板保持原样。",
    solution: hints || "请刷新字段列表，重新勾选仍然存在的字段后再运行。",
    actionLabel: "去重新选择字段",
    actionLink: ERROR_ROUTES.analysis,
  };
}

const TRAINING_PREFLIGHT: Array<{ test: RegExp; build: (raw: RawError) => AnalysisError }> = [
  {
    test: /目标列全为空|目标列\s*全部为空/,
    build: () => ({
      what: "目标列全部为空，没有可用于训练的真实标签",
      impact: "模型没有任何可学习的目标，训练无法开始。",
      solution: "请改选一个有取值的列作为目标列，或先到「数据处理」补齐该列。",
      actionLabel: "去处理缺失值",
      actionLink: ERROR_ROUTES.processing,
    }),
  },
  {
    test: /目标列无变化|所有取值相同/,
    build: () => ({
      what: "目标列所有取值都相同，没有区分度",
      impact: "回归模型无法从恒定不变的标签里学到任何规律，训练没有意义。",
      solution: "请换一个取值有波动的列作为目标列，或核对是否选错了列。",
      actionLabel: "去重新选择目标列",
      actionLink: ERROR_ROUTES.ml,
    }),
  },
  {
    test: /分类任务至少需要\s*2\s*个类别/,
    build: () => ({
      what: "目标列只有 1 个类别，无法做分类",
      impact: "分类模型至少需要 2 个类别才能区分，训练无法开始。",
      solution: "请确认目标列选得是否正确，或改用「回归 / 聚类」任务。",
    }),
  },
  {
    test: /训练样本数\s*<\s*2|样本数不足/,
    build: () => ({
      what: "可用于训练的样本不足（少于 2 行）",
      impact: "没有足够的数据让模型拟合，训练无法开始。",
      solution: "请放宽筛选条件、少排除一些列，或换一个行数更多的数据集。",
    }),
  },
  {
    test: /训练特征列为空|排除后无可用列|特征列为空/,
    build: () => ({
      what: "排除所选列之后，已经没有任何特征可用于训练",
      impact: "模型没有输入可用，训练无法开始。",
      solution: "请减少要排除的列，至少保留一个特征列，再重新训练。",
      actionLabel: "去调整特征列",
      actionLink: ERROR_ROUTES.ml,
    }),
  },
  {
    test: /X\s*与\s*y\s*行数不一致/,
    build: () => ({
      what: "特征与目标列的行数不一致",
      impact: "数据对齐异常，无法训练。",
      solution: "请刷新数据版本后重试；若仍失败，请重新导入该数据集。",
    }),
  },
];

/** 训练语境的关键词：用于识别「没带错误码」的训练失败（如 SSE error 事件原文）。 */
const TRAINING_HINT = /训练|train|拟合|estimator|sklearn/i;

function trainFailureError(what: string): AnalysisError {
  return {
    what: what || "训练过程中出错，模型没有产出结果",
    impact: "本次训练失败，没有生成可用的模型；已有的历史模型不受影响。",
    solution: "请检查目标列、特征与预处理配置后重试；若反复失败，请把错误信息提供给开发者。",
    actionLabel: "去查看运行详情",
    actionLink: ERROR_ROUTES.ml,
  };
}

function trainingError(raw: RawError): AnalysisError | null {
  const matched = TRAINING_PREFLIGHT.find((item) => item.test.test(raw.message));
  if (matched) return matched.build(raw);

  if (raw.code.startsWith("ML_") || raw.code.startsWith("TRAIN_") || raw.code.startsWith("MODEL_")) {
    return trainFailureError(raw.message);
  }

  // SSE error 事件只回一段文本、没有 code：命中训练语境且不是格式解析问题时按训练失败处理。
  if (TRAINING_HINT.test(raw.message) && !FORMAT_MESSAGE.test(raw.message)) {
    return trainFailureError(raw.message);
  }
  return null;
}

/* ------------------------------ 主入口 ------------------------------ */

/**
 * 把任意抛出的错误映射成标准错误对象（what / impact / solution + 可选行动入口）。
 *
 * 优先级（对应四类高频问题）：
 *   1. 数据格式错误   2. 目标列缺失（不存在 / 缺失值过多）
 *   3. 训练失败       4. API 超时
 * 之后依次是：网络错误 → 缺列 → 服务端 5xx → 状态码兜底。
 */
export function formatError(error: unknown): AnalysisError {
  const raw = normalizeError(error);

  if (!raw.message && !raw.status && !raw.code) {
    return {
      what: "操作没有完成",
      impact: "本次操作已中止，数据没有发生变化。",
      solution: "请稍后重试；若反复出现，请把错误信息提供给开发者。",
    };
  }

  // 4. API 超时（放在最前：超时文案不能被误判成训练失败）
  if (TIMEOUT_CODES.has(raw.code) || raw.status === 408 || raw.status === 504 || TIMEOUT_MESSAGE.test(raw.message)) {
    return timeoutError();
  }

  // 网络层错误
  if (NETWORK_CODES.has(raw.code) || (!raw.status && NETWORK_MESSAGE.test(raw.message))) {
    return networkError(raw);
  }

  // 2b. 目标列不存在
  const targetMatch = raw.message.match(TARGET_COLUMN_MESSAGE);
  if (targetMatch || raw.code === "TARGET_NOT_FOUND" || raw.code === "MISSING_TARGET") {
    return targetColumnMissingError((targetMatch?.[1] ?? "").trim());
  }

  // 3. 训练失败（先于格式判定：训练报错里也可能出现 "parse"）
  const training = trainingError(raw);
  if (training) return training;

  // 1. 数据格式错误
  if (FORMAT_CODES.has(raw.code) || FORMAT_MESSAGE.test(raw.message)) {
    return formatErrorOf(raw);
  }

  // 2a. 缺列
  const missing = isRecord(raw.details) ? readStringArray(raw.details.missing) : [];
  if (missing.length || MISSING_COLUMN_MESSAGE.test(raw.message)) {
    return missingColumnsError(raw);
  }

  // 服务端 5xx：不把内部实现吐给用户
  const hints = hintsAsSolution(detailHints(raw.details));
  if (raw.status && raw.status >= 500) {
    const serverHint = STATUS_HINTS[raw.status] ?? "服务端内部错误";
    return {
      what: serverHint,
      impact: "本次请求没有完成，相关结果未生成。",
      solution:
        raw.status === 503 || raw.status === 502
          ? "服务可能正在重启或过载，请稍后重试。"
          : "请稍后重试；若反复出现，请把错误信息提供给开发者。",
    };
  }

  // 状态码 + 业务文案兜底
  const what =
    raw.message ||
    (raw.status ? STATUS_HINTS[raw.status] ?? `请求失败（HTTP ${raw.status}）` : "操作失败");
  const solution =
    (raw.status ? STATUS_SOLUTIONS[raw.status] : undefined) ??
    "请按提示核对输入后重试；若反复出现，请把错误信息提供给开发者。";

  return {
    what,
    impact: "本次操作没有完成。",
    solution: hints ? `${solution}（${hints}）` : solution,
  };
}

