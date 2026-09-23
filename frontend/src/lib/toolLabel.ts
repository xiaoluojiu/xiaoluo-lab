/** 工具名 → 人话。

这个文件**刻意零依赖**（不 import 任何模块，也不碰 zustand / router），
因为它承载的是纯字符串逻辑，也是最容易写出 off-by-one 的地方：
「描述的首句」可能长达几百字，直接塞进流水线气泡里会把整行挤歪。
零依赖使得它可以被 Node 自带 test runner 直接单测，不必装 vitest。
*/

export interface ToolLike {
  name: string;
  description?: string;
}

/** 首句分隔符。
 *
 * - 中文句号 `。` / 中英文分号 `；;` / 换行；
 * - 英文句号：只认**后面跟着空白**的那个（`. `），否则会把版本号、
 *   缩写（`model.v2`、`e.g`）误切成首句。这也是过去只支持中文句号的原因，
 *   但纯英文描述会整段留下来——正是首句截断要解决的问题。
 */
const SENTENCE_BREAK = /[。；;\n]|\.\s/;

/** 显示宽度上限：超出就用省略号收尾。 */
export const MAX_LABEL_CHARS = 30;

/** 截断到 max 个字符，超出补省略号；max <= 0 时不截断。 */
export function truncateWithEllipsis(text: string, max: number): string {
  const value = (text ?? "").trim();
  if (max <= 0 || value.length <= max) return value;
  return `${value.slice(0, max)}…`;
}

/**
 * 取工具描述的第一句话（按中英文句号/分号/换行切分）。
 *
 * 历史实现只做「取首句」，没有长度上限：一旦某个工具的 description 第一句
 * 写得长（例如把适用条件一起写进去了），流水线气泡会被这行文本撑到换行，
 * 工具栏的下游布局跟着抖。这里补上长度上限，并保证一定有省略号标记被截断。
 */
export function firstSentence(description: string, max: number = MAX_LABEL_CHARS): string {
  const head = (description ?? "").split(SENTENCE_BREAK)[0]?.trim() ?? "";
  if (!head) return "";
  return truncateWithEllipsis(head, max);
}

/**
 * 工具名 → 友好名：优先 description 首句，回退工具名本身。
 *
 * @param name 工具名，如 `data.filter`
 * @param tools 已加载的工具清单；**允许为空数组**（工具目录还没拉到时），
 *              此时直接返回工具名，而不是渲染成 "undefined"。
 * @param max 显示宽度上限
 */
export function toolLabel(name: string, tools: readonly ToolLike[], max: number = MAX_LABEL_CHARS): string {
  if (!name) return "";
  const found = tools?.find((tool) => tool?.name === name);
  const candidate = firstSentence(found?.description ?? "", max);
  if (candidate) return candidate;
  return truncateWithEllipsis(name, max);
}
