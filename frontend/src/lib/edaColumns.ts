/**
 * EDA 图表选列 —— 与后端 `analysis.classify_columns` 对齐的「可用数值列」判定，
 * 以及把数据集 schema 变成「该选哪个字段」的文案建议。
 *
 * 背景（用户反馈的 heatmap 422）：
 * 后端 ``continuous_columns`` 会把「数值型但取值种类极少」的列（Month、DayofMonth、
 * DayOfWeek、CRSDepTime 这类按分类编码的整数列）排除在相关性之外。
 * 前端原先只按 dtype 判断"是不是数值列"，于是把 Month 也当数值列提交，
 * 后端按业务规则拒绝 —— 请求是前端组错的，不该让用户去猜。
 *
 * 本模块做三件事：
 * 1. `isContinuousNumeric` —— 与后端同一套判据，前端自己就能算「真正的数值列」；
 * 2. `splitHeatmapColumns` —— 拆出「能算相关性的列」与「会被后端拒绝的列」；
 * 3. `chartFieldAdvice` —— 依据当前 schema 给出可直接照做的选字段建议。
 *
 * 注意：dtype 无法反映基数，需配合 `schema.columns[].distinct / unique_count`
 * 之类的统计（有则用之，无则退化为「只信 dtype」）。
 */

import type { SchemaColumn } from "../types/dataset";

/** 后端同款阈值：低基数判据 n_unique <= 50 且 n_unique * 2 < 行数。 */
export const MAX_CATEGORICAL_CARDINALITY = 50;

/** 数值类型（与 VisualizationPanel.isNumericColumn 保持一致）。 */
export function isNumericColumn(c: SchemaColumn): boolean {
  return /int|float|double|decimal|number/i.test(c.dtype);
}

/** 时间类型。 */
export function isTemporalColumn(c: SchemaColumn): boolean {
  return /date|time/i.test(c.dtype);
}

function readCardinality(c: SchemaColumn): number | undefined {
  const bag = c as unknown as Record<string, unknown>;
  for (const key of ["distinct", "distinct_count", "unique_count", "n_unique"]) {
    const value = bag[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

/**
 * 是否是「可参与相关性/连续性统计」的数值列。
 *
 * 与后端 `continuous_columns` 语义对齐：
 * - 非数值 dtype ⇒ false；
 * - 数值 dtype 且已知基数、基数很低 ⇒ false（后端按分类编码列处理）；
 * - 基数未知 ⇒ true（前端信息不足时宁可放行，由后端做权威判定）。
 */
export function isContinuousNumeric(c: SchemaColumn, rowCount?: number): boolean {
  if (!isNumericColumn(c)) return false;
  const cardinality = readCardinality(c);
  if (cardinality === undefined) return true;
  if (cardinality > MAX_CATEGORICAL_CARDINALITY) return true;
  // 行数未知时，低基数本身即视为分类编码（与后端 rows 足够大时的结论一致）。
  if (rowCount === undefined || rowCount <= 0) return true;
  return !(cardinality * 2 < rowCount);
}

export interface HeatmapSplit {
  /** 可参与相关性矩阵的数值列。 */
  usable: string[];
  /** 数值型但会被后端按分类编码排除的列。 */
  rejected: string[];
  /** 非数值类型、且被勾选的列。 */
  nonNumeric: string[];
}

/**
 * 拆分热力图候选字段。
 *
 * `picked` 为左侧已勾选的列；为空时视作「未指定」。
 * `pool` 为当前数据集全部列。
 */
export function splitHeatmapColumns(
  picked: string[],
  pool: SchemaColumn[],
  rowCount?: number,
): HeatmapSplit {
  const byName = new Map(pool.map((c) => [c.column, c]));
  const scope = picked.length ? picked : pool.map((c) => c.column);
  const usable: string[] = [];
  const rejected: string[] = [];
  const nonNumeric: string[] = [];
  for (const name of scope) {
    const col = byName.get(name);
    if (!col) continue;
    if (!isNumericColumn(col)) {
      if (picked.includes(name)) nonNumeric.push(name);
      continue;
    }
    if (isContinuousNumeric(col, rowCount)) usable.push(name);
    else rejected.push(name);
  }
  return { usable, rejected, nonNumeric };
}

export interface ChartFieldAdvice {
  /** 建议里可直接高亮的字段名（前端用于一键勾选）。 */
  suggested: string[];
  /** 给用户看的建议文案。 */
  message: string;
}

/**
 * 依据当前数据集字段，给出「这张图该选什么」的具体建议。
 *
 * 返回的 `suggested` 是可直接替用户勾上的字段，`message` 是原因说明，
 * 让用户不必自己推断哪种 dtype 配哪种图。
 */
export function chartFieldAdvice(
  chart: string,
  pool: SchemaColumn[],
  rowCount?: number,
): ChartFieldAdvice {
  const numeric = pool.filter((c) => isContinuousNumeric(c, rowCount)).map((c) => c.column);
  const temporal = pool.filter(isTemporalColumn).map((c) => c.column);
  const categorical = pool
    .filter((c) => !isNumericColumn(c) && !isTemporalColumn(c))
    .map((c) => c.column);
  const shortFall = pool
    .filter((c) => isNumericColumn(c) && !isContinuousNumeric(c, rowCount))
    .map((c) => c.column);

  const preview = (names: string[], n = 3) =>
    names.slice(0, n).join("、") + (names.length > n ? " 等" : "");

  switch (chart) {
    case "heatmap":
    case "scatter":
      return {
        suggested: numeric.slice(0, 2),
        message:
          numeric.length >= 2
            ? `这张图需要 2 个以上"取值连续"的数值字段，当前可用：${preview(numeric)}。`
            : shortFall.length
              ? `当前数据集只有 ${numeric.length} 个取值连续的数值字段；${preview(shortFall)} 这类列取值种类过少，后端按分类编码处理，不能用于相关性，请换用其他数据集或改用柱状图。`
              : "当前数据集没有足够的数值字段，请换一个数据更完整的数据集，或改用柱状图。",
      };
    case "line":
      return {
        suggested: [temporal[0], numeric[0]].filter(Boolean) as string[],
        message:
          temporal.length && numeric.length
            ? `折线图建议用时间字段作横轴、数值字段作纵轴，例如 ${temporal[0]} × ${numeric[0]}。`
            : numeric.length
              ? `当前没有时间字段，折线图会退化为「数值 × 数值」，建议至少准备 2 个数值字段，例如 ${preview(numeric, 2)}。`
              : "折线图需要至少 1 个数值字段。",
      };
    case "bar":
      return {
        suggested: categorical.slice(0, 1),
        message:
          categorical.length
            ? `柱状图需要一个分类字段（非数值），当前可用：${preview(categorical)}。`
            : "当前数据集没有分类字段，柱状图不适用，可改用直方图或面积图查看数值字段分布。",
      };
    case "grouped_bar":
      return {
        suggested: [categorical[0], numeric[0], categorical[1]].filter(Boolean) as string[],
        message:
          categorical.length >= 2 && numeric.length
            ? `分组柱状图需要分类列 × 数值列 × 另一个分组列，例如 ${categorical[0]} × ${numeric[0]}（分组 ${categorical[1]}）。`
            : "分组柱状图需要至少 2 个分类字段与 1 个数值字段，当前字段不足。",
      };
    case "histogram":
    case "boxplot":
    case "qq":
    case "area":
      return {
        suggested: numeric.slice(0, 1),
        message:
          numeric.length
            ? `这张图作用于单个数值字段，建议：${numeric[0]}。`
            : "当前数据集没有可用的数值字段，请更换数据集。",
      };
    default:
      return { suggested: [], message: "" };
  }
}
