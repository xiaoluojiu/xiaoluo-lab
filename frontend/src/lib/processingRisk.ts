/**
 * 数据处理「影响评估」。
 *
 * 背景：筛选 / 去重 / 缺失值 / 类型转换等操作原先点「执行」就直接落新版本，
 * 用户没有机会看清这次操作会删掉多少行、多少列。
 * 预览弹窗需要把后端 dry-run 返回的 before / after 行列表数，
 * 翻译成「将影响 X 行，Y 列」以及「是否属于高风险（必须二次确认）」两个结论。
 *
 * 这里刻意做成**纯函数 + 无 React / DOM 依赖**：前端测试跑在
 * `node --experimental-strip-types` 下（没有 jsdom），只有纯逻辑才可被单测覆盖。
 * 阈值也都导出为常量，将来调整口径只改这一处。
 */

export interface FrameShape {
  rows: number;
  columns: number;
}

export type RiskLevel = "low" | "medium" | "high";

export interface OperationImpact {
  /** 被删除的行数（>= 0）。 */
  removedRows: number;
  /** 新增的行数（>= 0）。 */
  addedRows: number;
  /** 被删除的列数（>= 0）。 */
  removedColumns: number;
  /** 新增的列数（>= 0）。 */
  addedColumns: number;
  /** 原表非空但结果变成 0 行 —— 最危险的一类。 */
  emptiesTable: boolean;
  /** 删除行占输入行数的比例（0 ~ 1；输入为 0 时记 0）。 */
  removedRowRatio: number;
  /** 行数与列数都没变（只可能改单元格内容）。 */
  sameShape: boolean;
}

export interface RiskAssessment {
  level: RiskLevel;
  /** 是否需要「强制二次确认」（删除列 / 删除大量行 / 结果为空表）。 */
  requiresSecondConfirmation: boolean;
  /** 逐条展示给用户的影响说明。 */
  reasons: string[];
  impact: OperationImpact;
}

/** 删除行数达到该值即视为「删除大量行」。 */
export const HIGH_RISK_REMOVED_ROWS = 1000;
/** 删除行占输入行的比例达到该值即视为「删除大量行」。 */
export const HIGH_RISK_REMOVED_ROW_RATIO = 0.3;

const LEVEL_RANK: Record<RiskLevel, number> = { low: 0, medium: 1, high: 2 };

/** before / after 行列表数 -> 影响量。 */
export function assessImpact(before: FrameShape, after: FrameShape): OperationImpact {
  const rowDelta = after.rows - before.rows;
  const columnDelta = after.columns - before.columns;
  const removedRows = Math.max(0, -rowDelta);
  return {
    removedRows,
    addedRows: Math.max(0, rowDelta),
    removedColumns: Math.max(0, -columnDelta),
    addedColumns: Math.max(0, columnDelta),
    emptiesTable: before.rows > 0 && after.rows === 0,
    removedRowRatio: before.rows > 0 ? removedRows / before.rows : 0,
    sameShape: rowDelta === 0 && columnDelta === 0,
  };
}

/**
 * 风险判定。
 *
 * 只有「会丢数据」才升级为 high（需二次确认）：
 *   1. 结果为空表；
 *   2. 删除行数 >= 1000 或删除占比 >= 30%；
 *   3. 删除列（列数减少，含聚合 / 逆透视这类塌缩形状的操作）。
 * 新增行 / 列不判 high —— 它不破坏原始信息，但会提示（表格变宽/变长）。
 */
export function assessOperationRisk(before: FrameShape, after: FrameShape): RiskAssessment {
  const impact = assessImpact(before, after);
  const findings: Array<{ level: RiskLevel; reason: string }> = [];

  if (impact.emptiesTable) {
    findings.push({
      level: "high",
      reason: `结果将是空表：原有 ${formatInt(before.rows)} 行会被全部删除`,
    });
  } else if (impact.removedRows > 0) {
    const heavy =
      impact.removedRows >= HIGH_RISK_REMOVED_ROWS ||
      impact.removedRowRatio >= HIGH_RISK_REMOVED_ROW_RATIO;
    const ratio = formatPercent(impact.removedRowRatio);
    findings.push({
      level: heavy ? "high" : "medium",
      reason: heavy
        ? `删除大量行：${formatInt(impact.removedRows)} 行（占输入 ${ratio}）`
        : `删除 ${formatInt(impact.removedRows)} 行（占输入 ${ratio}）`,
    });
  }

  if (impact.removedColumns > 0) {
    findings.push({
      level: "high",
      reason: `删除列：${formatInt(impact.removedColumns)} 列（列一旦删除，本版本的数据无法再取回）`,
    });
  }

  if (impact.addedColumns > 0) {
    findings.push({ level: "medium", reason: `新增 ${formatInt(impact.addedColumns)} 列（表格会变宽）` });
  }

  if (impact.addedRows > 0) {
    findings.push({ level: "medium", reason: `新增 ${formatInt(impact.addedRows)} 行（行数会变多）` });
  }

  if (!findings.length) {
    // 行列表数不变不代表没动数据（缺失值填充 / 类型转换 / 字符串处理都在改单元格），
    // 所以这里只说「形状不变」，不声称「数据未变」。
    findings.push({ level: "low", reason: "行数与列数都不变，仅可能修改单元格内容" });
  }

  // 取最严重的一条作为整体等级：一次操作可能同时删行又删列。
  const level = findings.reduce<RiskLevel>(
    (worst, item) => (LEVEL_RANK[item.level] > LEVEL_RANK[worst] ? item.level : worst),
    "low",
  );

  return {
    level,
    requiresSecondConfirmation: level === "high",
    reasons: findings.map((item) => item.reason),
    impact,
  };
}

/** 需求指定的口径：「将影响 X 行，Y 列」。 */
export function impactSummary(before: FrameShape, after: FrameShape): string {
  const impact = assessImpact(before, after);
  const rows = Math.max(impact.removedRows, impact.addedRows);
  const columns = Math.max(impact.removedColumns, impact.addedColumns);
  return `将影响 ${formatInt(rows)} 行，${formatInt(columns)} 列`;
}

/** 「10,000 行 × 19 列 → 8 行 × 2 列」。 */
export function shapeTransition(before: FrameShape, after: FrameShape): string {
  return `${formatInt(before.rows)} 行 × ${formatInt(before.columns)} 列 → ${formatInt(after.rows)} 行 × ${formatInt(after.columns)} 列`;
}

export const RISK_LABEL: Record<RiskLevel, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
};

/** 千分位整数（自己实现，避免依赖环境的 Intl 中文数字分组差异）。 */
export function formatInt(value: number): string {
  const rounded = Math.trunc(value);
  const digits = Math.abs(rounded).toString();
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return rounded < 0 ? `-${grouped}` : grouped;
}

/** 比例 -> 百分比（只在这里舍入一次，避免"报告层双重舍入"）。 */
export function formatPercent(ratio: number): string {
  return `${(ratio * 100).toFixed(1)}%`;
}
