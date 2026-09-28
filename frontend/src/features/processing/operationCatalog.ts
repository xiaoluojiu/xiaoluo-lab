/**
 * 数据处理的操作目录与表单元数据。
 *
 * 定位：`features/processing/` 的领域元数据层 —— 只放「有哪些操作、每个操作的表单
 * 默认值、字段类型到可选算子的映射」这类纯数据/纯计算，不含任何请求与 React 状态。
 *
 * 来源：这些内容原先全部内联在 `pages/Processing/index.tsx`（页面既当路由壳、
 * 又当业务模块，516 行）。归并到 features 后，页面只剩「选数据集 → 切 Tab → 装配」。
 */
import type { SchemaColumn } from "../../types/dataset";
import { isNumericDtype, isTemporalDtype } from "../../lib/edaColumns";

export type Tab =
  | "clean"
  | "duplicate"
  | "cast"
  | "string"
  | "filter"
  | "transform"
  | "aggregate"
  | "pivot"
  | "melt"
  | "merge"
  | "multi_merge";
/** 需要走 OperationPanel 的表单型 Tab。 */
export type FormTab = Exclude<Tab, "merge" | "multi_merge">;
export type FormState = Record<string, unknown>;

export const TABS: Array<{ key: Tab; label: string; description: string; group: string }> = [
  { key: "clean", label: "缺失值", description: "删除或填充缺失值", group: "清洗" }, { key: "duplicate", label: "重复值", description: "按字段去除重复记录", group: "清洗" }, { key: "cast", label: "类型转换", description: "转换字段数据类型", group: "清洗" }, { key: "string", label: "字符串", description: "清理或替换文本字段", group: "清洗" },
  { key: "filter", label: "筛选", description: "按条件保留目标行", group: "行处理" }, { key: "transform", label: "字段转换", description: "计算新字段或修改字段", group: "字段处理" }, { key: "aggregate", label: "聚合", description: "按字段分组统计", group: "聚合与汇总" }, { key: "pivot", label: "透视", description: "长表转宽表", group: "表结构" }, { key: "melt", label: "逆透视", description: "宽表转长表", group: "表结构" }, { key: "merge", label: "合并", description: "连接两个数据集", group: "数据集合" }, { key: "multi_merge", label: "多文件合并", description: "勾选多个数据集，对齐共有/独有字段整合为一张表", group: "数据集合" },
];

export const DEFAULT_FORMS: Record<FormTab, FormState> = { clean: { strategy: "mean", columns: [], value: 0 }, duplicate: { subset: [], keep: "first" }, cast: { types: {}, formats: {} }, string: { column: "", op: "trim", params: {} }, filter: { conditions: [{ column: "", op: "gte", value: "" }], logic: "and" }, transform: { name: "", expression: { type: "binary", left: "", operator: "*", right: 1 }, overwrite: false }, aggregate: { group_by: [], aggregations: [{ column: "", func: "sum", alias: "" }] }, pivot: { index: [], columns: "", values: "", aggregation: "first" }, melt: { id_vars: [], value_vars: [], variable_name: "variable", value_name: "value" } };

/**
 * 筛选算子清单：按字段类型给出可选算子。
 * - 数值字段：大小比较算子 + 数字输入
 * - 文本字段：包含 / 等于 / 属于集合
 * - 时间字段：大小比较 + 日期/时间输入（后端支持多格式解析）
 * - 布尔字段：等于 true/false
 * - 为空：不需要取值
 */
export const FILTER_OPS: Record<string, Array<{ value: string; label: string }>> = {
  numeric: [
    { value: "gte", label: "≥" },
    { value: "lte", label: "≤" },
    { value: "gt", label: ">" },
    { value: "lt", label: "<" },
    { value: "eq", label: "等于" },
    { value: "neq", label: "不等于" },
    { value: "in", label: "属于(逗号分隔)" },
    { value: "is_null", label: "为空" },
  ],
  temporal: [
    { value: "gte", label: "≥" },
    { value: "lte", label: "≤" },
    { value: "gt", label: ">" },
    { value: "lt", label: "<" },
    { value: "eq", label: "等于" },
    { value: "neq", label: "不等于" },
    { value: "is_null", label: "为空" },
  ],
  text: [
    { value: "contains", label: "包含" },
    { value: "eq", label: "等于" },
    { value: "neq", label: "不等于" },
    { value: "in", label: "属于(逗号分隔)" },
    { value: "is_null", label: "为空" },
  ],
  boolean: [
    { value: "eq", label: "等于" },
    { value: "neq", label: "不等于" },
    { value: "is_null", label: "为空" },
  ],
  unknown: [
    { value: "gte", label: "≥" },
    { value: "lte", label: "≤" },
    { value: "gt", label: ">" },
    { value: "lt", label: "<" },
    { value: "eq", label: "等于" },
    { value: "neq", label: "不等于" },
    { value: "contains", label: "包含" },
    { value: "in", label: "属于(逗号分隔)" },
    { value: "is_null", label: "为空" },
  ],
};

/**
 * dtype → 字段类型分组。数值/时间两类复用 `lib/edaColumns` 的唯一正则
 * （此前本文件旁边还内联了一份 `isNumericDtype`，与 lib 里的逐字相同）。
 */
export function dtypeKind(dtype: string): keyof typeof FILTER_OPS {
  const t = dtype.toLowerCase();
  if (isNumericDtype(t)) return "numeric";
  if (isTemporalDtype(t)) return "temporal";
  if (/bool/.test(t)) return "boolean";
  if (/str|utf8|object|categorical/.test(t)) return "text";
  return "unknown";
}

/** 透视默认建议：行索引取首个非数值字段，列维度取另一个非数值字段，值取首个数值字段。 */
export function suggestPivot(columns: SchemaColumn[]) {
  const categorical = columns.filter((c) => !isNumericDtype(c.dtype));
  const numeric = columns.filter((c) => isNumericDtype(c.dtype));
  const index = categorical.slice(0, 1).map((c) => c.column);
  const columnsField = categorical.find((c) => !index.includes(c.column))?.column ?? "";
  const values = numeric[0]?.column ?? "";
  return {
    index,
    columns: columnsField,
    values,
    aggregation: values ? "sum" : "first",
  };
}

/** 逆透视默认建议：标识列取首个非数值字段，值列优先取数值字段。 */
export function suggestMelt(columns: SchemaColumn[]) {
  const categorical = columns.filter((c) => !isNumericDtype(c.dtype));
  const numeric = columns.filter((c) => isNumericDtype(c.dtype));
  const idVars = categorical.slice(0, 1).map((c) => c.column);
  const valueVars = (numeric.length ? numeric : columns)
    .map((c) => c.column)
    .filter((c) => !idVars.includes(c));
  return {
    id_vars: idVars,
    value_vars: valueVars,
    variable_name: "variable",
    value_name: "value",
  };
}

/** 深拷贝一份表单默认值 —— 直接把常量对象塞进 state 会被后续编辑就地改写。 */
export function cloneForm(form: FormState): FormState {
  return JSON.parse(JSON.stringify(form)) as FormState;
}
