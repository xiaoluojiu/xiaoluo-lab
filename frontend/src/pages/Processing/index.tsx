import { useEffect, useMemo, useState } from "react";
import { PageHeader } from "../../components/PageHeader";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { MergePanel } from "../../features/merge/MergePanel";
import { MultiMergePanel } from "../../features/merge/MultiMergePanel";
import { getSchema } from "../../api/datasets";
import type { SchemaColumn } from "../../types/dataset";
import {
  getProcessingHistory,
  listOperations,
  previewProcessing,
  runProcessing,
  OPERATION_KIND_MAP,
  type OperationMeta,
  type OperationHistoryItem,
  type OperationPreview,
  type OperationResult,
} from "../../api/processing";

type Tab = "clean" | "duplicate" | "cast" | "string" | "filter" | "transform" | "aggregate" | "pivot" | "melt" | "merge" | "multi_merge";
/** 需要走 OperationPanel 的表单型 Tab。 */
type FormTab = Exclude<Tab, "merge" | "multi_merge">;
type FormState = Record<string, unknown>;
const TABS: Array<{ key: Tab; label: string; description: string; group: string }> = [
  { key: "clean", label: "缺失值", description: "删除或填充缺失值", group: "清洗" }, { key: "duplicate", label: "重复值", description: "按字段去除重复记录", group: "清洗" }, { key: "cast", label: "类型转换", description: "转换字段数据类型", group: "清洗" }, { key: "string", label: "字符串", description: "清理或替换文本字段", group: "清洗" },
  { key: "filter", label: "筛选", description: "按条件保留目标行", group: "行处理" }, { key: "transform", label: "字段转换", description: "计算新字段或修改字段", group: "字段处理" }, { key: "aggregate", label: "聚合", description: "按字段分组统计", group: "聚合与汇总" }, { key: "pivot", label: "透视", description: "长表转宽表", group: "表结构" }, { key: "melt", label: "逆透视", description: "宽表转长表", group: "表结构" },   { key: "merge", label: "合并", description: "连接两个数据集", group: "数据集合" }, { key: "multi_merge", label: "多文件合并", description: "勾选多个数据集，对齐共有/独有字段整合为一张表", group: "数据集合" },
];
const DEFAULT_FORMS: Record<FormTab, FormState> = { clean: { strategy: "mean", columns: [], value: 0 }, duplicate: { subset: [], keep: "first" }, cast: { types: {}, formats: {} }, string: { column: "", op: "trim", params: {} }, filter: { conditions: [{ column: "", op: "gte", value: "" }], logic: "and" }, transform: { name: "", expression: { type: "binary", left: "", operator: "*", right: 1 }, overwrite: false }, aggregate: { group_by: [], aggregations: [{ column: "", func: "sum", alias: "" }] }, pivot: { index: [], columns: "", values: "", aggregation: "first" }, melt: { id_vars: [], value_vars: [], variable_name: "variable", value_name: "value" } };
// kind -> op_type 的映射由 api/processing.ts 提供（OPERATION_KIND_MAP），此处不再复制常量。

function isNumericDtype(dtype: string) {
  return /int|float|double|decimal|number/i.test(dtype);
}

/** 透视默认建议：行索引取首个非数值字段，列维度取另一个非数值字段，值取首个数值字段。 */
function suggestPivot(columns: SchemaColumn[]) {
  const categorical = columns.filter((c) => !isNumericDtype(c.dtype));
  const numeric = columns.filter((c) => isNumericDtype(c.dtype));
  const index = categorical.slice(0, 1).map((c) => c.column);
  const columnsField =
    categorical.find((c) => !index.includes(c.column))?.column ?? "";
  const values = numeric[0]?.column ?? "";
  return {
    index,
    columns: columnsField,
    values,
    aggregation: values ? "sum" : "first",
  };
}

/** 逆透视默认建议：标识列取首个非数值字段，值列优先取数值字段。 */
function suggestMelt(columns: SchemaColumn[]) {
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

export default function Processing() {
  const [tab, setTab] = useState<Tab>("clean");
  const [datasetIds, setDatasetIds] = useState<number[]>([]);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [inputVersion, setInputVersion] = useState<number | undefined>();
  const [loadingOperations, setLoadingOperations] = useState(true);
  const [operationsError, setOperationsError] = useState<string | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [operations, setOperations] = useState<OperationMeta[]>([]);
  const selectedDatasetId = datasetIds[0];

  useEffect(() => { if (!selectedDatasetId) { setColumns([]); return; } getSchema(selectedDatasetId).then((s) => setColumns(s.columns)).catch(() => setColumns([])); }, [selectedDatasetId]);
  useEffect(() => { let cancelled = false; async function loadOperations() { setLoadingOperations(true); setOperationsError(null); try { setOperations(await listOperations()); } catch (error) { if (!cancelled) setOperationsError(error instanceof Error ? error.message : "加载操作目录失败"); } finally { if (!cancelled) setLoadingOperations(false); } } void loadOperations(); return () => { cancelled = true; }; }, []);
  useEffect(() => { setInputVersion(undefined); }, [datasetIds]);

  const tabGroups = Array.from(new Set(TABS.map((item) => item.group)));

  return (
    <div>
      <PageHeader
        breadcrumbs={<>数据中心 / <b>数据处理</b></>}
        title="数据处理"
        description="对数据集执行清洗、转换与聚合，每次执行都会生成一个可回溯的新版本。"
      />
      <div className="processing-workspace">
        <div className="processing-toolbar">
          <div className="processing-toolbar-main"><span>数据集</span><DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} requireVersion /></div>
          {selectedDatasetId && <label className="field processing-toolbar-version">输入版本<input type="number" min={1} value={inputVersion ?? ""} placeholder="留空 = 最新版本" onChange={(event) => { const value = event.target.value; if (!value) { setInputVersion(undefined); return; } const parsed = Number(value); if (Number.isInteger(parsed) && parsed > 0) setInputVersion(parsed); }} /></label>}
        </div>
        <div className="processing-tabs">{tabGroups.map((group) => <div className="processing-tabs-group" key={group}><span className="processing-tabs-label">{group}</span><div className="processing-tabs-items">{TABS.filter((item) => item.group === group).map((item) => <button key={item.key} type="button" title={item.description} className={`processing-tab${tab === item.key ? " active" : ""}`} onClick={() => setTab(item.key)}>{item.label}</button>)}</div></div>)}</div>
        <div className="processing-body">
          {loadingOperations && <div className="muted">正在加载操作…</div>}
          {operationsError && <div style={{ color: "var(--danger)" }}>操作目录加载失败：{operationsError}</div>}
          {!loadingOperations && !operationsError && (tab === "merge" ? <MergePanel /> : tab === "multi_merge" ? <MultiMergePanel /> : <OperationPanel key={tab} tab={tab as FormTab} meta={operations.find((item) => item.op_type === OPERATION_KIND_MAP[tab])} datasetId={selectedDatasetId} columns={columns} inputVersion={inputVersion} onCompleted={() => setHistoryRefresh((value) => value + 1)} />)}
        </div>
      </div>
      {selectedDatasetId && <OperationHistory key={historyRefresh} datasetId={selectedDatasetId} />}
    </div>
  );
}

function OperationPanel({ tab, meta, datasetId, columns, inputVersion, onCompleted }: { tab: FormTab; meta?: OperationMeta; datasetId?: number; columns: SchemaColumn[]; inputVersion?: number; onCompleted: () => void; }) {
  const [form, setForm] = useState<FormState>(() => cloneForm(DEFAULT_FORMS[tab]));
  const [advancedMode, setAdvancedMode] = useState(false);
  const [jsonText, setJsonText] = useState(() => JSON.stringify(DEFAULT_FORMS[tab], null, 2));
  const [preview, setPreview] = useState<OperationPreview | null>(null);
  const [result, setResult] = useState<OperationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyAction, setBusyAction] = useState<"preview" | "execute" | null>(null);
  useEffect(() => { setForm(cloneForm(DEFAULT_FORMS[tab])); setJsonText(JSON.stringify(DEFAULT_FORMS[tab], null, 2)); setPreview(null); setResult(null); setError(null); }, [tab]);
  // 透视 / 逆透视：用户尚未选择任何字段时，按当前 schema 智能预填，避免点「预览」时参数为空而报错。
  useEffect(() => {
    if (tab !== "pivot" && tab !== "melt") return;
    if (!columns.length) return;
    setForm((current) => {
      if (tab === "pivot") {
        const hasChoice = (Array.isArray(current.index) && current.index.length > 0) || current.columns || current.values;
        if (hasChoice) return current;
        return { ...current, ...suggestPivot(columns) };
      }
      const hasChoice = (Array.isArray(current.id_vars) && current.id_vars.length > 0)
        || (Array.isArray(current.value_vars) && current.value_vars.length > 0);
      if (hasChoice) return current;
      return { ...current, ...suggestMelt(columns) };
    });
  }, [tab, columns]);
  const params = useMemo(() => { if (!advancedMode) return form; try { return JSON.parse(jsonText) as Record<string, unknown>; } catch { return null; } }, [advancedMode, form, jsonText]);
  function updateField(key: string, value: unknown) { setForm((current) => ({ ...current, [key]: value })); }
  function resetForm() { const next = cloneForm(DEFAULT_FORMS[tab]); setForm(next); setJsonText(JSON.stringify(next, null, 2)); setPreview(null); setResult(null); setError(null); }
  function switchAdvancedMode(enabled: boolean) { if (enabled) setJsonText(JSON.stringify(form, null, 2)); else { try { setForm(JSON.parse(jsonText) as Record<string, unknown>); } catch { setError("当前 JSON 不合法，无法切换回表单模式"); return; } } setAdvancedMode(enabled); setError(null); }
  function validateParams(): Record<string, unknown> | null {
    if (!params) { setError("参数 JSON 格式错误，请检查输入"); return null; }
    if (tab === "clean") { const strategy = params.strategy; if (typeof strategy !== "string" || !strategy) { setError("清洗操作必须选择缺失值处理策略"); return null; } if (strategy === "constant" && params.value === undefined) { setError("constant 策略必须填写替换值"); return null; } }
    if (tab === "filter") { const raw = params.conditions; if (!Array.isArray(raw) || raw.length === 0) { setError("筛选操作至少需要添加一个条件"); return null; } const conditions = raw.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && Boolean((item as Record<string, unknown>).column)).map((item) => ({ column: item.column, op: (item.op as string) ?? "gte", value: item.op === "is_null" ? null : item.value, })); if (conditions.length === 0) { setError("请为至少一个条件选择待筛选的字段"); return null; } return { ...params, conditions }; }
    if (tab === "pivot") { const index = params.index; if (!Array.isArray(index) || index.length === 0) { setError("透视操作请先选择「行索引」字段（可多选）"); return null; } if (!params.columns) { setError("透视操作请先选择「列维度」字段（会展开成列）"); return null; } if (!params.values) { setError("透视操作请先选择「值」字段（用于填充单元格）"); return null; } }
    if (tab === "melt") { const idVars = params.id_vars; const valueVars = params.value_vars; if ((!Array.isArray(idVars) || idVars.length === 0) && (!Array.isArray(valueVars) || valueVars.length === 0)) { setError("逆透视操作请至少选择「标识字段」（保持不动）或「值字段」（融化为 variable/value）"); return null; } }
    if (tab === "transform") { if (!params.name || typeof params.name !== "string") { setError("转换操作必须填写新字段名称"); return null; } if (!params.expression || typeof params.expression !== "object") { setError("转换操作必须填写结构化 expression"); return null; } }
    if (tab === "aggregate") { if (!Array.isArray(params.group_by)) { setError("聚合操作的 group_by 必须是数组"); return null; } if (!Array.isArray(params.aggregations) || params.aggregations.length === 0) { setError("聚合操作至少需要一个聚合字段"); return null; } }
    return params;
  }

  async function handlePreview() {
    setError(null); setResult(null); setPreview(null);
    if (!datasetId) { setError("请先选择数据集"); return; }
    const validated = validateParams(); if (!validated) return;
    setBusy(true); setBusyAction("preview");
    try {
      // previewProcessing(datasetId, kind, params, inputVersion?, pageSize?) —— 位置参数，不能传对象。
      const data = await previewProcessing(datasetId, tab, validated, inputVersion);
      setPreview(data);
    }
    catch (e) { setError(e instanceof Error ? e.message : "预览失败"); }
    finally { setBusy(false); setBusyAction(null); }
  }
  async function handleExecute() {
    setError(null); setResult(null);
    if (!datasetId) { setError("请先选择数据集"); return; }
    const validated = validateParams(); if (!validated) return;
    setBusy(true); setBusyAction("execute");
    try {
      // runProcessing(datasetId, kind, params, inputVersion?) —— 位置参数，不能传对象。
      const data = await runProcessing(datasetId, tab, validated, inputVersion);
      setResult(data);
      onCompleted();
    }
    catch (e) { setError(e instanceof Error ? e.message : "执行失败"); }
    finally { setBusy(false); setBusyAction(null); }
  }

  return <div className="processing-operation-panel">
    <div className="processing-config-header">
      <div>
        <h2>{TABS.find((item) => item.key === tab)?.label}</h2>
        <span className="muted op-catalog-desc">{meta?.description ?? TABS.find((item) => item.key === tab)?.description}</span>
      </div>
      <div className="processing-config-actions">
        {meta?.risk && <span className={`badge op-risk op-risk-${meta.risk}`}>风险：{meta.risk}</span>}
        <button className="btn" type="button" onClick={resetForm}>重置</button><button className={`btn ${advancedMode ? "primary" : ""}`} type="button" onClick={() => switchAdvancedMode(!advancedMode)}>{advancedMode ? "表单模式" : "高级 JSON"}</button></div></div>
    <div className="processing-config-grid"><div className="processing-config-main">{advancedMode ? <textarea className="processing-json-editor" value={jsonText} onChange={(event) => setJsonText(event.target.value)} spellCheck={false} /> : <OperationForm tab={tab} form={form} columns={columns} onChange={updateField} />}</div><div className="processing-preview"><div className="processing-preview-header"><strong>预览</strong></div>{preview ? <OperationPreviewView preview={preview} /> : result ? <OperationResultView result={result} /> : <div className="processing-empty">选择参数后运行预览。</div>}</div></div>
    {error && <div className="processing-error">{error}</div>}
    <div className="processing-footer-actions"><button className="btn" type="button" onClick={() => void handlePreview()} disabled={busy}>{busyAction === "preview" ? "预览中…" : "预览"}</button><button className="btn primary" type="button" onClick={() => void handleExecute()} disabled={busy}>{busyAction === "execute" ? "执行中…" : "执行并生成版本"}</button></div>
  </div>;
}

function OperationForm({ tab, form, columns, onChange }: { tab: FormTab; form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void; }) {
  switch (tab) {
    case "clean": return <CleanForm form={form} columns={columns} onChange={onChange} />;
    case "duplicate": return <DuplicateForm form={form} columns={columns} onChange={onChange} />;
    case "cast": return <CastForm form={form} columns={columns} onChange={onChange} />;
    case "string": return <StringForm form={form} columns={columns} onChange={onChange} />;
    case "filter": return <FilterForm form={form} columns={columns} onChange={onChange} />;
    case "transform": return <TransformForm form={form} columns={columns} onChange={onChange} />;
    case "aggregate": return <AggregateForm form={form} columns={columns} onChange={onChange} />;
    case "pivot": return <PivotForm form={form} columns={columns} onChange={onChange} />;
    case "melt": return <MeltForm form={form} columns={columns} onChange={onChange} />;
  }
}

function CleanForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const selected = Array.isArray(form.columns) ? form.columns as string[] : []; return <div className="processing-form-fields"><label className="field">处理策略<select value={String(form.strategy ?? "mean")} onChange={(e) => onChange("strategy", e.target.value)}><option value="drop">删除缺失行</option><option value="mean">均值填充</option><option value="median">中位数填充</option><option value="mode">众数填充</option><option value="constant">固定值填充</option></select></label><label className="field">字段范围<select multiple value={selected} onChange={(e) => onChange("columns", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label>{form.strategy === "constant" && <label className="field">填充值<input value={String(form.value ?? "")} onChange={(e) => onChange("value", e.target.value)} /></label>}</div>; }
function DuplicateForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const subset = Array.isArray(form.subset) ? form.subset as string[] : []; return <div className="processing-form-fields"><label className="field">依据字段<select multiple value={subset} onChange={(e) => onChange("subset", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">保留规则<select value={String(form.keep ?? "first")} onChange={(e) => onChange("keep", e.target.value)}><option value="first">第一条</option><option value="last">最后一条</option><option value="false">全部删除</option></select></label></div>; }
function CastForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const types = (form.types ?? {}) as Record<string, string>; return <div className="processing-form-fields">{columns.map((column) => <label className="field processing-type-row" key={column.column}><span>{column.column}</span><select value={types[column.column] ?? column.dtype} onChange={(e) => onChange("types", { ...types, [column.column]: e.target.value })}><option value="int">整数</option><option value="float">小数</option><option value="str">文本</option><option value="bool">布尔</option><option value="datetime">日期时间</option></select></label>)}</div>; }
function StringForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const params = (form.params ?? {}) as Record<string, unknown>; return <div className="processing-form-fields"><label className="field">文本字段<select value={String(form.column ?? "")} onChange={(e) => onChange("column", e.target.value)}><option value="">请选择</option>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">操作<select value={String(form.op ?? "trim")} onChange={(e) => onChange("op", e.target.value)}><option value="trim">去除首尾空格</option><option value="lower">转小写</option><option value="upper">转大写</option><option value="replace">替换文本</option></select></label>{form.op === "replace" && <><label className="field">查找文本<input value={String(params.find ?? "")} onChange={(e) => onChange("params", { ...params, find: e.target.value })} /></label><label className="field">替换为<input value={String(params.replace ?? "")} onChange={(e) => onChange("params", { ...params, replace: e.target.value })} /></label></>}</div>; }
/**
 * 筛选表单（重设计）：算子与取值输入随所选字段的类型自适应。
 * - 数值字段：大小比较算子 + 数字输入
 * - 文本字段：包含 / 等于 / 属于集合
 * - 时间字段：大小比较 + 日期/时间输入（后端支持多格式解析）
 * - 布尔字段：等于 true/false
 * - 为空：不需要取值，隐藏输入框
 * - 全为空字段的占位条件在本地就会被剔除，不再把无意义条件发给后端
 */
const FILTER_OPS: Record<string, Array<{ value: string; label: string }>> = {
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

function dtypeKind(dtype: string): keyof typeof FILTER_OPS {
  const t = dtype.toLowerCase();
  if (/int|float|double|decimal|number/.test(t)) return "numeric";
  if (/date|time/.test(t)) return "temporal";
  if (/bool/.test(t)) return "boolean";
  if (/str|utf8|object|categorical/.test(t)) return "text";
  return "unknown";
}

function FilterForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) {
  const conditions = Array.isArray(form.conditions) ? form.conditions as Array<Record<string, unknown>> : [];
  function update(index: number, patch: Record<string, unknown>) { onChange("conditions", conditions.map((item, i) => i === index ? { ...item, ...patch } : item)); }
  return <div className="processing-form-fields">
    <label className="field">条件关系<select value={String(form.logic ?? "and")} onChange={(e) => onChange("logic", e.target.value)}><option value="and">全部满足</option><option value="or">任一满足</option></select></label>
    {conditions.map((condition, index) => {
      const column = String(condition.column ?? "");
      const shape = columns.find((c) => c.column === column);
      const kind = shape ? dtypeKind(shape.dtype) : "unknown";
      const ops = FILTER_OPS[kind];
      const op = String(condition.op ?? "gte");
      const needsValue = op !== "is_null";
      return (
        <div className="processing-condition filter-condition" key={index}>
          <select value={column} onChange={(e) => update(index, { column: e.target.value, op: FILTER_OPS[dtypeKind(columns.find((c) => c.column === e.target.value)?.dtype ?? "")][0].value })}>
            <option value="">字段（可选）</option>
            {columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}
          </select>
          <select value={ops.some((o) => o.value === op) ? op : ops[0].value} onChange={(e) => update(index, { op: e.target.value })}>
            {ops.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          {needsValue && (kind === "boolean" ? (
            <input value={String(condition.value ?? "")} placeholder="true / false" onChange={(e) => update(index, { value: e.target.value })} />
          ) : kind === "numeric" ? (
            <input type="number" value={condition.value === undefined || condition.value === "" ? "" : Number(condition.value)} onChange={(e) => update(index, { value: e.target.value === "" ? "" : Number(e.target.value) })} />
          ) : kind === "temporal" ? (
            <input type="date" value={String(condition.value ?? "")} onChange={(e) => update(index, { value: e.target.value })} />
          ) : (
            <input value={String(condition.value ?? "")} placeholder={op === "in" ? "多个值用逗号分隔" : "取值"} onChange={(e) => update(index, { value: e.target.value })} />
          ))}
          {!needsValue && <span className="muted filter-null-note">无需取值</span>}
          <button className="btn" type="button" onClick={() => onChange("conditions", conditions.filter((_, i) => i !== index))}>删除</button>
          {shape && <span className="filter-dtype">{shape.dtype}</span>}
        </div>
      );
    })}
    <button className="btn" type="button" onClick={() => onChange("conditions", [...conditions, { column: "", op: "gte", value: "" }])}>+ 添加条件</button>
  </div>;
}
function TransformForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const expression = (form.expression ?? {}) as Record<string, unknown>; return <div className="processing-form-fields"><label className="field">新字段名称<input value={String(form.name ?? "")} onChange={(e) => onChange("name", e.target.value)} /></label><label className="field">左值<select value={String(expression.left ?? "")} onChange={(e) => onChange("expression", { ...expression, left: e.target.value })}><option value="">请选择字段</option>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">运算符<select value={String(expression.operator ?? "*")} onChange={(e) => onChange("expression", { ...expression, operator: e.target.value })}><option value="+">+</option><option value="-">−</option><option value="*">×</option><option value="/">÷</option></select></label><label className="field">右值<input type="number" value={Number(expression.right ?? 1)} onChange={(e) => onChange("expression", { ...expression, right: Number(e.target.value) })} /></label><label className="field"><span><input type="checkbox" checked={Boolean(form.overwrite)} onChange={(e) => onChange("overwrite", e.target.checked)} /> 覆盖同名字段</span></label></div>; }
function AggregateForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const groupBy = Array.isArray(form.group_by) ? form.group_by as string[] : []; const aggregations = Array.isArray(form.aggregations) ? form.aggregations as Array<Record<string, unknown>> : []; function updateAggregation(index: number, patch: Record<string, unknown>) { onChange("aggregations", aggregations.map((item, i) => i === index ? { ...item, ...patch } : item)); } return <div className="processing-form-fields"><label className="field">分组字段<select multiple value={groupBy} onChange={(e) => onChange("group_by", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label>{aggregations.map((item, index) => <div className="processing-condition" key={index}><select value={String(item.column ?? "")} onChange={(e) => updateAggregation(index, { column: e.target.value })}><option value="">字段</option>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select><select value={String(item.func ?? "sum")} onChange={(e) => updateAggregation(index, { func: e.target.value })}><option value="sum">求和</option><option value="mean">均值</option><option value="count">计数</option><option value="min">最小值</option><option value="max">最大值</option></select><input placeholder="别名（可选）" value={String(item.alias ?? "")} onChange={(e) => updateAggregation(index, { alias: e.target.value })} /><button className="btn" type="button" onClick={() => onChange("aggregations", aggregations.filter((_, i) => i !== index))}>删除</button></div>)}<button className="btn" type="button" onClick={() => onChange("aggregations", [...aggregations, { column: "", func: "sum", alias: "" }])}>+ 添加聚合</button></div>; }
function PivotForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const index = Array.isArray(form.index) ? form.index as string[] : []; return <div className="processing-form-fields"><label className="field">行索引<select multiple value={index} onChange={(e) => onChange("index", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">列字段<select value={String(form.columns ?? "")} onChange={(e) => onChange("columns", e.target.value)}><option value="">请选择</option>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">值字段<select value={String(form.values ?? "")} onChange={(e) => onChange("values", e.target.value)}><option value="">请选择</option>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">聚合方式<select value={String(form.aggregation ?? "first")} onChange={(e) => onChange("aggregation", e.target.value)}><option value="first">首个</option><option value="sum">求和</option><option value="mean">均值</option><option value="count">计数</option></select></label></div>; }
function MeltForm({ form, columns, onChange }: { form: FormState; columns: SchemaColumn[]; onChange: (key: string, value: unknown) => void }) { const idVars = Array.isArray(form.id_vars) ? form.id_vars as string[] : []; const valueVars = Array.isArray(form.value_vars) ? form.value_vars as string[] : []; return <div className="processing-form-fields"><label className="field">标识字段<select multiple value={idVars} onChange={(e) => onChange("id_vars", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">值字段<select multiple value={valueVars} onChange={(e) => onChange("value_vars", Array.from(e.target.selectedOptions).map((o) => o.value))}>{columns.map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}</select></label><label className="field">变量名称<input value={String(form.variable_name ?? "variable")} onChange={(e) => onChange("variable_name", e.target.value)} /></label><label className="field">值名称<input value={String(form.value_name ?? "value")} onChange={(e) => onChange("value_name", e.target.value)} /></label></div>; }

/**
 * 预览面板。
 * 字段契约以后端 OperationPreview 为准：
 *   rows    -> preview.preview.items（不是 preview_rows）
 *   columns -> preview.preview.columns
 *   总行数   -> preview.after.rows（after 是执行后的预计形态）
 */
function OperationPreviewView({ preview }: { preview: OperationPreview }) {
  const rows = preview.preview.items ?? [];
  const columns = preview.preview.columns ?? [];
  const delta = formatDelta(preview.row_delta);
  return (
    <div>
      <div className="processing-result-summary">
        <span>预览行数</span>
        <strong>{rows.length}</strong>
        <span>预计总行数</span>
        <strong>{preview.after.rows}</strong>
        <span>行数变化</span>
        <strong>{delta}</strong>
        <span>列数变化</span>
        <strong>{formatDelta(preview.column_delta)}</strong>
      </div>
      <PreviewTable rows={rows} columns={columns} />
    </div>
  );
}

/** 执行结果。版本信息在 result.version 内，result.rows/columns 并不存在。 */
function OperationResultView({ result }: { result: OperationResult }) {
  return (
    <div className="processing-result-summary">
      <span>新版本</span>
      <strong>v{result.version.version}</strong>
      <span>行数</span>
      <strong>{result.version.row_count}</strong>
      <span>列数</span>
      <strong>{result.version.column_count}</strong>
      <span>操作状态</span>
      <strong>{result.operation.status}</strong>
    </div>
  );
}

function formatDelta(value: number) {
  if (!value) return "±0";
  return value > 0 ? `+${value}` : String(value);
}
function PreviewTable({ rows, columns }: { rows: Record<string, unknown>[]; columns: string[] }) {
  if (!rows.length) return <div className="processing-empty">暂无预览数据</div>;
  return (
    <div className="processing-table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** 处理记录。字段为 operation_type / input_version_id / output_version_id。 */
function OperationHistory({ datasetId }: { datasetId: number }) {
  const [items, setItems] = useState<OperationHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState<string | null>(null);

  function load() {
    setLoading(true);
    setFailed(null);
    getProcessingHistory(datasetId)
      .then(setItems)
      .catch((error: unknown) => {
        setItems([]);
        setFailed(error instanceof Error ? error.message : "加载处理记录失败");
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, [datasetId]);

  if (loading) return <div className="card" style={{ marginTop: "var(--space-4)" }}><div className="muted">正在加载处理记录…</div></div>;

  return (
    <section className="card" style={{ marginTop: "var(--space-4)" }}>
      <div className="section-title">
        <h3>处理记录</h3>
        <span className="muted">{items.length} 条</span>
      </div>
      {failed && <div className="processing-error">{failed} <button className="btn btn-sm" type="button" onClick={load}>重试</button></div>}
      {!failed && items.length === 0 && <div className="processing-empty">暂无处理记录</div>}
      {items.length > 0 && (
        <div className="processing-history-list">
          {items.map((item) => (
            <div className="processing-history-item" key={item.id}>
              <div>
                <strong>{item.operation_type}</strong>
                <div className="muted">
                  {versionLabel(item.input_version_id)} → {versionLabel(item.output_version_id)} · {item.status}
                </div>
                {item.error && <div className="muted">{item.error}</div>}
              </div>
              <div className="muted">{item.created_at ?? "—"}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function versionLabel(version: number | null) {
  return version === null ? "最新" : `v${version}`;
}
function cloneForm(form: FormState): FormState { return JSON.parse(JSON.stringify(form)) as FormState; }
function formatCell(value: unknown) { if (value === null || value === undefined) return "—"; if (typeof value === "object") return JSON.stringify(value); return String(value); }
