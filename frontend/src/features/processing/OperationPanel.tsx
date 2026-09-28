/**
 * 数据处理操作面板（原 `pages/Processing/index.tsx` 的主体）。
 *
 * 归并原因：这个面板连同 9 个操作表单、预览/结果视图原本整段内联在页面里，
 * 页面因此承载了业务逻辑（违反「pages 只做路由级布局与数据初始化」）。
 * 现在页面只负责选数据集 / 切 Tab，面板自己管表单、校验、dry-run 与落库。
 *
 * 行为与拆分前完全一致：请求接口、参数校验、Toast 文案、空/错/忙三态均未改动。
 */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useToast } from "../../components/ToastProvider";
import { SampleTable } from "../../components/SampleTable";
import {
  previewProcessing,
  runProcessing,
  type OperationMeta,
  type OperationPreview,
  type OperationResult,
} from "../../api/processing";
import type { SchemaColumn } from "../../types/dataset";
import { OperationPreviewDialog, PREVIEW_ROWS } from "../dataset/OperationPreviewDialog";
import {
  DEFAULT_FORMS,
  FILTER_OPS,
  TABS,
  cloneForm,
  dtypeKind,
  suggestMelt,
  suggestPivot,
  type FormState,
  type FormTab,
} from "./operationCatalog";

export function OperationPanel({ tab, meta, datasetId, columns, inputVersion, onCompleted }: { tab: FormTab; meta?: OperationMeta; datasetId?: number; columns: SchemaColumn[]; inputVersion?: number; onCompleted: () => void; }) {
  const toast = useToast();
  const [form, setForm] = useState<FormState>(() => cloneForm(DEFAULT_FORMS[tab]));
  const [advancedMode, setAdvancedMode] = useState(false);
  const [jsonText, setJsonText] = useState(() => JSON.stringify(DEFAULT_FORMS[tab], null, 2));
  const [preview, setPreview] = useState<OperationPreview | null>(null);
  /** 产生当前 preview 时的参数指纹，用于判断预览是否已过期。 */
  const [previewSignature, setPreviewSignature] = useState<string | null>(null);
  const [result, setResult] = useState<OperationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyAction, setBusyAction] = useState<"preview" | "impact" | "execute" | null>(null);
  /** 待确认的执行：dry-run 结果 + 已校验的参数。非 null 即代表确认弹窗打开。 */
  const [pending, setPending] = useState<{ preview: OperationPreview; params: Record<string, unknown> } | null>(null);
  /** 执行失败的原因，展示在确认弹窗内（不关弹窗，便于直接重试）。 */
  const [dialogError, setDialogError] = useState<string | null>(null);
  useEffect(() => { setForm(cloneForm(DEFAULT_FORMS[tab])); setJsonText(JSON.stringify(DEFAULT_FORMS[tab], null, 2)); setPreview(null); setPreviewSignature(null); setResult(null); setError(null); setPending(null); setDialogError(null); }, [tab]);
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
  /** 参数指纹：数据集 / 输入版本 / 操作 / 参数任一变化，旧预览即失效。 */
  const signature = useMemo(() => JSON.stringify({ tab, datasetId: datasetId ?? null, inputVersion: inputVersion ?? null, params }), [tab, datasetId, inputVersion, params]);
  const previewStale = Boolean(preview && previewSignature !== signature);
  function updateField(key: string, value: unknown) { setForm((current) => ({ ...current, [key]: value })); }
  function resetForm() { const next = cloneForm(DEFAULT_FORMS[tab]); setForm(next); setJsonText(JSON.stringify(next, null, 2)); setPreview(null); setPreviewSignature(null); setResult(null); setError(null); setPending(null); setDialogError(null); }
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
    setError(null); setResult(null); setPreview(null); setPreviewSignature(null);
    if (!datasetId) { setError("请先选择数据集"); return; }
    const validated = validateParams(); if (!validated) return;
    setBusy(true); setBusyAction("preview");
    try {
      // previewProcessing(datasetId, kind, params, inputVersion?, pageSize?) —— 位置参数，不能传对象。
      const data = await previewProcessing(datasetId, tab, validated, inputVersion, PREVIEW_ROWS);
      setPreview(data);
      setPreviewSignature(signature);
    }
    catch (e) { setError(e instanceof Error ? e.message : "预览失败"); }
    finally { setBusy(false); setBusyAction(null); }
  }

  /**
   * 点「执行」不再直接写数据：先拿 dry-run 预览，再让用户在弹窗里确认。
   * 预览失败就**不放行**——宁可让用户看到报错，也不要盲执行一次不可预料的改动。
   */
  async function requestExecute() {
    setError(null); setResult(null); setDialogError(null);
    if (!datasetId) { setError("请先选择数据集"); return; }
    const validated = validateParams(); if (!validated) return;
    // 参数没变且已有预览：直接复用，避免在同一张表上白算第二遍整表操作。
    if (preview && previewSignature === signature) { setPending({ preview, params: validated }); return; }
    setBusy(true); setBusyAction("impact");
    try {
      const data = await previewProcessing(datasetId, tab, validated, inputVersion, PREVIEW_ROWS);
      setPreview(data);
      setPreviewSignature(signature);
      setPending({ preview: data, params: validated });
    }
    catch (e) { setError(e instanceof Error ? e.message : "影响预览失败，已阻止执行"); }
    finally { setBusy(false); setBusyAction(null); }
  }

  /** 弹窗里「确认执行」之后才真正落库。 */
  async function runConfirmed() {
    if (!pending || !datasetId) return;
    setBusy(true); setBusyAction("execute"); setDialogError(null);
    try {
      // runProcessing(datasetId, kind, params, inputVersion?) —— 位置参数，不能传对象。
      const data = await runProcessing(datasetId, tab, pending.params, inputVersion);
      setResult(data);
      setPending(null);
      onCompleted();
      const version = data.version.version;
      toast.success(`已生成新版本 v${version}`, {
        // 带上 dataset / version：路由入口的 RouteContextSync 会把「当前数据集 + 当前版本」
        // 一起恢复，用户点进去看到的正是刚生成的那个版本。
        link: `/datasets/${datasetId}?dataset=${datasetId}&version=${version}&tab=versions`,
        label: "查看新版本",
      });
    }
    catch (e) {
      // 保留弹窗：参数与预览都还在，用户可以直接重试或取消。
      setDialogError(e instanceof Error ? e.message : "执行失败");
    }
    finally { setBusy(false); setBusyAction(null); }
  }

  function cancelPending() { setPending(null); setDialogError(null); }

  return <div className="processing-operation-panel">
    <div className="processing-config-header">
      <div>
        <h2>{TABS.find((item) => item.key === tab)?.label}</h2>
        <span className="muted op-catalog-desc">{meta?.description ?? TABS.find((item) => item.key === tab)?.description}</span>
      </div>
      <div className="processing-config-actions">
        {meta?.risk && <span className={`badge op-risk op-risk-${meta.risk}`}>风险：{meta.risk}</span>}
        <button className="btn" type="button" onClick={resetForm}>重置</button><button className={`btn ${advancedMode ? "primary" : ""}`} type="button" onClick={() => switchAdvancedMode(!advancedMode)}>{advancedMode ? "表单模式" : "高级 JSON"}</button></div></div>
    <div className="processing-config-grid"><div className="processing-config-main">{advancedMode ? <textarea className="processing-json-editor" value={jsonText} onChange={(event) => setJsonText(event.target.value)} spellCheck={false} /> : <OperationForm tab={tab} form={form} columns={columns} onChange={updateField} />}</div><div className="processing-preview"><div className="processing-preview-header"><strong>预览</strong>{previewStale && <span className="badge warning" title="参数已修改，此预览不再对应当前参数；执行时会重新计算">已过期</span>}</div>{preview ? <OperationPreviewView preview={preview} /> : result ? <OperationResultView result={result} datasetId={datasetId} /> : <div className="processing-empty">选择参数后运行预览。</div>}</div></div>
    {error && <div className="processing-error">{error}</div>}
    <div className="muted">执行前会先展示这次操作的影响预览；删除列或删除大量行需要二次确认。</div>
    <div className="processing-footer-actions"><button className="btn" type="button" onClick={() => void handlePreview()} disabled={busy}>{busyAction === "preview" ? "预览中…" : "预览"}</button><button className="btn primary" type="button" onClick={() => void requestExecute()} disabled={busy}>{busyAction === "impact" ? "计算影响中…" : "执行并生成版本"}</button></div>
    {pending && (
      <OperationPreviewDialog
        open
        title={TABS.find((item) => item.key === tab)?.label ?? "数据处理"}
        preview={pending.preview}
        error={dialogError}
        executing={busyAction === "execute"}
        onCancel={cancelPending}
        onConfirm={() => void runConfirmed()}
      />
    )}
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
 * - 全为空字段的占位条件在本地就会被剔除，不再把无意义条件发给后端
 */
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
 * 表格本体复用共享 SampleTable（历史差异：空态用 processing-empty 空态块）。
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
      <SampleTable rows={rows} columns={columns} />
    </div>
  );
}

/** 执行结果。版本信息在 result.version 内，result.rows/columns 并不存在。 */
function OperationResultView({ result, datasetId }: { result: OperationResult; datasetId?: number }) {
  return (
    <div>
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
      {/* Toast 会自动消失，这里再留一个常驻入口，避免用户错过新版本。 */}
      {datasetId ? (
        <Link
          className="btn link"
          to={`/datasets/${datasetId}?dataset=${datasetId}&version=${result.version.version}&tab=versions`}
        >
          查看新版本 v{result.version.version}
        </Link>
      ) : null}
    </div>
  );
}

function formatDelta(value: number) {
  if (!value) return "±0";
  return value > 0 ? `+${value}` : String(value);
}
