/**
 * 结构化参数表单（替代原来的裸 JSON 编辑框）。
 *
 * 数据来源：
 * - 字段清单 / 类型 / 枚举 / 默认值 / 必填 / 条件依赖 → `nodeSpecs.ts`；
 * - 节点是否可运行、上游有哪些列 → 调用方传入（schema 来自数据集接口）。
 *
 * 保留「高级 JSON」折叠入口：结构化表单覆盖不到的表达（如 cast 的 types 映射）
 * 仍然走 JSON，避免把能力砍掉。
 */

import { useMemo, useState } from "react";

import { isContinuousNumeric, isNumericColumn, isTemporalColumn } from "../../lib/edaColumns";
import type { SchemaColumn } from "../../types/dataset";
import { AGGREGATION_FUNCS, type ParamSpec } from "./nodeSpecs";
import { flattenConfig } from "./configView";
// 自带样式：沉浸式编辑器不在 .content 内，拿不到 design-foundation 的控件兜底规则。
import "./params.css";

interface Props {
  specs: ParamSpec[];
  config: Record<string, unknown>;
  columns: SchemaColumn[];
  datasetOptions?: Array<{ id: number; name: string }>;
  onChange: (next: Record<string, unknown>) => void;
}

/** 按语义过滤列，供 columns 类型参数使用。 */
function filterColumns(columns: SchemaColumn[], filter?: ParamSpec["columnFilter"]): string[] {
  if (!filter || filter === "any") return columns.map((c) => c.column);
  if (filter === "numeric") return columns.filter((c) => isContinuousNumeric(c)).map((c) => c.column);
  if (filter === "categorical") {
    return columns.filter((c) => !isNumericColumn(c) && !isTemporalColumn(c)).map((c) => c.column);
  }
  return columns.filter((c) => isTemporalColumn(c)).map((c) => c.column);
}

/**
 * 多选列控件。
 *
 * 不用原生 multiple select：它在窄面板里几乎不可用（要 Ctrl 点选、无搜索）。
 * 这里做成可勾选的 chip 列表 + 已选计数。
 */
function ColumnsField({
  value,
  options,
  onChange,
}: {
  value: string[];
  options: string[];
  onChange: (next: string[]) => void;
}) {
  const [keyword, setKeyword] = useState("");
  const shown = useMemo(
    () => options.filter((name) => name.toLowerCase().includes(keyword.trim().toLowerCase())),
    [options, keyword],
  );
  const selected = new Set(value);

  if (!options.length) {
    return <p className="wf-param-empty">上游还没有可用列。请先配置并运行「加载数据」节点，或在上方选择数据集。</p>;
  }

  return (
    <div className="wf-columns-field">
      {options.length > 8 && (
        <input
          className="wf-param-input"
          value={keyword}
          onChange={(event) => setKeyword(event.target.value)}
          placeholder="搜索列名"
        />
      )}
      <div className="wf-columns-list">
        {shown.map((name) => (
          <button
            key={name}
            type="button"
            className={`wf-column-chip${selected.has(name) ? " active" : ""}`}
            aria-pressed={selected.has(name)}
            onClick={() => {
              const next = new Set(selected);
              if (next.has(name)) next.delete(name);
              else next.add(name);
              // 保持数据集原始列序，避免勾选顺序影响下游表达式的可比性。
              onChange(options.filter((option) => next.has(option)));
            }}
          >
            {name}
          </button>
        ))}
        {!shown.length && <p className="wf-param-empty">没有匹配「{keyword}」的列。</p>}
      </div>
      {value.length > 0 && (
        <button type="button" className="wf-columns-clear" onClick={() => onChange([])}>
          已选 {value.length} 列 · 清空
        </button>
      )}
    </div>
  );
}

function JsonField({
  value,
  placeholder,
  onChange,
}: {
  value: unknown;
  placeholder?: string;
  onChange: (next: unknown) => void;
}) {
  const [draft, setDraft] = useState(() => JSON.stringify(value ?? null, null, 2));
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="wf-param-json">
      <textarea
        className="wf-param-textarea"
        rows={4}
        value={draft}
        placeholder={placeholder}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          if (!text.trim()) {
            // 清空即视为「不设置该参数」，而不是写入 null。
            setError(null);
            onChange(undefined);
            return;
          }
          try {
            onChange(JSON.parse(text));
            setError(null);
          } catch {
            // 输入未完成时不覆盖真实 config，仅提示。
            setError("JSON 还没写完，暂未生效");
          }
        }}
      />
      {error && <p className="wf-param-error">{error}</p>}
    </div>
  );
}

/* ---------- enum 多选（固定选项，chips；产出 string[]） ---------- */
function MultiOptionsField({
  value, options, onChange,
}: {
  value: string[];
  options: NonNullable<ParamSpec["options"]>;
  onChange: (next: string[]) => void;
}) {
  const selected = new Set(value);
  return (
    <div className="wf-columns-list">
      {options.map((option) => {
        const active = selected.has(option.value);
        return (
          <button
            key={option.value}
            type="button"
            className={`wf-column-chip${active ? " active" : ""}`}
            aria-pressed={active}
            onClick={() => {
              const next = new Set(selected);
              if (active) next.delete(option.value);
              else next.add(option.value);
              onChange(options.filter((o) => next.has(o.value)).map((o) => o.value));
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------- 条件构造器（data.filter；产出 [{column, op, value}]） ---------- */
interface FilterConditionRow { column: string; op: string; value: string; }

const FILTER_OPERATORS: NonNullable<ParamSpec["options"]> = [
  { value: "eq", label: "等于" },
  { value: "neq", label: "不等于" },
  { value: "gt", label: "大于" },
  { value: "gte", label: "大于等于" },
  { value: "lt", label: "小于" },
  { value: "lte", label: "小于等于" },
  { value: "contains", label: "包含子串" },
  { value: "in", label: "在集合中" },
  { value: "is_null", label: "为空" },
];

const EMPTY_CONDITION: FilterConditionRow = { column: "", op: "eq", value: "" };

function ConditionsField({
  value, options, onChange,
}: {
  value: unknown;
  options: string[];
  onChange: (next: FilterConditionRow[]) => void;
}) {
  const rows: FilterConditionRow[] =
    Array.isArray(value) && value.length ? (value as FilterConditionRow[]) : [{ ...EMPTY_CONDITION }];
  const patch = (index: number, next: Partial<FilterConditionRow>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...next } : row)));

  return (
    <div className="wf-builder">
      {rows.map((row, index) => {
        const noValue = row.op === "is_null";
        return (
          <div className={`wf-builder-row${noValue ? " no-value" : ""}`} key={index}>
            <select
              className="wf-param-input" aria-label="选择列" value={row.column}
              onChange={(e) => patch(index, { column: e.target.value })}
            >
              <option value="">（选择列）</option>
              {options.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
            <select
              className="wf-param-input" aria-label="选择操作符" value={row.op}
              onChange={(e) => patch(index, { op: e.target.value })}
            >
              {FILTER_OPERATORS.map((op) => <option key={op.value} value={op.value}>{op.label}</option>)}
            </select>
            {!noValue && (
              <input
                className="wf-param-input" aria-label="条件值" value={row.value ?? ""}
                placeholder={row.op === "in" ? "逗号分隔多个值" : "值"}
                onChange={(e) => patch(index, { value: e.target.value })}
              />
            )}
            <button
              type="button" className="wf-builder-remove" title="删除条件" aria-label="删除条件"
              onClick={() =>
                onChange(rows.length === 1 ? [{ ...EMPTY_CONDITION }] : rows.filter((_, i) => i !== index))
              }
            >
              ×
            </button>
          </div>
        );
      })}
      <button type="button" className="wf-builder-add"
        onClick={() => onChange([...rows, { ...EMPTY_CONDITION }])}>
        ＋ 添加条件
      </button>
    </div>
  );
}

/* ---------- 聚合构造器（data.aggregate；产出 [{column, func}]） ---------- */
interface AggregationRow { column: string; func: string; }

function AggregationsField({
  value, options, onChange,
}: {
  value: unknown;
  options: string[];
  onChange: (next: AggregationRow[]) => void;
}) {
  const rows: AggregationRow[] =
    Array.isArray(value) && value.length ? (value as AggregationRow[]) : [{ column: "", func: "mean" }];
  const patch = (index: number, next: Partial<AggregationRow>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...next } : row)));

  return (
    <div className="wf-builder">
      {rows.map((row, index) => (
        <div className="wf-builder-row agg" key={index}>
          <select
            className="wf-param-input" aria-label="选择聚合列" value={row.column}
            onChange={(e) => patch(index, { column: e.target.value })}
          >
            <option value="">（选择列；count 可留空）</option>
            {options.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
          <select
            className="wf-param-input" aria-label="选择聚合函数" value={row.func}
            onChange={(e) => patch(index, { func: e.target.value })}
          >
            {AGGREGATION_FUNCS.map((func) => <option key={func.value} value={func.value}>{func.label}</option>)}
          </select>
          <button
            type="button" className="wf-builder-remove" title="删除聚合" aria-label="删除聚合"
            onClick={() =>
              onChange(rows.length === 1 ? [{ column: "", func: "mean" }] : rows.filter((_, i) => i !== index))
            }
          >
            ×
          </button>
        </div>
      ))}
      <button type="button" className="wf-builder-add"
        onClick={() => onChange([...rows, { column: "", func: "mean" }])}>
        ＋ 添加聚合
      </button>
    </div>
  );
}

export function NodeParamForm({ specs, config, columns, datasetOptions = [], onChange }: Props) {
  const flat = useMemo(() => flattenConfig(config), [config]);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advancedDraft, setAdvancedDraft] = useState("");
  const [advancedError, setAdvancedError] = useState<string | null>(null);

  const setValue = (key: string, value: unknown) => {
    const next = { ...flat };
    if (value === undefined || value === "" || (Array.isArray(value) && !value.length)) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  };

  if (!specs.length) {
    return <p className="wf-param-empty">该节点没有可配置的参数。</p>;
  }

  return (
    <div className="wf-param-form">
      {specs.map((spec) => {
        // 条件依赖：不满足显示条件就不渲染（审查 #4）。
        if (spec.visibleWhen) {
          const gate = flat[spec.visibleWhen.key];
          if (gate !== spec.visibleWhen.equals) return null;
        }
        const value = flat[spec.key];
        const inputId = `wf-param-${spec.key}`;
        return (
          <div className="wf-param-row" key={spec.key}>
            <label htmlFor={inputId}>
              <span>
                {spec.label}
                {spec.required && <em className="wf-param-required" title="必填">*</em>}
              </span>
              {value == null && spec.default !== undefined && (
                <button
                  type="button"
                  className="wf-param-use-default"
                  onClick={() => setValue(spec.key, spec.default)}
                >
                  用默认值 {String(spec.default)}
                </button>
              )}
            </label>

            {spec.kind === "enum" && spec.multi && (
              <MultiOptionsField
                value={Array.isArray(value) ? (value as string[]) : []}
                options={spec.options ?? []}
                onChange={(next) => setValue(spec.key, next)}
              />
            )}
            {spec.kind === "enum" && !spec.multi && (
              <>
                {spec.options?.length ? (
                  <select
                    id={inputId}
                    className="wf-param-input"
                    value={value == null ? "" : String(value)}
                    onChange={(event) => setValue(spec.key, event.target.value || undefined)}
                  >
                    <option value="">（未设置）</option>
                    {spec.options.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={inputId}
                    className="wf-param-input"
                    list={`${inputId}-list`}
                    value={value == null ? "" : String(value)}
                    onChange={(event) => setValue(spec.key, event.target.value)}
                  />
                )}
              </>
            )}

            {spec.kind === "boolean" && (
              <label className="wf-param-checkbox" htmlFor={inputId}>
                <input
                  id={inputId}
                  type="checkbox"
                  checked={value === true}
                  onChange={(event) => setValue(spec.key, event.target.checked)}
                />
                <span>{value === true ? "已开启" : "已关闭"}</span>
              </label>
            )}

            {spec.kind === "number" && (
              <input
                id={inputId}
                className="wf-param-input"
                type="number"
                step="any"
                value={value == null ? "" : String(value)}
                onChange={(event) => setValue(spec.key, event.target.value === "" ? undefined : Number(event.target.value))}
              />
            )}

            {spec.kind === "string" && (
              <input
                id={inputId}
                className="wf-param-input"
                value={value == null ? "" : String(value)}
                placeholder={spec.placeholder}
                onChange={(event) => setValue(spec.key, event.target.value)}
              />
            )}

            {spec.kind === "columns" && (
              <ColumnsField
                value={Array.isArray(value) ? (value as string[]) : value ? [String(value)] : []}
                options={filterColumns(columns, spec.columnFilter)}
                onChange={(next) => setValue(spec.key, spec.multi ? next : next[0])}
              />
            )}

            {spec.kind === "json" && (
              <JsonField value={value} placeholder={spec.placeholder} onChange={(next) => setValue(spec.key, next)} />
            )}

            {spec.kind === "dataset" && (
              <select
                id={inputId}
                className="wf-param-input"
                value={value == null ? "" : String(value)}
                onChange={(event) =>
                  setValue(spec.key, event.target.value === "" ? undefined : Number(event.target.value))
                }
              >
                <option value="">（未选择）</option>
                {datasetOptions.map((option) => (
                  <option key={option.id} value={option.id}>#{option.id} {option.name}</option>
                ))}
              </select>
            )}

            {spec.kind === "conditions" && (
              <ConditionsField
                value={value}
                options={columns.map((c) => c.column)}
                onChange={(next) => setValue(spec.key, next)}
              />
            )}

            {spec.kind === "aggregations" && (
              <AggregationsField
                value={value}
                options={columns.map((c) => c.column)}
                onChange={(next) => setValue(spec.key, next)}
              />
            )}

            {spec.hint && <p className="wf-param-hint">{spec.hint}</p>}
          </div>
        );
      })}

      <details
        className="wf-param-advanced"
        open={advancedOpen}
        onToggle={(event) => {
          const open = (event.target as HTMLDetailsElement).open;
          setAdvancedOpen(open);
          if (open) {
            setAdvancedDraft(JSON.stringify(flat, null, 2));
            setAdvancedError(null);
          }
        }}
      >
        <summary>高级：直接编辑 JSON</summary>
        {advancedOpen && (
          <>
            <textarea
              className="wf-param-textarea"
              rows={8}
              value={advancedDraft}
              onChange={(event) => {
                setAdvancedDraft(event.target.value);
                try {
                  onChange(JSON.parse(event.target.value) as Record<string, unknown>);
                  setAdvancedError(null);
                } catch {
                  setAdvancedError("JSON 解析失败，暂未生效");
                }
              }}
            />
            {advancedError && <p className="wf-param-error">{advancedError}</p>}
          </>
        )}
      </details>
    </div>
  );
}
