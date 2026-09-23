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
import type { ParamSpec } from "./nodeSpecs";
import { flattenConfig } from "./configView";
// 自带样式：沉浸式编辑器不在 .content 内，拿不到 design-foundation 的控件兜底规则。
import "./params.css";

interface Props {
  specs: ParamSpec[];
  config: Record<string, unknown>;
  columns: SchemaColumn[];
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

export function NodeParamForm({ specs, config, columns, onChange }: Props) {
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

            {spec.kind === "enum" && (
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
                  // 数据集下拉等运行期才知道选项的场景，由调用方以 datalist 提供。
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
