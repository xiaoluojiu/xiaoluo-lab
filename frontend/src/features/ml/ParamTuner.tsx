import { useMemo, useState } from "react";
import type { MlCatalog, MlParamCombo, MlParamSpec } from "../../types/ml";
import { ParamHint } from "./ParamGuide";

/** 参数值的编码：undefined 表示「不传，用 sklearn 默认」。 */
const keyOf = (v: unknown): string => (v === undefined ? "" : JSON.stringify(v));
const valOf = (raw: string): unknown => (raw === "" ? undefined : JSON.parse(raw));

/** 枚举候选：优先用 options（区分显示名与真实值），否则把数组型 range 当候选。 */
function optionsOf(spec: MlParamSpec): { label: string; value: unknown }[] {
  if (spec.options && spec.options.length) return spec.options;
  if (Array.isArray(spec.range)) {
    return (spec.range as unknown[]).map((v) => ({ label: String(v), value: v }));
  }
  if (spec.type === "bool") {
    return [
      { label: "true", value: true },
      { label: "false", value: false },
    ];
  }
  return [];
}

/** 单个参数的输入控件：按 type 决定用下拉还是数字框。 */
function ParamControl({
  spec,
  value,
  disabled,
  onChange,
}: {
  spec: MlParamSpec;
  value: unknown;
  disabled: boolean;
  onChange: (v: unknown) => void;
}) {
  const options = optionsOf(spec);
  if (options.length) {
    return (
      <select
        value={keyOf(value)}
        disabled={disabled}
        onChange={(e) => onChange(valOf(e.target.value))}
      >
        {/* 空选项＝该参数不写进请求，由 sklearn 用它自己的默认值 */}
        <option value="">（默认）</option>
        {options.map((o) => (
          <option key={keyOf(o.value)} value={keyOf(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    );
  }
  const isFloat = spec.type === "float";
  return (
    <input
      type="number"
      step={spec.step ?? (isFloat ? 0.1 : 1)}
      disabled={disabled}
      value={value === undefined ? "" : String(value)}
      onChange={(e) => {
        const raw = e.target.value;
        if (!raw.trim()) return onChange(undefined);
        const n = Number.parseFloat(raw);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
    />
  );
}

interface Props {
  catalog: MlCatalog | null;
  model: string;
  /** 当前已设置的参数（未出现的键＝用默认值，不会被发送）。 */
  values: Record<string, unknown>;
  /** 按当前数据规模算出的推荐值，仅用于「改动对比」与「恢复推荐值」。 */
  recommended: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  /** 当前命中「无效参数组合」的提示（由页面统一判定）。 */
  comboWarnings?: MlParamCombo[];
}

/**
 * 模型参数调整面板。
 *
 * 参数清单完全来自后端教学元数据（`/ml/catalog` 的模型参数），前端不再硬编码
 * 「这个模型能调哪几项」——否则元数据里写了说明、界面上却没有入口，说明就成了摆设。
 *
 * 三条交互约定：
 * 1. **核心参数常显，进阶参数按需启用**：不加限制地铺开会让表单长到无法阅读，
 *    但进阶参数又确实影响结果，所以用一个可展开的清单让用户自己决定启用哪些。
 * 2. **未设置的参数不写进请求**：显示「（默认）」即等价于交给 sklearn，
 *    这样「不动任何东西」与「显式填入默认值」在结果上完全等价，行为可预期。
 * 3. **每项都能单独恢复**：改过哪一项、能不能退回去，必须一眼可见——
 *    否则调参变成单向操作，用户不敢试。
 */
export function ParamTuner({
  catalog,
  model,
  values,
  recommended,
  onChange,
  comboWarnings = [],
}: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const specs = useMemo(
    () => catalog?.models.find((m) => m.name === model)?.params ?? [],
    [catalog, model],
  );
  const specOf = useMemo(() => {
    const map = new Map<string, MlParamSpec>();
    specs.forEach((s) => map.set(s.name, s));
    return map;
  }, [specs]);

  const core = specs.filter((s) => (s.tier ?? "core") === "core");
  const advanced = specs.filter((s) => s.tier === "advanced");
  // 已赋值的进阶参数提到主区，避免用户启用后找不到它在哪
  const activeAdvanced = advanced.filter((s) => values[s.name] !== undefined);
  const dormantAdvanced = advanced.filter((s) => values[s.name] === undefined);
  const visible = [...core, ...activeAdvanced];

  /** 参数的有效取值：已设置 → 用它；否则 → sklearn 默认值。 */
  function effective(name: string): unknown {
    if (values[name] !== undefined) return values[name];
    return specOf.get(name)?.sklearn_default;
  }

  /** 未满足的依赖（如 l1_ratio 需要 penalty=elasticnet）。 */
  function blockedBy(spec: MlParamSpec): string | null {
    if (!spec.requires) return null;
    for (const [dep, want] of Object.entries(spec.requires)) {
      const cur = effective(dep);
      const ok = Array.isArray(want)
        ? want.some((w) => keyOf(w) === keyOf(cur))
        : keyOf(want) === keyOf(cur);
      if (!ok) {
        const label = specOf.get(dep)?.label ?? dep;
        return `需先设置「${label}」为 ${want === true ? "开启" : String(want)}`;
      }
    }
    return null;
  }

  function setParam(name: string, v: unknown) {
    const next = { ...values };
    if (v === undefined) delete next[name];
    else next[name] = v;
    onChange(next);
  }

  /** 单参数恢复：有推荐值回到推荐值，否则清空（回到 sklearn 默认）。 */
  function restore(name: string) {
    setParam(name, recommended[name] !== undefined ? recommended[name] : undefined);
  }

  function isChanged(name: string): boolean {
    return values[name] !== undefined && keyOf(values[name]) !== keyOf(recommended[name]);
  }

  function hasAnyChange(): boolean {
    return Object.keys(values).some((k) => isChanged(k));
  }

  if (!specs.length) {
    return (
      <div className="ml-param-box">
        <div className="flex-between">
          <strong>模型参数</strong>
        </div>
        <p className="ml-tuner-empty">该模型没有需要调整的超参数。</p>
      </div>
    );
  }

  return (
    <div className="ml-param-box">
      <div className="ml-tuner-head">
        <strong>模型参数</strong>
        <span className="ml-tuner-count">
          可直接调 {specs.length} 项（核心 {core.length}）
        </span>
        <div className="ml-tuner-head-actions">
          <button
            type="button"
            className="btn"
            style={{ padding: "4px 10px" }}
            disabled={!hasAnyChange()}
            onClick={() => onChange({ ...recommended })}
          >
            恢复推荐值
          </button>
        </div>
      </div>

      {comboWarnings.map((w) => (
        <p className="ml-tuner-warn" key={`${w.model}-${w.message}`}>
          {w.message}
        </p>
      ))}

      <div className="ml-tuner-grid">
        {visible.map((spec) => {
          const blocked = blockedBy(spec);
          return (
            <div className="ml-tuner-field" key={spec.name}>
              <div className="ml-tuner-label">
                <span className="ml-tuner-name">{spec.label}</span>
                <ParamHint spec={spec} />
                {isChanged(spec.name) && (
                  <button
                    type="button"
                    className="ml-tuner-restore"
                    title="恢复本项"
                    onClick={() => restore(spec.name)}
                  >
                    ↺
                  </button>
                )}
              </div>
              <ParamControl
                spec={spec}
                value={values[spec.name]}
                disabled={!!blocked}
                onChange={(v) => setParam(spec.name, v)}
              />
              {blocked && <span className="ml-tuner-blocked">{blocked}</span>}
            </div>
          );
        })}
      </div>

      {dormantAdvanced.length > 0 && (
        <div className="ml-tuner-more">
          <button
            type="button"
            className="btn"
            style={{ padding: "4px 10px" }}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? "收起进阶参数" : `展开进阶参数（${dormantAdvanced.length}）`}
          </button>
          {showAdvanced && (
            <div className="ml-tuner-add-list">
              {dormantAdvanced.map((spec) => (
                <div className="ml-tuner-add-item" key={spec.name}>
                  <div className="ml-tuner-label">
                    <span className="ml-tuner-name">{spec.label}</span>
                    <ParamHint spec={spec} />
                  </div>
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: "3px 10px" }}
                    onClick={() => setParam(spec.name, spec.sklearn_default)}
                  >
                    + 启用
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
