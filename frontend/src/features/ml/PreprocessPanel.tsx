import type { MlCatalog } from "../../types/ml";
import { ParamHint, findParamSpec } from "./ParamGuide";

/**
 * 预处理配置（与后端 `build_pipeline` 接受的三个键一致）。
 *
 * 写成 `Record<string, unknown> & {...}` 而不是 interface：训练请求里
 * `preprocessing` 是自由字典，这样它可以直接传入而不必在调用处做断言。
 */
export type PreprocessValue = Record<string, unknown> & {
  missing?: { strategy: string };
  encoding?: { method: string };
  scaling?: { method: string };
};

type PreprocessKey = "missing" | "encoding" | "scaling";

/** 面板暴露的三个可调项，其余配置由后端按数据画像自动生成。 */
const FIELDS: { key: PreprocessKey; spec: string; label: string }[] = [
  { key: "missing", spec: "missing.strategy", label: "缺失值填补" },
  { key: "encoding", spec: "encoding.method", label: "类别编码" },
  { key: "scaling", spec: "scaling.method", label: "数值缩放" },
];

/** 每项对应的取值字段名（missing 用 strategy，其余用 method）。 */
function valueField(key: PreprocessKey): "strategy" | "method" {
  return key === "missing" ? "strategy" : "method";
}

/** 从 catalog 取候选项（array 型 range 即候选值）。 */
function candidatesOf(catalog: MlCatalog | null, specName: string): string[] {
  const spec = findParamSpec(catalog, specName);
  const range = spec?.range;
  return Array.isArray(range) ? (range as unknown[]).map((v) => String(v)) : [];
}

interface Props {
  catalog: MlCatalog | null;
  value: PreprocessValue;
  onChange: (next: PreprocessValue) => void;
  /** 数据集里是否存在需要该步骤的列；缺失时该项调了也不生效。 */
  availability?: { missing: boolean; encoding: boolean; scaling: boolean };
}

/**
 * 预处理配置面板。
 *
 * 预处理对最终效果的影响常常比模型超参数更大（类别编码方式会直接改变特征空间，
 * 缩放方式决定距离类模型能否正常工作），但它此前只能「自动」——
 * 后端其实早就接受这三个配置，缺的只是一个入口。
 *
 * 全部留空时行为与从前完全一致（后端按数据画像自动生成配置）；
 * 只有用户显式选择的项才会写进请求，因此这一步是可选的、可回退的。
 */
export function PreprocessPanel({ catalog, value, onChange, availability }: Props) {
  const anySet = FIELDS.some((f) => value[f.key] !== undefined);

  function setField(key: PreprocessKey, raw: string) {
    // 每个子配置整体替换，避免残留上一个策略的字段；空值＝交给后端自动决定
    const next: PreprocessValue = { ...value };
    if (key === "missing") {
      if (raw) next.missing = { strategy: raw };
      else delete next.missing;
    } else if (key === "encoding") {
      if (raw) next.encoding = { method: raw };
      else delete next.encoding;
    } else if (raw) {
      next.scaling = { method: raw };
    } else {
      delete next.scaling;
    }
    onChange(next);
  }

  function currentOf(key: PreprocessKey): string {
    const sub = value[key] as Record<string, string> | undefined;
    return sub?.[valueField(key)] ?? "";
  }

  return (
    <div className="ml-param-box">
      <div className="ml-tuner-head">
        <strong>预处理</strong>
        <span className="ml-tuner-count">留空则由后端按数据自动决定</span>
        <div className="ml-tuner-head-actions">
          <button
            type="button"
            className="btn"
            style={{ padding: "4px 10px" }}
            disabled={!anySet}
            onClick={() => onChange({})}
          >
            恢复自动
          </button>
        </div>
      </div>

      <div className="ml-tuner-grid">
        {FIELDS.map((f) => {
          const cur = currentOf(f.key);
          const missingData = availability && !availability[f.key];
          return (
            <div className="ml-tuner-field" key={f.key}>
              <div className="ml-tuner-label">
                <span className="ml-tuner-name">{f.label}</span>
                <ParamHint spec={findParamSpec(catalog, f.spec)} />
              </div>
              <select value={cur} onChange={(e) => setField(f.key, e.target.value)}>
                <option value="">（自动）</option>
                {candidatesOf(catalog, f.spec).map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              {missingData && <span className="ml-tuner-blocked">当前数据没有这类列，设置后不生效</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
