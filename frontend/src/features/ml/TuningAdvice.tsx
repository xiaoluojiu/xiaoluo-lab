import { useMemo } from "react";
import type { MlCatalog, MlParamSpec, MlTuningAdvice, MlTuningRule, TrainArtifacts } from "../../types/ml";
import { InfoHint } from "../../components/InfoHint";

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/**
 * 判定本次训练命中了哪些「调参信号」。
 *
 * 判定只用后端训练时真实落库的产物（测试集指标 / 训练集指标 / 类别分布 /
 * 特征规模 / 耗时），不做任何前端估算——否则建议会建立在猜测上。
 */
export function detectSignals(
  task: string,
  metrics: Record<string, number | string>,
  artifacts: TrainArtifacts | null,
  runtime: number | null | undefined,
): Set<string> {
  const out = new Set<string>();
  const train = (artifacts?.train_metrics ?? null) as Record<string, number> | null;

  if (task === "classification") {
    const accTest = num(metrics.accuracy);
    const f1Test = num(metrics.f1);
    // 过拟合取 accuracy / f1 中更大的那个差距：只要有一项明显回落就值得提示
    const gaps = [
      train?.accuracy != null && accTest != null ? train.accuracy - accTest : null,
      train?.f1 != null && f1Test != null ? train.f1 - f1Test : null,
    ].filter((g): g is number => g !== null);
    const gap = gaps.length ? Math.max(...gaps) : null;

    if (gap !== null && gap > 0.1) out.add("overfit");
    if (accTest !== null && accTest < 0.7 && (gap === null || gap <= 0.05)) out.add("underfit");
    const auc = num(metrics.roc_auc);
    if (auc !== null && auc <= 0.55) out.add("no_signal");

    const dist = artifacts?.class_distribution;
    if (dist && dist.majority_ratio >= 0.7) out.add("imbalanced");
  } else if (task === "regression") {
    const r2Test = num(metrics.r2);
    const gap = train?.r2 != null && r2Test != null ? train.r2 - r2Test : null;
    if (gap !== null && gap > 0.15) out.add("overfit");
    if (r2Test !== null && r2Test < 0.3 && (gap === null || gap <= 0.1)) out.add("underfit");
    if (r2Test !== null && r2Test <= 0) out.add("no_signal");
  } else if (task === "clustering") {
    const sil = num(metrics.silhouette);
    const k = num(metrics.cluster_count);
    if ((sil !== null && sil < 0.25) || k === 1) out.add("cluster_weak");
    if (sil !== null && sil < 0.1) out.add("no_signal");
  }

  const nFeatures = artifacts?.model_features?.length ?? 0;
  const nRows = artifacts?.train_rows ?? 0;
  if (
    (runtime != null && runtime > 30) ||
    (nFeatures > 0 && nRows > 0 && nFeatures > nRows)
  ) {
    out.add("costly");
  }

  // 「几乎没学到东西」比「欠拟合」更精确，同时给出会让用户困惑，只留前者
  if (out.has("no_signal")) out.delete("underfit");
  return out;
}

/**
 * 按 op 算出建议值。
 *
 * halve / double 以「当前值」为基准（未设置时取该参数的真实默认值），
 * 两者的 value 是当前值不可计算时的兜底；set 直接采用 value。
 */
function suggestValue(
  spec: MlParamSpec | undefined,
  current: unknown,
  op: MlTuningAdvice["op"],
  fallback: unknown,
): unknown {
  if (op === "set") return fallback;
  const base =
    typeof current === "number" ? current : typeof fallback === "number" ? fallback : null;
  if (base === null) return fallback;
  const scaled = op === "halve" ? base / 2 : base * 2;
  if (Number.isInteger(base)) {
    // 聚类簇数不能降到 1（等于没分），其余整数参数下限为 1
    const min = spec?.name === "n_clusters" ? 2 : 1;
    return Math.max(min, Math.floor(scaled));
  }
  return Math.max(0.0001, Number(scaled.toFixed(4)));
}

function show(v: unknown): string {
  if (v === null) return "None";
  if (v === undefined) return "—";
  if (typeof v === "boolean") return v ? "开" : "关";
  return String(v);
}

interface Props {
  catalog: MlCatalog | null;
  model: string;
  artifacts: TrainArtifacts | null;
  metrics: Record<string, number | string>;
  runtime?: number | null;
  onApply: (patch: Record<string, unknown>) => void;
}

/**
 * 调参建议：训练完之后，把「指标不理想」翻译成「下一步该动哪个参数」。
 *
 * 规则来自后端 `tuning_playbook`（单一事实源），这里只负责判定命中、按当前模型过滤
 * （元数据里写了但该模型没有的参数直接不显示，避免给出无效建议），以及把建议写成
 * 参数面板能直接接住的一次改动——调参因此从「凭感觉试」变成「有依据地试一次」。
 */
export function TuningAdvice({
  catalog,
  model,
  artifacts,
  metrics,
  runtime = null,
  onApply,
}: Props) {
  const task = artifacts?.task ?? "";
  // 建议的基准是「那一次训练实际用的参数」而不是参数面板的当前值：
  // 否则点一次「应用」后目标值会跟着变，用户会以为建议在漂移。
  const trainedParams = (artifacts?.model_summary?.params ?? {}) as Record<string, unknown>;
  const specs = useMemo(
    () => catalog?.models.find((m) => m.name === model)?.params ?? [],
    [catalog, model],
  );
  const specOf = useMemo(() => {
    const map = new Map<string, MlParamSpec>();
    specs.forEach((s) => map.set(s.name, s));
    return map;
  }, [specs]);

  const hits = useMemo(
    () => detectSignals(task, metrics, artifacts, runtime),
    [task, metrics, artifacts, runtime],
  );

  const matched = useMemo(() => {
    const rules = (catalog?.tuning_playbook ?? []) as MlTuningRule[];
    return rules
      .filter((r) => hits.has(r.signal))
      .map((r) => ({
        rule: r,
        // 只保留当前模型真实存在的参数，否则「应用」会传进一条 sklearn 不认识的参数
        items: r.advice
          .map((a) => ({ advice: a, spec: specOf.get(a.param) }))
          .filter((x): x is { advice: MlTuningAdvice; spec: MlParamSpec } => !!x.spec),
      }))
      .filter((r) => r.items.length > 0);
  }, [catalog, hits, specOf]);

  if (!matched.length) return null;

  function targetOf(item: { advice: MlTuningAdvice; spec: MlParamSpec }): unknown {
    const current = trainedParams[item.spec.name] !== undefined
      ? trainedParams[item.spec.name]
      : item.spec.sklearn_default;
    return suggestValue(item.spec, current, item.advice.op, item.advice.value);
  }

  return (
    <div className="ml-advice">
      <h4 style={{ margin: "0 0 var(--space-2)" }}>调参建议</h4>
      {matched.map(({ rule, items }) => (
        <div className="ml-advice-block" key={rule.id}>
          <div className="ml-advice-head">
            <span className="ml-advice-title">
              {rule.title}
              <InfoHint label={`${rule.title}：为什么会这样`}>
                <span>{rule.why}</span>
              </InfoHint>
            </span>
            <button
              type="button"
              className="btn"
              style={{ padding: "3px 10px" }}
              onClick={() =>
                onApply(Object.fromEntries(items.map((it) => [it.spec.name, targetOf(it)])))
              }
            >
              全部应用到参数
            </button>
          </div>
          <p className="ml-advice-detect">判定依据：{rule.detect}</p>
          <ul className="ml-advice-list">
            {items.map((it) => {
              const target = targetOf(it);
              const opText =
                it.advice.op === "halve" ? "调小" : it.advice.op === "double" ? "调大" : "设为";
              return (
                <li className="ml-advice-item" key={it.spec.name}>
                  <span className="ml-advice-param">{it.spec.label}</span>
                  <span className="ml-advice-action">
                    {opText} <b>{show(target)}</b>
                    <code>{it.spec.name}</code>
                  </span>
                  <span className="ml-advice-note">{it.advice.note}</span>
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: "3px 10px" }}
                    onClick={() => onApply({ [it.spec.name]: target })}
                  >
                    应用
                  </button>
                </li>
              );
            })}
          </ul>
          {rule.escalate && <p className="ml-advice-escalate">{rule.escalate}</p>}
          {/* 不用重训的替代路径：改超参要重跑训练，改决策阈值不用。
              两条路都能动同一个指标时，必须让用户知道有便宜的那条。 */}
          {rule.inference_alternative && (
            <p className="ml-advice-inference">{rule.inference_alternative}</p>
          )}
        </div>
      ))}
    </div>
  );
}
