import { useMemo, useState } from "react";
import type { MlCatalog, MlMetricGuide, MlParamSpec } from "../../types/ml";
import { InfoHint } from "../../components/InfoHint";
import { modelLabel, PARAM_LABELS } from "./modelMeta";

/** 把 unknown 类型的 `range` / `default` 渲染成一行可读文本。 */
function renderValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.map((x) => String(x)).join(" / ");
  if (typeof v === "object") {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, val]) => `${k}=${String(val)}`)
      .join(", ");
  }
  return String(v);
}

/**
 * 参数说明的折叠形态（范围 · 默认 · 影响）。
 *
 * 这些内容对第一次调参的人有用，但常驻在输入框下方会持续占位：
 * 一个模型动辄 3~5 个参数，每个都挂三行小字，表单会变得很长。
 * 因此改为一个 ⓘ：悬停或键盘聚焦展开，信息一条不少，默认不占视觉空间。
 */
export function ParamHint({ spec }: { spec?: MlParamSpec }) {
  if (!spec) return null;
  const typical = Array.isArray(spec.typical) && spec.typical.length ? spec.typical : null;
  return (
    <InfoHint label={`${spec.label} 参数说明`}>
      <span><b>范围</b>：{renderValue(spec.range)}</span>
      <span><b>默认</b>：{renderValue(spec.default)}</span>
      {typical && <span><b>常用</b>：{typical.map((v) => String(v)).join(" / ")}</span>}
      {spec.effect && <span><b>影响</b>：{spec.effect}</span>}
      {spec.when_to_change && <span><b>什么时候改</b>：{spec.when_to_change}</span>}
      {/* 「会不会影响结果」是调参时最先要确认的问题，有值就必须答，别让用户自己猜。 */}
      {spec.affects_result !== undefined && (
        <span>
          <b>是否影响结果</b>：
          {spec.affects_result ? "会影响预测结果" : "不影响预测结果，只影响展示或性能"}
        </span>
      )}
      {spec.evidence && <span><b>依据</b>：{spec.evidence}</span>}
    </InfoHint>
  );
}

interface ParamTableProps {
  title: string;
  params: MlParamSpec[];
  /** 是否展示「什么时候该改」一列（参数多于 3 项时更易读）。 */
  showWhenToChange?: boolean;
}

/** 参数说明表：名称 / 范围 / 默认 / 影响 / 什么时候改。 */
export function ParamTable({ title, params, showWhenToChange = true }: ParamTableProps) {
  if (!params.length) return null;
  return (
    <div className="ml-guide-block">
      <h5 className="ml-guide-subtitle">{title}</h5>
      <div className="ml-guide-table-wrap">
        <table className="data-table ml-guide-table">
          <thead>
            <tr>
              <th style={{ width: "18%" }}>参数</th>
              <th style={{ width: "16%" }}>取值范围</th>
              <th style={{ width: "12%" }}>默认值</th>
              <th>对结果的影响</th>
              {showWhenToChange && <th style={{ width: "22%" }}>什么时候改</th>}
            </tr>
          </thead>
          <tbody>
            {params.map((p) => (
              <tr key={p.name}>
                <td>
                  <strong>{p.label}</strong>
                  {p.tier === "advanced" && <span className="ml-guide-tier">进阶</span>}
                  {/* 只标「例外」：绝大多数参数都影响结果，逐个标反而变噪声；
                      真正需要提醒的是「看着像参数、其实不影响结果」的那几个。 */}
                  {p.affects_result === false && <span className="ml-guide-tier">不影响结果</span>}
                  {p.applies_to === "classification_binary" && (
                    <span className="ml-guide-tier">仅二分类</span>
                  )}
                  <div className="ml-guide-code">{p.name}</div>
                </td>
                <td>{renderValue(p.range)}</td>
                <td>
                  {renderValue(p.default)}
                  {Array.isArray(p.typical) && p.typical.length > 0 && (
                    <div className="ml-guide-typical">
                      常用 {p.typical.map((v) => String(v)).join(" / ")}
                    </div>
                  )}
                </td>
                <td>{p.effect ?? "—"}</td>
                {showWhenToChange && <td>{p.when_to_change ?? "—"}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** 指标解读表：指标 / 含义 / 怎么读 / 多少算好。 */
export function MetricTable({ metrics }: { metrics: Record<string, MlMetricGuide> }) {
  const entries = Object.entries(metrics ?? {});
  if (!entries.length) return null;
  const order = ["accuracy", "precision", "recall", "f1", "roc_auc", "mae", "rmse", "r2", "cluster_count", "silhouette"];
  const sorted = entries.sort((a, b) => {
    const ia = order.indexOf(a[0]);
    const ib = order.indexOf(b[0]);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return (
    <div className="ml-guide-block">
      <h5 className="ml-guide-subtitle">指标怎么读</h5>
      <div className="ml-guide-table-wrap">
        <table className="data-table ml-guide-table">
          <thead>
            <tr>
              <th style={{ width: "14%" }}>指标</th>
              <th style={{ width: "12%" }}>适用任务</th>
              <th>含义</th>
              <th style={{ width: "30%" }}>怎么读</th>
              <th style={{ width: "14%" }}>方向</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(([key, m]) => (
              <tr key={key}>
                <td>
                  <strong>{m.label}</strong>
                  <div className="ml-guide-code">{key}</div>
                </td>
                <td>{m.task}</td>
                <td>{m.meaning}</td>
                <td>{m.how_to_read}</td>
                <td>{m.better}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface Props {
  catalog: MlCatalog | null;
  /** 当前选中的模型（用于把该模型参数表置顶高亮）。 */
  model?: string;
  /** 默认展开还是收起。 */
  defaultOpen?: boolean;
}

/**
 * 「流程与参数说明」面板：直接渲染 `GET /ml/catalog` 的教学元数据。
 * 让「透明」在界面上可被看到 —— 每个环节做什么、每个参数影响什么、指标怎么读。
 */
export function ParamGuide({ catalog, model, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [tab, setTab] = useState<"pipeline" | "params" | "metrics">("pipeline");

  const currentModel = useMemo(
    () => catalog?.models.find((m) => m.name === model) ?? null,
    [catalog, model],
  );
  // 当前模型没有专属参数（如线性回归）时，退化为展示全部模型参数
  const otherModels = useMemo(
    () => (catalog?.models ?? []).filter((m) => m.name !== model && m.params.length > 0),
    [catalog, model],
  );

  if (!catalog) {
    return (
      <div className="card ml-guide-card">
        <div className="flex-between">
          <h3 style={{ margin: 0 }}>流程与参数说明</h3>
        </div>
        <div className="ml-guide-empty">教学目录暂不可用（后端 <code>/ml/catalog</code> 未响应）。</div>
      </div>
    );
  }

  const conventions = catalog.conventions;

  return (
    <div className="card ml-guide-card">
      <div className="ml-compare-toolbar">
        <div>
          <h3 style={{ margin: 0 }}>流程与参数说明</h3>
        </div>
        <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
          <button
            className={`btn ${tab === "pipeline" ? "primary" : ""}`}
            style={{ padding: "5px 12px" }}
            onClick={() => { setTab("pipeline"); setOpen(true); }}
          >
            流程环节（{catalog.pipeline_steps.length}）
          </button>
          <button
            className={`btn ${tab === "params" ? "primary" : ""}`}
            style={{ padding: "5px 12px" }}
            onClick={() => { setTab("params"); setOpen(true); }}
          >
            参数说明
          </button>
          <button
            className={`btn ${tab === "metrics" ? "primary" : ""}`}
            style={{ padding: "5px 12px" }}
            onClick={() => { setTab("metrics"); setOpen(true); }}
          >
            指标解读（{Object.keys(catalog.metrics).length}）
          </button>
          <button className="btn" style={{ padding: "5px 12px" }} onClick={() => setOpen((v) => !v)}>
            {open ? "收起" : "展开"}
          </button>
        </div>
      </div>

      {open && tab === "pipeline" && (
        <div className="mt">
          <ol className="ml-step-list">
            {catalog.pipeline_steps.map((step) => (
              <li key={step.id} className="ml-step-item">
                <div className="ml-step-head">
                  <strong>{step.name}</strong>
                  <span className="ml-step-id">{step.id}</span>
                </div>
                <p className="ml-step-summary">{step.summary}</p>
                <div className="ml-step-io">
                  <span><b>输入</b>{step.input}</span>
                  <span><b>输出</b>{step.output}</span>
                </div>
                {step.key_behaviour && step.key_behaviour.length > 0 && (
                  <ul className="ml-step-behaviour">
                    {step.key_behaviour.map((b, i) => (
                      <li key={i}>{b}</li>
                    ))}
                  </ul>
                )}
                {step.verify && (
                  <p className="ml-step-verify">
                    <b>可核验</b>
                    {step.verify}
                  </p>
                )}
                {step.code && <div className="ml-guide-code">{step.code}</div>}
              </li>
            ))}
          </ol>
        </div>
      )}

      {open && tab === "params" && (
        <div className="mt">
          {conventions && (
            <div className="ml-conv-row">
              <span>默认 test_size：<b>{String(conventions.default_test_size ?? 0.2)}</b></span>
              <span>默认 seed：<b>{String(conventions.default_seed ?? 42)}</b></span>
              {conventions.target_naming && (
                <span>目标列常用名：<b>{conventions.target_naming.join(" / ")}</b></span>
              )}
              {conventions.default_models && (
                <span>
                  任务默认模型：
                  <b>
                    {Object.entries(conventions.default_models)
                      .map(([t, m]) => `${t}→${modelLabel(m)}`)
                      .join("，")}
                  </b>
                </span>
              )}
            </div>
          )}

          <ParamTable title="训练参数" params={catalog.training_params} />
          <ParamTable title="预处理参数" params={catalog.preprocessing_params} />

          {currentModel && (
            currentModel.params.length > 0 ? (
              <ParamTable
                title={`当前模型参数 · ${modelLabel(currentModel.name)}`}
                params={currentModel.params}
              />
            ) : (
              <div className="ml-guide-block">
                <h5 className="ml-guide-subtitle">当前模型参数 · {modelLabel(currentModel.name)}</h5>
                <div className="ml-guide-empty">
                  该模型没有需要调整的超参数。{currentModel.note}
                </div>
              </div>
            )
          )}
          {currentModel && currentModel.params.length > 0 && currentModel.note && (
            <p className="ml-guide-model-note">{currentModel.note}</p>
          )}

          {/* 与超参数分开成表：它们作用在「已训练好的模型」上，改一次立刻见效，
              迭代代价与「改超参必须重训」完全不同，混在一张表里会误导调参节奏。 */}
          {(catalog.inference_params?.length ?? 0) > 0 && (
            <ParamTable
              title="推理阶段参数（训练完成之后才生效，改它不需要重新训练）"
              params={catalog.inference_params ?? []}
            />
          )}

          {otherModels.map((m) => (
            <ParamTable key={m.name} title={`其他模型参数 · ${modelLabel(m.name)}`} params={m.params} />
          ))}
        </div>
      )}

      {open && tab === "metrics" && (
        <div className="mt">
          <MetricTable metrics={catalog.metrics} />
        </div>
      )}
    </div>
  );
}

/** 供其他组件复用：按参数名在所有来源里查说明（训练参数 → 预处理参数 → 模型参数）。 */
export function findParamSpec(
  catalog: MlCatalog | null,
  key: string,
  model?: string,
): MlParamSpec | undefined {
  if (!catalog) return undefined;
  const direct = [...catalog.training_params, ...catalog.preprocessing_params].find((p) => p.name === key);
  if (direct) return direct;
  const byModel = catalog.models.find((m) => m.name === model)?.params.find((p) => p.name === key);
  if (byModel) return byModel;
  return catalog.models.flatMap((m) => m.params).find((p) => p.name === key);
}

/** 参数名的中文标签：优先用教学元数据，退回前端内置 PARAM_LABELS。 */
export function paramLabel(catalog: MlCatalog | null, key: string, model?: string): string {
  const spec = findParamSpec(catalog, key, model);
  return spec?.label ?? PARAM_LABELS[key] ?? key;
}
