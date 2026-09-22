import type { MlCatalog, TrainArtifacts } from "../../types/ml";
import { paramLabel } from "./ParamGuide";

/** 把 `unknown` 渲染成可读文本（对象走 JSON、数组走逗号）。 */
function show(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.length ? v.map((x) => String(x)).join(", ") : "（空）";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** 预处理三段的配置，逐段渲染成「策略 + 涉及列」。 */
function PreprocessConfig({ report }: { report: TrainArtifacts["preprocessing_report"] }) {
  if (!report) return null;
  const cfg = report.config ?? {};
  const rows: { key: string; label: string; value: string }[] = [];

  const missing = cfg.missing as { strategy?: string; columns?: string[] } | null | undefined;
  if (missing) {
    rows.push({
      key: "missing",
      label: "缺失填补",
      value: `策略 ${missing.strategy ?? "—"} · 列 ${show(missing.columns)}`,
    });
  }
  const enc = cfg.encoding as { method?: string; columns?: string[] } | null | undefined;
  if (enc) {
    rows.push({
      key: "encoding",
      label: "类别编码",
      value: `方式 ${enc.method ?? "—"} · 列 ${show(enc.columns)}`,
    });
  }
  const sc = cfg.scaling as { method?: string; columns?: string[] } | null | undefined;
  if (sc) {
    rows.push({
      key: "scaling",
      label: "数值缩放",
      value: `方式 ${sc.method ?? "—"} · 列 ${show(sc.columns)}`,
    });
  }

  if (!rows.length && !report.features_out?.length) return null;

  return (
    <div className="ml-trace-block">
      <h5 className="ml-guide-subtitle">④ 预处理（统计量只在训练集拟合，防数据泄漏）</h5>
      <p className="ml-trace-note">
        管道 fitted = <b>{String(report.fitted ?? false)}</b>
      </p>
      {rows.length > 0 && (
        <ul className="ml-trace-list">
          {rows.map((r) => (
            <li key={r.key}>
              <span className="ml-trace-label">{r.label}</span>
              <span>{r.value}</span>
            </li>
          ))}
        </ul>
      )}
      {report.features_out && report.features_out.length > 0 && (
        <details className="mt">
          <summary style={{ cursor: "pointer" }}>
            预处理后真正喂给模型的 {report.features_out.length} 个特征
          </summary>
          <div className="ml-chip-row mt">
            {report.features_out.map((f) => (
              <span className="ml-chip" key={f}>{f}</span>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

interface Props {
  artifacts: TrainArtifacts;
  catalog?: MlCatalog | null;
  model?: string;
}

/**
 * 训练过程透明化：把 `run.artifacts` 里的「怎么做的」摊开给人看 ——
 * 特征怎么选的、切分规模多少、预处理用了什么策略、模型实际用了哪些参数。
 * 数据全部来自后端训练时落库的产物，不是前端估算。
 */
export function TrainTrace({ artifacts, catalog = null, model }: Props) {
  const prep = artifacts.preprocessing_report;
  const summary = artifacts.model_summary;
  const features = artifacts.features ?? [];
  const modelFeatures = artifacts.model_features ?? [];
  const excluded = artifacts.excluded_columns ?? [];

  const paramEntries = Object.entries(summary?.params ?? {});

  return (
    <div className="ml-trace">
      <h4 style={{ margin: "0 0 var(--space-2)" }}>训练过程</h4>

      {/* ② 特征与目标划分 */}
      <div className="ml-trace-block">
        <h5 className="ml-guide-subtitle">② 特征与目标划分</h5>
        <div className="ml-trace-kv">
          <span><b>目标列</b>{artifacts.target_column ?? "无（聚类）"}</span>
          <span><b>参与训练的特征</b>{features.length} 列</span>
          <span><b>排除的列</b>{excluded.length ? `${excluded.length} 列` : "无"}</span>
        </div>
        {features.length > 0 && (
          <details className="mt">
            <summary style={{ cursor: "pointer" }}>查看参与训练的原始列清单</summary>
            <div className="ml-chip-row mt">
              {features.map((f) => <span className="ml-chip" key={f}>{f}</span>)}
            </div>
          </details>
        )}
        {excluded.length > 0 && (
          <details className="mt">
            <summary style={{ cursor: "pointer" }}>查看被排除的列（不参与训练）</summary>
            <div className="ml-chip-row mt">
              {excluded.map((f) => <span className="ml-chip ml-chip-muted" key={f}>{f}</span>)}
            </div>
          </details>
        )}
      </div>

      {/* ③ 训练/测试划分 */}
      <div className="ml-trace-block">
        <h5 className="ml-guide-subtitle">③ 训练 / 测试划分</h5>
        <div className="ml-trace-kv">
          <span><b>训练集</b>{show(artifacts.train_rows)} 行</span>
          <span><b>测试集</b>{show(artifacts.test_rows)} 行</span>
          <span><b>test_size</b>{show(artifacts.test_size)}</span>
          <span><b>是否分层</b>{artifacts.stratified ? "是（按类别比例）" : "否"}</span>
          {!!artifacts.dropped_rows && (
            <span className="ml-trace-warn"><b>剔除空标签行</b>{artifacts.dropped_rows} 行</span>
          )}
        </div>
      </div>

      {/* ④ 预处理 */}
      <PreprocessConfig report={prep} />

      {/* ⑤ 模型训练 */}
      <div className="ml-trace-block">
        <h5 className="ml-guide-subtitle">⑤ 模型训练</h5>
        <div className="ml-trace-kv">
          <span><b>模型</b>{summary?.name ?? artifacts.model ?? "—"}</span>
          <span><b>任务</b>{summary?.task ?? artifacts.task ?? "—"}</span>
          <span><b>特征数</b>{show(summary?.n_features ?? modelFeatures.length)}</span>
          <span><b>训练样本数</b>{show(summary?.n_samples ?? artifacts.train_rows)}</span>
          {summary?.fit_seconds != null && <span><b>拟合耗时</b>{summary.fit_seconds.toFixed(3)}s</span>}
        </div>
        {paramEntries.length > 0 ? (
          <>
            <p className="ml-trace-note">实际传给模型的参数：</p>
            <ul className="ml-trace-list">
              {paramEntries.map(([k, v]) => (
                <li key={k}>
                  <span className="ml-trace-label">{paramLabel(catalog, k, model)}</span>
                  <span><code>{k}</code> = <b>{show(v)}</b></span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="ml-trace-note">该模型没有额外超参数（或未传参，使用 sklearn 默认值）。</p>
        )}
        {modelFeatures.length > 0 && (
          <details className="mt">
            <summary style={{ cursor: "pointer" }}>
              编码后的模型输入列（{modelFeatures.length} 个，与上方原始列不一定一一对应）
            </summary>
            <div className="ml-chip-row mt">
              {modelFeatures.map((f) => <span className="ml-chip ml-chip-accent" key={f}>{f}</span>)}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
