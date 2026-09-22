import { useMemo, useState } from "react";
import { predictWithRun } from "../../api/ml";
import type {
  ExperimentRun,
  MlCatalog,
  MlThresholdCurve,
  PredictionResult,
  TrainArtifacts,
} from "../../types/ml";
import { ParamHint } from "./ParamGuide";
import { modelLabel } from "./modelMeta";

interface Props {
  /** 可用于推理的运行（调用方应只传 success 的） */
  runs: ExperimentRun[];
  datasetId?: number | null;
  /** 教学元数据：推理参数的 ⓘ 文案来自它（单一事实源），界面不另写一份。 */
  catalog?: MlCatalog | null;
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

/**
 * 曲线依据来源的说明 —— 「这条曲线是拿哪份数据算的」必须写在脸上。
 * 拿训练数据算出来的好看指标不是泛化能力，不标出来就会被当成模型水平。
 */
const BASIS_LABEL: Record<string, string> = {
  holdout_test: "依据训练时留出的测试集（代表泛化能力，是选阈值的正当依据）",
  inference_data: "依据本次推理数据本身（若它就是训练数据，指标会偏乐观，仅供参考）",
};

const fmtPct = (v: number | null | undefined): string =>
  typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";

/**
 * 模型推理面板：选择一次成功运行 -> 对数据集执行推理 -> 展示预测预览。
 * 后端会复用训练时保存的预处理管道（pipeline.pkl），因此无需手动对齐特征。
 *
 * 这里额外承担「训练后参数」的入口 —— 决策阈值：
 * 它不改模型、不需要重训，改的只是「概率多少算正类」那条线。
 * 概率列本来就已经返回给界面了，若没有入口能作用在它上面，用户就只能看着概率干瞪眼。
 */
export function PredictionPanel({ runs, datasetId, catalog }: Props) {
  const successRuns = useMemo(() => runs.filter((r) => r.status === "success"), [runs]);
  const [runId, setRunId] = useState<number | null>(successRuns[0]?.id ?? null);
  const [limit, setLimit] = useState(20);
  // 阈值用字符串保存：空串＝不传该参数（沿用默认），避免「0」与「没填」混淆
  const [thresholdInput, setThresholdInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const activeRunId = runId ?? successRuns[0]?.id ?? null;
  const activeRun = successRuns.find((r) => r.id === activeRunId) ?? null;
  const artifacts = (activeRun?.artifacts ?? {}) as TrainArtifacts;

  const thresholdSpec = catalog?.inference_params?.find((p) => p.name === "threshold");
  const limitSpec = catalog?.inference_params?.find((p) => p.name === "limit");
  const step = thresholdSpec?.step ?? 0.05;
  const defaultThreshold =
    typeof thresholdSpec?.sklearn_default === "number" ? thresholdSpec.sklearn_default : 0.5;

  const isClassification = String(artifacts.task ?? "") === "classification";
  const nClasses = artifacts.class_distribution?.n_classes ?? null;
  // 二分类才谈得上「一个阈值」。类别数未知（旧运行没有该产物）时先放开，
  // 真不适用时后端会返回可读报错 —— 比前端猜错了把入口锁死更可靠。
  const thresholdUsable = isClassification && (nClasses === null || nClasses === 2);
  const multiclassBlocked = isClassification && nClasses !== null && nClasses !== 2;

  const thresholdValue = thresholdInput.trim() === "" ? null : Number(thresholdInput);
  const thresholdInvalid =
    thresholdValue !== null &&
    (!Number.isFinite(thresholdValue) || thresholdValue <= 0 || thresholdValue >= 1);

  /** 当前实际生效的阈值：本次推理结果优先，其次输入框，最后默认值。 */
  const effectiveThreshold = result?.threshold ?? thresholdValue ?? defaultThreshold;

  // 曲线优先级：留出测试集（选阈值的正当依据）> 推理数据（仅在前者缺失时兜底）
  const runCurve: MlThresholdCurve | null = artifacts.threshold_curve ?? null;
  const curve: MlThresholdCurve | null =
    runCurve && runCurve.points?.length ? runCurve : (result?.threshold_curve ?? null);

  /** 曲线里离当前阈值最近的一档，用于在表里标出「你现在在哪」。 */
  const nearestIndex = useMemo(() => {
    if (!curve?.points?.length) return -1;
    let best = 0;
    let bestGap = Number.POSITIVE_INFINITY;
    curve.points.forEach((p, i) => {
      const gap = Math.abs(p.threshold - effectiveThreshold);
      if (gap < bestGap) {
        bestGap = gap;
        best = i;
      }
    });
    return best;
  }, [curve, effectiveThreshold]);

  async function run() {
    if (!activeRunId || thresholdInvalid) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await predictWithRun({
          run_id: activeRunId,
          dataset_id: datasetId ?? undefined,
          limit,
          threshold: thresholdUsable ? thresholdValue : null,
        }),
      );
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "推理失败");
    } finally {
      setBusy(false);
    }
  }

  if (!successRuns.length) {
    return <div className="muted">还没有成功的训练运行，先完成一次训练后再推理。</div>;
  }

  const previewColumns = result
    ? [...result.feature_columns, result.prediction_column, ...result.probability_columns]
    : [];
  const changedRatio = result?.label_changed_ratio ?? null;

  return (
    <div>
      <div className="form-row" style={{ alignItems: "flex-end", gap: "var(--space-3)", flexWrap: "wrap" }}>
        <label className="field" style={{ marginBottom: 0 }}>
          训练运行
          <select value={activeRunId ?? ""} onChange={(e) => setRunId(Number(e.target.value))}>
            {successRuns.map((r) => (
              <option key={r.id} value={r.id}>
                #{r.id} · {r.artifacts?.model ? modelLabel(String(r.artifacts.model)) : ""} · {String(r.artifacts?.task ?? "")}
              </option>
            ))}
          </select>
        </label>

        {isClassification && (
          <label className="field" style={{ marginBottom: 0 }}>
            <span className="ml-infer-label">
              决策阈值
              <ParamHint spec={thresholdSpec} />
            </span>
            <input
              type="number"
              min={step}
              max={1 - step}
              step={step}
              placeholder={`${defaultThreshold}（默认）`}
              disabled={!thresholdUsable}
              value={thresholdInput}
              onChange={(e) => setThresholdInput(e.target.value)}
              style={{ width: 150 }}
            />
          </label>
        )}

        <label className="field" style={{ marginBottom: 0 }}>
          <span className="ml-infer-label">
            预览行数
            <ParamHint spec={limitSpec} />
          </span>
          <input
            type="number"
            min={1}
            max={200}
            value={limit}
            onChange={(e) => setLimit(Math.max(1, Math.min(200, Number(e.target.value) || 20)))}
            style={{ width: 110 }}
          />
        </label>

        <button
          className="btn primary"
          disabled={busy || !activeRunId || !datasetId || thresholdInvalid}
          onClick={() => void run()}
        >
          {busy ? "推理中..." : "执行推理"}
        </button>

        {thresholdInput.trim() !== "" && (
          <button
            type="button"
            className="btn"
            style={{ padding: "5px 12px" }}
            onClick={() => setThresholdInput("")}
          >
            恢复默认阈值
          </button>
        )}
      </div>

      {multiclassBlocked && (
        <p className="muted">
          该模型有 {nClasses} 个类别，单个阈值无法表达多类之间的取舍，因此不提供这项调整。
        </p>
      )}
      {thresholdInvalid && <p style={{ color: "var(--danger)" }}>阈值需在 0 与 1 之间（不含两端）。</p>}
      {!datasetId && <p className="muted">请先选择用于推理的数据集。</p>}
      {error && <p style={{ color: "var(--danger)" }}>{error}</p>}

      {result && (
        <div className="mt">
          <div className="flex-between">
            <span>
              运行 #{result.run_id} · {modelLabel(result.model)}
              <span className="badge success" style={{ marginLeft: "var(--space-2)" }}>
                {result.pipeline_applied ? "已应用预处理管道" : "未应用预处理"}
              </span>
            </span>
            <span className="muted">
              {result.row_count} 行 · 耗时 {result.runtime.toFixed(3)}s
            </span>
          </div>

          {/* 阈值回执：改没改、正类是谁、到底改动了多少样本，一眼可查。
              「阈值设了但没生效」是最难自查的情况，所以这里必须回执而不是静默。 */}
          {result.threshold_supported && (
            <div className="ml-infer-receipt">
              <span>
                决策阈值{" "}
                <b>{result.threshold ?? result.threshold_default ?? defaultThreshold}</b>
                {result.threshold === null || result.threshold === undefined ? "（默认）" : ""}
              </span>
              <span>
                正类 <b>{String(result.positive_class ?? "—")}</b>
              </span>
              <span>
                标签变化 <b>{fmtPct(changedRatio)}</b>
                {changedRatio === 0 ? "（与默认行为等价）" : ""}
              </span>
            </div>
          )}

          <div className="kv-grid mt">
            <div className="kv-item">
              <div className="k">预测列</div>
              <div className="v">{result.prediction_column}</div>
            </div>
            <div className="kv-item">
              <div className="k">概率列</div>
              <div className="v">{result.probability_columns.length ? result.probability_columns.join(" / ") : "无"}</div>
            </div>
            <div className="kv-item">
              <div className="k">模型特征数</div>
              <div className="v">{result.model_features.length}</div>
            </div>
          </div>

          <div className="mt" style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  {previewColumns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.preview.map((row, i) => (
                  <tr key={i}>
                    {previewColumns.map((c) => (
                      <td key={c}>{cellText(row[c])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.row_count > result.preview.length && (
            <p className="muted">仅显示前 {result.preview.length} 行，共 {result.row_count} 行。</p>
          )}
        </div>
      )}

      {/* 阈值取舍曲线：把「该定多少」从拍脑袋变成看数字选。
          训练时已在留出测试集上算好，这里直接展示，不需要为了试阈值重训一次。 */}
      {curve && curve.points.length > 0 && (
        <div className="mt">
          <div className="flex-between">
            <span className="ml-infer-curve-title">
              阈值取舍依据
              <span className="ml-infer-basis">{BASIS_LABEL[String(curve.basis)] ?? "来源未知"}</span>
            </span>
            {typeof curve.suggested_threshold === "number" && (
              <button
                type="button"
                className="btn"
                style={{ padding: "4px 10px" }}
                onClick={() => setThresholdInput(String(curve.suggested_threshold))}
              >
                采用 F1 最高阈值 {curve.suggested_threshold}
              </button>
            )}
          </div>
          {curve.metric_average === "positive" && (
            <p className="muted">
              表中为正类（{String(curve.positive_class ?? "—")}）口径，与上方指标卡的 macro 口径不同，不要直接对比。
              {curve.basis_rows ? ` 依据样本 ${curve.basis_rows} 行。` : ""}
            </p>
          )}
          <div style={{ overflowX: "auto" }}>
            <table className="data-table ml-threshold-table">
              <thead>
                <tr>
                  <th>阈值</th>
                  <th>精确率</th>
                  <th>召回率</th>
                  <th>F1</th>
                  <th>判正类比例</th>
                </tr>
              </thead>
              <tbody>
                {curve.points.map((p, i) => (
                  <tr key={p.threshold} className={i === nearestIndex ? "is-current" : undefined}>
                    <td>
                      {p.threshold}
                      {i === nearestIndex && <span className="ml-guide-tier">当前</span>}
                    </td>
                    <td>{p.precision.toFixed(3)}</td>
                    <td>{p.recall.toFixed(3)}</td>
                    <td>{p.f1.toFixed(3)}</td>
                    <td>{p.positive_rate.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
