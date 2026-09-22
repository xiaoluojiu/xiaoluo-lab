import { useEffect, useMemo, useState } from "react";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { TaskSelector, modelsForTask } from "../../features/ml/TaskSelector";
import { FeatureSelector } from "../../features/ml/FeatureSelector";
import { ModelSelector } from "../../features/ml/ModelSelector";
import { ExperimentResult } from "../../features/ml/ExperimentResult";
import { ModelComparison } from "../../features/ml/ModelComparison";
import { PredictionPanel } from "../../features/ml/PredictionPanel";
import { TrainingProgress } from "../../features/ml/TrainingProgress";
import { ParamGuide, ParamHint, findParamSpec } from "../../features/ml/ParamGuide";
import { ParamTuner } from "../../features/ml/ParamTuner";
import { PreprocessPanel, type PreprocessValue } from "../../features/ml/PreprocessPanel";
import { TuningAdvice } from "../../features/ml/TuningAdvice";
import { TrainTrace } from "../../features/ml/TrainTrace";
import { getSchema, previewDataset } from "../../api/datasets";
import { compareRuns, getExperimentRuns, getMlCatalog, listExperiments, listModels, trainModelStream } from "../../api/ml";
import type { MlCatalog, MlModel, MlParamCombo, TrainResult, TrainProgress, CompareResult, Experiment, ExperimentRun, TrainArtifacts } from "../../types/ml";
import type { SchemaColumn } from "../../types/dataset";
import { modelLabel, modelDesc, recommendParams } from "../../features/ml/modelMeta";
import { PageHeader } from "../../components/PageHeader";

// 训练进度阶段顺序（与后端 _TRAIN_STAGE_ORDER 对齐；阶段名来自 /ml/catalog）。
const TRAIN_STAGE_IDS = ["load", "feature_select", "split", "preprocess", "train", "evaluate", "persist"];

// 机器学习：数据集 → 任务 → 特征/目标 → 模型/参数 → 训练 → 多 run 对比。
export default function ML() {
  const [datasetIds, setDatasetIds] = useState<number[]>([]);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [models, setModels] = useState<MlModel[]>([]);
  const [catalog, setCatalog] = useState<MlCatalog | null>(null);
  const [task, setTask] = useState("classification");
  const [model, setModel] = useState("");
  const [target, setTarget] = useState<string | null>(null);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [recommendedParams, setRecommendedParams] = useState<Record<string, unknown>>({});
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  // 预处理配置：留空即由后端按数据画像自动生成（与改动前的行为完全一致）
  const [preprocessing, setPreprocessing] = useState<PreprocessValue>({});
  const [testSize, setTestSize] = useState<number | null>(null);
  const [sampleRows, setSampleRows] = useState<Record<string, unknown>[]>([]);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runsByExp, setRunsByExp] = useState<Record<number, ExperimentRun[]>>({});
  const [selectedRuns, setSelectedRuns] = useState<number[]>([]);
  const [compare, setCompare] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [seed, setSeed] = useState<number | null>(42);
  const [error, setError] = useState<string | null>(null);
  const [trainProgress, setTrainProgress] = useState<TrainProgress[]>([]);

  const datasetId = datasetIds[0];
  const taskModels = modelsForTask(models, task);
  // 训练进度条骨架：从教学目录取训练阶段（load~persist），聚类任务去掉 split
  const trainStages = (catalog?.pipeline_steps ?? [])
    .filter((s) => TRAIN_STAGE_IDS.includes(s.id))
    .filter((s) => !(task === "clustering" && s.id === "split"))
    .map((s) => ({ id: s.id, name: s.name }));
  // 可用于推理的成功运行：本次训练结果 + 已加载的历史运行
  const successRuns = (() => {
    const list: ExperimentRun[] = [];
    if (trainResult?.run.status === "success") list.push(trainResult.run);
    for (const run of Object.values(runsByExp).flat()) {
      if (run.status === "success" && !list.some((r) => r.id === run.id)) list.push(run);
    }
    return list;
  })();

  // 无效参数组合（如 penalty=l1 必须配 liblinear/saga）：与其等 sklearn 抛英文错，
  // 不如在点训练之前就说明白。取值口径＝已设置值 → 否则该参数的真实默认值。
  const comboWarnings = useMemo<MlParamCombo[]>(() => {
    const combos = catalog?.param_combos ?? [];
    if (!combos.length || !model) return [];
    const specs = catalog?.models.find((m) => m.name === model)?.params ?? [];
    const specOf = new Map(specs.map((s) => [s.name, s]));
    const effective = (name: string) =>
      parameters[name] !== undefined ? parameters[name] : specOf.get(name)?.sklearn_default;
    const eq = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
    return combos
      .filter((c) => c.model === model)
      .filter((c) => {
        const triggered = Object.entries(c.when).every(([k, vals]) =>
          vals.some((v) => eq(v, effective(k))),
        );
        if (!triggered) return false;
        return Object.entries(c.require).some(
          ([k, vals]) => !vals.some((v) => eq(v, effective(k))),
        );
      });
  }, [catalog, model, parameters]);

  // 数据里没有对应类型的列时，该项设置了也不会生效 —— 提前说明，避免「调了没反应」
  const prepAvailability = useMemo(() => {
    const isCat = (d: string) => /string|categor|utf|object/i.test(d);
    const isNum = (d: string) => /int|float|decimal|double|date|datetime|bool/i.test(d);
    return {
      missing: true,
      encoding: columns.some((c) => isCat(c.dtype)),
      scaling: columns.some((c) => isNum(c.dtype)),
    };
  }, [columns]);

  useEffect(() => {
    listModels().then(setModels).catch(() => setModels([]));
    getMlCatalog().then(setCatalog).catch(() => setCatalog(null));
    listExperiments(1, 50).then((r) => setExperiments(r.items)).catch(() => setExperiments([]));
  }, []);

  useEffect(() => {
    setModel(taskModels[0]?.name ?? "");
    if (task === "clustering") setTarget(null);
  }, [task, models]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!columns.length) return;
    if (!target && task !== "clustering") {
      const recommended = task === "classification"
        ? columns.find((c) => c.unique_count <= 20 && !c.dtype.toLowerCase().includes("float"))?.column ?? null
        : columns.find((c) => c.dtype.toLowerCase().includes("float") && /target|label|price|score|amount|y/i.test(c.column))?.column
          ?? columns.find((c) => c.dtype.toLowerCase().includes("float"))?.column ?? null;
      if (recommended) setTarget(recommended);
    }
    const autoExcluded = columns
      .filter((c) => c.unique_count > 200 || c.unique_count === rowCount)
      .map((c) => c.column);
    setExcluded((prev) => (prev.length ? prev : autoExcluded));
  }, [columns, task, rowCount]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (model && columns.length) {
      const recommended = recommendParams(model, columns, target, rowCount);
      setRecommendedParams(recommended);
      setParameters(recommended);
    } else {
      setRecommendedParams({});
      setParameters({});
    }
  }, [model, target, columns, rowCount]);

  async function loadSchema() {
    if (!datasetId) return;
    setError(null);
    setColumns([]);
    setSampleRows([]);
    try {
      const schema = await getSchema(datasetId);
      setColumns(schema.columns);
      setRowCount(schema.row_count ?? 0);
      previewDataset(datasetId, { page: 1, page_size: 200 }).then((d) => setSampleRows(d.items)).catch(() => setSampleRows([]));
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载 Schema 失败");
    }
  }

  /** 把调参建议写回参数面板（只改建议涉及的那几项，其余保持不变）。 */
  function applyParamPatch(patch: Record<string, unknown>) {
    setParameters((prev) => ({ ...prev, ...patch }));
  }

  async function train() {
    if (!datasetId || !model) return;
    setBusy(true);
    setElapsed(0);
    setError(null);
    setTrainProgress([]);
    const startedAt = Date.now();
    const timer = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 200);
    try {
      const result = await trainModelStream(
        {
          dataset_id: datasetId,
          task,
          model,
          target_column: task === "clustering" ? null : target,
          parameters,
          preprocessing,
          excluded_columns: excluded,
          seed,
          test_size: testSize,
        },
        (p) => setTrainProgress((prev) => [...prev, p]),
      );
      setTrainResult(result);
      setCompare(null);
      if (result.run.status !== "success") {
        setError(result.run.error ?? "训练失败，请查看下方运行详情");
      }
      listExperiments(1, 50).then((r) => setExperiments(r.items)).catch(() => undefined);
    } catch (e) {
      setError(e instanceof Error ? e.message : "训练失败");
    } finally {
      window.clearInterval(timer);
      setBusy(false);
    }
  }

  async function toggleExpRuns(expId: number) {
    if (runsByExp[expId]) return;
    try {
      const runs = await getExperimentRuns(expId);
      setRunsByExp((prev) => ({ ...prev, [expId]: runs }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载运行失败");
    }
  }

  function toggleRun(runId: number) {
    setSelectedRuns((prev) => prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]);
    setCompare(null);
  }

  async function doCompare() {
    if (selectedRuns.length < 2) return;
    setError(null);
    try {
      setCompare(await compareRuns(selectedRuns));
    } catch (e) {
      setError(e instanceof Error ? e.message : "对比失败");
    }
  }

  return (
    <div>
      <PageHeader title="机器学习" description="选择数据集与任务，配置特征与模型后训练，并对比多次实验的运行结果。" />

      <DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} />

      <div className="card">
        <h3>训练配置</h3>
        <button className="btn" onClick={() => void loadSchema()} disabled={!datasetId}>加载列信息</button>
        {columns.length > 0 && (
          <div className="mt">
            <TaskSelector value={task} onChange={setTask} />
            <div className="mt">
              <FeatureSelector columns={columns} task={task} target={task === "clustering" ? null : target} onTargetChange={setTarget} excluded={excluded} onExcludedChange={setExcluded} />
            </div>
            <div className="mt">
              <h4>模型</h4>
              <ModelSelector models={taskModels} value={model} onChange={setModel} />
              {model && modelDesc(model) && (
                <p className="ml-model-desc">{modelDesc(model)}</p>
              )}
            </div>
            <div className="mt">
              <ParamTuner
                catalog={catalog}
                model={model}
                values={parameters}
                recommended={recommendedParams}
                onChange={setParameters}
                comboWarnings={comboWarnings}
              />
            </div>
            <div className="mt">
              <PreprocessPanel
                catalog={catalog}
                value={preprocessing}
                onChange={setPreprocessing}
                availability={prepAvailability}
              />
            </div>
            <div className="form-row mt" style={{ alignItems: "flex-start", gap: "var(--space-3)" }}>
              <label className="field ml-param-field" style={{ marginBottom: 0 }}>
                <span className="ml-param-label">
                  测试集比例 test_size
                  <ParamHint spec={findParamSpec(catalog, "test_size")} />
                </span>
                <input
                  type="number"
                  step="0.05"
                  min="0.05"
                  max="0.95"
                  placeholder="0.2"
                  value={testSize ?? ""}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (!raw.trim()) { setTestSize(null); return; }
                    const n = Number.parseFloat(raw);
                    setTestSize(Number.isFinite(n) ? n : null);
                  }}
                  style={{ width: 140 }}
                />
              </label>
              <label className="field ml-param-field" style={{ marginBottom: 0 }}>
                <span className="ml-param-label">
                  随机种子 seed
                  <ParamHint spec={findParamSpec(catalog, "seed")} />
                </span>
                <input
                  type="number"
                  value={seed ?? ""}
                  onChange={(e) => setSeed(e.target.value === "" ? null : Number(e.target.value))}
                  style={{ width: 140 }}
                />
              </label>
              {taskModels.find((m) => m.name === model)?.supports_random_state === false && (
                <span className="muted ml-param-note">
                  当前模型不支持随机种子，seed 仅用于数据切分
                </span>
              )}
            </div>
            {testSize !== null && (testSize <= 0 || testSize >= 1) && (
              <p className="ml-param-error">test_size 必须在 0 与 1 之间（不含端点），当前 {testSize}。</p>
            )}
            <button className="btn primary mt" disabled={busy || !model || (task !== "clustering" && !target) || (testSize !== null && (testSize <= 0 || testSize >= 1))} onClick={() => void train()}>
              {busy ? "训练中..." : "开始训练"}
            </button>
            <TrainingProgress active={busy} elapsed={elapsed} stages={trainStages} completed={trainProgress} />
          </div>
        )}
        {error && <p style={{ color: "var(--danger)" }}>{error}</p>}
      </div>

      <div className="card">
        <h3>训练结果</h3>
        <ExperimentResult result={trainResult} columns={columns} sampleRows={sampleRows} target={target} />
        {trainResult?.run.status === "success" && (
          <TrainTrace
            artifacts={(trainResult.run.artifacts ?? {}) as TrainArtifacts}
            catalog={catalog}
            model={trainResult.experiment.model}
          />
        )}
        {trainResult?.run.status === "success" && (
          <TuningAdvice
            catalog={catalog}
            model={trainResult.experiment.model}
            artifacts={(trainResult.run.artifacts ?? {}) as TrainArtifacts}
            metrics={trainResult.run.metrics ?? {}}
            runtime={trainResult.run.runtime}
            onApply={applyParamPatch}
          />
        )}
      </div>

      <ParamGuide catalog={catalog} model={model} />

      <div className="card">
        <h3>模型推理</h3>
        <p className="muted">
          复用训练时保存的模型与预处理管道，对指定数据集执行批量预测。
        </p>
        <PredictionPanel runs={successRuns} datasetId={datasetId} catalog={catalog} />
      </div>

      <div className="card">
        <div className="ml-compare-toolbar">
          <div>
            <h3 style={{ margin: 0 }}>实验历史与运行选择</h3>
          </div>
          <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
            <button className="btn" disabled={!selectedRuns.length} onClick={() => { setSelectedRuns([]); setCompare(null); }}>清空选择</button>
            <button className="btn primary" disabled={selectedRuns.length < 2} onClick={() => void doCompare()}>对比选中运行（{selectedRuns.length}）</button>
          </div>
        </div>

        {selectedRuns.length > 0 && (
          <div className="ml-selected-runs">
            {selectedRuns.map((id) => <span className="ml-selected-run" key={id}>Run #{id}</span>)}
          </div>
        )}

        {!experiments.length && <div className="muted">暂无实验。先完成一次训练。</div>}
        {experiments.map((exp) => (
          <details key={exp.id} className="mt" onToggle={() => void toggleExpRuns(exp.id)}>
            <summary style={{ cursor: "pointer" }}>#{exp.id} · {exp.task} · {modelLabel(exp.model)}{exp.target_column ? ` · target=${exp.target_column}` : ""}</summary>
            <table className="data-table mt">
              <thead><tr><th>选择</th><th>Run</th><th>状态</th><th>指标</th><th>耗时(s)</th></tr></thead>
              <tbody>
                {(runsByExp[exp.id] ?? []).map((run) => (
                  <tr key={run.id}>
                    <td><input type="checkbox" checked={selectedRuns.includes(run.id)} onChange={() => toggleRun(run.id)} /></td>
                    <td>#{run.id}</td>
                    <td><span className={`badge ${run.status === "success" ? "success" : run.status === "failed" ? "failed" : "warning"}`}>{run.status}</span></td>
                    <td>{Object.entries(run.metrics ?? {}).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join("  ")}</td>
                    <td>{run.runtime != null ? run.runtime.toFixed(3) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ))}
      </div>

      <div className="card">
        <h3>运行对比</h3>
        <ModelComparison result={compare} />
      </div>
    </div>
  );
}
