import { useEffect, useMemo, useRef, useState } from "react";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { TaskSelector, modelsForTask } from "../../features/ml/TaskSelector";
import { FeatureSelector } from "../../features/ml/FeatureSelector";
import { ModelSelector } from "../../features/ml/ModelSelector";
import { ExperimentResult } from "../../features/ml/ExperimentResult";
import { ModelComparison } from "../../features/ml/ModelComparison";
import { PredictionPanel } from "../../features/ml/PredictionPanel";
import { TaskProgress } from "../../components/TaskProgress";
import { ParamGuide, ParamHint, findParamSpec } from "../../features/ml/ParamGuide";
import { ParamTuner } from "../../features/ml/ParamTuner";
import { PreprocessPanel, type PreprocessValue } from "../../features/ml/PreprocessPanel";
import { TuningAdvice } from "../../features/ml/TuningAdvice";
import { TrainTrace } from "../../features/ml/TrainTrace";
import { getSchema, previewDataset } from "../../api/datasets";
import { compareRuns, deleteExperiment, deleteExperiments, deleteRuns, getExperimentRuns, getMlCatalog, listExperiments, listModels, trainModelStream, ML_TRAIN_ESTIMATE } from "../../api/ml";
import type { MlCatalog, MlModel, MlParamCombo, TrainResult, TrainProgress, CompareResult, Experiment, ExperimentRun, TrainArtifacts } from "../../types/ml";
import type { SchemaColumn } from "../../types/dataset";
import { modelLabel, modelDesc, recommendParams, recommendTargetColumn, labelLeakWarnings, formatMetric } from "../../features/ml/modelMeta";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { ErrorNotice } from "../../components/ErrorNotice";
import { useToast } from "../../components/ToastProvider";
import { formatError, targetMissingValuesError, type AnalysisError } from "../../lib/analysisError";

// 训练进度阶段顺序（与后端 _TRAIN_STAGE_ORDER 对齐；阶段名来自 /ml/catalog）。
const TRAIN_STAGE_IDS = ["load", "feature_select", "split", "preprocess", "train", "evaluate", "persist"];

// 页面分区锚点：这一页从配置到对比有六屏，没有目录就只能靠滚轮找。
// id 与下方卡片的 id 一一对应，改一处必须改另一处（故集中在这里声明）。
const SECTION_ANCHORS: { id: string; label: string }[] = [
  { id: "ml-config", label: "训练配置" },
  { id: "ml-result", label: "训练结果" },
  { id: "ml-guide", label: "参数说明" },
  { id: "ml-predict", label: "模型推理" },
  { id: "ml-history", label: "实验历史" },
  { id: "ml-compare", label: "运行对比" },
];

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
  // 训练数据预算：auto=系统上限自动治理（默认，与旧行为一致）；rows=指定行数；fraction=指定占比
  const [dataLimitMode, setDataLimitMode] = useState<"auto" | "rows" | "fraction">("auto");
  const [maxRowsInput, setMaxRowsInput] = useState<number | null>(null);
  const [trainFracInput, setTrainFracInput] = useState<number | null>(null);
  // 学习曲线：需要额外最多 5 次拟合，默认关闭（不勾选时训练耗时与旧版完全一致）
  const [learningCurveOn, setLearningCurveOn] = useState(false);
  // 交叉验证：5 折 × 每折重新拟合预处理与模型，默认关闭（耗时约为普通训练的 5 倍）
  const [cvOn, setCvOn] = useState(false);
  const [sampleRows, setSampleRows] = useState<Record<string, unknown>[]>([]);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runsByExp, setRunsByExp] = useState<Record<number, ExperimentRun[]>>({});
  const [selectedRuns, setSelectedRuns] = useState<number[]>([]);
  const [compare, setCompare] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [seed, setSeed] = useState<number | null>(42);
  const [error, setError] = useState<AnalysisError | null>(null);
  const [trainProgress, setTrainProgress] = useState<TrainProgress[]>([]);
  // 训练任务自身的三态与失败原因：与页面级 error（Schema 加载、运行对比等）分开维护。
  // 混用会让「对比失败」把训练进度条染成红色，用户以为训练挂了。
  const [trainStatus, setTrainStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  // 训练失败原因：存结构化的三层信息（what/impact/solution），既供进度条取一句话，
  // 也供下方 ErrorNotice 展示完整影响与解决建议。
  const [trainError, setTrainError] = useState<AnalysisError | null>(null);
  // 实验历史区的删除：单条运行 / 批量运行 / 单个实验 / 全部实验。
  // 待删运行统一存成 id 列表 —— 单条与批量共用同一个确认框，
  // 两套入口各写一个对话框迟早会出现「单条删没有二次确认」这类不一致。
  const [pendingDeleteExp, setPendingDeleteExp] = useState<Experiment | null>(null);
  const [pendingDeleteExpAll, setPendingDeleteExpAll] = useState(false);
  const [pendingDeleteRunIds, setPendingDeleteRunIds] = useState<number[] | null>(null);
  const [deleting, setDeleting] = useState(false);
  const toast = useToast();
  // 「查看结果」的滚动目标：结果卡片在页面下方，训练完不定位过去用户看不到产出。
  const resultRef = useRef<HTMLDivElement>(null);

  const datasetId = datasetIds[0];
  const taskModels = modelsForTask(models, task);
  // 训练进度条骨架：从教学目录取训练阶段（load~persist），聚类任务去掉 split
  const trainStages = (catalog?.pipeline_steps ?? [])
    .filter((s) => TRAIN_STAGE_IDS.includes(s.id))
    .filter((s) => !(task === "clustering" && s.id === "split"))
    .map((s) => ({ id: s.id, name: s.name }));
  // —— 训练进度的派生口径（全部来自 /ml/train/stream 的 progress 事件，不伪造）——
  // 后端 `done` 的语义是「已完成阶段数（含当前）」⇒ 下一个待执行阶段的下标就是 done，
  // 于是「当前阶段」= done，「已完成阶段」= 事件里出现过的阶段 id。
  const latestTrainProgress = trainProgress[trainProgress.length - 1];
  const trainCurrentStage = latestTrainProgress ? latestTrainProgress.done : 0;
  const trainDoneStages = trainProgress.map((p) => p.stage);
  const trainStageDetails = Object.fromEntries(trainProgress.map((p) => [p.stage, p.detail]));
  const trainPercent =
    latestTrainProgress && latestTrainProgress.total > 0
      ? Math.min(100, Math.round((latestTrainProgress.done / latestTrainProgress.total) * 100))
      : null;
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
      const rec = recommendTargetColumn(columns, task);
      if (rec.column) setTarget(rec.column);
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
    setDataLimitMode("auto");
    setMaxRowsInput(null);
    setTrainFracInput(null);
    setLearningCurveOn(false);
    try {
      const schema = await getSchema(datasetId);
      setColumns(schema.columns);
      setRowCount(schema.row_count ?? 0);
      previewDataset(datasetId, { page: 1, page_size: 200 }).then((d) => setSampleRows(d.items)).catch(() => setSampleRows([]));
    } catch (e) {
      setError(formatError(e));
    }
  }

  /** 把调参建议写回参数面板（只改建议涉及的那几项，其余保持不变）。 */
  function applyParamPatch(patch: Record<string, unknown>) {
    setParameters((prev) => ({ ...prev, ...patch }));
  }

  /**
   * 目标列缺失值偏多的前端预检（后端不会主动报这个错，但它是训练质量的大坑）。
   * 命中后给出「建议先处理 + 去处理缺失值」的三层提示；**不阻断训练** ——
   * 后端会自动剔除目标列为空的行，用户仍可选择继续。
   */
  function targetMissingAdvisory(): AnalysisError | null {
    if (task === "clustering" || !target || rowCount <= 0) return null;
    const column = columns.find((c) => c.column === target);
    if (!column) return null;
    const missing = column.null_count ?? 0;
    if (missing <= 0) return null;
    // 缺失比例低于 20% 时剔除对应行无伤大雅，不必打扰用户。
    if (missing / rowCount < 0.2) return null;
    return targetMissingValuesError({ target, missing, total: rowCount });
  }

  async function train() {
    if (!datasetId || !model) return;
    setBusy(true);
    setElapsed(0);
    setError(null);
    // 预检：目标列缺失值过多时先给出「建议先处理 + 去处理缺失值」；训练仍继续（不阻断）。
    const advisory = targetMissingAdvisory();
    if (advisory) setError(advisory);
    setTrainError(null);
    setTrainStatus("running");
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
          max_rows: dataLimitMode === "rows" ? maxRowsInput : null,
          train_fraction: dataLimitMode === "fraction" ? trainFracInput : null,
          enable_learning_curve: learningCurveOn,
          enable_cv: cvOn,
        },
        (p) => setTrainProgress((prev) => [...prev, p]),
      );
      setTrainResult(result);
      setCompare(null);
      if (result.run.status === "success") {
        setTrainStatus("success");
      } else {
        // run 已落库为 failed：后端通常附了一段人能读的说明，优先用它。
        setTrainStatus("error");
        setTrainError(formatError(result.run.error ?? "训练失败，请查看下方运行详情"));
      }
      listExperiments(1, 50).then((r) => setExperiments(r.items)).catch(() => undefined);
    } catch (e) {
      // 网络中断 / 后端报错：失败原因 + 「重试」都留在进度条里。
      // 散落到页面别处的话，用户看到的是一行红字，不知道该点哪里重来。
      setTrainStatus("error");
      setTrainError(formatError(e));
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
      setError(formatError(e));
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
      setError(formatError(e));
    }
  }

  /**
   * 删除后重取实验列表 + 已展开实验的运行列表。
   *
   * 只重取「用户已经展开过」的实验：未展开的保持未展开，不替用户做选择。
   * 已删除实验的运行缓存会被丢弃，避免界面上残留打不开的幽灵条目。
   */
  async function refreshHistory() {
    const list = await listExperiments(1, 50)
      .then((r) => r.items)
      .catch(() => [] as Experiment[]);
    setExperiments(list);
    const alive = new Set(list.map((e) => e.id));
    const loaded = Object.keys(runsByExp).map(Number).filter((id) => alive.has(id));
    const pairs = await Promise.all(
      loaded.map(async (id) => [id, await getExperimentRuns(id).catch(() => [] as ExperimentRun[])] as const),
    );
    setRunsByExp(Object.fromEntries(pairs));
    // 对比结果里引用了「可能已被删掉」的运行，一律作废重来。
    setCompare(null);
  }

  /** 从选中集合中剔除已删除的运行（对比按钮不能拿着一批已经不存在的 id）。 */
  function dropSelectedRuns(removedRunIds: Set<number>) {
    setSelectedRuns((prev) => prev.filter((id) => !removedRunIds.has(id)));
  }

  /** 本次训练结果对应的运行被删掉后，结果卡片必须一并作废 ——
   *  它的模型产物已经从存储里清掉，继续展示会引导用户去点必然失败的「推理」。 */
  function invalidateTrainResultIfRemoved(removedRunIds: Set<number>) {
    if (trainResult && removedRunIds.has(trainResult.run.id)) {
      setTrainResult(null);
      setTrainStatus("idle");
      setTrainError(null);
      setTrainProgress([]);
    }
  }

  async function removeExperiment(exp: Experiment) {
    setPendingDeleteExp(null);
    setDeleting(true);
    setError(null);
    try {
      await deleteExperiment(exp.id);
      const removed = new Set((runsByExp[exp.id] ?? []).map((r) => r.id));
      dropSelectedRuns(removed);
      invalidateTrainResultIfRemoved(removed);
      toast.success(`实验 #${exp.id} 已删除`);
      await refreshHistory();
    } catch (e) {
      setError(formatError(e));
    } finally {
      setDeleting(false);
    }
  }

  async function removeAllExperiments() {
    setPendingDeleteExpAll(false);
    const targets = experiments.map((e) => e.id);
    if (!targets.length) return;
    setDeleting(true);
    setError(null);
    try {
      const result = await deleteExperiments(targets);
      const removed = new Set(Object.values(runsByExp).flat().map((r) => r.id));
      dropSelectedRuns(removed);
      invalidateTrainResultIfRemoved(removed);
      toast.success(`已删除 ${result.count} 个实验及其运行记录`);
      await refreshHistory();
    } catch (e) {
      setError(formatError(e));
    } finally {
      setDeleting(false);
    }
  }

  async function removeRuns(runIds: number[]) {
    if (!runIds.length) return;
    setDeleting(true);
    setError(null);
    try {
      const result = await deleteRuns(runIds);
      const removed = new Set(runIds);
      dropSelectedRuns(removed);
      invalidateTrainResultIfRemoved(removed);
      toast.success(`已删除 ${result.count} 条运行记录`);
      await refreshHistory();
    } catch (e) {
      setError(formatError(e));
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div>
      <PageHeader title="机器学习" description="选择数据集与任务，配置特征与模型后训练，并对比多次实验的运行结果。" />

      <nav className="ml-anchor-nav" aria-label="机器学习页分区导航">
        {SECTION_ANCHORS.map((s) => (
          <a key={s.id} className="ml-anchor-link" href={`#${s.id}`}>
            {s.label}
          </a>
        ))}
      </nav>

      <DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} />

      <div className="card ml-section" id="ml-config">
        <h3>训练配置</h3>
        <button className="btn" onClick={() => void loadSchema()} disabled={!datasetId}>加载列信息</button>
        {columns.length > 0 && (
          <div className="mt">
            <TaskSelector value={task} onChange={setTask} />
            <div className="mt">
              <FeatureSelector
                columns={columns}
                task={task}
                target={task === "clustering" ? null : target}
                onTargetChange={setTarget}
                excluded={excluded}
                onExcludedChange={setExcluded}
                suggestedExcluded={
                  excluded.length
                    ? []
                    : columns
                        .filter((c) => c.unique_count > 200 || c.unique_count === rowCount)
                        .map((c) => c.column)
                }
              />
              {task !== "clustering" &&
                labelLeakWarnings(columns, target, excluded, rowCount).map((w) => (
                  <p key={w} className="ml-tuner-warn" style={{ marginTop: "var(--space-1)" }}>⚠️ {w}</p>
                ))}
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
            <label className="field ml-param-field">
              <span className="ml-param-label">训练数据量</span>
              <select
                className="input"
                value={dataLimitMode}
                onChange={(e) => setDataLimitMode(e.target.value as "auto" | "rows" | "fraction")}
                style={{ width: 260 }}
              >
                <option value="auto">自动（超过 20 万行时随机抽样）</option>
                <option value="rows">指定行数</option>
                <option value="fraction">指定占比</option>
              </select>
            </label>
            {dataLimitMode === "rows" && (
              <label className="field ml-param-field">
                <span className="ml-param-label">行数上限</span>
                <input
                  type="number"
                  min="1"
                  step="1000"
                  placeholder="如 50000"
                  value={maxRowsInput ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    setMaxRowsInput(v.trim() === "" ? null : Number(v));
                  }}
                  style={{ width: 160 }}
                />
                <span className="muted ml-param-note">
                  {rowCount > 0 && maxRowsInput && maxRowsInput > 0
                    ? maxRowsInput >= rowCount
                      ? "行数 ≥ 数据总量，将使用全部数据"
                      : `约为全部数据的 ${((maxRowsInput / rowCount) * 100).toFixed(1)}%`
                    : `当前数据集共 ${rowCount.toLocaleString()} 行`}
                </span>
              </label>
            )}
            {dataLimitMode === "fraction" && (
              <label className="field ml-param-field">
                <span className="ml-param-label">训练数据占比</span>
                <input
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.05"
                  placeholder="0~1，如 0.5"
                  value={trainFracInput ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    setTrainFracInput(v.trim() === "" ? null : Number(v));
                  }}
                  style={{ width: 160 }}
                />
                <span className="muted ml-param-note">
                  {rowCount > 0 && trainFracInput && trainFracInput > 0
                    ? `约使用 ${Math.round(rowCount * trainFracInput).toLocaleString()} 行（切分前）`
                    : `当前数据集共 ${rowCount.toLocaleString()} 行`}
                </span>
              </label>
            )}
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
            <div
              className="form-row mt"
              style={{ alignItems: "center", gap: "var(--space-3)" }}
            >
              <label
                className="field"
                style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 8 }}
              >
                <input
                  type="checkbox"
                  checked={learningCurveOn}
                  onChange={(e) => setLearningCurveOn(e.target.checked)}
                />
                <span>生成学习曲线（最多额外拟合 5 次，训练耗时会增加）</span>
              </label>
              <label
                className="field"
                style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 8 }}
              >
                <input
                  type="checkbox"
                  checked={cvOn}
                  onChange={(e) => setCvOn(e.target.checked)}
                />
                <span>5 折交叉验证（耗时增加）</span>
              </label>
            </div>
            {cvOn && (
              <p className="muted ml-param-note">
                交叉验证在切分前的全量数据上做 5 折，每折都重新拟合预处理与模型
                （不复用整份数据的统计量），用于判断「换个切分还稳不稳」；
                它与上方测试集指标是两个独立口径，不是谁替代谁。
              </p>
            )}
            {testSize !== null && (testSize <= 0 || testSize >= 1) && (
              <p className="ml-param-error">test_size 必须在 0 与 1 之间（不含端点），当前 {testSize}。</p>
            )}
            <button className="btn primary mt" disabled={busy || !model || (task !== "clustering" && !target) || (testSize !== null && (testSize <= 0 || testSize >= 1)) || (dataLimitMode === "rows" && !(maxRowsInput && maxRowsInput > 0)) || (dataLimitMode === "fraction" && !(trainFracInput && trainFracInput > 0 && trainFracInput <= 1))} onClick={() => void train()}>
              {busy ? "训练中..." : "开始训练"}
            </button>
            {/* 分阶段训练进度：阶段骨架取自 /ml/catalog，进度由 /ml/train/stream 的真实事件推进 */}
            {trainStatus !== "idle" && (
              <TaskProgress
                className="mt"
                stages={trainStages}
                currentStage={trainCurrentStage}
                doneStages={trainDoneStages}
                progress={trainPercent}
                status={trainStatus}
                title="模型训练中"
                estimate={ML_TRAIN_ESTIMATE}
                elapsed={elapsed}
                stageDetails={trainStageDetails}
                error={trainError?.what ?? null}
                onViewResult={() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                onRetry={() => void train()}
              />
            )}
            {/* 训练失败的三层说明：进度条里只放一句话 + 重试，这里补全「影响 + 怎么解决」
                并在有明确去处时给一个可点击按钮。 */}
            {trainStatus === "error" && trainError && (
              <ErrorNotice error={trainError} title="训练没能完成" />
            )}
          </div>
        )}
        {error && <ErrorNotice error={error} compact />}
      </div>

      <div className="card ml-section" id="ml-result" ref={resultRef}>
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
            experimentId={trainResult.run.experiment_id}
            onApply={applyParamPatch}
          />
        )}
      </div>

      <div className="ml-section" id="ml-guide">
        <ParamGuide catalog={catalog} model={model} />
      </div>

      <div className="card ml-section" id="ml-predict">
        <h3>模型推理</h3>
        <p className="muted">
          复用训练时保存的模型与预处理管道，对指定数据集执行批量预测。
        </p>
        <PredictionPanel runs={successRuns} datasetId={datasetId} catalog={catalog} />
      </div>

      <div className="card ml-section" id="ml-history">
        <div className="ml-compare-toolbar">
          <div>
            <h3 style={{ margin: 0 }}>实验历史与运行选择</h3>
            <div className="muted" style={{ marginTop: "var(--space-1)" }}>
              勾选 2 条以上运行可对比指标；删除运行只清掉这一次记录，实验配置保留。
            </div>
          </div>
          <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
            <button className="btn" disabled={!selectedRuns.length} onClick={() => { setSelectedRuns([]); setCompare(null); }}>清空选择</button>
            <button className="btn primary" disabled={selectedRuns.length < 2} onClick={() => void doCompare()}>对比选中运行（{selectedRuns.length}）</button>
            <button
              className="btn danger"
              type="button"
              disabled={deleting || !selectedRuns.length}
              onClick={() => setPendingDeleteRunIds([...selectedRuns])}
            >
              删除选中运行（{selectedRuns.length}）
            </button>
            <button
              className="btn danger"
              type="button"
              disabled={deleting || !experiments.length}
              onClick={() => setPendingDeleteExpAll(true)}
            >
              一键删除全部实验（{experiments.length}）
            </button>
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
            <summary style={{ cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "var(--space-2)" }}>
              <span>#{exp.id} · {exp.task} · {modelLabel(exp.model)}{exp.target_column ? ` · target=${exp.target_column}` : ""}</span>
              {/* summary 内放按钮：不阻断默认行为的话，点「删除」会顺带展开/收起这个实验。 */}
              <button
                className="btn btn-sm danger"
                type="button"
                disabled={deleting}
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setPendingDeleteExp(exp); }}
              >
                删除实验
              </button>
            </summary>
            <table className="data-table mt">
              <thead><tr><th>选择</th><th>Run</th><th>状态</th><th>指标</th><th>耗时(s)</th><th>操作</th></tr></thead>
              <tbody>
                {(runsByExp[exp.id] ?? []).map((run) => (
                  <tr key={run.id}>
                    <td><input type="checkbox" aria-label={`选择运行 #${run.id} 参与对比`} checked={selectedRuns.includes(run.id)} onChange={() => toggleRun(run.id)} /></td>
                    <td>#{run.id}</td>
                    <td><span className={`badge ${run.status === "success" ? "success" : run.status === "failed" ? "failed" : "warning"}`}>{run.status}</span></td>
                    <td>{Object.entries(run.metrics ?? {}).map(([k, v]) => `${k}=${formatMetric(k, v)}`).join("  ")}</td>
                    <td>{run.runtime != null ? run.runtime.toFixed(3) : "-"}</td>
                    <td>
                      <button
                        className="btn btn-sm danger"
                        type="button"
                        disabled={deleting}
                        onClick={() => setPendingDeleteRunIds([run.id])}
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
                {!(runsByExp[exp.id] ?? []).length && (
                  <tr><td colSpan={6} className="muted">该实验暂无运行记录。</td></tr>
                )}
              </tbody>
            </table>
          </details>
        ))}
      </div>

      <div className="card ml-section" id="ml-compare">
        <h3>运行对比</h3>
        <ModelComparison result={compare} />
      </div>

      <ConfirmDialog
        open={pendingDeleteExp !== null}
        title="删除这个实验？"
        message={<>实验 #{pendingDeleteExp?.id} 的所有运行记录与模型产物会一并清理，不可恢复。</>}
        confirmText="删除"
        danger
        onConfirm={() => { if (pendingDeleteExp) void removeExperiment(pendingDeleteExp); }}
        onCancel={() => setPendingDeleteExp(null)}
      />
      <ConfirmDialog
        open={pendingDeleteRunIds !== null}
        title={
          pendingDeleteRunIds && pendingDeleteRunIds.length > 1
            ? `删除选中的 ${pendingDeleteRunIds.length} 条运行记录？`
            : `删除运行 #${pendingDeleteRunIds?.[0] ?? ""}？`
        }
        message={
          pendingDeleteRunIds && pendingDeleteRunIds.length > 1
            ? <>选中的 {pendingDeleteRunIds.length} 条运行记录及其模型产物将被清理；实验配置保留，仍可重新运行。不可恢复。</>
            : <>这次运行的记录与模型产物将被清理；实验配置保留，仍可重新运行。不可恢复。</>
        }
        confirmText={pendingDeleteRunIds && pendingDeleteRunIds.length > 1 ? `删除 ${pendingDeleteRunIds.length} 条` : "删除"}
        danger
        onConfirm={() => {
          const ids = pendingDeleteRunIds ?? [];
          setPendingDeleteRunIds(null);
          void removeRuns(ids);
        }}
        onCancel={() => setPendingDeleteRunIds(null)}
      />
      <ConfirmDialog
        open={pendingDeleteExpAll}
        title="删除全部实验？"
        message={<>将删除当前 {experiments.length} 个实验及其全部运行记录与模型产物，不可恢复。</>}
        confirmText="全部删除"
        danger
        onConfirm={() => void removeAllExperiments()}
        onCancel={() => setPendingDeleteExpAll(false)}
      />
    </div>
  );
}
