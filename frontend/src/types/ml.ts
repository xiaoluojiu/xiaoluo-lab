export interface MlModel {
  name: string;
  task?: string;
  desc?: string;
  supports_random_state?: boolean;
  supports_probability?: boolean;
}

/** 一次推理的返回结果。 */
export interface PredictionResult {
  run_id: number;
  experiment_id: number;
  model: string;
  task: string;
  row_count: number;
  prediction_column: string;
  probability_columns: string[];
  feature_columns: string[];
  model_features: string[];
  pipeline_applied: boolean;
  runtime: number;
  preview: Record<string, unknown>[];
  /** 该模型（二分类 + 有概率输出）是否支持调决策阈值。 */
  threshold_supported?: boolean;
  /** 本次推理实际使用的阈值；null 表示沿用默认行为。 */
  threshold?: number | null;
  /** 默认阈值（0.5），用于判断用户是否改过。 */
  threshold_default?: number;
  /** 正类标签（概率大于阈值即判为它），界面用它给控件命名。 */
  positive_class?: string | number | null;
  /** 换阈值后标签发生变化的样本占比（0 表示该阈值在当前数据上等价于默认行为）。 */
  label_changed_ratio?: number | null;
  /** 推理数据自带真实标签时的就地曲线；无标签则为 null。 */
  threshold_curve?: MlThresholdCurve | null;
}

/** 特征重要性解释结果。 */
export interface ExplainResult {
  source: "run_artifacts" | "model";
  run_id: number;
  method: string;
  importances: { feature: string; importance: number }[];
}

/** 分类混淆矩阵（行=真实，列=预测）。 */
export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface TrainResult {
  experiment: Experiment;
  run: ExperimentRun;
}

/** 训练流式进度事件（`POST /ml/train/stream` 的 progress 事件）。 */
export interface TrainProgress {
  /** 阶段 id（与后端 _TRAIN_STAGE_ORDER 对齐）。 */
  stage: string;
  /** 阶段中文名（来自 /ml/catalog 的 pipeline_steps，单一事实源）。 */
  label: string;
  /** 已完成阶段数（含当前）。 */
  done: number;
  /** 总阶段数（聚类任务无 split，比监督少一步）。 */
  total: number;
  /** 该阶段的一行真实摘要（透明化：展示实际数值）。 */
  detail: string;
}

export interface Experiment {
  id: number;
  dataset_id: number;
  dataset_version_id: number;
  task: string;
  model: string;
  target_column: string | null;
  parameters: Record<string, unknown>;
  preprocessing: Record<string, unknown>;
  seed: number | null;
  description: string;
  created_at: string | null;
  /** 仅 `with_metrics=true` 时返回：最近一次**成功**运行；从未成功运行过为 null。 */
  latest_run?: {
    run_id: number;
    status: string;
    metrics: Record<string, number | string>;
    runtime: number | null;
    created_at: string | null;
  } | null;
}

export interface ExperimentRun {
  id: number;
  experiment_id: number;
  status: "pending" | "running" | "success" | "failed";
  metrics: Record<string, number | string>;
  artifacts: Record<string, unknown>;
  runtime: number | null;
  error: string | null;
  created_at: string | null;
}

export interface CompareEntry {
  run_id: number;
  experiment_id?: number;
  status?: "pending" | "running" | "success" | "failed" | string;
  metrics?: Record<string, number | string>;
  runtime?: number | null;
  parameters?: Record<string, unknown>;
}

export interface CompareResult {
  entries?: CompareEntry[];
  best?: Record<string, number>;
  best_by_metric?: Record<string, number>;
  parameter_diff?: Record<string, Record<number, unknown>>;
  runtime_ranking?: number[];
  [key: string]: unknown;
}

/** ------------------------------------------------------------------
 * 教学元数据（`GET /ml/catalog`）
 * 后端单一事实源在 `app/ml_engine/metadata.py`，此处只做类型镜像。
 * 同一份数据还供 Agent 工具 `ml.explain_config` 与 `docs/ML_GUIDE.md` 使用。
 * ------------------------------------------------------------------ */

/** 枚举参数的候选项：label 给显示，value 是**真实传给 sklearn 的值**。 */
export interface MlParamOption {
  label: string;
  value: unknown;
}

/**
 * 单个参数的结构化说明。`default` / `range` / `typical` 可能是数字、字符串或数组，故为 unknown。
 *
 * 关键区别（渲染控件时必须区分）：
 * - `default` 是**人类可读的默认值**（如 "None（不限）"），只用于展示；
 * - `sklearn_default` 才是**可回填进请求**的真实值（如 null / 1.0 / true），
 *   用户启用某个可选参数时以它作为初始值。
 */
export interface MlParamSpec {
  name: string;
  label: string;
  type?: "int" | "float" | "enum" | "bool" | string;
  default?: unknown;
  sklearn_default?: unknown;
  range?: unknown;
  /** 枚举候选；缺省时回退为「range 是数组 ⇒ 直接当候选」。 */
  options?: MlParamOption[];
  /** core = 常显，最值得先调；advanced = 进阶，折叠后按需展开。 */
  tier?: "core" | "advanced";
  step?: number;
  /** 常用候选值，给输入框做快捷提示。 */
  typical?: unknown[];
  /** 依赖约束 {其它参数名: 期望值}，不满足时该项不可调。 */
  requires?: Record<string, unknown>;
  effect?: string;
  when_to_change?: string;
  code?: string;
  /** false 表示该项不需要前端控件（由别的组件负责，如 excluded_columns）。 */
  tunable?: boolean;
  /**
   * 以下三项由推理阶段参数（metadata.INFERENCE_PARAMS）使用，超参数不带。
   * 放在这里是为了让 ⓘ / 参数表这类共用渲染器不必区分两种 spec。
   */
  /** 依据说明：这个取值该怎么定、依据来自哪个数据集、口径有何不同。 */
  evidence?: string;
  /** true = 会改变预测结果；false = 只影响展示或性能。 */
  affects_result?: boolean;
  /** classification_binary = 仅二分类可用；all = 任意任务。 */
  applies_to?: string;
}

/** 流程中的一个环节，含「中间产物落在哪里」与可核验点。 */
export interface MlPipelineStep {
  id: string;
  name: string;
  summary: string;
  input: string;
  output: string;
  key_behaviour?: string[];
  verify?: string;
  code?: string;
}

/** 单个模型的教学条目（参数为该模型真实会透传给 sklearn 的常用参数）。 */
export interface MlModelGuide {
  name: string;
  task: string;
  note: string;
  params: MlParamSpec[];
}

/** 指标解读：是什么 / 怎么读 / 多少算好。 */
export interface MlMetricGuide {
  label: string;
  task: string;
  meaning: string;
  how_to_read: string;
  better: string;
}

export interface MlCatalogConventions {
  target_naming?: string[];
  default_test_size?: number;
  default_seed?: number;
  default_models?: Record<string, string>;
  [key: string]: unknown;
}

/** 一条可一键应用的调参动作。 */
export interface MlTuningAdvice {
  param: string;
  /** halve / double 基于当前值（未设置时取 sklearn_default）；set 直接设为 value。 */
  op: "halve" | "double" | "set";
  /** set 的目标值；halve / double 在当前值不可计算时作为兜底。 */
  value?: unknown;
  note?: string;
}

/**
 * 调参手册规则：把「指标不理想」翻译成「具体该动哪个参数」。
 * `signal` 由前端按 run 的指标与产物判定（见 features/ml/TuningAdvice.tsx）。
 */
export interface MlTuningRule {
  id: string;
  title: string;
  signal: string;
  detect: string;
  why: string;
  advice: MlTuningAdvice[];
  escalate?: string;
  /**
   * 不重新训练的替代路径：部分信号（如类别不均衡）既能靠改超参数解决，
   * 也能靠改推理阶段的决策阈值解决，而后者代价低一个数量级。
   */
  inference_alternative?: string;
}

/** 无效参数组合：选了 when 里的值就必须同时满足 require，否则 sklearn 直接报错。 */
export interface MlParamCombo {
  model: string;
  when: Record<string, unknown[]>;
  require: Record<string, unknown[]>;
  message: string;
  severity?: string;
}

/**
 * 推理阶段参数：训练完成之后才生效的旋钮（决策阈值等）。
 *
 * 与超参数的本质区别：
 * - 超参数改一次**必须重训**；推理参数改一次**立刻见效**；
 * - 字段 `affects_result` 显式区分「会改变预测结果」与「只影响展示/性能」——
 *   用户问「这个参数会不会影响结果」时界面必须能直接回答，而不是让人自己猜。
 *
 * 形状完全复用 `MlParamSpec`（故 ⓘ 注解与超参数同一套渲染），
 * 这里单独命名只是为了在 catalog 层面标出「它是另一类东西」。
 */
export interface MlInferenceParam extends MlParamSpec {}

/** 阈值取舍曲线上的一个档位。指标口径为正类，不是 run.metrics 的 macro。 */
export interface MlThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  /** 该阈值下被判为正类的样本占比。 */
  positive_rate: number;
}

/**
 * 阈值取舍曲线。
 * `basis` 标明依据来自哪个数据集：holdout_test = 训练时留出的测试集（代表泛化能力）；
 * inference_data = 推理数据本身（若它就是训练数据，指标会偏乐观，界面须提示）。
 */
export interface MlThresholdCurve {
  points: MlThresholdPoint[];
  /** 曲线不可用时的原因（非二分类、真实标签缺正类等），便于界面说明而非留白。 */
  reason?: string | null;
  positive_class?: string | number | null;
  basis_rows?: number;
  basis?: "holdout_test" | "inference_data" | string;
  /** 固定为 "positive"：与 run.metrics 的 macro 口径不同，避免被直接比较。 */
  metric_average?: string;
  suggested_threshold?: number | null;
  suggested_f1?: number | null;
}

export interface MlCatalog {
  pipeline_steps: MlPipelineStep[];
  preprocessing_params: MlParamSpec[];
  training_params: MlParamSpec[];
  models: MlModelGuide[];
  metrics: Record<string, MlMetricGuide>;
  tuning_playbook?: MlTuningRule[];
  param_combos?: MlParamCombo[];
  /** 训练后才生效的旋钮（决策阈值等），与 models[].params 分开返回。 */
  inference_params?: MlInferenceParam[];
  conventions?: MlCatalogConventions;
}

/** ------------------------------------------------------------------
 * 训练产物（`run.artifacts`）—— 透明化的载体。
 * 这里的字段由 `ExperimentService._execute` 组装，`preprocessing_report`
 * 来自 `PreprocessingPipeline.report`，`model_summary` 来自 `ModelAdapter.summary`。
 * ------------------------------------------------------------------ */

export interface PreprocessingReport {
  fitted?: boolean;
  features_out?: string[];
  config?: {
    missing?: Record<string, unknown> | null;
    encoding?: Record<string, unknown> | null;
    scaling?: Record<string, unknown> | null;
  };
}

export interface ModelSummary {
  name?: string;
  task?: string;
  params?: Record<string, unknown>;
  trained?: boolean;
  feature_names?: string[];
  n_features?: number;
  n_samples?: number;
  fit_seconds?: number;
}

/** 训练产物里与「怎么做的」相关的部分（用于结果区的透明化展示）。 */
export interface TrainArtifacts {
  model?: string;
  task?: string;
  /** 排除后实际参与训练的**原始**列清单。 */
  features?: string[];
  /** 预处理后真正喂给模型的列清单（类别编码会展开成多列）。 */
  model_features?: string[];
  excluded_columns?: string[];
  target_column?: string | null;
  train_rows?: number;
  test_rows?: number;
  test_size?: number;
  /** 因目标列为空而被剔除的行数。 */
  dropped_rows?: number;
  /** 分类任务是否按类别分层切分。 */
  stratified?: boolean;
  preprocessing_report?: PreprocessingReport;
  model_summary?: ModelSummary;
  feature_importance?: { method?: string; importances?: { feature: string; importance: number }[] } | null;
  confusion_matrix?: ConfusionMatrixData;
  per_class?: Record<string, unknown>;
  /**
   * 训练集指标（仅监督任务；聚类显式为 null）。
   * 它**不进 run.metrics**（那里始终是测试集口径），只用来算 train/test 差距，
   * 判断模型是「学过头」还是「还没学会」。
   */
  train_metrics?: Record<string, number> | null;
  /** 分类任务的目标列分布（按样本量降序），majority_ratio 用于判断类别不均衡。 */
  class_distribution?: {
    total: number;
    n_classes: number;
    majority_ratio: number;
    items: { label: string; count: number; ratio: number }[];
  } | null;
  /**
   * 决策阈值的依据曲线，在**留出测试集**上扫出（仅二分类）。
   * 它是「阈值定多少」的选型依据，口径为正类，与 run.metrics 的 macro 不同。
   */
  threshold_curve?: MlThresholdCurve | null;
  residual_stats?: {
    count?: number;
    mean?: number | null;
    std?: number | null;
    max_abs?: number | null;
    p50?: number | null;
    p95?: number | null;
    histogram?: { range: string; count: number }[];
    note?: string;
  };
  model_key?: string;
  pipeline_key?: string;
  [key: string]: unknown;
}
