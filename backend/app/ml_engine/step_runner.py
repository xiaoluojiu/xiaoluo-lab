"""ML 单步执行器（step runner）。

设计目标（对应用户诉求「每一步可独立运行与验证」）：
- 把 ML 流程拆成 4 个可单独调用的步骤：load / preprocess / train / predict。
- 每步都返回统一的 ``StepResult``：status / output / 中间产物 / 说明 / 可核验信息。
- 不依赖 FastAPI，可被 API、Agent 工具、脚本、单元测试直接调用。

与 ``ExperimentService`` 的关系：
- ``ExperimentService.run`` 是「一键全流程」，本模块是「分步可控」版本；
- 两者共用同一套 ``ml_engine`` 组件（预处理 / 模型 / 评估），不做第二套实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from app.ml_engine.exceptions import MLEngineException
from app.ml_engine.inference import batch_predict, batch_predict_proba
from app.ml_engine.preprocessing import (
    PreprocessingPipeline,
    build_pipeline,
    cap_training_rows,
    default_preprocessing_config,
)
from app.ml_engine.registry import MODEL_REGISTRY


@dataclass
class StepResult:
    """单步执行结果（统一结构，便于 UI / 日志 / 测试三处消费）。"""

    step: str
    status: str  # ok / failed
    output: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "output": self.output,
            "artifacts": self.artifacts,
            "note": self.note,
            "error": self.error,
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def _can_stratify(y: pl.Series | None) -> bool:
    """分类任务是否可分层切分（每类至少 2 样本）。与 ExperimentService 口径一致。"""
    if y is None:
        return False
    non_null = y.drop_nulls()
    if non_null.len() < 4 or non_null.n_unique() < 2:
        return False
    counts = non_null.value_counts()
    return bool(counts.height) and int(counts["count"].min()) >= 2


def _preflight(task: str, X: pl.DataFrame, y: pl.Series | None) -> str | None:
    """训练前校验；返回可读错误文案，通过则返回 None。

    与 ExperimentService._preflight 保持同一判断集合（此处返回字符串而非抛异常，
    因为 step runner 以 StepResult 表达失败）。
    """
    if X.width == 0:
        return "训练特征列为空（排除后无可用列）"
    if X.height < 2:
        return "训练样本数 < 2，无法训练"
    if y is None:
        return None
    if y.len() != X.height:
        return "X 与 y 行数不一致"
    if y.null_count() == y.len():
        return "目标列全为空"
    if task == "classification" and int(y.drop_nulls().n_unique()) < 2:
        return "分类任务至少需要 2 个类别"
    if task == "regression" and int(y.drop_nulls().n_unique()) < 2:
        return "目标列无变化（所有取值相同），回归模型无法学习"
    return None


# ----------------------------------------------------------------------
# 步骤 1：数据输入
# ----------------------------------------------------------------------
def run_load_step(
    df: pl.DataFrame,
    *,
    target: str | None = None,
    excluded_columns: list[str] | None = None,
) -> StepResult:
    """校验数据可用性并划分 X / y，不训练任何模型。

    可独立验证：给定任意 DataFrame 即可调用，返回数据规模、列类型画像、
    以及排除特征后的实际 X 列清单——这是「数据是否适合建模」的第一步体检。
    """
    excluded = [c for c in (excluded_columns or [])]
    missing_excluded = [c for c in excluded if c not in df.columns]
    if missing_excluded:
        return StepResult(
            step="load",
            status="failed",
            error=f"排除列不存在：{missing_excluded}",
            artifacts={"columns": list(df.columns)},
        )
    if target is not None and target not in df.columns:
        return StepResult(
            step="load",
            status="failed",
            error=f"目标列 {target!r} 不存在",
            artifacts={"columns": list(df.columns)},
        )
    if target is not None and target in excluded:
        # 目标列本就不参与特征（下面的 feature_cols 已按 `c != target` 过滤），
        # 因此「把目标列列进排除列」是幂等的冗余指令，不是错误。
        # 这里自动剔除而不是让整条链路失败 —— 与 ExperimentService._execute 同一口径。
        excluded = [c for c in excluded if c != target]

    feature_cols = [c for c in df.columns if c != target and c not in excluded]
    schema_profile = [
        {
            "column": c,
            "dtype": str(df.schema[c]),
            "null_count": int(df[c].null_count()),
            "n_unique": int(df[c].n_unique()),
            "is_numeric": bool(df.schema[c].is_numeric()),
        }
        for c in df.columns
    ]
    empty_label_rows = int(df[target].null_count()) if target else 0
    return StepResult(
        step="load",
        status="ok",
        output={
            "row_count": df.height,
            "column_count": df.width,
            "target": target,
            "feature_columns": feature_cols,
        },
        artifacts={
            "schema_profile": schema_profile,
            "empty_label_rows": empty_label_rows,
        },
        note=(
            f"{df.height} 行 × {df.width} 列；目标列 {target or '（无，聚类）'}；"
            f"参与训练的特征 {len(feature_cols)} 列"
            + (f"；目标列有 {empty_label_rows} 个空值（训练时会剔除对应行）" if empty_label_rows else "")
        ),
    )


# ----------------------------------------------------------------------
# 步骤 2：预处理（可独立 fit 并观察变换前后）
# ----------------------------------------------------------------------
def run_preprocess_step(
    df: pl.DataFrame,
    *,
    target: str | None = None,
    excluded_columns: list[str] | None = None,
    config: dict[str, Any] | None = None,
    preview_rows: int = 5,
) -> StepResult:
    """构造并拟合预处理管道，输出变换前后对照。

    可独立验证：不训练模型，只看「原始列 → 模型特征」的映射是否符合预期，
    例如类别列是否被 one-hot 展开、数值列是否被标准化、空值是否被填补。
    """
    loaded = run_load_step(df, target=target, excluded_columns=excluded_columns)
    if loaded.status != "ok":
        return StepResult(step="preprocess", status="failed", error=loaded.error)

    feature_cols = loaded.output["feature_columns"]
    X = df.select(feature_cols)
    effective_cfg = config if isinstance(config, dict) and config else default_preprocessing_config(X)
    # 本步只需要「列映射 + 前 N 行对照」，却要 fit 整份数据。
    # 而 one-hot 的 fit 会物化整份变换结果（10M 行 × 767 列 = 57 GiB）。
    # ⇒ 在大数据上按上限抽样来 fit（统计量在抽样上足够稳），
    #   transform 也只在抽样集上做，不再为 5 行预览付出全量代价。
    X_fit, _, sampling = cap_training_rows(X)
    pipeline = build_pipeline(effective_cfg, X_fit).fit(X_fit)
    Xp = pipeline.transform(X_fit)

    # 变换前后对照：取前 N 行原始特征与模型特征
    before = X.head(preview_rows).to_dicts()
    after = Xp.head(preview_rows).to_dicts()
    return StepResult(
        step="preprocess",
        status="ok",
        output={
            "input_features": feature_cols,
            "output_features": list(pipeline.feature_names_out_),
            "input_feature_count": len(feature_cols),
            "output_feature_count": len(pipeline.feature_names_out_),
        },
        artifacts={
            "config_used": effective_cfg,
            "pipeline_report": pipeline.report,
            "before_preview": before,
            "after_preview": after,
            "sampling": sampling,
        },
        note=(
            f"预处理后特征由 {len(feature_cols)} 列变为 {len(pipeline.feature_names_out_)} 列"
            f"（{'含' if effective_cfg.get('encoding') else '不含'}类别编码，"
            f"{'含' if effective_cfg.get('scaling') else '不含'}数值缩放"
            + (
                f"；因数据量 {sampling['original_rows']:,} 行，管道仅在 "
                f"{sampling['used_rows']:,} 行抽样上拟合）"
                if sampling["sampled"]
                else "）"
            )
        ),
    )


# ----------------------------------------------------------------------
# 步骤 3：训练（可独立运行，返回指标 + 决策依据）
# ----------------------------------------------------------------------
def run_train_step(
    df: pl.DataFrame,
    *,
    task: str,
    model: str,
    target: str | None = None,
    parameters: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    excluded_columns: list[str] | None = None,
    test_size: float = 0.2,
    seed: int | None = 42,
) -> StepResult:
    """端到端训练一步完成：切分 → 预处理 → 拟合 → 评估。

    可独立验证：返回指标、切分规模、实际生效的预处理配置、模型摘要，
    足以判断「这次训练是否可信」。
    """
    from app.ml_engine.evaluation import (
        classification_report,
        confusion_matrix,
        evaluate_classification,
        evaluate_clustering,
        evaluate_regression,
        regression_residuals,
    )

    loaded = run_load_step(df, target=target, excluded_columns=excluded_columns)
    if loaded.status != "ok":
        return StepResult(step="train", status="failed", error=loaded.error)
    if not (0 < float(test_size) < 1):
        return StepResult(step="train", status="failed", error="test_size 必须在 (0,1) 区间")

    try:
        MODEL_REGISTRY.get(model)
    except MLEngineException as exc:
        return StepResult(step="train", status="failed", error=str(exc), artifacts=exc.details)

    feature_cols = loaded.output["feature_columns"]
    if not feature_cols:
        return StepResult(step="train", status="failed", error="没有可用特征列")
    X = df.select(feature_cols)
    y = df[target] if target else None

    dropped_rows = 0
    if y is not None and y.null_count() > 0:
        keep = y.is_not_null()
        dropped_rows = int((~keep).sum())
        X, y = X.filter(keep), y.filter(keep)

    if X.height < 2:
        return StepResult(step="train", status="failed", error="训练样本数 < 2，无法训练")

    # 与 ExperimentService 同一套 preflight 口径：把 sklearn 难懂错误提前转成可读提示
    task_of_model = MODEL_REGISTRY.get(model).task
    effective_task = "clustering" if (task == "clustering" or task_of_model == "clustering") else task
    preflight_error = _preflight(effective_task, X, y)
    if preflight_error:
        return StepResult(step="train", status="failed", error=preflight_error)

    # 训练集规模治理（与 ExperimentService 同一口径）：超限即随机抽样，
    # 并把抽样信息写进 artifacts —— 静默改训练集规模会让指标无法解释。
    X, y, sampling = cap_training_rows(X, y, seed=seed)

    try:
        pipeline = build_pipeline(preprocessing, X)
        model_obj = MODEL_REGISTRY.create(model, dict(parameters or {}))
        # seed 注入 random_state
        if seed is not None and MODEL_REGISTRY.supports_param(model, "random_state"):
            model_obj.params.setdefault("random_state", seed)
            model_obj = MODEL_REGISTRY.create(model, model_obj.params)

        details: dict[str, Any] = {}
        details["sampling"] = sampling
        if effective_task == "clustering":
            Xp = pipeline.fit_transform(X)
            model_obj.fit(Xp)
            metrics = evaluate_clustering(Xp, model_obj.labels_)
            train_rows, test_rows, stratified = X.height, 0, False
        else:
            # 与 ExperimentService 同口径：分层切分只对分类任务有意义，
            # 回归任务按类别分层会变成「要求测试集覆盖目标的所有取值」并莫名失败。
            stratify = effective_task == "classification" and _can_stratify(y)
            # 分层切分的硬约束：每类样本数 × test_size 至少能分到 1 个测试样本，
            # 否则 sklearn 会抛难懂的 "test_size = N should be greater or equal to
            # the number of classes"。此处给出可操作的中文提示而非裸错。
            # 注意：sklearn 内部按 ceil(N * test_size) 计算测试集大小，这里必须同样向上取整。
            if stratify and y is not None:
                n_classes = int(y.drop_nulls().n_unique())
                n_test = math.ceil(X.height * float(test_size))
                if n_test < n_classes:
                    return StepResult(
                        step="train",
                        status="failed",
                        error=(
                            f"测试集样本不足以覆盖 {n_classes} 个类别（当前仅有 "
                            f"{X.height} 行、test_size={test_size}，仅能划分 {n_test} 个测试样本）。"
                            f"请调大 test_size 或补充数据。"
                        ),
                    )
            try:
                X_train, X_test, y_train, y_test = PreprocessingPipeline.train_test_split(
                    X, y, test_size=float(test_size), seed=seed, stratify=y if stratify else None
                )
            except MLEngineException as exc:
                return StepResult(step="train", status="failed", error=str(exc))
            Xtr = pipeline.fit_transform(X_train)
            Xte = pipeline.transform(X_test)
            model_obj.fit(Xtr, y_train)
            y_pred = model_obj.predict(Xte)
            proba = None
            try:
                proba = model_obj.predict_proba(Xte)
            except MLEngineException:
                proba = None
            if model_obj.task == "classification":
                metrics = evaluate_classification(y_test, y_pred, proba)
                details["confusion_matrix"] = confusion_matrix(y_test, y_pred)
                details["per_class"] = classification_report(y_test, y_pred)["per_class"]
            else:
                metrics = evaluate_regression(y_test, y_pred)
                details["residual_stats"] = regression_residuals(y_test, y_pred)
            train_rows, test_rows, stratified = X_train.height, X_test.height, stratify
    except Exception as exc:  # noqa: BLE001 - 统一转成可读失败
        return StepResult(step="train", status="failed", error=str(exc))

    model_summary = model_obj.summary()
    return StepResult(
        step="train",
        status="ok",
        output={
            "task": model_obj.task,
            "model": model,
            "metrics": metrics,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "test_size": float(test_size),
            "seed": seed,
            "stratified": stratified,
        },
        artifacts={
            "features": feature_cols,
            "model_features": list(pipeline.feature_names_out_),
            "preprocessing_report": pipeline.report,
            "model_summary": model_summary,
            "dropped_rows": dropped_rows,
            **details,
        },
        note=(
            f"{model} 训练完成：训练 {train_rows} 行 / 测试 {test_rows} 行"
            f"（test_size={test_size}, seed={seed}{'，分层切分' if stratified else ''}）"
        ),
    )


# ----------------------------------------------------------------------
# 步骤 4：推理（可独立运行）
# ----------------------------------------------------------------------
def run_predict_step(
    model_obj: Any,
    pipeline: PreprocessingPipeline | None,
    df: pl.DataFrame,
    *,
    feature_columns: list[str],
    limit: int = 20,
) -> StepResult:
    """用已训练模型 + 管道对新数据推理。

    可独立验证：给定模型对象与数据即可跑，返回预测预览、概率列、耗时。
    """
    import time

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return StepResult(
            step="predict",
            status="failed",
            error=f"推理数据缺少训练时的特征列：{missing}",
            artifacts={"required": feature_columns},
        )
    X_raw = df.select(feature_columns)
    started = time.perf_counter()
    try:
        # 推理不能抽样，只能分块（见 ml_engine.inference）：1000 万行 × 767 列的
        # 稠密矩阵是 57.1 GiB，一次性 transform + predict 必崩。
        prediction = batch_predict(model_obj, X_raw, pipeline=pipeline)
    except Exception as exc:  # noqa: BLE001
        return StepResult(step="predict", status="failed", error=str(exc))
    elapsed = round(time.perf_counter() - started, 6)

    preview = max(1, int(limit))
    payload = df.head(preview).to_dicts()
    for i, value in enumerate(prediction.head(preview).to_list()):
        payload[i]["prediction"] = _jsonable(value)

    probability_columns: list[str] = []
    if getattr(model_obj, "task", "") == "classification":
        try:
            proba = batch_predict_proba(model_obj, X_raw, pipeline=pipeline)
            probability_columns = list(proba.columns)
            for i, row in enumerate(proba.head(preview).to_dicts()):
                if i < len(payload):
                    payload[i].update({k: _jsonable(v) for k, v in row.items()})
        except MLEngineException:
            probability_columns = []

    return StepResult(
        step="predict",
        status="ok",
        output={
            "row_count": df.height,
            "prediction_column": "prediction",
            "probability_columns": probability_columns,
            "pipeline_applied": pipeline is not None,
            "runtime": elapsed,
            "preview": payload,
        },
        note=f"预测完成：{df.height} 行，耗时 {elapsed:.4f}s",
    )
