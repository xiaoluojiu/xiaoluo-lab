"""Prompt 081：ExperimentService。

负责实验的 create / run / get / list / compare。
实验与 DatasetVersion 严格绑定：run 时从绑定的版本快照读取数据，
配合 seed 保证实验可复现。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import sys
import time
from typing import Any, Callable

import numpy as np
import polars as pl
import sklearn
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import NotFoundException, ValidationException
from app.ml_engine.evaluation import (
    classification_report,
    confusion_matrix,
    evaluate_clustering,
    regression_residuals,
)
from app.ml_engine.evaluation import (
    evaluate_classification as eval_classification,
)
from app.ml_engine.evaluation import evaluate_regression as eval_regression
from app.ml_engine.exceptions import MLEngineException
from app.ml_engine.inference import batch_predict, batch_predict_proba
from app.ml_engine.metadata import PIPELINE_STEPS
from app.ml_engine.preprocessing import (
    PreprocessingPipeline,
    build_pipeline,
    cap_training_rows,
    default_preprocessing_config,
)
from app.ml_engine.registry import MODEL_REGISTRY
from app.ml_engine.threshold import (
    DEFAULT_THRESHOLD,
    apply_threshold,
    is_binary,
    label_shift_ratio,
    model_classes,
    threshold_curve,
)
from app.models.dataset_version import DatasetVersion
from app.models.experiment import Experiment
from app.models.experiment_run import ExperimentRun
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

TASKS = {"classification", "regression", "clustering"}

# 训练进度阶段顺序（id 与 metadata.PIPELINE_STEPS 对齐，保证「单一事实源」）。
# 聚类任务没有「训练/测试划分」，进度条里会跳过 split。
_TRAIN_STAGE_ORDER = [
    "load", "feature_select", "split", "preprocess", "train", "evaluate", "persist"
]
_STAGE_LABELS = {s["id"]: s["name"] for s in PIPELINE_STEPS}


def _metric_summary(task: str, metrics: dict[str, Any]) -> str:
    """把指标 dict 压成进度条上的一行可读摘要（透明化：展示真实数值）。"""
    if task == "classification":
        acc, f1 = metrics.get("accuracy"), metrics.get("f1")
        if acc is not None and f1 is not None:
            return f"accuracy={acc:.4f} / f1={f1:.4f}"
        return "分类评估完成"
    if task == "regression":
        rmse, r2 = metrics.get("rmse"), metrics.get("r2")
        if rmse is not None and r2 is not None:
            return f"rmse={rmse:.4f} / r2={r2:.4f}"
        return "回归评估完成"
    sil = metrics.get("silhouette")
    n_cluster = metrics.get("cluster_count")
    if sil is not None:
        return f"轮廓系数={sil:.4f}（{n_cluster} 簇）"
    return f"{n_cluster} 簇（轮廓系数不可用）"


def _class_distribution(y: pl.Series) -> dict[str, Any]:
    """训练集类别分布（按样本量降序），供前端判断「类别不均衡」。

    返回 majority_ratio 而不只是一个布尔，是为了让界面能直接解释
    「为什么 accuracy 很高但 f1 很低」——那是多数类占比撑起来的假象。
    """
    non_null = y.drop_nulls()
    counts = non_null.value_counts().sort("count", descending=True)
    total = int(counts["count"].sum()) if counts.height else 0
    items = [
        {
            "label": str(row.get(y.name)),
            "count": int(row["count"]),
            "ratio": round(int(row["count"]) / total, 4) if total else 0.0,
        }
        for row in counts.to_dicts()
    ]
    return {
        "total": total,
        "n_classes": len(items),
        "majority_ratio": items[0]["ratio"] if items else 0.0,
        "items": items[:20],
    }


def _jsonable(value: Any) -> Any:
    """numpy 标量 -> Python 原生类型，保证 JSON 序列化安全。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def library_versions() -> dict[str, str]:
    """软件版本快照。

    同一份数据 + 同一个 seed，在不同版本的 sklearn/polars 上也可能跑出不同结果
    （默认参数调整、数值实现变化）。记录版本不是形式主义，而是让「上次和这次
    指标不一样」这类问题有第一条可查的线索。
    """
    return {
        "python": sys.version.split()[0],
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def dataset_snapshot(version_row: DatasetVersion) -> dict[str, Any]:
    """实验所用**不可变数据快照**的指纹。

    回答的问题只有一个：这次结果究竟是用哪一份数据跑出来的。因此必须绑到
    ``DatasetVersion`` 而不是 ``dataset_id`` —— 数据集会持续产生新版本，
    只记 dataset_id 的话，三个月后回看完全无法还原当时的输入。

    刻意做**轻量指纹**（版本行元数据）而不是读整个 Parquet 算内容摘要：
    千万行表上那是一次全量 IO，而本地应用场景下「哪个版本 + 多少行 +
    什么 schema」已足以定位输入。
    """
    schema = version_row.schema_json or {}
    payload = "|".join(
        [
            str(version_row.id),
            str(version_row.dataset_id),
            str(version_row.version),
            str(version_row.row_count),
            str(version_row.column_count),
            json.dumps(schema, sort_keys=True, ensure_ascii=False),
        ]
    )
    return {
        "dataset_id": version_row.dataset_id,
        "dataset_version_id": version_row.id,
        "version": version_row.version,
        "row_count": version_row.row_count,
        "column_count": version_row.column_count,
        "fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }


class ExperimentService:
    """实验全生命周期管理。"""

    def __init__(self, db: Session, dataset_service: DatasetService | None = None):
        self.db = db
        self.dataset_service = dataset_service

    # ------------------------------------------------------------------
    # create / get / list
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        dataset_id: int,
        dataset_version_id: int,
        task: str,
        model: str,
        parameters: dict[str, Any] | None = None,
        preprocessing: dict[str, Any] | None = None,
        excluded_columns: list[str] | None = None,
        seed: int | None = None,
        description: str = "",
        target_column: str | None = None,
        test_size: float | None = None,
    ) -> Experiment:
        if task not in TASKS:
            raise ValidationException(
                f"task 必须是 {sorted(TASKS)} 之一", details={"task": task}
            )
        if task == "clustering" and target_column:
            raise ValidationException("聚类任务不应指定 target_column")
        if task != "clustering" and not target_column:
            raise ValidationException("分类 / 回归任务必须指定 target_column")
        if test_size is not None and not (0 < float(test_size) < 1):
            raise ValidationException(
                "test_size 必须在 (0, 1) 区间", details={"test_size": test_size}
            )
        MODEL_REGISTRY.get(model)  # 未注册模型直接报错
        version = self.db.get(DatasetVersion, dataset_version_id)
        if version is None or version.dataset_id != dataset_id:
            raise ValidationException(
                "dataset_version 与 dataset 不匹配",
                details={"dataset_id": dataset_id, "dataset_version_id": dataset_version_id},
            )
        # excluded_columns 存入 preprocessing JSON，避免新增 ORM 字段/迁移
        pp = dict(preprocessing or {})
        if excluded_columns:
            pp["excluded_columns"] = list(excluded_columns)
        # 防御：parameters 中残留的 excluded_columns 必须剥离（兼容旧前端请求）
        params = dict(parameters or {})
        params.pop("excluded_columns", None)
        exp = Experiment(
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            task=task,
            model=model,
            target_column=target_column,
            parameters=params,
            preprocessing=pp,
            seed=seed,
            description=description,
        )
        if test_size is not None:
            # 复用既有 JSON 列承载切分比例，避免为单个数值新增 ORM 字段/迁移
            exp.preprocessing = {**pp, "test_size": float(test_size)}
        self.db.add(exp)
        self.db.commit()
        return exp

    def get(self, experiment_id: int) -> Experiment:
        exp = self.db.get(Experiment, experiment_id)
        if exp is None:
            raise NotFoundException(
                "experiment not found", details={"experiment_id": experiment_id}
            )
        return exp

    def list(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[Experiment], int]:
        total = self.db.scalar(select(sa_func.count()).select_from(Experiment)) or 0
        stmt = (
            select(Experiment)
            .order_by(Experiment.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        return list(self.db.scalars(stmt)), int(total)

    # ------------------------------------------------------------------
    # delete：先清理 runs 产物（model/pipeline），再依赖 ORM 级联删除
    # ------------------------------------------------------------------
    def _delete_run_artifacts(self, experiment: Experiment) -> None:
        if self.dataset_service is None:
            return
        storage = self.dataset_service.storage
        for run in experiment.runs:
            artifacts = run.artifacts or {}
            for key_name in ("model_key", "pipeline_key"):
                key = artifacts.get(key_name)
                if isinstance(key, str) and key:
                    try:
                        storage.delete(key)
                    except Exception:  # noqa: BLE001 - 存储清理失败不阻断删除
                        pass

    def delete(self, experiment_id: int) -> None:
        exp = self.get(experiment_id)
        self._delete_run_artifacts(exp)
        self.db.delete(exp)
        self.db.commit()

    def delete_all(self) -> int:
        experiments = list(self.db.scalars(select(Experiment)))
        for exp in experiments:
            self._delete_run_artifacts(exp)
            self.db.delete(exp)
        self.db.commit()
        return len(experiments)

    # ------------------------------------------------------------------
    # run：加载数据 -> 预处理（防泄漏） -> 训练 -> 评估 -> 记录
    # ------------------------------------------------------------------
    def run(
        self,
        experiment_id: int,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExperimentRun:
        exp = self.get(experiment_id)
        run = ExperimentRun(experiment_id=exp.id, status="running")
        self.db.add(run)
        self.db.commit()
        logger.info(
            "实验运行开始 experiment_id=%s run_id=%s task=%s model=%s target=%s seed=%s",
            exp.id, run.id, exp.task, exp.model, exp.target_column, exp.seed,
        )

        # 训练进度：聚类任务没有「训练/测试划分」，进度条里跳过 split 阶段。
        is_cluster = exp.task == "clustering"
        stage_order = [
            s for s in _TRAIN_STAGE_ORDER if not (is_cluster and s == "split")
        ]
        total_stages = len(stage_order)
        completed: list[str] = []

        def emit(stage_id: str, detail: str = "") -> None:
            if on_progress is None:
                return
            completed.append(stage_id)
            on_progress(
                {
                    "stage": stage_id,
                    "label": _STAGE_LABELS.get(stage_id, stage_id),
                    "done": len(completed),
                    "total": total_stages,
                    "detail": detail,
                }
            )

        started = time.perf_counter()
        try:
            metrics, artifacts, model, pipeline = self._execute(exp, on_progress=emit)
            # 保存模型产物 + 预处理管道（供推理 / 解释复用）
            if self.dataset_service is not None and model is not None:
                base = f"experiments/{exp.id}/runs/{run.id}"
                model_key = f"{base}/model.pkl"
                self.dataset_service.storage.save(model_key, model.to_bytes())
                artifacts["model_key"] = model_key
                if pipeline is not None:
                    pipe_key = f"{base}/pipeline.pkl"
                    self.dataset_service.storage.save(
                        pipe_key, pickle.dumps(pipeline)
                    )
                    artifacts["pipeline_key"] = pipe_key
            run.status = "success"
            run.metrics = metrics
            run.artifacts = artifacts
            emit("persist", "模型与预处理管道已保存，可供推理 / 解释复用")
            logger.info(
                "实验运行成功 run_id=%s 耗时=%.3fs 指标=%s",
                run.id, time.perf_counter() - started, metrics,
            )
        except Exception as exc:  # noqa: BLE001 - 失败也要完整记录
            run.status = "failed"
            run.error = str(exc)
            logger.exception(
                "实验运行失败 experiment_id=%s run_id=%s: %s",
                exp.id, run.id, exc,
            )
        run.runtime = round(time.perf_counter() - started, 6)
        self.db.commit()
        return run

    def _execute(
        self,
        exp: Experiment,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
        emit = on_progress or (lambda stage_id, detail="": None)
        df = self._load_data(exp)
        emit("load", f"加载 {df.height} 行 × {df.width} 列")
        if exp.target_column and exp.target_column not in df.columns:
            raise MLEngineException(
                "目标列不存在于数据版本中",
                details={"target_column": exp.target_column},
            )

        # ---- T0-1: 特征排除（excluded_columns 存于 preprocessing JSON）----
        excluded = list((exp.preprocessing or {}).get("excluded_columns") or [])
        target = exp.target_column
        if target and target in excluded:
            # 目标列只做标签、永远不参与特征（下面 `df.drop([target])` 已把它移除），
            # 所以「把目标列也列进 excluded_columns」是一条幂等的冗余指令，不是错误。
            # 原实现直接抛 MLEngineException，使这条常见且无害的计划在 0.2 秒内必然失败；
            # 真实事故：DepDelay 回归任务连续 3 次 run failed（experiment 26/27/29），
            # 用户看到的只是「训练失败」，模型链路整条没跑起来。
            # 这里改为自动剔除并留痕。
            logger.info(
                "excluded_columns 中的目标列 %s 已自动剔除（目标列不参与特征，指令冗余）",
                target,
            )
            excluded = [c for c in excluded if c != target]
            pp = dict(exp.preprocessing or {})
            pp["excluded_columns"] = excluded
            exp.preprocessing = pp
        missing_cols = [c for c in excluded if c not in df.columns]
        if missing_cols:
            raise MLEngineException(
                "excluded_columns 中存在数据集没有的列",
                details={"missing": missing_cols},
            )
        X = df.drop([target] if target else []) if target else df
        if excluded:
            X = X.drop(excluded)

        y = df[target] if target else None

        # 目标列空值清理：丢弃标签缺失的整行（而不是让 sklearn 抛 NaN 错）
        dropped_rows = 0
        if y is not None and y.null_count() > 0:
            keep = y.is_not_null()
            dropped_rows = int((~keep).sum())
            X = X.filter(keep)
            y = y.filter(keep)
            logger.info(
                "目标列 %s 含空值，训练前剔除 %s 行 experiment_id=%s",
                target, dropped_rows, exp.id,
            )
        emit(
            "feature_select",
            f"特征 {X.width} 列" + (f"，目标 {target}" if target else "（聚类，无目标）")
            + (f"，剔除空标签 {dropped_rows} 行" if dropped_rows else ""),
        )

        # ---- 训练集规模治理：超过 ML_MAX_TRAIN_ROWS 时随机抽样 ----
        # sklearn 的估计器都要求稠密矩阵，1000 万行 × one-hot 展开后的列数
        # 会直接把 numpy 撑爆（实测 57.1 GiB）。抽样是**有损**的，
        # 因此必须 emit 出去并写进 artifacts，绝不静默。
        X, y, sampling = cap_training_rows(X, y, seed=exp.seed)
        # 到这里 X 已经是抽样后的小表（训练全程只用 X / y，本方法不再触碰整表 df）。
        # 显式释放 df：1000 万行 × 10 列常驻约 1.3 GB，与下游预处理矩阵的峰值叠加
        # 会白白吃掉一份内存预算 —— 在 16 GB 机器上直接决定「跑得通 / 跑不通」。
        # 聚类场景此前 X is df，正是它让整表在 fit 期间一直活着。
        del df
        if sampling["sampled"]:
            emit(
                "sample",
                f"数据量 {sampling['original_rows']:,} 行，超过单次训练上限，"
                f"已随机抽样 {sampling['used_rows']:,} 行"
                f"（{sampling['sample_rate']:.1%}）用于训练",
            )

        # ---- T0-3: 训练前 preflight 校验 ----
        self._preflight(exp.task, X, y)

        # ---- T0-2: 默认预处理（用户未提供配置时自动套用）----
        # test_size 共用 preprocessing JSON 承载，需从预处理配置中剥离后再交给管道
        pp_cfg = {
            k: v for k, v in (exp.preprocessing or {}).items()
            if k not in ("excluded_columns", "test_size")
        }
        test_size = float((exp.preprocessing or {}).get("test_size") or 0.2)
        pipeline = build_pipeline(pp_cfg, X)
        model = self._build_model(exp)

        stratified = False
        details: dict[str, Any] = {}
        # 抽样信息透出到 artifacts：训练指标必须能解释（用户要能看出
        # 指标是在全量还是在抽样上算出来的）。
        details["sampling"] = sampling
        if exp.task == "clustering":
            Xp = pipeline.fit_transform(X)
            emit("preprocess", f"{X.width} 列 → {len(pipeline.feature_names_out_)} 列")
            model.fit(Xp)
            emit("train", f"{exp.model} 在 {X.height} 样本上拟合完成")
            metrics = evaluate_clustering(Xp, model.labels_)
            emit("evaluate", _metric_summary("clustering", metrics))
            # 聚类是全量拟合，没有 holdout，也就无所谓 train/test 差距 ⇒ 显式置空
            # （保持 artifacts 键一致，前端据此跳过过拟合诊断）
            details["train_metrics"] = None
            train_rows, test_rows = X.height, 0
            model_features = list(pipeline.feature_names_out_)
        else:
            # ---- T0-4: 分类场景尝试 stratify split ----
            # ★ 分层切分只对**分类**有意义。回归目标即使取值较少（延误分钟被取整成
            # 几十档、「件数/人数」这类离散计数），也绝不能按类别分层：那会要求
            # 「测试集样本数 ≥ 目标唯一值数」，把一次完全正常的回归训练在中小数据上
            # 直接判死；而且报错文案「测试集样本不足以覆盖 N 个类别」会把回归说成分类，
            # 让人朝完全错误的方向排查（真实事故：200 行 / 70 档延误分钟的回归训练
            # 报「不足以覆盖 70 个类别」）。
            stratify = exp.task == "classification" and self._can_stratify(y)
            # 分层切分的硬约束：每类至少要能分到 1 个测试样本，否则 sklearn 抛
            # "The test_size = N should be greater or equal to the number of classes = M"
            # 这类难懂英文错。此处提前转成可操作的中文提示（与 step_runner 同口径）。
            # 注意：sklearn 内部按 ceil(N * test_size) 计算测试集大小，这里必须同样向上取整，
            # 否则 N=8/test_size=0.2（实际测试集 2 个）会被 int() 截断成 1 而误判为不足。
            if stratify and y is not None:
                n_classes = int(y.drop_nulls().n_unique())
                n_test = math.ceil(X.height * test_size)
                if n_test < n_classes:
                    raise MLEngineException(
                        f"测试集样本不足以覆盖 {n_classes} 个类别"
                        f"（当前 {X.height} 行、test_size={test_size}，"
                        f"仅能划分 {n_test} 个测试样本）。"
                        f"请调大 test_size 或补充数据。",
                        details={
                            "n_classes": n_classes,
                            "n_test": n_test,
                            "rows": X.height,
                            "test_size": test_size,
                        },
                    )
            X_train, X_test, y_train, y_test = PreprocessingPipeline.train_test_split(
                X, y, test_size=test_size, seed=exp.seed,
                stratify=y if stratify else None,
            )
            emit(
                "split",
                f"训练 {X_train.height} / 测试 {X_test.height} 行"
                f"（test_size={test_size}{'，分层' if stratify else ''}）",
            )
            Xtr = pipeline.fit_transform(X_train)
            Xte = pipeline.transform(X_test)
            emit("preprocess", f"{X_train.width} 列 → {len(pipeline.feature_names_out_)} 列")
            model.fit(Xtr, y_train)
            emit("train", f"{exp.model} 在 {X_train.height} 样本上拟合完成")
            y_pred = model.predict(Xte)
            y_proba: pl.DataFrame | None = None
            try:
                y_proba = model.predict_proba(Xte)
            except MLEngineException:
                y_proba = None
            if exp.task == "classification":
                metrics = eval_classification(y_test, y_pred, y_proba)
                details["confusion_matrix"] = confusion_matrix(y_test, y_pred)
                details["per_class"] = classification_report(y_test, y_pred)["per_class"]
                # 训练集类别分布：既是「模型偏向多数类」的判定依据，
                # 也是界面上解释「为什么 accuracy 高但 f1 低」的直接证据。
                details["class_distribution"] = _class_distribution(y_train)
                # 决策阈值的依据：y_proba 上面已经算好了，扫一遍阈值几乎不增加成本，
                # 却能把「阈值该定多少」从拍脑袋变成看曲线选。
                details["threshold_curve"] = self._threshold_basis(y_test, y_proba, model)
            else:
                metrics = eval_regression(y_test, y_pred)
                details["residual_stats"] = regression_residuals(y_test, y_pred)
            # 训练集指标：只在 artifacts 里落一份，供前端做「过拟合诊断」用
            # （判断依据是 train/test 指标差距，而不是单看测试集绝对值）。
            # 它不参与 run.metrics，因此不改变任何既有指标口径。
            details["train_metrics"] = self._train_metrics(model, Xtr, y_train, exp.task)
            emit("evaluate", _metric_summary(exp.task, metrics))
            train_rows, test_rows = X_train.height, X_test.height
            stratified = stratify
            model_features = list(pipeline.feature_names_out_)

        artifacts = {
            # ---- 可复现性快照（§可复现性）：数据 / 参数 / 切分 / 版本 ----
            "dataset_snapshot": dataset_snapshot(
                self.db.get(DatasetVersion, exp.dataset_version_id)
            ),
            "seed": exp.seed,
            "library_versions": library_versions(),
            "model": exp.model,
            "task": exp.task,
            "features": list(X.columns),  # T0-1: 排除后实际参与训练的列
            "model_features": model_features,  # 预处理后真正喂给模型的列
            "excluded_columns": excluded,
            "target_column": exp.target_column,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "test_size": test_size,
            "dropped_rows": dropped_rows,
            "stratified": stratified,
            "preprocessing_report": pipeline.report,
            "model_summary": model.summary(),
            "feature_importance": self._safe_feature_importance(model),
            **details,
        }
        return metrics, artifacts, model, pipeline

    @staticmethod
    def _safe_feature_importance(model: Any) -> dict[str, Any] | None:
        """尽力产出特征重要性；模型不支持时返回 None（不阻断训练）。"""
        try:
            return model.feature_importance()
        except Exception as exc:  # noqa: BLE001 - 解释失败不影响训练结果
            logger.info("特征重要性不可用 model=%s: %s", getattr(model, "name", "?"), exc)
            return None

    @staticmethod
    def _threshold_basis(
        y_test: pl.Series, y_proba: pl.DataFrame | None, model: Any
    ) -> dict[str, Any] | None:
        """留出测试集上的阈值取舍曲线，作为「阈值该定多少」的依据。

        复用上面已经算好的 y_proba，不额外跑一次预测，所以不给训练增加可感知的耗时。
        曲线口径为正类，与 run.metrics 的 macro 口径不同，因此曲线自带
        metric_average 标记，避免两处数字被当成同一口径直接比较。

        仅在二分类落这份曲线：多分类下「一个阈值」没有明确语义（见 threshold 模块）。
        任何失败都返回 None —— 它只是调参依据，不该让训练本身失败。
        """
        if y_proba is None:
            return None
        classes = model_classes(model)
        if not is_binary(classes):
            return None
        try:
            curve = threshold_curve(y_test, y_proba, classes)
        except Exception as exc:  # noqa: BLE001 - 诊断失败不影响训练结果
            logger.info("阈值曲线不可用 model=%s: %s", getattr(model, "name", "?"), exc)
            return None
        curve["basis"] = "holdout_test"
        return curve

    @staticmethod
    def _inference_threshold_curve(
        df: pl.DataFrame,
        target_column: str | None,
        y_proba: pl.DataFrame,
        classes: list[Any],
    ) -> dict[str, Any] | None:
        """推理数据自带目标列时，就地扫一条阈值曲线。

        与训练时的曲线（依据=留出测试集）刻意分开：这里的依据是**推理数据本身**，
        若推理数据恰好就是训练数据，指标会偏乐观。因此返回结果带
        basis="inference_data"，界面据此提示「仅供参考，不代表泛化性能」。

        目标列缺失时返回 None —— 无标签数据本就无法评估阈值，这是正常情况而非错误。
        """
        if not target_column or target_column not in df.columns:
            return None
        try:
            curve = threshold_curve(df[target_column], y_proba, classes)
        except Exception as exc:  # noqa: BLE001 - 依据缺失不该阻断推理
            logger.info("推理数据阈值曲线不可用: %s", exc)
            return None
        if not curve.get("points"):
            return None
        curve["basis"] = "inference_data"
        return curve

    @staticmethod
    def _train_metrics(
        model: Any,
        Xtr: pl.DataFrame,
        y_train: pl.Series | None,
        task: str,
    ) -> dict[str, Any] | None:
        """在训练集上评估一次，供前端诊断「过拟合 / 欠拟合」。

        与 `run.metrics`（测试集口径）严格分开：这里只回答「模型在它见过的数据上
        表现如何」，两者相减才有诊断意义 —— 单看测试集绝对值无法判断是模型太弱
        还是已经学过头。结果落在 artifacts.train_metrics，不进入 run.metrics。

        刻意不算概率类指标（roc_auc）：那需要再跑一次 predict_proba，
        为一个辅助信号把训练耗时翻倍不划算，accuracy / f1 的差距已足够判断。
        评估失败返回 None —— 它只是调参依据，不该让训练本身失败。
        """
        if y_train is None:
            return None
        try:
            y_pred = model.predict(Xtr)
            if task == "classification":
                return eval_classification(y_train, y_pred, None)
            if task == "regression":
                return eval_regression(y_train, y_pred)
        except Exception as exc:  # noqa: BLE001 - 诊断失败不影响训练结果
            logger.info("训练集指标不可用 model=%s: %s", getattr(model, "name", "?"), exc)
        return None

    @staticmethod
    def _default_preprocessing(X: pl.DataFrame) -> dict[str, Any]:
        """T0-2: 用户未配置时的默认预处理（与 Workflow 共用同一份实现）。"""
        return default_preprocessing_config(X)

    @staticmethod
    def _preflight(task: str, X: pl.DataFrame, y: pl.Series | None) -> None:
        """T0-3: 训练前校验，把 sklearn 难懂错误转成 MLEngineException。"""
        if X.width == 0:
            raise MLEngineException("训练特征列为空（排除后无可用列）")
        if X.height < 2:
            raise MLEngineException("训练样本数 < 2，无法训练")
        if y is not None:
            if y.len() != X.height:
                raise MLEngineException("X 与 y 行数不一致")
            if y.null_count() == y.len():
                raise MLEngineException("目标列全为空")
            if task == "classification":
                n_classes = int(y.drop_nulls().n_unique())
                if n_classes < 2:
                    raise MLEngineException(
                        "分类任务至少需要 2 个类别",
                        details={"n_classes": n_classes},
                    )
            if task == "regression":
                n_unique = int(y.drop_nulls().n_unique())
                if n_unique < 2:
                    raise MLEngineException(
                        "目标列无变化（所有取值相同），回归模型无法学习",
                        details={"n_unique": n_unique},
                    )

    @staticmethod
    def _can_stratify(y: pl.Series | None) -> bool:
        """T0-4: 判断分类任务是否可分层切分（每类≥2 样本）。"""
        if y is None:
            return False
        non_null = y.drop_nulls()
        if non_null.len() < 4:
            return False
        counts = non_null.value_counts()
        return int(counts["count"].min()) >= 2 if counts.height else False

    def _load_data(self, exp: Experiment) -> pl.DataFrame:
        if self.dataset_service is None:
            raise MLEngineException(
                "ExperimentService 缺少 dataset_service，无法读取数据版本"
            )
        version = self.db.get(DatasetVersion, exp.dataset_version_id)
        if version is None:
            raise NotFoundException(
                "dataset version not found",
                details={"dataset_version_id": exp.dataset_version_id},
            )
        return self.dataset_service.load_version(exp.dataset_id, version.version)

    def _build_model(self, exp: Experiment):
        """创建模型实例；seed 注入 random_state（不支持的模型自动回退）。"""
        params = dict(exp.parameters or {})
        params.pop("excluded_columns", None)  # 防御：绝不让 excluded_columns 进入 sklearn
        if exp.seed is not None and "random_state" not in params:
            # 公开探测参数支持（签名级，不构造实例、不走私有方法）
            if MODEL_REGISTRY.supports_param(exp.model, "random_state"):
                params["random_state"] = exp.seed
            else:
                logger.debug(
                    "模型 %s 不支持 random_state，seed=%s 仅用于数据切分",
                    exp.model, exp.seed,
                )
        return MODEL_REGISTRY.create(exp.model, params)

    # ------------------------------------------------------------------
    # 推理：加载训练产物（model.pkl + pipeline.pkl）对新数据做预测
    # ------------------------------------------------------------------
    def predict(
        self,
        run_id: int,
        df: pl.DataFrame | None = None,
        *,
        dataset_id: int | None = None,
        version: int | None = None,
        limit: int | None = 200,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """用一次成功运行的模型做批量推理。

        threshold 是**训练后**参数：作用在模型输出的类别概率上，不动模型本身，
        因此改它不需要重新训练。只对二分类有意义（多分类单个阈值语义不明），
        传了却在不适用的场景下会显式报错，而不是被静默忽略。

        必须同时加载 pipeline：训练特征是 one-hot/标准化后的列，
        直接把原始列喂给模型会因列不匹配而失败（已实证）。
        """
        run = self.get_run(run_id)
        if run.status != "success":
            raise ValidationException(
                "只能对状态为 success 的运行做推理",
                details={"run_id": run_id, "status": run.status},
            )
        if df is None:
            if dataset_id is None or self.dataset_service is None:
                raise ValidationException("推理需要提供 dataset_id 或内联数据")
            df = self.dataset_service.load_version(int(dataset_id), version)

        exp = self.get(run.experiment_id)
        model = self._load_model(run, exp)
        pipeline = self._load_pipeline(run)

        feature_columns = list((run.artifacts or {}).get("features") or [])
        if not feature_columns:
            feature_columns = list(model.feature_names_)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise MLEngineException(
                "推理数据缺少训练时的特征列",
                details={"missing_columns": missing},
            )
        X_raw = df.select(feature_columns)

        started = time.perf_counter()
        # 推理**不能抽样**（少预测一行就是少一行结果），只能分块：每块独立
        # transform + predict，结果按行拼接，语义与一次性推理逐样本一致。
        # 这样 1000 万行 × 767 列也不会一次性稠密化（那是 57.1 GiB）。
        prediction = batch_predict(model, X_raw, pipeline=pipeline)
        elapsed = round(time.perf_counter() - started, 6)

        probabilities: list[dict[str, Any]] | None = None
        probability_columns: list[str] = []
        proba_frame: pl.DataFrame | None = None
        rows = df.height
        preview = int(limit) if limit and limit > 0 else rows
        if exp.task == "classification":
            try:
                proba_frame = batch_predict_proba(model, X_raw, pipeline=pipeline)
                probability_columns = list(proba_frame.columns)
                # 只物化要返回的前 preview 行：整表 to_dicts() 会为百万行数据集构造等量 Python dict，
                # 而下面只有前 preview 行被写进 payload。
                probabilities = proba_frame.head(preview).to_dicts()
            except MLEngineException:
                proba_frame = None
                probabilities = None

        # ---- 训练后参数：决策阈值 ----
        # 模型负责输出概率，阈值负责决定「概率多少算正类」。两者分开的好处是
        # 换阈值不动模型，改一次立刻见效。不适用时显式报错而不是静默忽略 ——
        # 静默忽略会让用户以为阈值生效了、结果却没变，是最难排查的一类问题。
        classes = model_classes(model)
        binary = is_binary(classes)
        positive_class: Any = classes[-1] if binary and classes else None
        applied_threshold: float | None = None
        shift_ratio: float | None = None
        inference_curve: dict[str, Any] | None = None
        if threshold is not None:
            if exp.task != "classification":
                raise ValidationException(
                    f"决策阈值只适用于分类任务，当前任务是 {exp.task}："
                    "回归输出的是数值、聚类输出的是簇编号，都不是「正类概率」，没有阈值可调",
                    details={"task": exp.task, "model": exp.model},
                )
            if not binary:
                raise ValidationException(
                    "决策阈值只适用于二分类，当前模型有 "
                    f"{len(classes) if classes else 0} 个类别："
                    "单个阈值无法表达多类之间的取舍",
                    details={"n_classes": len(classes) if classes else 0, "model": exp.model},
                )
            if proba_frame is None:
                raise ValidationException(
                    "该模型没有输出类别概率，无法按阈值判定标签",
                    details={"model": exp.model},
                )
            applied_threshold = float(threshold)
            prediction = apply_threshold(proba_frame, classes, applied_threshold)
            shift_ratio = label_shift_ratio(proba_frame, classes, applied_threshold)
            # 推理数据自带目标列时，就地再扫一条曲线：这样训练时还没落曲线的旧运行
            # 也能看到依据，不必为了试一个阈值重训一次。
            inference_curve = self._inference_threshold_curve(
                df, exp.target_column, proba_frame, classes
            )

        payload = df.head(preview).to_dicts()
        for i, value in enumerate(prediction.head(preview).to_list()):
            payload[i]["prediction"] = _jsonable(value)
            if probabilities is not None and i < preview:
                for col, v in probabilities[i].items():
                    payload[i][col] = _jsonable(v)

        logger.info(
            "推理完成 run_id=%s 模型=%s 行数=%s 耗时=%.3fs",
            run_id, exp.model, rows, elapsed,
        )
        return {
            "run_id": run.id,
            "experiment_id": exp.id,
            "model": exp.model,
            "task": exp.task,
            "row_count": rows,
            "prediction_column": "prediction",
            "probability_columns": probability_columns,
            "feature_columns": feature_columns,
            "model_features": list(model.feature_names_),
            "pipeline_applied": pipeline is not None,
            "runtime": elapsed,
            "preview": payload,
            # ---- 推理阶段参数的回执 ----
            # 回执而非「悄悄生效」：界面上必须能看出这次推理用的是哪个阈值、
            # 正类是哪个类别、以及换阈值到底改动了多少样本的标签。
            "threshold_supported": binary and proba_frame is not None,
            "threshold": applied_threshold,
            "threshold_default": DEFAULT_THRESHOLD,
            "positive_class": _jsonable(positive_class),
            "label_changed_ratio": shift_ratio,
            "threshold_curve": inference_curve,
        }

    def explain(self, run_id: int) -> dict[str, Any]:
        """解释一次成功运行的模型：优先复用训练时落库的特征重要性。"""
        run = self.get_run(run_id)
        if run.status != "success":
            raise ValidationException(
                "只能解释状态为 success 的运行",
                details={"run_id": run_id, "status": run.status},
            )
        cached = (run.artifacts or {}).get("feature_importance")
        if cached:
            return {"source": "run_artifacts", "run_id": run.id, **cached}
        exp = self.get(run.experiment_id)
        model = self._load_model(run, exp)
        return {"source": "model", "run_id": run.id, **model.feature_importance()}

    def _load_model(self, run: ExperimentRun, exp: Experiment):
        """从存储加载模型产物；缺失时给出可操作的错误说明。"""
        storage = getattr(self.dataset_service, "storage", None)
        model_key = (run.artifacts or {}).get("model_key")
        if not model_key or storage is None:
            raise MLEngineException(
                "该运行没有保存模型产物，无法推理（请重新训练一次）",
                details={"run_id": run.id},
            )
        try:
            return MODEL_REGISTRY.get(exp.model).from_bytes(storage.read(model_key))
        except FileNotFoundError as exc:
            raise MLEngineException(
                "模型产物文件已丢失，请重新训练",
                details={"run_id": run.id, "model_key": model_key},
            ) from exc

    def _load_pipeline(self, run: ExperimentRun) -> PreprocessingPipeline | None:
        """加载预处理管道；产物缺失或反序列化失败时降级为「不预处理」。

        训练若未产生 pipeline（例如旧版本运行或无需预处理），
        直接用原始列推理；列不匹配会由模型层给出明确报错。
        """
        storage = getattr(self.dataset_service, "storage", None)
        pipeline_key = (run.artifacts or {}).get("pipeline_key")
        if not pipeline_key or storage is None:
            logger.warning("运行 %s 无 pipeline 产物，推理将跳过预处理", run.id)
            return None
        try:
            obj = pickle.loads(storage.read(pipeline_key))  # noqa: S301 - 内部可信数据
        except Exception as exc:  # noqa: BLE001
            logger.warning("pipeline 加载失败 run_id=%s: %s", run.id, exc)
            return None
        return obj if isinstance(obj, PreprocessingPipeline) else None

    # ------------------------------------------------------------------
    # runs 与 compare
    # ------------------------------------------------------------------
    def list_runs(self, experiment_id: int) -> list[ExperimentRun]:
        self.get(experiment_id)
        stmt = (
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id == experiment_id)
            .order_by(ExperimentRun.id.desc())
        )
        return list(self.db.scalars(stmt))

    def get_run(self, run_id: int) -> ExperimentRun:
        run = self.db.get(ExperimentRun, run_id)
        if run is None:
            raise NotFoundException("run not found", details={"run_id": run_id})
        return run

    def compare_runs(self, run_ids: list[int]) -> Any:
        from app.experiments.comparator import ExperimentComparator

        runs = [self.get_run(rid) for rid in run_ids]
        self._load_experiment_params(runs)
        return ExperimentComparator().compare(runs)

    def compare_experiments(self, experiment_ids: list[int]) -> Any:
        """比较多个实验：各取最近一次成功运行（无成功则取最近一次运行）。"""
        from app.experiments.comparator import ExperimentComparator

        runs: list[ExperimentRun] = []
        for exp_id in experiment_ids:
            self.get(exp_id)
            stmt = (
                select(ExperimentRun)
                .where(ExperimentRun.experiment_id == exp_id)
                .order_by(ExperimentRun.id.desc())
            )
            all_runs = list(self.db.scalars(stmt))
            chosen = next((r for r in all_runs if r.status == "success"), None)
            if chosen is None and all_runs:
                chosen = all_runs[0]
            if chosen is not None:
                runs.append(chosen)
        if not runs:
            raise NotFoundException("no runs to compare")
        self._load_experiment_params(runs)
        return ExperimentComparator().compare(runs)

    def _load_experiment_params(self, runs: list[ExperimentRun]) -> None:
        """预加载实验参数，供比较器读取。"""
        exp_ids = {r.experiment_id for r in runs}
        stmt = (
            select(Experiment)
            .where(Experiment.id.in_(exp_ids))
            .options(selectinload(Experiment.runs))
        )
        self.db.scalars(stmt).all()
