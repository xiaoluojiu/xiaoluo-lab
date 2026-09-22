"""Prompt 108-113：ML 工具。

ml.detect_task / ml.prepare / ml.train / ml.predict / ml.evaluate / ml.compare / ml.explain。
ml.train 走 ExperimentService（复现实验）；ml.predict / ml.explain 读取训练产物。
"""

from __future__ import annotations

from typing import Any

import polars as pl
from app.ml_engine.metadata import (
    INFERENCE_PARAMS,
    METRIC_GUIDE,
    MODEL_PARAMS,
    PARAM_COMBOS,
    PIPELINE_STEPS,
    PREPROCESSING_PARAMS,
    TRAINING_PARAMS,
    TUNING_PLAYBOOK,
)
from app.ml_engine.registry import MODEL_REGISTRY
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

# 推理预览行数上限：与 API 层（experiments.py PredictRequest.limit）保持一致，
# 避免 Agent 路径用超大 limit 让 to_dicts() 全量物化而放大内存。
PREDICT_PREVIEW_LIMIT_MAX = 5000
PREDICT_PREVIEW_LIMIT_DEFAULT = 20


def _load_df(services: ToolServices, dataset_id: int, version: int | None) -> pl.DataFrame:
    ds = services.require("dataset_service")
    return ds.load_version(dataset_id, int(version) if version else None)


# 目标列命名约定：ml.detect_task 未显式指定 target 时的唯一候选判定
_TARGET_COLUMN_NAMES = ("target", "label", "y", "class")


class MlDetectTaskTool(Tool):
    """根据 Schema / Profile / Target 判断任务类型，结果必须包含理由。"""

    name = "ml.detect_task"
    description = "根据目标列类型与数据分布判断 ML 任务类型，并给出理由。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "target": {"type": "string"},
            "infer_target": {"type": "boolean"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {
        "type": "object",
        "properties": {"task": {"type": "string"}, "reasons": {"type": "array"}},
    }
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        target = params.get("target")
        df = _load_df(services, dataset_id, params.get("version"))
        reasons: list[str] = []
        if target is None:
            if params.get("infer_target"):
                # 显式开启目标列推断：按常见命名约定识别（target/label/y/class）。
                # Agent 的"训练模型"链路会传 infer_target=True，让未指明目标的请求
                # 也能走通监督学习；直接调用（默认关闭）保持"未指定即聚类"语义。
                candidates = [c for c in df.columns if str(c).lower() in _TARGET_COLUMN_NAMES]
                if len(candidates) == 1:
                    target = candidates[0]
                    reasons.append(f"未指定目标列，但识别到唯一候选目标列 {target!r}（命名约定）")
                elif len(candidates) > 1:
                    reasons.append(f"发现多个候选目标列 {candidates}，无法自动选择")
                else:
                    reasons.append("未指定目标列且无命名约定候选：无监督任务")
            else:
                reasons.append("未指定目标列：无监督任务")
            if target is None:
                reasons.append(f"数据规模 {df.height} 行 x {df.width} 列，适合聚类探索")
                return ToolResult.ok(
                    {"task": "clustering", "target": None, "reasons": reasons},
                    summary="任务类型：clustering",
                )
        if target not in df.columns:
            return ToolResult.fail(
                f"目标列 {target!r} 不存在", metadata={"columns": list(df.columns)}
            )
        dtype = df.schema[target]
        n_unique = df[target].n_unique()
        if dtype == pl.String or dtype == pl.Boolean or dtype == pl.Categorical:
            task = "classification"
            reasons.append(f"目标列 {target} 为 {dtype}（类别型）")
        elif dtype.is_integer() and n_unique <= 20:
            task = "classification"
            reasons.append(
                f"目标列 {target} 为整数且仅 {n_unique} 个取值（低基数，视为类别）"
            )
        elif dtype.is_numeric():
            task = "regression"
            reasons.append(
                f"目标列 {target} 为数值型且有 {n_unique} 个不同取值（连续值）"
            )
        else:
            return ToolResult.fail(
                f"目标列 {target} 类型 {dtype} 无法判断任务类型",
                metadata={"supported": ["数值", "字符串", "布尔"]},
            )
        reasons.append(f"目标列非空值比例 {(1 - df[target].null_count() / df.height):.1%}")
        return ToolResult.ok(
            {"task": task, "target": target, "reasons": reasons},
            summary=f"任务类型：{task}",
        )


class MlPrepareTool(Tool):
    """根据数据画像建议预处理配置。"""

    name = "ml.prepare"
    description = "分析数据缺失/类别/数值列，生成防泄漏的预处理配置（missing/encoding/scaling）。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "target": {"type": "string"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        df = _load_df(services, dataset_id, params.get("version"))
        target = params.get("target")
        feature_df = df.drop(target) if target else df

        missing_cols = [c for c in feature_df.columns if feature_df[c].null_count() > 0]
        missing_config = None
        if missing_cols:
            numeric_missing = [
                c for c in missing_cols if feature_df.schema[c].is_numeric()
            ]
            strategy = "median" if numeric_missing else "mode"
            missing_config = {
                "strategy": strategy,
                "columns": missing_cols,
            }
        categorical = [
            c
            for c, d in feature_df.schema.items()
            if d == pl.String or d == pl.Categorical
        ]
        encoding_config = (
            {"method": "one_hot", "columns": categorical} if categorical else None
        )
        # 时间列与布尔列同样按数值列处理，避免带着 datetime64 直接进模型导致训练失败
        numeric = [
            c
            for c, d in feature_df.schema.items()
            if d.is_numeric() or isinstance(d, (pl.Datetime, pl.Date, pl.Time)) or d == pl.Boolean
        ]
        scaling_config = {"method": "standard", "columns": numeric} if numeric else None

        config = {
            "missing": missing_config,
            "encoding": encoding_config,
            "scaling": scaling_config,
        }
        notes = []
        if categorical and numeric:
            notes.append("类别列将 one-hot 编码，数值列将标准化；统计量仅在训练集拟合")
        elif categorical:
            notes.append("类别列将 one-hot 编码")
        elif numeric:
            notes.append("数值列将标准化")
        # preprocessing 是 config 的同义别名：下游（尤其是 LLM 规划器）写
        # {{stepN.preprocessing}} 比 {{stepN.config}} 自然得多，缺这个别名会让
        # 「prepare -> train」这条链在引用解析阶段直接失败（字段不存在）。
        return ToolResult.ok(
            {
                "config": config,
                "preprocessing": config,
                "feature_columns": list(feature_df.columns),
                "notes": notes,
            },
            summary=f"预处理建议：{', '.join(notes) or '数据无需预处理'}",
        )


# 「选用合适的模型」这类请求落到计划里必然是 model="auto"（或干脆省略）。
# 不接受它 ⇒ 规划出来的第一步就必然失败（"模型 'auto' 未注册"），
# 而失败会触发重规划 —— 2026-09-22 r-22 的死循环就是这么起头的。
_AUTO_MODEL_ALIASES = {"", "auto", "自动", "default"}


class MlTrainTool(Tool):
    """创建实验并立即训练（复用 ExperimentService，保证可复现）。"""

    name = "ml.train"
    description = "在数据集版本上训练模型：创建 Experiment 并运行，返回评估指标。model 填 auto 表示按任务类型自动选择。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "target": {"type": "string"},
            "model": {"type": "string", "description": "模型名；填 auto 表示按任务类型自动选择（分类→logistic_regression，回归→linear_regression，聚类→kmeans）"},
            "params": {"type": "object"},
            "preprocessing": {"type": "object"},
            "test_size": {"type": "number", "description": "测试集比例，缺省 0.2"},
            "seed": {"type": "integer"},
            "description": {"type": "string"},
        },
        "required": ["dataset_id", "model"],
    }
    output_schema = {"type": "object"}
    permission = "train_model"
    risk_level = "high"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        exp_service = services.require("experiment_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        ds = services.require("dataset_service")
        version_row = ds.get_version_row(dataset_id, params.get("version"))

        # 目标列缺省时自动检测任务类型
        target = params.get("target")
        task = params.get("task")
        model_adjusted: dict[str, str] | None = None
        task_defaults = {
            "clustering": "kmeans",
            "classification": "logistic_regression",
            "regression": "linear_regression",
        }
        task_inferred = False
        if task is None:
            detect = MlDetectTaskTool().execute(
                {"dataset_id": dataset_id, "version": version_row.version, "target": target},
                context,
                services,
            )
            if not detect.success:
                return detect
            task = detect.data["task"]
            task_inferred = True

        requested_model = str(params.get("model") or "").strip().lower()
        if requested_model in _AUTO_MODEL_ALIASES:
            # ★「选用合适的模型」：按任务类型挑默认模型，并在结果里记一笔，便于追溯。
            fallback = task_defaults.get(task)
            if not fallback:
                return ToolResult.fail(f"无法确定任务类型 {task} 对应的默认模型，请显式指定 model。")
            model_adjusted = {"from": requested_model or "auto", "to": fallback, "task": task}
            params["model"] = fallback
        elif task_inferred:
            # 任务类型是推断出来的，就可能与模型不匹配：
            # 例「训练一个模型」未指明目标列 -> 判为 clustering，但模型默认给了
            # logistic_regression，直接训练会抛 sklearn 底层错误
            # （LogisticRegression.fit() missing 1 required positional argument: 'y'）。
            # 推断场景下自动换成该任务的默认模型，并在结果里记录这次替换，便于追溯。
            try:
                model_task = MODEL_REGISTRY.metadata(params["model"]).get("task")
            except Exception:  # noqa: BLE001 - 未知模型由后续 MODEL_REGISTRY.get 统一报错
                model_task = None
            fallback = task_defaults.get(task)
            if model_task and fallback and model_task != task:
                model_adjusted = {"from": params["model"], "to": fallback, "task": task}
                params["model"] = fallback

        # T0-1: excluded_columns 从 params 剥离，作为独立特征选择配置
        raw_params = dict(params.get("params") or {})
        excluded = raw_params.pop("excluded_columns", None) or params.get("excluded_columns")
        exp = exp_service.create(
            dataset_id=dataset_id,
            dataset_version_id=version_row.id,
            task=task,
            model=params["model"],
            parameters=raw_params or None,
            preprocessing=params.get("preprocessing"),
            excluded_columns=excluded,
            seed=params.get("seed"),
            description=params.get("description", ""),
            target_column=target if task != "clustering" else None,
            test_size=params.get("test_size"),
        )
        run = exp_service.run(exp.id)
        # T0-8: 训练失败必须返回 ToolResult.fail，不能让 Agent 误判为成功
        if run.status != "success":
            return ToolResult.fail(
                run.error or "训练失败",
                data={
                    "experiment_id": exp.id,
                    "run_id": run.id,
                    "status": run.status,
                    "error": run.error,
                },
                metadata={"task": task, "model": params["model"]},
            )
        return ToolResult.ok(
            {
                "experiment_id": exp.id,
                "run_id": run.id,
                "status": run.status,
                "metrics": run.metrics,
                "error": run.error,
                **({"model_adjusted": model_adjusted} if model_adjusted else {}),
            },
            summary=f"训练成功：{exp.model}",
            metadata={"task": task, "model": params["model"], "model_adjusted": model_adjusted},
        )


class MlPredictTool(Tool):
    """用已训练模型推理：加载 model.pkl + pipeline.pkl 对新数据预测。"""

    name = "ml.predict"
    description = (
        "用一次成功训练的模型对新数据做推理，返回预测结果与概率预览。"
        "二分类可通过 threshold 调整决策阈值（训练后参数，改它不需要重新训练）。"
    )
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "run_id": {"type": "integer", "description": "成功训练的运行 ID"},
            "dataset_id": {"type": "integer", "description": "待推理的数据集（缺省用训练数据集）"},
            "version": {"type": "integer"},
            "limit": {"type": "integer", "description": "预览行数，默认 20"},
            "threshold": {
                "type": "number",
                "description": (
                    "决策阈值（仅二分类，0~1 开区间）：正类概率大于阈值才判为正类。"
                    "调高＝精确率升、召回率降。缺省沿用 sklearn 默认的 0.5。"
                ),
            },
        },
        "required": ["run_id"],
    }
    output_schema = {"type": "object"}
    permission = "train_model"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        exp_service = services.require("experiment_service")
        run_id = int(params["run_id"])
        run = exp_service.get_run(run_id)
        exp = exp_service.get(run.experiment_id)
        dataset_id = params.get("dataset_id")
        if dataset_id is None:
            dataset_id = exp.dataset_id
        self.assert_dataset_access(context, int(dataset_id))
        # 钳制预览行数上限，避免超大 limit 让整表 to_dicts() 放大内存
        limit = int(params.get("limit") or PREDICT_PREVIEW_LIMIT_DEFAULT)
        limit = max(1, min(limit, PREDICT_PREVIEW_LIMIT_MAX))
        try:
            result = exp_service.predict(
                run_id,
                dataset_id=int(dataset_id),
                version=params.get("version"),
                limit=limit,
                threshold=params.get("threshold"),
            )
        except Exception as exc:  # noqa: BLE001 - 转成工具可读失败
            return ToolResult.fail(
                f"推理失败：{exc}",
                data={"run_id": run_id, "dataset_id": dataset_id},
            )
        # 阈值回执：Agent 必须能看出这次推理到底用了哪个阈值、改动了多少样本，
        # 否则它会把「阈值没生效」当成「模型结果就是这样」。
        curve = result.get("threshold_curve") or {}
        return ToolResult.ok(
            {
                "run_id": run_id,
                "model": result["model"],
                "task": result["task"],
                "row_count": result["row_count"],
                "prediction_column": result["prediction_column"],
                "probability_columns": result["probability_columns"],
                "pipeline_applied": result["pipeline_applied"],
                "runtime": result["runtime"],
                "threshold_supported": result.get("threshold_supported"),
                "threshold": result.get("threshold"),
                "positive_class": result.get("positive_class"),
                "label_changed_ratio": result.get("label_changed_ratio"),
                "threshold_curve": {
                    "basis": curve.get("basis"),
                    "basis_rows": curve.get("basis_rows"),
                    "suggested_threshold": curve.get("suggested_threshold"),
                    "suggested_f1": curve.get("suggested_f1"),
                    "metric_average": curve.get("metric_average"),
                    "points": curve.get("points"),
                }
                if curve
                else None,
                "preview": result["preview"][:limit],
            },
            summary=self._summarize(result, limit),
            metadata={"task": result["task"], "model": result["model"]},
        )

    @staticmethod
    def _summarize(result: dict[str, Any], limit: int) -> str:
        """把阈值回执压成一句人话，Agent 转述时不必自己拼。"""
        base = f"推理完成：{result['row_count']} 行 · {result['model']}"
        threshold = result.get("threshold")
        if threshold is None:
            return base
        shifted = result.get("label_changed_ratio")
        changed = f"，{shifted:.1%} 的样本标签随之改变" if isinstance(shifted, float) else ""
        return f"{base} · 阈值 {threshold}{changed}"


class MlEvaluateTool(Tool):
    name = "ml.evaluate"
    description = "查看一次训练运行的评估指标与解读。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {"run_id": {"type": "integer"}},
        "required": ["run_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        exp_service = services.require("experiment_service")
        run = exp_service.get_run(int(params["run_id"]))
        exp = exp_service.get(run.experiment_id)
        return ToolResult.ok(
            {
                "run_id": run.id,
                "experiment_id": run.experiment_id,
                "task": exp.task,
                "model": exp.model,
                "status": run.status,
                "metrics": run.metrics,
                "runtime": run.runtime,
                "error": run.error,
            },
            summary=(
                f"运行 {run.id}（{exp.task}/{exp.model}）状态 {run.status}"
            ),
        )


class MlCompareTool(Tool):
    name = "ml.compare"
    description = "比较多次训练运行：指标最优、参数差异、耗时排名。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "run_ids": {"type": "array", "items": {"type": "integer"}},
            "experiment_ids": {"type": "array", "items": {"type": "integer"}},
        },
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        exp_service = services.require("experiment_service")
        if params.get("run_ids"):
            result = exp_service.compare_runs([int(r) for r in params["run_ids"]])
        elif params.get("experiment_ids"):
            result = exp_service.compare_experiments(
                [int(e) for e in params["experiment_ids"]]
            )
        else:
            return ToolResult.fail("需要提供 run_ids 或 experiment_ids")
        return ToolResult.ok(result.to_dict(), summary=f"比较了 {len(result.entries)} 次运行")


class MlExplainTool(Tool):
    """模型解释：读取训练产物（模型字节）+ 训练特征。"""

    name = "ml.explain"
    description = "解释已训练模型：特征重要性（可扩展 SHAP）。"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {"run_id": {"type": "integer"}},
        "required": ["run_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        exp_service = services.require("experiment_service")
        run = exp_service.get_run(int(params["run_id"]))
        if run.status != "success":
            return ToolResult.fail("只能解释成功完成的训练运行")
        try:
            # 优先复用训练时落库的特征重要性，避免每次都反序列化整个模型
            explanation = exp_service.explain(int(params["run_id"]))
        except Exception as exc:  # noqa: BLE001 - 转成工具可读失败
            return ToolResult.fail(f"解释失败：{exc}", data={"run_id": run.id})
        top = explanation["importances"][:5]
        summary = "、".join(
            f"{item['feature']}({item['importance']:.3f})" for item in top
        )
        return ToolResult.ok(
            {"method": explanation["method"], "importances": explanation["importances"]},
            summary=f"特征重要性 Top5：{summary}",
        )


class MlExplainConfigTool(Tool):
    """教学工具：解释 ML 流程步骤、参数含义与指标口径。

    数据源与 `GET /ml/catalog` 完全一致（`app/ml_engine/metadata.py`），
    因此 Agent 回答"这个参数是做什么的"时不会与前端文档各说一套。
    """

    name = "ml.explain_config"
    description = (
        "解释机器学习流程与参数：可查询流程步骤、预处理参数、训练参数、"
        "某模型的可调参数、评估指标的含义与好坏判断、"
        "指标不理想时该调整哪些参数（调参手册），"
        "以及训练完成后不需重训即可生效的推理参数（如决策阈值）。"
    )
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": [
                    "pipeline",
                    "preprocessing",
                    "training",
                    "model",
                    "metrics",
                    "tuning",
                    "inference",
                ],
                "description": (
                    "要解释的主题：流程 / 预处理 / 训练参数 / 模型参数 / 指标 / "
                    "调参手册（指标不理想时该动哪个参数）/ "
                    "推理阶段参数（训练之后才生效、改了不用重训的旋钮，如决策阈值）"
                ),
            },
            "model": {"type": "string", "description": "topic=model 时必填"},
            "metric": {"type": "string", "description": "topic=metrics 时可指定单个指标"},
        },
        "required": ["topic"],
    }
    output_schema = {"type": "object"}
    permission = "read_data"
    risk_level = "low"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        topic = str(params.get("topic") or "pipeline")
        if topic == "pipeline":
            data: Any = PIPELINE_STEPS
            summary = f"ML 流程共 {len(PIPELINE_STEPS)} 步：数据输入 → 预处理 → 训练 → 评估 → 持久化 → 推理 → 解释"
        elif topic == "preprocessing":
            data = PREPROCESSING_PARAMS
            summary = f"预处理参数 {len(PREPROCESSING_PARAMS)} 项"
        elif topic == "training":
            data = TRAINING_PARAMS
            summary = f"训练参数 {len(TRAINING_PARAMS)} 项"
        elif topic == "model":
            name = params.get("model")
            if not name or name not in MODEL_PARAMS:
                return ToolResult.fail(
                    "请提供有效的 model 名称",
                    metadata={"available": sorted(MODEL_PARAMS)},
                )
            data = MODEL_PARAMS[name]
            summary = f"{name} 可调参数 {len(data.get('params', []))} 项"
        elif topic == "metrics":
            name = params.get("metric")
            if name:
                if name not in METRIC_GUIDE:
                    return ToolResult.fail(
                        f"未知指标 {name!r}",
                        metadata={"available": sorted(METRIC_GUIDE)},
                    )
                data = {name: METRIC_GUIDE[name]}
                summary = f"指标 {name} 说明"
            else:
                data = METRIC_GUIDE
                summary = f"共 {len(METRIC_GUIDE)} 个指标说明"
        elif topic == "tuning":
            # 修：schema 里一直声明支持 tuning，但这里原本没有对应分支，
            # TUNING_PLAYBOOK 被 import 却从未使用 —— 调用会落进 else 报「未知 topic」。
            # 把「该动哪个参数」与「哪些参数不能一起用」放在同一主题下，
            # 因为它们回答的是同一个问题：接下来该怎么改配置。
            data = {"playbook": TUNING_PLAYBOOK, "invalid_combos": PARAM_COMBOS}
            summary = (
                f"调参手册 {len(TUNING_PLAYBOOK)} 条信号 + "
                f"{len(PARAM_COMBOS)} 条非法参数组合"
            )
        elif topic == "inference":
            data = INFERENCE_PARAMS
            affects = [p for p in INFERENCE_PARAMS if p.get("affects_result")]
            summary = (
                f"推理阶段参数 {len(INFERENCE_PARAMS)} 项"
                f"（其中 {len(affects)} 项会改变预测结果）"
            )
        else:
            return ToolResult.fail(
                f"未知 topic {topic!r}",
                metadata={
                    "supported": [
                        "pipeline",
                        "preprocessing",
                        "training",
                        "model",
                        "metrics",
                        "tuning",
                        "inference",
                    ]
                },
            )
        return ToolResult.ok({"topic": topic, "data": data}, summary=summary)
