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
from app.ml_engine.target_inference import (
    recommend_target,
    target_candidates as _target_candidates,
)
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


def _dataset_name(services: ToolServices, dataset_id: int) -> str:
    """数据集名称（拿不到就返回空串）：只作为任务类型的提示，不阻断主流程。"""
    try:
        return str(services.require("dataset_service").get(dataset_id).name or "")
    except Exception:  # noqa: BLE001
        return ""


# 目标列命名约定、日历/时间/标识列判据、诉求语义词典与排序逻辑
# 全部收敛在 `app.ml_engine.target_inference`（那里有完整的分级证据说明）。
# 本模块只负责「把推断接进工具链」。

#: 数据集名称里的任务关键词。用户常在命名时就写明了任务类型。
#: 真实事故：数据集名「航空公司出发延误预测（回归）」——名称已经写明是回归任务，
#: 却因为没有 target/label/y/class 字段而被判成「无监督聚类」，
#: 于是 DepDelay（延误分钟，本该是目标列）降级成普通特征，用户要求的
#: 「选择适合的机器学习模型预测延误」整条链路跑偏（Experiment 28/30 都是 kmeans）。
_TASK_NAME_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("regression", ("回归", "regression", "数值预测", "预测数值", "forecast")),
    ("classification", ("分类", "classification", "二分类", "多分类")),
    ("clustering", ("聚类", "clustering", "分群", "无监督")),
)

_TASK_LABELS = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def _dataset_task_hint(name: Any) -> str | None:
    """从数据集名称里读出任务类型提示（读不出返回 None）。"""
    text = str(name or "").lower()
    for task, keywords in _TASK_NAME_HINTS:
        if any(k in text for k in keywords):
            return task
    return None


def _intent_of(params: dict[str, Any], context: ToolExecutionContext) -> str:
    """取本次推断可用的「用户诉求」文本。

    优先用调用方显式传的 `goal`（LLM 规划器可以填）；
    没填则用运行时注入到 `context.extra` 的原始用户请求 ——
    工具不该要求 LLM **记得**传参才能正确工作。
    """
    return str(params.get("goal") or (context.extra or {}).get("user_request") or "")


def _description_with_target_note(
    description: Any, target_inferred: dict[str, Any] | None
) -> str:
    """把「目标列是推断出来的」写进实验描述，落库可追溯。"""
    base = str(description or "").strip()
    if not target_inferred:
        return base
    reasons = "；".join(str(r) for r in (target_inferred.get("reasons") or [])[:2])
    note = (
        f"[目标列 {target_inferred.get('target')} 由推断得出："
        f"依据 {target_inferred.get('source')}，"
        f"置信度 {target_inferred.get('confidence')}"
        + (f"；{reasons}" if reasons else "")
        + "]"
    )
    return f"{base} {note}".strip()


class MlDetectTaskTool(Tool):
    """根据 Schema / Profile / Target 判断任务类型，结果必须包含理由。"""

    name = "ml.detect_task"
    description = (
        "根据目标列类型与数据分布判断 ML 任务类型，并给出理由。"
        "未指定 target 时按「显式指定 > 命名约定(target/label/y/class) > 诉求语义匹配 > "
        "排除日历·时间·标识列后的唯一候选」分级推断目标列，理由写在 reasons 里；"
        "推断不出来才失败并回传 target_candidates，绝不默认按聚类处理监督任务。"
    )
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "target": {"type": "string"},
            "infer_target": {"type": "boolean"},
            "goal": {
                "type": "string",
                "description": (
                    "用户诉求原文（如「预测出发延误」）。用于按语义匹配目标列；"
                    "不填则回退到运行时注入的用户请求。"
                ),
            },
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
        dataset_name = _dataset_name(services, dataset_id)
        dataset_hint = _dataset_task_hint(dataset_name)
        if dataset_hint:
            reasons.append(f"数据集名称提示这是{_TASK_LABELS.get(dataset_hint, dataset_hint)}任务")
        guess = None
        if target is None:
            if not params.get("infer_target"):
                # 直接调用且未开启推断：保持「未指定即聚类」语义
                reasons.append("未指定目标列：无监督任务")
            else:
                # 分级推断目标列。规划器在规划阶段看不到列名，拿不到 target 是常态，
                # 所以「推断不出来」才是例外 —— 这里必须尽力给出结论与理由。
                guess = recommend_target(
                    df,
                    goal=_intent_of(params, context),
                    dataset_name=dataset_name,
                    dataset_hint=dataset_hint,
                )
                reasons.extend(guess.reasons)
                if guess.column and guess.column in df.columns:
                    target = guess.column
                    reasons.append(
                        f"据此选定目标列 {target}"
                        f"（依据 {guess.source}，置信度 {guess.confidence:.2f}）"
                    )
            if target is None:
                options = _target_candidates(df)
                excluded = list(getattr(guess, "excluded", []) or [])
                # 名称写明是监督任务时，绝不能悄悄退回聚类：那会把本该预测的列
                # 当成普通特征，产出一份「看起来完成了、其实没回答用户问题」的结果。
                if params.get("infer_target") and dataset_hint in ("regression", "classification"):
                    label = _TASK_LABELS.get(dataset_hint, dataset_hint)
                    return ToolResult.fail(
                        f"未能唯一确定目标列：数据集名称提示这是{label}任务，"
                        f"但诉求语义与类型分布都不足以选定唯一目标。"
                        f"请显式指定 target 后重试；"
                        f"回归候选：{options['regression'] or '无'}；"
                        f"分类候选：{options['classification'] or '无'}",
                        data={
                            "task": dataset_hint,
                            "target": None,
                            "needs_target": True,
                            "dataset_hint": dataset_hint,
                            "target_candidates": options,
                            "excluded_columns": excluded,
                            "reasons": reasons,
                        },
                        compact_data={
                            "needs_target": True,
                            "dataset_hint": dataset_hint,
                            "target_candidates": options,
                            "reasons": reasons[-3:],
                        },
                        metadata={"columns": list(df.columns)},
                    )
                reasons.append(f"数据规模 {df.height} 行 x {df.width} 列，适合聚类探索")
                if params.get("infer_target"):
                    reasons.append(
                        "如需回归/分类，请显式指定 target（当前未识别到目标列，按无监督聚类执行）"
                    )
                return ToolResult.ok(
                    {
                        "task": "clustering",
                        "target": None,
                        "reasons": reasons,
                        "dataset_hint": dataset_hint,
                        "target_candidates": options,
                        "excluded_columns": excluded,
                    },
                    summary="任务类型：clustering",
                    warnings=(
                        ["未指定目标列，已按无监督聚类执行；如需回归/分类请显式指定 target"]
                        if params.get("infer_target")
                        else None
                    ),
                )
        if target not in df.columns:
            return ToolResult.fail(
                f"目标列 {target!r} 不存在",
                data={"target_candidates": _target_candidates(df)},
                metadata={"columns": list(df.columns)},
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
        # 名称为「回归」却给了字符串目标列这类冲突，必须写进理由让上层看见，但不覆盖判定
        if dataset_hint and dataset_hint != task:
            reasons.append(
                f"注意：数据集名称提示为{_TASK_LABELS.get(dataset_hint, dataset_hint)}任务，"
                f"但目标列 {target} 的类型指向{_TASK_LABELS.get(task, task)}；"
                "已按目标列实际类型判定，请确认目标列是否选对"
            )
        reasons.append(f"目标列非空值比例 {(1 - df[target].null_count() / df.height):.1%}")
        payload: dict[str, Any] = {
            "task": task,
            "target": target,
            "reasons": reasons,
            "dataset_hint": dataset_hint,
        }
        if guess is not None:
            # 推断出来的目标列必须带出处：谁选的、凭什么、还有哪些备选。
            # 静默替用户决定是这份报告上一轮最要命的问题之一。
            payload.update(
                {
                    "target_source": guess.source,
                    "target_confidence": round(guess.confidence, 2),
                    "target_alternatives": list(guess.alternatives),
                    "target_excluded": list(guess.excluded),
                }
            )
        return ToolResult.ok(
            payload,
            summary=f"任务类型：{task}（目标列 {target}）",
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


def _prune_params_for_model(
    raw: dict[str, Any], model: str
) -> tuple[dict[str, Any], list[str]]:
    """裁掉不属于该模型的超参数；返回 (保留的参数, 被丢弃的参数名)。

    只在「模型被自动替换」时调用：用户为 kmeans 填的 n_clusters 直接喂给
    logistic_regression 会得到一条指向 sklearn 内部签名的报错，排查成本极高。
    """
    meta = MODEL_PARAMS.get(str(model))
    if not isinstance(meta, dict):
        return dict(raw), []  # 未知模型不做裁剪，交给它自己报错
    allowed = {
        str(p.get("name")) for p in (meta.get("params") or []) if isinstance(p, dict)
    }
    if not allowed:
        return dict(raw), []
    kept = {k: v for k, v in raw.items() if k in allowed}
    dropped = sorted(str(k) for k in raw if k not in allowed)
    return kept, dropped


class MlTrainTool(Tool):
    """创建实验并立即训练（复用 ExperimentService，保证可复现）。"""

    name = "ml.train"
    description = (
        "在数据集版本上训练模型：创建 Experiment 并运行，返回评估指标。"
        "model 填 auto 表示按任务类型自动选择（分类→logistic_regression，回归→linear_regression，聚类→kmeans）。"
        "监督任务未给 target 时，会调用 ml.detect_task 按「命名约定 → 诉求语义 → "
        "排除日历·时间·标识列后的唯一候选」自动推断目标列，并把推断依据一并回传"
        "（target_inferred）；确实推断不出来才会失败并回传候选目标列，"
        "而不会默认按聚类执行。"
    )
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "target": {"type": "string", "description": "预测目标列；未填则自动推断，推断不出才报错"},
            "task": {
                "type": "string",
                "description": "任务类型；留空则按 target 类型自动判定",
                "enum": ["classification", "regression", "clustering"],
            },
            "goal": {
                "type": "string",
                "description": "用户诉求原文（如「预测出发延误」），用于自动推断目标列",
            },
            "model": {"type": "string", "description": "模型名；填 auto 表示按任务类型自动选择（分类→logistic_regression，回归→linear_regression，聚类→kmeans）"},
            "params": {"type": "object"},
            "preprocessing": {"type": "object"},
            "excluded_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "训练前排除的特征列（如 id、主键、高基数字符串）；目标列无需列入，列入也等价于不列",
            },
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
        target_inferred: dict[str, Any] | None = None
        if task is None:
            # 这里不再另起一套「预检」：`ml.detect_task` 自己就会在
            # 「数据集名写明是监督任务、但目标列无法唯一确定」时失败并回传候选。
            # 两处重复只会让口径分叉 —— 旧预检比 detect 少一层语义推断，
            # 于是同一件事出现了两条不同的报错文案（事故里就是这么来的）。
            detect = MlDetectTaskTool().execute(
                {
                    "dataset_id": dataset_id,
                    "version": version_row.version,
                    "target": target,
                    # ★ 必须传：ml.detect_task 默认关闭目标列推断（"未指定即聚类"）。
                    # 少了这个参数，Agent 的「训练一个模型」链路会静默退化成聚类
                    # （注释声称会传、代码没传，是真实事故的直接原因）。
                    "infer_target": True,
                    # 用户诉求用于语义匹配目标列；LLM 没填时由运行时注入兜底。
                    "goal": _intent_of(params, context),
                },
                context,
                services,
            )
            if not detect.success:
                return detect
            task = detect.data["task"]
            task_inferred = True
            if detect.data.get("target") and not target:
                target = detect.data["target"]
                params["target"] = target
                target_inferred = {
                    "target": target,
                    "source": detect.data.get("target_source"),
                    "confidence": detect.data.get("target_confidence"),
                    "reasons": list(detect.data.get("reasons") or [])[:5],
                }

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
        if model_adjusted:
            # 换过模型就必须裁掉不属于新模型的参数。真实缺陷：任务被判成分类后模型
            # 从 kmeans 换成 logistic_regression，params 里的 n_clusters 被一起带过去，
            # 训练在 sklearn 抛 `LogisticRegression.__init__() got an unexpected keyword
            # argument 'n_clusters'` —— 报错文本完全看不出是「换了模型」引起的。
            raw_params, dropped = _prune_params_for_model(raw_params, params["model"])
            if dropped:
                model_adjusted["dropped_params"] = dropped
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
            description=_description_with_target_note(
                params.get("description"), target_inferred
            ),
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
                # 目标列是推断出来的时候必须回执：否则报告里出现一个
                # 用户从没指定过的目标列，谁也说不清它从哪来。
                **({"target_inferred": target_inferred} if target_inferred else {}),
            },
            summary=(
                f"训练成功：{exp.model}"
                + (f"（目标列 {target} 为推断结果）" if target_inferred else "")
            ),
            metadata={
                "task": task,
                "model": params["model"],
                "model_adjusted": model_adjusted,
                "target_inferred": target_inferred,
            },
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
