"""Workflow 默认节点 Runner。

节点白名单与前端 Workflow Studio 注册表保持一致：数据处理复用 Data Engine，
机器学习复用现有 Model Registry，AI 节点复用项目已有 LLM Provider。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.agent.llm.base import LLMMessage
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.core.config import settings
from app.core.exceptions import WorkflowException
from app.data_engine.json_utils import json_safe
from app.ml_engine.evaluation import evaluate_clustering
from app.ml_engine.preprocessing import (
    PreprocessingPipeline,
    build_pipeline,
)
from app.ml_engine.registry import MODEL_REGISTRY
from app.workflow.models import flatten_config

logger = logging.getLogger(__name__)

NodeRunner = Callable[[Any, dict[str, Any], dict[str, Any]], Any]


def _remember_df(ctx: dict[str, Any] | None, df) -> None:
    """把当前数据集写入共享上下文，供下游分析节点兜底取用。

    链式 DAG（dataset.read → data.quality_check → data.statistics → report.summary）
    里，中间的分析节点并不「产出」数据集，只产出结论。若下游严格按照直接前驱的
    输出找 DataFrame，就会在第 3 个节点报「上游节点未产生可处理的数据集」，
    这正是 Workflow 跑不起来的根因之一。这里把最近一次的数据集记在上下文里兜底。
    """
    if isinstance(ctx, dict):
        ctx["_last_df"] = df


def _pull_df(upstream: dict[str, Any], ctx: dict[str, Any] | None = None):
    # 1) 直接前驱里带 _df 的优先（最近的数据加工结果）
    for output in upstream.values():
        if isinstance(output, dict) and "_df" in output:
            return output["_df"]
    # 2) 兜底：共享上下文里最近一次载入/加工的数据集
    if isinstance(ctx, dict) and ctx.get("_last_df") is not None:
        return ctx["_last_df"]
    raise WorkflowException("上游节点未产生可处理的数据集（请先连接 data.load 节点）")


def _pull_model(upstream: dict[str, Any]):
    for output in upstream.values():
        if isinstance(output, dict) and "_model" in output:
            return output["_model"]
    raise WorkflowException("上游节点未产生模型，请先连接 ml.train 节点")


def _pull_pipeline(upstream: dict[str, Any]):
    """上游训练节点产出的预处理管道（可能缺失）。"""
    for output in upstream.values():
        if isinstance(output, dict) and output.get("_pipeline") is not None:
            return output["_pipeline"]
    return None


def _summarize(df) -> dict[str, Any]:
    return {"row_count": df.height, "column_count": df.width, "columns": df.columns}


def _ml_config(node) -> dict[str, Any]:
    raw = dict(node.config or {})
    params = raw.get("params")
    return dict(params) if isinstance(params, dict) else {k: v for k, v in raw.items() if k != "__ui"}


# 每个节点类型「缺少就一定跑不起来」的必需 config 键（单一事实源）。
# 取值按 flatten_config 归一化后的平铺视图判断，因此 {"dataset_id": 1} 与
# {"params": {"dataset_id": 1}} 都算齐全。
# 这些键与对应 runner 里真正读取的键必须一致，改动时两边一起改：
#   data.load / dataset.read → _load_dataset 读 config["dataset_id"]
#   ml.train                 → ml_train 读 config["target_column"]（否则直接抛错）
#   ml.evaluate              → ml_evaluate 读 config["target_column"]
# 其余节点（ml.predict / ml.cluster / ml.pca / data.* 操作 / ai.analyze /
# report.summary / data.quality_check / data.statistics / noop）全部有默认值或可
# 从上游推断，刻意不列为必需 —— 宁可运行时报明确错误，也不要误拒合法配置。
NODE_REQUIRED_CONFIG: dict[str, set[str]] = {
    "data.load": {"dataset_id"},
    "dataset.read": {"dataset_id"},
    "ml.train": {"target_column"},
    "ml.evaluate": {"target_column"},
}


def build_default_runners() -> dict[str, NodeRunner]:
    def noop(node, upstream, ctx) -> dict[str, Any]:
        return {"ok": True, "config": {k: v for k, v in dict(node.config or {}).items() if k != "__ui"}}

    def data_load(node, upstream, ctx) -> dict[str, Any]:
        return _load_dataset(node, upstream, ctx)

    def make_op(op_type: str) -> NodeRunner:
        def runner(node, upstream, ctx) -> dict[str, Any]:
            engine = ctx.get("data_engine_service")
            if engine is None:
                raise WorkflowException("缺少 data_engine_service 上下文")
            df = _pull_df(upstream, ctx)
            raw_config = dict(node.config or {})
            params = raw_config.get("params")
            if not isinstance(params, dict):
                params = {k: v for k, v in raw_config.items() if k not in {"dataset_id", "version", "__ui"}}
            new_df = engine.apply_operation(df, op_type, params)
            _remember_df(ctx, new_df)
            return {"_df": new_df, "op_type": op_type, **_summarize(new_df)}
        return runner

    def ml_train(node, upstream, ctx) -> dict[str, Any]:
        df = _pull_df(upstream, ctx)
        config = _ml_config(node)
        model_name = str(config.pop("model", "random_forest_regressor"))
        target = config.pop("target_column", None)
        test_size = float(config.pop("test_size", 0.2))
        random_state = int(config.pop("random_state", 42))
        feature_columns = config.pop("feature_columns", None)
        preprocessing = config.pop("preprocessing", None)
        if not target or target not in df.columns:
            raise WorkflowException("ml.train 需要 params.target_column，并且目标列必须存在")
        if not isinstance(feature_columns, list) or not feature_columns:
            feature_columns = [c for c in df.columns if c != target]
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise WorkflowException(f"训练特征不存在：{missing}")
        # seed 同时用于切分与模型本身，保证 workflow 结果可复现
        if MODEL_REGISTRY.supports_param(model_name, "random_state"):
            config.setdefault("random_state", random_state)
        model = MODEL_REGISTRY.create(model_name, config)
        X, y = df.select(feature_columns), df[target]

        started = time.perf_counter()
        try:
            if model.task == "clustering":
                pipeline = build_pipeline(preprocessing, X)
                Xp = pipeline.fit_transform(X)
                model.fit(Xp)
                metrics = evaluate_clustering(Xp, model.labels_)
                train_rows, test_rows = X.height, 0
            else:
                X_train, X_test, y_train, y_test = PreprocessingPipeline.train_test_split(
                    X, y, test_size=test_size, seed=random_state
                )
                pipeline = build_pipeline(preprocessing, X_train)
                Xtr = pipeline.fit_transform(X_train)
                Xte = pipeline.transform(X_test)
                model.fit(Xtr, y_train)
                metrics = (
                    model.evaluate(Xte, y_test)
                    if model.task in {"classification", "regression"}
                    else {}
                )
                train_rows, test_rows = X_train.height, X_test.height
        except ValueError as exc:
            raise WorkflowException(f"训练失败：{exc}") from exc

        logger.info(
            "ml.train 完成 model=%s task=%s 训练行=%s 测试行=%s 耗时=%.3fs",
            model_name, model.task, train_rows, test_rows, time.perf_counter() - started,
        )
        return {
            "_model": model,
            "_pipeline": pipeline,
            "_df": df,
            "model": model_name,
            "task": model.task,
            "target_column": target,
            "feature_columns": feature_columns,
            "model_features": list(pipeline.feature_names_out_),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "random_state": random_state,
            "metrics": metrics,
            "preprocessing_report": pipeline.report,
        }

    def ml_predict(node, upstream, ctx) -> dict[str, Any]:
        df = _pull_df(upstream, ctx)
        model = _pull_model(upstream)
        pipeline = _pull_pipeline(upstream)
        config = _ml_config(node)
        # 训练若经过预处理，推理必须复用同一管道把原始列转换到模型特征空间
        raw_columns = config.get("feature_columns")
        if not raw_columns:
            for output in upstream.values():
                if isinstance(output, dict) and output.get("feature_columns"):
                    raw_columns = output["feature_columns"]
                    break
        if not raw_columns:
            raw_columns = [c for c in model.feature_names_ if c in df.columns]
        missing = [c for c in raw_columns if c not in df.columns]
        if missing:
            raise WorkflowException(f"ml.predict 推理数据缺少特征列：{missing}")
        X = pipeline.transform(df.select(raw_columns)) if pipeline is not None else df.select(raw_columns)
        prediction = model.predict(X)
        output_column = str(config.get("output_column", "prediction"))
        out = df.with_columns(prediction.alias(output_column))
        logger.info(
            "ml.predict 完成 model=%s 行数=%s 输出列=%s 预处理=%s",
            getattr(model, "name", ""), out.height, output_column, pipeline is not None,
        )
        return {"_df": out, "_model": model, "_pipeline": pipeline, "prediction_column": output_column, **_summarize(out)}

    def ml_evaluate(node, upstream, ctx) -> dict[str, Any]:
        df = _pull_df(upstream, ctx)
        model = _pull_model(upstream)
        pipeline = _pull_pipeline(upstream)
        config = _ml_config(node)
        target = config.get("target_column")
        if not target or target not in df.columns:
            raise WorkflowException("ml.evaluate 需要 params.target_column")
        raw_columns = getattr(model, "feature_names_", [])
        if not raw_columns:
            raise WorkflowException("模型没有记录训练特征")
        raw_columns = [c for c in raw_columns if c in df.columns]
        X = pipeline.transform(df.select(raw_columns)) if pipeline is not None else df.select(raw_columns)
        return {
            "metrics": model.evaluate(X, df[target]),
            "model": getattr(model, "name", ""),
            "rows": df.height,
        }

    def ml_cluster(node, upstream, ctx) -> dict[str, Any]:
        df = _pull_df(upstream, ctx)
        config = _ml_config(node)
        model_name = str(config.pop("model", "kmeans"))
        output_column = str(config.pop("output_column", "cluster"))
        feature_columns = config.pop("feature_columns", None) or list(df.columns)
        preprocessing = config.pop("preprocessing", None)
        if "random_state" not in config:
            config.setdefault("random_state", 42)
        if not MODEL_REGISTRY.supports_param(model_name, "random_state"):
            config.pop("random_state", None)
        model = MODEL_REGISTRY.create(model_name, config)
        X = df.select(feature_columns)
        pipeline = build_pipeline(preprocessing, X)
        Xp = pipeline.fit_transform(X)
        model.fit(Xp)
        # DBSCAN 无 predict：降级使用训练样本的 labels_
        try:
            labels = model.predict(Xp)
        except Exception:  # noqa: BLE001 - DBSCAN 明确不支持 predict
            labels = model.labels_
            if labels is None:
                raise WorkflowException(f"{model_name} 无法产出簇标签") from None
        metrics = evaluate_clustering(Xp, labels)
        out = df.with_columns(labels.alias(output_column))
        logger.info(
            "ml.cluster 完成 model=%s 簇数=%s 轮廓系数=%s",
            model_name, metrics.get("cluster_count"), metrics.get("silhouette"),
        )
        return {
            "_df": out, "_model": model, "_pipeline": pipeline,
            "model": model_name, "output_column": output_column,
            "metrics": metrics, **_summarize(out),
        }

    def ml_pca(node, upstream, ctx) -> dict[str, Any]:
        df = _pull_df(upstream, ctx)
        config = _ml_config(node)
        feature_columns = config.pop("feature_columns", None) or list(df.columns)
        preprocessing = config.pop("preprocessing", None)
        model = MODEL_REGISTRY.create("pca", config)
        X = df.select(feature_columns)
        pipeline = build_pipeline(preprocessing, X)
        Xp = pipeline.fit_transform(X)
        model.fit(Xp)
        out = model.transform(Xp)
        return {
            "_df": out, "_model": model, "_pipeline": pipeline,
            "model": "pca",
            "explained_variance_ratio": model.explained_variance_ratio_,
            **_summarize(out),
        }

    def ai_analyze(node, upstream, ctx) -> dict[str, Any]:
        config = _ml_config(node)
        provider = ctx.get("llm_provider")
        if provider is None:
            # 这里是唯一会「自建 Provider」的分支（工作流的 ctx 从不注入 llm_provider）。
            # 必须先问总开关：否则设置页关掉「启用远程 API 大模型」后，
            # 工作流里的 ai.analyze 仍会偷偷调用远程大模型 —— 开关形同虚设。
            if not settings.remote_llm_available():
                raise WorkflowException(
                    "远程 API 大模型已停用（设置 → AI 与模型），ai.analyze 节点无法执行"
                )
            if settings.LLM_PROVIDER not in {"openai", "openai_compatible"}:
                raise WorkflowException("当前 LLM Provider 未接入 Workflow")
            provider = OpenAICompatibleProvider(settings.LLM_BASE_URL, settings.LLM_MODEL, settings.LLM_API_KEY)
        summaries = []
        for output in upstream.values():
            if isinstance(output, dict):
                summaries.append(
                    {k: v for k, v in output.items() if k not in _PRIVATE_KEYS}
                )
        prompt = str(config.get("prompt", "请分析这个机器学习 Workflow 的结果，指出主要发现、风险和下一步建议。"))
        response = provider.chat([
            LLMMessage(role="system", content="你是小洛实验室的数据科学助手。请基于给定 Workflow 结果回答，避免编造不存在的数据。"),
            LLMMessage(role="user", content=f"{prompt}\nWorkflow 上游结果：{json_safe(summaries)}"),
        ])
        return {"analysis": response.content, "model": response.model}

    default_runners: dict[str, NodeRunner] = {
        "noop": noop,
        "data.load": data_load,
        "data.clean": make_op("missing"),
        "data.duplicate": make_op("duplicate"),
        "data.cast": make_op("cast"),
        "data.string": make_op("string"),
        "data.filter": make_op("filter"),
        "data.transform": make_op("transform"),
        "data.aggregate": make_op("aggregate"),
        "data.pivot": make_op("pivot"),
        "data.melt": make_op("melt"),
        "ml.train": ml_train,
        "ml.predict": ml_predict,
        "ml.evaluate": ml_evaluate,
        "ml.cluster": ml_cluster,
        "ml.pca": ml_pca,
        "ai.analyze": ai_analyze,
    }
    # Agent 语义节点（dataset.read / data.quality_check / data.statistics / report.summary）
    # 必须与原生节点一起在进程启动即注册：惰性补挂会导致部分入口拿不到这些类型，
    # 表现为「Workflow 已创建但执行时报未注册的节点类型」。
    default_runners.update(build_agent_semantic_runners())
    # Agent 常用的数据载入语义别名
    default_runners.setdefault("dataset.read", default_runners["data.load"])
    return default_runners


def _load_dataset(node, upstream, ctx) -> dict[str, Any]:
    """data.load / dataset.read 共用实现：按 dataset_id 载入某个版本快照。

    走 flatten_config 取值：本节点过去只认平铺写法，于是
    ``{"params": {"dataset_id": 1}}``（ml.* 与 data.* 操作节点的通行写法）
    会被当成「没给 dataset_id」而报错 —— 同一份 config 换个节点就失效。
    """
    ds = ctx.get("dataset_service")
    if ds is None:
        raise WorkflowException("缺少 dataset_service 上下文")
    config = flatten_config(node.config)
    dataset_id = config.get("dataset_id")
    if dataset_id is None:
        raise WorkflowException("data.load 节点需要 config.dataset_id")
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError) as exc:
        raise WorkflowException(f"dataset_id 必须是整数，实际为 {dataset_id!r}") from exc
    row = ds.get_version_row(dataset_id, config.get("version"))
    df = ds.load_version(row.dataset_id, row.version)
    _remember_df(ctx, df)
    return {"_df": df, "dataset_id": row.dataset_id, "version": row.version, **_summarize(df)}


def build_agent_semantic_runners() -> dict[str, NodeRunner]:
    """Agent 语义节点 runner。

    Agent 在规划工作流时习惯用「dataset.read / data.quality_check / data.statistics /
    report.summary」这套语义命名。过去这些 runner 是在 workflow 工具的 execute() 里
    惰性补挂到 WorkflowService 上的，存在两个致命问题：

    1. 补挂发生在工具执行时，但 WorkflowService 在 **进程启动时** 就用
       build_default_runners() 建好了 —— 只要 Agent 的计划里第一个 workflow 节点
       不是 list/create/run（例如直接 workflow.run），或服务端走了 REST 的
       /workflows/{id}/run 通道，就会以「未注册的节点类型」被 validator 拒绝，
       这正是「Workflow 工具流无法被触发与执行」的根因之一。
    2. 补挂依赖 `dataset.read not in service.node_runners and "data.load" in ...`，
       任一条件不满足就静默跳过，失败不可见。

    现在把它们作为默认 runner 的一部分，进程启动即注册，任何入口都能跑。
    """
    def quality_check(node, upstream, ctx):
        df = _pull_df(upstream, ctx)
        config = dict(node.config or {})
        checks = config.get("checks") or ["row_count", "duplicate_rows", "missing_values"]
        result: dict[str, Any] = {}
        if "row_count" in checks:
            result["row_count"] = int(df.height)
        result["column_count"] = int(df.width)
        if "duplicate_rows" in checks:
            result["duplicate_rows"] = int(_count_duplicates(df))
        if "missing_values" in checks:
            result["missing_values"] = {c: int(df[c].null_count()) for c in df.columns}
            total_cells = max(int(df.height) * int(df.width), 1)
            missing_cells = sum(result["missing_values"].values())
            result["missing_rate"] = round(missing_cells / total_cells, 6)
        if "unique_keys" in checks and config.get("key_column"):
            key = config["key_column"]
            if key not in df.columns:
                raise WorkflowException(f"唯一键字段不存在：{key}")
            result["unique_keys"] = {
                "column": key,
                "unique_count": int(df[key].n_unique()),
                "null_count": int(df[key].null_count()),
            }
        issues = [
            {"type": "missing_values", "column": c, "count": n}
            for c, n in (result.get("missing_values") or {}).items()
            if n
        ]
        if result.get("duplicate_rows"):
            issues.append({"type": "duplicate_rows", "count": result["duplicate_rows"]})
        result["issues"] = issues
        result["issue_count"] = len(issues)
        # 质量核查不改变数据本身，把数据集透传给下游统计/汇总节点使用
        result["_df"] = df
        return result

    def statistics(node, upstream, ctx):
        df = _pull_df(upstream, ctx)
        config = dict(node.config or {})
        columns = config.get("columns") or df.columns
        metrics = config.get("metrics") or ["count", "mean", "median", "min", "max", "value_counts"]
        output: dict[str, Any] = {}
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col]
            item: dict[str, Any] = {"dtype": str(series.dtype), "null_count": int(series.null_count())}
            if series.dtype.is_numeric():
                if "count" in metrics:
                    item["count"] = int(series.len() - series.null_count())
                if "mean" in metrics:
                    item["mean"] = series.mean()
                if "median" in metrics:
                    item["median"] = series.median()
                if "min" in metrics:
                    item["min"] = series.min()
                if "max" in metrics:
                    item["max"] = series.max()
                if "std" in metrics:
                    item["std"] = series.std()
            elif "value_counts" in metrics:
                try:
                    item["value_counts"] = series.value_counts().sort("count", descending=True).head(10).to_dicts()
                except Exception:  # noqa: BLE001 - 极少数 dtype 无法排序
                    item["value_counts"] = []
            output[col] = item
        # 统计同样不改变数据，透传给下游（不进报告：sanitize_output 会剥离 _df）
        return {"_df": df, "row_count": int(df.height), "columns": output}

    def report_summary(node, upstream, ctx):
        """把上游各节点的关键结论压缩成一段可读摘要，而不是原样回吐。"""
        config = dict(node.config or {})
        sections = config.get("sections") or ["data_quality", "key_statistics", "issues"]
        facts: dict[str, Any] = {"sections": sections}
        for key, value in upstream.items():
            if not isinstance(value, dict):
                continue
            if "row_count" in value and "shape" not in facts:
                facts["shape"] = {"rows": value.get("row_count"), "columns": value.get("column_count")}
            if value.get("issues") is not None:
                issues = value["issues"]
                facts.setdefault("data_quality", {})
                facts["data_quality"][key] = {"issue_count": len(issues), "issues": issues[:20]}
            if isinstance(value.get("columns"), dict):
                facts.setdefault("key_statistics", {})
                facts["key_statistics"][key] = {
                    col: {k: v for k, v in stat.items() if k != "value_counts"}
                    for col, stat in list(value["columns"].items())[:20]
                }
            if value.get("metrics") is not None:
                facts.setdefault("metrics", {})[key] = value["metrics"]
        return facts

    return {
        "data.quality_check": quality_check,
        "data.statistics": statistics,
        "report.summary": report_summary,
    }


def _count_duplicates(df) -> int:
    """兼容不同 polars 版本的重复行计数。"""
    try:
        return int(df.is_duplicated().sum())
    except AttributeError:
        return int(df.height - df.unique().height)


_PRIVATE_KEYS = {"_df", "_model", "_pipeline"}


def sanitize_output(output: Any) -> Any:
    if isinstance(output, dict):
        cleaned = {k: v for k, v in output.items() if k not in _PRIVATE_KEYS}
        return json_safe(cleaned)
    return json_safe(output)
