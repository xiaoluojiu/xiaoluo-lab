"""大规模推理：分块 transform + 分块 predict，把峰值内存钉在预算内。

背景：训练侧可以用随机抽样把行数压到 ``ML_MAX_TRAIN_ROWS`` 以内，但**推理不能抽样**
—— 少预测一行就是少一行结果，抽样会把批量推理变成「随机丢弃样本」。
而 10,000,000 行 × 767 列的稠密矩阵是 57.1 GiB，直接
``numpy._core._exceptions._ArrayMemoryError``。

唯一可行且语义无损的办法是把推理切块：每块独立 ``transform`` + ``predict``，
结果按行拼接。这对估计器是安全的 —— ``predict`` 是逐样本的前向计算，
块与块之间没有依赖（``transform`` 用的也是 fit 阶段定好的参数，不做全局统计）。

与预检（``_assert_dense_fits``）的分工：预检负责**明确拒绝**不该跑的任务，
分块负责**让本该能跑的批量推理真的跑得动**。两者都在，缺一不可。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import polars as pl

from app.core.config import settings
from app.ml_engine.exceptions import MLEngineException

logger = logging.getLogger(__name__)

# 分块时的安全系数。同一时刻会同时存在：原始列数组、transform 输出矩阵、
# sklearn 内部拷贝（ColumnTransformer 会先 column_stack 再切片）、估计器内部副本。
# 只按「单个矩阵」估算一定低估，这里取 4 倍余量；实际仍不够时 batch_* 会自动折半重试。
_SAFETY_FACTOR = 4


def _budget_bytes() -> int:
    """稠密内存预算（字节）。<=0 表示不设限、不分块。"""
    try:
        return int(settings.ML_MAX_DENSE_BYTES)
    except Exception:  # pragma: no cover - 配置层异常时按默认兜底
        return 2 * 1024 * 1024 * 1024


def plan_chunk_rows(n_cols: int, *, budget_bytes: int | None = None) -> int:
    """按稠密内存预算算出单块最大行数。

    返回 ``0`` 表示「不分块」（预算被显式关掉），调用方据此走原路径。
    """
    budget = _budget_bytes() if budget_bytes is None else int(budget_bytes)
    if budget <= 0:
        return 0
    per_row = max(1, int(n_cols)) * 8 * _SAFETY_FACTOR
    return max(1, budget // per_row)


def _out_width(model: Any, pipeline: Any, fallback: int) -> int:
    """分块宽度按**变换后**的列数算（one-hot 展开后的宽度才是真实占用）。"""
    if pipeline is not None and getattr(pipeline, "fitted_", False):
        width = len(getattr(pipeline, "feature_names_out_", None) or [])
        if width:
            return width
    width = len(getattr(model, "feature_names_", None) or [])
    return width or max(1, int(fallback))


def _transform(pipeline: Any, X: pl.DataFrame) -> pl.DataFrame:
    return pipeline.transform(X) if pipeline is not None else X


def _chunked_apply(
    apply: Callable[[pl.DataFrame], Any],
    X: pl.DataFrame,
    *,
    width: int,
    chunk_rows: int | None,
    combine: Callable[[list[Any]], Any],
) -> Any:
    total = X.height
    step = int(chunk_rows) if chunk_rows else plan_chunk_rows(width)
    # 不分块（预算关闭）或整表本来就装得下 —— 两条都走原路径，行为零变化。
    if step <= 0 or total <= step:
        return apply(X)

    parts: list[Any] = []
    start = 0
    while start < total:
        stop = min(start + step, total)
        chunk = X[start:stop]
        try:
            parts.append(apply(chunk))
        except (MLEngineException, MemoryError):
            # 估算偏乐观时自动折半重试（同一块），直到能装下或退无可退。
            # 只在「内存类」异常上重试；真实业务错误直接抛，不做无意义的重试。
            if step <= 1:
                raise
            step = max(1, step // 2)
            logger.warning("推理分块过大，块行数减半为 %s 后重试", step)
            continue
        start = stop
    if not parts:  # pragma: no cover - total > step 时至少产出一块
        return apply(X)
    return combine(parts)


def batch_predict(
    model: Any,
    X: pl.DataFrame,
    *,
    pipeline: Any = None,
    chunk_rows: int | None = None,
) -> pl.Series:
    """分块推理，返回与 ``model.predict`` 完全同形的 Series。

    语义与一次性推理逐样本一致：估计器的 predict 不做跨样本聚合。
    """
    return _chunked_apply(
        lambda chunk: model.predict(_transform(pipeline, chunk)),
        X,
        width=_out_width(model, pipeline, X.width),
        chunk_rows=chunk_rows,
        combine=lambda parts: pl.concat(parts, how="vertical"),
    )


def batch_transform(
    model: Any,
    X: pl.DataFrame,
    *,
    pipeline: Any = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame:
    """分块 transform（PCA 这类降维模型的批量版本）。

    注意：``transformed`` 的输出宽度通常远小于输入宽度，但真正吃内存的是**输入**，
    所以宽度估算取 ``model.feature_names_``（= 输入列）而不是输出列，天然偏保守。
    """
    return _chunked_apply(
        lambda chunk: model.transform(_transform(pipeline, chunk)),
        X,
        width=_out_width(model, pipeline, X.width),
        chunk_rows=chunk_rows,
        combine=lambda parts: pl.concat(parts, how="vertical"),
    )


def batch_predict_proba(
    model: Any,
    X: pl.DataFrame,
    *,
    pipeline: Any = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame:
    """分块预测类别概率（``predict_proba`` 的批量版本）。

    不做异常吞掉：模型本身不支持概率时由 ``model.predict_proba`` 抛
    ``MLEngineException``，调用方按原有方式降级。
    """
    return _chunked_apply(
        lambda chunk: model.predict_proba(_transform(pipeline, chunk)),
        X,
        width=_out_width(model, pipeline, X.width),
        chunk_rows=chunk_rows,
        combine=lambda parts: pl.concat(parts, how="vertical"),
    )


def batch_evaluate(
    model: Any,
    X: pl.DataFrame,
    y: pl.Series,
    *,
    pipeline: Any = None,
    chunk_rows: int | None = None,
) -> dict[str, Any]:
    """分块推理后整体评估。

    指标（accuracy / mae / roc_auc …）是**全局量**，必须见全量预测结果 ——
    所以这里只把「产生预测」这一步分块，指标计算仍在完整序列上做一次。
    """
    from app.ml_engine import evaluation  # 延迟导入，避免与 evaluation 形成循环

    y_pred = batch_predict(model, X, pipeline=pipeline, chunk_rows=chunk_rows)
    task = getattr(model, "task", "")
    if task == "regression":
        return evaluation.evaluate_regression(y, y_pred)
    if task == "classification":
        try:
            y_proba = batch_predict_proba(model, X, pipeline=pipeline, chunk_rows=chunk_rows)
        except MLEngineException:
            y_proba = None
        return evaluation.evaluate_classification(y, y_pred, y_proba)
    raise MLEngineException(
        f"任务类型 {task!r} 不支持 evaluate（仅分类 / 回归）"
    )
