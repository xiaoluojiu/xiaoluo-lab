"""多文件合并（N 个数据集纵向整合 / 字段并集）。

与 MergeExecutor（两表 join）互补：本模块面向「多文件整合到一张宽表」场景——
勾选多个数据集后，分别计算共有字段（全部文件都含）与独有字段（仅部分文件含），
用户选择要保留的字段，按列名纵向 concat（diagonal，缺失字段补 null）。
输出可为新数据集，或写入指定已有数据集的新版本。

设计要点：
- 共有字段 common_columns：出现在所有文件中的字段。
- 独有字段 unique_columns：只出现在部分文件中的字段（per-file 维度可见 only_in_this）。
- 类型归一：同一字段在不同文件 dtype 不一致时，统一为兼容超类型，避免 concat 失败。
"""

from __future__ import annotations

from typing import Any

import polars as pl

from app.data_engine.exceptions import TransformError


def _common_dtype(dtypes: list[pl.DataType]) -> pl.DataType:
    """为同一字段在多个文件中的 dtype 选取兼容超类型。"""
    dtypes = [d for d in dtypes if d is not None]
    if not dtypes:
        return pl.String
    if all(d.is_numeric() for d in dtypes):
        if all(d.is_integer() for d in dtypes):
            return pl.Int64
        return pl.Float64
    if all(d.is_temporal() for d in dtypes):
        return pl.Datetime if any(d == pl.Datetime for d in dtypes) else pl.Date
    # 字符串 / 布尔 / 混合 -> 统一为字符串，确保可拼接
    return pl.String


def analyze_multi_merge(frames: list[pl.DataFrame]) -> dict[str, Any]:
    """分析多个数据集的字段交集 / 独有字段，供前端勾选。"""
    if not frames:
        raise TransformError("多文件合并至少需要一个数据集")

    column_sets = [set(f.columns) for f in frames]
    all_columns = sorted(set().union(*column_sets))
    common = sorted(set.intersection(*column_sets)) if column_sets else []

    files: list[dict[str, Any]] = []
    for idx, f in enumerate(frames):
        files.append(
            {
                "index": idx,
                "row_count": f.height,
                "column_count": f.width,
                "columns": list(f.columns),
                "only_in_this": [c for c in f.columns if c not in common],
            }
        )

    field_presence: list[dict[str, Any]] = []
    for c in all_columns:
        present = [i for i, cs in enumerate(column_sets) if c in cs]
        field_presence.append(
            {
                "column": c,
                "present_in": present,
                "in_all": len(present) == len(frames),
                "in_count": len(present),
            }
        )

    return {
        "file_count": len(frames),
        "all_columns": all_columns,
        "common_columns": common,
        "unique_columns": [c for c in all_columns if c not in common],
        "field_presence": field_presence,
        "files": files,
    }


def execute_multi_merge(
    frames: list[pl.DataFrame],
    selected_columns: list[str] | None = None,
    *,
    add_source: bool = False,
    source_column: str = "source",
    source_labels: list[str] | None = None,
) -> pl.DataFrame:
    """纵向合并多个数据集，按列名对齐；缺失字段在不存在的文件中补 null。"""
    if not frames:
        raise TransformError("多文件合并至少需要一个数据集")

    union = set().union(*[set(f.columns) for f in frames])
    if selected_columns is None:
        selected_columns = sorted(union)
    selected_columns = [c for c in selected_columns if c in union]

    if not selected_columns:
        raise TransformError("多文件合并至少需要选择一个字段")

    for col in selected_columns:
        if not any(col in f.columns for f in frames):
            raise TransformError(
                f"所选字段在所有数据集中都不存在：{col!r}",
                details={"column": col},
            )

    # 每个所选字段的超类型（用于跨文件 dtype 归一）
    common_dtypes = {
        col: _common_dtype([f.schema.get(col) for f in frames if col in f.columns])
        for col in selected_columns
    }

    parts: list[pl.DataFrame] = []
    for i, f in enumerate(frames):
        present = [c for c in selected_columns if c in f.columns]
        part = f.select(present)
        # dtype 归一：避免同名字段跨文件类型不一致导致 concat 失败
        cast_exprs = [
            pl.col(c).cast(common_dtypes[c]).alias(c)
            for c in present
            if f.schema[c] != common_dtypes[c]
        ]
        if cast_exprs:
            part = part.with_columns(cast_exprs)
        if add_source:
            label = (
                source_labels[i]
                if source_labels and i < len(source_labels) and source_labels[i]
                else f"file_{i + 1}"
            )
            part = part.with_columns(pl.lit(label).alias(source_column))
        parts.append(part)

    try:
        merged = pl.concat(parts, how="diagonal")
    except Exception as exc:  # noqa: BLE001
        raise TransformError(
            f"多文件合并失败：{exc}",
            details={"reason": str(exc)},
        ) from exc

    return merged
