"""Join Key 分析（Prompt 056 / T0-M5 修复 composite key）。

检查 Key 的：唯一性 / 重复 / 空值 / 类型 / 覆盖率，
并识别连接基数：one-to-one / one-to-many / many-to-one / many-to-many。

T0-M5 修复：原版只接受单个 string key，但 MergePlan.keys 是 list[JoinKey]。
现在新增 composite key 支持：
- analyze_join_keys_composite 接受 list[str] 作为左右 keys
- cardinality 基于组合 key 元组的唯一性判断
- 旧版 analyze_join_keys / infer_cardinality 保留为薄壳兼容
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

CARDINALITIES = ("one-to-one", "one-to-many", "many-to-one", "many-to-many")


@dataclass
class KeyProfile:
    """单侧 Key 列画像（单列）。"""

    column: str
    dtype: str
    row_count: int
    unique_count: int
    null_count: int
    duplicate_count: int
    coverage: float  # 非空覆盖率 [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "row_count": self.row_count,
            "unique_count": self.unique_count,
            "null_count": self.null_count,
            "duplicate_count": self.duplicate_count,
            "coverage": round(self.coverage, 6),
        }


def analyze_key(df: pl.DataFrame, column: str) -> KeyProfile:
    """分析单侧 Join Key（单列画像）。"""
    if column not in df.columns:
        raise KeyError(f"key column not found: {column!r}")
    s = df[column]
    row_count = s.len()
    unique_count = int(s.n_unique())
    null_count = int(s.null_count())
    duplicate_count = row_count - unique_count if null_count == 0 else max(
        0, row_count - null_count - int(s.drop_nulls().n_unique())
    )
    coverage = (row_count - null_count) / row_count if row_count else 0.0
    return KeyProfile(
        column=column,
        dtype=str(s.dtype),
        row_count=row_count,
        unique_count=unique_count,
        null_count=null_count,
        duplicate_count=duplicate_count,
        coverage=coverage,
    )


def infer_cardinality(
    left_df: pl.DataFrame, left_key: str, right_df: pl.DataFrame, right_key: str
) -> str:
    """识别连接基数（单 key 版本，保留兼容）。"""
    return infer_cardinality_composite(left_df, [left_key], right_df, [right_key])


def infer_cardinality_composite(
    left_df: pl.DataFrame,
    left_keys: list[str],
    right_df: pl.DataFrame,
    right_keys: list[str],
) -> str:
    """识别连接基数（composite key 版本）。

    基于非空 key 元组的重复情况判断：
    - l_multi：左侧存在重复的 key 元组
    - r_multi：右侧存在重复的 key 元组
    """
    if not left_keys or not right_keys:
        return "many-to-many"

    l_null_mask = pl.any_horizontal([left_df[c].is_null() for c in left_keys])
    r_null_mask = pl.any_horizontal([right_df[c].is_null() for c in right_keys])
    l_non_null_df = left_df.filter(~l_null_mask)
    r_non_null_df = right_df.filter(~r_null_mask)
    l_non_null = l_non_null_df.height
    r_non_null = r_non_null_df.height
    l_unique = int(l_non_null_df.select(left_keys).n_unique()) if l_non_null else 0
    r_unique = int(r_non_null_df.select(right_keys).n_unique()) if r_non_null else 0

    l_multi = l_non_null > l_unique
    r_multi = r_non_null > r_unique

    if not l_multi and not r_multi:
        return "one-to-one"
    if not l_multi and r_multi:
        return "one-to-many"
    if l_multi and not r_multi:
        return "many-to-one"
    return "many-to-many"


def analyze_join_keys(
    left_df: pl.DataFrame, left_key: str, right_df: pl.DataFrame, right_key: str
) -> dict[str, Any]:
    """完整 Key 分析（单 key 版本，薄壳转发到 composite 版本）。"""
    return analyze_join_keys_composite(left_df, [left_key], right_df, [right_key])


def analyze_join_keys_composite(
    left_df: pl.DataFrame,
    left_keys: list[str],
    right_df: pl.DataFrame,
    right_keys: list[str],
) -> dict[str, Any]:
    """完整 composite key 分析：双侧画像 + 基数 + 覆盖交集。

    单 key 场景下退化为单列画像；composite 场景下 left/right 画像
    描述整个 key 元组（column 用 ","-join 标识）。
    """
    if len(left_keys) != len(right_keys):
        raise ValueError(
            f"left_keys 与 right_keys 数量不一致：{len(left_keys)} vs {len(right_keys)}"
        )
    for k in left_keys:
        if k not in left_df.columns:
            raise KeyError(f"left key column not found: {k!r}")
    for k in right_keys:
        if k not in right_df.columns:
            raise KeyError(f"right key column not found: {k!r}")

    # 单 key 时复用单列画像；composite 时构造合成画像
    if len(left_keys) == 1:
        left_profile = analyze_key(left_df, left_keys[0]).to_dict()
        right_profile = analyze_key(right_df, right_keys[0]).to_dict()
    else:
        left_profile = _composite_profile(left_df, left_keys)
        right_profile = _composite_profile(right_df, right_keys)

    cardinality = infer_cardinality_composite(left_df, left_keys, right_df, right_keys)

    # 覆盖交集：左右非空 key 元组互相 is_in 的命中率
    l_null_mask = pl.any_horizontal([left_df[c].is_null() for c in left_keys])
    r_null_mask = pl.any_horizontal([right_df[c].is_null() for c in right_keys])
    l_tuples = left_df.filter(~l_null_mask).select(left_keys).unique()
    r_tuples = right_df.filter(~r_null_mask).select(right_keys).unique()

    l_matched = _composite_overlap_count(l_tuples, left_keys, r_tuples, right_keys)
    r_matched = _composite_overlap_count(r_tuples, right_keys, l_tuples, left_keys)

    l_cov = round(l_matched / l_tuples.height, 6) if l_tuples.height else 0.0
    r_cov = round(r_matched / r_tuples.height, 6) if r_tuples.height else 0.0
    return {
        "left": left_profile,
        "right": right_profile,
        "cardinality": cardinality,
        "left_key_coverage_in_right": l_cov,
        "right_key_coverage_in_left": r_cov,
    }


def _composite_profile(df: pl.DataFrame, keys: list[str]) -> dict[str, Any]:
    """构造 composite key 的合成画像。"""
    null_count = int(df.select(pl.any_horizontal([df[c].is_null() for c in keys]).sum()).item())
    row_count = df.height
    non_null_df = df.filter(~pl.any_horizontal([df[c].is_null() for c in keys]))
    unique_count = int(non_null_df.select(keys).n_unique()) if non_null_df.height else 0
    duplicate_count = max(0, row_count - null_count - unique_count)
    coverage = (row_count - null_count) / row_count if row_count else 0.0
    return {
        "column": ",".join(keys),
        "columns": list(keys),
        "dtype": "composite",
        "row_count": row_count,
        "unique_count": unique_count,
        "null_count": null_count,
        "duplicate_count": duplicate_count,
        "coverage": round(coverage, 6),
    }


def _composite_overlap_count(
    left_tuples: pl.DataFrame,
    left_keys: list[str],
    right_tuples: pl.DataFrame,
    right_keys: list[str],
) -> int:
    """计算 left_tuples 中有多少出现在 right_tuples 中（按位置对应）。"""
    if left_tuples.height == 0 or right_tuples.height == 0:
        return 0
    # 重命名 left 列对齐 right 列名
    rename_map = {lk: rk for lk, rk in zip(left_keys, right_keys, strict=False)}
    renamed = left_tuples.rename(rename_map)
    return int(renamed.join(right_tuples, on=right_keys, how="semi").height)
