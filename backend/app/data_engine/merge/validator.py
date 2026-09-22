"""MergePlan Validator（Prompt 058）。

检查：Key 存在 / 类型兼容 / 重复 / 多对多 / 字段冲突 / 数据量风险。
返回 ValidationResult（ok + errors + warnings），不执行合并。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from app.data_engine.exceptions import MergeError
from app.data_engine.merge.plan import MergePlan
from app.data_engine.merge.schema_mapper import _category

# 多对多 + 大表时的数据量风险提示阈值（行数）
RISK_ROWS = 100_000


@dataclass
class ValidationResult:
    """校验结果。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_merge_plan(
    plan: MergePlan, left_df: pl.DataFrame, right_df: pl.DataFrame
) -> ValidationResult:
    """校验合并计划；只检查，不修改数据、不执行合并。"""
    errors: list[str] = []
    warnings: list[str] = list(plan.warnings)
    details: dict[str, Any] = {}

    # 1. Key 存在
    for key in plan.keys:
        if key.left not in left_df.columns:
            errors.append(f"left key column not found: {key.left!r}")
        if key.right not in right_df.columns:
            errors.append(f"right key column not found: {key.right!r}")
    if not plan.keys:
        errors.append("merge plan requires at least one join key")

    # 1b. 重复 Key（同一侧列名重复，会导致 join 语义歧义 + 执行层重复列名崩溃）
    from collections import Counter

    dup_left = [c for c, n in Counter(plan.left_keys()).items() if n > 1]
    dup_right = [c for c, n in Counter(plan.right_keys()).items() if n > 1]
    if dup_left:
        errors.append(
            f"duplicate left join key: {dup_left}; each left column may only appear once"
        )
    if dup_right:
        errors.append(
            f"duplicate right join key: {dup_right}; "
            "multiple left columns mapped to the same right column is ambiguous — "
            "specify a single join key per right column"
        )

    # 2. 类型兼容（按类别）
    for key in plan.keys:
        if key.left in left_df.columns and key.right in right_df.columns:
            lc, rc = _category(left_df.schema[key.left]), _category(right_df.schema[key.right])
            compatible = lc == rc or {lc, rc} <= {"int", "float"}
            if not compatible:
                errors.append(
                    f"key type conflict: left {key.left!r} is {lc}, "
                    f"right {key.right!r} is {rc}"
                )

    # 3. Key 空值
    for key in plan.keys:
        if key.left in left_df.columns and left_df[key.left].null_count() > 0:
            warnings.append(f"left key {key.left!r} contains nulls; those rows never match")
        if key.right in right_df.columns and right_df[key.right].null_count() > 0:
            warnings.append(f"right key {key.right!r} contains nulls; those rows never match")

    # 4. 重复 / 多对多（T0-M5：基于全部 keys 的组合基数判断）
    #    注意：key 有重复（dup_left / dup_right）时跳过——key_analyzer 会对重复列名
    #    `select([...])` 触发 polars DuplicateError（重复列名崩溃），而 plan 已注定失败。
    from app.data_engine.merge.key_analyzer import infer_cardinality_composite

    valid_keys = [
        k
        for k in plan.keys
        if k.left in left_df.columns and k.right in right_df.columns
    ]
    if valid_keys and not dup_left and not dup_right:
        lk_list = [k.left for k in valid_keys]
        rk_list = [k.right for k in valid_keys]
        cardinality = infer_cardinality_composite(left_df, lk_list, right_df, rk_list)
        details["cardinality"] = cardinality
        if cardinality == "many-to-many":
            msg = "join is many-to-many; result may explode in size"
            errors.append(msg)
        elif cardinality in ("one-to-many", "many-to-one"):
            warnings.append(
                f"join cardinality is {cardinality}; output may contain duplicated keys"
            )

    # 5. 字段冲突（除 Key 外的同名列）
    right_keys = set(plan.right_keys())
    left_keys = set(plan.left_keys())
    mapped_outputs = {m.output_column for m in plan.mapping}
    right_columns = plan.right_columns or [
        c for c in right_df.columns if c not in right_keys
    ]
    conflict_columns = [
        c
        for c in right_columns
        if c in left_df.columns and c not in left_keys and c not in mapped_outputs
    ]
    if conflict_columns:
        warnings.append(
            f"same-name columns will be suffixed: {conflict_columns} "
            f"(left{plan.left_suffix} / right{plan.right_suffix})"
        )
        details["conflict_columns"] = conflict_columns

    # 6. mapping 列存在性
    for m in plan.mapping:
        if m.right_column not in right_df.columns:
            errors.append(f"mapping right column not found: {m.right_column!r}")
        if m.output_column in left_df.columns and m.output_column not in left_keys:
            errors.append(
                f"mapping output column {m.output_column!r} conflicts with an existing left column"
            )

    # 7. 数据量风险
    if valid_keys and left_df.height * right_df.height > RISK_ROWS * RISK_ROWS // 1000:
        warnings.append(
            f"large join: left {left_df.height} rows x right {right_df.height} rows; "
            "check cardinality before executing"
        )

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, details=details)


def validate_or_raise(
    plan: MergePlan, left_df: pl.DataFrame, right_df: pl.DataFrame
) -> ValidationResult:
    result = validate_merge_plan(plan, left_df, right_df)
    if not result.ok:
        raise MergeError(
            "merge plan validation failed",
            details={"errors": result.errors, "warnings": result.warnings},
        )
    return result
