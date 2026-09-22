"""MergeExecutor（Prompt 059）。

只有 Validation 通过后才允许执行。支持 inner / left / right / outer。
输出：DataFrame + MergeReport（版本化与 Operation 记录由 Service 层完成）。
"""

from __future__ import annotations

import polars as pl

from app.data_engine.exceptions import MergeError
from app.data_engine.merge.plan import MergePlan
from app.data_engine.merge.report import MergeReport
from app.data_engine.merge.validator import validate_or_raise

JOIN_TYPES = ("inner", "left", "right", "outer")


class MergeExecutor:
    """合并执行器。"""

    def execute(
        self, left_df: pl.DataFrame, right_df: pl.DataFrame, plan: MergePlan
    ) -> tuple[pl.DataFrame, MergeReport]:
        """校验并执行合并；返回 (merged_df, report)。"""
        if plan.join_type not in JOIN_TYPES:
            raise MergeError(
                f"unsupported join type: {plan.join_type!r}",
                details={"join_type": plan.join_type, "allowed": list(JOIN_TYPES)},
            )

        # 强制：先校验后执行
        result = validate_or_raise(plan, left_df, right_df)

        report = MergeReport(
            join_type=plan.join_type,
            input_rows_left=left_df.height,
            input_rows_right=right_df.height,
            warnings=result.warnings,
            conflicts=list(plan.conflicts),
        )

        left_keys, right_keys = plan.left_keys(), plan.right_keys()

        # Key 重复组数（用于报告）
        report.duplicate_keys_left = self._duplicate_key_groups(left_df, left_keys)
        report.duplicate_keys_right = self._duplicate_key_groups(right_df, right_keys)

        # 右表参与合并的列：Key + 映射列 +（可选）其余列
        right_selected = self._right_columns(plan, left_df, right_df)
        right_part = right_df.select(right_selected)

        # 冲突列（同名列且未映射）加后缀
        rename_map = self._conflict_renames(plan, left_df, right_selected, right_keys)
        if rename_map:
            right_part = right_part.rename(rename_map)
            join_right_keys = [rename_map.get(k, k) for k in right_keys]
        else:
            join_right_keys = right_keys

        # 映射列统一重命名为 output_column
        mapping_renames = {
            m.right_column: m.output_column
            for m in plan.mapping
            if m.right_column in right_part.columns and m.right_column != m.output_column
        }
        if mapping_renames:
            join_right_keys = [mapping_renames.get(k, k) for k in join_right_keys]
            right_part = right_part.rename(mapping_renames)

        try:
            merged = left_df.join(
                right_part,
                left_on=left_keys,
                right_on=join_right_keys,
                how=plan.join_type,
                coalesce=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise MergeError(
                f"join execution failed: {exc}", details={"reason": str(exc)}
            ) from exc

        report.output_rows = merged.height
        report.output_columns = merged.width
        report.matched_rows, report.unmatched_rows_left, report.unmatched_rows_right = (
            self._match_stats(left_df, right_df, left_keys, join_right_keys, merged, plan.join_type)
        )
        return merged, report

    # ---- 内部工具 ----
    @staticmethod
    def _duplicate_key_groups(df: pl.DataFrame, keys: list[str]) -> int:
        if not keys or df.height == 0:
            return 0
        counts = df.group_by(keys).len()
        return int((counts["len"] > 1).sum())

    @staticmethod
    def _right_columns(plan: MergePlan, left_df: pl.DataFrame, right_df: pl.DataFrame) -> list[str]:
        right_keys = plan.right_keys()
        if plan.right_columns is not None:
            selected = plan.right_columns
        else:
            selected = [c for c in right_df.columns if c not in right_keys]
        # 始终带上 Key 列与映射列
        for k in right_keys:
            if k not in selected:
                selected.append(k)
        for m in plan.mapping:
            if m.right_column not in selected:
                selected.append(m.right_column)
        unknown = [c for c in selected if c not in right_df.columns]
        if unknown:
            raise MergeError(
                "right columns not found", details={"missing": unknown}
            )
        return selected

    @staticmethod
    def _conflict_renames(
        plan: MergePlan, left_df: pl.DataFrame, right_selected: list[str], right_keys: list[str]
    ) -> dict[str, str]:
        """给同名列（非 Key、非映射输出名）加后缀（T0-M6：保证最终列名唯一）。

        若右表已有 amount 与 amount_right，原版会把两者都重命名为 amount_right
        造成静默冲突。新版依次尝试 amount_right、amount_right_2、amount_right_3...
        直到找到未占用名。
        """
        left_cols = set(left_df.columns)
        # 已被占用的输出名集合（左表列名 + 右表 selected + 计划映射输出 + 已生成的 rename）
        used: set[str] = set(left_cols) | set(right_selected)
        for m in plan.mapping:
            used.add(m.output_column)
        # 右表 selected 的原列名（在 rename 之前也算占用，避免 right_to_right 命中）
        used |= set(right_selected)

        renames: dict[str, str] = {}
        for c in right_selected:
            if c in right_keys:
                continue
            target = c
            # 已被 mapping 重命名的按 output_column 判断
            for m in plan.mapping:
                if m.right_column == c:
                    target = m.output_column
                    break
            if target in left_cols and target not in plan.left_keys():
                # T0-M6：依次尝试 target_suffix、target_suffix_2、target_suffix_3...
                candidate = f"{target}{plan.right_suffix}"
                suffix_idx = 2
                while candidate in used:
                    candidate = f"{target}{plan.right_suffix}_{suffix_idx}"
                    suffix_idx += 1
                renames[c] = candidate
                used.add(candidate)
        return renames

    @staticmethod
    def _match_stats(
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        left_keys: list[str],
        right_keys: list[str],
        merged: pl.DataFrame,
        join_type: str,
    ) -> tuple[int, int, int]:
        """计算 matched / unmatched 统计。

        防御：left_keys 与 right_keys 已由 validator 保证一一对应且列名唯一；
        但为防任何绕过校验的路径（历史 Bug：多个 left 列映射到同一 right 列时，
        rename 会产生重复列名导致 polars 崩溃），这里再做一次「右 key 去重」——
        重复的 right key 只保留第一次出现的那个，对应地同步裁剪 left_keys。
        """
        seen_right: dict[str, str] = {}
        dedup_left: list[str] = []
        for lk, rk in zip(left_keys, right_keys, strict=False):
            if rk in seen_right:
                continue
            seen_right[rk] = rk
            dedup_left.append(lk)
        left_keys = dedup_left
        right_keys = list(seen_right.keys())
        if not left_keys:
            return merged.height, left_df.height, right_df.height

        l_keys = left_df.select(left_keys).rename(
            dict(zip(left_keys, right_keys, strict=False))
        )
        matched_pairs = l_keys.join(
            right_df.select(right_keys), on=right_keys, how="inner"
        ).height
        l_null_mask = pl.any_horizontal([left_df[c].is_null() for c in left_keys])
        r_null_mask = pl.any_horizontal([right_df[c].is_null() for c in right_keys])
        matched_left = int(
            left_df.filter(~l_null_mask)[left_keys]
            .rename(dict(zip(left_keys, right_keys, strict=False)))
            .join(right_df.filter(~r_null_mask).select(right_keys), on=right_keys, how="semi")
            .height
        )
        unmatched_left = left_df.height - matched_left
        matched_right = int(
            right_df.filter(~r_null_mask)[right_keys]
            .join(
                left_df.select(left_keys).rename(dict(zip(left_keys, right_keys, strict=False))),
                on=right_keys,
                how="semi",
            )
            .height
        )
        unmatched_right = right_df.height - matched_right
        if join_type == "inner":
            matched = merged.height
        else:
            matched = matched_pairs
        return matched, unmatched_left, unmatched_right
