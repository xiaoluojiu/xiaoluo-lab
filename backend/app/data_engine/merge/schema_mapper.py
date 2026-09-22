"""字段映射候选生成（Prompt 055 / T0-M7 schema 文档）。

比较字段名 / 字段类型 / 样本值 / 唯一性 / 缺失率，
输出 source_column / target_column / confidence / reason。
只生成候选，不直接执行 Merge。

T0-M7：schema 统一约定
=====================
本模块的 MappingCandidate 描述"哪一列对应哪一列"（语义：左列 ↔ 右列的相似度候选）：
    - source_column：左表（left_df）的列
    - target_column：右表（right_df）的列

而执行阶段的 MergePlan.mapping 使用 ColumnMapping 描述"右列重命名为输出列名"：
    - right_column：右表的列
    - output_column：合并后该列的新名字

两套 schema 在前端层面必须区分：
    - MappingCandidate → UI 展示"建议对应关系"
    - ColumnMapping     → UI 让用户决定"右表这列在输出里叫什么"

如果用户基于 MappingCandidate 的 target_column 作为 right_column 构造 ColumnMapping，
应当显式给出 output_column，不要让 UI 猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import polars as pl


@dataclass
class MappingCandidate:
    """一条字段映射候选。"""

    source_column: str
    target_column: str
    confidence: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "target_column": self.target_column,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _category(dtype: pl.DataType) -> str:
    if dtype == pl.String or str(dtype) == "Categorical":
        return "string"
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if dtype.is_temporal():
        return "temporal"
    if dtype == pl.Boolean:
        return "bool"
    return "other"


def _name_similarity(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # 前缀 / 后缀强信号（如 user_id vs user_id_right）
    if a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a):
        ratio = max(ratio, 0.85)
    return ratio


def _value_overlap(left: pl.Series, right: pl.Series, max_unique: int = 50) -> float | None:
    """低基数字符串列的样本值重合率；基数过高或非字符串返回 None。"""
    if left.dtype != pl.String or right.dtype != pl.String:
        return None
    ls, rs = left.drop_nulls().unique(), right.drop_nulls().unique()
    if ls.len() == 0 or rs.len() == 0 or ls.len() > max_unique or rs.len() > max_unique:
        return None
    overlap = ls.is_in(rs).sum()
    return float(overlap) / max(ls.len(), rs.len())


def suggest_mappings(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    *,
    min_confidence: float = 0.4,
    max_candidates: int = 50,
) -> list[MappingCandidate]:
    """为左右两表生成字段映射候选（不执行 Merge）。

    打分维度：字段名相似度（主）、类型一致性、样本值重合、
    唯一率 / 缺失率接近程度。
    """
    candidates: list[MappingCandidate] = []
    for lcol in left_df.columns:
        ls = left_df[lcol]
        l_unique_rate = ls.n_unique() / ls.len() if ls.len() else 0.0
        l_missing_rate = ls.null_count() / ls.len() if ls.len() else 0.0
        l_cat = _category(ls.dtype)

        for rcol in right_df.columns:
            rs = right_df[rcol]
            r_cat = _category(rs.dtype)
            name_sim = _name_similarity(lcol, rcol)
            if name_sim < min_confidence:
                continue

            # 类型：一致 1.0，数值族之间 0.7，其余 0.0
            if l_cat == r_cat:
                type_score = 1.0
            elif {l_cat, r_cat} <= {"int", "float"}:
                type_score = 0.7
            else:
                type_score = 0.0

            reasons: list[str] = [f"name similarity {name_sim:.2f}"]
            score = 0.6 * name_sim + 0.25 * type_score

            overlap = _value_overlap(ls, rs)
            evidence: dict[str, Any] = {
                "name_similarity": round(name_sim, 4),
                "left_dtype": str(ls.dtype),
                "right_dtype": str(rs.dtype),
            }
            if overlap is not None:
                score += 0.1 * overlap
                evidence["value_overlap"] = round(overlap, 4)
                if overlap > 0.5:
                    reasons.append(f"sample value overlap {overlap:.0%}")

            r_unique_rate = rs.n_unique() / rs.len() if rs.len() else 0.0
            r_missing_rate = rs.null_count() / rs.len() if rs.len() else 0.0
            profile_sim = 1.0 - abs(l_unique_rate - r_unique_rate) / 2 - abs(
                l_missing_rate - r_missing_rate
            ) / 2
            score += 0.05 * profile_sim
            evidence.update(
                {
                    "left_unique_rate": round(l_unique_rate, 4),
                    "right_unique_rate": round(r_unique_rate, 4),
                    "left_missing_rate": round(l_missing_rate, 4),
                    "right_missing_rate": round(r_missing_rate, 4),
                }
            )

            reason = "; ".join(reasons) + (f"; type {l_cat}={r_cat}" if type_score == 1.0 else "")
            candidates.append(
                MappingCandidate(
                    source_column=lcol,
                    target_column=rcol,
                    confidence=min(score, 1.0),
                    reason=reason,
                    evidence=evidence,
                )
            )

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:max_candidates]
