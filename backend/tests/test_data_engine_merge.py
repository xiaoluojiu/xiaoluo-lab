"""Prompt 055-060：Merge 系统（映射 / Key 分析 / 计划 / 校验 / 执行 / 报告）测试。"""

import polars as pl
import pytest
from app.data_engine.exceptions import MergeError
from app.data_engine.merge.executor import MergeExecutor
from app.data_engine.merge.key_analyzer import analyze_key, infer_cardinality
from app.data_engine.merge.plan import ColumnMapping, JoinKey, MergePlan
from app.data_engine.merge.report import MergeReport
from app.data_engine.merge.schema_mapper import suggest_mappings
from app.data_engine.merge.validator import validate_merge_plan


@pytest.fixture()
def left_df() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amount": [10.0, 20.0, 30.0]})


@pytest.fixture()
def right_df() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2, 4], "tag": ["x", "y", "z"], "amount": [99.0, 88.0, 77.0]})


def plan(join_type: str = "inner", **kwargs) -> MergePlan:
    return MergePlan(
        left={"name": "left"}, right={"name": "right"},
        keys=[JoinKey(left="id", right="id")], join_type=join_type, **kwargs,
    )


class TestSchemaMapper:
    def test_exact_name_match(self):
        left = pl.DataFrame({"user_id": [1, 2], "v": [1, 2]})
        right = pl.DataFrame({"user_id": [1, 2], "w": [3, 4]})
        candidates = suggest_mappings(left, right)
        assert any(
            c.source_column == "user_id" and c.target_column == "user_id" and c.confidence > 0.8
            for c in candidates
        )

    def test_no_execution(self):
        """映射候选只是描述，不修改数据。"""
        left = pl.DataFrame({"a": [1]})
        right = pl.DataFrame({"a": [2]})
        before = (left.height, right.height)
        suggest_mappings(left, right)
        assert (left.height, right.height) == before


class TestKeyAnalyzer:
    def test_analyze_key(self):
        df = pl.DataFrame({"k": ["a", "a", "b", None]})
        profile = analyze_key(df, "k")
        assert profile.unique_count == 3  # a, b, null
        assert profile.null_count == 1
        assert profile.coverage == pytest.approx(0.75)

    def test_cardinality(self):
        left = pl.DataFrame({"k": [1, 2, 3]})
        right = pl.DataFrame({"k": [1, 2, 3]})
        assert infer_cardinality(left, "k", right, "k") == "one-to-one"

        one, many = pl.DataFrame({"k": [1, 2]}), pl.DataFrame({"k": [1, 1, 2, 2]})
        assert infer_cardinality(one, "k", many, "k") == "one-to-many"
        assert infer_cardinality(many, "k", one, "k") == "many-to-one"
        assert infer_cardinality(many, "k", many, "k") == "many-to-many"


class TestValidator:
    def test_valid_plan(self, left_df, right_df):
        result = validate_merge_plan(plan(), left_df, right_df)
        assert result.ok is True
        assert result.errors == []

    def test_missing_key(self, left_df, right_df):
        p = plan()
        p.keys = [JoinKey(left="id", right="nope")]
        result = validate_merge_plan(p, left_df, right_df)
        assert result.ok is False
        assert any("right key column not found" in e for e in result.errors)

    def test_type_conflict(self, left_df, right_df):
        p = plan()
        right_df = right_df.with_columns(pl.col("id").cast(pl.String))
        result = validate_merge_plan(p, left_df, right_df)
        assert any("type conflict" in e for e in result.errors)

    def test_many_to_many_rejected(self):
        left = pl.DataFrame({"k": [1, 1]})
        right = pl.DataFrame({"k": [1, 1]})
        p = MergePlan(keys=[JoinKey("k", "k")])
        result = validate_merge_plan(p, left, right)
        assert result.ok is False
        assert any("many-to-many" in e for e in result.errors)


class TestExecutor:
    def test_inner_join(self, left_df, right_df):
        merged, report = MergeExecutor().execute(left_df, right_df, plan("inner"))
        assert merged.height == 2
        assert report.input_rows_left == 3
        assert report.input_rows_right == 3
        assert report.output_rows == 2
        assert report.matched_rows == 2
        assert report.unmatched_rows_left == 1
        assert report.unmatched_rows_right == 1

    def test_left_join(self, left_df, right_df):
        merged, report = MergeExecutor().execute(left_df, right_df, plan("left"))
        assert merged.height == 3
        assert report.unmatched_rows_left == 1

    def test_outer_join(self, left_df, right_df):
        merged, report = MergeExecutor().execute(left_df, right_df, plan("outer"))
        assert merged.height == 4
        assert report.matched_rows == 2

    def test_conflict_columns_suffixed(self, left_df, right_df):
        merged, _ = MergeExecutor().execute(left_df, right_df, plan("left"))
        assert "amount_right" in merged.columns
        assert "amount" in merged.columns  # 左表保留原名

    def test_mapping_renames(self, left_df, right_df):
        p = plan("inner", mapping=[ColumnMapping(right_column="tag", output_column="label")])
        merged, _ = MergeExecutor().execute(left_df, right_df, p)
        assert "label" in merged.columns

    def test_validation_blocks_execution(self, left_df, right_df):
        p = plan()
        p.keys = [JoinKey(left="nope", right="id")]
        with pytest.raises(MergeError):
            MergeExecutor().execute(left_df, right_df, p)

    def test_invalid_join_type(self, left_df, right_df):
        with pytest.raises(MergeError):
            MergeExecutor().execute(left_df, right_df, plan("cross"))

    def test_duplicate_keys_reported(self):
        left = pl.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
        right = pl.DataFrame({"k": [1, 2], "w": [1, 2]})
        p = MergePlan(
            keys=[JoinKey("k", "k")],
            right_columns=["w"],
        )
        _, report = MergeExecutor().execute(left, right, p)
        assert report.duplicate_keys_left == 1
        assert report.duplicate_keys_right == 0


class TestPlanAndReport:
    def test_plan_to_dict_roundtrip(self):
        p = plan("left", mapping=[ColumnMapping("tag", "label")])
        p.warnings.append("test")
        data = p.to_dict()
        assert data["join_type"] == "left"
        assert data["keys"][0] == {"left": "id", "right": "id"}
        assert data["mapping"][0]["output_column"] == "label"

    def test_report_to_dict(self):
        report = MergeReport(join_type="inner", output_rows=5)
        data = report.to_dict()
        assert data["output_rows"] == 5
