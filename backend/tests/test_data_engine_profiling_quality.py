"""Prompt 036-045：Preview / Schema / Profile / Quality 测试。"""

from __future__ import annotations

import io
import json

import polars as pl
import pytest
from app.analysis import (
    DuplicateChecker,
    MissingChecker,
    OutlierChecker,
    SchemaChecker,
    analyze_schema,
    build_report,
    profile,
)
from app.data_engine.exceptions import TransformError
from app.data_engine.operations import preview


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["a", "b", "a", "c", None, "b"],
            "age": [25, 30, 25, 40, None, 22],
            "city": ["BJ", "SH", "BJ", "GZ", "SZ", "SH"],
        }
    )


class TestPreview:
    def test_pagination(self, df):
        page1 = preview(df, page=1, page_size=2)
        page2 = preview(df, page=2, page_size=2)
        assert page1["total"] == 6
        assert page1["total_pages"] == 3
        assert [r["id"] for r in page1["items"]] == [1, 2]
        assert [r["id"] for r in page2["items"]] == [3, 4]

    def test_columns(self, df):
        result = preview(df, columns=["id", "name"])
        assert result["columns"] == ["id", "name"]
        assert "city" not in result["items"][0]

    def test_sort(self, df):
        result = preview(df, sort={"column": "age", "desc": True})
        ages = [r["age"] for r in result["items"]]
        assert ages[0] == 40

    def test_filter(self, df):
        result = preview(
            df, filter_conditions=[{"column": "city", "op": "eq", "value": "BJ"}]
        )
        assert result["total"] == 2

    def test_filter_hidden_column(self, df):
        """隐藏列也可用于过滤。"""
        result = preview(
            df,
            columns=["id"],
            filter_conditions=[{"column": "age", "op": "gte", "value": 25}],
        )
        assert result["total"] == 4
        assert result["columns"] == ["id"]

    def test_unknown_column(self, df):
        with pytest.raises(TransformError):
            preview(df, columns=["nope"])

    def test_invalid_page(self, df):
        with pytest.raises(TransformError):
            preview(df, page=0)

    def test_json_serializable(self, df):
        result = preview(df)
        json.dumps(result)  # 不应抛异常


class TestSchemaEngine:
    def test_analyze_schema(self, df):
        result = analyze_schema(df)
        assert result["row_count"] == 6
        assert result["column_count"] == 4
        by_name = {c["column"]: c for c in result["columns"]}
        assert by_name["id"]["dtype"] == "Int64"
        assert by_name["name"]["nullable"] is True
        assert by_name["id"]["nullable"] is False
        assert by_name["name"]["unique_count"] == 4  # a, b, c, null
        assert by_name["name"]["sample_values"] == ["a", "b", "c"]
        json.dumps(result)  # JSON 可序列化


class TestProfile:
    def test_profile_numeric(self, df):
        result = profile(df)
        assert result["row_count"] == 6
        assert result["column_count"] == 4
        age = next(c for c in result["columns"] if c["column"] == "age")
        assert age["type_class"] == "numeric"
        assert age["missing_count"] == 1
        assert age["min"] == 22
        assert age["max"] == 40
        assert age["quantiles"]["0.5"] == 25  # 中位数：22,25,25,30,40
        assert age["missing_rate"] == pytest.approx(1 / 6, abs=1e-5)

    def test_profile_categorical(self, df):
        result = profile(df)
        city = next(c for c in result["columns"] if c["column"] == "city")
        assert city["type_class"] == "categorical"
        assert city["top_values"][0]["value"] in ("BJ", "SH")

    def test_json_serializable(self, df):
        json.dumps(profile(df))


class TestQualityCheckers:
    def test_missing_checker(self, df):
        issues = MissingChecker().check(df)
        cols = {i.column: i for i in issues}
        assert set(cols) == {"name", "age"}
        assert cols["age"].details["missing_count"] == 1
        assert cols["age"].severity in ("low", "medium")

    def test_missing_checker_severity_thresholds(self):
        df = pl.DataFrame({"a": [None] * 9 + [1]})  # 90% 缺失
        issues = MissingChecker().check(df)
        assert issues[0].severity == "critical"

    def test_duplicate_checker_full_row(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        issues = DuplicateChecker().check(df)
        assert len(issues) == 1
        assert issues[0].details["duplicate_rows"] == 2

    def test_duplicate_checker_subset(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        issues = DuplicateChecker(subset=["a"]).check(df)
        assert issues[0].details["duplicate_rows"] == 2

    def test_outlier_checker_iqr(self):
        df = pl.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
        issues = OutlierChecker(method="iqr").check(df)
        assert len(issues) == 1
        assert issues[0].details["method"] == "iqr"
        assert issues[0].details["outlier_count"] == 1

    def test_outlier_checker_zscore(self):
        df = pl.DataFrame({"v": [10.0, 10.1, 10.2, 10.3, 50.0]})
        issues = OutlierChecker(method="zscore", z_threshold=1.0).check(df)
        assert issues[0].details["method"] == "zscore"
        assert issues[0].details["outlier_count"] == 1

    def test_schema_checker(self, df):
        checker = SchemaChecker({"id": "int", "name": "string", "missing_col": "float"})
        issues = checker.check(df)
        kinds = [i.details for i in issues]
        assert any("missing_columns" in d for d in kinds)
        assert any("extra_columns" in d for d in kinds)
        assert not any("type_conflicts" in d for d in kinds)

    def test_schema_checker_type_conflict(self, df):
        checker = SchemaChecker({"name": "int"})
        issues = checker.check(df)
        assert any("type_conflicts" in i.details for i in issues)

    def test_checkers_do_not_modify_data(self, df):
        snapshot = df.clone()
        MissingChecker().check(df)
        DuplicateChecker().check(df)
        OutlierChecker().check(df)
        assert df.equals(snapshot)


class TestQualityReport:
    def test_build_report(self, df):
        report = build_report(df)
        assert report.statistics["row_count"] == 6
        assert report.statistics["issue_count"] == len(report.issues)
        assert report.severity["medium"] >= 1  # name/age 有缺失
        assert report.has_errors is True  # age 含 IQR 意义上的异常值（40）
        assert isinstance(report.recommendations, list)
        data = report.to_dict()
        json.dumps(data)

    def test_report_with_expected_schema(self, df):
        report = build_report(df, expected_schema={"id": "int", "name": "string"})
        assert any(i.check == "schema" for i in report.issues)

    def test_recommendations(self, df):
        report = build_report(df.vstack(df))  # 复制一份制造重复行
        assert any("缺失值" in r for r in report.recommendations)
        assert any("重复数据" in r for r in report.recommendations)


def parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()
