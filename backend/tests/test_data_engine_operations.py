"""Prompt 046-054：清洗 / 过滤 / 变换 / 聚合 / 透视 / 逆透视 测试。"""

from datetime import datetime

import polars as pl
import pytest
from app.data_engine.exceptions import TransformError
from app.data_engine.operations import (
    add_column,
    aggregate,
    apply_filter,
    apply_string_op,
    cast_columns,
    drop_duplicates,
    handle_missing,
    melt,
    pivot,
)


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": [" Alice ", "BOB", "cathy", None],
            "age": [25, None, 35, 40],
            "score": [88.0, 92.5, 75.0, None],
        }
    )


class TestHandleMissing:
    def test_drop(self, df):
        out = handle_missing(df, strategy="drop")
        assert out.height == 2  # 只有 id=1,3 全行非空

    def test_mean(self):
        df = pl.DataFrame({"v": [1.0, 2.0, None, 4.0]})
        out = handle_missing(df, strategy="mean", columns=["v"])
        assert out["v"][2] == pytest.approx(7 / 3)

    def test_median(self):
        df = pl.DataFrame({"v": [1.0, 2.0, None, 100.0]})
        out = handle_missing(df, strategy="median", columns=["v"])
        assert out["v"][2] == 2.0

    def test_mode(self):
        df = pl.DataFrame({"s": ["a", "a", "b", None]})
        out = handle_missing(df, strategy="mode", columns=["s"])
        assert out["s"][3] == "a"

    def test_constant(self, df):
        out = handle_missing(df, strategy="constant", columns=["name"], value="unknown")
        assert out["name"][3] == "unknown"

    def test_constant_requires_value(self, df):
        with pytest.raises(TransformError):
            handle_missing(df, strategy="constant", columns=["name"])

    def test_mean_rejects_string(self, df):
        with pytest.raises(TransformError):
            handle_missing(df, strategy="mean", columns=["name"])

    def test_unknown_strategy(self, df):
        with pytest.raises(TransformError):
            handle_missing(df, strategy="nope")

    def test_input_not_modified(self, df):
        snapshot = df.clone()
        handle_missing(df, strategy="drop")
        assert df.equals(snapshot)


class TestDropDuplicates:
    def test_first_last(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x1", "x2", "y"]})
        first = drop_duplicates(df, subset=["a"], keep="first")
        assert first["b"].to_list() == ["x1", "y"]
        last = drop_duplicates(df, subset=["a"], keep="last")
        assert last["b"].to_list() == ["x2", "y"]

    def test_full_row_unique(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        out = drop_duplicates(df)  # 全行重复才去重
        assert out.height == 3  # 三行内容各不相同

    def test_subset(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        out = drop_duplicates(df, subset=["a"])
        assert out.height == 2

    def test_invalid_keep(self):
        with pytest.raises(TransformError):
            drop_duplicates(pl.DataFrame({"a": [1]}), keep="nope")


class TestCast:
    def test_str_to_int(self):
        df = pl.DataFrame({"s": ["1", "2", "3"]})
        out = cast_columns(df, {"s": "int"})
        assert out.schema["s"] == pl.Int64
        assert out["s"].to_list() == [1, 2, 3]

    def test_str_to_float(self):
        df = pl.DataFrame({"s": ["1.5", "2.5"]})
        out = cast_columns(df, {"s": "float"})
        assert out["s"].to_list() == [1.5, 2.5]

    def test_str_to_bool(self):
        df = pl.DataFrame({"s": ["yes", "no", "true", "0"]})
        out = cast_columns(df, {"s": "bool"})
        assert out["s"].to_list() == [True, False, True, False]

    def test_str_to_date(self):
        df = pl.DataFrame({"s": ["2024-01-15", "2024-02-20"]})
        out = cast_columns(df, {"s": "date"})
        assert out.schema["s"] == pl.Date

    def test_str_to_date_custom_format(self):
        df = pl.DataFrame({"s": ["15/01/2024", "20/02/2024"]})
        out = cast_columns(df, {"s": "date"}, formats={"s": "%d/%m/%Y"})
        assert out["s"][0].isoformat() == "2024-01-15"

    def test_str_to_datetime(self):
        df = pl.DataFrame({"s": ["2024-01-15 10:30:00"]})
        out = cast_columns(df, {"s": "datetime"})
        assert str(out.schema["s"]).startswith("Datetime")

    def test_float_to_int(self):
        df = pl.DataFrame({"v": [1.0, 2.0, 3.9]})
        out = cast_columns(df, {"v": "int"})
        assert out["v"].to_list() == [1, 2, 3]

    def test_failed_cast_detailed_error(self):
        df = pl.DataFrame({"s": ["1", "abc", "3"]})
        with pytest.raises(TransformError) as exc_info:
            cast_columns(df, {"s": "int"})
        failures = exc_info.value.details["failures"]
        assert failures[0]["column"] == "s"
        assert "abc" in failures[0]["failed_values"]

    def test_unknown_target(self):
        with pytest.raises(TransformError):
            cast_columns(pl.DataFrame({"s": ["1"]}), {"s": "decimal"})

    def test_unknown_column(self):
        with pytest.raises(TransformError):
            cast_columns(pl.DataFrame({"s": ["1"]}), {"nope": "int"})


class TestStringOps:
    def test_trim(self, df):
        out = apply_string_op(df, column="name", op="trim")
        assert out["name"][0] == "Alice"

    def test_lower_upper(self):
        df = pl.DataFrame({"s": ["ABC", "def"]})
        assert apply_string_op(df, column="s", op="lower")["s"].to_list() == ["abc", "def"]
        assert apply_string_op(df, column="s", op="upper")["s"].to_list() == ["ABC", "DEF"]

    def test_replace(self):
        df = pl.DataFrame({"s": ["a-b-c", "a-b"]})
        out = apply_string_op(df, column="s", op="replace", params={"old": "-", "new": "+"})
        assert out["s"].to_list() == ["a+b+c", "a+b"]

    def test_regex(self):
        df = pl.DataFrame({"s": ["abc123", "xyz456"]})
        out = apply_string_op(
            df, column="s", op="regex", params={"pattern": r"\d+", "replacement": "#"}
        )
        assert out["s"].to_list() == ["abc#", "xyz#"]

    def test_regex_requires_pattern(self):
        with pytest.raises(TransformError):
            apply_string_op(pl.DataFrame({"s": ["a"]}), column="s", op="regex")

    def test_non_string_column_rejected(self, df):
        with pytest.raises(TransformError):
            apply_string_op(df, column="age", op="lower")

    def test_unknown_op(self, df):
        with pytest.raises(TransformError):
            apply_string_op(df, column="name", op="eval")


class TestFilter:
    def test_comparison_ops(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4], "s": ["x", "y", "z", "w"]})
        out = apply_filter(df, [{"column": "a", "op": "gt", "value": 2}])
        assert out["a"].to_list() == [3, 4]
        out = apply_filter(df, [{"column": "a", "op": "lte", "value": 2}])
        assert out["a"].to_list() == [1, 2]
        out = apply_filter(df, [{"column": "a", "op": "neq", "value": 1}])
        assert out.height == 3

    def test_contains_in_is_null(self):
        df = pl.DataFrame({"s": ["apple", "banana", None]})
        out = apply_filter(df, [{"column": "s", "op": "contains", "value": "an"}])
        assert out["s"].to_list() == ["banana"]
        out = apply_filter(df, [{"column": "s", "op": "in", "value": ["apple", "banana"]}])
        assert out.height == 2
        out = apply_filter(df, [{"column": "s", "op": "is_null"}])
        assert out.height == 1

    def test_and_or_logic(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4]})
        out = apply_filter(
            df,
            [{"column": "a", "op": "gt", "value": 1}, {"column": "a", "op": "lt", "value": 4}],
            logic="and",
        )
        assert out["a"].to_list() == [2, 3]
        out = apply_filter(
            df,
            [{"column": "a", "op": "eq", "value": 1}, {"column": "a", "op": "eq", "value": 4}],
            logic="or",
        )
        assert out["a"].to_list() == [1, 4]

    def test_unknown_op_rejected(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TransformError):
            apply_filter(df, [{"column": "a", "op": "regex_eval", "value": "x"}])

    def test_unknown_column_rejected(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TransformError):
            apply_filter(df, [{"column": "b", "op": "eq", "value": 1}])

    def test_in_requires_list(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TransformError):
            apply_filter(df, [{"column": "a", "op": "in", "value": 1}])


class TestTransform:
    def test_math_column_and_value(self):
        df = pl.DataFrame({"a": [1, 2], "b": [10, 20]})
        out = add_column(
            df,
            name="total",
            expression={
                "type": "math",
                "op": "add",
                "left": {"type": "column", "column": "a"},
                "right": {"type": "column", "column": "b"},
            },
        )
        assert out["total"].to_list() == [11, 22]

    def test_div_by_zero_gives_null(self):
        df = pl.DataFrame({"a": [10, 20], "b": [2, 0]})
        out = add_column(
            df,
            name="ratio",
            expression={
                "type": "math",
                "op": "div",
                "left": {"type": "column", "column": "a"},
                "right": {"type": "column", "column": "b"},
            },
        )
        assert out["ratio"][0] == 5.0
        assert out["ratio"][1] is None

    def test_date_part(self):
        df = pl.DataFrame({"ts": [datetime(2024, 3, 15, 10, 30, 0)]})
        out = add_column(df, name="month", expression={
            "type": "date_part", "part": "month", "column": "ts"})
        assert out["month"][0] == 3

    def test_duplicate_name_rejected(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TransformError):
            add_column(df, name="a", expression={"type": "value", "value": 1})

    def test_overwrite_allowed(self):
        df = pl.DataFrame({"a": [1]})
        out = add_column(df, name="a", expression={"type": "value", "value": 9}, overwrite=True)
        assert out["a"].to_list() == [9]

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TransformError):
            add_column(df, name="x", expression={"type": "column", "column": "nope"})


class TestAggregate:
    def test_group_aggregations(self):
        df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 3, 10]})
        out = aggregate(
            df,
            group_by=["g"],
            aggregations=[
                {"column": "v", "func": "sum"},
                {"column": "v", "func": "mean"},
                {"func": "count"},
            ],
        )
        rows = {r["g"]: r for r in out.to_dicts()}
        assert rows["a"]["v_sum"] == 4
        assert rows["a"]["v_mean"] == 2.0
        assert rows["a"]["count"] == 2
        assert rows["b"]["v_sum"] == 10

    def test_numeric_only(self):
        df = pl.DataFrame({"g": ["a"], "s": ["x"]})
        with pytest.raises(TransformError):
            aggregate(df, group_by=["g"], aggregations=[{"column": "s", "func": "sum"}])

    def test_unknown_func(self):
        df = pl.DataFrame({"g": ["a"], "v": [1]})
        with pytest.raises(TransformError):
            aggregate(df, group_by=["g"], aggregations=[{"column": "v", "func": "median2"}])

    def test_missing_group_by(self):
        with pytest.raises(TransformError):
            aggregate(pl.DataFrame({"v": [1]}), group_by=[], aggregations=[{"func": "count"}])


class TestPivot:
    def test_pivot_sum(self):
        df = pl.DataFrame(
            {"month": ["1", "1", "2"], "city": ["BJ", "SH", "BJ"], "sales": [10, 20, 30]}
        )
        out = pivot(df, index=["month"], columns="city", values="sales", aggregation="sum")
        assert set(out.columns) == {"month", "BJ", "SH"}

    def test_too_many_columns(self):
        df = pl.DataFrame(
            {"k": ["a", "b"] * 100, "c": [str(i) for i in range(200)], "x": [1] * 200}
        )
        with pytest.raises(TransformError):
            pivot(df, index=["k"], columns="c", values="x")

    def test_invalid_aggregation(self):
        df = pl.DataFrame({"k": ["a"], "c": ["x"], "v": [1]})
        with pytest.raises(TransformError):
            pivot(df, index=["k"], columns="c", values="v", aggregation="nope")


class TestMelt:
    def test_basic(self):
        df = pl.DataFrame({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
        out = melt(df, id_vars=["id"])
        assert out.height == 4
        assert set(out.columns) == {"id", "variable", "value"}

    def test_value_vars(self):
        df = pl.DataFrame({"id": [1], "a": [10], "b": [30]})
        out = melt(df, id_vars=["id"], value_vars=["a"])
        assert out.height == 1

    def test_overlap_rejected(self):
        df = pl.DataFrame({"id": [1], "a": [10]})
        with pytest.raises(TransformError):
            melt(df, id_vars=["id"], value_vars=["id"])

    def test_requires_spec(self):
        with pytest.raises(TransformError):
            melt(pl.DataFrame({"a": [1]}))
