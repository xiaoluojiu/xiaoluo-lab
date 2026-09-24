"""数据中心模块后端验证：筛选类型转换、多文件合并、聚合等。

运行：PYTHONPATH=<backend> python -m pytest backend/tests/test_data_center.py -q
（使用隔离 venv 中的 polars 运行）
"""

from __future__ import annotations

import polars as pl
import pytest

from app.analysis import (
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    DistributionAnalyzer,
    EdaOutlierAnalyzer,
    MissingChecker,
    VisualizationBuilder,
    _norm_ppf,
    analyze_schema,
    build_report,
    profile,
)
from app.core.exceptions import ValidationException
from app.data_engine.exceptions import TransformError
from app.data_engine.merge_multi import analyze_multi_merge, execute_multi_merge
from app.data_engine.operations import aggregate, apply_filter, melt, pivot
from app.data_engine.param_validation import validate_selector_params


def _df_numeric() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "score": [10, 20, 30, 40, 50],
            "name": ["a", "b", "c", "d", "e"],
            "active": [True, False, True, False, True],
        }
    )


# ---------------- 筛选（Task #16）----------------


def test_filter_numeric_with_string_value_no_400():
    """根因修复：前端传入字符串 "30"，数值列应能正确比较，不再 400。"""
    df = _df_numeric()
    out = apply_filter(df, [{"column": "score", "op": "gte", "value": "30"}], logic="and")
    assert out.height == 3
    assert set(out["score"].to_list()) == {30, 40, 50}


def test_filter_string_column_with_numeric_input_coerces_to_string():
    """字符串列传入数字形输入时，应自动转为字符串再比较，且不报错/不 400。"""
    df = _df_numeric()
    # value=2 会被转为字符串 "2"，name 列(a/b/c/d/e)无匹配 → 0 行，且不抛异常
    out = apply_filter(df, [{"column": "name", "op": "eq", "value": 2}], logic="and")
    assert out.height == 0
    # 反向验证：字符串列传入字符串值仍正常匹配
    out2 = apply_filter(df, [{"column": "name", "op": "eq", "value": "c"}], logic="and")
    assert out2.height == 1
    assert out2["name"][0] == "c"


def test_filter_contains_on_numeric_column_casts_to_string():
    df = _df_numeric()
    out = apply_filter(df, [{"column": "score", "op": "contains", "value": "4"}], logic="and")
    assert out.height == 1
    assert out["score"][0] == 40


def test_filter_bool_value_coercion():
    df = _df_numeric()
    out = apply_filter(df, [{"column": "active", "op": "eq", "value": "true"}], logic="and")
    assert out.height == 3


def test_filter_placeholder_empty_column_is_skipped():
    """占位条件（column 为空）不应导致 400。"""
    df = _df_numeric()
    out = apply_filter(
        df,
        [{"column": "", "op": "gte", "value": 0}, {"column": "score", "op": "gt", "value": 25}],
        logic="and",
    )
    assert out.height == 3


def test_filter_all_placeholder_returns_all_rows():
    df = _df_numeric()
    out = apply_filter(df, [{"column": "", "op": "gte", "value": 0}], logic="and")
    assert out.height == df.height


def test_filter_in_accepts_comma_separated_string():
    """重设计后的筛选：in 支持文本输入的逗号分隔串（按列类型逐项转换）。"""
    df = _df_numeric()
    out = apply_filter(df, [{"column": "name", "op": "in", "value": "a,c"}], logic="and")
    assert out.height == 2
    assert set(out["name"].to_list()) == {"a", "c"}
    # 数值列：字符串项应转为数值后比较
    out2 = apply_filter(df, [{"column": "score", "op": "in", "value": "10,30"}], logic="and")
    assert out2.height == 2
    assert set(out2["score"].to_list()) == {10, 30}


def test_filter_invalid_value_raises_clear_error():
    df = _df_numeric()
    try:
        apply_filter(df, [{"column": "score", "op": "gt", "value": "abc"}], logic="and")
    except TransformError as exc:
        assert "score" in exc.message
    else:
        raise AssertionError("expected TransformError for non-numeric value on numeric column")


# ---------------- 聚合（Task #17）----------------


def test_aggregate_sum_and_count():
    df = pl.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
    out = aggregate(df, group_by=["g"], aggregations=[{"column": "v", "func": "sum"}])
    assert out.height == 2
    assert "v_sum" in out.columns
    out2 = aggregate(df, group_by=["g"], aggregations=[{"column": None, "func": "count"}])
    assert "count" in out2.columns
    assert out2.filter(pl.col("g") == "x")["count"][0] == 2


def test_aggregate_std_included():
    df = pl.DataFrame({"g": ["x", "y", "x", "y"], "v": [1.0, 2.0, 3.0, 4.0]})
    out = aggregate(df, group_by=["g"], aggregations=[{"column": "v", "func": "std"}])
    assert "v_std" in out.columns


# ---------------- 多文件合并（Task #15）----------------


def test_multi_merge_analyze_common_and_unique():
    a = pl.DataFrame({"k": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"k": [3, 4], "y": [30, 40]})
    info = analyze_multi_merge([a, b])
    assert info["common_columns"] == ["k"]
    assert set(info["unique_columns"]) == {"x", "y"}
    assert info["files"][0]["only_in_this"] == ["x"]
    assert info["files"][1]["only_in_this"] == ["y"]


def test_multi_merge_union_by_selected_columns():
    a = pl.DataFrame({"k": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"k": [3, 4], "y": [30, 40]})
    merged = execute_multi_merge([a, b], selected_columns=["k", "x", "y"])
    assert merged.height == 4
    assert set(merged.columns) == {"k", "x", "y"}
    # 缺失字段补 null
    assert merged.filter(pl.col("k") == 3)["x"][0] is None
    assert merged.filter(pl.col("k") == 3)["y"][0] == 30


def test_multi_merge_add_source():
    a = pl.DataFrame({"k": [1], "x": [10]})
    b = pl.DataFrame({"k": [2], "x": [20]})
    merged = execute_multi_merge(
        [a, b], selected_columns=["k", "x"], add_source=True, source_column="src",
        source_labels=["A", "B"],
    )
    assert merged["src"].to_list() == ["A", "B"]


def test_multi_merge_dtype_normalization():
    """同名字段跨文件类型不一致（int vs float）应归一为 Float64 而非报错。"""
    a = pl.DataFrame({"k": [1], "v": [1]})
    b = pl.DataFrame({"k": [2], "v": [2.5]})
    merged = execute_multi_merge([a, b], selected_columns=["k", "v"])
    assert merged.schema["v"] == pl.Float64
    assert merged.height == 2


def test_multi_merge_selected_column_not_in_any_raises():
    a = pl.DataFrame({"k": [1]})
    b = pl.DataFrame({"m": [2]})
    try:
        execute_multi_merge([a, b], selected_columns=["nope"])
    except TransformError:
        pass
    else:
        raise AssertionError("expected TransformError for non-existent column")


# ---------------- 分析模块（Task #19）：确认为真实实现而非伪代码 ----------------


def _df_analysis() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [22, 25, 27, 30, 33, 35, 38, 41, 44, 47],
            "score": [10, 12, 11, 15, 18, 17, 20, 22, 21, 25],
            "city": ["BJ", "SH", "BJ", "SH", "BJ", "SH", "BJ", "SH", "BJ", "SH"],
            "grp": ["A", "B"] * 5,
        }
    )


def test_norm_ppf_matches_standard_normal():
    """自包含逆正态 CDF（Acklam）对照标准值。"""
    assert abs(_norm_ppf(0.5)) < 1e-9
    assert abs(_norm_ppf(0.975) - 1.959963985) < 1e-6
    assert abs(_norm_ppf(0.9) - 1.281551566) < 1e-6
    assert abs(_norm_ppf(0.25) + 0.674489750) < 1e-6


def test_descriptive_analyzer_real_stats():
    """DescriptiveAnalyzer 产出真实统计量（数值/布尔/类别三类分支）。"""
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None],
            "s": ["a", "a", "b", "b", "c"],
            "b": [True, False, True, False, True],
        }
    )
    out = DescriptiveAnalyzer().analyze(df)
    assert out["row_count"] == 5
    by = {c["column"]: c for c in out["columns"]}
    assert by["v"]["mean"] == 2.5
    assert by["v"]["missing_count"] == 1
    assert set(by["v"]["quantiles"]) == {"0.25", "0.5", "0.75"}
    assert by["b"]["true_count"] == 3
    assert by["b"]["false_count"] == 2
    assert by["s"]["top"] in ("a", "b")
    assert by["s"]["freq"] == 2


def test_correlation_analyzer_real_matrix():
    """CorrelationAnalyzer 产出对称矩阵，完全线性相关为 1.0。"""
    df = pl.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 4, 6, 8, 10]})
    out = CorrelationAnalyzer().analyze(df, method="pearson")
    assert out["method"] == "pearson"
    assert abs(out["matrix"]["x"]["y"] - 1.0) < 1e-9
    assert out["matrix"]["x"]["x"] == 1.0
    auto = CorrelationAnalyzer().analyze(df, method="auto")
    assert auto["method"] == "auto"
    assert "methods_used" in auto


def test_distribution_analyzer_counts_sum_to_total():
    """数值分布各桶计数之和 == 非空样本数；类别分布带占比。"""
    df = _df_analysis()
    num = DistributionAnalyzer().analyze(df, column="age", bins=5)
    assert num["type"] == "numeric"
    assert sum(b["count"] for b in num["bins"]) == df.height
    cat = DistributionAnalyzer().analyze(df, column="city")
    assert cat["type"] == "categorical"
    assert sum(i["count"] for i in cat["values"]) == df.height
    assert all("ratio" in i for i in cat["values"])


def test_outlier_analyzer_real_detection():
    """EdaOutlierAnalyzer 真实检出离群点，并对常数列给出明确跳过原因。"""
    df = pl.DataFrame({"flat": [5.0] * 10, "spike": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 99]})
    iqr = EdaOutlierAnalyzer().analyze(df, method="iqr")
    iqr_by = {c["column"]: c for c in iqr["columns"]}
    # 常数列（方差为 0）在**两种**方法下都跳过，并给出明确原因：
    # 前端因此不必区分「有结果 / 无结果」两种行结构，用户也知道为什么没有数值。
    assert iqr_by["flat"]["status"] == "skipped"
    assert iqr_by["flat"]["outlier_count"] == 0
    assert iqr_by["flat"]["reason"]
    # IQR=0 时对稀疏离群仍能检出（spike 的 99 被标为离群）
    assert iqr_by["spike"]["status"] == "ok"
    assert iqr_by["spike"]["outlier_count"] == 1
    assert 99 in iqr_by["spike"]["sample_outliers"]

    z = EdaOutlierAnalyzer().analyze(df, method="zscore")
    z_by = {c["column"]: c for c in z["columns"]}
    # 跳过原因按平台口径为中文（前端直接展示），不再比对英文字符串
    assert z_by["flat"]["status"] == "skipped"
    assert "常数列" in z_by["flat"]["reason"]


def test_quality_build_report_real_issues():
    """Quality build_report 真实汇总缺失等问题与严重级别。"""
    df = pl.DataFrame({"a": [1, 1, 1, None, None], "b": [1, 2, 3, 4, 5]})
    rep = build_report(df).to_dict()
    assert rep["statistics"]["row_count"] == 5
    assert "missing" in {i["check"] for i in rep["issues"]}
    # a 列缺失率 40% → high
    assert rep["severity"]["high"] >= 1


def test_quality_missing_checker_rejects_unknown_columns():
    df = pl.DataFrame({"b": [1, 2, 3]})
    # 业务错误统一走 ValidationException（422），不是裸 ValueError：
    # 未知列名是「请求参数不合法」而非程序错误，必须能作为业务错误返回给前端。
    with pytest.raises(ValidationException) as excinfo:
        MissingChecker(columns=["nope"]).check(df)
    assert "nope" in str(excinfo.value)


def test_profiling_real_schema_and_profile():
    """Profiling：analyze_schema / profile 产出真实画像。"""
    df = pl.DataFrame({"v": [1, 2, 3], "s": ["a", "b", None]})
    schema = analyze_schema(df)
    assert schema["row_count"] == 3 and schema["column_count"] == 2
    cols = {c["column"]: c for c in schema["columns"]}
    assert cols["s"]["nullable"] is True
    assert cols["s"]["null_count"] == 1
    prof = profile(df)
    assert prof["missing_cells"] == 1
    assert prof["complete_row_count"] == 2


def test_visualization_builder_all_chart_types_real():
    """VisualizationBuilder 全部 9 类图表均产出真实结构（Task #18 新增类型）。"""
    df = _df_analysis()
    vb = VisualizationBuilder()

    h = vb.analyze(df, chart="histogram", column="age", bins=5)
    assert h["chart"] == "histogram" and len(h["x"]) == 5 and sum(h["y"]) == df.height

    b = vb.analyze(df, chart="bar", column="city")
    assert b["chart"] == "bar" and set(b["x"]) == {"BJ", "SH"} and sum(b["y"]) == df.height

    ln = vb.analyze(df, chart="line", x="age", y="score")
    assert ln["chart"] == "line" and len(ln["x"]) == df.height

    sc = vb.analyze(df, chart="scatter", x="age", y="score")
    assert sc["chart"] == "scatter" and len(sc["x"]) == df.height

    bx = vb.analyze(df, chart="boxplot", column="score")
    assert bx["chart"] == "boxplot"
    assert bx["boxes"][0]["q1"] <= bx["boxes"][0]["median"] <= bx["boxes"][0]["q3"]

    hm = vb.analyze(df, chart="heatmap", columns=["age", "score"])
    assert hm["chart"] == "heatmap" and sorted(hm["columns"]) == ["age", "score"]

    qq = vb.analyze(df, chart="qq", column="age")
    assert qq["chart"] == "qq" and len(qq["x"]) == len(qq["y"]) == df.height
    assert qq["y"] == sorted(qq["y"])  # 样本按升序排列

    gb = vb.analyze(df, chart="grouped_bar", column="city", y="score", group_by="grp")
    assert gb["chart"] == "grouped_bar"
    assert set(gb["x"]) == {"BJ", "SH"} and len(gb["groups"]) == 2

    ar = vb.analyze(df, chart="area", column="age", bins=5)
    assert ar["chart"] == "area" and ar["y"][-1] == 1.0 and ar["total"] == df.height

    # 不支持的图表类型属于「请求参数不合法」⇒ ValidationException（422），
    # 不是数据处理异常（TransformError）。用例早先按 TransformError 编写，
    # 与平台统一的业务错误契约不一致，这里对齐到契约。
    with pytest.raises(ValidationException):
        vb.analyze(df, chart="not_a_chart")


# ---------------- 透视 / 逆透视（Task #24：预览 400）----------------


def test_pivot_missing_params_raises_validation_not_400():
    """透视必填维度缺失属于用户输入不完整 → 422 ValidationException（不是 400）。"""
    for params, keyword in [
        ({"index": [], "columns": "", "values": ""}, "行索引"),
        ({"index": ["city"], "columns": "", "values": "v"}, "列维度"),
        ({"index": ["city"], "columns": "q", "values": ""}, "值"),
    ]:
        try:
            validate_selector_params("pivot", params)
        except ValidationException as exc:
            assert keyword in exc.message, f"expected {keyword} in {exc.message}"
        else:
            raise AssertionError(f"expected ValidationException mentioning {keyword}")


def test_melt_missing_params_raises_validation_not_400():
    try:
        validate_selector_params(
            "melt",
            {
                "id_vars": [],
                "value_vars": [],
                "variable_name": "variable",
                "value_name": "value",
            },
        )
    except ValidationException as exc:
        assert "逆透视" in exc.message
    else:
        raise AssertionError("expected ValidationException for melt without any vars")


def test_pivot_melt_valid_params_pass_validation():
    """参数完整时不应拦截。"""
    validate_selector_params(
        "pivot",
        {"index": ["city"], "columns": "q", "values": "v", "aggregation": "sum"},
    )
    validate_selector_params(
        "melt",
        {
            "id_vars": ["city"],
            "value_vars": ["v"],
            "variable_name": "variable",
            "value_name": "value",
        },
    )
    # 其他操作类型不受影响（不做必填维度拦截）
    validate_selector_params("filter", {})


def test_pivot_and_melt_real_transform():
    """透视 / 逆透视链路真实可用（round-trip）。"""
    df = pl.DataFrame(
        {
            "city": ["BJ", "BJ", "SH", "SH"],
            "quarter": ["Q1", "Q2", "Q1", "Q2"],
            "sales": [10, 20, 30, 40],
        }
    )
    wide = pivot(df, index=["city"], columns="quarter", values="sales", aggregation="sum")
    assert wide.height == 2
    assert set(wide.columns) == {"city", "Q1", "Q2"}
    assert wide.filter(pl.col("city") == "BJ")["Q1"][0] == 10
    assert wide.filter(pl.col("city") == "SH")["Q2"][0] == 40

    long = melt(wide, id_vars=["city"], value_vars=["Q1", "Q2"])
    assert long.height == 4
    assert set(long.columns) == {"city", "variable", "value"}
    assert set(long["variable"].to_list()) == {"Q1", "Q2"}


def test_pivot_rejects_non_numeric_values_for_sum():
    df = pl.DataFrame({"city": ["a", "b"], "q": ["Q1", "Q2"], "txt": ["x", "y"]})
    try:
        pivot(df, index=["city"], columns="q", values="txt", aggregation="sum")
    except TransformError:
        pass
    else:
        raise AssertionError("expected TransformError for sum on non-numeric values")


if __name__ == "__main__":
    import traceback

    funcs = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print("FAIL", fn.__name__, "->", repr(exc))
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
