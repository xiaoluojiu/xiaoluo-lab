"""Prompt 064-069：EDA（描述统计 / 分布 / 相关性 / 异常 / 可视化）测试。"""

import json

import polars as pl
import pytest
from app.analysis import (
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    DistributionAnalyzer,
    EdaOutlierAnalyzer,
    VisualizationBuilder,
)
from app.core.exceptions import ValidationException


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [22, 25, 28, 30, 35, 40],
            "salary": [3000.0, 3500.0, 4000.0, 5000.0, 8000.0, 12000.0],
            "dept": ["A", "A", "B", "B", "C", "C"],
            "active": [True, False, True, True, False, True],
        }
    )


class TestDescriptive:
    def test_numeric_and_categorical(self, df):
        result = DescriptiveAnalyzer().analyze(df)
        by_name = {c["column"]: c for c in result["columns"]}
        assert by_name["age"]["mean"] == pytest.approx(30.0)
        assert by_name["age"]["min"] == 22
        assert by_name["dept"]["unique"] == 3
        assert by_name["dept"]["freq"] == 2
        assert by_name["active"]["true_count"] == 4

    def test_selected_columns(self, df):
        result = DescriptiveAnalyzer().analyze(df, columns=["age"])
        assert len(result["columns"]) == 1


class TestDistribution:
    def test_numeric_bins(self, df):
        result = DistributionAnalyzer().numeric_distribution(df, "age", bins=4)
        assert len(result["bins"]) == 4
        assert sum(b["count"] for b in result["bins"]) == 6

    def test_numeric_constant(self):
        df = pl.DataFrame({"v": [5, 5, 5]})
        result = DistributionAnalyzer().numeric_distribution(df, "v")
        assert result["bins"][0]["count"] == 3

    def test_categorical(self, df):
        result = DistributionAnalyzer().categorical_distribution(df, "dept")
        assert result["unique_count"] == 3
        assert sum(i["count"] for i in result["values"]) == 6
        assert all("ratio" in i for i in result["values"])

    def test_dispatch_by_type(self, df):
        assert DistributionAnalyzer().analyze(df, column="dept")["type"] == "categorical"
        assert DistributionAnalyzer().analyze(df, column="age")["type"] == "numeric"


class TestCorrelation:
    def test_pearson_known_value(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
        result = CorrelationAnalyzer().analyze(df, method="pearson")
        assert result["matrix"]["x"]["y"] == pytest.approx(1.0)

    def test_auto_methods(self, df):
        result = CorrelationAnalyzer().analyze(df, method="auto")
        assert result["method"] == "auto"
        assert result["methods_used"]["age"] == "spearman"  # 整数列
        assert result["methods_used"]["salary"] == "pearson"  # 浮点列
        # age 与 salary 强相关
        assert result["matrix"]["age"]["salary"] > 0.9

    def test_requires_numeric(self):
        df = pl.DataFrame({"s": ["a", "b"]})
        with pytest.raises(ValidationException):
            CorrelationAnalyzer().analyze(df)

    def test_invalid_method(self, df):
        with pytest.raises(ValidationException):
            CorrelationAnalyzer().analyze(df, method="nope")


class TestEdaOutlier:
    def test_outlier_profile(self):
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
        result = EdaOutlierAnalyzer().analyze(df, method="iqr")
        entry = result["columns"][0]
        assert entry["outlier_count"] == 1
        assert entry["sample_outliers"] == [100.0]
        assert entry["bounds"]["upper"] < 100

    def test_constant_column_skipped(self):
        df = pl.DataFrame({"v": [5, 5, 5]})
        result = EdaOutlierAnalyzer().analyze(df, method="zscore")
        assert result["columns"][0]["status"] == "skipped"


class TestVisualization:
    def test_histogram(self, df):
        data = VisualizationBuilder().analyze(df, chart="histogram", column="age", bins=3)
        assert data["chart"] == "histogram"
        assert len(data["x"]) == 3
        assert sum(data["y"]) == 6

    def test_bar(self, df):
        data = VisualizationBuilder().analyze(df, chart="bar", column="dept")
        assert data["chart"] == "bar"
        assert len(data["x"]) == 3

    def test_line(self, df):
        data = VisualizationBuilder().analyze(df, chart="line", x="age", y="salary")
        assert data["chart"] == "line"
        assert data["y"] == sorted(data["y"])

    def test_scatter_downsample(self):
        n = 5000
        df = pl.DataFrame({"x": list(range(n)), "y": [i * 2.0 for i in range(n)]})
        data = VisualizationBuilder().analyze(df, chart="scatter", x="x", y="y", sample_limit=100)
        assert data["sampled"] is True
        assert len(data["x"]) == 100

    def test_boxplot(self, df):
        data = VisualizationBuilder().analyze(df, chart="boxplot", column="salary")
        box = data["boxes"][0]
        assert box["q1"] <= box["median"] <= box["q3"]

    def test_boxplot_grouped(self, df):
        data = VisualizationBuilder().analyze(df, chart="boxplot", column="age", group_by="dept")
        assert len(data["boxes"]) == 3

    def test_heatmap(self, df):
        data = VisualizationBuilder().analyze(df, chart="heatmap")
        assert data["chart"] == "heatmap"
        assert set(data["columns"]) == {"age", "salary"}
        assert data["matrix"]["age"]["salary"] is not None

    def test_unknown_chart(self, df):
        with pytest.raises(ValidationException):
            VisualizationBuilder().analyze(df, chart="3d-surface")

    def test_all_json_serializable(self, df):
        builder = VisualizationBuilder()
        json.dumps(builder.analyze(df, chart="histogram", column="age"))
        json.dumps(builder.analyze(df, chart="bar", column="dept"))
        json.dumps(builder.analyze(df, chart="line", x="age", y="salary"))
        json.dumps(builder.analyze(df, chart="boxplot", column="salary"))
        json.dumps(builder.analyze(df, chart="heatmap"))
