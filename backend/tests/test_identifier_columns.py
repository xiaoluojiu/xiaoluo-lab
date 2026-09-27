"""标识符列（customer_id 之类）不该混进相关性矩阵。

实测：1220 行的信贷样本上跑「看看各数值列之间的相关性」，10 对里 4 对是
``customer_id`` 的（customer_id × age = 0.0419、customer_id × credit_amount
= 0.0266 …）。它是行标识，取值连续且唯一，算法照算不误，但对业务毫无意义，
还把真正有信息量的那几对挤到了后面。
"""

from __future__ import annotations

import polars as pl
import pytest

from app.analysis import CorrelationAnalyzer, VisualizationBuilder, is_identifier_like
from app.core.exceptions import ValidationException


def _frame() -> pl.DataFrame:
    rows = 60
    return pl.DataFrame(
        {
            "customer_id": list(range(1000, 1000 + rows)),
            "credit_amount": [1000 + (i * 37) % 5000 for i in range(rows)],
            "installment_commitment": [50 + (i * 37) % 5000 // 12 for i in range(rows)],
            "age": [20 + (i * 7) % 50 for i in range(rows)],
        }
    )


def test_identifier_columns_are_dropped_when_columns_are_not_named():
    result = CorrelationAnalyzer().analyze(_frame(), method="pearson")

    assert "customer_id" not in result["columns"]
    assert set(result["columns"]) == {"credit_amount", "installment_commitment", "age"}


def test_named_identifier_column_is_still_honored():
    """用户点名了就必须照办 —— 他要算 id 的相关性就给他算。"""
    result = CorrelationAnalyzer().analyze(
        _frame(), method="pearson", columns=["customer_id", "age"]
    )

    assert result["columns"] == ["customer_id", "age"]


def test_identifier_filter_never_breaks_a_runnable_analysis():
    """剔除后不足 2 列时必须放弃剔除：宁可有噪声，也不能让能跑的分析失败。"""
    df = pl.DataFrame({"customer_id": [1, 2, 3, 4], "user_id": [4, 3, 2, 1]})

    result = CorrelationAnalyzer().analyze(df, method="pearson")

    assert len(result["columns"]) == 2


def test_too_few_numeric_columns_still_raises():
    df = pl.DataFrame({"customer_id": [1, 2, 3], "name": ["a", "b", "c"]})

    with pytest.raises(ValidationException):
        CorrelationAnalyzer().analyze(df)


@pytest.mark.parametrize(
    "column,expected",
    [
        ("id", True),
        ("customer_id", True),
        ("user-id", True),
        ("编号", True),
        ("序号", True),
        ("index", True),
        ("credit_amount", False),
        ("duration", False),
        ("valid", False),
        ("", False),
    ],
)
def test_is_identifier_like(column: str, expected: bool):
    assert is_identifier_like(column) is expected


# ---------------------------------------------------------------- 图表降级


def test_histogram_on_text_column_degrades_to_bar():
    """文本列画不出直方图。

    旧行为是一路抛到 Polars 的 ``could not convert string to float: 'bad'`` ——
    用户既不知道哪一步错了，也不知道该换成什么图。这里应当直接给出类别频次图，
    并如实标注发生了降级。
    """
    df = pl.DataFrame({"class": ["good", "bad", "good", "good", "bad"]})

    result = VisualizationBuilder().analyze(df, chart="histogram", column="class")

    assert result["chart"] == "bar"
    assert result["degraded_from"] == "histogram"
    assert result["y"] == [3, 2]


def test_histogram_on_numeric_column_is_unchanged():
    df = pl.DataFrame({"age": [20, 21, 22, 23, 24, 25, 60]})

    result = VisualizationBuilder().analyze(df, chart="histogram", column="age")

    assert result["chart"] == "histogram"
    assert "degraded_from" not in result
