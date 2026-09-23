"""目标列推断的回归测试。

真实事故（2026-09-23）：数据集名「航空公司出发延误预测（回归）」，
列里没有 target/label/y/class，规划器又看不到列名 ⇒ 链路死锁在
「`ml.detect_task` 说 needs_target，但没人能替它选」。

生产环境回传的候选面（原样抄下来当测试基线）：

    回归候选：['DepDelay', 'CRSDepTime', 'CRSArrTime', 'Distance']
    分类候选：['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier']

本文件钉住两件事：
1. `target_candidates` 仍然复现这份原始候选面（不做任何隐藏排除）；
2. `recommend_target` 能在其上**唯一地**推出 `DepDelay`，并给出可读理由；
   拿不到证据时**不猜**，返回 None + 候选。
"""

from __future__ import annotations

import polars as pl
import pytest

from app.ml_engine.target_inference import (
    feature_like_reason,
    goal_concepts,
    infer_task_from_column,
    name_tokens,
    recommend_target,
    target_candidates,
)


def _airlines(n: int = 2000) -> pl.DataFrame:
    """复刻 dataset 9 读进来后的形状（10 列，全部 Float64 / String）。"""
    return pl.DataFrame(
        {
            "DepDelay": [float(i % 300 - 20) for i in range(n)],
            "Month": [float(i % 12 + 1) for i in range(n)],
            "DayofMonth": [float(i % 31 + 1) for i in range(n)],
            "DayOfWeek": [float(i % 7 + 1) for i in range(n)],
            # 真实数据里 CRS*Time 是 HHMM 编码，唯一值远多于 50 ⇒ 会被列进回归候选
            "CRSDepTime": [float((i % 288) * 5) for i in range(n)],
            "CRSArrTime": [float((i % 240) * 5) for i in range(n)],
            "UniqueCarrier": [f"C{i % 30}" for i in range(n)],
            "Origin": [f"A{i % 400}" for i in range(n)],
            "Dest": [f"B{i % 369}" for i in range(n)],
            # 真实 Distance 只有几百个取值（0.2% 唯一度），不能做成逐行唯一，
            # 否则会被「每行唯一 ⇒ 疑似主键」规则挡掉。
            "Distance": [float((i % 500) * 1.37) for i in range(n)],
        }
    )


class TestNameTokens:
    @pytest.mark.parametrize(
        "column,expected",
        [
            # `DayofMonth` 里 "of" 是小写，驼峰切分切不出它 —— 但 `month` 已经足够
            # 把它认成日历分量，所以不影响排除判定。
            ("DayofMonth", ["dayof", "month"]),
            ("DayOfWeek", ["day", "of", "week"]),
            ("CRSDepTime", ["crs", "dep", "time"]),
            ("crs_dep_time", ["crs", "dep", "time"]),
            ("DepDelay", ["dep", "delay"]),
            ("MonthlyCharges", ["monthly", "charges"]),
            ("  total delay  ", ["total", "delay"]),
        ],
    )
    def test_split(self, column, expected):
        assert name_tokens(column) == expected

    def test_cjk_name_is_kept_whole(self):
        assert name_tokens("是否流失") == ["是否流失"]

    def test_empty(self):
        assert name_tokens("") == []


class TestFeatureLikeReason:
    @pytest.mark.parametrize(
        "column", ["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime"]
    )
    def test_calendar_and_time_are_not_targets(self, column):
        assert feature_like_reason(column) is not None

    @pytest.mark.parametrize(
        "column", ["DepDelay", "Distance", "UniqueCarrier", "Churn", "MonthlyCharges"]
    )
    def test_real_candidates_are_not_excluded(self, column):
        assert feature_like_reason(column) is None

    def test_word_boundary_is_respected(self):
        """`Holiday` 含子串 day 但不是日历分量 —— 分词边界必须挡住误伤。"""
        assert feature_like_reason("Holiday") is None
        assert feature_like_reason("Payday") is None


class TestGoalConcepts:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("预测出发延误", {"delay"}),
            ("predict flight delay", {"delay"}),
            ("预测客户是否会流失", {"churn"}),
            ("航空公司出发延误预测（回归）", {"delay"}),
            ("预测价格", {"price"}),
            ("", set()),
        ],
    )
    def test_detect(self, text, expected):
        assert goal_concepts(text) == expected

    def test_pure_meta_request_has_no_concept(self):
        """步骤意图那种自身描述（不含预测对象）不应该匹配出概念。"""
        assert goal_concepts("根据目标列类型与数据分布判断 ML 任务类型，并给出理由") == set()

    def test_short_alias_needs_word_boundary(self):
        """`age` 不能命中 `average` —— 否则「预测年龄」会选中平均停留时长。"""
        assert goal_concepts("average_tenure") == set()


class TestInferTaskFromColumn:
    @pytest.mark.parametrize(
        "dtype,n_unique,expected",
        [
            (pl.String, 5, "classification"),
            (pl.Boolean, 2, "classification"),
            (pl.Int64, 3, "classification"),
            (pl.Int64, 1000, "regression"),
            (pl.Float64, 300, "regression"),
            (pl.Float64, 12, "regression"),
        ],
    )
    def test_basic(self, dtype, n_unique, expected):
        assert infer_task_from_column(dtype, n_unique) == expected


class TestRawCandidateSurface:
    def test_reproduces_production_candidate_lists(self):
        df = _airlines()
        options = target_candidates(df)
        assert options["regression"] == ["DepDelay", "CRSDepTime", "CRSArrTime", "Distance"]
        assert options["classification"] == [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "UniqueCarrier",
        ]


class TestRecommendTargetAirlineCase:
    """事故复现：这一格之前必然是「needs_target 失败」。"""

    def test_dataset_name_resolves_the_target(self):
        df = _airlines()
        guess = recommend_target(
            df,
            goal="选择适合的机器学习模型进行处理，并给出数据分析报告",
            dataset_name="航空公司出发延误预测（回归）",
            dataset_hint="regression",
        )
        assert guess.column == "DepDelay"
        assert guess.task == "regression"
        assert guess.source == "goal_match"
        assert guess.confidence >= 0.8
        assert guess.reasons, "必须给出理由"
        assert any("延误" in r or "delay" in r for r in guess.reasons)

    def test_goal_alone_is_enough(self):
        df = _airlines()
        guess = recommend_target(
            df, goal="帮我预测出发延误有多少分钟", dataset_hint="regression"
        )
        assert guess.column == "DepDelay"
        assert guess.source == "goal_match"

    def test_without_any_semantic_evidence_it_does_not_guess(self):
        """没有诉求、没有数据集名 ⇒ 两个回归候选，绝不瞎选。"""
        df = _airlines()
        guess = recommend_target(df, dataset_hint="regression")
        assert guess.column is None
        assert set(guess.alternatives) >= {"DepDelay", "Distance"}
        assert any("无法唯一确定" in r for r in guess.reasons)

    def test_calendar_columns_are_excluded_with_reasons(self):
        df = _airlines()
        guess = recommend_target(df, dataset_hint="regression")
        excluded = {item["column"]: item["reason"] for item in guess.excluded}
        for col in ("Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime"):
            assert col in excluded

    def test_task_mismatch_falls_through_conservatively(self):
        """数据集说是回归，但语义命中的列是二元标签 ⇒ 不硬选，交给调用方。"""
        df = pl.DataFrame(
            {
                "delay_flag": [i % 2 for i in range(500)],
                "distance": [float(i % 300) for i in range(500)],
            }
        )
        guess = recommend_target(df, goal="预测延误", dataset_hint="regression")
        assert guess.column is None
        assert "delay_flag" in guess.alternatives


class TestRecommendTargetOtherShapes:
    def test_naming_convention_wins(self):
        df = pl.DataFrame(
            {"a": [float(i) for i in range(100)], "label": [i % 2 for i in range(100)]}
        )
        guess = recommend_target(df)
        assert guess.column == "label"
        assert guess.source == "naming_convention"
        assert guess.confidence > 0.9

    def test_churn_classification(self):
        df = pl.DataFrame(
            {
                "customerID": [f"c{i}" for i in range(1000)],
                "tenure": [i % 72 for i in range(1000)],
                "MonthlyCharges": [float(i % 90) + 20 for i in range(1000)],
                "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(1000)],
            }
        )
        guess = recommend_target(
            df, goal="预测客户是否会流失", dataset_name="客户流失分类", dataset_hint="classification"
        )
        assert guess.column == "Churn"
        assert guess.task == "classification"

    def test_single_candidate_fallback(self):
        """兜底通道：排除日历列后只剩一个类型相符的候选，可以用。"""
        df = pl.DataFrame(
            {
                "Month": [float(i % 12 + 1) for i in range(500)],
                "sales": [float(i % 300) for i in range(500)],
            }
        )
        guess = recommend_target(df, dataset_hint="regression")
        assert guess.column == "sales"
        assert guess.source == "single_candidate"
        assert 0 < guess.confidence < 0.8

    def test_explicit_target_is_authoritative(self):
        df = _airlines()
        guess = recommend_target(df, goal="预测延误", explicit="Distance")
        assert guess.column == "Distance"
        assert guess.source == "explicit"
        assert guess.confidence == 1.0

    def test_explicit_missing_column_is_reported(self):
        df = _airlines()
        guess = recommend_target(df, explicit="不存在")
        assert guess.column == "不存在"
        assert guess.resolved is True
        assert any("不在数据中" in r for r in guess.reasons)

    def test_no_hint_and_no_goal_yields_nothing(self):
        df = _airlines()
        guess = recommend_target(df)
        assert guess.column is None
        assert guess.reasons

    def test_ambiguous_semantic_match_is_not_guessed(self):
        df = pl.DataFrame(
            {
                "dep_delay": [float(i % 300) for i in range(500)],
                "arr_delay": [float((i * 2) % 300) for i in range(500)],
            }
        )
        guess = recommend_target(df, goal="预测延误", dataset_hint="regression")
        assert guess.column is None
        assert set(guess.alternatives) >= {"dep_delay", "arr_delay"}
        assert any("命中多列" in r for r in guess.reasons)

    def test_unique_key_column_is_never_a_target(self):
        df = pl.DataFrame(
            {
                "row_key": [f"k{i}" for i in range(500)],
                "delay": [float(i % 300) for i in range(500)],
            }
        )
        guess = recommend_target(df, goal="预测延误", dataset_hint="regression")
        assert guess.column == "delay"

    def test_receipt_is_serializable(self):
        df = _airlines()
        guess = recommend_target(df, goal="预测延误", dataset_hint="regression")
        receipt = guess.as_receipt()
        assert receipt["target"] == guess.column
        assert isinstance(receipt["reasons"], list)
        assert 0.0 <= receipt["target_confidence"] <= 1.0
