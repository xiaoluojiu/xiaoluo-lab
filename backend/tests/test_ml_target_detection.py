"""分类列判定 / ML 任务类型判定 / 目标列冗余排除 的回归测试。

三个真实缺陷：
1. `is_categorical_like` 只放行整型列，ARFF 里被读成 Float64 的 Month / DayofMonth /
   DayOfWeek 因此被当连续量：进了相关性热力图、被算均值、还生成了正态 Q-Q 图
   （事故报告里那张「Month 正态性 Q-Q 图」357 KB，纯粹是伪影）；
2. `ml.train` 调用 `ml.detect_task` 时漏传 `infer_target`（注释声称会传），
   使「训练一个模型」链路静默退化成聚类；
3. `target_column` 被同时写进 `excluded_columns` 会让训练直接失败，而这是一个
   幂等的冗余指令 —— 事故里同一个回归请求连续 3 次 run failed（0.2 秒/次）。
4. 【2026-09-23 第二轮】`ml.detect_task` 在「数据集名写明是回归、但列名没有
   target/label/y/class」时只能失败并回传候选，而规划器**看不到列名**根本无法
   据此改计划 ⇒ 链路死锁在「我不知道该预测哪一列」。现在由
   `app.ml_engine.target_inference` 分级推断（命名约定 → 诉求语义 → 类型分布）。
5. 【同轮】回归任务被按类别分层切分：分层判定只看目标唯一值数，于是
   「延误分钟被取整成几十档」这种低基数回归目标会被当成多分类，
   以「测试集样本不足以覆盖 N 个类别」失败 —— 把回归说成分类。
"""

from __future__ import annotations

import polars as pl
import pytest

from app.analysis import (
    categorical_columns,
    classify_columns,
    continuous_columns,
    is_categorical_like,
)
from app.ml_engine.step_runner import run_load_step
from app.tools.context import ToolExecutionContext
from app.tools.ml_tools import _dataset_task_hint, _intent_of, _target_candidates


def _airlines_like(n: int = 2000) -> pl.DataFrame:
    """模拟 dataset 9 读进来后的形状：10 列全部是 Float64 / String。"""
    return pl.DataFrame(
        {
            "DepDelay": [float(i % 300 - 20) for i in range(n)],
            "Month": [float(i % 12 + 1) for i in range(n)],
            "DayofMonth": [float(i % 31 + 1) for i in range(n)],
            "DayOfWeek": [float(i % 7 + 1) for i in range(n)],
            "CRSDepTime": [float((i % 24) * 100) for i in range(n)],
            "UniqueCarrier": [f"C{i % 30}" for i in range(n)],
            "Origin": [f"A{i % 400}" for i in range(n)],
            "Distance": [float(i * 1.37) for i in range(n)],
        }
    )


class TestCategoricalLike:
    def test_integer_valued_float_codes_are_categorical(self):
        df = _airlines_like()
        for col in ("Month", "DayofMonth", "DayOfWeek"):
            assert df.schema[col] == pl.Float64
            assert is_categorical_like(df, col) is True, col

    def test_high_cardinality_float_stays_continuous(self):
        df = _airlines_like()
        # DepDelay 有 300 个取值、Distance 基本一一对应 —— 都不能被判成编码列
        assert is_categorical_like(df, "DepDelay") is False
        assert is_categorical_like(df, "Distance") is False

    def test_fractional_float_is_never_categorical(self):
        df = pl.DataFrame({"price": [float(i) + 0.5 for i in range(2000)]})
        assert is_categorical_like(df, "price") is False

    def test_small_table_is_left_alone(self):
        """小表里「唯一值少」只说明采样少，不能据此把低基数浮点列判成编码列。"""
        df = pl.DataFrame({"a": [float(i % 11) for i in range(100)], "b": [float(i) for i in range(100)]})
        assert is_categorical_like(df, "a") is False

    def test_classify_batch_matches_per_column(self):
        df = _airlines_like()
        batch = classify_columns(df)
        for col in df.columns:
            assert batch[col] is is_categorical_like(df, col), col

    def test_column_partitions_are_disjoint_and_complete(self):
        df = _airlines_like()
        cat = set(categorical_columns(df))
        cont = set(continuous_columns(df))
        assert not (cat & cont)
        assert cat | cont == set(df.columns)
        assert {"Month", "DayofMonth", "DayOfWeek"} <= cat
        assert {"DepDelay", "Distance"} <= cont


class TestDatasetTaskHint:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("航空公司出发延误预测（回归）", "regression"),
            ("airlines regression", "regression"),
            ("german credit classification", "classification"),
            ("iris（分类）", "classification"),
            ("客户分群聚类分析", "clustering"),
            ("某数据集", None),
        ],
    )
    def test_hint(self, name, expected):
        assert _dataset_task_hint(name) == expected


class TestTargetCandidates:
    def test_lists_both_task_shapes(self):
        df = _airlines_like()
        options = _target_candidates(df)
        assert "DepDelay" in options["regression"]
        assert "Month" in options["classification"]


class TestLoadStepToleratesRedundantTargetExclusion:
    def test_target_in_excluded_is_auto_dropped(self):
        df = pl.DataFrame({"a": [float(i) for i in range(20)], "label": [i % 2 for i in range(20)]})
        result = run_load_step(df, target="label", excluded_columns=["label", "a"])
        assert result.status == "ok", result.error
        assert result.output["feature_columns"] == []

    def test_missing_excluded_column_still_fails(self):
        df = pl.DataFrame({"a": [float(i) for i in range(20)], "label": [i % 2 for i in range(20)]})
        result = run_load_step(df, target="label", excluded_columns=["不存在"])
        assert result.status == "failed"


class TestUserRequestReachesTools:
    """事故的直接成因之一：规划阶段看不到列名，工具又拿不到用户诉求 ⇒ 无法推断目标列。

    规划器拿不到列名是**架构约束**（ContextBuilder 禁止规划阶段读 Schema），
    所以诉求必须由运行时注入，而不是指望 LLM 记得把原文抄进参数里。
    """

    def test_tool_context_carries_user_request(self):
        from app.agent.runtime.models import AgentSession
        from app.agent.runtime.runtime import AgentRuntime

        session = AgentSession(id="s-1", user_id="u-1", dataset_ids=[9])
        # 该方法不触碰 self，直接以未绑定方式调用即可验证注入逻辑
        ctx = AgentRuntime._tool_context(None, session, "analyst", user_request="预测出发延误")
        assert ctx.extra["user_request"] == "预测出发延误"
        assert ctx.dataset_ids == {9}

    def test_tool_context_without_request_is_empty_string(self):
        from app.agent.runtime.models import AgentSession
        from app.agent.runtime.runtime import AgentRuntime

        session = AgentSession(id="s-2", user_id="u-1")
        ctx = AgentRuntime._tool_context(None, session, "analyst")
        assert ctx.extra["user_request"] == ""

    def test_intent_prefers_explicit_goal(self):
        ctx = ToolExecutionContext(user_id="u", extra={"user_request": "来自运行时的诉求"})
        assert _intent_of({"goal": "显式给的诉求"}, ctx) == "显式给的诉求"
        assert _intent_of({}, ctx) == "来自运行时的诉求"
        assert _intent_of({}, ToolExecutionContext(user_id="u")) == ""
        assert _intent_of({"goal": ""}, ctx) == "来自运行时的诉求"
