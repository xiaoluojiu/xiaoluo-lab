"""槽位抽取：规则优先，LLM 只在规则读不出时才出手。

关键回归点是「零 Token」：没有 provider 时，抽取必须照常工作并且绝不发起调用。
"""

from __future__ import annotations

import pytest

from app.agent.playbooks import PlaybookStep
from app.agent.slots import _parse_json_loose, extract_slots, llm_extract, rule_extract

COLUMNS = ["id", "age", "user_id", "city", "income"]


def _step(required: tuple[str, ...] = ()) -> PlaybookStep:
    return PlaybookStep(tool="eda.distribution", title="分布分析", required_slots=required)


# ---------------------------------------------------------------- 规则抽取


def test_column_prefers_longest_match():
    """「user_id 的分布」不能抽成 id —— 取最长匹配。"""
    found = rule_extract("分析 user_id 的分布", _step(("column",)), schema_columns=COLUMNS)
    assert found["column"] == "user_id"


def test_column_unknown_returns_nothing():
    """列名表里没有的词，宁可不抽，也不瞎猜。"""
    assert rule_extract("分析 手机号 的分布", _step(("column",)), schema_columns=COLUMNS) == {}


def test_chart_alias():
    assert rule_extract("画个直方图", _step(("chart",)))["chart"] == "histogram"
    assert rule_extract("来个箱线图", _step(("chart",)))["chart"] == "box"


def test_run_id():
    assert rule_extract("评估模型 run id 42", _step(("run_id",)))["run_id"] == 42


def test_test_size():
    """「测试集占 30%」这类测试集比例必须被抽取，不能静默忽略。"""
    step = _step(("test_size",))
    assert rule_extract("测试集占30%", step)["test_size"] == 0.3
    assert rule_extract("test_size=0.25", step)["test_size"] == 0.25
    assert rule_extract("30%测试集", step)["test_size"] == 0.3
    assert rule_extract("验证集 20%", step)["test_size"] == 0.2
    # 非法比例（不在 0~1 开区间）不抽
    assert rule_extract("测试集占120%", step) == {}
    assert rule_extract("测试集占0%", step) == {}


def test_group_by_and_aggregation():
    found = rule_extract("按 city 分组统计 income 的平均值", _step(("group_by", "aggregations")), schema_columns=COLUMNS)
    assert found["group_by"] == ["city"]
    assert found["aggregations"] == [{"income": "mean"}]


def test_aggregation_without_known_column_is_skipped():
    """只知道聚合函数、不知道列 → 不猜。"""
    assert rule_extract("求平均", _step(("group_by", "aggregations")), schema_columns=COLUMNS) == {}


def test_rule_extract_without_schema_is_empty():
    assert rule_extract("分析 age 的分布", _step(("column",))) == {}


# ---------------------------------------------------------------- 复数列槽位


def _columns_step() -> PlaybookStep:
    return PlaybookStep(tool="eda.correlation", title="相关性", tuned_slots=("columns",))


def test_named_columns_are_all_extracted():
    """「看看 customer_id 和 age 的相关性」必须两个列都传进去。

    规则抽取此前只认单数 ``column``，而相关性/描述统计/异常值用的都是 ``columns``
    ——点名的列一个都传不到工具，答案只能写「矩阵里没有你点名的列」。
    """
    found = rule_extract(
        "看看 customer_id 和 age 的相关性", _columns_step(),
        schema_columns=["customer_id", "age", "credit_amount"],
    )
    assert found["columns"] == ["customer_id", "age"]


def test_no_named_column_means_no_filter():
    """没点名列就不该传 columns：让工具自己挑全表的连续数值列。"""
    found = rule_extract(
        "看看各数值列之间的相关性", _columns_step(),
        schema_columns=["customer_id", "age", "credit_amount"],
    )
    assert "columns" not in found


def test_columns_keep_dataset_order():
    """结果顺序按数据集列序，不随用户说话顺序变 —— 否则同一句问法的输出不稳定。"""
    found = rule_extract(
        "分析 income 与 age 的关系", _columns_step(), schema_columns=COLUMNS
    )
    assert found["columns"] == ["age", "income"]


def test_short_column_is_not_counted_twice():
    """``user_id`` 只能算一次：短列 ``id`` 是它的子串，不能额外再计一列。"""
    found = rule_extract("看看 user_id 和 age", _columns_step(), schema_columns=COLUMNS)
    assert found["columns"] == ["age", "user_id"]  # 按数据集列序，且没有单独的 "id"


# ---------------------------------------------------------------- 无 LLM 路径


def test_extract_slots_without_provider_never_calls_llm():
    """provider 为 None 时必须静默返回，不能抛异常、不能假装调用。"""
    found = extract_slots("分析 age 的分布", _step(("column",)), schema_columns=COLUMNS, provider=None)
    assert found == {"column": "age"}


def test_llm_extract_without_provider_returns_empty():
    assert llm_extract("分析 age 的分布", _step(("column",)), provider=None) == {}


def test_auto_slots_win_over_rule():
    """自动槽位（会话绑定的 dataset_id）优先级最高，规则不得覆盖。"""
    found = extract_slots("看看数据", _step(("dataset_id",)), auto={"dataset_id": 7})
    assert found["dataset_id"] == 7


# ---------------------------------------------------------------- LLM 降级


class _ExplodingProvider:
    """任何调用都抛异常的 provider —— 用来证明失败不会中断链路。"""

    def chat(self, *args, **kwargs):
        raise RuntimeError("网络不可用")


def test_llm_failure_degrades_to_empty():
    result = llm_extract("分析 age 的分布", _step(("column",)), provider=_ExplodingProvider())
    assert result == {}


class _GarbageProvider:
    def chat(self, *args, **kwargs):
        class _Resp:
            content = "我觉得应该是 age 吧，不太确定"
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

        return _Resp()


def test_llm_non_json_output_degrades_to_empty():
    assert llm_extract("分析 age 的分布", _step(("column",)), provider=_GarbageProvider()) == {}


class _JsonProvider:
    def chat(self, *args, **kwargs):
        class _Resp:
            content = '```json\n{"column": "age", "evil": "rm -rf"}\n```'
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

        return _Resp()


def test_llm_output_is_filtered_to_declared_slots():
    """模型顺手多给的参数必须被丢弃，否则会传进工具造成未知行为。"""
    result = llm_extract("分析 age 的分布", _step(("column",)), provider=_JsonProvider())
    assert result == {"column": "age"}


# ---------------------------------------------------------------- 宽松解析


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('好的：{"a": 1} 就这样', {"a": 1}),
        ("抱歉，我做不到", None),
        ("", None),
    ],
)
def test_parse_json_loose(raw: str, expected):
    assert _parse_json_loose(raw) == expected
