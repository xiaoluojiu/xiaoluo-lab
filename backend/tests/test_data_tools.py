"""data.aggregate 的参数兜底。

槽位抽取（尤其是模型抽取）给出的聚合配置形状并不稳定：实测「按 purpose 分组
统计 credit_amount 的均值和笔数」抽出来的是 ``[{"credit_amount": "mean"}]``
—— 一个单键字典。直接交给数据引擎的结果是整次运行失败，
用户只看到 ``unsupported aggregation func: None``。
"""

from __future__ import annotations

import polars as pl
import pytest

from app.agent.permission import Permission
from app.tools.base import ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.data_tools import DataAggregateTool, _normalize_aggregations


def test_single_key_shape_is_expanded():
    """ ``{"列名": "函数名"}`` 必须翻成 ``{"column": ..., "func": ...}``。"""
    out = _normalize_aggregations({"aggregations": [{"credit_amount": "mean"}]})
    assert out["aggregations"] == [{"column": "credit_amount", "func": "mean"}]


def test_missing_func_falls_back_to_mean():
    """没写函数不该让整次运行失败 —— 「统计 x」默认就是求均值。"""
    out = _normalize_aggregations({"aggregations": [{"column": "credit_amount"}]})
    assert out["aggregations"][0]["func"] == "mean"


def test_chinese_func_name_is_translated():
    out = _normalize_aggregations({"aggregations": [{"column": "credit_amount", "func": "平均值"}]})
    assert out["aggregations"][0] == {"column": "credit_amount", "func": "mean"}


def test_item_without_column_is_dropped():
    """连列名都没有的聚合项不能编造一个假列 —— 丢弃，让上层去澄清。"""
    params = {"aggregations": [{"func": "笔数"}]}
    out = _normalize_aggregations(params)
    assert out is params or out["aggregations"] != [{"column": None, "func": "count"}]
    # 但只要还有一条能用的，就得把它留下（不能整单作废）
    mixed = _normalize_aggregations({"aggregations": [{"func": "笔数"}, {"credit_amount": "mean"}]})
    assert mixed["aggregations"] == [{"column": "credit_amount", "func": "mean"}]


@pytest.mark.parametrize("bad", [[], "not-a-list", [None], [{}], ["x"]])
def test_unrecoverable_input_is_left_untouched(bad):
    """救不回来的输入原样交回：让它走正常的必填槽位澄清，而不是静默改写成假参数。"""
    params = {"dataset_id": 8, "aggregations": bad}
    assert _normalize_aggregations(params) is params


def test_other_params_are_preserved():
    params = {"dataset_id": 8, "group_by": ["purpose"], "aggregations": [{"credit_amount": "mean"}]}
    out = _normalize_aggregations(params)
    assert out["dataset_id"] == 8
    assert out["group_by"] == ["purpose"]


# ---------------------------------------------------------------- 只读性


def test_aggregate_returns_rows_without_creating_a_dataset_version():
    """★ 回归：聚合曾把结果写成数据集的**新版本**，把整份数据顶掉。

    实机事故：1220 行 × 14 列的信贷样本上跑一次「按 purpose 分组求平均」，
    数据集的当前版本就变成了一张 7 行 × 2 列的分组表 —— 随后的相关性分析报
    「至少需要 2 个数值字段」、质量检查说「共 7 行、2 列」、用户点名的 age 列
    根本不存在。聚合是看数字，不是改数据。
    """
    df = pl.DataFrame(
        {
            "purpose": ["car", "car", "education", "education"],
            "credit_amount": [1000, 3000, 2000, 4000],
        }
    )

    class _DatasetService:
        def __init__(self) -> None:
            self.created: list[int] = []

        def load_version(self, dataset_id: int, version: int | None = None):
            return df

        def create_version(self, *args, **kwargs):  # pragma: no cover - 不该被调用
            self.created.append(1)
            raise AssertionError("聚合不应创建数据集版本")

    service = _DatasetService()
    services = ToolServices(dataset_service=service)
    context = ToolExecutionContext(
        user_id="tester", session_id="s-1", dataset_ids={1}, permissions=set(Permission)
    )

    result = DataAggregateTool().execute(
        {"dataset_id": 1, "group_by": ["purpose"], "aggregations": [{"credit_amount": "mean"}]},
        context,
        services,
    )

    assert result.success, result.errors
    assert service.created == [], "聚合绝不能产出数据集新版本"
    # 结果必须真的到得了答案里：只能说出「几行几列」等于没算
    assert result.data["rows"] == 2
    assert {"purpose": "car", "credit_amount_mean": 2000.0} in result.data["preview"]
    assert "未改动" in result.summary


def test_aggregate_is_declared_read_only():
    """自描述必须与实际行为一致：前端据此决定要不要弹确认框。"""
    describe = DataAggregateTool().describe()
    # 分析权限即只读权限：它不写数据集，也因此不需要用户确认
    assert str(describe["permission"]) == str(Permission.ANALYZE_DATA)
    assert describe["requires_confirmation"] is False
