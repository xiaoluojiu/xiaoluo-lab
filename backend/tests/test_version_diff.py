"""Dataset Version Diff / 时间线测试。

覆盖目标：把「版本差异」做成产品能力，而不是又一个只读页面。

    - 时间线要能说清每个版本**从哪来**（import / 哪个 operation）；
    - Diff 要能说清**变了什么**：行数、列数、Schema（新增/删除/类型变化）、质量；
    - 不新建版本系统 —— 全部复用 DatasetVersion + Operation。

这些断言刻意只依赖「建版本 → 跑操作 → 读 Diff」这条真实链路，
不直接往库里塞 DatasetVersion/Operation 行（那样测的是序列化，不是功能）。
"""

from __future__ import annotations

import polars as pl
import pytest
from app.api.deps import get_storage_service
from app.core.database import get_db
from app.core.exceptions import NotFoundException
from app.data_engine.service import DataEngineService
from app.main import app
from app.services.dataset_service import DatasetService
from fastapi.testclient import TestClient


@pytest.fixture()
def engine(db, storage):
    """真实链路：DatasetService -> DataEngineService（走 operation 产生新版本）。"""
    dataset_service = DatasetService(db, storage)
    dataset = dataset_service.create("diff-demo")
    return {
        "ds": dataset_service,
        "engine": DataEngineService(dataset_service),
        "dataset_id": dataset.id,
    }


def _import(engine, df: pl.DataFrame):
    return engine["ds"].create_version(engine["dataset_id"], df)


def test_timeline_reports_origin_import_then_operation(engine):
    """首个版本来源是 import，之后每个版本挂着产出它的 operation。"""
    _import(engine, pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    engine["engine"].run_operation(
        engine["dataset_id"],
        "filter",
        {"conditions": [{"column": "a", "op": "gt", "value": 1}]},
    )

    timeline = engine["engine"].version_timeline(engine["dataset_id"])

    assert timeline["total"] == 2
    assert timeline["latest_version"] == 2

    first, second = timeline["versions"]

    assert first["origin"] == "import"
    assert first["operation"] is None
    # 首版本没有父版本，delta 无从计算（不是 0，0 是「没变」的意思）
    assert first["delta_rows"] is None

    assert second["origin"] == "filter"
    assert second["operation"]["operation_type"] == "filter"
    assert second["operation"]["status"] == "success"
    assert second["parent_version"] == 1
    assert second["delta_rows"] == -1  # 3 行 -> 2 行


def test_diff_reports_row_and_column_changes(engine):
    _import(
        engine,
        pl.DataFrame({"a": [1, 2, 3, 3], "b": ["x", "y", "z", "z"]}),
    )
    # 去重：4 行 -> 3 行
    engine["engine"].run_operation(
        engine["dataset_id"],
        "duplicate",
        {"keep": "first"},
    )
    # 新增列：3 列（表达式是结构化对象，不允许字符串公式）
    engine["engine"].run_operation(
        engine["dataset_id"],
        "transform",
        {
            "name": "c",
            "overwrite": False,
            "expression": {
                "type": "math",
                "op": "mul",
                "left": {"type": "column", "column": "a"},
                "right": {"type": "value", "value": 2},
            },
        },
    )

    diff = engine["engine"].version_diff(engine["dataset_id"], 1, 3)

    assert diff["row_count"] == {"base": 4, "target": 3, "delta": -1}
    assert diff["column_count"] == {"base": 2, "target": 3, "delta": 1}
    assert [c["column"] for c in diff["schema_diff"]["added"]] == ["c"]
    assert diff["schema_diff"]["removed"] == []
    assert diff["schema_diff"]["type_changed"] == []


def test_diff_reports_removed_and_type_changed_columns(engine):
    """列被删掉 / 类型被改，都要在 schema_diff 里点名。"""
    _import(engine, pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]}))
    engine["engine"].run_operation(
        engine["dataset_id"],
        "cast",
        {"types": {"a": "float"}},
    )
    # aggregate 只保留分组列 + 聚合列：c 消失，b 变成聚合结果
    engine["engine"].run_operation(
        engine["dataset_id"],
        "aggregate",
        {
            "group_by": ["a"],
            "aggregations": [{"func": "sum", "column": "b", "alias": "b_sum"}],
        },
    )

    diff = engine["engine"].version_diff(
        engine["dataset_id"], 1, 3, include_quality=False
    )

    removed = [c["column"] for c in diff["schema_diff"]["removed"]]

    assert "c" in removed
    # a: Int64 -> Float64（类型变化），b 也经过聚合
    changed = {c["column"]: (c["from"], c["to"]) for c in diff["schema_diff"]["type_changed"]}
    assert "a" in changed
    assert changed["a"][1] == "Float64"


def test_diff_quality_reflects_missing_values_being_fixed(engine):
    """质量变化：处理缺失值后 missing_cells 必须下降。"""
    _import(engine, pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1, 2, 3, 4]}))
    engine["engine"].run_operation(
        engine["dataset_id"],
        "missing",
        {"strategy": "drop"},
    )

    diff = engine["engine"].version_diff(engine["dataset_id"], 1, 2)

    assert diff["quality"]["base"]["missing_cells"] == 1
    assert diff["quality"]["target"]["missing_cells"] == 0
    assert diff["quality"]["base"]["issue_count"] >= 1


def test_diff_can_skip_quality_for_large_tables(engine):
    _import(engine, pl.DataFrame({"a": [1, 2, 3]}))
    engine["engine"].run_operation(
        engine["dataset_id"],
        "filter",
        {"conditions": [{"column": "a", "op": "gt", "value": 1}]},
    )

    diff = engine["engine"].version_diff(
        engine["dataset_id"], 1, 2, include_quality=False
    )

    assert diff["quality"] is None
    # 结构差异照样算得出来
    assert diff["row_count"]["delta"] == -1


def test_diff_of_same_version_is_empty_not_error(engine):
    """自己和自己比是合法请求：所有 delta 为 0，不报错。"""
    _import(engine, pl.DataFrame({"a": [1, 2]}))

    diff = engine["engine"].version_diff(engine["dataset_id"], 1, 1)

    assert diff["row_count"]["delta"] == 0
    assert diff["schema_diff"]["added"] == []
    assert diff["schema_diff"]["removed"] == []


def test_diff_records_the_operation_that_produced_target(engine):
    """Diff 要说清 target 版本是**怎么来的**（操作来源）。"""
    _import(engine, pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    engine["engine"].run_operation(
        engine["dataset_id"],
        "filter",
        {"conditions": [{"column": "a", "op": "gte", "value": 2}]},
    )

    diff = engine["engine"].version_diff(engine["dataset_id"], 1, 2)

    assert len(diff["operations"]) == 1
    assert diff["operations"][0]["operation_type"] == "filter"
    assert diff["operations"][0]["output_version_id"] == diff["target"]["id"]
    assert diff["operations"][0]["parameters"]["conditions"][0]["value"] == 2


def test_diff_rejects_unknown_version(engine):
    _import(engine, pl.DataFrame({"a": [1, 2]}))

    with pytest.raises(NotFoundException):
        engine["engine"].version_diff(engine["dataset_id"], 1, 99)


def test_timeline_of_dataset_without_version_is_empty(engine):
    """空数据集是正常状态：时间线为空，不抛异常。"""
    timeline = engine["engine"].version_timeline(engine["dataset_id"])

    assert timeline["versions"] == []
    assert timeline["latest_version"] == 0
    assert timeline["total"] == 0


# =========================================================
# API 层：路由可达 + 参数校验（服务层已在上面覆盖）
# =========================================================


@pytest.fixture()
def api(db, storage) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_storage_service] = lambda: storage
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_api_version_timeline_and_diff(api, db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("api-diff")
    ds.create_version(dataset.id, pl.DataFrame({"a": [1, 2, 3, 3]}))
    DataEngineService(ds).run_operation(dataset.id, "duplicate", {"keep": "first"})

    timeline = api.get(f"/api/v1/datasets/{dataset.id}/versions").json()
    assert timeline["success"] is True
    assert [v["version"] for v in timeline["data"]["versions"]] == [1, 2]

    diff = api.get(
        f"/api/v1/datasets/{dataset.id}/versions/diff",
        params={"base": 1, "target": 2},
    ).json()
    assert diff["success"] is True
    assert diff["data"]["row_count"] == {"base": 4, "target": 3, "delta": -1}


def test_api_diff_requires_base_and_target(api, db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("api-diff-params")
    ds.create_version(dataset.id, pl.DataFrame({"a": [1, 2]}))

    resp = api.get(f"/api/v1/datasets/{dataset.id}/versions/diff")
    assert resp.status_code == 422  # base / target 缺失


def test_api_diff_unknown_version_is_404_not_500(api, db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("api-diff-404")
    ds.create_version(dataset.id, pl.DataFrame({"a": [1, 2]}))

    resp = api.get(
        f"/api/v1/datasets/{dataset.id}/versions/diff",
        params={"base": 1, "target": 42},
    )
    # 版本不存在是「没找到」，不是服务端异常
    assert resp.status_code == 404
