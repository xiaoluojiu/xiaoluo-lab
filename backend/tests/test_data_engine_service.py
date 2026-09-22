"""Prompt 061：DataEngineService 整合测试（版本级操作 + Merge + Operation 记录）。"""

import polars as pl
import pytest
from app.data_engine.exceptions import TransformError
from app.data_engine.merge.plan import JoinKey, MergePlan
from app.data_engine.service import DataEngineService
from app.services.dataset_service import DatasetService


@pytest.fixture()
def dataset_service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


@pytest.fixture()
def engine(dataset_service) -> DataEngineService:
    return DataEngineService(dataset_service)


@pytest.fixture()
def dataset_id(dataset_service) -> int:
    ds = dataset_service.create("source")
    df = pl.DataFrame({"id": [1, 2, 3, 4], "name": ["a", None, "c", "d"], "v": [1, 2, 3, 4]})
    dataset_service.create_version(ds.id, df)
    return ds.id


class TestVersionedOperations:
    def test_run_operation_creates_new_version(self, engine, dataset_service, dataset_id):
        versions_before = dataset_service.get_versions(dataset_id)[1]
        output, operation = engine.run_operation(
            dataset_id, "missing", {"strategy": "drop"}
        )
        assert output.version == versions_before + 1
        assert output.row_count == 3  # name 为 null 的行被删除
        assert operation.input_version_id is not None
        assert operation.output_version_id == output.id
        assert operation.status == "success"

    def test_original_version_unchanged(self, engine, dataset_service, dataset_id):
        before = dataset_service.load_version(dataset_id)
        engine.run_operation(dataset_id, "missing", {"strategy": "drop"})
        # 原版本（v1）仍可按版本号读取且内容不变
        original_df = dataset_service.load_version(dataset_id, version=1)
        assert original_df.height == before.height

    def test_failed_operation_recorded(self, engine, dataset_service, dataset_id):
        with pytest.raises(TransformError):
            engine.run_operation(dataset_id, "missing", {"strategy": "nope"})
        ops = engine.operation_history(dataset_id)
        assert any(op.status == "failed" for op in ops)

    def test_unknown_operation(self, engine, dataset_id):
        with pytest.raises(TransformError):
            engine.run_operation(dataset_id, "nope", {})

    def test_available_operations(self, engine):
        ops = engine.available_operations()
        names = {o["op_type"] for o in ops}
        assert {"missing", "duplicate", "cast", "filter", "aggregate", "pivot", "melt"} <= names


class TestAnalysis:
    def test_schema_profile_quality(self, engine, dataset_service, dataset_id):
        df = dataset_service.load_version(dataset_id)
        schema = engine.schema(df)
        prof = engine.profile(df)
        quality = engine.quality(df)
        assert schema["column_count"] == 3
        assert prof["row_count"] == 4
        assert quality["statistics"]["row_count"] == 4

    def test_preview(self, engine, dataset_id):
        result = engine.preview(
            engine.dataset_service.load_version(dataset_id),
            page=1, page_size=2, sort={"column": "id", "desc": True},
        )
        assert result["total"] == 4
        assert result["items"][0]["id"] == 4


class TestMergeFlow:
    def test_run_merge(self, engine, dataset_service):
        left = dataset_service.create("left-ds")
        right = dataset_service.create("right-ds")
        dataset_service.create_version(left.id, pl.DataFrame({"k": [1, 2], "a": ["x", "y"]}))
        dataset_service.create_version(right.id, pl.DataFrame({"k": [1, 2], "b": [10, 20]}))

        plan = MergePlan(
            left={"dataset_id": left.id}, right={"dataset_id": right.id},
            keys=[JoinKey(left="k", right="k")], join_type="inner",
        )
        output, report = engine.run_merge(left.id, right.id, plan)
        assert output.dataset_id == left.id
        assert output.row_count == 2
        assert report.matched_rows == 2
        assert report.output_columns == 3  # k, a, b

        merged = dataset_service.load_version(left.id)
        assert "b" in merged.columns

    def test_validate_merge(self, engine, dataset_service):
        left = pl.DataFrame({"k": [1]})
        right = pl.DataFrame({"k": [1]})
        plan = MergePlan(keys=[JoinKey("k", "k")])
        result = engine.validate_merge(left, right, plan)
        assert result["ok"] is True
