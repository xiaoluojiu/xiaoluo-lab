"""删除数据集必须连带清理**所有**外键依赖（2026-09-27 线上 500 回归）。

事故：前端删除数据集 28，连续两次返回 500，日志是

    sqlite3.IntegrityError: FOREIGN KEY constraint failed
    [SQL: DELETE FROM dataset_versions WHERE dataset_versions.id = ?]
    [parameters: [(89,), (90,)]]

根因是三层互相独立的引用，只修任何一层删除依然失败：

1. ``operations.input_version_id`` / ``operations.output_version_id`` 指向被删版本；
2. ``experiments.dataset_version_id`` 指向被删版本；
3. ``dataset_versions.parent_version_id`` 是**自引用**（v2.parent = v1），
   先删 v1 就违约。

另有两处把问题放大的错误：

4. 本项目 Session 是 ``autoflush=False``，只 ``db.delete(exp)`` 不会真正发出 DELETE，
   必须在删版本之前显式 ``flush()``；
5. 旧实现**先删快照文件、后提交**：提交一失败就留下「列表里还在、Parquet 已经没了」
   的幽灵数据集（dataset 28 的 ``data/datasets/28/`` 当时确实已被清空，且每次重试
   都仍然 500）。现在改为提交成功后再清理存储。
"""

from __future__ import annotations

from app.api.deps import get_storage_service
from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.experiment import Experiment
from app.models.experiment_run import ExperimentRun
from app.models.operation import Operation
from sqlalchemy import func, select


def _seed_dataset_graph(db, storage, name: str) -> dict:
    """造一个「有依赖」的数据集。

    两个版本（v2 以 v1 为父）+ 操作审计行 + 两个实验（各带一个 run）+ 两个快照文件。
    这套结构把线上那三层外键依赖一次性覆盖，缺任何一层就复现不出原始错误。
    """

    dataset = Dataset(name=name)
    db.add(dataset)
    db.flush()

    keys: list[str] = []
    versions: list[DatasetVersion] = []

    for index, version in enumerate((1, 2)):
        key = f"datasets/{dataset.id}/v{version:06d}.parquet"
        storage.save(key, b"fake-parquet")
        keys.append(key)

        row = DatasetVersion(
            dataset_id=dataset.id,
            version=version,
            # v2 的父版本是 v1 —— 自引用，删除顺序不对就会违约
            parent_version_id=versions[-1].id if versions else None,
            storage_path=key,
            row_count=100 - index,
            column_count=3,
            schema_json={"a": "Int64", "b": "String", "c": "Float64"},
        )
        db.add(row)
        db.flush()
        versions.append(row)

    v1, v2 = versions

    db.add(
        Operation(
            dataset_id=dataset.id,
            input_version_id=v1.id,
            output_version_id=v2.id,
            operation_type="filter",
            parameters={"condition": "a > 1"},
            status="success",
        )
    )

    for model in ("linear_regression", "random_forest_classifier"):
        experiment = Experiment(
            dataset_id=dataset.id,
            dataset_version_id=v1.id,
            task="classification",
            model=model,
            target_column="c",
            preprocessing={},
        )
        db.add(experiment)
        db.flush()
        db.add(ExperimentRun(experiment_id=experiment.id, status="success"))

    db.commit()

    return {
        "dataset_id": dataset.id,
        "version_ids": [v1.id, v2.id],
        "keys": keys,
    }


def _counts(db, dataset_id: int) -> dict[str, int]:
    db.expire_all()

    return {
        "datasets": int(
            db.scalar(
                select(func.count()).select_from(Dataset).where(Dataset.id == dataset_id)
            )
            or 0
        ),
        "versions": int(
            db.scalar(
                select(func.count())
                .select_from(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
            )
            or 0
        ),
        "operations": int(
            db.scalar(
                select(func.count())
                .select_from(Operation)
                .where(Operation.dataset_id == dataset_id)
            )
            or 0
        ),
        "experiments": int(
            db.scalar(
                select(func.count())
                .select_from(Experiment)
                .where(Experiment.dataset_id == dataset_id)
            )
            or 0
        ),
    }


def test_delete_dataset_clears_all_fk_dependents(client):
    """带版本/操作/实验/运行的数据集可以被删掉，且不留下任何悬挂引用。"""

    db = SessionLocal()
    storage = get_storage_service()
    try:
        seeded = _seed_dataset_graph(db, storage, "删库回归-有依赖")
        dataset_id = seeded["dataset_id"]

        assert _counts(db, dataset_id) == {
            "datasets": 1,
            "versions": 2,
            "operations": 1,
            "experiments": 2,
        }

        resp = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 200, resp.text

        data = resp.json()["data"]
        assert data["deleted"] is True
        assert data["deleted_versions"] == 2
        assert data["deleted_operations"] == 1
        assert data["deleted_experiments"] == 2
        assert data["deleted_runs"] == 2
        assert data["deleted_storage_objects"] == 2

        assert _counts(db, dataset_id) == {
            "datasets": 0,
            "versions": 0,
            "operations": 0,
            "experiments": 0,
        }

        # 悬挂引用检查：任何一张表都不该再指向已删除的行
        db.expire_all()
        orphan_runs = db.scalar(
            select(func.count())
            .select_from(ExperimentRun)
            .where(~ExperimentRun.experiment_id.in_(select(Experiment.id)))
        )
        dangling_parent = db.scalar(
            select(func.count())
            .select_from(DatasetVersion)
            .where(
                DatasetVersion.parent_version_id.is_not(None),
                ~DatasetVersion.parent_version_id.in_(select(DatasetVersion.id)),
            )
        )
        assert orphan_runs == 0
        assert dangling_parent == 0

        # 快照文件在提交之后一并清理
        assert all(not storage.exists(key) for key in seeded["keys"])
    finally:
        db.close()


def test_delete_dataset_removes_linked_experiment_runs(client):
    """实验被级联删除时，其 experiment_runs 不能留下（experiment_runs -> experiments）。"""

    db = SessionLocal()
    storage = get_storage_service()
    try:
        seeded = _seed_dataset_graph(db, storage, "删库回归-运行记录")
        dataset_id = seeded["dataset_id"]

        run_ids = list(
            db.scalars(
                select(ExperimentRun.id).where(
                    ExperimentRun.experiment_id.in_(
                        select(Experiment.id).where(Experiment.dataset_id == dataset_id)
                    )
                )
            )
        )
        assert run_ids

        resp = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 200, resp.text

        db.expire_all()
        assert (
            db.scalar(
                select(func.count())
                .select_from(ExperimentRun)
                .where(ExperimentRun.id.in_(run_ids))
            )
            == 0
        )
    finally:
        db.close()


def test_delete_missing_dataset_returns_404(client):
    """再删一次给 404，而不是 500 或半删状态。"""

    resp = client.delete("/api/v1/datasets/999999")
    assert resp.status_code == 404, resp.text
