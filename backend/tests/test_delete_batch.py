"""删除能力回归：实验/运行的批量与单条删除、已保存报告的批量删除（2026-09-28）。

改造前的缺口（用户反馈「删除功能不够完善」）：

1. 报告中心只有单条删除，没有「删除选中」「一键删除全部」；
2. 机器学习页的「实验历史与运行选择」**完全没有删除入口** ——
   既删不掉一次跑废的运行，也删不掉整条实验；
3. 实验中心的勾选只服务「横向对比」，不能用它批量删。

本文件钉住三件事：

* **删除运行 ≠ 删除实验**：删 run 只掉这一次记录与它的 pkl，实验配置仍在，
  还可以重新「运行实验」；
* **产物清理不能漏**：单条删与批量删必须共用同一份产物清理实现，
  否则会出现「界面里没了、存储里还躺着 model.pkl」这类看不见的泄漏；
* **批量删除是幂等的**：id 不存在直接跳过，不让整批失败（用户界面上的列表是快照，
  期间别处删掉其中一条属正常竞态）。
"""

from __future__ import annotations

from app.api.deps import get_storage_service
from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.experiment import Experiment
from app.models.experiment_run import ExperimentRun
from sqlalchemy import func, select


def _seed(db, name: str, *, experiments: int = 2, runs_per_experiment: int = 2) -> dict:
    """造 dataset + version + N 个实验（各带 M 个运行，每个运行都有自己的产物文件）。"""
    storage = get_storage_service()

    dataset = Dataset(name=name)
    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.id,
        version=1,
        storage_path=f"datasets/{dataset.id}/v000001.parquet",
        row_count=120,
        column_count=3,
        schema_json={"a": "Int64", "b": "String", "c": "Float64"},
    )
    db.add(version)
    db.flush()

    run_artifact_keys: dict[int, list[str]] = {}
    experiment_ids: list[int] = []

    for _ in range(experiments):
        exp = Experiment(
            dataset_id=dataset.id,
            dataset_version_id=version.id,
            task="classification",
            model="logistic_regression",
            target_column="c",
            preprocessing={},
        )
        db.add(exp)
        db.flush()
        experiment_ids.append(exp.id)

        for _ in range(runs_per_experiment):
            run = ExperimentRun(
                experiment_id=exp.id, status="success", metrics={"accuracy": 0.9}
            )
            db.add(run)
            db.flush()
            # 产物 key 用真实 run id，才能让「删 A 不影响 B 的产物」成为可断言的事实
            keys = [
                f"experiments/{exp.id}/runs/{run.id}/model.pkl",
                f"experiments/{exp.id}/runs/{run.id}/pipeline.pkl",
            ]
            for key in keys:
                storage.save(key, b"fake-model-bytes")
            run.artifacts = {"model_key": keys[0], "pipeline_key": keys[1]}
            run_artifact_keys[run.id] = keys

    db.commit()
    return {
        "dataset_id": dataset.id,
        "version_id": version.id,
        "experiment_ids": experiment_ids,
        "run_artifact_keys": run_artifact_keys,
    }


def _run_ids(db, experiment_id: int) -> list[int]:
    db.expire_all()
    return list(
        db.scalars(
            select(ExperimentRun.id)
            .where(ExperimentRun.experiment_id == experiment_id)
            .order_by(ExperimentRun.id)
        )
    )


def _count(table, where) -> int:
    db = SessionLocal()
    try:
        return int(db.scalar(select(func.count()).select_from(table).where(where)) or 0)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 运行（run）级删除
# ---------------------------------------------------------------------------
def test_delete_single_run_keeps_experiment_and_clears_artifacts(client):
    """删一次运行：记录消失、产物文件清掉，实验本身与它的其它运行都还在。"""
    db = SessionLocal()
    storage = get_storage_service()
    try:
        seeded = _seed(db, "删运行回归", experiments=1, runs_per_experiment=2)
        exp_id = seeded["experiment_ids"][0]
        run_ids = _run_ids(db, exp_id)
        assert len(run_ids) == 2
        target, survivor = run_ids

        resp = client.delete(f"/api/v1/experiments/runs/{target}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == {"deleted": True, "run_id": target}

        db.expire_all()
        assert db.get(ExperimentRun, target) is None
        assert db.get(ExperimentRun, survivor) is not None
        # 实验配置必须保留 —— 这就是「删运行」与「删实验」的分界
        assert db.get(Experiment, exp_id) is not None

        assert all(not storage.exists(k) for k in seeded["run_artifact_keys"][target])
        # 幸存运行自己的产物不受牵连
        assert all(storage.exists(k) for k in seeded["run_artifact_keys"][survivor])
    finally:
        db.close()


def test_delete_run_missing_returns_404(client):
    resp = client.delete("/api/v1/experiments/runs/999999")
    assert resp.status_code == 404, resp.text


def test_batch_delete_runs_is_idempotent(client):
    """批量删运行：一次清掉多条，未知 id 只被跳过、不让整批失败。"""
    db = SessionLocal()
    storage = get_storage_service()
    try:
        seeded = _seed(db, "删运行批量回归", experiments=2, runs_per_experiment=2)
        first, second = seeded["experiment_ids"]
        first_runs = _run_ids(db, first)
        second_runs = _run_ids(db, second)

        payload = {"run_ids": [*first_runs, *second_runs[:1], 999999]}
        resp = client.post("/api/v1/experiments/runs/batch-delete", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["count"] == 3
        assert data["requested"] == len(payload["run_ids"])

        for run_id in (*first_runs, *second_runs[:1]):
            assert db.get(ExperimentRun, run_id) is None
            assert all(not storage.exists(k) for k in seeded["run_artifact_keys"][run_id])
        # 两个实验都还在，只是各自少了一条运行
        assert db.get(Experiment, first) is not None
        assert db.get(Experiment, second) is not None
        assert _run_ids(db, first) == []
        assert len(_run_ids(db, second)) == 1
    finally:
        db.close()


def test_batch_delete_runs_with_empty_list_is_noop(client):
    resp = client.post("/api/v1/experiments/runs/batch-delete", json={"run_ids": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["count"] == 0


# ---------------------------------------------------------------------------
# 实验级批量删除
# ---------------------------------------------------------------------------
def test_batch_delete_experiments_removes_runs_and_artifacts(client):
    """批量删实验：「删除选中」要连带运行记录与产物一起清掉，不留悬挂引用。"""
    db = SessionLocal()
    storage = get_storage_service()
    try:
        seeded = _seed(db, "删实验批量回归", experiments=3, runs_per_experiment=2)
        keep, drop_a, drop_b = seeded["experiment_ids"]
        drop_runs = [*_run_ids(db, drop_a), *_run_ids(db, drop_b)]

        resp = client.post(
            "/api/v1/experiments/batch-delete",
            json={"experiment_ids": [drop_a, drop_b, 999999]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["count"] == 2

        db.expire_all()
        assert db.get(Experiment, drop_a) is None
        assert db.get(Experiment, drop_b) is None
        assert db.get(Experiment, keep) is not None
        for run_id in drop_runs:
            assert db.get(ExperimentRun, run_id) is None
            assert all(not storage.exists(k) for k in seeded["run_artifact_keys"][run_id])

        # 悬挂引用检查：没有 run 再指向已被删除的实验
        assert (
            _count(ExperimentRun, ~ExperimentRun.experiment_id.in_(select(Experiment.id))) == 0
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 已保存报告的批量删除
# ---------------------------------------------------------------------------
def _save_report_pair(storage, key: str) -> None:
    """落一份「正文 + 元数据副本」，复刻 save_report 的真实产出结构。"""
    storage.save(key, b'{"title": "t", "dataset": {}, "metadata": {}}')
    storage.save(key[: -len(".json")] + ".meta.json", b'{"title": "t"}')


def test_saved_report_batch_delete_clears_body_and_meta(client):
    """报告批量删除：正文与 .meta.json 都要清掉，缺失 key 只回执不报错。"""
    storage = get_storage_service()
    keys = ["reports/test-a.json", "reports/test-b.json", "reports/test-c.json"]
    for key in keys:
        _save_report_pair(storage, key)

    resp = client.post(
        "/api/v1/reports/saved/batch-delete",
        json={
            "keys": [
                keys[0],
                keys[1],
                "reports/does-not-exist.json",
                "data/other.json",  # 非法 key：不在 reports/ 下
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["count"] == 2
    assert data["deleted"] is False  # 有失败项时整体不宣称「全部成功」
    assert set(data["deleted_keys"]) == {keys[0], keys[1]}
    failed_keys = {item["key"] for item in data["failed"]}
    assert failed_keys == {"reports/does-not-exist.json", "data/other.json"}

    for key in (keys[0], keys[1]):
        assert not storage.exists(key)
        assert not storage.exists(key[: -len(".json")] + ".meta.json")
    # 未在请求里的报告不能被动到
    assert storage.exists(keys[2])
    assert storage.exists(keys[2][: -len(".json")] + ".meta.json")


def test_saved_report_batch_delete_rejects_path_traversal(client):
    """路径穿越的 key 必须被逐条拒绝，且不影响同批里的合法 key。"""
    storage = get_storage_service()
    storage.save("reports/keep.json", b'{"title": "keep"}')

    resp = client.post(
        "/api/v1/reports/saved/batch-delete",
        json={"keys": ["reports/../../etc/passwd.json", "reports/keep.json"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["count"] == 1
    assert data["deleted_keys"] == ["reports/keep.json"]
    assert any("非法" in item["error"] for item in data["failed"])


def test_saved_report_single_delete_repeats_as_404(client):
    """单条删除：不存在给 404、非法 key 给 400。

    批量接口刻意不同（逐条回执而非整批报错），这里把两条口径都钉住 ——
    否则日后「统一错误处理」时很容易把单条的 404 也改掉。

    回归点：同一份报告删第二次必须是 404，而不是 500。存储层在对象不存在时抛的是
    ``StorageException(code="NOT_FOUND")``，而接口最初只捕获 ``FileNotFoundError``，
    于是「再删一次」会变成 500 —— 只有真正连删两次才看得到的缺陷。
    """
    storage = get_storage_service()
    _save_report_pair(storage, "reports/repeat.json")

    first = client.delete("/api/v1/reports/saved", params={"key": "reports/repeat.json"})
    assert first.status_code == 200, first.text

    second = client.delete("/api/v1/reports/saved", params={"key": "reports/repeat.json"})
    assert second.status_code == 404, second.text

    missing = client.delete("/api/v1/reports/saved", params={"key": "reports/nope.json"})
    assert missing.status_code == 404, missing.text

    illegal = client.delete("/api/v1/reports/saved", params={"key": "not-a-report.json"})
    assert illegal.status_code == 400, illegal.text
