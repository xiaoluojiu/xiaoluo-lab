"""Phase 5（Prompt 079-082）Experiment 测试。"""

from __future__ import annotations

import polars as pl
import pytest
from app.core.exceptions import NotFoundException, ValidationException
from app.experiments.comparator import ExperimentComparator
from app.experiments.service import ExperimentService
from app.ml_engine.exceptions import MLEngineException
from app.models.experiment import Experiment
from app.models.experiment_run import ExperimentRun
from app.services.dataset_service import DatasetService


@pytest.fixture()
def dataset_service(db, storage) -> DatasetService:
    return DatasetService(db, storage)


@pytest.fixture()
def seeded_dataset(db, dataset_service: DatasetService):
    """创建含分类数据版本的数据集，返回 (dataset_id, version_row_id)。"""
    ds = dataset_service.create("cls-data")
    df = pl.DataFrame(
        {
            "a": [1.0, 1.5, 2.0, 8.0, 8.5, 9.0, 1.2, 8.2],
            "b": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5, 1.6, 8.6],
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )
    version = dataset_service.create_version(ds.id, df)
    return ds.id, version.id


@pytest.fixture()
def experiment_service(db, dataset_service) -> ExperimentService:
    return ExperimentService(db, dataset_service)


def _make_exp(service: ExperimentService, seeded_dataset, **overrides):
    dataset_id, version_id = seeded_dataset
    payload = dict(
        dataset_id=dataset_id,
        dataset_version_id=version_id,
        task="classification",
        model="logistic_regression",
        target_column="label",
        seed=42,
        description="demo",
    )
    payload.update(overrides)
    return service.create(**payload)


# ----------------------------------------------------------------------
# Prompt 079/080: 模型字段
# ----------------------------------------------------------------------
def test_experiment_models_roundtrip(db, seeded_dataset):
    dataset_id, version_id = seeded_dataset
    exp = Experiment(
        dataset_id=dataset_id,
        dataset_version_id=version_id,
        task="classification",
        model="logistic_regression",
        target_column="label",
        parameters={"C": 1.0},
        preprocessing={"scaling": {"method": "standard"}},
        seed=7,
    )
    db.add(exp)
    db.commit()
    run = ExperimentRun(experiment_id=exp.id, status="running", runtime=1.5)
    db.add(run)
    db.commit()
    loaded = db.get(Experiment, exp.id)
    assert loaded.parameters == {"C": 1.0}
    assert loaded.preprocessing == {"scaling": {"method": "standard"}}
    assert loaded.seed == 7
    assert loaded.runs[0].status == "running"
    assert loaded.runs[0].runtime == 1.5


# ----------------------------------------------------------------------
# Prompt 081: ExperimentService
# ----------------------------------------------------------------------
def test_create_experiment_validation(experiment_service, seeded_dataset):
    with pytest.raises(ValidationException, match="task"):
        _make_exp(experiment_service, seeded_dataset, task="forecasting")
    # 分类缺 target_column
    with pytest.raises(ValidationException, match="target_column"):
        _make_exp(experiment_service, seeded_dataset, target_column=None)
    # 聚类带 target_column
    with pytest.raises(ValidationException, match="聚类"):
        _make_exp(
            experiment_service,
            seeded_dataset,
            task="clustering",
            model="kmeans",
            target_column="label",
        )
    # 未注册模型
    with pytest.raises(MLEngineException, match="未注册"):
        _make_exp(experiment_service, seeded_dataset, model="xgboost")
    # 版本不属于数据集
    with pytest.raises(ValidationException, match="不匹配"):
        _make_exp(experiment_service, seeded_dataset, dataset_version_id=99999)


def test_run_classification_success(experiment_service, seeded_dataset):
    exp = _make_exp(
        experiment_service,
        seeded_dataset,
        preprocessing={"scaling": {"method": "standard"}},
    )
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    assert run.error is None
    assert run.runtime is not None and run.runtime > 0
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(run.metrics)
    assert run.artifacts["model"] == "logistic_regression"
    assert "label" in run.artifacts["target_column"]
    assert run.artifacts["train_rows"] == 6
    assert run.artifacts["test_rows"] == 2


def test_run_does_not_mutate_experiment_definition(experiment_service, seeded_dataset):
    """★ P1 回归：跑一次实验不能改写实验定义（Experiment ≠ Run）。

    历史缺陷：`_execute()` 在「自动剔除目标列」后执行 `exp.preprocessing = pp` 并随
    `db.commit()` 落库 —— 用户定义的实验配置被某一次运行永久改写，再次编辑或对比
    历史实验时看到的是被改过的 `excluded_columns`。

    正确语义：运行期的规范化只存在于本次运行的局部副本 + Run artifacts。
    """
    exp = _make_exp(
        experiment_service,
        seeded_dataset,
        preprocessing={"scaling": {"method": "standard"}, "excluded_columns": ["b", "label"]},
    )
    original_preprocessing = dict(exp.preprocessing)
    original_target = exp.target_column

    run = experiment_service.run(exp.id)
    assert run.status == "success", run.error

    # 1) 实验定义保持创建时的配置（以数据库为准，避免 ORM 内存态掩盖回写）
    experiment_service.db.refresh(exp)
    assert exp.preprocessing == original_preprocessing
    assert exp.target_column == original_target

    # 2) 运行期实际生效的排除列记录在 artifacts：目标列被剔除，其余保留
    assert run.artifacts["excluded_columns"] == ["b"]


def test_run_regression_success(db, storage, dataset_service, experiment_service):
    ds = dataset_service.create("reg-data")
    df = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]}
    )
    version = dataset_service.create_version(ds.id, df)
    exp = experiment_service.create(
        dataset_id=ds.id,
        dataset_version_id=version.id,
        task="regression",
        model="linear_regression",
        target_column="y",
    )
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    assert {"mae", "mse", "rmse", "r2"} <= set(run.metrics)


def test_run_clustering_success(db, storage, dataset_service, experiment_service):
    ds = dataset_service.create("clu-data")
    df = pl.DataFrame(
        {"a": [1.0, 1.2, 1.1, 9.0, 9.2, 9.1], "b": [1.0, 1.1, 0.9, 9.0, 9.1, 8.9]}
    )
    version = dataset_service.create_version(ds.id, df)
    exp = experiment_service.create(
        dataset_id=ds.id,
        dataset_version_id=version.id,
        task="clustering",
        model="kmeans",
        parameters={"n_clusters": 2},
        seed=0,
    )
    run = experiment_service.run(exp.id)
    assert run.status == "success"
    assert run.metrics["cluster_count"] == 2
    # 聚类不应注入 random_state 失败（kmeans 支持）
    assert run.artifacts["target_column"] is None


def test_run_failure_recorded(db, storage, dataset_service, experiment_service):
    """运行失败时：run 状态 failed + error 记录，实验行不消失。"""
    ds = dataset_service.create("bad-data")
    df = pl.DataFrame({"city": ["北京", "上海"], "label": [0, 1]})
    version = dataset_service.create_version(ds.id, df)
    exp = experiment_service.create(
        dataset_id=ds.id,
        dataset_version_id=version.id,
        task="classification",
        model="logistic_regression",
        target_column="label",
    )
    run = experiment_service.run(exp.id)
    assert run.status == "failed"
    assert run.error  # 有错误信息
    assert run.runtime is not None


def test_run_without_target_missing_column(experiment_service, seeded_dataset):
    exp = _make_exp(experiment_service, seeded_dataset, target_column="nope")
    run = experiment_service.run(exp.id)
    assert run.status == "failed"
    assert "目标列" in run.error


def test_get_list_and_runs(experiment_service, seeded_dataset):
    exp = _make_exp(experiment_service, seeded_dataset)
    experiment_service.run(exp.id)
    got = experiment_service.get(exp.id)
    assert got.id == exp.id
    items, total = experiment_service.list()
    assert total >= 1 and any(e.id == exp.id for e in items)
    runs = experiment_service.list_runs(exp.id)
    assert len(runs) == 1
    with pytest.raises(NotFoundException):
        experiment_service.get(99999)
    with pytest.raises(NotFoundException):
        experiment_service.get_run(99999)


# ----------------------------------------------------------------------
# Prompt 082: 比较器
# ----------------------------------------------------------------------
def test_comparator_best_and_param_diff(experiment_service, seeded_dataset):
    """两个不同参数的实验：比较 metrics/parameters/runtime。"""
    exp1 = _make_exp(experiment_service, seeded_dataset, parameters={"C": 1.0})
    exp2 = _make_exp(experiment_service, seeded_dataset, parameters={"C": 0.1})
    run1 = experiment_service.run(exp1.id)
    run2 = experiment_service.run(exp2.id)

    result = experiment_service.compare_runs([run1.id, run2.id])
    assert len(result.entries) == 2
    assert result.best.get("accuracy") in {run1.id, run2.id}
    assert result.parameter_diff["C"] == {run1.id: 1.0, run2.id: 0.1}
    assert sorted(result.runtime_ranking) == sorted([run1.id, run2.id])


def test_comparator_direction_aware():
    """mae 越低越好、f1 越高越好。"""

    class FakeRun:
        def __init__(self, rid, metrics, runtime):
            self.id = rid
            self.experiment_id = rid
            self.status = "success"
            self.metrics = metrics
            self.parameters = {}
            self.runtime = runtime
            self.experiment = None

    runs = [
        FakeRun(1, {"mae": 0.5, "f1": 0.8}, 2.0),
        FakeRun(2, {"mae": 0.1, "f1": 0.9}, 1.0),
    ]
    result = ExperimentComparator().compare(runs)
    assert result.best["mae"] == 2
    assert result.best["f1"] == 2
    assert result.runtime_ranking == [2, 1]


def test_comparator_ignores_failed_runs(db, experiment_service, seeded_dataset):
    """failed run 不参与最优指标竞争。"""
    exp = _make_exp(experiment_service, seeded_dataset)
    success_run = experiment_service.run(exp.id)
    failed_run = ExperimentRun(experiment_id=exp.id, status="failed", error="x")
    db.add(failed_run)
    db.commit()
    result = experiment_service.compare_runs([success_run.id, failed_run.id])
    # 只有 success run 参与 best 计算
    for rid in result.best.values():
        assert rid == success_run.id
    # 比较实验（取最近成功运行）
    cmp2 = experiment_service.compare_experiments([exp.id])
    assert cmp2.entries[0].run_id == success_run.id


# ----------------------------------------------------------------------
# 实验驾驶舱：列表要能回答「哪个实验效果更好」
# ----------------------------------------------------------------------
def test_list_with_latest_runs_returns_last_successful_run(
    experiment_service, seeded_dataset
):
    """列表带出的是**最近一次成功**运行的指标，失败运行不算数。"""
    exp = _make_exp(experiment_service, seeded_dataset)
    first = experiment_service.run(exp.id)
    # 再跑一次失败的运行：它的 id 更大，但不该顶掉成功的那条
    failed = ExperimentRun(
        experiment_id=exp.id, status="failed", error="boom"
    )
    experiment_service.db.add(failed)
    experiment_service.db.commit()

    items, latest, total = experiment_service.list_with_latest_runs()

    assert total == 1
    assert items[0].id == exp.id
    assert latest[exp.id].id == first.id
    assert latest[exp.id].status == "success"
    assert latest[exp.id].metrics  # 真实指标，不是占位


def test_list_with_latest_runs_is_empty_when_never_run(
    experiment_service, seeded_dataset
):
    _make_exp(experiment_service, seeded_dataset)

    items, latest, total = experiment_service.list_with_latest_runs()

    assert total == 1
    assert latest == {}  # 没跑过就是没有，不编造指标


def test_list_with_latest_runs_is_two_queries_not_n_plus_1(
    db, storage, dataset_service, experiment_service
):
    """5 个实验只应该有 2 条查询（实验 + 最近成功 run），不能每实验查一次。"""
    from sqlalchemy import event

    ds = dataset_service.create("n-plus-1")
    version = dataset_service.create_version(
        ds.id,
        pl.DataFrame(
            {
                "a": [1.0, 1.5, 2.0, 8.0, 8.5, 9.0, 1.2, 8.2],
                "label": [0, 0, 0, 1, 1, 1, 0, 1],
            }
        ),
    )
    for _ in range(5):
        exp = experiment_service.create(
            dataset_id=ds.id,
            dataset_version_id=version.id,
            task="classification",
            model="logistic_regression",
            target_column="label",
            seed=42,
        )
        experiment_service.run(exp.id)

    statements: list[str] = []

    def _before(conn, cursor, statement, params, context, executemany_or_many):
        statements.append(statement)

    event.listen(db.bind, "before_cursor_execute", _before)
    try:
        experiment_service.list_with_latest_runs(page=1, page_size=20)
    finally:
        event.remove(db.bind, "before_cursor_execute", _before)

    selects = [s for s in statements if s.strip().upper().startswith("SELECT")]
    assert len(selects) <= 3, f"查询次数异常：{len(selects)}（疑似 N+1）"
