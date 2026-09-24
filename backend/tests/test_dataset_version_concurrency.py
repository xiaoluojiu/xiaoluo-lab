"""DatasetVersion 并发正确性回归测试。

这些用例覆盖的是**曾经真实存在的竞态**，不是为了覆盖率凑数：

    latest = max(version)          # 请求 A 读到 v7
    next   = latest + 1            # 请求 B 也读到 v7
    insert(next)                   # A、B 都写 v8

在 ``storage_path UNIQUE`` 单独存在时这个竞态**拦不住**：storage key 由
``dataset_id`` + ``version`` 派生，version 相同则 key 相同，两个请求写的是同一个
文件，唯一约束在「后写覆盖先写」中完全失效。因此这里断言的是更强的命题：

    同一个 dataset 永远不能出现两个 version=8。

并发用例使用**文件型 SQLite**（而非 conftest 里的内存库 + StaticPool）：
内存库的 StaticPool 把所有连接压成一条，根本复现不出多连接竞态。
"""

from __future__ import annotations

import threading
from pathlib import Path

import polars as pl
import pytest
from app.core.database import Base
from app.core.exceptions import DatasetException, NotFoundException, ValidationException
from app.models.dataset_version import DatasetVersion
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

# 并发线程数。远大于「真实用户同时点两次」的量级，用于把竞态窗口压到必然命中。
CONCURRENCY = 8


@pytest.fixture()
def env(tmp_path: Path):
    """文件型 SQLite + 独立 storage，返回 (factory, storage, dataset_id)。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrency.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    storage = StorageService(LocalStorage(root=tmp_path / "storage"))

    session = factory()
    dataset = DatasetService(session, storage).create("concurrency")
    dataset_id = dataset.id
    session.close()

    return factory, storage, dataset_id


def _versions(factory, dataset_id: int) -> list[int]:
    session = factory()
    try:
        rows = session.scalars(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version)
        ).all()
        return [r.version for r in rows]
    finally:
        session.close()


# ---------------------------------------------------------
# 基线：串行创建
# ---------------------------------------------------------


def test_sequential_versions_are_monotonic(env):
    """普通创建 v1/v2/v3：版本号连续递增，父版本串成链。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    v1 = service.create_version(dataset_id, pl.DataFrame({"a": [1, 2, 3]}))
    v2 = service.create_version(dataset_id, pl.DataFrame({"a": [4, 5]}))
    v3 = service.create_version(dataset_id, pl.DataFrame({"a": [6]}))

    assert [v1.version, v2.version, v3.version] == [1, 2, 3]
    # 父版本正确性：v1 无父，v2 的父是 v1 的行 id，v3 的父是 v2 的行 id。
    assert v1.parent_version_id is None
    assert v2.parent_version_id == v1.id
    assert v3.parent_version_id == v2.id
    # 每个版本的存储 key 互不相同（不可变快照，不允许复用）
    assert len({v1.storage_path, v2.storage_path, v3.storage_path}) == 3
    session.close()


def test_explicit_parent_version_is_preserved(env):
    """显式传入 parent_version_id 时不得被 latest 覆盖。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    v1 = service.create_version(dataset_id, pl.DataFrame({"a": [1]}))
    service.create_version(dataset_id, pl.DataFrame({"a": [2]}))
    v3 = service.create_version(
        dataset_id, pl.DataFrame({"a": [3]}), parent_version_id=v1.id
    )

    assert v3.parent_version_id == v1.id
    session.close()


# ---------------------------------------------------------
# P0：并发创建
# ---------------------------------------------------------


def test_concurrent_create_version_never_duplicates(env):
    """并发创建：8 个线程同时建版本，必须恰好得到 v1..v8，无重号。"""
    factory, storage, dataset_id = env

    barrier = threading.Barrier(CONCURRENCY)
    lock = threading.Lock()
    outcomes: list[tuple[int, int]] = []  # (worker, version)
    errors: list[str] = []

    def worker(index: int) -> None:
        session = factory()
        service = DatasetService(session, storage)
        try:
            barrier.wait()
            row = service.create_version(
                dataset_id, pl.DataFrame({"tag": [f"w{index}"], "n": [index]})
            )
            with lock:
                outcomes.append((index, row.version))
        except Exception as exc:  # noqa: BLE001 - 收集并发异常供断言
            with lock:
                errors.append(f"worker {index}: {exc!r}")
        finally:
            session.close()

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # 核心断言：同 dataset 不可能出现两个相同的 version
    assert sorted(v for _, v in outcomes) == list(range(1, CONCURRENCY + 1))
    assert _versions(factory, dataset_id) == list(range(1, CONCURRENCY + 1))


def test_concurrent_versions_have_uncorrupted_snapshots(env):
    """并发下每个版本的快照内容必须属于它自己的写入者。

    这条断言针对的是「后写覆盖先写」：若两个线程拿到同一 version，
    它们会写同一个 ``v00000N.parquet``，其中一个的内容会被另一个静默覆盖。
    """
    factory, storage, dataset_id = env

    barrier = threading.Barrier(CONCURRENCY)
    lock = threading.Lock()
    outcomes: list[tuple[int, int]] = []

    def worker(index: int) -> None:
        session = factory()
        service = DatasetService(session, storage)
        try:
            barrier.wait()
            row = service.create_version(dataset_id, pl.DataFrame({"tag": [f"w{index}"]}))
            with lock:
                outcomes.append((index, row.version))
        finally:
            session.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    verify = factory()
    service = DatasetService(verify, storage)
    for worker_index, version in outcomes:
        frame = service.load_version(dataset_id, version)
        assert frame["tag"].to_list() == [f"w{worker_index}"], (
            f"version {version} 的快照被别的写入者覆盖了"
        )
    verify.close()


def test_concurrent_reservation_conflict_is_retried(monkeypatch, env):
    """并发 version reservation：抢号失败的一方必须重算，而不是抛给用户。

    这里把重试次数压到 1，再人为制造一次「第一次 flush 撞唯一约束」，
    验证循环确实会重新读 max 并取下一个号，而不是把 IntegrityError 抛出去。
    """
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    real_flush = session.flush
    calls = {"n": 0}

    def flaky_flush(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # 消息必须是真实 SQLite 形态：_reserve_version 靠它区分「版本竞争」与
            # 其它约束冲突，伪造一条不带列名的消息会让重试逻辑被误判成真错误。
            raise IntegrityError(
                "stmt",
                {},
                Exception(
                    "UNIQUE constraint failed: "
                    "dataset_versions.dataset_id, dataset_versions.version"
                ),
            )
        return real_flush(*args, **kwargs)

    monkeypatch.setattr(session, "flush", flaky_flush)

    row = service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    # 版本号仍然是 1：说明第一次撞约束后回滚并重算，而不是把错误抛给用户。
    # （计数器 >=2 而非 ==2：最终 commit 内部还会再 flush 一次。）
    assert row.version == 1
    assert calls["n"] >= 2
    session.close()


# ---------------------------------------------------------
# 失败路径：不留半成品
# ---------------------------------------------------------


def test_storage_conflict_does_not_delete_existing_snapshot(env):
    """storage 冲突：目标 key 已存在时拒绝创建，且不动已有快照。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    # 手工占据 v1 的存储位置（模拟历史遗留 / 崩溃残留）
    occupied = storage.local_path(f"datasets/{dataset_id}/v000001.parquet")
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_bytes(b"pre-existing")

    with pytest.raises(DatasetException) as exc:
        service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    assert exc.value.code == "VERSION_CONFLICT"
    # 已存在的快照必须完好：冲突保护不能反过来把旧版本的数据删掉
    assert occupied.read_bytes() == b"pre-existing"
    assert _versions(factory, dataset_id) == []
    session.close()


def test_db_commit_failure_leaves_no_orphan_snapshot(env):
    """DB 写入失败：不能留下「磁盘有文件、库里没记录」的孤儿快照。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    def boom():
        raise RuntimeError("db down")

    monkeypatch_commit = boom
    session.commit = monkeypatch_commit  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="db down"):
        service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    snapshot = storage.local_path(f"datasets/{dataset_id}/v000001.parquet")
    assert not snapshot.exists(), "DB 失败后残留了孤儿快照"
    assert _versions(factory, dataset_id) == []
    session.close()


def test_failure_after_commit_keeps_snapshot(env):
    """★ commit-before / commit-after 边界：commit 成功后的异常不得删除快照。

    ``stage_version`` 里有两条方向相反的一致性要求：
      - commit **之前**失败 → 占位行 rollback + 回收快照，本次创建整体消失（正确）；
      - commit **之后**失败 → 版本行已经落库，此时删快照会留下
        「库里有版本行、磁盘上没有文件」的孤儿版本，之后每次读取都失败。

    旧实现只用一个 ``promoted`` 标记决定要不要回收快照，于是 commit 成功但随后
    ``db.refresh()`` 抛异常时也会走进「回滚 + 删快照」—— 而 rollback 对已提交的
    行无效，不一致性被藏进了磁盘。
    """
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    def refresh_boom(*_args, **_kwargs):
        raise RuntimeError("refresh failed after commit")

    session.refresh = refresh_boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="refresh failed after commit"):
        service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    # 版本行已提交，rollback 撤不掉它 —— 所以快照必须留着，两者才是一致的
    assert _versions(factory, dataset_id) == [1]
    snapshot = storage.local_path(f"datasets/{dataset_id}/v000001.parquet")
    assert snapshot.exists(), "commit 之后的异常删除了已提交版本的快照，会产生孤儿版本行"
    session.close()


def test_staging_failure_creates_no_version(env):
    """快照产出失败：暂存文件被清掉，且不产生任何版本行。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    with pytest.raises(RuntimeError, match="extract failed"):
        with service.stage_version(dataset_id) as staging:
            staging.staging_path.write_bytes(b"half-written")
            raise RuntimeError("extract failed")

    assert not staging.staging_path.exists(), "失败后暂存文件未清理"
    assert _versions(factory, dataset_id) == []
    session.close()


def test_incomplete_staging_is_rejected(env):
    """未填行列数的暂存不得提交（否则会写下 row_count=0 的错误版本行）。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    with pytest.raises(DatasetException) as exc:
        with service.stage_version(dataset_id) as staging:
            staging.staging_path.write_bytes(b"whatever")

    assert exc.value.code == "STAGING_INCOMPLETE"
    assert _versions(factory, dataset_id) == []
    session.close()


def test_duplicate_version_is_rejected_by_database(env):
    """重复 version：绕过 Service 直接插库也必须被唯一约束挡下。"""
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    session.add(
        DatasetVersion(
            dataset_id=dataset_id,
            version=1,
            parent_version_id=None,
            storage_path="datasets/1/v999999.parquet",  # 绕开 storage_path 唯一约束
            format="parquet",
            row_count=0,
            column_count=0,
            schema_json={},
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.close()


def test_unique_constraint_exists(env):
    """唯一约束必须真实存在于 schema 上（防止迁移被漏掉/回滚）。"""
    factory, _, _ = env
    session = factory()
    table = DatasetVersion.__table__
    names = {
        c.name if not isinstance(c, str) else c
        for constraint in table.constraints
        for c in getattr(constraint, "columns", [])
    }
    assert "dataset_id" in names and "version" in names

    unique = [
        c
        for c in table.constraints
        if c.__class__.__name__ == "UniqueConstraint"
        and {col.name for col in c.columns} == {"dataset_id", "version"}
    ]
    assert unique, "dataset_versions 缺少 (dataset_id, version) 唯一约束"
    session.close()


# ---------------------------------------------------------
# P1：血缘（parent_version_id）必须同数据集
# ---------------------------------------------------------


def test_cross_dataset_parent_is_rejected(env):
    """父版本必须属于同一个数据集。

    版本链是血缘：B 的版本声明自己源自 A 的 v1，回溯输入时会直接指到另一份数据上。
    这种「看起来合理」的错误血缘比直接报错更危险，因此必须显式拒绝。
    """
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    other = service.create("另一个数据集")
    v1_a = service.create_version(dataset_id, pl.DataFrame({"a": [1]}))
    v1_b = service.create_version(other.id, pl.DataFrame({"b": [1]}))

    # B 的版本不能把 A 的版本当父节点
    with pytest.raises(ValidationException) as exc:
        service.create_version(
            other.id, pl.DataFrame({"b": [2]}), parent_version_id=v1_a.id
        )
    assert "数据集" in str(exc.value.message)
    assert str(other.id) in str(exc.value.details)

    # 父版本不存在：明确 NotFound，而不是静默当 NULL 处理
    with pytest.raises(NotFoundException):
        service.create_version(dataset_id, pl.DataFrame({"a": [9]}), parent_version_id=999999)

    # 同数据集的正常血缘不受影响
    v2_a = service.create_version(dataset_id, pl.DataFrame({"a": [2]}), parent_version_id=v1_a.id)
    assert v2_a.parent_version_id == v1_a.id
    assert v1_b.parent_version_id is None
    session.close()


# ---------------------------------------------------------
# P1：IntegrityError 不能被一律当成版本竞争
# ---------------------------------------------------------


def test_version_unique_conflict_is_retried(env):
    """(dataset_id, version) 冲突 = 版本号被抢 ⇒ 必须换号重试，而不是报错。

    用受控的 ``latest_version`` 伪造一次「另一位写者刚拿走 v2」的竞争窗口，
    让重试路径可确定性复现（不依赖真实线程时序）。
    """
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    v1 = service.create_version(dataset_id, pl.DataFrame({"a": [1]}))
    v2 = service.create_version(dataset_id, pl.DataFrame({"a": [2]}))

    real_latest = service.latest_version
    calls = {"n": 0}

    def fake_latest(ds_id):
        calls["n"] += 1
        # 第一次仍返回 v1（于是算出 version=2，与已存在的 v2 撞车）
        return v1 if calls["n"] == 1 else real_latest(ds_id)

    service.latest_version = fake_latest  # type: ignore[method-assign]
    v3 = service.create_version(dataset_id, pl.DataFrame({"a": [3]}))

    assert v3.version == 3, "冲突后应换到下一个可用版本号"
    assert calls["n"] >= 2, "唯一约束冲突必须触发重试，而不是直接失败"
    assert v2.version == 2
    session.close()


def test_non_version_integrity_error_is_not_retried(env, monkeypatch):
    """非版本冲突的 IntegrityError 必须原样抛出，不能被伪装成并发冲突。

    让 ``_version_key`` 返回 None ⇒ storage_path 触发 NOT NULL。这与版本号无关，
    重试 5 次也永远不会成功；若被当成版本竞争，调用方只会看到
    ``VERSION_RESERVATION_CONFLICT``，真正的数据库原因彻底丢失。
    """
    factory, storage, dataset_id = env
    session = factory()
    service = DatasetService(session, storage)

    monkeypatch.setattr(DatasetService, "_version_key", lambda self, ds_id, version: None)

    with pytest.raises(DatasetException) as exc:
        service.create_version(dataset_id, pl.DataFrame({"a": [1]}))

    assert exc.value.code == "DATASET_VERSION_CONSTRAINT_FAILED", (
        f"应暴露真实的数据库约束错误，实际：{exc.value.code}"
    )
    assert "storage_path" in str(exc.value.details), "原始原因必须留在 details 里"
    session.close()
