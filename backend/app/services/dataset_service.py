"""DatasetService。

负责：

- Dataset 创建
- Dataset 查询
- Dataset 列表
- Dataset 元数据修改
- DatasetVersion 创建
- DatasetVersion 查询
- DatasetVersion 加载
- Dataset 删除

核心原则：

1. 原始数据不可覆盖
2. 每次数据修改都产生新的 DatasetVersion
3. DatasetVersion 使用 Parquet 快照
4. 数据库只保存元数据
5. DataFrame 不直接存入数据库

大数据接入（本文件的性能关键路径）
----------------------------------
两条写路径、两条读路径，共同点是**都不再让整表常驻内存**：

- 写①``create_version(df)``：DataFrame 已在内存（数据处理/建模产出），
  直接 ``write_parquet`` 到临时文件后 ``promote`` 进 Storage，省掉
  ``io.BytesIO`` 里的那一份完整拷贝。
- 写②``create_version_from_source(...)``：源文件在磁盘（上传 / 数据库抽取），
  走 ``app.data_engine.ingest`` 流式转为 Parquet，内存占用与文件体积无关。
- 读①``load_version(...)``：优先按**本地路径**读（Parquet 内存映射），
  避免先把整个文件读成 bytes 再解码。
- 读②``scan_version(...)``：返回 LazyFrame，让「只读几列」「按范围过滤」
  真正下推到 Parquet 层，而不是先物化再过滤。
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    DatasetException,
    NotFoundException,
    ValidationException,
)
from app.data_engine import ingest as ingest_engine
from app.data_engine.cache import (
    VersionFrameCache,
    get_version_frame_cache,
)
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.storage.service import StorageService

# 是否启用版本快照缓存。DatasetVersion 不可变，因此缓存不会失效，只需淘汰。
_VERSION_CACHE_ENABLED = True
# 列裁剪阈值：请求的列数低于总列数的该比例时才值得重新读盘（否则全量解码更划算）。
_COLUMN_PRUNE_RATIO = 0.6
# 版本号预留的最大重试次数。只有并发请求撞到同一版本号时才会 >1，
# 正常路径循环一次即返回。
_MAX_VERSION_RESERVE_ATTEMPTS = 5

#: (dataset_id, version) 唯一约束名。判定 IntegrityError 是否为「版本号竞争」时
#: 以它为准 —— 不能把所有 IntegrityError 都当成版本冲突（见 `_is_version_unique_conflict`）。
_VERSION_UNIQUE_CONSTRAINT = "uq_dataset_versions_dataset_id_version"


def _is_version_unique_conflict(exc: IntegrityError) -> bool:
    """判定 IntegrityError 是否来自 ``(dataset_id, version)`` 唯一约束冲突。

    为什么要区分：``_reserve_version`` 的 INSERT 可能触发**任何**约束冲突 ——
    外键、``storage_path`` 唯一、NOT NULL、CHECK。把它们一律当成「版本号被抢」会
    静默重试 5 次后抛 ``VERSION_RESERVATION_CONFLICT``，把真正的数据库错误伪装成
    并发冲突：调用方以为「再试一次就好」，实际永远不可能成功，且原始原因丢失。

    判定顺序：
    1. 驱动暴露的结构化约束名（PostgreSQL 的 ``diag.constraint_name``）——最可靠；
    2. 异常文本里出现约束名；
    3. 退化到 SQLite 的 ``UNIQUE constraint failed: <表>.<列>[, <表>.<列>]`` 形态，
       只认 dataset_versions 表上「(dataset_id, version) 两列」或「派生列
       storage_path」这两种同源冲突，其它表/列的唯一冲突一律不算。
    """
    orig = getattr(exc, "orig", None)
    constraint_name = getattr(getattr(orig, "diag", None), "constraint_name", None)
    if constraint_name:
        return str(constraint_name) == _VERSION_UNIQUE_CONSTRAINT

    text = str(orig or exc)
    if _VERSION_UNIQUE_CONSTRAINT in text:
        return True
    if "UNIQUE constraint failed" not in text:
        return False
    # SQLite 形态：UNIQUE constraint failed: dataset_versions.<列>[, dataset_versions.<列>]
    # storage_path 也算同源：它由 (dataset_id, version) 派生，两个写者算出同一个
    # version 时存储 key 必然相同；且 SQLite 在两条约束同时违反时**先**报列级的
    # storage_path，所以漏掉它就会把并发建版本误判成「约束校验失败」而中止。
    return (
        ("dataset_versions.dataset_id" in text and "dataset_versions.version" in text)
        or "dataset_versions.storage_path" in text
    )


def _lazy_scan_enabled() -> bool:
    """是否允许 scan_parquet 懒执行（配置层异常时按「允许」处理）。"""
    try:
        return bool(settings.DATASET_LAZY_SCAN_ENABLED)
    except Exception:  # pragma: no cover
        return True


@dataclass
class VersionStaging:
    """``stage_version()`` 交给产出方的暂存句柄。

    产出方只关心三件事：往 ``staging_path`` 写 Parquet、填行列数、
    （可选）填 schema。存储 key 与版本号由 DatasetService 负责，
    产出方无需知道存储布局。
    """

    staging_path: Path
    version: int
    storage_key: str
    row_count: int | None = None
    column_count: int | None = None
    schema_json: dict = field(default_factory=dict)
    version_row: DatasetVersion | None = None

    def set_stats(
        self,
        *,
        row_count: int,
        column_count: int,
        schema_json: dict | None = None,
    ) -> None:
        self.row_count = int(row_count)
        self.column_count = int(column_count)
        if schema_json:
            self.schema_json = dict(schema_json)


class DatasetService:
    """Dataset 业务服务。"""

    def __init__(
        self,
        db: Session,
        storage: StorageService,
        cache: VersionFrameCache | None = None,
    ) -> None:
        self.db = db
        self.storage = storage
        # 缓存默认取进程级单例；测试可注入独立实例以避免相互影响。
        self.cache = cache if cache is not None else get_version_frame_cache()

    # =========================================================
    # Dataset
    # =========================================================

    def create(
        self,
        name: str,
        description: str = "",
        source_file_id: int | None = None,
    ) -> Dataset:
        """创建逻辑数据集。"""

        name = name.strip()

        if not name:
            raise DatasetException(
                "dataset name cannot be empty",
                code="DATASET_NAME_REQUIRED",
            )

        dataset = Dataset(
            name=name,
            description=description,
            source_file_id=source_file_id,
        )

        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)

        return dataset

    def get(
        self,
        dataset_id: int,
    ) -> Dataset:
        """获取数据集。"""

        dataset = self.db.get(
            Dataset,
            dataset_id,
        )

        if dataset is None:
            raise NotFoundException(
                "dataset not found",
                details={
                    "dataset_id": dataset_id,
                },
            )

        return dataset

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Dataset], int]:
        """分页获取数据集。"""

        if page < 1:
            page = 1

        if page_size < 1:
            page_size = 20

        stmt = (
            select(Dataset)
            .order_by(Dataset.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )

        items = list(self.db.scalars(stmt))

        total = self.db.scalar(
            select(func.count(Dataset.id))
        ) or 0

        return items, total

    def update_metadata(
        self,
        dataset_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Dataset:
        """修改数据集元数据。

        不修改 DatasetVersion。
        """

        dataset = self.get(dataset_id)

        if name is not None:
            name = name.strip()

            if not name:
                raise DatasetException(
                    "dataset name cannot be empty",
                    code="DATASET_NAME_REQUIRED",
                )

            dataset.name = name

        if description is not None:
            dataset.description = description

        self.db.commit()
        self.db.refresh(dataset)

        return dataset

    # =========================================================
    # DatasetVersion
    # =========================================================

    def latest_version(
        self,
        dataset_id: int,
    ) -> DatasetVersion | None:
        """返回最新版本行；没有版本时返回 None。

        ``with_for_update()`` 在 PostgreSQL 上对命中的行加排他锁，与
        ``_reserve_version`` 的占位 INSERT 形成完整互斥；SQLite 方言不渲染
        FOR UPDATE（无副作用），唯一性由唯一约束兜底，因此两种库都成立。
        """

        stmt = (
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id
                == dataset_id,
            )
            .order_by(
                DatasetVersion.version.desc(),
            )
            .limit(1)
            .with_for_update()
        )

        return self.db.scalars(stmt).first()

    def create_version(
        self,
        dataset_id: int,
        df: pl.DataFrame,
        parent_version_id: int | None = None,
    ) -> DatasetVersion:
        """创建不可变 DatasetVersion（数据已在内存时使用）。

        DataFrame -> 暂存 Parquet -> promote 进 Storage -> 补全版本行。

        改造要点：不再用 ``io.BytesIO`` 攒一份完整 Parquet 字节串。对 1 GB 级
        数据，那等于在「DataFrame 常驻」之外**再多占一份压缩后体积的内存峰值**，
        而且写入 Storage 时还要再拷一次。改为落暂存文件后 ``promote``
        （同盘为原子 rename，零拷贝）。
        """

        self.get(dataset_id)

        if not isinstance(
            df,
            pl.DataFrame,
        ):
            raise DatasetException(
                "dataset version requires a Polars DataFrame",
                code="INVALID_DATAFRAME",
            )

        with self.stage_version(dataset_id, parent_version_id) as staging:
            df.write_parquet(staging.staging_path, compression="zstd")
            staging.set_stats(
                row_count=df.height,
                column_count=df.width,
                schema_json={name: str(dtype) for name, dtype in df.schema.items()},
            )

        return staging.version_row

    def create_version_from_source(
        self,
        dataset_id: int,
        source_path: str | Path,
        *,
        fmt: str | None = None,
        options: dict | None = None,
        parent_version_id: int | None = None,
    ) -> tuple[DatasetVersion, ingest_engine.IngestResult]:
        """从**磁盘上的源文件**创建版本（上传 / 数据库抽取走这里）。

        与 ``create_version`` 的区别：数据从未被完整读进内存——``ingest`` 用
        Polars 流式引擎「读一批、转一批、写一批」，常驻内存与文件体积无关。

        返回 ``(DatasetVersion, IngestResult)``；后者携带 strategy / 行数 /
        耗时 / 吞吐 / 告警，供 API 与前端展示「这份数据是怎么进来的」。
        """

        self.get(dataset_id)

        source_path = Path(str(source_path))
        if not source_path.is_file():
            raise DatasetException(
                "source file not found",
                code="SOURCE_NOT_FOUND",
                details={"source": str(source_path)},
            )

        with self.stage_version(dataset_id, parent_version_id) as staging:
            result = ingest_engine.ingest_to_parquet(
                source_path, staging.staging_path, fmt=fmt, options=options
            )
            staging.set_stats(
                row_count=result.row_count,
                column_count=result.column_count,
                schema_json=result.schema,
            )

        return staging.version_row, result

    # ---------------------------------------------------------
    # 写路径内部工具
    # ---------------------------------------------------------

    @contextmanager
    def stage_version(
        self,
        dataset_id: int,
        parent_version_id: int | None = None,
    ) -> Iterator["VersionStaging"]:
        """预留版本号 + 提供暂存路径的上下文管理器（**所有写路径的唯一入口**）。

        三条写路径（内存 DataFrame / 磁盘源文件 / 数据库连接器流式抽取）
        都收敛到这里，避免「预定版本号 → 写快照 → 登记版本行」这套并发敏感的
        逻辑在多个地方各写一遍。

        用法::

            with service.stage_version(dataset_id) as staging:
                result = extract(..., staging.staging_path)
                staging.set_stats(row_count=result.row_count, ...)
            # 无异常退出即提交（promote 快照 + 补全版本行 + commit）
            # 有异常则整体回滚：快照与版本行一起消失，不留下半成品
        """
        self.get(dataset_id)

        row = self._reserve_version(dataset_id, parent_version_id)
        storage_key = row.storage_path
        tmp_path = self._temp_target(storage_key)

        staging = VersionStaging(
            staging_path=tmp_path,
            version=row.version,
            storage_key=storage_key,
        )

        # 只有 promote 成功过，快照才算"本次创建的"，异常时才有权回收。
        # 否则 ``VERSION_CONFLICT`` 保护的可能是**已存在的旧版本快照**。
        promoted = False
        # ★★ 提交边界（commit-before / commit-after）：
        #   - commit 之前失败 → 占位行 rollback + 回收快照，本次创建整体消失（正确）；
        #   - commit 之后失败 → 版本行**已经落库**，此时删快照会留下
        #     「库里有版本行、磁盘上没有文件」的孤儿版本，后续读取必然失败。
        #     旧实现只靠 `promoted` 一个标记，把 commit 之后的 ``db.refresh`` 异常也
        #     当成「没提交成功」去删快照，正是这条边界没划清导致的。
        #   commit 之后的异常只能如实上抛（版本行与快照都在，是可查可修的一致状态），
        #   绝不能靠删文件把不一致藏起来。
        committed = False

        try:
            yield staging

            if staging.row_count is None or staging.column_count is None:
                raise DatasetException(
                    "暂存版本未提供行列数，拒绝提交",
                    code="STAGING_INCOMPLETE",
                    details={"storage_key": storage_key},
                )

            self._promote_or_save(storage_key, tmp_path)
            promoted = True

            row.row_count = int(staging.row_count)
            row.column_count = int(staging.column_count)
            row.schema_json = staging.schema_json or {}

            self.db.commit()
            committed = True
            self.db.refresh(row)
            staging.version_row = row
        except BaseException:
            self._abort_version(storage_key, drop_snapshot=promoted and not committed)
            raise
        finally:
            # 提交成功后 promote 已把暂存文件移走；失败/异常时在这里兜底清理。
            tmp_path.unlink(missing_ok=True)

    def _reserve_version(
        self,
        dataset_id: int,
        parent_version_id: int | None,
    ) -> DatasetVersion:
        """占用下一个版本号：立即 INSERT 一行占位记录，**不提交**。

        这是 DatasetVersion 并发正确性的支点，三个设计要点：

        1. **不用 Python 锁**。进程内锁挡不住多进程 / 多副本部署，而版本号唯一
           属于数据正确性，必须落在数据库上。
        2. **由 ``uq_dataset_versions_dataset_id_version`` 裁决**。并发双方会算出
           同一个 version，但只有一方的 INSERT 成功；另一方收到 IntegrityError
           后回滚重算，自然拿到下一个号。相比「先 SELECT max 再 INSERT」的读后写，
           把判定权交给数据库后无需依赖 SQLite 的库级写锁语义，切到 PostgreSQL
           同样成立。
        3. **占位行的 INSERT 立即开启写事务**并持有到快照写完一起提交，于是
           「预定版本号 → 写快照 → 提交」是一次原子操作：并发者要么看到提交后的
           新版本（拿到 +1），要么被唯一约束挡下重试；不可能出现两个请求同时写
           同一个 ``v000008.parquet``、后写者覆盖先写者快照的情况。

        返回的 DatasetVersion 尚未填行列数；占位行未提交，对其它连接不可见，
        因此不存在「半成品版本」被外部读到。
        """
        # 血缘校验放在循环外：它与版本号无关，先失败可以省掉一次无谓的 INSERT，
        # 也避免占位行回滚把真正的校验错误冲掉。
        if parent_version_id is not None:
            self._assert_valid_parent(dataset_id, parent_version_id)

        for _ in range(_MAX_VERSION_RESERVE_ATTEMPTS):
            latest = self.latest_version(dataset_id)

            version = latest.version + 1 if latest is not None else 1
            parent = (
                parent_version_id
                if parent_version_id is not None
                else (latest.id if latest is not None else None)
            )

            row = DatasetVersion(
                dataset_id=dataset_id,
                version=version,
                parent_version_id=parent,
                storage_path=self._version_key(dataset_id, version),
                format="parquet",
                row_count=0,
                column_count=0,
                schema_json={},
            )

            try:
                self.db.add(row)
                # flush 而非 commit：INSERT 立刻执行（唯一约束此刻裁决），
                # 但整个事务仍由 stage_version 在快照写完后统一提交。
                self.db.flush()
                return row
            except IntegrityError as exc:
                self.db.rollback()
                # 只有「版本号被并发者抢走」才重算版本号重试；其它约束冲突是
                # 真错误，重试只会把原因伪装成并发冲突。
                if not _is_version_unique_conflict(exc):
                    raise DatasetException(
                        "创建数据版本时数据库约束校验失败",
                        code="DATASET_VERSION_CONSTRAINT_FAILED",
                        details={
                            "dataset_id": dataset_id,
                            "version": version,
                            "reason": str(getattr(exc, "orig", exc))[:300],
                        },
                    ) from exc
                continue

        raise DatasetException(
            "无法为数据集分配版本号：并发冲突过多",
            code="VERSION_RESERVATION_CONFLICT",
            details={
                "dataset_id": dataset_id,
                "attempts": _MAX_VERSION_RESERVE_ATTEMPTS,
            },
        )

    def _assert_valid_parent(self, dataset_id: int, parent_version_id: int) -> None:
        """校验父版本：必须存在，且必须属于同一个数据集。

        版本链是 DatasetVersion 的血缘。跨数据集挂父节点会让「v3 由 v2 派生」这条
        断言失去意义：数据集 B 的版本在物理上是另一份数据，却声明自己源自数据集 A，
        回溯来源时会直接指到错误的输入上。这里显式拒绝，宁可失败也不留错误血缘。
        """
        parent = self.db.get(DatasetVersion, parent_version_id)
        if parent is None:
            raise NotFoundException(
                f"父版本 {parent_version_id} 不存在",
                details={"parent_version_id": parent_version_id, "dataset_id": dataset_id},
            )
        if int(parent.dataset_id) != int(dataset_id):
            raise ValidationException(
                f"父版本 {parent_version_id} 属于数据集 {parent.dataset_id}，"
                f"不能作为数据集 {dataset_id} 的版本父节点",
                details={
                    "parent_version_id": parent_version_id,
                    "parent_dataset_id": int(parent.dataset_id),
                    "dataset_id": dataset_id,
                },
            )

    def _abort_version(
        self,
        storage_key: str,
        *,
        drop_snapshot: bool,
    ) -> None:
        """回滚版本创建：撤销未提交的占位行，并回收本次已提升的快照。

        两条方向相反的一致性要求在这里汇合：

        - DB 写入失败 -> 不能留下孤儿快照（磁盘有文件、库里没记录）
        - 快照写入失败 -> 不能留下错误版本行（库里有记录、磁盘没文件）

        占位行靠 ``rollback`` 撤销（它从未提交）；快照只有在确认是本次
        promote 出来的之后才删除，避免误删已有版本的快照。
        """
        try:
            self.db.rollback()
        except Exception:
            pass

        if not drop_snapshot:
            return

        try:
            if self.storage.exists(storage_key):
                self.storage.delete(storage_key)
        except Exception:
            pass

    @staticmethod
    def _version_key(dataset_id: int, version: int) -> str:
        return f"datasets/{dataset_id}/v{version:06d}.parquet"

    def _temp_target(self, storage_key: str) -> Path:
        """为即将写入的存储 key 分配一个临时落点。

        优先与目标**同目录**（同盘 ⇒ ``os.replace`` 原子且零拷贝）；
        后端不支持本地路径时退回系统临时目录（随后走 ``save_stream`` 拷贝）。
        """
        local = self.storage.local_path(storage_key)

        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            return local.with_name(f".{local.name}.writing")

        fd, raw = tempfile.mkstemp(prefix="xiaoluo_ver_", suffix=".parquet")
        # mkstemp 已经创建了文件；这里只需要路径，句柄立即关闭。
        os.close(fd)
        return Path(raw)

    def _promote_or_save(self, storage_key: str, tmp_path: Path) -> None:
        """把临时文件落到 Storage（存在冲突时拒绝，保持「版本不可覆盖」语义）。"""
        if self.storage.exists(storage_key):
            raise DatasetException(
                "version snapshot already exists",
                code="VERSION_CONFLICT",
                details={"storage_path": storage_key},
            )

        target = self.storage.local_path(storage_key)
        if target is not None:
            self.storage.promote(storage_key, tmp_path)
            return

        self.storage.save_stream(storage_key, tmp_path)

    def get_versions(
        self,
        dataset_id: int,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[
        list[DatasetVersion],
        int,
    ]:
        """分页获取数据集版本。"""

        self.get(dataset_id)

        stmt = (
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id
                == dataset_id,
            )
            .order_by(
                DatasetVersion.version.desc(),
            )
            .limit(page_size)
            .offset((page - 1) * page_size)
        )

        items = list(self.db.scalars(stmt))

        total = self.db.scalar(
            select(func.count(DatasetVersion.id)).where(
                DatasetVersion.dataset_id == dataset_id,
            )
        ) or 0

        return items, total

    def get_version_row(
        self,
        dataset_id: int,
        version: int | None = None,
    ) -> DatasetVersion:
        """获取指定版本。

        version=None 时获取最新版本。
        """

        self.get(dataset_id)

        if version is None:
            current = self.latest_version(
                dataset_id
            )

            if current is None:
                raise NotFoundException(
                    "dataset has no versions",
                    details={
                        "dataset_id": dataset_id,
                    },
                )

            return current

        stmt = (
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id
                == dataset_id,
                DatasetVersion.version
                == version,
            )
            .limit(1)
        )

        result = self.db.scalars(stmt).first()

        if result is None:
            raise NotFoundException(
                "dataset version not found",
                details={
                    "dataset_id": dataset_id,
                    "version": version,
                },
            )

        return result

    def load_version(
        self,
        dataset_id: int,
        version: int | None = None,
        *,
        columns: list[str] | None = None,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """读取指定 DatasetVersion。

        优化点：

        1. ``use_cache`` 命中 ``VersionFrameCache``（key = dataset_id + version）时
           直接返回内存副本，跳过「读盘 + Parquet 全量解码」。DatasetVersion 不可变，
           因此缓存永不失效，只需 LRU 淘汰。
        2. ``columns`` 指定需要的列时，走 Parquet **列裁剪** 读取，
           I/O 与解码量按列数等比下降（只分析 3 列不必解全表）。

        注意：``columns`` 存在时会跳过缓存（返回的是列子集，与缓存的全量表语义不同），
        避免把子集写进缓存污染后续全量请求。
        """

        version_row = self.get_version_row(
            dataset_id,
            version,
        )

        # 列裁剪路径：直接按需读列，不经过缓存。
        if columns and self._should_prune_columns(version_row, columns):
            return self._read_version_columns(version_row, columns)

        if use_cache and _VERSION_CACHE_ENABLED:
            cached = self.cache.get(dataset_id, version_row.version)
            if cached is not None:
                return cached

        frame = self._read_version(version_row)

        if use_cache and _VERSION_CACHE_ENABLED:
            self.cache.put(dataset_id, version_row.version, frame)

        return frame

    def _should_prune_columns(
        self,
        version_row: DatasetVersion,
        columns: list[str],
    ) -> bool:
        """列数明显少于总列数时才值得重新读盘（否则命中缓存更快）。"""
        total = int(version_row.column_count or 0)
        if total <= 0:
            return True
        wanted = len({c for c in columns if c})
        return 0 < wanted < total * _COLUMN_PRUNE_RATIO

    # ---------------------------------------------------------
    # 读路径：本地直通 / 懒加载
    # ---------------------------------------------------------

    def version_local_path(
        self,
        version_row: DatasetVersion,
    ) -> Path | None:
        """版本快照的本地绝对路径；后端不支持本地直通时返回 None。

        拿到路径的意义：Parquet 可以**内存映射**读取并做谓词/投影下推，
        跳过「整文件读成 bytes → 再从 bytes 解码」这一份多余的完整拷贝。
        """
        try:
            return self.storage.local_path(version_row.storage_path)
        except Exception:  # noqa: BLE001 - 探测失败按不支持处理
            return None

    def scan_version(
        self,
        dataset_id: int,
        version: int | None = None,
        *,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """返回版本快照的懒执行帧（**不物化数据，不经过内存缓存**）。

        适用于「大数据集只需要少量列 / 少量行」的只读分析：过滤与投影会被
        Polars 优化器下推到 Parquet 扫描层，只需解码命中的行组。

        注意：懒帧会持有文件句柄；调用方应在同一请求内 ``collect()`` 完成。
        若后端不支持本地路径，则降级为「先读进内存再转 lazy」。
        """
        version_row = self.get_version_row(dataset_id, version)

        local = self.version_local_path(version_row)

        if local is not None and _lazy_scan_enabled():
            lf = pl.scan_parquet(str(local))
            return lf.select(columns) if columns else lf

        frame = self.load_version(dataset_id, version_row.version, columns=columns)
        return frame.lazy()

    def _read_version(
        self,
        version_row: DatasetVersion,
    ) -> pl.DataFrame:
        """从 Storage 读取并解码完整版本快照。"""
        local = self.version_local_path(version_row)

        if local is not None and local.is_file():
            try:
                return pl.read_parquet(str(local))
            except Exception as exc:
                self._raise_read_error(version_row, exc)

        content = self.storage.read(version_row.storage_path)

        try:
            return pl.read_parquet(io.BytesIO(content))
        except Exception as exc:
            self._raise_read_error(version_row, exc)

    def _read_version_columns(
        self,
        version_row: DatasetVersion,
        columns: list[str],
    ) -> pl.DataFrame:
        """按列投影读取版本快照（Parquet 列裁剪）。"""
        wanted = [c for c in dict.fromkeys(columns) if c]

        schema_json = version_row.schema_json or {}
        known = [c for c in wanted if c in schema_json]

        # 请求的列全部不存在时退回全量读取，让上层给出"列不存在"的准确报错。
        if not known:
            return self._read_version(version_row)

        local = self.version_local_path(version_row)

        if local is not None and local.is_file():
            try:
                return pl.read_parquet(str(local), columns=known)
            except Exception:
                # 列裁剪失败（如 schema 元数据过期）时降级为全量读取，保证正确性。
                return self._read_version(version_row)

        try:
            return pl.read_parquet(self._storage_path(version_row), columns=known)
        except Exception:
            return self._read_version(version_row)

    def _storage_path(self, version_row: DatasetVersion) -> str:
        """返回版本快照的可读路径（列裁剪需要按路径读取）。

        与 ``version_local_path`` 的差别：这里在「拿不到路径」时**直接报错**，
        因为调用方已经承诺按路径读取（历史行为，保持兼容）。
        """
        local = self.version_local_path(version_row)

        if local is not None:
            return str(local)

        raise DatasetException(
            "column pruning requires local storage backend",
            code="COLUMN_PRUNE_UNSUPPORTED",
            details={"storage_path": version_row.storage_path},
        )

    @staticmethod
    def _raise_read_error(version_row: DatasetVersion, exc: Exception) -> None:
        raise DatasetException(
            "failed to read dataset snapshot",
            code="VERSION_READ_ERROR",
            details={
                "dataset_id": version_row.dataset_id,
                "version": version_row.version,
                "storage_path": version_row.storage_path,
                "error": str(exc),
            },
        ) from exc

    # =========================================================
    # Delete
    # =========================================================

    def delete(
        self,
        dataset_id: int,
    ) -> None:
        """删除 Dataset 及其版本快照。"""

        self.get(dataset_id)

        prefix = (
            f"datasets/{dataset_id}/"
        )

        metadata = list(
            self.storage.list(
                prefix
            )
        )

        for item in metadata:
            self.storage.delete(
                item.key
            )

        self.db.delete(
            self.get(dataset_id)
        )
        self.db.commit()

        # 数据集已删除：清理其全部版本缓存，避免内存残留与 id 复用时的脏读。
        self.cache.invalidate_dataset(dataset_id)
