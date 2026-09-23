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
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    DatasetException,
    NotFoundException,
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
        """返回最新版本行；没有版本时返回 None。"""

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
        )

        return self.db.scalars(stmt).first()

    def create_version(
        self,
        dataset_id: int,
        df: pl.DataFrame,
        parent_version_id: int | None = None,
    ) -> DatasetVersion:
        """创建不可变 DatasetVersion（数据已在内存时使用）。

        DataFrame -> 临时 Parquet 文件 -> Storage -> DatasetVersion。

        改造要点：不再用 ``io.BytesIO`` 攒一份完整 Parquet 字节串。对 1 GB 级
        数据，那等于在「DataFrame 常驻」之外**再多占一份压缩后体积的内存峰值**，
        而且写入 Storage 时还要再拷一次。改为落临时文件后 ``promote``
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

        next_version, resolved_parent = self._reserve_version(
            dataset_id, parent_version_id
        )
        storage_key = self._version_key(dataset_id, next_version)

        tmp_path = self._temp_target(storage_key)
        try:
            df.write_parquet(tmp_path, compression="zstd")
            self._promote_or_save(storage_key, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return self._insert_version_row(
            dataset_id=dataset_id,
            version=next_version,
            parent_version_id=resolved_parent,
            storage_key=storage_key,
            row_count=df.height,
            column_count=df.width,
            schema_json={name: str(dtype) for name, dtype in df.schema.items()},
        )

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

        next_version, resolved_parent = self._reserve_version(
            dataset_id, parent_version_id
        )
        storage_key = self._version_key(dataset_id, next_version)

        tmp_path = self._temp_target(storage_key)
        try:
            result = ingest_engine.ingest_to_parquet(
                source_path, tmp_path, fmt=fmt, options=options
            )
            self._promote_or_save(storage_key, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        version = self._insert_version_row(
            dataset_id=dataset_id,
            version=next_version,
            parent_version_id=resolved_parent,
            storage_key=storage_key,
            row_count=result.row_count,
            column_count=result.column_count,
            schema_json=result.schema,
        )

        return version, result

    # ---------------------------------------------------------
    # 写路径内部工具
    # ---------------------------------------------------------

    @contextmanager
    def stage_version(
        self,
        dataset_id: int,
        parent_version_id: int | None = None,
    ) -> Iterator["VersionStaging"]:
        """预留版本号 + 提供暂存路径的上下文管理器（写盘产出方统一入口）。

        动机：除了「上传文件」和「DataFrame 已在内存」这两种写路径，还有
        「外部数据源流式产出 Parquet」（数据库连接器抽取）这类场景。它们的
        共同点是——**必须在写之前就确定最终存储 key**，否则暂存文件与目标
        不在同一卷，无法用原子 rename 落地（大数据量下多一次全量拷贝）。

        用法::

            with service.stage_version(dataset_id) as staging:
                result = extract(..., staging.path)
                staging.set_stats(row_count=result.row_count, ...)
            # 无异常时退出即提交（promote + 写 DatasetVersion 行）
            # 有异常时丢弃暂存文件，不产生任何版本
        """
        self.get(dataset_id)

        version, resolved_parent = self._reserve_version(dataset_id, parent_version_id)
        storage_key = self._version_key(dataset_id, version)
        tmp_path = self._temp_target(storage_key)

        staging = VersionStaging(
            staging_path=tmp_path,
            version=version,
            storage_key=storage_key,
        )

        try:
            yield staging

            if staging.row_count is None or staging.column_count is None:
                raise DatasetException(
                    "暂存版本未提供行列数，拒绝提交",
                    code="STAGING_INCOMPLETE",
                    details={"storage_key": storage_key},
                )

            self._promote_or_save(storage_key, tmp_path)

            staging.version_row = self._insert_version_row(
                dataset_id=dataset_id,
                version=version,
                parent_version_id=resolved_parent,
                storage_key=storage_key,
                row_count=int(staging.row_count),
                column_count=int(staging.column_count),
                schema_json=staging.schema_json or {},
            )
        finally:
            # 提交成功后 promote 已经把暂存文件移走；失败/异常时在这里清掉。
            tmp_path.unlink(missing_ok=True)

    def _reserve_version(
        self,
        dataset_id: int,
        parent_version_id: int | None,
    ) -> tuple[int, int | None]:
        """计算下一个版本号并补默认父版本。"""
        latest = self.latest_version(dataset_id)

        next_version = latest.version + 1 if latest is not None else 1

        if parent_version_id is None and latest is not None:
            parent_version_id = latest.id

        return next_version, parent_version_id

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

    def _insert_version_row(
        self,
        *,
        dataset_id: int,
        version: int,
        parent_version_id: int | None,
        storage_key: str,
        row_count: int,
        column_count: int,
        schema_json: dict,
    ) -> DatasetVersion:
        """写入 DatasetVersion 行；DB 失败时回收已写入的 Storage 对象。"""
        model = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            parent_version_id=parent_version_id,
            storage_path=storage_key,
            format="parquet",
            row_count=row_count,
            column_count=column_count,
            schema_json=schema_json,
        )

        try:
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return model
        except Exception:
            # DB 写入失败时清理已经写入 Storage 的对象，避免产生「孤儿快照」。
            try:
                if self.storage.exists(storage_key):
                    self.storage.delete(storage_key)
            except Exception:
                pass

            raise

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
