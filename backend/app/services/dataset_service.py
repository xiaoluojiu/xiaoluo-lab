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
"""

from __future__ import annotations

import io

import polars as pl
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DatasetException,
    NotFoundException,
)
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
        """创建不可变 DatasetVersion。

        DataFrame -> Parquet -> Storage -> DatasetVersion。
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

        latest = self.latest_version(
            dataset_id
        )

        next_version = (
            latest.version + 1
            if latest is not None
            else 1
        )

        if (
            parent_version_id is None
            and latest is not None
        ):
            parent_version_id = latest.id

        buffer = io.BytesIO()

        df.write_parquet(
            buffer
        )

        content = buffer.getvalue()

        storage_key = (
            f"datasets/"
            f"{dataset_id}/"
            f"v{next_version:06d}.parquet"
        )

        if self.storage.exists(
            storage_key
        ):
            raise DatasetException(
                "version snapshot already exists",
                code="VERSION_CONFLICT",
                details={
                    "dataset_id": dataset_id,
                    "version": next_version,
                    "storage_path": storage_key,
                },
            )

        self.storage.save(
            storage_key,
            content,
        )

        version = DatasetVersion(
            dataset_id=dataset_id,
            version=next_version,
            parent_version_id=parent_version_id,
            storage_path=storage_key,
            format="parquet",
            row_count=df.height,
            column_count=df.width,
            schema_json={
                name: str(dtype)
                for name, dtype
                in df.schema.items()
            },
        )

        try:
            self.db.add(version)
            self.db.commit()
            self.db.refresh(version)

            return version
        except Exception:
            # DB 写入失败时清理已经写入 Storage
            try:
                self.storage.delete(
                    storage_key
                )
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

    def _read_version(
        self,
        version_row: DatasetVersion,
    ) -> pl.DataFrame:
        """从 Storage 读取并解码完整版本快照。"""
        content = self.storage.read(
            version_row.storage_path
        )

        try:
            return pl.read_parquet(
                io.BytesIO(content)
            )
        except Exception as exc:
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

    def _read_version_columns(
        self,
        version_row: DatasetVersion,
        columns: list[str],
    ) -> pl.DataFrame:
        """按列投影读取版本快照（Parquet 列裁剪）。"""
        wanted = [
            c for c in dict.fromkeys(columns) if c
        ]

        schema_json = version_row.schema_json or {}
        known = [c for c in wanted if c in schema_json]

        # 请求的列全部不存在时退回全量读取，让上层给出"列不存在"的准确报错。
        if not known:
            return self._read_version(version_row)

        try:
            return pl.read_parquet(
                self._storage_path(version_row),
                columns=known,
            )
        except Exception:
            # 列裁剪失败（如 schema 元数据过期）时降级为全量读取，保证正确性。
            return self._read_version(version_row)

    def _storage_path(self, version_row: DatasetVersion):
        """返回版本快照的本地路径（列裁剪需要按路径读取，而非 bytes）。"""
        backend = getattr(self.storage, "_backend", None)
        if backend is not None and hasattr(backend, "_resolve"):
            return backend._resolve(version_row.storage_path)
        # 非本地 Storage 后端：先读 bytes 再解码，无法投影（保持正确性）。
        raise DatasetException(
            "column pruning requires local storage backend",
            code="COLUMN_PRUNE_UNSUPPORTED",
            details={"storage_path": version_row.storage_path},
        )

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
