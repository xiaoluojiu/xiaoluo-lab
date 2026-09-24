"""DataEngineService。

Data Engine 统一门面。

职责：

Loader
Reader
Schema
Profile
Quality
Operation
Merge

原则：

1. Data Engine 不依赖 FastAPI
2. 数据操作只接受结构化参数
3. 禁止 eval / exec
4. 操作通过 OPERATION_REGISTRY 调度
5. 数据修改产生新的 DatasetVersion
6. 原版本永远不覆盖
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sqlalchemy import select

from app.analysis import analyze_schema, build_report, profile
from app.data_engine.exceptions import (
    TransformError,
)
from app.data_engine.param_validation import validate_selector_params
from app.data_engine.loaders import REGISTRY, LoadedTable, LoaderRegistry
from app.data_engine.merge.executor import (
    MergeExecutor,
)
from app.data_engine.merge.key_analyzer import (
    analyze_join_keys,
)
from app.data_engine.merge.plan import (
    MergePlan,
)
from app.data_engine.merge.report import (
    MergeReport,
)
from app.data_engine.merge.schema_mapper import (
    suggest_mappings,
)
from app.data_engine.merge.validator import (
    ValidationResult,
    validate_merge_plan,
)
from app.data_engine.merge_multi import (
    analyze_multi_merge,
    execute_multi_merge,
)
from app.data_engine.serializers import (
    operation_dict,
    version_dict,
)
from app.data_engine.operations import (
    add_column,
    aggregate,
    apply_filter,
    apply_string_op,
    cast_columns,
    drop_duplicates,
    handle_missing,
    melt,
    pivot,
    preview,
)
from app.models.dataset_version import DatasetVersion
from app.models.operation import (
    Operation,
)
from app.services.dataset_service import (
    DatasetService,
)

# =========================================================
# Operation Registry
# =========================================================

OPERATION_REGISTRY: dict[
    str,
    tuple[Any, tuple[str, ...]],
] = {
    "missing": (
        handle_missing,
        (
            "strategy",
            "columns",
            "value",
        ),
    ),
    "duplicate": (
        drop_duplicates,
        (
            "subset",
            "keep",
        ),
    ),
    "cast": (
        cast_columns,
        (
            "types",
            "formats",
        ),
    ),
    "string": (
        apply_string_op,
        (
            "column",
            "op",
            "params",
        ),
    ),
    "filter": (
        apply_filter,
        (
            "conditions",
            "logic",
        ),
    ),
    "transform": (
        add_column,
        (
            "name",
            "expression",
            "overwrite",
        ),
    ),
    "aggregate": (
        aggregate,
        (
            "group_by",
            "aggregations",
        ),
    ),
    "pivot": (
        pivot,
        (
            "index",
            "columns",
            "values",
            "aggregation",
        ),
    ),
    "melt": (
        melt,
        (
            "id_vars",
            "value_vars",
            "variable_name",
            "value_name",
        ),
    ),
}


OPERATION_METADATA = {
    "missing": {
        "label": "缺失值处理",
        "description": "处理数据中的缺失值。",
        "risk": "medium",
    },
    "duplicate": {
        "label": "重复数据",
        "description": "删除重复记录。",
        "risk": "medium",
    },
    "cast": {
        "label": "类型转换",
        "description": "转换字段数据类型。",
        "risk": "medium",
    },
    "string": {
        "label": "字符串处理",
        "description": "对文本字段执行结构化字符串操作。",
        "risk": "low",
    },
    "filter": {
        "label": "数据筛选",
        "description": "按照结构化条件过滤数据。",
        "risk": "medium",
    },
    "transform": {
        "label": "字段转换",
        "description": "通过安全表达式创建派生字段。",
        "risk": "medium",
    },
    "aggregate": {
        "label": "分组聚合",
        "description": "按照字段进行统计聚合。",
        "risk": "medium",
    },
    "pivot": {
        "label": "透视",
        "description": "将长表转换为宽表。",
        "risk": "medium",
    },
    "melt": {
        "label": "逆透视",
        "description": "将宽表转换为长表。",
        "risk": "medium",
    },
}


def _flat_schema(raw: Any) -> dict[str, str]:
    """``DatasetVersion.schema_json`` -> ``{列名: 类型}``。

    建版本时落库的 schema 形态是 ``{"col": "Int64", ...}``；
    个别老版本可能存成列表形态，这里统一摊平，缺失/异常一律按空处理，
    不让「描述差异」这种只读功能因为脏数据炸掉。
    """

    if not isinstance(raw, dict):
        return {}

    return {
        str(key): str(value)
        for key, value in raw.items()
    }


def _quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    """质量报告 -> Diff 里要展示的几个数字。

    只取「可比」的统计量：问题数、缺失单元格、重复行。
    完整 issue 列表不进 Diff（那是「质量」页的职责）。
    """

    statistics = report.get("statistics") or {}

    return {
        "issue_count": int(
            statistics.get("issue_count") or 0
        ),
        "missing_cells": int(
            statistics.get("missing_cells") or 0
        ),
        "duplicate_rows": int(
            statistics.get("duplicate_rows") or 0
        ),
        "severity": report.get("severity") or {},
    }


class DataEngineService:
    """Data Engine 门面服务。"""

    def __init__(
        self,
        dataset_service: DatasetService | None = None,
        registry: LoaderRegistry = REGISTRY,
    ) -> None:
        self.dataset_service = (
            dataset_service
        )

        self.registry = registry

        self.executor = MergeExecutor()

    # =========================================================
    # Load
    # =========================================================

    def load(
        self,
        source: Any,
        data: bytes | None = None,
        **options: Any,
    ) -> LoadedTable:
        return self.registry.load(
            source,
            data,
            **options,
        )

    # =========================================================
    # Analysis
    # =========================================================

    def schema(
        self,
        df: pl.DataFrame,
    ) -> dict[str, Any]:
        return analyze_schema(
            df
        )

    def column_schema(
        self,
        dataset_id: int,
        version: int | None = None,
    ) -> list[dict[str, Any]]:
        """只读列结构（列名 + dtype），**不加载任何数据行**。

        为 Pre-flight 而设：目标列是否明确、任务类型是否与列类型矛盾，
        这两项判定只需要列名与类型。用 `schema()` 会去算每列唯一值，
        在千万行表上是几十秒级开销 —— 那是「为了防错反而把链路拖垮」。

        拿不到本地路径（如远端存储）时返回空列表，调用方据此跳过相关检查。
        """
        try:
            version_row = self.dataset_service.get_version_row(dataset_id, version)
            path = self.dataset_service.version_local_path(version_row)
        except Exception:  # noqa: BLE001 - 探测失败按「拿不到」处理
            return []
        if path is None:
            return []
        try:
            collected = pl.scan_parquet(str(path)).collect_schema()
        except Exception:  # noqa: BLE001
            return []
        return [{"name": str(name), "dtype": str(dtype)} for name, dtype in collected.items()]

    def profile(
        self,
        df: pl.DataFrame,
    ) -> dict[str, Any]:
        return profile(
            df
        )

    def preview(
        self,
        df: pl.DataFrame
        | pl.LazyFrame,
        **params: Any,
    ) -> dict[str, Any]:
        return preview(
            df,
            **params,
        )

    def quality(
        self,
        df: pl.DataFrame,
        *,
        expected_schema: dict[
            str,
            str,
        ]
        | None = None,
        target: str | None = None,
        business_rules: list[Any] | None = None,
        adaptive_outlier: bool = False,
    ) -> dict[str, Any]:
        """质量检查。

        ``target`` 传入后，异常值检查会把该列标为「只描述、不清洗」，
        并在结果里说明它未进入清洗建议（第二层：目标列保护）。
        """
        return build_report(
            df,
            expected_schema=expected_schema,
            target=target,
            business_rules=business_rules,
            adaptive_outlier=adaptive_outlier,
        ).to_dict()

    # =========================================================
    # Operation
    # =========================================================

    def available_operations(
        self,
    ) -> list[dict[str, Any]]:
        """返回前端 / Agent 可用操作目录。"""

        result = []

        for (
            op_type,
            (
                _func,
                params,
            ),
        ) in sorted(
            OPERATION_REGISTRY.items()
        ):
            metadata = (
                OPERATION_METADATA.get(
                    op_type,
                    {},
                )
            )

            result.append(
                {
                    "op_type": op_type,
                    "params": list(
                        params
                    ),
                    "label": metadata.get(
                        "label",
                        op_type,
                    ),
                    "description": metadata.get(
                        "description",
                        "",
                    ),
                    "risk": metadata.get(
                        "risk",
                        "medium",
                    ),
                }
            )

        return result

    def _validate_operation_params(
        self,
        op_type: str,
        params: dict[str, Any],
    ) -> None:
        """严格检查操作参数。

        过去这里会静默忽略未知参数，
        容易导致前端传错参数却不知道。

        现在：

        未知参数 -> 直接报错。
        """

        if not isinstance(
            params,
            dict,
        ):
            raise TransformError(
                "operation params must be an object",
                details={
                    "op_type": op_type,
                },
            )

        entry = OPERATION_REGISTRY.get(
            op_type
        )

        if entry is None:
            raise TransformError(
                f"unknown operation: {op_type!r}",
                details={
                    "op_type": op_type,
                    "allowed": sorted(
                        OPERATION_REGISTRY
                    ),
                },
            )

        _func, allowed = entry

        unknown = sorted(
            set(params)
            - set(allowed)
        )

        if unknown:
            raise TransformError(
                "unknown operation parameters",
                details={
                    "op_type": op_type,
                    "unknown": unknown,
                    "allowed": list(
                        allowed
                    ),
                },
            )

        # 操作专属基础校验
        if op_type == "missing":
            strategy = params.get(
                "strategy"
            )

            if not strategy:
                raise TransformError(
                    "missing operation requires strategy"
                )

        elif op_type == "filter":
            conditions = params.get(
                "conditions"
            )

            if not conditions:
                raise TransformError(
                    "filter operation requires conditions"
                )

        elif op_type == "transform":
            if not params.get(
                "name"
            ):
                raise TransformError(
                    "transform operation requires name"
                )

            if not isinstance(
                params.get(
                    "expression"
                ),
                dict,
            ):
                raise TransformError(
                    "transform expression must be a structured object"
                )

        elif op_type == "aggregate":
            if not params.get(
                "group_by"
            ):
                raise TransformError(
                    "aggregate operation requires group_by"
                )

            if not params.get(
                "aggregations"
            ):
                raise TransformError(
                    "aggregate operation requires aggregations"
                )

        # 透视 / 逆透视：必填维度缺失属于「用户输入不完整」，
        # 交给 param_validation 抛 422 + 可执行的中文提示（不是 400）。
        validate_selector_params(op_type, params)

    def apply_operation(
        self,
        df: pl.DataFrame,
        op_type: str,
        params: dict[str, Any],
    ) -> pl.DataFrame:
        """执行 DataFrame 级操作。"""

        entry = OPERATION_REGISTRY.get(
            op_type
        )

        if entry is None:
            raise TransformError(
                f"unknown operation: {op_type!r}",
                details={
                    "op_type": op_type,
                    "allowed": sorted(
                        OPERATION_REGISTRY
                    ),
                },
            )

        self._validate_operation_params(
            op_type,
            params,
        )

        func, param_names = entry

        kwargs = {
            key: value
            for key, value
            in params.items()
            if key in param_names
        }

        try:
            result = func(
                df,
                **kwargs,
            )
        except TransformError:
            raise
        except Exception as exc:
            raise TransformError(
                f"operation {op_type!r} failed",
                details={
                    "op_type": op_type,
                    "error": str(exc),
                },
            ) from exc

        if not isinstance(
            result,
            pl.DataFrame,
        ):
            raise TransformError(
                "operation did not return a Polars DataFrame",
                details={
                    "op_type": op_type,
                },
            )

        return result

    # =========================================================
    # Operation Preview
    # =========================================================

    def preview_operation(
        self,
        dataset_id: int,
        op_type: str,
        params: dict[str, Any],
        *,
        input_version: int | None = None,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """执行操作但不创建新版本。

        用于：

        用户点击「预览结果」

        而不是直接修改数据。
        """

        if self.dataset_service is None:
            raise RuntimeError(
                "DatasetService is required"
            )

        ds = self.dataset_service

        version_row = (
            ds.get_version_row(
                dataset_id,
                input_version,
            )
        )

        before = ds.load_version(
            dataset_id,
            version_row.version,
        )

        after = self.apply_operation(
            before,
            op_type,
            params,
        )

        before_profile = self.profile(
            before
        )

        after_profile = self.profile(
            after
        )

        return {
            "input_version": (
                version_row.version
            ),
            "operation": op_type,
            "before": {
                "rows": before.height,
                "columns": before.width,
            },
            "after": {
                "rows": after.height,
                "columns": after.width,
            },
            "row_delta": (
                after.height
                - before.height
            ),
            "column_delta": (
                after.width
                - before.width
            ),
            "preview": self.preview(
                after,
                page=1,
                page_size=page_size,
            ),
            "before_profile": before_profile,
            "after_profile": after_profile,
        }

    # =========================================================
    # Versioned Operation
    # =========================================================

    def run_operation(
        self,
        dataset_id: int,
        op_type: str,
        params: dict[str, Any],
        *,
        input_version: int | None = None,
    ):
        """执行版本级数据操作。

        DatasetVersion -> 新 DatasetVersion
        """

        if self.dataset_service is None:
            raise RuntimeError(
                "DataEngineService requires DatasetService"
            )

        ds = self.dataset_service

        input_row = (
            ds.get_version_row(
                dataset_id,
                input_version,
            )
        )

        df = ds.load_version(
            dataset_id,
            input_row.version,
        )

        try:
            new_df = self.apply_operation(
                df,
                op_type,
                params,
            )

            output_version = (
                ds.create_version(
                    dataset_id,
                    new_df,
                    parent_version_id=input_row.id,
                )
            )

            operation = (
                self._record_operation(
                    dataset_id,
                    op_type,
                    params,
                    input_version_id=input_row.id,
                    output_version_id=output_version.id,
                    status="success",
                )
            )

            return (
                output_version,
                operation,
            )

        except Exception as exc:
            self._record_operation(
                dataset_id,
                op_type,
                params,
                input_version_id=input_row.id,
                output_version_id=None,
                status="failed",
                error=str(exc),
            )

            raise

    # =========================================================
    # Operation History
    # =========================================================

    def operation_history(
        self,
        dataset_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        """获取数据集操作历史。"""

        if self.dataset_service is None:
            return []

        self.dataset_service.get(
            dataset_id
        )

        stmt = (
            select(Operation)
            .where(
                Operation.dataset_id
                == dataset_id,
            )
            .order_by(Operation.id.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(
            self.dataset_service.db.scalars(stmt)
        )

    # =========================================================
    # Version Timeline / Diff
    #
    # 完全复用已有的 DatasetVersion + Operation：
    # 版本是 DatasetVersion 行，操作来源是 Operation 行（output_version_id -> version.id），
    # 血缘是 DatasetVersion.parent_version_id。这里不新增任何「版本系统」，
    # 只是把已经落库的信息读出来，按时间线组织并算出两个版本之间的差异。
    # =========================================================

    def version_timeline(
        self,
        dataset_id: int,
        limit: int = 200,
    ) -> dict[str, Any]:
        """数据集版本时间线（从旧到新）。

        每项包含：版本号、规模、创建时间、来源操作、相对父版本的变化。
        首个版本没有 Operation 行（它是导入产生的），``origin`` 记为 ``import``。
        """

        if self.dataset_service is None:
            raise RuntimeError(
                "DataEngineService requires DatasetService"
            )

        ds = self.dataset_service

        ds.get(dataset_id)

        rows = list(
            ds.db.scalars(
                select(DatasetVersion)
                .where(
                    DatasetVersion.dataset_id
                    == dataset_id,
                )
                .order_by(DatasetVersion.version.asc())
                .limit(limit)
            )
        )

        # output_version_id -> Operation（一个版本最多由一个操作产出）
        operations = {
            op.output_version_id: op
            for op in ds.db.scalars(
                select(Operation).where(
                    Operation.dataset_id == dataset_id,
                    Operation.output_version_id.is_not(None),
                )
            )
        }

        by_id = {row.id: row for row in rows}

        items: list[dict[str, Any]] = []

        for row in rows:
            parent = (
                by_id.get(row.parent_version_id)
                if row.parent_version_id is not None
                else None
            )
            op = operations.get(row.id)

            items.append(
                {
                    **version_dict(row),
                    "created_at": (
                        str(row.created_at)
                        if row.created_at
                        else None
                    ),
                    "origin": (
                        op.operation_type
                        if op is not None
                        else "import"
                    ),
                    # 展示名复用 OPERATION_METADATA，前端不必再抄一份 op_type -> 中文名
                    "origin_label": (
                        OPERATION_METADATA.get(
                            op.operation_type, {}
                        ).get("label")
                        if op is not None
                        else "导入"
                    ),
                    "operation": operation_dict(op),
                    "parent_version": (
                        parent.version
                        if parent is not None
                        else (
                            row.parent_version_id
                            if row.parent_version_id
                            is not None
                            else None
                        )
                    ),
                    # 相对父版本的变化；父版本不在本次取回范围内时为 None
                    "delta_rows": (
                        row.row_count - parent.row_count
                        if parent is not None
                        else None
                    ),
                    "delta_columns": (
                        row.column_count
                        - parent.column_count
                        if parent is not None
                        else None
                    ),
                }
            )

        latest = rows[-1] if rows else None

        return {
            "dataset_id": dataset_id,
            "total": len(items),
            "latest_version": (
                latest.version if latest is not None else 0
            ),
            "versions": items,
        }

    def version_diff(
        self,
        dataset_id: int,
        base: int,
        target: int,
        *,
        include_quality: bool = True,
    ) -> dict[str, Any]:
        """两个版本之间的差异。

        只做「描述差异」，不修改任何数据：

        - ``schema_diff`` 来自 ``DatasetVersion.schema_json``（建版本时落库），
          **不读数据文件**；
        - ``quality`` 需要真正加载两个版本的数据，所以是可关的（默认开），
          复用 ``quality()`` 本身，保证与「质量」页看到的是同一套口径。
        """

        if self.dataset_service is None:
            raise RuntimeError(
                "DataEngineService requires DatasetService"
            )

        ds = self.dataset_service

        base_row = ds.get_version_row(dataset_id, base)
        target_row = ds.get_version_row(dataset_id, target)

        base_schema = _flat_schema(base_row.schema_json)
        target_schema = _flat_schema(target_row.schema_json)

        added = sorted(
            set(target_schema) - set(base_schema)
        )
        removed = sorted(
            set(base_schema) - set(target_schema)
        )
        changed = sorted(
            column
            for column in set(base_schema)
            & set(target_schema)
            if base_schema[column]
            != target_schema[column]
        )

        quality: dict[str, Any] | None = None

        if include_quality:
            quality = {
                "base": _quality_summary(
                    self.quality(
                        ds.load_version(
                            dataset_id,
                            base_row.version,
                        )
                    )
                ),
                "target": _quality_summary(
                    self.quality(
                        ds.load_version(
                            dataset_id,
                            target_row.version,
                        )
                    )
                ),
            }

        return {
            "dataset_id": dataset_id,
            "base": version_dict(base_row),
            "target": version_dict(target_row),
            "row_count": {
                "base": base_row.row_count,
                "target": target_row.row_count,
                "delta": (
                    target_row.row_count
                    - base_row.row_count
                ),
            },
            "column_count": {
                "base": base_row.column_count,
                "target": target_row.column_count,
                "delta": (
                    target_row.column_count
                    - base_row.column_count
                ),
            },
            "schema_diff": {
                "added": [
                    {
                        "column": column,
                        "dtype": target_schema[column],
                    }
                    for column in added
                ],
                "removed": [
                    {
                        "column": column,
                        "dtype": base_schema[column],
                    }
                    for column in removed
                ],
                "type_changed": [
                    {
                        "column": column,
                        "from": base_schema[column],
                        "to": target_schema[column],
                    }
                    for column in changed
                ],
                "unchanged_count": len(
                    set(base_schema) & set(target_schema)
                )
                - len(changed),
            },
            "quality": quality,
            # 从 base 到 target 之间实际执行过的操作（按 id 升序）。
            "operations": [
                operation_dict(op)
                for op in ds.db.scalars(
                    select(Operation)
                    .where(
                        Operation.dataset_id == dataset_id,
                        Operation.output_version_id
                        == target_row.id,
                    )
                    .order_by(Operation.id.asc())
                )
            ],
        }

    # =========================================================
    # Merge
    # =========================================================

    def suggest_mappings(
        self,
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        **options: Any,
    ):
        return [
            candidate.to_dict()
            for candidate in suggest_mappings(
                left_df,
                right_df,
                **options,
            )
        ]

    def analyze_keys(
        self,
        left_df: pl.DataFrame,
        left_key: str,
        right_df: pl.DataFrame,
        right_key: str,
    ) -> dict[str, Any]:
        return analyze_join_keys(
            left_df,
            left_key,
            right_df,
            right_key,
        )

    def validate_merge(
        self,
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        plan: MergePlan,
    ) -> dict[str, Any]:
        result: ValidationResult = (
            validate_merge_plan(
                plan,
                left_df,
                right_df,
            )
        )

        return result.to_dict()

    def run_merge(
        self,
        left_dataset_id: int,
        right_dataset_id: int,
        plan: MergePlan,
        *,
        input_version: int | None = None,
        right_version: int | None = None,
    ) -> tuple[
        Any,
        MergeReport,
    ]:
        if self.dataset_service is None:
            raise RuntimeError(
                "DataEngineService requires DatasetService"
            )

        ds = self.dataset_service

        left_row = (
            ds.get_version_row(
                left_dataset_id,
                input_version,
            )
        )

        right_row = (
            ds.get_version_row(
                right_dataset_id,
                right_version,
            )
        )

        left_df = ds.load_version(
            left_dataset_id,
            left_row.version,
        )

        right_df = ds.load_version(
            right_dataset_id,
            right_row.version,
        )

        validation = (
            validate_merge_plan(
                plan,
                left_df,
                right_df,
            )
        )

        if not validation.ok:
            raise TransformError(
                "merge plan validation failed",
                details=validation.to_dict(),
            )

        merged_df, report = (
            self.executor.execute(
                left_df,
                right_df,
                plan,
            )
        )

        output_version = (
            ds.create_version(
                left_dataset_id,
                merged_df,
                parent_version_id=left_row.id,
            )
        )

        parameters = {
            **plan.to_dict(),
            "right_dataset_id": (
                right_dataset_id
            ),
            "right_version": (
                right_row.version
            ),
        }

        self._record_operation(
            left_dataset_id,
            "merge",
            parameters,
            input_version_id=left_row.id,
            output_version_id=output_version.id,
            status="success",
        )

        return (
            output_version,
            report,
        )

    # =========================================================
    # Multi-file Merge（N 个数据集纵向整合）
    # =========================================================

    def _load_refs(
        self, refs: list[dict[str, Any]]
    ) -> tuple[list[pl.DataFrame], list[dict[str, Any]]]:
        """加载多个数据集引用为 DataFrame 列表，并附带元信息。"""
        if self.dataset_service is None:
            raise RuntimeError("DataEngineService requires DatasetService")

        frames: list[pl.DataFrame] = []
        meta: list[dict[str, Any]] = []
        for ref in refs:
            dataset_id = int(ref["dataset_id"])
            version = ref.get("version")
            version_row = self.dataset_service.get_version_row(dataset_id, version)
            df = self.dataset_service.load_version(dataset_id, version_row.version)
            frames.append(df)
            meta.append(
                {
                    "dataset_id": dataset_id,
                    "version": version_row.version,
                    "name": self.dataset_service.get(dataset_id).name,
                }
            )
        return frames, meta

    def analyze_multi_merge(
        self, refs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """分析多文件共有 / 独有字段，供前端勾选。"""
        frames, meta = self._load_refs(refs)
        result = analyze_multi_merge(frames)
        for i, file_info in enumerate(result["files"]):
            file_info["dataset_id"] = meta[i]["dataset_id"]
            file_info["name"] = meta[i]["name"]
        return result

    def preview_multi_merge(
        self,
        refs: list[dict[str, Any]],
        selected_columns: list[str] | None = None,
        *,
        add_source: bool = False,
        source_column: str = "source",
        source_labels: list[str] | None = None,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """内存预览多文件合并结果（不创建版本）。"""
        frames, _ = self._load_refs(refs)
        merged = execute_multi_merge(
            frames,
            selected_columns,
            add_source=add_source,
            source_column=source_column,
            source_labels=source_labels,
        )
        return {
            "row_count": merged.height,
            "column_count": merged.width,
            "columns": merged.columns,
            "input_rows": [f.height for f in frames],
            "preview": self.preview(
                merged,
                page=1,
                page_size=page_size,
            ),
        }

    def run_multi_merge(
        self,
        refs: list[dict[str, Any]],
        selected_columns: list[str] | None = None,
        *,
        add_source: bool = False,
        source_column: str = "source",
        source_labels: list[str] | None = None,
        create_new: bool = True,
        target_dataset_id: int | None = None,
        name: str | None = None,
    ) -> tuple[int, Any, bool]:
        """执行多文件合并。

        create_new=True  -> 生成新数据集（返回 is_new=True）
        create_new=False -> 写入 target_dataset_id 的新版本（返回 is_new=False）
        """
        if self.dataset_service is None:
            raise RuntimeError("DataEngineService requires DatasetService")

        ds = self.dataset_service
        frames, _ = self._load_refs(refs)
        merged = execute_multi_merge(
            frames,
            selected_columns,
            add_source=add_source,
            source_column=source_column,
            source_labels=source_labels,
        )

        if create_new or target_dataset_id is None:
            dataset = ds.create(
                name=name or "多文件合并结果",
                description=f"由 {len(frames)} 个数据集合并生成",
                source_file_id=None,
            )
            output_version = ds.create_version(dataset.id, merged)
            dataset_id = dataset.id
            is_new = True
        else:
            dataset_id = int(target_dataset_id)
            ds.get(dataset_id)  # 校验存在
            output_version = ds.create_version(dataset_id, merged)
            is_new = False

        self._record_operation(
            dataset_id,
            "multi_merge",
            {
                "refs": refs,
                "selected_columns": selected_columns,
                "add_source": add_source,
                "source_column": source_column,
                "create_new": create_new,
                "target_dataset_id": target_dataset_id,
            },
            input_version_id=None,
            output_version_id=output_version.id,
            status="success",
        )

        return dataset_id, output_version, is_new

    # =========================================================
    # Internal
    # =========================================================

    def _record_operation(
        self,
        dataset_id: int,
        operation_type: str,
        parameters: dict[str, Any],
        *,
        input_version_id: int | None,
        output_version_id: int | None,
        status: str,
        error: str | None = None,
    ):
        if self.dataset_service is None:
            return None

        operation = Operation(
            dataset_id=dataset_id,
            input_version_id=input_version_id,
            output_version_id=output_version_id,
            operation_type=operation_type,
            parameters=parameters,
            status=status,
            error=error,
        )

        db = self.dataset_service.db

        db.add(operation)
        db.commit()
        db.refresh(operation)

        return operation
