"""数据层对象的序列化辅助（ORM -> dict）。

放在 data_engine 而不是 api 层：
服务层（DataEngineService 的版本时间线 / Diff）也要用它，
而依赖方向必须是 api -> service，**不能**反过来让 service import api。
"""

from __future__ import annotations

from typing import Any

from app.models.dataset_version import DatasetVersion
from app.models.operation import Operation


def version_dict(v: DatasetVersion) -> dict[str, Any]:
    return {
        "id": v.id,
        "dataset_id": v.dataset_id,
        "version": v.version,
        "parent_version_id": v.parent_version_id,
        "storage_path": v.storage_path,
        "format": v.format,
        "row_count": v.row_count,
        "column_count": v.column_count,
        "schema": v.schema_json or {},
    }


def operation_dict(op: Operation | None) -> dict[str, Any] | None:
    if op is None:
        return None
    return {
        "id": op.id,
        "dataset_id": op.dataset_id,
        "input_version_id": op.input_version_id,
        "output_version_id": op.output_version_id,
        "operation_type": op.operation_type,
        "parameters": op.parameters or {},
        "status": op.status,
        "error": op.error,
        "created_at": str(op.created_at) if op.created_at else None,
    }
