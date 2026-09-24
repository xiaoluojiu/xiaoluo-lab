"""API 层共享序列化辅助（Router 内不写业务逻辑，仅做 ORM/对象 -> dict）。"""

from __future__ import annotations

from typing import Any

from app.data_engine.merge.plan import MergePlan
# version_dict / operation_dict 的真源在 data_engine.serializers：
# 服务层（版本时间线 / Diff）也要用它们，依赖方向不能是 service -> api。
from app.data_engine.serializers import (
    operation_dict,
    version_dict,
)
from app.models.experiment import Experiment
from app.models.experiment_run import ExperimentRun


__all__ = [
    "experiment_dict",
    "merge_plan_from_dict",
    "operation_dict",
    "run_dict",
    "version_dict",
]


def experiment_dict(exp: Experiment) -> dict[str, Any]:
    return {
        "id": exp.id,
        "dataset_id": exp.dataset_id,
        "dataset_version_id": exp.dataset_version_id,
        "task": exp.task,
        "model": exp.model,
        "target_column": exp.target_column,
        "parameters": exp.parameters or {},
        "preprocessing": exp.preprocessing or {},
        "seed": exp.seed,
        "description": exp.description,
        "created_at": str(exp.created_at) if exp.created_at else None,
    }


def run_dict(run: ExperimentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "status": run.status,
        "metrics": run.metrics or {},
        "artifacts": run.artifacts or {},
        "runtime": run.runtime,
        "error": run.error,
        "created_at": str(run.created_at) if run.created_at else None,
    }


def merge_plan_from_dict(data: dict[str, Any]) -> MergePlan:
    """请求体中的 plan dict -> MergePlan（计划与执行分离）。"""
    from app.data_engine.merge.plan import ColumnMapping, JoinKey

    return MergePlan(
        left=dict(data.get("left") or {}),
        right=dict(data.get("right") or {}),
        keys=[JoinKey(left=str(k["left"]), right=str(k["right"])) for k in data.get("keys", [])],
        mapping=[
            ColumnMapping(
                right_column=str(m["right_column"]), output_column=str(m["output_column"])
            )
            for m in data.get("mapping", [])
        ],
        join_type=str(data.get("join_type", "inner")),
        conflicts=list(data.get("conflicts", [])),
        warnings=list(data.get("warnings", [])),
        left_suffix=str(data.get("left_suffix", "_left")),
        right_suffix=str(data.get("right_suffix", "_right")),
        right_columns=(
            [str(c) for c in data["right_columns"]] if data.get("right_columns") else None
        ),
    )
