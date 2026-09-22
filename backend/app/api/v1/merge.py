"""Merge API（Prompt 198 / T0-M3 / T0-M4 / T0-M5 / T0-M7）。

- POST /merge/mapping   字段映射候选（source/target/confidence/reason）
- POST /merge/keys      Join Key 画像与基数分析（支持 composite key）
- POST /merge/preview   内存执行 join + 返回统计与预览（T0-M3：不创建版本）
- POST /merge/validate  计划校验（不执行）
- POST /merge/execute   执行合并 -> 左数据集新版本 + MergeReport

T0-M4：execute 必须显式指定 left_version / right_version，
禁止 fallback 到 latest（避免 validate 用 vN 而 execute 用 vN+1）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_data_engine_service
from app.api.v1._serializers import merge_plan_from_dict, version_dict
from app.core.exceptions import ValidationException
from app.data_engine.merge.executor import MergeExecutor
from app.data_engine.merge.key_analyzer import analyze_join_keys_composite
from app.data_engine.service import DataEngineService
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/merge", tags=["merge"])

PREVIEW_ROWS = 20


class MergePairRequest(BaseModel):
    left_dataset_id: int
    right_dataset_id: int
    left_version: int | None = None
    right_version: int | None = None


class KeyAnalysisRequest(MergePairRequest):
    # T0-M5：单 key 字段保留兼容，新增 *_keys 列表用于 composite key
    left_key: str | None = None
    right_key: str | None = None
    left_keys: list[str] | None = None
    right_keys: list[str] | None = None


class MergePlanBody(BaseModel):
    keys: list[dict[str, str]] = Field(default_factory=list)
    mapping: list[dict[str, str]] = Field(default_factory=list)
    join_type: str = "inner"
    left_suffix: str = "_left"
    right_suffix: str = "_right"
    right_columns: list[str] | None = None
    warnings: list[str] = Field(default_factory=list)


class MergeValidateRequest(MergePairRequest):
    plan: MergePlanBody


class MergeExecuteRequest(MergeValidateRequest):
    pass


class MergePreviewRequest(MergeValidateRequest):
    """T0-M3：预览合并，不创建版本/Operation，不改原数据。"""


def _load_pair(
    service: DataEngineService, body: MergePairRequest
) -> tuple[Any, Any, int, int]:
    ds = service.dataset_service
    left_row = ds.get_version_row(body.left_dataset_id, body.left_version)
    right_row = ds.get_version_row(body.right_dataset_id, body.right_version)
    left_df = ds.load_version(body.left_dataset_id, left_row.version)
    right_df = ds.load_version(body.right_dataset_id, right_row.version)
    return left_df, right_df, left_row.version, right_row.version


def _resolve_key_lists(body: KeyAnalysisRequest) -> tuple[list[str], list[str]]:
    """T0-M5：兼容单 key 字段与 composite key 列表。"""
    if body.left_keys and body.right_keys:
        return list(body.left_keys), list(body.right_keys)
    if body.left_key and body.right_key:
        return [body.left_key], [body.right_key]
    raise ValidationException(
        "必须提供 left_key/right_key 或 left_keys/right_keys",
        details={"left_key": body.left_key, "right_key": body.right_key},
    )


@router.post("/mapping", response_model=ApiResponse[list])
def suggest_mapping(
    body: MergePairRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[list]:
    """字段映射候选：source_column / target_column / confidence / reason。

    注意 schema：source_column = 左表列，target_column = 右表列（仅描述对应关系）。
    真正执行时的列重命名使用 MergePlan.mapping[right_column -> output_column]。
    """
    left_df, right_df, _, _ = _load_pair(service, body)
    return ApiResponse[list](data=service.suggest_mappings(left_df, right_df))


@router.post("/keys", response_model=ApiResponse[dict])
def analyze_keys(
    body: KeyAnalysisRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """Key 画像（唯一/重复/空值/覆盖）+ 连接基数（T0-M5：支持 composite key）。"""
    left_df, right_df, _, _ = _load_pair(service, body)
    left_keys, right_keys = _resolve_key_lists(body)
    data = analyze_join_keys_composite(left_df, left_keys, right_df, right_keys)
    return ApiResponse[dict](data=data)


@router.post("/preview", response_model=ApiResponse[dict])
def preview_merge(
    body: MergePreviewRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """T0-M3：内存合并预览。

    不创建 DatasetVersion、不创建 Operation、不修改原数据。
    返回输入/输出 行列数、warnings、conflicts、前若干行预览。
    """
    left_df, right_df, left_version, right_version = _load_pair(service, body)
    plan = merge_plan_from_dict(body.plan.model_dump())
    executor = MergeExecutor()
    merged, report = executor.execute(left_df, right_df, plan)
    preview_rows = merged.head(PREVIEW_ROWS).to_dicts()
    return ApiResponse[dict](
        data={
            "left_version": left_version,
            "right_version": right_version,
            "input_rows_left": left_df.height,
            "input_rows_right": right_df.height,
            "input_columns_left": left_df.width,
            "input_columns_right": right_df.width,
            "output_rows": merged.height,
            "output_columns": merged.width,
            "matched_rows": report.matched_rows,
            "unmatched_rows_left": report.unmatched_rows_left,
            "unmatched_rows_right": report.unmatched_rows_right,
            "warnings": report.warnings,
            "conflicts": report.conflicts,
            "output_columns_list": merged.columns,
            "preview": {
                "columns": merged.columns,
                "items": preview_rows,
                "total": merged.height,
            },
        }
    )


@router.post("/validate", response_model=ApiResponse[dict])
def validate_merge(
    body: MergeValidateRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """校验合并计划：Key 存在 / 类型兼容 / 重复 / 多对多 / 冲突。"""
    left_df, right_df, _, _ = _load_pair(service, body)
    plan = merge_plan_from_dict(body.plan.model_dump())
    return ApiResponse[dict](data=service.validate_merge(left_df, right_df, plan))


@router.post("/execute", response_model=ApiResponse[dict])
def execute_merge(
    body: MergeExecuteRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """校验并执行合并；结果写入左数据集的新不可变版本。

    T0-M4：必须显式指定 left_version / right_version（禁止自动取 latest），
    防止 validate 使用 vN 而 execute 使用更新后的 vN+1。
    """
    if body.left_version is None or body.right_version is None:
        raise ValidationException(
            "execute 必须显式指定 left_version 与 right_version"
            "（建议由 preview/validate 返回的版本号固定传入）",
            details={
                "left_version": body.left_version,
                "right_version": body.right_version,
            },
        )
    left_df, right_df, left_version, right_version = _load_pair(service, body)
    plan = merge_plan_from_dict(body.plan.model_dump())
    plan.left = {"dataset_id": body.left_dataset_id, "version": left_version}
    plan.right = {"dataset_id": body.right_dataset_id, "version": right_version}
    output_version, report = service.run_merge(
        body.left_dataset_id,
        body.right_dataset_id,
        plan,
        input_version=body.left_version,
        right_version=body.right_version,
    )
    return ApiResponse[dict](
        data={
            "version": version_dict(output_version),
            "report": report.to_dict(),
        }
    )


# =========================================================
# Multi-file Merge（N 个数据集纵向整合）
# =========================================================


class MultiMergeDatasetRef(BaseModel):
    dataset_id: int
    version: int | None = None
    label: str | None = None  # 可选，用于 add_source 的源标记


class MultiMergeAnalyzeRequest(BaseModel):
    datasets: list[MultiMergeDatasetRef]


class MultiMergePreviewRequest(MultiMergeAnalyzeRequest):
    columns: list[str] | None = None
    add_source: bool = False
    source_column: str = "source"


class MultiMergeExecuteRequest(MultiMergePreviewRequest):
    create_new: bool = True
    target_dataset_id: int | None = None
    name: str | None = None


@router.post("/multi/analyze", response_model=ApiResponse[dict])
def analyze_multi_merge(
    body: MultiMergeAnalyzeRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """分析多文件共有字段 / 独有字段，供前端勾选。"""
    refs = [r.model_dump() for r in body.datasets]
    return ApiResponse[dict](data=service.analyze_multi_merge(refs))


@router.post("/multi/preview", response_model=ApiResponse[dict])
def preview_multi_merge(
    body: MultiMergePreviewRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """内存预览多文件合并结果（不创建版本）。"""
    refs = [r.model_dump() for r in body.datasets]
    source_labels = [r.label for r in body.datasets] if body.datasets else None
    data = service.preview_multi_merge(
        refs,
        body.columns,
        add_source=body.add_source,
        source_column=body.source_column,
        source_labels=source_labels,
    )
    return ApiResponse[dict](data=data)


@router.post("/multi/execute", response_model=ApiResponse[dict])
def execute_multi_merge(
    body: MultiMergeExecuteRequest,
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    """执行多文件合并：生成新数据集 或 写入指定已有数据集的新版本。"""
    if not body.create_new and body.target_dataset_id is None:
        raise ValidationException(
            "合并至已有文件时必须指定 target_dataset_id",
            details={"create_new": body.create_new},
        )

    refs = [r.model_dump() for r in body.datasets]
    source_labels = [r.label for r in body.datasets] if body.datasets else None
    dataset_id, output_version, is_new = service.run_multi_merge(
        refs,
        body.columns,
        add_source=body.add_source,
        source_column=body.source_column,
        source_labels=source_labels,
        create_new=body.create_new,
        target_dataset_id=body.target_dataset_id,
        name=body.name,
    )
    return ApiResponse[dict](
        data={
            "dataset_id": dataset_id,
            "version": version_dict(output_version),
            "is_new": is_new,
        }
    )
