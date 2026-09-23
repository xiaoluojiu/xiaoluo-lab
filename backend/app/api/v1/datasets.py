"""数据集 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_dataset_service, get_file_service
from app.data_engine.loaders import REGISTRY
from app.schemas.common import ApiResponse, PageInfo, Pagination
from app.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    DatasetUpdate,
    DatasetVersionResponse,
)
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _to_response(
    dataset,
    service: DatasetService,
    *,
    ingest: dict | None = None,
) -> DatasetResponse:
    """ORM Dataset 转 API Schema。"""
    latest = service.latest_version(dataset.id)

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        source_file_id=dataset.source_file_id,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        latest_version=(
            DatasetVersionResponse.from_model(latest)
            if latest is not None
            else None
        ),
        ingest=ingest,
    )


@router.post("", response_model=ApiResponse[DatasetResponse])
def create_dataset(
    payload: DatasetCreate,
    service: DatasetService = Depends(get_dataset_service),
    file_service: FileService = Depends(get_file_service),
) -> ApiResponse[DatasetResponse]:
    dataset = service.create(
        name=payload.name,
        description=payload.description,
        source_file_id=payload.source_file_id,
    )

    # 有 source_file_id 时自动加载文件并创建初始版本
    ingest_meta: dict | None = None

    if payload.source_file_id is not None:
        file_record = file_service.get(payload.source_file_id)

        # 快路径：存储后端支持本地直通时，走「流式解析 → 直接落 Parquet」，
        # 常驻内存与文件体积无关（不再把整份文件读成 bytes 再解析）。
        source_path = file_service.local_path(payload.source_file_id)

        if source_path is not None and source_path.is_file():
            _, result = service.create_version_from_source(
                dataset.id,
                source_path,
                fmt=file_record.format or None,
            )
            ingest_meta = result.to_metadata()
        else:
            # 慢路径：远程/非本地存储后端 —— 先取回 bytes，再按格式物化解析。
            # 这里保留旧行为作为兜底，保证存储后端可替换。
            content = file_service.read(payload.source_file_id)
            loaded = REGISTRY.load(file_record.original_name, data=content)
            service.create_version(dataset.id, loaded.df)
            ingest_meta = {
                "strategy": "materialized(non-local-storage)",
                "row_count": loaded.df.height,
                "column_count": loaded.df.width,
                "warnings": ["存储后端不支持本地直通，已退回内存物化路径"],
            }

    return ApiResponse(
        data=_to_response(dataset, service, ingest=ingest_meta)
    )


@router.get("", response_model=ApiResponse[Pagination[DatasetResponse]])
def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[Pagination[DatasetResponse]]:
    items, total = service.list(
        page=page,
        page_size=page_size,
    )

    return ApiResponse(
        data=Pagination(
            items=[
                _to_response(dataset, service)
                for dataset in items
            ],
            page_info=PageInfo.build(
                page,
                page_size,
                total,
            ),
        )
    )


@router.get(
    "/{dataset_id}",
    response_model=ApiResponse[DatasetResponse],
)
def get_dataset(
    dataset_id: int,
    service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[DatasetResponse]:
    dataset = service.get(dataset_id)

    return ApiResponse(
        data=_to_response(dataset, service)
    )


@router.patch(
    "/{dataset_id}",
    response_model=ApiResponse[DatasetResponse],
)
def update_dataset(
    dataset_id: int,
    payload: DatasetUpdate,
    service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[DatasetResponse]:
    dataset = service.update_metadata(
        dataset_id,
        name=payload.name,
        description=payload.description,
    )

    return ApiResponse(
        data=_to_response(dataset, service)
    )


@router.delete(
    "/{dataset_id}",
    response_model=ApiResponse[dict],
)
def delete_dataset(
    dataset_id: int,
    service: DatasetService = Depends(get_dataset_service),
) -> ApiResponse[dict]:
    service.delete(dataset_id)

    return ApiResponse(
        data={
            "deleted": True,
            "dataset_id": dataset_id,
        }
    )
