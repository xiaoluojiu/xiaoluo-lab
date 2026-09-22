"""数据处理 API。

核心流程：

选择 DatasetVersion
        ↓
预览操作
        ↓
确认操作
        ↓
生成新的 DatasetVersion
        ↓
记录 Operation

原版本永不覆盖。
"""

from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Query,
)
from pydantic import (
    BaseModel,
    Field,
)

from app.api.deps import (
    get_data_engine_service,
)
from app.api.v1._serializers import (
    operation_dict,
    version_dict,
)
from app.data_engine.service import (
    DataEngineService,
)
from app.schemas.common import (
    ApiResponse,
)

router = APIRouter(
    prefix="/processing",
    tags=["processing"],
)


class OperationRequest(
    BaseModel
):
    params: dict[str, Any] = Field(
        default_factory=dict
    )

    input_version: int | None = None


class OperationPreviewRequest(
    BaseModel
):
    params: dict[str, Any] = Field(
        default_factory=dict
    )

    input_version: int | None = None

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
    )


# =========================================================
# Operation Catalog
# =========================================================


@router.get(
    "/operations",
    response_model=ApiResponse[list],
)
def list_operations(
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[list]:
    return ApiResponse[list](
        data=service.available_operations()
    )


# =========================================================
# Operation Preview
# =========================================================


@router.post(
    "/datasets/{dataset_id}/preview",
    response_model=ApiResponse[dict],
)
def preview_operation(
    dataset_id: int,
    op_type: str,
    body: OperationPreviewRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    result = (
        service.preview_operation(
            dataset_id,
            op_type,
            body.params,
            input_version=body.input_version,
            page_size=body.page_size,
        )
    )

    return ApiResponse[dict](
        data=result
    )


# =========================================================
# Run Operation
# =========================================================


def _run(
    service: DataEngineService,
    dataset_id: int,
    op_type: str,
    body: OperationRequest,
) -> ApiResponse[dict]:
    version, operation = (
        service.run_operation(
            dataset_id,
            op_type,
            body.params,
            input_version=body.input_version,
        )
    )

    return ApiResponse[dict](
        data={
            "version": version_dict(
                version
            ),
            "operation": operation_dict(
                operation
            ),
        }
    )


@router.post(
    "/datasets/{dataset_id}/clean",
    response_model=ApiResponse[dict],
)
def clean_dataset(
    dataset_id: int,
    body: OperationRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    return _run(
        service,
        dataset_id,
        "missing",
        body,
    )


@router.post(
    "/datasets/{dataset_id}/filter",
    response_model=ApiResponse[dict],
)
def filter_dataset(
    dataset_id: int,
    body: OperationRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    return _run(
        service,
        dataset_id,
        "filter",
        body,
    )


@router.post(
    "/datasets/{dataset_id}/transform",
    response_model=ApiResponse[dict],
)
def transform_dataset(
    dataset_id: int,
    body: OperationRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    return _run(
        service,
        dataset_id,
        "transform",
        body,
    )


@router.post(
    "/datasets/{dataset_id}/aggregate",
    response_model=ApiResponse[dict],
)
def aggregate_dataset(
    dataset_id: int,
    body: OperationRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    return _run(
        service,
        dataset_id,
        "aggregate",
        body,
    )


@router.post(
    "/datasets/{dataset_id}/operation/{op_type}",
    response_model=ApiResponse[dict],
)
def run_named_operation(
    dataset_id: int,
    op_type: str,
    body: OperationRequest,
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[dict]:
    """统一执行 Data Engine 已注册操作。

    保留 clean/filter/transform/aggregate 旧端点，
    新版前端通过本端点直接暴露完整操作目录。
    """
    return _run(
        service,
        dataset_id,
        op_type,
        body,
    )


# =========================================================
# History
# =========================================================


@router.get(
    "/datasets/{dataset_id}/history",
    response_model=ApiResponse[list],
)
def operation_history(
    dataset_id: int,
    limit: int = Query(
        100,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        0,
        ge=0,
    ),
    service: DataEngineService = Depends(
        get_data_engine_service
    ),
) -> ApiResponse[list]:
    operations = (
        service.operation_history(
            dataset_id,
            limit=limit,
            offset=offset,
        )
    )

    return ApiResponse[list](
        data=[
            operation_dict(
                operation
            )
            for operation in operations
        ]
    )
