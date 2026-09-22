"""数据集分析 API。"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_data_engine_service
from app.core.exceptions import ValidationException
from app.data_engine.service import DataEngineService
from app.schemas.common import ApiResponse

router = APIRouter(
    prefix="/datasets",
    tags=["dataset-analysis"],
)


def _load_df(
    service: DataEngineService,
    dataset_id: int,
    version: int | None,
):
    version_row = service.dataset_service.get_version_row(
        dataset_id,
        version,
    )

    df = service.dataset_service.load_version(
        dataset_id,
        version_row.version,
    )

    return df, version_row


def _parse_filter(
    raw: str | None,
) -> list[dict[str, Any]] | None:
    if not raw:
        return None

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationException(
            "filter must be valid JSON",
            details={"reason": str(exc)},
        ) from exc

    if not isinstance(value, list):
        raise ValidationException(
            "filter must be a JSON array"
        )

    return value


def _versioned_result(
    data: dict[str, Any],
    version_row,
) -> ApiResponse[dict]:
    data["version"] = version_row.version
    return ApiResponse(data=data)


@router.get(
    "/{dataset_id}/preview",
    response_model=ApiResponse[dict],
)
def preview_dataset(
    dataset_id: int,
    version: int | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    columns: str | None = Query(None),
    sort_column: str | None = Query(None),
    sort_desc: bool = Query(False),
    filter: str | None = Query(None),
    filter_logic: str = Query("and"),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    df, version_row = _load_df(
        service,
        dataset_id,
        version,
    )

    column_list = (
        [item.strip() for item in columns.split(",") if item.strip()]
        if columns
        else None
    )

    sort = (
        {
            "column": sort_column,
            "desc": sort_desc,
        }
        if sort_column
        else None
    )

    data = service.preview(
        df,
        page=page,
        page_size=page_size,
        columns=column_list,
        sort=sort,
        filter_conditions=_parse_filter(filter),
        filter_logic=filter_logic,
        known_total=(
            None if filter else getattr(version_row, "row_count", None)
        ),
    )

    return _versioned_result(data, version_row)


@router.get(
    "/{dataset_id}/schema",
    response_model=ApiResponse[dict],
)
def schema_dataset(
    dataset_id: int,
    version: int | None = Query(None),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    df, version_row = _load_df(
        service,
        dataset_id,
        version,
    )

    return _versioned_result(
        service.schema(df),
        version_row,
    )


@router.get(
    "/{dataset_id}/profile",
    response_model=ApiResponse[dict],
)
def profile_dataset(
    dataset_id: int,
    version: int | None = Query(None),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    df, version_row = _load_df(
        service,
        dataset_id,
        version,
    )

    return _versioned_result(
        service.profile(df),
        version_row,
    )


def _parse_expected_schema(
    raw: str | None,
) -> dict[str, str] | None:
    """解析 expected_schema 查询参数。

    形如 ``{"amount": "float", "ts": "datetime"}`` 的 JSON 对象；
    提供后质量报告会额外执行 SchemaChecker 校验列的存在性与类型。
    """
    if not raw:
        return None

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationException(
            "expected_schema 必须是合法 JSON",
            details={"reason": str(exc)},
        ) from exc

    if not isinstance(value, dict):
        raise ValidationException(
            "expected_schema 必须是 JSON 对象（列名 -> 期望类型）"
        )

    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValidationException(
                "expected_schema 的键与值都必须是字符串",
                details={"offending": str(key)},
            )
        normalized[key] = item

    return normalized


@router.get(
    "/{dataset_id}/quality",
    response_model=ApiResponse[dict],
)
def quality_dataset(
    dataset_id: int,
    version: int | None = Query(None),
    expected_schema: str | None = Query(
        None,
        description=(
            '期望的列结构 JSON，如 {"amount":"float","ts":"datetime"}；'
            "提供后会追加 SchemaChecker 校验，缺列/类型不符会作为质量问题上报"
        ),
    ),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    df, version_row = _load_df(
        service,
        dataset_id,
        version,
    )

    return _versioned_result(
        service.quality(
            df,
            expected_schema=_parse_expected_schema(expected_schema),
        ),
        version_row,
    )
