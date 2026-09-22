"""Dataset 相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.schemas.common import (
    PageInfo,
)


class DatasetCreate(
    BaseModel
):
    name: str = Field(
        min_length=1,
        max_length=255,
    )

    description: str = ""

    source_file_id: int | None = None


class DatasetUpdate(
    BaseModel
):
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    description: str | None = None


class DatasetVersionResponse(
    BaseModel
):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    dataset_id: int

    version: int

    parent_version_id: int | None

    storage_path: str

    format: str

    row_count: int

    column_count: int

    schema: dict = Field(
        default_factory=dict
    )

    @classmethod
    def from_model(
        cls,
        version,
    ) -> DatasetVersionResponse:
        return cls(
            id=version.id,
            dataset_id=version.dataset_id,
            version=version.version,
            parent_version_id=version.parent_version_id,
            storage_path=version.storage_path,
            format=version.format,
            row_count=version.row_count,
            column_count=version.column_count,
            schema=(
                version.schema_json
                or {}
            ),
        )


class DatasetResponse(
    BaseModel
):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    name: str

    description: str

    source_file_id: int | None

    created_at: datetime

    updated_at: datetime

    latest_version: (
        DatasetVersionResponse
        | None
    ) = None


class DatasetListResponse(
    BaseModel
):
    items: list[
        DatasetResponse
    ] = Field(
        default_factory=list
    )

    page_info: PageInfo = Field(
        default_factory=PageInfo
    )
