"""File 相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import PageInfo


class FileResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    name: str
    original_name: str
    path: str
    size: int
    format: str
    checksum: str
    created_at: datetime


class FileListResponse(BaseModel):
    items: list[FileResponse] = Field(
        default_factory=list
    )
    page_info: PageInfo = Field(
        default_factory=PageInfo
    )
