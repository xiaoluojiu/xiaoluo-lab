"""统一 API Schema。"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiError(BaseModel):
    """统一错误结构。"""

    code: str = "INTERNAL_ERROR"
    message: str = "Internal Server Error"
    details: dict | list | str | int | float | bool | None = None


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应。"""

    success: bool = True
    data: T | None = None
    error: ApiError | None = None
    request_id: str | None = Field(
        default=None,
        description="请求链路 ID",
    )


class PageInfo(BaseModel):
    """分页信息。"""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(
        default=20,
        ge=1,
        le=200,
    )
    total: int = Field(
        default=0,
        ge=0,
    )
    total_pages: int = Field(
        default=0,
        ge=0,
    )

    @classmethod
    def build(
        cls,
        page: int,
        page_size: int,
        total: int,
    ) -> PageInfo:
        total_pages = (
            (total + page_size - 1) // page_size
            if page_size
            else 0
        )

        return cls(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )


class Pagination(BaseModel, Generic[T]):
    """统一分页结果。"""

    items: list[T] = Field(
        default_factory=list
    )

    page_info: PageInfo = Field(
        default_factory=PageInfo
    )
