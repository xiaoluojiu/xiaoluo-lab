"""文件 API。"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import get_file_service
from app.schemas.common import ApiResponse, PageInfo, Pagination
from app.schemas.file import FileResponse
from app.services.file_service import FileService

router = APIRouter(prefix="/files", tags=["files"])


def _content_disposition(filename: str) -> str:
    """构造下载响应头：先剔除引号/反斜杠/控制字符，再用 RFC 5987 filename* 保留原名。

    original_name 是用户可控输入（_sanitize_filename 只作用于落盘的 stored_name），
    直接拼进 `attachment; filename="..."` 时，文件名里的 `"` 或 CRLF 会截断/注入响应头。
    """
    safe = "".join(
        "_" if (ch in '"\\\r\n' or ord(ch) < 32) else ch for ch in (filename or "")
    ).strip() or "download"
    ascii_name = "".join(
        ch if ((ch.isascii() and ch.isalnum()) or ch in "._-") else "_" for ch in safe
    )
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(safe)}'


@router.post(
    "/upload",
    response_model=ApiResponse[FileResponse],
)
def upload_file(
    file: UploadFile,
    service: FileService = Depends(get_file_service),
) -> ApiResponse[FileResponse]:
    """上传文件。"""
    saved = service.upload_stream(
        original_name=file.filename or "",
        file_obj=file.file,
    )

    return ApiResponse(
        data=FileResponse.model_validate(saved)
    )


@router.get(
    "",
    response_model=ApiResponse[Pagination[FileResponse]],
)
def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: FileService = Depends(get_file_service),
) -> ApiResponse[Pagination[FileResponse]]:
    """分页获取文件。"""
    items, total = service.list(
        page=page,
        page_size=page_size,
    )

    return ApiResponse(
        data=Pagination(
            items=[
                FileResponse.model_validate(item)
                for item in items
            ],
            page_info=PageInfo.build(
                page,
                page_size,
                total,
            ),
        )
    )


@router.get(
    "/{file_id}",
    response_model=ApiResponse[FileResponse],
)
def get_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> ApiResponse[FileResponse]:
    """获取文件信息。"""
    return ApiResponse(
        data=FileResponse.model_validate(
            service.get(file_id)
        )
    )


@router.get("/{file_id}/content")
def get_file_content(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> Response:
    """下载文件。"""
    file = service.get(file_id)

    return Response(
        content=service.read(file_id),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": _content_disposition(file.original_name)
        },
    )


@router.delete(
    "/{file_id}",
    response_model=ApiResponse[dict],
)
def delete_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
) -> ApiResponse[dict]:
    """删除文件。"""
    service.delete(file_id)

    return ApiResponse(
        data={
            "deleted": True,
            "file_id": file_id,
        }
    )
