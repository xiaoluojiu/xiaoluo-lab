"""文件 API。"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.api.deps import get_file_service
from app.core.config import settings
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


@router.get(
    "/limits",
    response_model=ApiResponse[dict],
)
def get_upload_limits() -> ApiResponse[dict]:
    """上传限额（前端不再硬编码 100 MB，避免前后端上限不一致）。

    注意：本路由必须声明在 ``/{file_id}`` **之前**，否则 ``limits`` 会被
    当作 ``file_id`` 去解析并返回 422。
    """
    return ApiResponse[dict](
        data={
            **settings.upload_limits(),
            "allowed_extensions": [".csv", ".json", ".xlsx", ".xls", ".parquet", ".arff"],
        }
    )


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
    """流式下载文件。

    此前是 ``Response(content=service.read(...))``：整个文件先读成 bytes 再交给
    框架，2 GiB 的文件会直接顶爆内存。改为分块迭代器后，内存占用恒为块大小。
    """
    file, chunks = service.open_stream(file_id)

    return StreamingResponse(
        chunks,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": _content_disposition(file.original_name),
            "Content-Length": str(file.size),
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
