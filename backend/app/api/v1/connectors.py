"""数据库连接器 API（拓展功能 · 大数据流接入）。

路由总览
--------
    GET    /connectors                     连接器列表
    GET    /connectors/catalog             可用数据库类型（含驱动就绪状态）
    POST   /connectors                     新建连接器
    POST   /connectors/test                试连（可未保存）
    GET    /connectors/{id}                详情
    PATCH  /connectors/{id}                局部更新
    DELETE /connectors/{id}                删除配置（不动已导入的数据集）
    POST   /connectors/{id}/test           试连已保存的连接器并回写状态
    GET    /connectors/{id}/tables         表 / 视图清单
    GET    /connectors/{id}/tables/{t}/columns   列结构
    POST   /connectors/{id}/preview        预览若干行
    POST   /connectors/{id}/import         导入为数据集（→ 新的 DatasetVersion）

路由顺序注意：``/catalog`` 与 ``/test`` 必须声明在 ``/{connector_id}`` 之前，
否则会被路径参数吞掉（``catalog`` 会被当作 int 解析并返回 422）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_connector_service
from app.connectors.service import ConnectorService
from app.schemas.common import ApiResponse, PageInfo, Pagination
from app.schemas.connector import (
    ConnectorCreate,
    ConnectorImportRequest,
    ConnectorImportResponse,
    ConnectorPreviewRequest,
    ConnectorPreviewResponse,
    ConnectorResponse,
    ConnectorTestRequest,
    ConnectorTestResponse,
    ConnectorUpdate,
)

router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.get("/catalog", response_model=ApiResponse[list[dict]])
def get_dialect_catalog() -> ApiResponse[list[dict]]:
    """可用数据库类型与驱动状态（前端下拉框数据源）。"""
    return ApiResponse[list[dict]](data=ConnectorService.catalog())


@router.get("", response_model=ApiResponse[Pagination[ConnectorResponse]])
def list_connectors(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[Pagination[ConnectorResponse]]:
    items, total = service.list(page=page, page_size=page_size)

    return ApiResponse[Pagination[ConnectorResponse]](
        data=Pagination(
            items=[ConnectorService.to_response(item) for item in items],
            page_info=PageInfo.build(page, page_size, total),
        )
    )


@router.post("", response_model=ApiResponse[ConnectorResponse])
def create_connector(
    payload: ConnectorCreate,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorResponse]:
    record = service.create(payload)
    return ApiResponse[ConnectorResponse](data=ConnectorService.to_response(record))


@router.post("/test", response_model=ApiResponse[ConnectorTestResponse])
def test_connector(
    payload: ConnectorTestRequest,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorTestResponse]:
    """试连（不落库）。连接失败是**正常业务结果**，返回 ok=false 而非 HTTP 错误。"""
    return ApiResponse[ConnectorTestResponse](data=service.test_request(payload))


@router.get("/{connector_id}", response_model=ApiResponse[ConnectorResponse])
def get_connector(
    connector_id: int,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorResponse]:
    record = service.get(connector_id)
    return ApiResponse[ConnectorResponse](data=ConnectorService.to_response(record))


@router.patch("/{connector_id}", response_model=ApiResponse[ConnectorResponse])
def update_connector(
    connector_id: int,
    payload: ConnectorUpdate,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorResponse]:
    record = service.update(connector_id, payload)
    return ApiResponse[ConnectorResponse](data=ConnectorService.to_response(record))


@router.delete("/{connector_id}", response_model=ApiResponse[dict])
def delete_connector(
    connector_id: int,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[dict]:
    service.delete(connector_id)
    return ApiResponse[dict](data={"deleted": True, "connector_id": connector_id})


@router.post("/{connector_id}/test", response_model=ApiResponse[ConnectorTestResponse])
def test_saved_connector(
    connector_id: int,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorTestResponse]:
    return ApiResponse[ConnectorTestResponse](data=service.test_saved(connector_id))


@router.get("/{connector_id}/tables", response_model=ApiResponse[dict])
def list_connector_tables(
    connector_id: int,
    schema: str | None = Query(default=None),
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](data=service.list_tables(connector_id, schema=schema))


@router.get("/{connector_id}/tables/{table}/columns", response_model=ApiResponse[dict])
def describe_connector_table(
    connector_id: int,
    table: str,
    schema: str | None = Query(default=None),
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[dict]:
    return ApiResponse[dict](
        data=service.describe_table(connector_id, table=table, schema=schema)
    )


@router.post("/{connector_id}/preview", response_model=ApiResponse[ConnectorPreviewResponse])
def preview_connector_table(
    connector_id: int,
    payload: ConnectorPreviewRequest,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorPreviewResponse]:
    data = service.preview(connector_id, payload)
    return ApiResponse[ConnectorPreviewResponse](
        data=ConnectorPreviewResponse(**data)
    )


@router.post("/{connector_id}/import", response_model=ApiResponse[ConnectorImportResponse])
def import_connector_table(
    connector_id: int,
    payload: ConnectorImportRequest,
    service: ConnectorService = Depends(get_connector_service),
) -> ApiResponse[ConnectorImportResponse]:
    """把表 / 查询抽取为数据集版本。

    这是「突破文件体积限制」的用户可见入口：抽取走分块 + 增量写 row group，
    单次导入规模只受目标磁盘容量约束。
    """
    return ApiResponse[ConnectorImportResponse](data=service.import_table(connector_id, payload))
