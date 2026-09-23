"""数据库连接器 Agent 工具。

把「接入外部数据库」变成 Agent 可用的一等能力——用户可以直接说
「把生产库的 orders 表导进来看看」，而不必先去拓展中心配好再手动导入。

风险分级（与 app/agent/permission/rules.py 登记一致）
----------------------------------------------------
- ``connector.list`` / ``connector.tables`` / ``connector.preview``：只读，LOW。
  注意 preview 虽然只读，但会真实打到外部库上，因此仍需在 context 中
  具备对应权限（沿用 READ_DATA）。
- ``connector.import``：会把外部数据落成新的 DatasetVersion（写类操作、
  可能抽很多行），按项目的风险惯例定 **HIGH** 并强制用户确认。
"""

from __future__ import annotations

from typing import Any

from app.schemas.connector import ConnectorImportRequest, ConnectorPreviewRequest
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult


class ConnectorListTool(Tool):
    """列出已配置的外部数据库连接器。"""

    name = "connector.list"
    description = "列出已配置的数据库连接器（外部数据源），查看其类型与最近一次连接状态。"
    category = "connector"
    input_schema = {
        "type": "object",
        "properties": {"page": {"type": "integer", "default": 1}, "page_size": {"type": "integer", "default": 20}},
    }
    output_schema = {"type": "object", "properties": {"items": {"type": "array"}, "total": {"type": "integer"}}}
    permission = "read_data"

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        svc = services.require("connector_service")
        page = int(params.get("page", 1))
        page_size = int(params.get("page_size", 20))
        items, total = svc.list(page=page, page_size=page_size)

        data = {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "dialect": item.dialect,
                    "database": item.database,
                    "last_status": item.last_status,
                    "dataset_id": item.dataset_id,
                }
                for item in items
            ],
            "total": total,
        }
        return ToolResult.ok(data, summary=f"共 {total} 个数据库连接器")


class ConnectorTablesTool(Tool):
    """列出连接器下的表 / 视图。"""

    name = "connector.tables"
    description = "列出某个数据库连接器中的表与视图，用于确定要导入哪张表。"
    category = "connector"
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "integer"},
            "schema": {"type": "string"},
        },
        "required": ["connector_id"],
    }
    output_schema = {"type": "object", "properties": {"tables": {"type": "array"}, "total": {"type": "integer"}}}
    permission = "read_data"

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        svc = services.require("connector_service")
        connector_id = int(params["connector_id"])
        data = svc.list_tables(connector_id, schema=params.get("schema"))
        return ToolResult.ok(
            data,
            summary=f"连接器 {connector_id} 下有 {data.get('total', 0)} 张表 / 视图",
        )


class ConnectorPreviewTool(Tool):
    """预览外部表的少量行。"""

    name = "connector.preview"
    description = "预览数据库表的前若干行（只读，不导入数据），用于确认字段含义。"
    category = "connector"
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "integer"},
            "table": {"type": "string"},
            "schema_name": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "where": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["connector_id", "table"],
    }
    output_schema = {"type": "object"}
    permission = "read_data"

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        svc = services.require("connector_service")
        payload = ConnectorPreviewRequest(
            table=str(params["table"]),
            schema_name=params.get("schema_name"),
            columns=params.get("columns"),
            where=params.get("where"),
            limit=int(params.get("limit", 50)),
        )
        data = svc.preview(int(params["connector_id"]), payload)
        return ToolResult.ok(
            data,
            summary=f"预览返回 {data.get('row_count', 0)} 行、{len(data.get('columns') or [])} 列",
        )


class ConnectorImportTool(Tool):
    """把外部表导入为数据集（高风险）。"""

    name = "connector.import"
    description = (
        "把数据库连接器中的表 / 查询抽取为平台内的数据集版本（会写入数据，"
        "可能需要较长时间）。抽取按批进行，可处理远超上传上限的数据量。"
    )
    category = "connector"
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "integer"},
            "table": {"type": "string"},
            "schema_name": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "where": {"type": "string"},
            "order_by": {"type": "string"},
            "keyset_column": {"type": "string"},
            "max_rows": {"type": "integer"},
            "dataset_id": {"type": "integer"},
            "dataset_name": {"type": "string"},
        },
        "required": ["connector_id", "table"],
    }
    output_schema = {"type": "object"}
    permission = "read_data"
    requires_confirmation = True

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        svc = services.require("connector_service")

        payload = ConnectorImportRequest(
            table=str(params["table"]),
            schema_name=params.get("schema_name"),
            columns=params.get("columns"),
            where=params.get("where"),
            order_by=params.get("order_by"),
            keyset_column=params.get("keyset_column"),
            max_rows=params.get("max_rows"),
            dataset_id=params.get("dataset_id"),
            dataset_name=params.get("dataset_name"),
        )

        result = svc.import_table(int(params["connector_id"]), payload)

        # 导入后新数据集对本会话可见，否则后续步骤（profile / EDA）会被权限拦下。
        context.dataset_ids.add(result.dataset_id)

        data = result.model_dump()
        return ToolResult.ok(
            data,
            summary=(
                f"已导入 {result.row_count} 行 / {result.column_count} 列到数据集 "
                f"{result.dataset_id}（v{result.dataset_version}，策略 {result.strategy}）"
            ),
        )
