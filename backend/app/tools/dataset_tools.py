"""Prompt 092-097：Dataset 只读工具。

dataset.list / dataset.inspect / dataset.preview / dataset.schema /
dataset.profile / dataset.quality —— 全部只读，不产生新版本。
"""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult


class DatasetListTool(Tool):
    """列出数据集（只能读取列表）。"""

    name = "dataset.list"
    description = "列出当前用户可见的数据集（分页），仅读取元信息。"
    category = "dataset"
    input_schema = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 20},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}, "total": {"type": "integer"}},
    }
    permission = "read_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        page = int(params.get("page", 1))
        page_size = int(params.get("page_size", 20))
        items, total = ds.list(page=page, page_size=page_size)
        visible = [d for d in items if context.can_access_dataset(d.id)]
        data = {
            "items": [
                {"id": d.id, "name": d.name, "description": d.description}
                for d in visible
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        return ToolResult.ok(data, summary=f"共 {total} 个数据集，当前页 {len(visible)} 个")


class DatasetInspectTool(Tool):
    """查看数据集详情与版本信息。"""

    name = "dataset.inspect"
    description = "查看数据集元信息、版本列表与最新版本规模。"
    category = "dataset"
    input_schema = {
        "type": "object",
        "properties": {"dataset_id": {"type": "integer"}},
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "read_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        dataset = ds.get(dataset_id)
        versions, total = ds.get_versions(dataset_id, page=1, page_size=20)
        latest = ds.get_version_row(dataset_id)
        data = {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "version_count": total,
            "latest_version": {
                "version": latest.version,
                "rows": latest.row_count,
                "columns": latest.column_count,
                "created_at": str(latest.created_at),
            },
            "versions": [
                {"version": v.version, "rows": v.row_count, "columns": v.column_count}
                for v in versions
            ],
        }
        return ToolResult.ok(data, summary=f"数据集 {dataset.name} 共 {total} 个版本")


class _VersionDataTool(Tool):
    """公共：按 dataset_id(+version) 读取 DataFrame 的只读工具基类。"""

    category = "dataset"
    permission = "read_data"

    def _load_df(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ):
        ds = services.require("dataset_service")
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        version = params.get("version")
        return ds.load_version(dataset_id, int(version) if version else None), dataset_id


class DatasetPreviewTool(_VersionDataTool):
    name = "dataset.preview"
    description = "预览数据集内容（分页，可筛选列与行条件）。"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "page": {"type": "integer", "default": 1},
            "page_size": {"type": "integer", "default": 20},
            "columns": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        preview_params = {
            k: params[k]
            for k in ("page", "page_size", "columns", "sort", "filter_logic")
            if k in params
        }
        if params.get("filters"):
            # 工具面向 Agent 的参数名为 filters，preview 内部参数为 filter_conditions
            preview_params["filter_conditions"] = params["filters"]
        result = engine.preview(df, **preview_params)
        return ToolResult.ok(
            result,
            summary=(
                f"数据集 {dataset_id} 预览：{result['total']} 行，"
                f"第 {result['page']}/{result['total_pages']} 页"
            ),
        )


class DatasetSchemaTool(_VersionDataTool):
    name = "dataset.schema"
    description = "查看数据集列结构（列名、类型、空值、唯一值、样本值）。"
    input_schema = {
        "type": "object",
        "properties": {"dataset_id": {"type": "integer"}, "version": {"type": "integer"}},
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        return ToolResult.ok(engine.schema(df), summary=f"数据集 {dataset_id} 共 {df.width} 列")


class DatasetProfileTool(_VersionDataTool):
    name = "dataset.profile"
    description = "生成数据集统计画像（数值分布、缺失、类别 Top 等）。"
    input_schema = {
        "type": "object",
        "properties": {"dataset_id": {"type": "integer"}, "version": {"type": "integer"}},
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        return ToolResult.ok(engine.profile(df), summary=f"数据集 {dataset_id} 画像完成")


class DatasetQualityTool(_VersionDataTool):
    name = "dataset.quality"
    description = (
        "数据质量检查（缺失、重复、异常值、Schema）。"
        "传 target 时该列只做描述、不产出清洗建议；异常值方法会按字段语义标注适用性。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            # 第二层：目标列保护。用户/Agent 已知目标列时必须传，
            # 否则报告会把「要预测的对象」当成需要清洗的脏数据。
            "target": {"type": "string", "description": "目标列名（可选，传入后该列不做清洗建议）"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(
        self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices
    ) -> ToolResult:
        engine = services.require("data_engine_service")
        df, dataset_id = self._load_df(params, context, services)
        target = str(params.get("target") or "") or None
        report = engine.quality(df, target=target)
        issue_count = len(report.get("issues", []))
        summary = f"数据集 {dataset_id} 质量检查：{issue_count} 个问题"
        if target:
            report.setdefault("target_protection", {
                "target": target,
                "note": f"{target} 已标记为目标列，只做描述、未进入清洗建议",
            })
            summary += f"；目标列 {target} 已排除出清洗建议"
        return ToolResult.ok(report, summary=summary)
