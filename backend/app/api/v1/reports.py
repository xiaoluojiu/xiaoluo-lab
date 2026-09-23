"""Reports API（Prompt 203）。

- POST /reports/generate  生成结构化报告（概览/质量/EDA/ML/结论）
- POST /reports/export    导出 markdown / html / pdf
- GET/DELETE /reports/saved  管理已保存报告
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.analysis import CorrelationAnalyzer, DescriptiveAnalyzer
from app.api.deps import get_data_engine_service, get_storage_service
from app.core.exceptions import ValidationException
from app.data_engine.service import DataEngineService
from app.notifications import notify
from app.reports.generator import ReportGenerator
from app.reports.html import export_html
from app.reports.markdown import export_markdown
from app.reports.models import Report, ReportSection
from app.reports.pdf import export_pdf
from app.reports.report_charts import build_report_charts
from app.reports.saved import invalidate_cache, list_metadata, meta_key, save_report
from app.schemas.common import ApiResponse
from app.storage.service import StorageService

router = APIRouter(prefix="/reports", tags=["reports"])


class GenerateRequest(BaseModel):
    dataset_id: int
    version: int | None = None
    title: str = "数据分析实验报告"
    include_quality: bool = True
    include_eda: bool = True
    include_ml: bool = True
    conclusions: list[str] = Field(default_factory=list)
    narrate: bool = True
    request: str = ""


class ExportRequest(BaseModel):
    report: dict[str, Any]
    format: Literal["markdown", "html", "pdf"] = "markdown"


def _section_from_dict(item: Any) -> ReportSection:
    """小节 dict -> ReportSection。

    `ReportSection(**s)` 直接解包外部输入：只要 sections 里有非 dict 元素、或带
    ReportSection 不认识的键，就会抛 TypeError 并被兜底成 500。这里改为按字段取值，
    非法元素报 422 并带上位置，便于前端定位。
    """
    if isinstance(item, ReportSection):
        return item
    if not isinstance(item, dict):
        raise ValidationException(
            "报告小节格式不正确（应为对象）",
            details={"section": str(item)[:200]},
        )
    return ReportSection(
        heading=str(item.get("heading") or ""),
        content=str(item.get("content") or ""),
        tables=[t for t in (item.get("tables") or []) if isinstance(t, dict)],
        charts=[c for c in (item.get("charts") or []) if isinstance(c, dict)],
    )


def _report_from_dict(data: dict[str, Any]) -> Report:
    """前端/上游传回的报告 dict -> Report（导出器只渲染不改动内容）。"""
    return Report(
        title=str(data.get("title", "报告")),
        dataset=dict(data.get("dataset") or {}),
        sections=[_section_from_dict(s) for s in data.get("sections", [])],
        experiments=list(data.get("experiments", [])),
        charts=list(data.get("charts", [])),
        conclusions=[str(c) for c in data.get("conclusions", [])],
        metadata=dict(data.get("metadata") or {}),
    )


@router.post("/generate", response_model=ApiResponse[dict])
def generate_report(
    body: GenerateRequest,
    service: DataEngineService = Depends(get_data_engine_service),
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[dict]:
    version_row = service.dataset_service.get_version_row(body.dataset_id, body.version)
    ds_service = service.dataset_service
    df = ds_service.load_version(body.dataset_id, version_row.version)

    dataset_info: dict[str, Any] = {
        "dataset_id": body.dataset_id,
        "name": ds_service.get(body.dataset_id).name,
        "rows": df.height,
        "columns": df.width,
        "version": version_row.version,
    }
    quality = service.quality(df) if body.include_quality else None
    eda = None
    if body.include_eda:
        describe = DescriptiveAnalyzer().analyze(df)
        corr = CorrelationAnalyzer().analyze(df)
        pairs: list[dict[str, Any]] = []
        matrix = corr.get("matrix") or {}
        cols = list(matrix)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = matrix[a].get(b)
                if r is not None and abs(r) >= 0.8:
                    pairs.append({"a": a, "b": b, "correlation": round(r, 4)})
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        eda = {
            "describe": {"stats": describe.get("columns", [])},
            "correlation": {"pairs": pairs[:10]},
        }
    ml: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = []
    if body.include_ml:
        # 与 Agent 工具 report.generate 共用同一份发现逻辑（app.reports.discovery），
        # 否则两条入口会分叉：Agent 报告曾完全没有「建模与评估」章。
        from app.reports.discovery import discover_ml_context

        ctx = discover_ml_context(ds_service.db, ds_service, body.dataset_id)
        ml, experiments = ctx.ml, ctx.experiments

    report = ReportGenerator().generate(
        title=body.title,
        dataset_info=dataset_info,
        quality=quality,
        eda=eda,
        ml=ml,
        experiments=experiments,
        conclusions=body.conclusions or None,
        charts=build_report_charts(df),
    )
    report_dict = report.to_dict()

    # LLM 基于真实工具结果撰写正文（失败自动降级为模板报告）
    if body.narrate:
        from app.reports.narrator import default_provider, narrate_and_apply

        report_dict = narrate_and_apply(report_dict, default_provider(), user_request=body.request)

    report_key = f"reports/{uuid.uuid4().hex}.json"
    report_dict.setdefault("metadata", {})["report_key"] = report_key

    # 落正文 + 轻量元数据副本（列表接口只读后者），并使列表缓存失效。
    save_report(storage, report_key, report_dict)
    notify(
        "report",
        "报告已生成",
        f"《{report_dict.get('title', '报告')}》已就绪，可查看、导出或分享。",
        link="/reports",
    )
    return ApiResponse[dict](data=report_dict)


@router.post("/export")
def export_report(body: ExportRequest) -> Response:
    """按格式导出报告。markdown/html 返回文本，pdf 返回二进制流。"""
    report = _report_from_dict(body.report)
    if body.format == "markdown":
        return Response(content=export_markdown(report), media_type="text/markdown; charset=utf-8")
    if body.format == "html":
        return HTMLResponse(content=export_html(report))
    content = export_pdf(report)
    return Response(content=content, media_type="application/pdf")


def _validate_saved_key(key: str) -> str:
    """只允许 reports/*.json，拒绝路径穿越。"""
    if (
        not key.startswith("reports/")
        or not key.endswith(".json")
        or ".." in key
        or "\\" in key
        or key.count("/") != 1
    ):
        raise HTTPException(status_code=400, detail="非法的报告 key")
    return key


def _meta_key(report_key: str) -> str:
    """报告正文 key -> 轻量元数据 key（reports/xxx.json -> reports/xxx.meta.json）。"""
    return meta_key(report_key)


@router.get("/saved", response_model=ApiResponse[list[dict[str, Any]]])
def list_saved_reports(
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[list[dict[str, Any]]]:
    """报告列表（**只返回元数据**，不读正文）。

    实现与缓存策略在 :mod:`app.reports.saved`；Agent 工具的写路径也走同一处，
    避免两条入口各自优化一半。
    """
    return ApiResponse[list[dict[str, Any]]](data=list_metadata(storage))


@router.get("/saved/detail", response_model=ApiResponse[dict])
def get_saved_report(
    key: str = Query(...),
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[dict]:
    _validate_saved_key(key)
    try:
        payload = json.loads(storage.read(key).decode("utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="报告不存在")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"报告读取失败：{exc}")
    return ApiResponse[dict](data=payload)


@router.delete("/saved", response_model=ApiResponse[dict])
def delete_saved_report(
    key: str = Query(...),
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[dict]:
    """删除一份已保存的正式报告。"""
    _validate_saved_key(key)
    try:
        storage.delete(key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="报告不存在")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"报告删除失败：{exc}")
    # 同步清理元数据副本（不存在则忽略，兼容历史报告）。
    try:
        storage.delete(_meta_key(key))
    except Exception:
        pass
    invalidate_cache()
    return ApiResponse[dict](data={"deleted": True, "key": key})
