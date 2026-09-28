"""Reports API（Prompt 203）。

- POST /reports/generate  生成结构化报告（概览/质量/EDA/ML/结论）
- POST /reports/export    导出 markdown / html / pdf
- GET/DELETE /reports/saved  管理已保存报告
- POST /reports/saved/batch-delete  批量删除（删除选中 / 一键删除全部）
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


class BatchDeleteRequest(BaseModel):
    """批量删除已保存报告的入参。

    「删除选中」与「一键删除全部」共用这一个端点：后者把当前列表里所有 key 传进来，
    不必再维护第二条清空路径（两条路径迟早分叉，见 ``_delete_saved_report_files``）。
    """

    keys: list[str] = Field(default_factory=list)


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


def _delete_saved_report_files(storage: StorageService, key: str) -> None:
    """删掉一份报告的正文 + 元数据副本（key 合法性由调用方先行校验）。

    单条删除与批量删除**共用这一份实现**：早先「正文删了、.meta.json 留着」
    这类分叉会让列表接口读到幽灵条目（缓存签名变了、文件却还在）。
    元数据副本不存在时忽略 —— 兼容没有副本的历史报告。

    ⚠️ 存在性必须先探再删，不能靠捕获 ``FileNotFoundError``：存储层
    （``LocalStorage.delete``）在对象不存在时抛的是 ``StorageException(code="NOT_FOUND")``，
    于是「再删一次同一份报告」原本会返回 500 而不是 404 —— 一个真实存在过、
    但只在重复删除时才暴露的缺陷。这里统一成显式 ``FileNotFoundError``，
    让两条调用路径都能按「不存在」处理。
    """
    if not storage.exists(key):
        raise FileNotFoundError(key)
    storage.delete(key)
    try:
        storage.delete(_meta_key(key))
    except Exception:  # noqa: BLE001 - 副本缺失/清理失败不影响正文已删除的事实
        pass


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
        raise HTTPException(status_code=404, detail="报告不存在") from None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"报告读取失败：{exc}") from exc
    return ApiResponse[dict](data=payload)


@router.delete("/saved", response_model=ApiResponse[dict])
def delete_saved_report(
    key: str = Query(...),
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[dict]:
    """删除一份已保存的正式报告。"""
    _validate_saved_key(key)
    try:
        _delete_saved_report_files(storage, key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="报告不存在") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"报告删除失败：{exc}") from exc
    invalidate_cache()
    return ApiResponse[dict](data={"deleted": True, "key": key})


@router.post("/saved/batch-delete", response_model=ApiResponse[dict])
def delete_saved_reports(
    body: BatchDeleteRequest,
    storage: StorageService = Depends(get_storage_service),
) -> ApiResponse[dict]:
    """批量删除已保存报告（前端「删除选中」与「一键删除全部」共用）。

    逐条回执而不是「一失败就整批 500」：所选报告里有几条已被别处删掉是常见情况，
    整批失败会让用户面对一堆「删不掉」却不知道是哪几条、也不知道其余到底删没删。
    这里返回 ``deleted_keys`` / ``failed`` 两份清单，前端据此提示。
    """
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    # dict.fromkeys 去重并保序：前端可能因「先点单条删除再点全选」传入重复 key。
    for key in dict.fromkeys(body.keys):
        try:
            _validate_saved_key(key)
        except HTTPException as exc:
            failed.append({"key": str(key), "error": str(exc.detail)})
            continue
        try:
            _delete_saved_report_files(storage, key)
        except FileNotFoundError:
            failed.append({"key": key, "error": "报告不存在或已被删除"})
            continue
        except Exception as exc:  # noqa: BLE001 - 单条失败不阻断其余
            failed.append({"key": key, "error": f"删除失败：{exc}"})
            continue
        deleted.append(key)
    if deleted:
        invalidate_cache()
    return ApiResponse[dict](
        data={
            "deleted": not failed,
            "count": len(deleted),
            "deleted_keys": deleted,
            "failed": failed,
        }
    )
