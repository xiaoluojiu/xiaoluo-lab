"""Report Agent Tool：基于真实工具结果生成并持久化结构化报告。

职责：
1. 跑真实分析（质量 / 描述统计 / 相关性）+ 自动图表（含散点图、相关系数热力图、分布直方图）；
2. 调 LLM 基于**这些真实结果**撰写报告正文与结论（不是模板句，也不是空话）；
3. 落盘到报告中心，并返回 report_key 供前端直接打开。
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import polars as pl

from app.agent.permission.models import Permission
from app.data_engine.json_utils import json_safe
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

logger = logging.getLogger(__name__)

# 规划器有时会把「待基于 xxx 汇总」这类占位句当作结论传进来。
# 这类文本会被写进报告正文，读者无法分辨，宁可在入口就拦掉。
_PLACEHOLDER_MARKERS = ("待基于", "待补充", "待填写", "占位", "todo", "tbd", "xxxx")


def _clean_conclusions(raw: Any) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        text = str(item).strip()
        if not text:
            continue
        if any(marker in text.lower() for marker in _PLACEHOLDER_MARKERS):
            logger.info("丢弃占位结论：%s", text[:60])
            continue
        out.append(text)
    return out


def _stratified_stats(df: Any, *, max_groups: int = 3, max_metrics: int = 2) -> list[dict[str, Any]]:
    """生成「低基数分类列 × 连续列」的分层统计（均值/中位数交叉表）。

    让报告的分层分析落到实处：对每个低基数分类列，计算各取值下连续指标的均值与中位数，
    输出成 {dimension, metric, rows:[{value, mean, median, count}]} 列表，供报告渲染。
    只取前 max_groups 个分类列、前 max_metrics 个连续列，避免 300 万行数据过度聚合。
    """
    from app.analysis import categorical_columns, continuous_columns

    cat_cols = categorical_columns(df)
    cont_cols = continuous_columns(df)
    out: list[dict[str, Any]] = []
    for dim in cat_cols[:max_groups]:
        n = int(df[dim].drop_nulls().n_unique())
        if not (0 < n <= 20):
            continue
        for metric in cont_cols[:max_metrics]:
            try:
                agg = (
                    df.select([dim, metric])
                    .drop_nulls()
                    .group_by(dim)
                    .agg([
                        pl.col(metric).mean().alias("mean"),
                        pl.col(metric).median().alias("median"),
                        pl.col(metric).count().alias("count"),
                    ])
                    .sort("mean", descending=True)
                )
                rows = [
                    {
                        "value": json_safe(row[0]),
                        "mean": round(float(row[1]), 4) if row[1] is not None else None,
                        "median": round(float(row[2]), 4) if row[2] is not None else None,
                        "count": int(row[3]),
                    }
                    for row in agg.iter_rows()
                ]
                if rows:
                    out.append({"dimension": dim, "metric": metric, "rows": rows})
            except Exception:  # noqa: BLE001 - 分层聚合失败不阻断报告
                logger.debug("分层统计失败：%s × %s", dim, metric, exc_info=True)
    return out


_STORAGE_KEY_PREFIX = "reports/"


class ReportGenerateTool(Tool):
    name = "report.generate"
    description = (
        "基于真实数据质量、EDA 和相关分析结果生成并保存一份完整的结构化分析报告，"
        "报告自动嵌入多张图表（分布直方图、类别柱状图、相关系数热力图、相关性散点图、正态 Q-Q 图、累积分布图），"
        "并由大模型针对真实数值撰写章节正文与结论。request 参数建议填写用户的原始诉求，"
        "模型会据此组织正文侧重点。返回 report_key 用于在报告中心打开。"
    )
    category = "report"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "title": {"type": "string"},
            "request": {"type": "string", "description": "用户的原始分析诉求，用于让 LLM 撰写有针对性的正文"},
            "conclusions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = Permission.EXPORT_REPORT

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.analysis import CorrelationAnalyzer, DescriptiveAnalyzer
        from app.reports.generator import ReportGenerator
        from app.reports.narrator import default_provider, narrate_and_apply
        from app.reports.report_charts import build_report_charts
        from app.storage.service import get_storage

        dataset_id = int(params["dataset_id"])
        ds = services.require("dataset_service")
        engine = services.require("data_engine_service")
        version = ds.get_version_row(dataset_id, None)
        df = ds.load_version(dataset_id, version.version)
        info = {
            "dataset_id": dataset_id,
            "name": ds.get(dataset_id).name,
            "rows": df.height,
            "columns": df.width,
            "version": version.version,
            "schema": list(df.columns),
        }
        quality = engine.quality(df)
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
        charts = build_report_charts(df)
        # 分层统计：低基数分类列 × 连续列的均值/中位数交叉，让报告的分层分析「已做」而非「建议做」
        # （回归：报告反复建议「按 Borough/VendorID/payment_type 分层评估」，正文却只有单变量分布图）。
        stratified = _stratified_stats(df)
        report = ReportGenerator().generate(
            title=str(params.get("title") or "AI 数据分析实验报告"),
            dataset_info=info,
            quality=quality,
            eda={
                "describe": {"stats": describe.get("columns", [])},
                "correlation": {"pairs": sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:10]},
                "stratified": stratified,
            },
            charts=charts,
            conclusions=_clean_conclusions(params.get("conclusions")),
        )
        data = report.to_dict()

        # LLM 撰写正文：输入只有真实工具结果，输出追加到对应小节
        user_request = str(params.get("request") or "").strip()
        narration_applied = False
        try:
            payload_before = json.dumps(data, ensure_ascii=False, default=str)
            data = narrate_and_apply(data, default_provider(), user_request=user_request)
            narration_applied = json.dumps(data, ensure_ascii=False, default=str) != payload_before
        except Exception:  # noqa: BLE001 - 叙述失败不能让报告生成失败
            logger.exception("报告叙述失败，降级为模板正文")
            data = report.to_dict()

        key = f"{_STORAGE_KEY_PREFIX}{uuid.uuid4().hex}.json"
        get_storage().save(key, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        data.setdefault("metadata", {})["report_key"] = key

        chart_types = sorted({c.get("type") for c in charts if isinstance(c, dict)})
        summary = (
            f"数据集 {dataset_id} 报告生成完成：{len(data.get('sections') or [])} 个章节、"
            f"{len(charts)} 张图表（{'/'.join(chart_types) or '无'}），"
            f"{'含 LLM 正文叙述' if narration_applied else '模板正文'}，已保存为 {key}"
        )
        # 给 LLM 的视图必须极小：完整报告含图表 SVG/数据表，直接进上下文会击穿 Token 预算。
        compact = {
            "report_key": key,
            "title": data.get("title"),
            "sections": [
                {
                    "heading": s.get("heading"),
                    "charts": [c.get("title") for c in (s.get("charts") or [])],
                }
                for s in (data.get("sections") or [])
            ],
            "chart_types": chart_types,
            "conclusions": (data.get("conclusions") or [])[:8],
            "narrated": narration_applied,
        }
        return ToolResult.ok(data=data, summary=summary, compact_data=compact, metadata={"report_key": key, "chart_count": len(charts)})
