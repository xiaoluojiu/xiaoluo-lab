"""Prompt 135：ReportGenerator。

输入 EDA / ML / Quality / Experiment 的结构化结果，
生成 Report（中文、面向毕业设计与正式实验报告的结构）。
只做内容组织，不做任何渲染。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.reports.models import Report, ReportSection


class ReportGenerator:
    """报告生成器。"""

    def generate(
        self,
        *,
        title: str = "数据分析实验报告",
        dataset_info: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        eda: dict[str, Any] | None = None,
        ml: dict[str, Any] | None = None,
        experiments: list[dict[str, Any]] | None = None,
        conclusions: list[str] | None = None,
        charts: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Report:
        dataset_info = dataset_info or {}
        sections: list[ReportSection] = []

        # 一、数据概览
        sections.append(self._overview_section(dataset_info))
        # 二、数据质量
        if quality:
            sections.append(self._quality_section(quality))
        # 三、探索性分析
        eda_section: ReportSection | None = None
        if eda:
            eda_section = self._eda_section(eda)
            sections.append(eda_section)
        # 四、建模与评估
        if ml:
            sections.append(self._ml_section(ml))

        report = Report(
            title=title,
            dataset=dataset_info,
            sections=sections,
            experiments=list(experiments or []),
            charts=list(charts or (eda or {}).get("charts", [])),
            conclusions=list(conclusions or []),
            metadata={
                **(metadata or {}),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generator": "xiaoluo-lab ReportGenerator",
            },
        )
        # 图表自动嵌入「探索性分析」小节（Agent 生成报告时图表产出应自动入报告）
        if eda_section is not None and charts:
            eda_section.charts = list(charts)
        # 结论：外部给定 + 自动归纳
        auto = self._auto_conclusions(dataset_info, quality, eda, ml)
        report.conclusions.extend(c for c in auto if c not in report.conclusions)
        return report

    # ------------------------------------------------------------------
    def _overview_section(self, ds: dict[str, Any]) -> ReportSection:
        name = ds.get("name") or (f"数据集 {ds.get('dataset_id', '-')}" if ds else "未指定")
        lines = [f"本报告的分析对象为数据集「{name}」。"]
        if ds:
            if "rows" in ds:
                lines.append(f"数据规模：{ds['rows']} 行 × {ds.get('columns', '?')} 列。")
            if ds.get("version") is not None:
                lines.append(f"分析版本：v{ds['version']}（版本快照不可变，保证可复现）。")
            schema = ds.get("schema")
            if schema:
                if isinstance(schema, str):  # 上下文截断后的字符串
                    lines.append(f"字段结构（摘要）：{schema}")
                else:
                    cols = "、".join(str(c) for c in schema)
                    lines.append(f"包含字段：{cols}。")
        return ReportSection(heading="一、数据概览", content="\n".join(lines))

    def _quality_section(self, quality: dict[str, Any]) -> ReportSection:
        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []
        issues = quality.get("issues") or quality.get("problems") or []
        score = quality.get("score")
        if score is not None:
            text_parts.append(f"数据质量得分：{score}。")
        if issues:
            text_parts.append(f"共发现 {len(issues)} 个质量问题，明细如下表。")
            rows = (
                [
                    [
                        str(i.get("column", i.get("type", "-"))),
                        str(i.get("type", "-")),
                        str(i.get("message", i.get("detail", "-"))),
                    ]
                    for i in issues
                ]
                if isinstance(issues, list)
                else []
            )
            tables.append(
                {
                    "title": "质量问题明细",
                    "headers": ["位置", "类型", "说明"],
                    "rows": rows,
                }
            )
        else:
            text_parts.append("未发现明显质量问题。")
        # 兜底：把未识别的键做成通用表
        extra = {
            k: v
            for k, v in quality.items()
            if k not in ("issues", "problems", "score") and not isinstance(v, (dict, list))
        }
        if extra:
            tables.append(
                {
                    "title": "其他质量指标",
                    "headers": ["指标", "值"],
                    "rows": [[k, str(v)] for k, v in extra.items()],
                }
            )
        return ReportSection(heading="二、数据质量", content="\n".join(text_parts), tables=tables)

    def _eda_section(self, eda: dict[str, Any]) -> ReportSection:
        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []
        describe = eda.get("describe")
        if describe:
            stats = describe.get("stats") if isinstance(describe, dict) else None
            if isinstance(stats, list):
                # 只把「连续变量」放进描述统计表（均值/std/min/max 对分类编码列无业务意义，
                # 回归：报告曾给 VendorID 算均值 1.754 / 支付方式算均值 1.161）。
                cont_rows = [
                    r for r in stats
                    if not r.get("categorical_encoding") and r.get("mean") is not None
                ]
                if cont_rows:
                    tables.append(
                        {
                            "title": "描述性统计（连续字段）",
                            "headers": ["列", "均值", "标准差", "最小值", "最大值"],
                            "rows": [
                                [
                                    str(r.get("column", "-")),
                                    self._fmt(r.get("mean")),
                                    self._fmt(r.get("std")),
                                    self._fmt(r.get("min")),
                                    self._fmt(r.get("max")),
                                ]
                                for r in cont_rows
                            ],
                        }
                    )
                # 分类编码列单独一张频次表（不掺进均值表）
                cat_rows = [
                    r for r in stats
                    if r.get("categorical_encoding") or (r.get("unique") is not None and r.get("mean") is None)
                ]
                if cat_rows:
                    tables.append(
                        {
                            "title": "分类/编码字段",
                            "headers": ["列", "取值数", "最常见值", "频次"],
                            "rows": [
                                [
                                    str(r.get("column", "-")),
                                    str(r.get("unique", r.get("count", "-"))),
                                    str(r.get("top", "-")),
                                    self._fmt(r.get("freq")),
                                ]
                                for r in cat_rows
                            ],
                        }
                    )
        correlation = eda.get("correlation")
        if correlation:
            pairs = correlation.get("top_pairs") or correlation.get("pairs") or []
            if pairs:
                text_parts.append(
                    "相关性分析显示存在显著相关字段组合（详见下表），建模时需注意多重共线性。"
                )
                tables.append(
                    {
                        "title": "高相关字段对",
                        "headers": ["字段 A", "字段 B", "相关系数"],
                        "rows": [
                            [
                                str(p.get("a", p.get("column_a", "-"))),
                                str(p.get("b", p.get("column_b", "-"))),
                                self._fmt(p.get("correlation", p.get("r"))),
                            ]
                            for p in pairs
                        ],
                    }
                )
        # 分层统计：让「按 X 分层评估」真正落地为一张交叉表，而非只停在建议。
        for strat in eda.get("stratified") or []:
            dim, metric = strat.get("dimension"), strat.get("metric")
            rows = strat.get("rows") or []
            if not rows:
                continue
            text_parts.append(
                f"按 {dim} 分层看 {metric} 的均值与中位数（详见下表），可据此识别结构性差异。"
            )
            tables.append(
                {
                    "title": f"{metric} 分层统计（按 {dim}）",
                    "headers": [str(dim), "均值", "中位数", "样本量"],
                    "rows": [
                        [
                            str(r.get("value", "-")),
                            self._fmt(r.get("mean")),
                            self._fmt(r.get("median")),
                            str(r.get("count", "-")),
                        ]
                        for r in rows
                    ],
                }
            )
        if not text_parts:
            text_parts.append("完成了对数据分布与统计特征的探索（详见各小节数据）。")
        return ReportSection(heading="三、探索性分析", content="\n".join(text_parts), tables=tables)

    def _ml_section(self, ml: dict[str, Any]) -> ReportSection:
        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []
        task = ml.get("task")
        model = ml.get("model")
        if task or model:
            text_parts.append(f"建模任务：{task or '-'}，模型：{model or '-'}。")
        metrics = ml.get("metrics")
        if isinstance(metrics, dict) and metrics:
            text_parts.append("在留出集上的评估指标如下表（训练/评估流程防泄漏，统计量仅在训练集拟合）。")
            tables.append(
                {
                    "title": "评估指标",
                    "headers": ["指标", "值"],
                    "rows": [[k, self._fmt(v)] for k, v in metrics.items()],
                }
            )
        elif metrics:
            text_parts.append(f"评估指标：{metrics}。")
        if ml.get("error"):
            text_parts.append(f"注意：训练过程报告了错误：{ml['error']}。")
        if not text_parts:
            text_parts.append("建模与评估结果待补充。")
        return ReportSection(heading="四、建模与评估", content="\n".join(text_parts), tables=tables)

    # ------------------------------------------------------------------
    def _auto_conclusions(
        self,
        ds: dict[str, Any],
        quality: dict[str, Any] | None,
        eda: dict[str, Any] | None,
        ml: dict[str, Any] | None,
    ) -> list[str]:
        out: list[str] = []
        if ds:
            out.append(
                f"本次分析基于数据集「{ds.get('name', ds.get('dataset_id', '-'))}」"
                f"（{ds.get('rows', '?')} 行 × {ds.get('columns', '?')} 列）。"
            )
        if quality:
            issues = quality.get("issues") or quality.get("problems") or []
            if issues:
                out.append(
                    f"数据存在 {len(issues)} 个质量问题，"
                    "建议在建模前清洗（data.clean / data.filter）。"
                )
            else:
                out.append("数据质量整体良好，未发现明显问题。")
        if ml:
            metrics = ml.get("metrics") or {}
            for key in ("accuracy", "r2", "f1"):
                if key in metrics:
                    label = {"accuracy": "准确率", "r2": "R²", "f1": "F1"}[key]
                    out.append(
                        f"模型{label}为 {self._fmt(metrics[key])}，"
                        "可作为该任务的基线结果，建议进一步做特征工程与参数调优。"
                    )
                    break
        if not out:
            out.append("本次报告未包含足够的分析结果，无法给出结论。")
        return out

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
