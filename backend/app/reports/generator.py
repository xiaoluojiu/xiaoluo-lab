"""Prompt 135：ReportGenerator。

输入 EDA / ML / Quality / Experiment 的结构化结果，
生成 Report（中文、面向毕业设计与正式实验报告的结构）。
只做内容组织，不做任何渲染。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.reports.layering import build_layers, dedupe
from app.reports.models import Report, ReportSection
from app.reports.numbering import chapter_heading, chapter_status


_CHECK_LABELS = {
    "missing": "缺失值",
    "duplicate": "重复行",
    "outlier": "离群值",
    "schema": "类型/结构",
    "type": "类型",
    "range": "取值范围",
    "unique": "唯一性",
    "constant": "常量列",
}

_SEVERITY_LABELS = {"critical": "严重", "high": "高", "medium": "中", "low": "低", "info": "提示"}

_STAT_LABELS = {
    "row_count": "总行数",
    "column_count": "总列数",
    "missing_cells": "缺失单元格",
    "duplicate_rows": "重复行数",
    "issue_count": "问题条数",
    "checked_by": "已执行检查",
}

#: 离群比例超过这个值就说明「IQR 方法对长尾分布不适用」，而不是数据有问题。
#: 真实事故：DepDelay 被 IQR 判出 1347955 个异常值（13.48%，边界 [-16.5, 19.5]），
#: 报告把它当成 severity=high 的数据质量问题 —— 而它其实是「延误本身」的分布，
#: 按这个口径清洗会把待预测现象整段删掉。
_SKEWED_OUTLIER_RATIO = 0.10


def _skewed_outlier(issue: dict[str, Any]) -> bool:
    """该条是否为「长尾分布被 IQR 误报」的离群问题。"""
    check = str(issue.get("check") or issue.get("type") or "")
    if check != "outlier":
        return False
    details = issue.get("details")
    ratio = details.get("outlier_ratio") if isinstance(details, dict) else None
    return isinstance(ratio, (int, float)) and float(ratio) >= _SKEWED_OUTLIER_RATIO


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
        # 目标列：质量章需要它来判断「被 IQR 判成离群的那一列，是不是待预测现象本身」。
        target_column = (ml or {}).get("target_column") if isinstance(ml, dict) else None
        # 章节标题与序号一律取自 app.reports.numbering 的计划表：
        # 序号不再写死在生成器/导出器里，避免某章缺失时出现「一/二/三/六」跳号。
        sections: list[ReportSection] = []

        # 一、数据概览
        sections.append(self._overview_section(dataset_info))
        # 二、数据质量
        if quality:
            sections.append(self._quality_section(quality, target_column))
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
        # 结论：外部给定 + 自动归纳
        auto = self._auto_conclusions(dataset_info, quality, eda, ml)
        report.conclusions.extend(c for c in auto if c not in report.conclusions)
        # 第六层：结论分「事实 / 判断 / 行动」三层并去重。
        # 去重放在这里而不是渲染层：分层与去重都只依赖文本，
        # 放在生成期一次算好，三个渲染器（md/html/pdf）取同一份结果，不会分叉。
        layers = build_layers(
            list(report.conclusions),
            next_action=ReportGenerator._next_action(dataset_info, quality, eda, ml),
        )
        report.conclusions = dedupe(list(report.conclusions))
        report.metadata["layers"] = layers.to_dict()
        report.metadata["next_action"] = layers.next_action
        # 图表自动嵌入「探索性分析」小节（Agent 生成报告时图表产出应自动入报告）
        if eda_section is not None and charts:
            eda_section.charts = list(charts)
        # 章节计划快照：记录每一章是否生成、未生成的原因。
        # 结论在上一行才补齐，因此状态必须在结论之后再算。
        report.metadata["chapters"] = chapter_status(report)
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
        return ReportSection(heading=chapter_heading("overview"), content="\n".join(lines))

    def _quality_section(
        self, quality: dict[str, Any], target_column: str | None = None
    ) -> ReportSection:
        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []
        issues = quality.get("issues") or quality.get("problems") or []
        severity = quality.get("severity") if isinstance(quality.get("severity"), dict) else {}
        if quality.get("score") is not None:
            text_parts.append(f"数据质量得分：{quality['score']}。")
        if issues:
            summary = "、".join(
                f"{_SEVERITY_LABELS.get(k, k)} {v} 条"
                for k, v in sorted(severity.items(), key=lambda x: -x[1])
                if v
            )
            line = f"共发现 {len(issues)} 个质量问题"
            if summary:
                line += f"（严重度分布：{summary}）"
            text_parts.append(line + "，明细如下表。")
            # 列必须与 QualityIssue 的字段对齐：check/severity/message/column。
            # 回归：渲染器此前读的是不存在的 `type`，导致「类型」列全是 "-"，
            # 「位置」列把 column=None 直接渲染成字符串 "None"。
            rows = []
            for i in issues if isinstance(issues, list) else []:
                if not isinstance(i, dict):
                    continue
                check = str(i.get("check") or i.get("type") or "-")
                column = i.get("column")
                rows.append(
                    [
                        str(column) if column else "全表",
                        _CHECK_LABELS.get(check, check),
                        _SEVERITY_LABELS.get(str(i.get("severity") or ""), str(i.get("severity") or "-")),
                        str(i.get("message") or i.get("detail") or "-"),
                    ]
                )
            tables.append(
                {
                    "title": "质量问题明细",
                    "headers": ["位置", "类型", "严重度", "说明"],
                    "rows": rows,
                }
            )
            # 离群值的口径说明：离群 ≠ 脏数据。长尾分布（延误分钟/距离/金额）按
            # IQR 会判出大量「异常值」，这类问题必须先解释再谈清洗。
            for i in issues:
                if not isinstance(i, dict) or not _skewed_outlier(i):
                    continue
                details = i.get("details") if isinstance(i.get("details"), dict) else {}
                ratio = details.get("outlier_ratio")
                column = i.get("column")
                note = (
                    f"注意：{column or '该字段'} 的离群比例达 {float(ratio):.2%}，"
                    "说明该列是明显右偏/长尾分布，IQR 边界在此不适用 —— "
                    "这部分记录是分布本身的尾部，不等于录入错误，清洗前需确认业务含义。"
                )
                if target_column and column == target_column:
                    note += (
                        f"此外该列正是本次建模的目标列（{target_column}），"
                        "其长尾就是待预测现象本身，不应作为异常值删除。"
                    )
                text_parts.append(note)
        else:
            text_parts.append("未发现明显质量问题。")
        # 统计口径：statistics 是 dict，此前被「排除 dict/list」的兜底逻辑整段丢掉，
        # 「其他质量指标」表只剩一个布尔标志。
        stats = quality.get("statistics")
        stat_rows: list[list[str]] = []
        if isinstance(stats, dict):
            for key, value in stats.items():
                label = _STAT_LABELS.get(key, key)
                if isinstance(value, (list, tuple)):
                    value = "、".join(str(v) for v in value)
                elif isinstance(value, dict):
                    continue
                stat_rows.append([label, str(value)])
        if severity:
            dist = "、".join(
                f"{_SEVERITY_LABELS.get(k, k)} {v}" for k, v in severity.items() if v
            )
            stat_rows.append(["严重度分布", dist or "无"])
        if "has_errors" in quality:
            # has_errors 只看 severity，不区分「真脏数据」与「长尾列被 IQR 误报」。
            # 报告要给出可执行的口径，所以这里额外算一次「真正需要先处理」的条数。
            blocking = [
                i
                for i in issues
                if isinstance(i, dict)
                and str(i.get("severity")) in ("high", "critical")
                and not _skewed_outlier(i)
            ]
            stat_rows.append(["是否存在高严重度问题", "是" if quality["has_errors"] else "否"])
            stat_rows.append(
                [
                    "其中需要建模前处理的问题数",
                    str(len(blocking))
                    + ("（其余为长尾列的 IQR 离群判定，见上方说明）" if quality["has_errors"] and not blocking else ""),
                ]
            )
        if stat_rows:
            tables.append({"title": "质量统计口径", "headers": ["指标", "值"], "rows": stat_rows})
        recs = quality.get("recommendations")
        if isinstance(recs, list) and recs:
            tables.append(
                {
                    "title": "质量修复建议",
                    "headers": ["#", "建议"],
                    "rows": [[str(i + 1), str(r)] for i, r in enumerate(recs)],
                }
            )
        # 兜底：把未识别的标量键做成通用表（跳过内部标志位）
        extra = {
            k: v
            for k, v in quality.items()
            if k not in ("issues", "problems", "score", "severity", "statistics", "recommendations", "has_errors")
            and not isinstance(v, (dict, list))
        }
        if extra:
            tables.append(
                {
                    "title": "其他质量指标",
                    "headers": ["指标", "值"],
                    "rows": [[k, str(v)] for k, v in extra.items()],
                }
            )
        return ReportSection(heading=chapter_heading("quality"), content="\n".join(text_parts), tables=tables)

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
        return ReportSection(heading=chapter_heading("eda"), content="\n".join(text_parts), tables=tables)

    def _ml_section(self, ml: dict[str, Any]) -> ReportSection:
        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []
        task = ml.get("task")
        model = ml.get("model")
        if task or model:
            text_parts.append(f"建模任务：{task or '-'}，模型：{model or '-'}。")
        # 必须写明「预测的是哪一列」：回归任务里目标列一旦被误判（例如本该预测
        # DepDelay 却被当成普通特征），读者只看指标数字完全无法发现。
        target = ml.get("target_column")
        if task == "clustering":
            text_parts.append("聚类任务无目标列，模型在全部特征上学习分组结构。")
        elif target:
            text_parts.append(f"预测目标列：{target}。")
        elif task in ("classification", "regression"):
            text_parts.append("注意：本次实验未记录预测目标列，指标口径无法复核。")
        sampling = ml.get("sampling")
        if isinstance(sampling, dict) and sampling.get("sampled"):
            text_parts.append(
                f"训练集抽样：原始 {sampling.get('original_rows')} 行，"
                f"实际使用 {sampling.get('used_rows')} 行"
                f"（抽样率 {self._fmt(sampling.get('sample_rate'))}）——"
                "大规模数据下为避免密集矩阵爆内存而抽样，指标反映的是该样本上的表现。"
            )
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
        status = ml.get("status")
        if status and status != "success":
            text_parts.append(f"本次运行的最终状态为 {status}，指标可能不完整。")
        if ml.get("error"):
            text_parts.append(f"注意：训练过程报告了错误：{ml['error']}。")
        if not text_parts:
            text_parts.append("建模与评估结果待补充。")
        return ReportSection(heading=chapter_heading("ml"), content="\n".join(text_parts), tables=tables)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 第六层：报告结尾必须回答「下一步该跑什么实验」
    # ------------------------------------------------------------------
    @staticmethod
    def _next_action(
        ds: dict[str, Any],
        quality: dict[str, Any] | None,
        eda: dict[str, Any] | None,
        ml: dict[str, Any] | None,
    ) -> str:
        """给出可执行的下一步（不是「建议进一步分析」这类空话）。"""
        metrics = dict((ml or {}).get("metrics") or {})
        target = (ml or {}).get("target_column") or ""
        issues = [i for i in ((quality or {}).get("issues") or []) if isinstance(i, dict)]
        blocking = [
            i for i in issues
            if str(i.get("severity")) in ("high", "critical") and not _skewed_outlier(i)
        ]
        if blocking and not metrics:
            return (
                f"先处理 {len(blocking)} 个阻塞性质量问题（data.clean / data.filter），"
                "再执行建模链路（ml.detect_task → ml.prepare → ml.train → ml.evaluate）。"
            )
        if metrics:
            scope = f"（目标列 {target}）" if target else ""
            return (
                f"以当前结果{scope}为基线：做时间字段解析与高基数类别编码后重跑同任务，"
                "用 MAE 与分层误差复核大值区间，再决定是否换非线性模型。"
            )
        if ml:
            return "建模步骤已执行但没有产出指标：先看 run.error 定位失败原因，再重跑。"
        return (
            "执行建模链路（ml.detect_task → ml.prepare → ml.train → ml.evaluate），"
            "先把基线跑出来再谈调优。"
        )

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
            issues = [i for i in issues if isinstance(i, dict)]
            if issues:
                blocking = [
                    i
                    for i in issues
                    if str(i.get("severity")) in ("high", "critical") and not _skewed_outlier(i)
                ]
                if blocking:
                    out.append(
                        f"数据存在 {len(issues)} 个质量问题（其中 {len(blocking)} 个需在建模前处理），"
                        "建议清洗（data.clean / data.filter）后再建模。"
                    )
                else:
                    out.append(
                        f"数据存在 {len(issues)} 个质量问题，但均为长尾列的 IQR 离群判定"
                        "（非录入错误），是否清洗应结合业务含义决定，"
                        "注意不要删除本身就是待预测现象的尾部记录。"
                    )
            else:
                out.append("数据质量整体良好，未发现明显问题。")
        if ml:
            metrics = ml.get("metrics") or {}
            target = ml.get("target_column")
            for key in ("accuracy", "r2", "f1"):
                if key in metrics:
                    label = {"accuracy": "准确率", "r2": "R²", "f1": "F1"}[key]
                    scope = f"（目标列 {target}）" if target else ""
                    out.append(
                        f"模型{label}为 {self._fmt(metrics[key])}{scope}，"
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
