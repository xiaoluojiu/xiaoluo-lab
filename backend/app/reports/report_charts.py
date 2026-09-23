"""report_charts.py —— 自动策划报告图表（科研级）并渲染为内联 SVG。

供 report.generate 工具与 POST /reports/generate 调用：基于数据集**特征**生成一组
「科研级」图表并渲染为内联 SVG，供 HTML / Markdown / PDF 导出嵌入。

历史问题（两轮修复）：
1. 图表组合写死（前 2 数值列直方图 + 热力图 + 散点 + 前 2 分类柱状 + Q-Q + CDF），
   不同数据集产出的图千篇一律 → 改为特征感知策划（line/boxplot/grouped_bar 按需出图）。
2. 无「变量类型意识」：VendorID/payment_type/RatecodeID/PULocationID/DOLocationID 这类
   低基数整型编码列被当成连续变量，算均值/std、进相关性热力图、做 Q-Q/箱线图，
   直方图长尾不裁剪导致正常值全挤进首 bin → 本文件据此引入连续/分类分离 + 长尾裁剪。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from app.analysis import (
    CorrelationAnalyzer,
    VisualizationBuilder,
    classify_columns,
    is_categorical_like,
)
from app.reports.chart_svg import to_svg

logger = logging.getLogger(__name__)

MAX_CHARTS = 12

# 必保图表类型：这三类是「一次完整分析」的硬性交付，必须优先占位，
# 其余补充图只能在剩余额度里插空，避免被 MAX_CHARTS 截断挤掉。
_REQUIRED_TYPES = ("histogram", "heatmap", "scatter")

# 分类列基数上限：基数过大的分类列画柱状图没有信息量（几百根柱子），直接跳过。
_MAX_CATEGORY_CARDINALITY = 30

# 直方图长尾裁剪分位：明显长尾（max > 3×该分位）时截到该分位再分桶。
_HIST_CLIP_QUANTILE = 0.99


def _column_classes(df: pl.DataFrame) -> dict[str, list[str]]:
    """按类型给列分桶，供策划器决定出哪些图。

    「连续变量」与「分类变量」分离：低基数数值编码列（VendorID/ratecodeID/区域 ID/
    Month/DayOfWeek 这类浮点日历字段）归入 categorical，不再被当连续变量算均值、
    做 Q-Q/箱线/热力图。

    用一次 ``classify_columns`` 拿到全部判定，而不是 ``continuous_columns`` +
    逐个 ``is_categorical_like``：后者会把每列的唯一值数重复算很多遍。
    """
    classes = classify_columns(df)
    continuous: list[str] = []
    categorical: list[str] = []
    temporal: list[str] = []
    for c, d in df.schema.items():
        if d.is_numeric() and not classes.get(c, False):
            continuous.append(c)
        elif d.is_temporal():
            temporal.append(c)
        elif d != pl.Boolean:
            categorical.append(c)
    return {"continuous": continuous, "categorical": categorical, "temporal": temporal}


def _cardinality(df: pl.DataFrame, col: str) -> int:
    """分类列的不同取值个数（用于判断是否适合画柱状图）。"""
    try:
        return int(df[col].drop_nulls().n_unique())
    except Exception:  # noqa: BLE001
        return 0


def build_report_charts(df: pl.DataFrame, max_charts: int = MAX_CHARTS) -> list[dict[str, Any]]:
    """为数据集生成一组报告图表（含 SVG 字符串），图表组合随数据特征变化。

    策划规则（按优先级，额度耗尽即停）：
      必保：分布直方图 / 相关性热力图 / 相关性散点图（至少各一张）
      时间序列：有「时间列 + 连续列」时出折线图（趋势）
      分布形态：连续列出箱线图（离群点 / 分位），有低基数分类列则分组对比
      分组对比：有「低基数分类列 × 连续列」时出分组柱状图
      补充：类别柱状图、正态 Q-Q、CDF 面积图
    类型约束：直方图/箱线图/Q-Q/CDF 只对**连续变量**；分类编码列只画柱状图。
    """
    classes = _column_classes(df)
    cont = classes["continuous"]
    categorical = classes["categorical"]
    temporal = classes["temporal"]
    vb = VisualizationBuilder()
    state = {"charts": [], "max": max_charts}
    used_hist: list[str] = []

    def _add(chart_type: str, title: str, data: dict[str, Any]) -> None:
        if len(state["charts"]) >= state["max"]:
            return
        svg = to_svg(data)
        if not svg:
            return
        state["charts"].append({"type": chart_type, "title": title, "svg": svg, "data": data})

    # ---------- 必保 1：分布直方图（前 2 个连续列，长尾裁剪） ----------
    for col in cont[:2]:
        try:
            data = vb.analyze(df, chart="histogram", column=col, bins=12, clip_quantile=_HIST_CLIP_QUANTILE)
            title = f"{col} 分布直方图"
            if data.get("clipped"):
                title += "（已裁剪极端值）"
            _add("histogram", title, data)
            used_hist.append(col)
        except Exception:
            continue

    # ---------- 必保 2：相关系数热力图（仅连续变量） ----------
    if len(cont) >= 2:
        try:
            data = vb.analyze(df, chart="heatmap", columns=cont)
            if data.get("matrix"):
                _add("heatmap", "连续字段相关性热力图", data)
        except Exception:
            pass

    # ---------- 必保 3：相关性散点图（相关性最强的一对，无论强弱都出图） ----------
    best: tuple[str, str, float | None] | None = None
    if len(cont) >= 2:
        try:
            corr = CorrelationAnalyzer().analyze(df, method="auto", columns=cont)
            matrix = corr.get("matrix") or {}
            cols = list(matrix) or cont[:2]
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    r = matrix[a].get(b)
                    if r is None:
                        continue
                    if best is None or abs(r) > abs(best[2] or 0.0):
                        best = (a, b, float(r))
        except Exception:
            pass
        if best is None:
            best = (cont[0], cont[1], None)
        x_col, y_col, r = best
        try:
            data = vb.analyze(df, chart="scatter", x=x_col, y=y_col)
            suffix = f"（r={r:.2f}）" if r is not None else ""
            _add("scatter", f"相关性散点：{x_col} × {y_col}{suffix}", data)
        except Exception:
            pass

    # ---------- 时间序列：折线图（趋势） ----------
    # 时间列可作为有序 x 轴，配合一个连续列画趋势。
    if temporal and cont:
        x = temporal[0]
        y = next((c for c in cont if c not in used_hist), cont[0])
        try:
            data = vb.analyze(df, chart="line", x=x, y=y)
            # 附加趋势摘要：让 LLM 叙述能基于真实数值解读趋势，而不是贴了图却说「无法判断」
            # （回归：报告既贴了〈trip_distance 随时间变化趋势〉图，正文却写「趋势暂无法判断」）。
            ys = [v for v in (data.get("y") or []) if v is not None]
            if len(ys) >= 2:
                first, last = float(ys[0]), float(ys[-1])
                trend = "上升" if last > first * 1.02 else ("下降" if last < first * 0.98 else "基本平稳")
                data["trend_summary"] = {
                    "x": x, "y": y,
                    "first": round(first, 4), "last": round(last, 4),
                    "avg": round(sum(ys) / len(ys), 4),
                    "direction": trend,
                }
            _add("line", f"{y} 随时间变化趋势", data)
        except Exception:
            pass

    # ---------- 分布形态：箱线图（仅连续列） ----------
    # 有低基数分类列时按分类分组对比更直观。
    box_col = cont[0] if cont else None
    if box_col:
        try:
            group_col = next(
                (c for c in categorical if 0 < _cardinality(df, c) <= 12), None
            )
            if group_col:
                data = vb.analyze(df, chart="boxplot", column=box_col, group_by=group_col)
                _add("boxplot", f"{box_col} 分位分布（按 {group_col}）", data)
            else:
                data = vb.analyze(df, chart="boxplot", column=box_col)
                _add("boxplot", f"{box_col} 箱线图（离群点）", data)
        except Exception:
            pass

    # ---------- 分组对比：分组柱状图 / 分类柱状图 ----------
    usable_cat = [c for c in categorical if 0 < _cardinality(df, c) <= _MAX_CATEGORY_CARDINALITY]
    if len(usable_cat) >= 2 and cont:
        col, group_by = usable_cat[0], usable_cat[1]
        y = cont[0]
        try:
            data = vb.analyze(df, chart="grouped_bar", column=col, group_by=group_by, y=y, agg="mean")
            _add("grouped_bar", f"{y} 均值：{col} × {group_by}", data)
        except Exception:
            pass
    # 剩余低基数分类列画普通柱状图
    for col in usable_cat[:3]:
        try:
            data = vb.analyze(df, chart="bar", column=col, top_n=12)
            _add("bar", f"{col} 类别分布", data)
        except Exception:
            continue

    # ---------- 补充：更多连续列的直方图（累计最多 4 个） ----------
    for col in cont:
        if col in used_hist or len(used_hist) >= 4:
            continue
        try:
            data = vb.analyze(df, chart="histogram", column=col, bins=12, clip_quantile=_HIST_CLIP_QUANTILE)
            title = f"{col} 分布直方图"
            if data.get("clipped"):
                title += "（已裁剪极端值）"
            _add("histogram", title, data)
            used_hist.append(col)
        except Exception:
            continue

    # ---------- 补充：正态 Q-Q 图（仅连续列，前 2 个） ----------
    # 分类编码列（VendorID/ratecodeID 等）不做 Q-Q：离散值做正态检验是伪影，无意义。
    for col in cont[:2]:
        try:
            data = vb.analyze(df, chart="qq", column=col)
            _add("qq", f"{col} 正态性 Q-Q 图", data)
        except Exception:
            continue

    # ---------- 补充：累积分布 CDF 面积图（仅连续列） ----------
    if cont:
        try:
            data = vb.analyze(df, chart="area", column=cont[0], bins=12)
            _add("area", f"{cont[0]} 累积分布 (CDF)", data)
        except Exception:
            pass

    charts = state["charts"]
    # 兜底自检：若因异常导致必保类型缺失，标记为缺失以便上层感知（不静默吞掉）
    missing = [t for t in _REQUIRED_TYPES if not any(c["type"] == t for c in charts)]
    if missing:
        logger.warning("report_charts: 缺少必保图表类型 %s", missing)
    return charts[:max_charts]
