"""chart_svg.py —— 将 VisualizationBuilder 的图表输出渲染为内联 SVG（零依赖）。

服务于「Agent 生成报告时图表自动嵌入报告」：所有图表统一 800×450（16:9）
视口、科研级配色与坐标轴，输出 <svg> 字符串，可直接嵌入 HTML / Markdown。

支持类型：histogram / bar / line / scatter / boxplot / heatmap / qq /
grouped_bar / area。
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import quote

# 画布
W, H = 800, 450
ML, MR, MT, MB = 72, 26, 42, 62  # 左/右/上/下边距
PW = W - ML - MR
PH = H - MT - MB

# 配色（与前端 chartColors 同源）
PALETTE = [
    "#4f46e5", "#16a34a", "#d97706", "#dc2626",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
]
PRIMARY = "#4f46e5"
GRID = "#eef2f7"
AXIS = "#94a3b8"
TEXT = "#475569"
TITLE = "#1e293b"
_BG = "#ffffff"

FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif"

#: 单张图的点上限。超过就等距抽样。
#: 一张 800×450 的图里 3000 个点已经互相盖满（纯噪声），而每个点写成一个
#: ``<circle>`` 要 ~74 字符。真实事故：报告里两张 Q-Q 图各 5000 点，光点集就
#: 366 KB，占全部 SVG 的 87%，整份 Markdown 报告被撑到 1.39 MB。
_MAX_POINTS = 3000

#: 点集用「零长度路径 + 圆头线帽」绘制，直径即 stroke-width（约等于 r=2.6 的圆点）。
_POINT_DIAMETER = 5.2


def _downsample(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], int]:
    """点数超过上限时等距抽样；返回 (抽样后点集, 原始点数或 0)。"""
    n = len(points)
    if n <= _MAX_POINTS:
        return points, 0
    step = math.ceil(n / _MAX_POINTS)
    kept = points[::step]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept, n


def _points_layer(
    points: list[tuple[float, float]],
    *,
    color: str,
    opacity: float,
) -> tuple[str, str]:
    """把点集渲染成**单条** ``<path>``，并返回 (svg, 抽样说明)。

    ``h0`` 是零长度水平线，配合 ``stroke-linecap="round"`` 即一个圆点；
    坐标取整后用相对移动 ``m dx dy`` 连接，每点约 10 字符（``<circle>`` 约 74），
    在 5000 点的 Q-Q 图上实测体积降到约 1/6。
    """
    if not points:
        return "", ""
    kept, original = _downsample(points)
    xi, yi = int(round(kept[0][0])), int(round(kept[0][1]))
    segs = [f"M{xi} {yi}h0"]
    for px, py in kept[1:]:
        nx, ny = int(round(px)), int(round(py))
        segs.append(f"m{nx - xi} {ny - yi}h0")
        xi, yi = nx, ny
    svg = (
        f'<path d="{"".join(segs)}" fill="none" stroke="{color}" '
        f'stroke-width="{_POINT_DIAMETER:g}" stroke-linecap="round" '
        f'stroke-opacity="{opacity:g}"/>'
    )
    note = ""
    if original:
        note = (
            f'<text x="{ML}" y="{H - 10}" font-size="10" fill="{TEXT}">'
            f"共 {original} 个点，为控制体积等距抽样显示 {len(kept)} 个</text>"
        )
    return svg, note


def to_svg(chart: dict[str, Any]) -> str:
    """把单张图表 dict 渲染为完整 <svg> 字符串。"""
    builders = {
        "histogram": _bar,
        "bar": _bar,
        "line": _xy_line,
        "scatter": _scatter,
        "boxplot": _box,
        "heatmap": _heat,
        "qq": _qq,
        "grouped_bar": _grouped_bar,
        "area": _area,
    }
    fn = builders.get(chart.get("chart"))
    if not fn:
        return _unsupported(chart.get("chart"))
    return fn(chart)


def to_svg_data_uri(chart: dict[str, Any]) -> str:
    """渲染为可直接用于 <img src> 的 data URI。"""
    raw = to_svg(chart)
    return "data:image/svg+xml;utf8," + quote(raw)


# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------

def _svg_open(title: str | None = None, height: int = H) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" role="img" font-family="{FONT}">',
        f'<rect x="0" y="0" width="{W}" height="{height}" fill="{_BG}"/>',
    ]
    if title:
        parts.append(
            f'<text x="{ML}" y="26" font-size="15" font-weight="600" fill="{TITLE}">'
            f"{_esc(title)}</text>"
        )
    return "".join(parts)


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v: float) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e9:
            return str(int(v))
        return f"{v:.3g}"
    return str(v)


def _nice_num(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / (10 ** exp)
    nf = 1.0 if f < 1.5 else 2.0 if f < 3 else 5.0 if f < 7 else 10.0
    return nf * (10 ** exp)


def _ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if vmax <= vmin:
        vmax = vmin + 1.0
    step = _nice_num((vmax - vmin) / max(1, count - 1))
    start = math.floor(vmin / step) * step
    out: list[float] = []
    v = start
    while v <= vmax + step * 1e-6:
        out.append(round(v, 6))
        v += step
    return out


def _y_axis(vmin: float, vmax: float, count: int = 5) -> tuple[str, float, float]:
    """绘制左 Y 轴网格与刻度，返回 (svg片段, 像素/单位y比例, y_min像素)。"""
    ticks = _ticks(vmin, vmax, count)
    parts = []
    y_lo = MT + PH
    y_hi = MT
    for t in ticks:
        y = y_lo + (vmin - t) / (vmax - vmin) * (y_hi - y_lo)
        y = max(y_hi, min(y_lo, y))
        parts.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ML - 8}" y="{y + 4:.1f}" font-size="11" fill="{TEXT}" '
            f'text-anchor="end">{_fmt(t)}</text>'
        )
    parts.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{y_lo:.1f}" stroke="{AXIS}" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{ML + PW}" y1="{MT}" x2="{ML + PW}" y2="{y_lo:.1f}" stroke="{AXIS}" stroke-width="1"/>'
    )
    return "".join(parts), (y_lo - y_hi) / (vmax - vmin), y_lo


def _x_categorical(n: int) -> list[float]:
    """返回 n 个分类 x 的中心像素坐标。"""
    if n <= 0:
        return []
    band = PW / n
    return [ML + band * (i + 0.5) for i in range(n)]


# ----------------------------------------------------------------------
# 柱状 / 直方图
# ----------------------------------------------------------------------

def _bar(chart: dict[str, Any]) -> str:
    x = [str(v) for v in (chart.get("x") or [])]
    y = [float(v) if v is not None else 0.0 for v in (chart.get("y") or [])]
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not x or not y:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    ymax = max(y) * 1.05 or 1.0
    grid, sy, _ = _y_axis(0, ymax)
    out.append(grid)
    centers = _x_categorical(len(x))
    band = PW / len(x)
    bw = band * 0.78
    for cx, val in zip(centers, y):
        top = MT + PH - val * sy
        out.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{max(0.0, MT + PH - top):.1f}" fill="{PRIMARY}" rx="2"/>'
        )
    out.append(_x_labels(centers, x))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 折线图
# ----------------------------------------------------------------------

def _xy_line(chart: dict[str, Any]) -> str:
    x = [str(v) for v in (chart.get("x") or [])]
    y = [float(v) if v is not None else 0.0 for v in (chart.get("y") or [])]
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not x or not y:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    ymax = max(y) * 1.05 or 1.0
    ymin = min(0.0, min(y))
    grid, sy, y_lo = _y_axis(ymin, ymax)
    out.append(grid)
    centers = _x_categorical(len(x))
    pts = " ".join(f"{cx:.1f},{y_lo - val * sy:.1f}" for cx, val in zip(centers, y))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="{PRIMARY}" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    layer, note = _points_layer(
        [(cx, y_lo - val * sy) for cx, val in zip(centers, y)],
        color=PRIMARY,
        opacity=1.0,
    )
    out.append(layer)
    if note:
        out.append(note)
    out.append(_x_labels(centers, x))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 散点图
# ----------------------------------------------------------------------
def _scatter(chart: dict[str, Any]) -> str:
    xs = [float(v) for v in (chart.get("x") or []) if v is not None]
    ys = [float(v) for v in (chart.get("y") or []) if v is not None]
    title = _chart_title(chart)
    out = [_svg_open(title)]
    pairs = list(zip(xs, ys))
    if not pairs:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmin, xmax = _pad(xmin, xmax)
    ymin, ymax = _pad(ymin, ymax)
    grid, sy, y_lo = _y_axis(ymin, ymax)
    out.append(grid)
    sx = PW / (xmax - xmin)
    x0 = ML
    # x 轴刻度
    for t in _ticks(xmin, xmax, 5):
        px = x0 + (t - xmin) * sx
        out.append(
            f'<text x="{px:.1f}" y="{y_lo + 18:.1f}" font-size="11" fill="{TEXT}" '
            f'text-anchor="middle">{_fmt(t)}</text>'
        )
    out.append(f'<line x1="{ML}" y1="{y_lo:.1f}" x2="{ML + PW}" y2="{y_lo:.1f}" stroke="{AXIS}"/>')
    px_pts = [
        (x0 + (xv - xmin) * sx, y_lo - (yv - ymin) * sy) for xv, yv in pairs
    ]
    layer, note = _points_layer(px_pts, color=PRIMARY, opacity=0.62)
    out.append(layer)
    if note:
        out.append(note)
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 箱线图
# ----------------------------------------------------------------------

def _box(chart: dict[str, Any]) -> str:
    boxes = chart.get("boxes") or []
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not boxes:
        out.append(_empty("无可绘制的箱线数据"))
        out.append("</svg>")
        return "".join(out)
    vals = []
    for b in boxes:
        for key in ("whisker_low", "whisker_high", "q1", "q3", "median"):
            v = b.get(key)
            if v is not None:
                vals.append(float(v))
        for o in (b.get("outliers") or []):
            if o is not None:
                vals.append(float(o))
    ymin, ymax = min(vals), max(vals)
    ymin, ymax = _pad(ymin, ymax)
    grid, sy, y_lo = _y_axis(ymin, ymax)
    out.append(grid)
    centers = _x_categorical(len(boxes))
    band = PW / len(boxes)
    bw = min(band * 0.5, 60)
    for cx, b in zip(centers, boxes):
        wl = float(b.get("whisker_low", b.get("q1")))
        wh = float(b.get("whisker_high", b.get("q3")))
        q1 = float(b.get("q1"))
        q3 = float(b.get("q3"))
        med = float(b.get("median"))
        y_wl = y_lo - (wl - ymin) * sy
        y_wh = y_lo - (wh - ymin) * sy
        y_q1 = y_lo - (q1 - ymin) * sy
        y_q3 = y_lo - (q3 - ymin) * sy
        y_med = y_lo - (med - ymin) * sy
        out.append(f'<line x1="{cx:.1f}" y1="{y_wl:.1f}" x2="{cx:.1f}" y2="{y_wh:.1f}" stroke="{AXIS}"/>')
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{y_q3:.1f}" width="{bw:.1f}" height="{max(0.0, y_q1 - y_q3):.1f}" fill="{PRIMARY}" fill-opacity="0.28" stroke="{PRIMARY}" stroke-width="1.4"/>')
        out.append(f'<line x1="{cx - bw / 2:.1f}" y1="{y_med:.1f}" x2="{cx + bw / 2:.1f}" y2="{y_med:.1f}" stroke="{PRIMARY}" stroke-width="2.2"/>')
        out.append(f'<line x1="{cx - bw * 0.18:.1f}" y1="{y_wl:.1f}" x2="{cx + bw * 0.18:.1f}" y2="{y_wl:.1f}" stroke="{AXIS}"/>')
        out.append(f'<line x1="{cx - bw * 0.18:.1f}" y1="{y_wh:.1f}" x2="{cx + bw * 0.18:.1f}" y2="{y_wh:.1f}" stroke="{AXIS}"/>')
        # 离群点可能上千个，同样压成单条 path（每个 <circle> 约 74 字符）
        oy = [
            (cx, y_lo - (float(o) - ymin) * sy)
            for o in (b.get("outliers") or [])
            if o is not None
        ]
        layer, _ = _points_layer(oy, color=PRIMARY, opacity=0.55)
        out.append(layer)
    out.append(_x_labels(centers, [str(b.get("label", "")) for b in boxes]))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 热力图（相关性）
# ----------------------------------------------------------------------

def _heat(chart: dict[str, Any]) -> str:
    cols = [str(c) for c in (chart.get("columns") or [])]
    matrix = chart.get("matrix") or {}
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if len(cols) < 2:
        out.append(_empty("暂无可绘制的相关性矩阵"))
        out.append("</svg>")
        return "".join(out)
    label_w = 130
    top = MT + 10
    side = ML - 20
    cell = min((W - side - MR - label_w) / len(cols), (H - top - MB) / (len(cols) + 1))
    cell = max(28, min(cell, 70))
    # 列头
    for j, c in enumerate(cols):
        cx = side + label_w + cell * (j + 0.5)
        out.append(
            f'<text x="{cx:.1f}" y="{top - 8:.1f}" font-size="11" fill="{TEXT}" '
            f'text-anchor="middle" transform="rotate(-35 {cx:.1f} {top - 8:.1f})">{_esc(c)}</text>'
        )
    for i, r in enumerate(cols):
        ry = top + cell * i + cell / 2
        out.append(f'<text x="{side + label_w - 8:.1f}" y="{ry + 4:.1f}" font-size="11" fill="{TEXT}" text-anchor="end">{_esc(r)}</text>')
        for j, c in enumerate(cols):
            v = matrix.get(r, {}).get(c) if isinstance(matrix.get(r), dict) else None
            if isinstance(matrix, list):
                v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            v = float(v) if isinstance(v, (int, float)) else None
            fill = _heat_color(v)
            x = side + label_w + cell * j
            out.append(f'<rect x="{x:.1f}" y="{top + cell * i:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" fill="{fill}" stroke="#ffffff"/>')
            if v is not None:
                out.append(
                    f'<text x="{x + cell / 2 - 1:.1f}" y="{ry + 4:.1f}" font-size="10.5" '
                    f'fill="#0f172a" text-anchor="middle">{v:.2f}</text>'
                )
    out.append("</svg>")
    return "".join(out)


def _heat_color(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "rgba(148,163,184,0.12)"
    t = min(1, max(0, (v + 1) / 2))
    r = round(255 + (79 - 255) * t)
    g = round(255 + (70 - 255) * t)
    b = round(255 + (229 - 255) * t)
    return f"rgb({r},{g},{b})"


# ----------------------------------------------------------------------
# Q-Q 图
# ----------------------------------------------------------------------

def _qq(chart: dict[str, Any]) -> str:
    xs = [float(v) for v in (chart.get("x") or [])]
    ys = [float(v) for v in (chart.get("y") or [])]
    line = chart.get("line") or {}
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not xs or not ys:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    xmin, xmax = _pad(min(xs), max(xs))
    ymin, ymax = _pad(min(ys), max(ys))
    grid, sy, y_lo = _y_axis(ymin, ymax)
    out.append(grid)
    sx = PW / (xmax - xmin)
    x0 = ML
    for t in _ticks(xmin, xmax, 5):
        px = x0 + (t - xmin) * sx
        out.append(f'<text x="{px:.1f}" y="{y_lo + 18:.1f}" font-size="11" fill="{TEXT}" text-anchor="middle">{_fmt(t)}</text>')
    out.append(f'<line x1="{ML}" y1="{y_lo:.1f}" x2="{ML + PW}" y2="{y_lo:.1f}" stroke="{AXIS}"/>')
    qq_pts = [(x0 + (xv - xmin) * sx, y_lo - (yv - ymin) * sy) for xv, yv in zip(xs, ys)]
    layer, note = _points_layer(qq_pts, color=PRIMARY, opacity=0.6)
    out.append(layer)
    if note:
        out.append(note)
    # 参考线
    lx0, ly0 = float(line.get("x0", xmin)), float(line.get("y0", ymin))
    lx1, ly1 = float(line.get("x1", xmax)), float(line.get("y1", ymax))
    px0 = x0 + (lx0 - xmin) * sx
    py0 = y_lo - (ly0 - ymin) * sy
    px1 = x0 + (lx1 - xmin) * sx
    py1 = y_lo - (ly1 - ymin) * sy
    out.append(f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{px1:.1f}" y2="{py1:.1f}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="6 4"/>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 分组柱状图
# ----------------------------------------------------------------------

def _grouped_bar(chart: dict[str, Any]) -> str:
    x = [str(v) for v in (chart.get("x") or [])]
    groups = [str(g) for g in (chart.get("groups") or [])]
    series = chart.get("series") or {}
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not x or not groups:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    all_vals = [v for g in groups for v in series.get(g, []) if v is not None]
    ymax = (max(all_vals) * 1.05 if all_vals else 1.0) or 1.0
    grid, sy, _ = _y_axis(0, ymax)
    out.append(grid)
    centers = _x_categorical(len(x))
    band = PW / len(x)
    ng = len(groups)
    gw = min(band * 0.82, 120)
    slot = gw / ng
    for ci, cx in enumerate(centers):
        for gi, g in enumerate(groups):
            val = series.get(g, [None] * len(x))[ci]
            if val is None:
                continue
            bx = cx - gw / 2 + gi * slot
            top = MT + PH - float(val) * sy
            out.append(
                f'<rect x="{bx + 1:.1f}" y="{top:.1f}" width="{slot - 2:.1f}" '
                f'height="{max(0.0, MT + PH - top):.1f}" fill="{PALETTE[gi % len(PALETTE)]}" rx="1.5"/>'
            )
    out.append(_x_labels(centers, x))
    # 图例
    lx = ML
    ly = MT - 14
    for gi, g in enumerate(groups):
        out.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="11" height="11" fill="{PALETTE[gi % len(PALETTE)]}"/>')
        out.append(f'<text x="{lx + 16:.1f}" y="{ly:.1f}" font-size="11" fill="{TEXT}">{_esc(g)}</text>')
        lx += 22 + len(g) * 11 + 14
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 面积图（CDF）
# ----------------------------------------------------------------------

def _area(chart: dict[str, Any]) -> str:
    x = [str(v) for v in (chart.get("x") or [])]
    y = [float(v) if v is not None else 0.0 for v in (chart.get("y") or [])]
    title = _chart_title(chart)
    out = [_svg_open(title)]
    if not x or not y:
        out.append(_empty("无可用数据"))
        out.append("</svg>")
        return "".join(out)
    grid, sy, y_lo = _y_axis(0, 1.0)
    out.append(grid)
    centers = _x_categorical(len(x))
    pts = [(cx, y_lo - val * sy) for cx, val in zip(centers, y)]
    poly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in pts)
    baseline = f"{centers[0]:.1f},{y_lo:.1f} {centers[-1]:.1f},{y_lo:.1f}"
    out.append(f'<polygon points="{baseline} {poly}" fill="{PRIMARY}" fill-opacity="0.16"/>')
    out.append(f'<polyline points="{poly}" fill="none" stroke="{PRIMARY}" stroke-width="2.2"/>')
    layer, note = _points_layer(pts, color=PRIMARY, opacity=1.0)
    out.append(layer)
    if note:
        out.append(note)
    out.append(_x_labels(centers, x))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------

def _chart_title(chart: dict[str, Any]) -> str | None:
    col = chart.get("column")
    name = chart.get("chart")
    if name == "heatmap":
        return "相关性热力图"
    if name == "qq":
        return f"正态性 Q-Q 图 · {col}" if col else "正态性 Q-Q 图"
    if name == "area":
        return f"累积分布 (CDF) · {col}" if col else "累积分布"
    if name == "grouped_bar":
        return "分组柱状图"
    if name in ("histogram", "bar"):
        return f"分布 · {col}" if col else "分布"
    if name == "boxplot":
        return f"箱线图 · {col}" if col else "箱线图"
    if name == "scatter":
        return "散点图"
    return None


def _x_labels(centers: list[float], labels: list[str]) -> str:
    parts = []
    rotate = len(labels) > 7
    for cx, lab in zip(centers, labels):
        if rotate:
            parts.append(
                f'<text x="{cx:.1f}" y="{MT + PH + 16:.1f}" font-size="10.5" fill="{TEXT}" '
                f'text-anchor="end" transform="rotate(-32 {cx:.1f} {MT + PH + 16:.1f})">{_esc(lab)}</text>'
            )
        else:
            parts.append(
                f'<text x="{cx:.1f}" y="{MT + PH + 18:.1f}" font-size="11" fill="{TEXT}" '
                f'text-anchor="middle">{_esc(lab)}</text>'
            )
    return "".join(parts)


def _pad(lo: float, hi: float) -> tuple[float, float]:
    if hi == lo:
        return lo - 1.0, hi + 1.0
    span = hi - lo
    pad = span * 0.05
    return lo - pad, hi + pad


def _empty(msg: str) -> str:
    return (
        f'<text x="{W / 2}" y="{H / 2}" font-size="13" fill="{TEXT}" '
        f'text-anchor="middle">{_esc(msg)}</text>'
    )


def _unsupported(chart: Any) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
        f'<rect width="{W}" height="{H}" fill="#fff"/>'
        f'<text x="{W / 2}" y="{H / 2}" font-size="13" fill="{TEXT}" text-anchor="middle">'
        f"不支持的图表类型：{_esc(chart)}</text></svg>"
    )
