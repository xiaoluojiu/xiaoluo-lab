"""单次 ML Run 详细报告装配。

把一次训练运行（run.metrics + run.artifacts）装配成 app.reports.models.Report，
随后复用既有导出器（HTML / Markdown / PDF）渲染。

与 ReportGenerator 的分工：后者面向"整份数据集分析报告"（质量/EDA/建模），
本模块面向"一次训练的完整留痕"——配置、指标、混淆矩阵、特征重要性、
学习曲线、校准、阈值曲线、复现信息全部落表。
"""

from __future__ import annotations

import base64
import io as _io
import json
import logging
import re
from datetime import datetime
from typing import Any

import matplotlib

# 无显示器的服务端必须显式选 Agg；否则 matplotlib 会在导入时去找 GUI 后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - 必须在 use("Agg") 之后
from matplotlib import font_manager  # noqa: E402

from app.reports.models import Report, ReportSection

logger = logging.getLogger(__name__)

# 中文标签默认会渲染成方块（DejaVu Sans 没有汉字），这里按系统实际有的字体挑一个。
# 找不到就保持默认 —— 报告仍能读（数字与英文正常），只是中文会退化成方框，
# 总好过因为字体缺失让整份报告生成失败。
for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(_font in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font]
        break
# 负号在部分中文字体里会显示成方框
plt.rcParams["axes.unicode_minus"] = False

# 中文序号按**小节**取值，不能按字符下标取：
# 写成字符串后 _CN[10] 只会拿到「十」里的第二个字符，第十一节会显示成「十、」。
_CN = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
       "十一", "十二", "十三", "十四", "十五")
_INTEGER_KEY = re.compile(r"(_seed$|_count$|^n_[a-z]+$|^k$|^cluster_count$)")


def _cn(index: int) -> str:
    return _CN[index] if index < len(_CN) else str(index + 1)


def _fmt(key: str, value: Any) -> str:
    """指标/配置值的展示格式。

    整数型指标（计数、seed、簇数）必须显示为整数：报告里出现
    「cluster_count 1534.0000」会让读者怀疑这个数是算出来的还是编出来的。
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if _INTEGER_KEY.search(key):
            return str(int(round(value)))
        return f"{value:.4f}"
    return str(value)


def _png(fig: Any) -> str:
    """Figure -> base64 PNG 字符串（HTML 内联用，单文件分发不需要附带图片文件）。"""
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_chart(builder: Any, payload: Any) -> list[dict[str, str]]:
    """画一张图；失败只记日志并返回空列表。

    图是「锦上添花」：表格里的数字才是报告的主体，任何绘图异常
    （数据形状异常、字体缺失、后端问题）都不该让整份报告导不出来。
    """
    try:
        image = builder(payload)
    except Exception as exc:  # noqa: BLE001 - 绘图失败不影响报告主体
        logger.info("报告图表生成失败 %s: %s", getattr(builder, "__name__", "?"), exc)
        return []
    return [image] if image else []


def _learning_curve_chart(lc: dict[str, Any]) -> dict[str, str] | None:
    """学习曲线：训练集 / 测试集分数 vs 训练样本数（两条线的间距就是过拟合程度）。"""
    points = lc.get("points") or []
    if len(points) < 2:
        return None
    rows = [p.get("rows") for p in points]
    train = [p.get("train_score") for p in points]
    test = [p.get("test_score") for p in points]
    metric = lc.get("metric") or "分数"
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(rows, train, marker="o", label="训练集")
    ax.plot(rows, test, marker="s", label="测试集")
    ax.set_xlabel("训练样本数")
    ax.set_ylabel(metric)
    ax.set_title(f"学习曲线（{metric}）")
    ax.grid(alpha=0.3)
    ax.legend()
    return {"title": f"学习曲线（{metric} vs 训练样本量）", "png_base64": _png(fig)}


def _confusion_chart(cm: dict[str, Any]) -> dict[str, str] | None:
    """混淆矩阵热力图：行=真实，列=预测（与表格同一份 matrix）。"""
    labels = [str(x) for x in (cm.get("labels") or [])]
    matrix = cm.get("matrix") or []
    if not labels or not matrix:
        return None
    flat = [v for row in matrix for v in row if isinstance(v, (int, float))]
    half = (max(flat) / 2) if flat else 0
    size = max(3.0, 0.55 * len(labels) + 2.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（行=真实，列=预测）")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j, i, str(value), ha="center", va="center",
                color="white" if isinstance(value, (int, float)) and value > half else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    return {"title": "混淆矩阵热力图", "png_base64": _png(fig)}


def _importance_chart(fi: dict[str, Any]) -> dict[str, str] | None:
    """特征重要性横向条形图（前 15 项；barh 自下而上画，故先升序）。"""
    items = list(fi.get("importances") or [])[:15]
    if not items:
        return None
    items.sort(key=lambda it: float(it.get("importance") or 0))
    names = [str(it.get("feature")) for it in items]
    values = [float(it.get("importance") or 0) for it in items]
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.34 * len(items) + 1.2)))
    ax.barh(names, values, color="#2c5aa0")
    ax.set_xlabel("重要性")
    ax.set_title(f"特征重要性（{fi.get('method') or '-'}，前 {len(items)} 项）")
    ax.grid(axis="x", alpha=0.3)
    return {"title": f"特征重要性（{fi.get('method') or '-'}）", "png_base64": _png(fig)}


def _calibration_chart(cal: dict[str, Any]) -> dict[str, str] | None:
    """可靠性曲线：贴对角线=校准好，偏离越大越不能按概率做决策。"""
    points = cal.get("points") or []
    if not points:
        return None
    x = [float(p.get("mean_predicted") or 0) for p in points]
    y = [float(p.get("observed_frequency") or 0) for p in points]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "k--", label="完全校准")
    ax.plot(x, y, marker="o", label="本模型")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际正类比例")
    ax.set_title(f"可靠性曲线（ECE={cal.get('ece')}，MCE={cal.get('mce')}）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    return {"title": "概率校准（可靠性曲线）", "png_base64": _png(fig)}


def _threshold_chart(tc: dict[str, Any]) -> dict[str, str] | None:
    """阈值扫描：精确率 / 召回率 / F1 随阈值的变化，最高点就是推荐阈值的来源。"""
    points = tc.get("points") or []
    if len(points) < 2:
        return None
    t = [float(p.get("threshold") or 0) for p in points]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, label in (("precision", "精确率"), ("recall", "召回率"), ("f1", "F1")):
        values = [p.get(key) for p in points]
        if any(isinstance(v, (int, float)) for v in values):
            ax.plot(t, values, label=label)
    suggested = tc.get("suggested_threshold")
    if isinstance(suggested, (int, float)):
        ax.axvline(float(suggested), color="#999", linestyle=":", label=f"推荐 {suggested:g}")
    ax.set_xlabel("阈值")
    ax.set_ylabel("分数（正类口径）")
    ax.set_title("阈值取舍曲线")
    ax.grid(alpha=0.3)
    ax.legend()
    return {"title": "决策阈值取舍曲线", "png_base64": _png(fig)}


def build_run_report(run: Any, *, dataset_name: str = "") -> Report:
    metrics = dict(run.metrics or {})
    art = dict(run.artifacts or {})
    task = str(art.get("task") or "")
    model_name = str(art.get("model") or "")
    target = art.get("target_column")
    sections: list[ReportSection] = []

    # 一、实验配置与数据预算
    config_rows: list[list[Any]] = [
        ["任务类型", task],
        ["模型", model_name],
        ["目标列", target or "（无，聚类）"],
        ["测试集比例 test_size", art.get("test_size")],
        ["随机种子 seed", art.get("seed")],
        ["是否分层切分", art.get("stratified")],
        ["训练集行数", art.get("train_rows")],
        ["测试集行数", art.get("test_rows")],
        ["目标列空值剔除行数", art.get("dropped_rows")],
        ["排除的特征列", "、".join(map(str, art.get("excluded_columns") or [])) or "无"],
        ["参与训练的原始特征列", "、".join(map(str, art.get("features") or []))],
        ["预处理后模型特征数", len(art.get("model_features") or [])],
    ]
    sampling = art.get("sampling")
    if isinstance(sampling, dict) and sampling.get("sampled"):
        config_rows.extend([
            ["训练集抽样", f"是（{sampling.get('limit_source') or '系统上限'}）"],
            ["抽样前行数", sampling.get("original_rows")],
            ["实际使用行数", sampling.get("used_rows")],
            ["抽样比例", _fmt("rate", sampling.get("sample_rate"))],
        ])
    sections.append(ReportSection(
        heading="实验配置与数据预算",
        content="本次训练的完整配置如下；存在抽样时，指标口径以实际使用行数为准。",
        tables=[{"title": "训练配置", "headers": ["配置项", "值"],
                 "rows": [[k, _fmt(k, v)] for k, v in config_rows]}],
    ))

    # 二、评估指标（附 95% 置信区间）
    # 只给点估计会让人把差 0.01 的两个模型当成有高下，而那点差别常是抽样噪声；
    # 区间来自测试集有放回重采样，与上面的点估计同源。
    ci_map = art.get("metric_ci") if isinstance(art.get("metric_ci"), dict) else {}
    metric_rows: list[list[Any]] = []
    for key, value in metrics.items():
        item = ci_map.get(key)
        interval = (
            f"[{item['lower']:.4f}, {item['upper']:.4f}]"
            if isinstance(item, dict)
            and isinstance(item.get("lower"), (int, float))
            and isinstance(item.get("upper"), (int, float))
            else "-"
        )
        metric_rows.append([key, _fmt(key, value), interval])
    sections.append(ReportSection(
        heading="评估指标",
        content="指标在留出测试集上计算（聚类为全量拟合，无测试集）；"
                "95% CI 由测试集有放回重采样得到，区间宽说明该指标对样本敏感。",
        tables=[{"title": "指标总表", "headers": ["指标", "值", "95% CI"],
                 "rows": metric_rows}],
    ))

    # 三、混淆矩阵
    cm = art.get("confusion_matrix")
    if isinstance(cm, dict) and cm.get("labels"):
        labels = [str(x) for x in cm["labels"]]
        rows = []
        for i, row in enumerate(cm.get("matrix", [])):
            head = labels[i] if i < len(labels) else str(i)
            rows.append([head, *[str(x) for x in row]])
        sections.append(ReportSection(
            heading="混淆矩阵",
            content="行＝真实类别，列＝预测类别；对角线为预测正确的样本数。",
            tables=[{"title": "混淆矩阵", "headers": ["真实＼预测", *labels], "rows": rows}],
            images=_safe_chart(_confusion_chart, cm),
        ))

    # 四、分类别指标
    per_class = art.get("per_class")
    if isinstance(per_class, list) and per_class:
        rows = [
            [m.get("label"), _fmt("precision", m.get("precision")),
             _fmt("recall", m.get("recall")), _fmt("f1", m.get("f1")),
             _fmt("support", m.get("support"))]
            for m in per_class if isinstance(m, dict)
        ]
        sections.append(ReportSection(
            heading="分类别指标",
            content="逐类别的精确率/召回率/F1/样本数（按 support 降序）。",
            tables=[{"title": "每类指标", "headers": ["类别", "精确率", "召回率", "F1", "样本数"],
                     "rows": rows}],
        ))

    # 五、训练集类别分布
    dist = art.get("class_distribution")
    if isinstance(dist, dict):
        items = dist.get("items") or []
        rows = [[it.get("label"), _fmt("count", it.get("count")),
                 _fmt("ratio", it.get("ratio"))] for it in items]
        sections.append(ReportSection(
            heading="训练集类别分布",
            content=f"类别数 {dist.get('n_classes', '-')}；"
                    f"多数类占比 {_fmt('ratio', dist.get('majority_ratio'))}"
                    + ("（≥0.70，类别不均衡，accuracy 可能被多数类撑起）"
                       if isinstance(dist.get("majority_ratio"), (int, float))
                       and dist["majority_ratio"] >= 0.7 else ""),
            tables=[{"title": "类别分布", "headers": ["类别", "样本数", "占比"],
                     "rows": rows}] if rows else [],
        ))

    # 六、回归残差统计
    res = art.get("residual_stats")
    if isinstance(res, dict):
        text = str(res.get("note") or "")
        rows = [
            ["样本数", _fmt("count", res.get("count"))],
            ["残差均值", _fmt("mean", res.get("mean"))],
            ["残差标准差", _fmt("std", res.get("std"))],
            ["最大绝对误差", _fmt("max_abs", res.get("max_abs"))],
            ["P50", _fmt("p50", res.get("p50"))],
            ["P95", _fmt("p95", res.get("p95"))],
        ]
        sections.append(ReportSection(
            heading="残差统计（真实 − 预测）", content=text,
            tables=[{"title": "残差统计", "headers": ["统计项", "值"], "rows": rows}],
        ))

    # 七、特征重要性
    fi = art.get("feature_importance")
    if isinstance(fi, dict) and fi.get("importances"):
        rows = [[it.get("feature"), _fmt("importance", it.get("importance"))]
                for it in fi["importances"][:25]]
        sections.append(ReportSection(
            heading="特征重要性",
            content=f"口径：{fi.get('method', '-')}；列出前 {len(rows)} 项。",
            tables=[{"title": "特征重要性", "headers": ["特征", "重要性"], "rows": rows}],
            images=_safe_chart(_importance_chart, fi),
        ))

    # 八、学习曲线
    lc = art.get("learning_curve")
    if isinstance(lc, dict) and lc.get("points"):
        rows = [[_fmt("rows", p.get("rows")),
                 _fmt("train", p.get("train_score")),
                 _fmt("test", p.get("test_score"))] for p in lc["points"]]
        sections.append(ReportSection(
            heading="学习曲线",
            content=str(lc.get("note") or "每个规模点为独立拟合的真实结果。"),
            tables=[{"title": f"学习曲线（指标：{lc.get('metric', '-')}）",
                     "headers": ["训练样本数", "训练集分数", "测试集分数"], "rows": rows}],
            images=_safe_chart(_learning_curve_chart, lc),
        ))

    # 九、概率校准
    cal = art.get("calibration")
    if isinstance(cal, dict) and cal.get("points"):
        rows = [[_fmt("p", p.get("mean_predicted")),
                 _fmt("o", p.get("observed_frequency")),
                 _fmt("count", p.get("count"))] for p in cal["points"]]
        sections.append(ReportSection(
            heading="概率校准（可靠性曲线）",
            content=f"ECE={_fmt('ece', cal.get('ece'))}，MCE={_fmt('mce', cal.get('mce'))}；"
                    "ECE≤0.05 校准良好，>0.10 不宜直接按概率决策。",
            tables=[{"title": "校准分箱", "headers": ["平均预测概率", "实际正类比例", "样本数"],
                     "rows": rows}],
            images=_safe_chart(_calibration_chart, cal),
        ))

    # 十、决策阈值取舍曲线
    tc = art.get("threshold_curve")
    if isinstance(tc, dict) and tc.get("points"):
        rows = [[_fmt("threshold", p.get("threshold")),
                 _fmt("precision", p.get("precision")),
                 _fmt("recall", p.get("recall")),
                 _fmt("f1", p.get("f1")),
                 _fmt("rate", p.get("positive_rate"))] for p in tc["points"]]
        sections.append(ReportSection(
            heading="决策阈值取舍曲线",
            content=f"正类 {tc.get('positive_class', '-')} 口径（非 macro，勿与指标卡混比）；"
                    f"推荐阈值 {_fmt('t', tc.get('suggested_threshold'))}"
                    f"（F1={_fmt('f1', tc.get('suggested_f1'))}）。",
            tables=[{"title": "阈值扫描",
                     "headers": ["阈值", "精确率", "召回率", "F1", "判正类比例"], "rows": rows}],
            images=_safe_chart(_threshold_chart, tc),
        ))

    # 十一、复现信息
    model_summary = art.get("model_summary") or {}
    params = model_summary.get("params") if isinstance(model_summary, dict) else None
    repro_rows = [
        ["模型参数（实际生效）",
         json.dumps(params, ensure_ascii=False, sort_keys=True) if params else "无"],
        ["数据版本快照", json.dumps(art.get("dataset_snapshot") or {}, ensure_ascii=False, sort_keys=True)],
        ["依赖库版本", json.dumps(art.get("library_versions") or {}, ensure_ascii=False, sort_keys=True)],
        ["训练耗时(s)", _fmt("runtime", getattr(run, "runtime", None))],
    ]
    sections.append(ReportSection(
        heading="复现信息",
        content="凭数据版本快照、参数、seed 与依赖版本可完整复现本次运行。",
        tables=[{"title": "复现信息", "headers": ["项目", "内容"], "rows": repro_rows}],
    ))

    # 统一中文序号（条件小节缺失也不会跳号）
    for i, section in enumerate(sections):
        section.heading = f"{_cn(i)}、{section.heading}"

    # 自动结论（只写可追溯的事实）
    conclusions: list[str] = []
    primary = next((k for k in ("accuracy", "r2", "f1", "silhouette") if k in metrics), None)
    if primary:
        conclusions.append(
            f"Run #{run.id}（{model_name}，目标列 {target or '无'}）"
            f"{primary}={_fmt(primary, metrics[primary])}。"
        )
    if isinstance(dist, dict) and isinstance(dist.get("majority_ratio"), (int, float)) \
            and dist["majority_ratio"] >= 0.7:
        conclusions.append("训练集类别不均衡，解读指标时需同时参考分类别指标。")
    if isinstance(sampling, dict) and sampling.get("sampled"):
        conclusions.append("本次指标基于抽样数据，不代表全量数据上的表现。")

    return Report(
        title=f"机器学习训练报告 · Run #{run.id}",
        dataset={"name": dataset_name, "rows": art.get("train_rows"),
                 "columns": len(art.get("features") or [])},
        sections=sections,
        conclusions=conclusions,
        metadata={
            "run_id": run.id,
            "experiment_id": getattr(run, "experiment_id", None),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
