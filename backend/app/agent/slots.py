"""槽位（工具参数）抽取：规则优先，LLM 只在规则读不出时兜底。

为什么不让 LLM 抽全部参数
--------------------------
每次都调 LLM 意味着：每次用户说「看看质量」都要付一次 Token，
而 ``dataset_id`` 这种参数根本不需要模型参与（会话里已经绑定了）。

抽取顺序（严格按此，命中即停）：

1. **自动槽位**：``dataset_id`` 取会话绑定的数据集，零成本零风险
2. **规则抽取**：从文本里直接读（列名、图表类型、run id、数据集 id）
3. **LLM 抽取**：仅在必填槽位仍缺失 **且** 配置了 LLM 时，发起**一次**调用
4. **放弃**：返回空，由引擎决定改用 fallback_tool 还是向用户澄清

LLM 抽取的降级
--------------
- 模型不可用 / 超时 / 返回非法 JSON → 返回空 dict，**绝不抛异常中断链路**
- 优先用 ``response_format`` 的 JSON 模式；厂商不支持时自动退回宽松解析
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.llm import LLMMessage, LLMProvider
from app.agent.playbooks import PlaybookStep

logger = logging.getLogger(__name__)

#: 图表类型别名：用户说法 → eda.visualize 的 chart 取值
_CHART_ALIASES: dict[str, str] = {
    "直方图": "histogram", "histogram": "histogram", "hist": "histogram", "分布图": "histogram",
    "柱状图": "bar", "条形图": "bar", "bar": "bar",
    "折线图": "line", "趋势图": "line", "line": "line",
    "散点图": "scatter", "scatter": "scatter",
    "箱线图": "box", "箱型图": "box", "box": "box",
    "饼图": "pie", "pie": "pie",
    "热力图": "heatmap", "heatmap": "heatmap",
}

_RUN_ID_RE = re.compile(r"(?:run[\s_-]?id|运行|运行id)\s*[:：]?\s*(\d{1,9})", re.IGNORECASE)
_DATASET_ID_RE = re.compile(r"(?:数据集|dataset)\s*(?:id)?\s*[:：]?\s*(\d{1,9})", re.IGNORECASE)
_AGG_WORDS: dict[str, str] = {
    "求和": "sum", "总和": "sum", "sum": "sum",
    "平均": "mean", "均值": "mean", "mean": "mean", "avg": "mean",
    "计数": "count", "数量": "count", "count": "count",
    "最大": "max", "max": "max",
    "最小": "min", "min": "min",
    "中位数": "median", "median": "median",
}

#: 用户说法 → 模型**族名**（不带任务后缀）。
#:
#: 为什么只到族名：规则抽取时还不知道是分类还是回归（那要等 ml.detect_task），
#: 而注册模型名是带后缀的。族名交给 ml.train 按任务解析 —— 分层清楚，
#: 也避免规则层去猜任务类型。
_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"随机森林|random[\s_\-]?forest", re.I), "random_forest"),
    (re.compile(r"逻辑回归|logistic[\s_\-]?regression", re.I), "logistic_regression"),
    (re.compile(r"线性回归|linear[\s_\-]?regression|最小二乘", re.I), "linear_regression"),
    (re.compile(r"决策树|decision[\s_\-]?tree", re.I), "decision_tree"),
    (re.compile(r"(?<![a-z])knn(?![a-z])|k\s*近邻|k[\s_\-]?nearest|最近邻", re.I), "knn"),
    (re.compile(r"(?<![a-z])k[\s_\-]?means(?![a-z])|k[\s_\-]?均值", re.I), "kmeans"),
    (re.compile(r"(?<![a-z])dbscan(?![a-z])|密度聚类", re.I), "dbscan"),
    (re.compile(r"(?<![a-z])pca(?![a-z])|主成分", re.I), "pca"),
)

#: 超参说法 → (参数名, 正则)。命中即写进 ``params``。
#:
#: 「分成 5 簇」「200 棵树」这种诉求在旧链路里全被丢弃：
#: model 永远取默认 auto，params 从来不是槽位，于是用户调的参数一个都没生效。
_PARAM_PATTERNS: tuple[tuple[str, re.Pattern[str], Any], ...] = (
    ("n_clusters", re.compile(r"n[\s_\-]?clusters\s*[:=]\s*(\d+)", re.I), int),
    ("n_clusters", re.compile(r"(\d+)\s*个?簇|簇(?:数|个数)?\s*(?:为|是|[:=])?\s*(\d+)|分成\s*(\d+)\s*(?:个)?(?:簇|类|组)", re.I), int),
    ("n_estimators", re.compile(r"n[\s_\-]?estimators\s*[:=]\s*(\d+)", re.I), int),
    ("n_estimators", re.compile(r"(\d+)\s*棵?树", re.I), int),
    ("max_depth", re.compile(r"max[\s_\-]?depth\s*[:=]\s*(\d+)|(?:树)?深(?:度)?\s*(?:为|[:=])?\s*(\d+)", re.I), int),
    ("n_neighbors", re.compile(r"n[\s_\-]?neighbors\s*[:=]\s*(\d+)", re.I), int),
    ("random_state", re.compile(r"(?:random[\s_\-]?state|随机种子|seed)\s*[:=为]?\s*(\d+)", re.I), int),
    ("C", re.compile(r"(?<![a-z])C\s*[:=]\s*([0-9.]+)", 0), float),
)


def _match_model(text: str) -> str | None:
    """从文本里认出模型族名。多个命中时取**先声明**的（表里按特异性排序）。"""
    for pattern, family in _MODEL_PATTERNS:
        if pattern.search(text or ""):
            return family
    return None


def _match_model_params(text: str) -> dict[str, Any]:
    """从文本里读出具体超参。读不出就返回空 dict —— 不猜。"""
    out: dict[str, Any] = {}
    for name, pattern, cast in _PARAM_PATTERNS:
        if name in out:
            continue
        match = pattern.search(text or "")
        if not match:
            continue
        raw = next((g for g in match.groups() if g), None)
        if raw is None:
            continue
        try:
            out[name] = cast(raw)
        except (TypeError, ValueError):
            continue
    return out


def _match_test_size(text: str) -> float | None:
    """从文本里读出「测试集占 30%」这类测试集比例。读不出返回 None。

    两种写法都要认：百分比（测试集占 30% / 30% 测试集）与小数（test_size=0.3）。
    比例必须落在开区间 (0,1)，否则视为没说。
    """
    patterns = (
        # test_size=0.3 / test_size: 0.25（小数优先，最明确）
        re.compile(r"test[\s_]?size\s*[:=]\s*(0?\.\d+)", re.I),
        # 测试集占 30% / 测试集 20% / 验证集 25%
        re.compile(r"(?:测试集|验证集)\s*(?:占|为|是|约)?\s*[:=]?\s*(\d{1,2})\s*%", re.I),
        # 30% 测试集 / 30% 验证集
        re.compile(r"(\d{1,2})\s*%\s*(?:的)?\s*(?:测试集|验证集)", re.I),
    )
    for pattern in patterns:
        match = pattern.search(text or "")
        if not match:
            continue
        raw = match.group(1)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        # 小数写法直接是比例；百分比写法要除以 100（「30%」→ 0.3）
        if "%" in match.group(0) or "测试集" in match.group(0) or "验证集" in match.group(0):
            if "%" in match.group(0):
                value = value / 100.0
        if 0.0 < value < 1.0:
            return round(value, 3)
    return None


def _match_column(text: str, columns: list[str] | None) -> str | None:
    """在已知列名里做匹配。

    优先用真实列名做子串匹配 —— 用户说「分析 age 的分布」，
    ``age`` 就在列名表里，这是最可靠的信号。
    有多个命中时取**最长的**（避免 ``id`` 命中 ``user_id`` 这种前缀误判）。
    """
    if not columns:
        return None
    lowered = text.lower()
    best: str | None = None
    for column in columns:
        if column and column.lower() in lowered:
            if best is None or len(column) > len(best):
                best = column
    return best


def _match_columns(text: str, columns: list[str] | None) -> list[str] | None:
    """抽出用户**点名的所有列**（复数槽位，如相关性矩阵的 ``columns``）。

    规则抽取此前只认单数 ``column``，而 eda.correlation / eda.describe /
    eda.outlier 用的都是 ``columns`` —— 于是「看看 customer_id 和 age 的相关性」
    一个列都传不进去，工具只能自作主张算全表，最后答案写「矩阵里没有你点名的列」。

    只认**在文本里真实出现过**的列名，不猜、不补全；一个都没点到就返回 None，
    由工具按默认（全表）处理。
    """
    if not columns:
        return None
    lowered = (text or "").lower()
    hits = [c for c in columns if c and c.lower() in lowered]
    if not hits:
        return None
    # 长列名优先，避免 "id" 这类短列压在前面（也避免 user_id 与 id 重复计入）
    hits.sort(key=lambda c: -len(c))
    ordered: list[str] = []
    for column in hits:
        if not any(column.lower() in kept.lower() for kept in ordered):
            ordered.append(column)
    # 保持数据集里的原始顺序，结果更稳定（不随用户说话顺序变）
    index = {c: i for i, c in enumerate(columns)}
    ordered.sort(key=lambda c: index.get(c, 0))
    return ordered or None


def _match_chart(text: str) -> str | None:
    lowered = text.lower()
    for alias, value in _CHART_ALIASES.items():
        if alias.lower() in lowered:
            return value
    return None


def _match_aggregations(text: str, columns: list[str] | None) -> dict[str, Any] | None:
    """抽取聚合配置。

    只处理「分组列在文本里、聚合函数在文本里」这种能确定的情况；
    其余交给 LLM 或澄清。半懂不懂地猜出一个聚合配置，比不猜更糟。
    """
    lowered = text.lower()
    func = next((v for k, v in _AGG_WORDS.items() if k in lowered), None)
    if not func:
        return None
    column = _match_column(text, columns)
    if not column:
        return None
    return {"group_by": _match_group_by(text, columns) or [], "aggregations": [{column: func}]}


#: 「创建一个叫 客户流失预警 的 Workflow」「新建名为 X 的流程」里的名字。
#: 名字是**自由文本**，只能靠前后标记切，不能靠词表。
_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:叫|叫做|名为|命名[为]?)\s*([^\s，,。;；的]{1,30})"),
    re.compile(r"(?:创建|新建|搭建|建个|建一个|创建个)\s*(?:一个)?\s*([^\s，,。;；]{1,30}?)\s*(?:工作?流|流程|pipeline)"),
)

#: 用户在推荐方案里选了哪一个：「我选 2」「2」「第二个」「方案3」。
_PLAN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:我选|选|选择|用|要|就来?)\s*(?:第)?\s*([1-9１-９一二三])\s*(?:号|个|方案)?"),
    re.compile(r"^\s*([1-9１-９])\s*[。.、]?\s*$"),
    re.compile(r"第\s*([1-9１-９一二三])\s*个"),
)
_CN_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
              "１": "1", "２": "2", "３": "3", "４": "4", "５": "5"}
#: 不该被当成方案序号的噪音（用户其实在说别的数字）。数字本身已在别处抽取。
_NAME_STOPWORDS: tuple[str, ...] = ("一个", "这个", "那个", "什么", "怎么", "如何")


def _match_name(text: str) -> str | None:
    """抽 Workflow/报告的名字。

    ★ 历史缺陷：``name`` 从来没有被抽取过。用户说「创建一个叫 客户流失预警 的
      Workflow」，``workflow.build_and_run`` 的 required_slots 直接判缺，
      于是退化到 ``fallback_tool``（workflow.list）只列了一圈已有流程 ——
      用户点名的名字被丢得干干净净。
    """
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        value = (match.group(1) or "").strip(" 「」\"'《》")
        if not value or value in _NAME_STOPWORDS:
            continue
        return value
    return None


def _match_plan(text: str) -> str | None:
    """抽用户选中的方案序号（归一成阿拉伯数字字符串）。"""
    stripped = (text or "").strip()
    if not stripped:
        return None
    for pattern in _PLAN_PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue
        raw = (match.group(1) or "").strip()
        return _CN_DIGITS.get(raw, raw)
    # 「方案1」「1号」这类裸写法
    bare = re.match(r"^方案?\s*([1-9１-９])\s*[号个]?$", stripped)
    if bare:
        return _CN_DIGITS.get(bare.group(1), bare.group(1))
    return None


#: 相关性方法：「用 spearman」「斯皮尔曼」。
_METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("spearman", "spearman"),
    ("斯皮尔曼", "spearman"),
    ("pearson", "pearson"),
    ("皮尔逊", "pearson"),
)


def _match_method(text: str) -> str | None:
    """抽相关性方法。

    ★ 「再帮我跑一次，这次用 spearman」里方法名是**唯一的新信息**；
      method 不是槽位时它会被静默丢掉，用户拿到的还是上一次的 pearson 结果，
      还以为系统按他说的做了。
    """
    low = (text or "").lower()
    for key, value in _METHOD_PATTERNS:
        if key.lower() in low:
            return value
    return None


def _match_bins(text: str) -> int | None:
    """抽直方图的分箱数：「30 个分箱」「分箱数改成 50」「bins=20」。"""
    match = re.search(r"(\d{1,3})\s*(?:个)?\s*分箱", text or "")
    if match:
        value = int(match.group(1))
        if 2 <= value <= 200:
            return value
    match = re.search(r"分箱\s*(?:数)?\s*(?:改成|换成|设为|改为)?\s*(\d{1,3})", text or "")
    if match:
        value = int(match.group(1))
        if 2 <= value <= 200:
            return value
    match = re.search(r"bins\s*[:=]?\s*(\d{1,3})", text or "", re.IGNORECASE)
    if match:
        value = int(match.group(1))
        if 2 <= value <= 200:
            return value
    return None


#: 「A 和 B 的散点图 / 关系图」里的两个列名。顺序保留：先说的是 x，后说的是 y。
_XY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"([\w\u4e00-\u9fff\.]+)\s*(?:和|与|跟|vs|versus)\s*([\w\u4e00-\u9fff\.]+)"),
    re.compile(r"x\s*[:=]\s*([\w\u4e00-\u9fff\.]+)[,，\s]+y\s*[:=]\s*([\w\u4e00-\u9fff\.]+)"),
    re.compile(r"([\w\u4e00-\u9fff\.]+)\s*[,、]\s*([\w\u4e00-\u9fff\.]+)\s*(?:的)?\s*(?:散点图|关系图|折线图|对比图)"),
)


def _match_xy(text: str, columns: list[str] | None) -> dict[str, Any] | None:
    """抽双列图表（scatter / line / grouped_bar）的 x 与 y。

    ★ 历史缺陷：``x`` / ``y`` 从来不是槽位。「画 tenure_months 和 monthly_charges
      的散点图」只抽出了 ``column=monthly_charges``，工具随即报
      「该图表类型必须指定 x 轴字段」—— 用户点名了两列，系统一列都没用上。
    """
    for pattern in _XY_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        first, second = (match.group(1) or "").strip(), (match.group(2) or "").strip()
        if not first or not second or first == second:
            continue
        # 列名优先按真实列名对齐（用户可能说中文别名或大小写不一致）。
        # ★ 对不上就必须**丢弃**，不能原样塞进去：
        #   「刚才那个字段和 TotalCharges 做散点图」里 x 被抽成「刚才那个字段」，
        #   工具随即报「指定的字段不存在：刚才那个字段」—— 一次图都没画成。
        #   丢掉之后引擎的回指兜底（_recent_columns）才有机会补上真正的列名。
        if columns:
            first = _align_column(first, columns) or ""
            second = _align_column(second, columns) or ""
        if not first or not second or first == second:
            continue
        return {"x": first, "y": second}
    return None


#: 「把 tenure 作为目标列」「目标列换成 Churn」「以 Churn 为目标」「预测 Churn」
#: 这些词不是列名，是「目标列」这个词本身的一部分（「训练一个预测模型」）
_TARGET_STOPWORDS = frozenset({"模型", "目标", "结果", "数据", "变量", "列", "字段", "一下", "什么", "值"})

_TARGET_PATTERNS = (
    re.compile(r"(?:把|将|用)\s*([\w\u4e00-\u9fff\.]+)\s*(?:作为|当作|设为|做成|做|当)?\s*(?:目标列|目标变量|目标|target)"),
    re.compile(r"(?:目标列|目标变量|target)\s*(?:换成|改为|改成|设为|指定为|取|是)\s*([\w\u4e00-\u9fff\.]+)"),
    re.compile(r"以\s*([\w\u4e00-\u9fff\.]+)\s*为(?:目标列|目标变量|目标)"),
    re.compile(r"(?:预测|预估)\s*([\w\u4e00-\u9fff\.]+)"),
)


def _match_target(text: str, columns: list[str] | None) -> str | None:
    """抽用户**点名**的目标列。

    ★ 历史缺陷：``ml.train`` 的 tuned_slots 里没有 ``target``，于是
      「帮我把 tenure 作为目标列，跑一个回归模型」里用户点名的列**从来没被抽出来**，
      目标列一律来自 ``ml.detect_task`` 的推断（这里推断成了 Churn）⇒
      task=classification + model=linear_regression 直接冲突失败，一次训练都没发生
      （Telco 压测实测）。用户点名的目标列必须能盖过推断值。

    只认**对得上真实列名**的候选，避免把「预测下个月的股票价格」当成列名。
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        raw = (match.group(1) or "").strip()
        if not raw or raw in _TARGET_STOPWORDS:
            continue
        if columns:
            aligned = _align_column(raw, columns)
            if aligned:
                return aligned
            continue
        return raw
    return None


def _align_column(text: str, columns: list[str]) -> str | None:
    """把用户写法的列名对齐到真实列名（忽略大小写与下划线）。"""
    wanted = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(text or "").lower())
    if not wanted:
        return None
    for col in columns:
        if re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(col).lower()) == wanted:
            return col
    return None


def _match_group_by(text: str, columns: list[str] | None) -> list[str] | None:
    """抽取分组列：匹配「按 X 分组 / 按 X 统计 / 每个 X」这类说法。"""
    if not columns:
        return None
    patterns = (
        re.compile(r"按\s*([\w\u4e00-\u9fff]+)\s*(?:分组|统计|聚合|汇总)"),
        re.compile(r"(?:每个|每种)\s*([\w\u4e00-\u9fff]+)"),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        token = match.group(1)
        for column in columns:
            if column.lower() == token.lower():
                return [column]
    return None


def rule_extract(
    user_request: str,
    step: PlaybookStep,
    *,
    schema_columns: list[str] | None = None,
) -> dict[str, Any]:
    """纯规则抽取。不联网、不调模型、无副作用。"""
    text = user_request or ""
    found: dict[str, Any] = {}
    # 必填槽位 + 可微调槽位一起走规则：model / params 同样能从文本里读出来，
    # 读不出来就交给默认值，绝不会走到反问。
    slots: list[str] = []
    for slot in (*step.required_slots, *getattr(step, "tuned_slots", ())):
        if slot not in slots:
            slots.append(slot)

    for slot in slots:
        if slot == "model":
            value = _match_model(text)
            if value:
                found[slot] = value
        elif slot == "params":
            value = _match_model_params(text)
            if value:
                found[slot] = value
        elif slot == "column":
            value = _match_column(text, schema_columns)
            if value:
                found[slot] = value
        elif slot == "columns":
            value = _match_columns(text, schema_columns)
            if value:
                found[slot] = value
        elif slot == "chart":
            value = _match_chart(text)
            if value:
                found[slot] = value
        elif slot == "run_id":
            match = _RUN_ID_RE.search(text)
            if match:
                found[slot] = int(match.group(1))
        elif slot == "right_dataset_id":
            # 「合并数据集9和数据集5」里 right 是第二个（5），search 取第一个会错填成 9。
            # findall 取最后一个覆盖「合并/关联 A 和 B」的常见语序。
            matches = _DATASET_ID_RE.findall(text)
            if matches:
                found[slot] = int(matches[-1])
        elif slot == "test_size":
            value = _match_test_size(text)
            if value:
                found[slot] = value
        elif slot in ("group_by", "aggregations"):
            if "group_by" in found or "aggregations" in found:
                continue
            value = _match_aggregations(text, schema_columns)
            if value:
                found.update(value)
        elif slot == "name":
            value = _match_name(text)
            if value:
                found[slot] = value
        elif slot == "plan":
            value = _match_plan(text)
            if value:
                found[slot] = value
        elif slot == "bins":
            value = _match_bins(text)
            if value:
                found[slot] = value
        elif slot == "method":
            value = _match_method(text)
            if value:
                found[slot] = value
        elif slot in ("x", "y"):
            if "x" in found or "y" in found:
                continue
            value = _match_xy(text, schema_columns)
            if value:
                found.update(value)
        elif slot == "target":
            value = _match_target(text, schema_columns)
            if value:
                found[slot] = value
        elif slot == "goal":
            # 诉求本身就是原始这句话：推荐方案时要按它排序。
            # 抽别的没有意义，只会把用户的原话改写得面目全非。
            if text.strip():
                found[slot] = text.strip()
    return found


def _slot_help(schema: dict[str, Any], slots: list[str]) -> str:
    """把工具 schema 里每个待抽参数的**取值约束**翻成模型能读的一行说明。

    为什么必须带上：只给参数名（``["model"]``）等于让模型盲填。
    它不知道这里能填 random_forest_classifier，只能老实写 ``auto`` 或干脆省略 ——
    看起来就是「Agent 只会调工具，不会按诉求调参数」。
    有了 enum 与描述，模型才有得选。
    """
    props = (schema or {}).get("properties") or {}
    lines: list[str] = []
    for name in slots:
        spec = props.get(name)
        if not isinstance(spec, dict):
            lines.append(f"- {name}")
            continue
        parts: list[str] = [str(spec.get("type") or "any")]
        enum = spec.get("enum")
        if isinstance(enum, list) and enum:
            shown = [str(v) for v in enum[:24]]
            parts.append(
                "只能取：" + "、".join(shown) + (f"（共 {len(enum)} 个）" if len(enum) > 24 else "")
            )
        # array of object：展开元素字段。否则模型只看到「conditions 是数组」，
        # 不知道内部要填 column/op/value，就会凭感觉写 field/operator（真实事故：
        # filter 收到 {"field": "departure_delay", "operator": ">"} 而 schema 要 column/op）。
        items = spec.get("items")
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            item_segs: list[str] = []
            for fname, fspec in items["properties"].items():
                seg = [fname]
                fenum = fspec.get("enum")
                if isinstance(fenum, list) and fenum:
                    seg.append("取" + "、".join(str(v) for v in fenum[:12]))
                fdesc = " ".join(str(fspec.get("description") or "").split())
                if fdesc:
                    seg.append(fdesc[:60])
                item_segs.append("：".join(seg[:2]))
            if item_segs:
                parts.append("元素字段：" + "；".join(item_segs))
        desc = " ".join(str(spec.get("description") or "").split())
        if desc:
            parts.append(desc[:220])
        lines.append(f"- {name}（{'；'.join(parts)}）")
    return "\n".join(lines)


def _extraction_budget(schema: dict[str, Any], slots: list[str]) -> int:
    """这次抽取给模型多少输出 token。

    默认 300 对「抽一个列名 / 一个模型名」绰绰有余，但对
    ``workflow.build_and_run`` 的 ``nodes`` / ``edges`` 这种**结构化数组**来说
    一定不够 —— 300 token 写不完三个节点，模型被截断后返回半个 JSON，
    解析失败 ⇒ 抽取结果为空 ⇒ 链路退化。这里按槽位类型放大预算。
    """
    props = (schema or {}).get("properties") or {}
    for name in slots:
        spec = props.get(name)
        if isinstance(spec, dict) and str(spec.get("type") or "") in ("array", "object"):
            return 1500
    return 300


def llm_extract(
    user_request: str,
    step: PlaybookStep,
    *,
    schema_columns: list[str] | None = None,
    tool_schema: dict[str, Any] | None = None,
    tool_description: str = "",
    hint: str = "",
    provider: LLMProvider | None = None,
    usage: Any | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """用一次 LLM 调用补齐剩余必填槽位。失败一律返回空 dict。

    :param tool_schema: 工具的 ``input_schema``，用于把 enum / 描述喂给模型
    :param tool_description: 工具的 ``description``。编排类工具（如
        ``workflow.build_and_run``）的**合法取值写在描述里而不是 schema 里**
        （可用节点类型清单），只给 schema 模型只能瞎编节点类型。
    :param hint: 引擎已知、但模型看不到的上下文（如会话绑定的数据集 id）
    :param usage: 可选，``TokenUsage``；调用成功时累加 llm_calls 与 token
    """
    missing = [*step.required_slots, *getattr(step, "tuned_slots", ())]
    seen: set[str] = set()
    missing = [s for s in missing if not (s in seen or seen.add(s))]
    if not missing or provider is None:
        return {}

    columns_hint = ""
    if schema_columns:
        shown = schema_columns[:60]
        columns_hint = f"\n数据集的列名（column 只能从这里面选）：{json.dumps(shown, ensure_ascii=False)}"
    desc_hint = ""
    if tool_description:
        desc_hint = f"\n工具说明（含合法取值，务必遵守）：{tool_description}"
    context_hint = f"\n已知上下文：{hint}" if hint else ""
    schema_hint = _slot_help(tool_schema or {}, missing)
    budget = _extraction_budget(tool_schema or {}, missing)
    system = (
        "你是参数抽取器。根据用户的请求，为工具抽取参数。\n"
        "只输出一个 JSON 对象，不要任何解释文字、不要用代码块包裹。\n"
        "只能使用给定的键名；无法确定的键**直接省略**，不要猜测、不要填 null。\n"
        "给出可选值清单的参数，取值必须落在清单内，不要自创造值。"
    )
    user = (
        f"工具：{step.tool}\n"
        f"参数说明：\n{schema_hint}\n"
        f"需要抽取的参数（只用这些键）：{json.dumps(missing, ensure_ascii=False)}"
        f"{columns_hint}{desc_hint}{context_hint}\n"
        f"用户请求：{user_request}"
    )
    try:
        try:
            response = provider.chat(
                [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                temperature=0,
                max_tokens=budget,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - JSON 模式不被支持时退回普通对话
            logger.info("LLM 参数抽取的 JSON 模式不可用，退回普通对话：%s", exc)
            response = provider.chat(
                [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                temperature=0,
                max_tokens=budget,
                timeout=timeout,
            )
    except Exception as exc:  # noqa: BLE001
        # 抽取失败不是致命错误：上层会走 fallback_tool 或澄清，链路继续。
        logger.warning("LLM 参数抽取失败，改用规则结果：%s", exc)
        return {}

    if usage is not None:
        usage.llm_calls += 1
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens

    parsed = _parse_json_loose(response.content)
    if not isinstance(parsed, dict):
        return {}
    # 只接受声明过的键，避免模型顺手编造工具不认识的参数
    allowed = set(missing)
    return {k: v for k, v in parsed.items() if k in allowed and v not in (None, "", [])}


def _parse_json_loose(text: str) -> Any | None:
    """宽松解析 LLM 的 JSON 输出。

    模型常见的三种「看起来像 JSON」的输出：```json 围栏、前后带说明文字、
    JSON 前后有多余字符。逐个处理；全都失败返回 None（调用方按「没抽到」处理）。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except ValueError:
            return None
    return None


def extract_slots(
    user_request: str,
    step: PlaybookStep,
    *,
    schema_columns: list[str] | None = None,
    provider: LLMProvider | None = None,
    usage: Any | None = None,
    auto: dict[str, Any] | None = None,
    tool_schema: dict[str, Any] | None = None,
    tool_description: str = "",
    hint: str = "",
) -> dict[str, Any]:
    """完整抽取：自动 → 规则 → LLM。

    :param auto: 自动槽位（``dataset_id`` 等），优先级最高
    :param tool_schema: 工具 ``input_schema``，让 LLM 看到 enum / 描述
    :param tool_description: 工具 ``description``（合法取值常写在这里）
    :param hint: 引擎侧的已知上下文（会话绑定的数据集 id 等）

    返回值里**包含可微调槽位**：调用方据此覆盖 :attr:`PlaybookStep.defaults`。
    """
    slots: dict[str, Any] = dict(auto or {})
    slots.update(rule_extract(user_request, step, schema_columns=schema_columns))

    still_missing = [s for s in step.required_slots if slots.get(s) in (None, "", [])]
    tuned = tuple(getattr(step, "tuned_slots", ()))
    # 规则已经读出来的微调槽位不再问模型：省一次 Token，也避免模型改掉明确诉求
    tuned_missing = [s for s in tuned if slots.get(s) in (None, "", [])]
    if (still_missing or tuned_missing) and provider is not None:
        partial = llm_extract(
            user_request,
            PlaybookStep(
                tool=step.tool,
                title=step.title,
                required_slots=tuple(still_missing),
                tuned_slots=tuple(tuned_missing),
            ),
            schema_columns=schema_columns,
            tool_schema=tool_schema,
            tool_description=tool_description,
            hint=hint,
            provider=provider,
            usage=usage,
        )
        slots.update(partial)
    return slots
