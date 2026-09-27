"""最终答案渲染：模板优先，LLM 只做可选润色。

两条路径
--------
1. **模板渲染（0 Token）**：把工具结果的 ``summary`` 与关键字段组织成中文。
   未配置 LLM 时走这条路，链路完整可用 —— 这是「离线也能用」的底线。
2. **LLM 润色（1 次调用）**：把**摘要**而不是原始数据喂给模型。
   工具结果可能包含上千行表格，全量喂进去既贵又容易超出上下文；
   ``ToolResult.for_llm`` 已经做过压缩，这里再取关键字段，实际入参通常几百字。

不许编造
--------
模板路径只输出工具真实返回的内容，缺什么就写「本次未获取」。
LLM 路径的 system prompt 明确要求「只使用给出的事实」——
这不保证模型一定不编，但把可核验的事实放在同一段上下文里，
是最有效的降低幻觉手段；剩下的靠用户核对。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.llm import LLMMessage, LLMProvider
from app.agent.models import ANSWER_SOURCE_RULE, AnswerSource, answer_source_llm
from app.agent.playbooks import PlaybookStep
from app.agent.intents import strip_result_reference

logger = logging.getLogger(__name__)

#: 模板答案里每个步骤最多展示的关键字段条数。
#:
#: 为什么从 12 提到 20：关系分析的结果结构是「left / right / 共享字段 / 候选键」，
#: 左侧光 dataset_id + name + rows + column_count + column_names 就占掉 5 条，
#: 12 条刚好在右侧的 column_count 与 column_names 之前用完 ——
#: 用户问「这两个数据集有哪些语义相同的字段」，答案里却只列得出左表的列，
#: 模型如实回答「右表列名本次未获取」。这不是模型编造，是我们先把事实掐掉了。
_MAX_FACTS_PER_STEP = 20
#: 喂给 LLM 的事实摘要字符上限。超过就截断 —— 宁可少说，不可超预算。
#:
#: 曾长期是 3000：关系分析这类「结果里就是一串字段名」的工具会被截到只剩结论，
#: 模型于是写出「共有 3 个同名字段（具体字段名本次未列出）」这种废话。
#: 9000 字符约 3~4k token，仍在单次润色可接受的预算内。
_FACTS_MAX_CHARS = 9000

#: 指标解读阈值（经验区间，只给方向性判断，不做绝对结论）。
#:
#: 为什么答案里必须自带解读：真实事故里模型跑出 ``silhouette = 0.1316``，
#: 答案原样把它列在「训练成功」下面。用户拿不到任何判断依据 ——
#: 这个数到底是好是坏、接下来该干什么，全靠他自己猜。
#: 数字本身不是结论，读得懂才算答案。
_SILHOUETTE_WEAK = 0.25
_SILHOUETTE_FAIR = 0.50
_R2_WEAK = 0.50
_F1_WEAK = 0.50
#: 超过这个行数就提醒先抽样：全量跑一遍常要几分钟，方案不对时沉没成本极高。
_LARGE_ROW_COUNT = 1_000_000
#: 建议最多几条。堆十条建议等于没有建议。
_MAX_ADVICE = 5
#: 对话路径最多带多少条历史、每条最多多少字符（与引擎侧的上限同口径）
_CHAT_HISTORY_MESSAGES = 8
_CHAT_HISTORY_ITEM_CHARS = 800


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        # 统一 4 位有效小数并去掉尾零：0.8364929 → 0.8365，避免长尾噪音
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return str(value)
    return str(value)


#: 一条列表最多枚举多少项、枚举出的文本最多多少字符。
#:
#: 为什么改成「字符预算」而不是固定条数：固定取前 5 条的老写法在关系分析里
#: 直接丢掉关键信息 —— 两个数据集有 20 个同名字段，喂给模型的只有前 5 个，
#: 剩下的变成一句「等 20 项」。模型于是只能写「字段名本次未列出」。
#: 字段名这类值很短，20 条也占不了多少字；反过来长文本按字数收口，不会撑爆预算。
_MAX_LIST_ITEMS = 20
_MAX_LIST_CHARS = 800
#: 单条长文本最多保留多少字符。超过就截断，**不能整条丢弃** ——
#: 报告正文、建议文案这些值动辄几百字，丢掉它们模型就只能回答「本次未获取」。
_MAX_FACT_VALUE_CHARS = 160


def _join_items(texts: list[str], total: int, *, sep: str = "、") -> str:
    out = ""
    kept = 0
    for text in texts:
        if out and len(out) + len(sep) + len(text) > _MAX_LIST_CHARS:
            break
        out = f"{out}{sep}{text}" if out else text
        kept += 1
    return out + (f"（共 {total} 条）" if total > kept else "")


def _fmt_list(value: list[Any], limit: int = _MAX_LIST_ITEMS) -> str:
    return _join_items([_fmt_scalar(v) for v in value[:limit]], len(value))


#: 结果里「量大且无信息量」的键：它们会把事实条数与上下文预算占满，
#: 真正有用的数值反而被挤掉（见 :func:`_correlation_pairs` 的注释）。
_SKIP_KEYS = frozenset({"pair_methods", "methods_used"})


def _correlation_pairs(matrix: Any, limit: int = 8) -> list[str]:
    """相关性矩阵原样展开是 N² 个数字 —— 7 列就是 49 条，直接占满事实条数，
    模型读完也挑不出重点；实测答案会写成「未给出任何相关系数数值」。

    这里按 |r| 降序取最强的若干对（去重对称项、跳过自相关），
    让模型拿到的是「哪几对最值得关注」而不是一堆裸数字。
    """
    if not isinstance(matrix, dict):
        return []
    pairs: list[tuple[float, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, row in matrix.items():
        if not isinstance(row, dict):
            continue
        for right, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if str(left) == str(right):
                continue
            key = tuple(sorted((str(left), str(right))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((abs(float(value)), f"{left} × {right} 相关系数 {float(value):.4f}"))
    pairs.sort(key=lambda item: -item[0])
    return [f"强相关 {text}" if score >= 0.5 else text for score, text in pairs[:limit]]


#: 比较运算符的中文写法。写进事实摘要的条件里，避免模型把「大于」读成「等于」。
_OP_SYMBOLS = {
    "gt": ">", "gte": "≥", "lt": "<", "lte": "≤",
    "eq": "=", "neq": "≠", "contains": "包含", "in": "属于",
    "is_null": "为空", "not_null": "非空", "startswith": "以…开头", "endswith": "以…结尾",
}


def _op_symbol(op: Any) -> str:
    return _OP_SYMBOLS.get(str(op).strip().lower(), str(op))


def _collect_facts(data: Any, *, limit: int = _MAX_FACTS_PER_STEP) -> list[str]:
    """从工具结果里抽出可展示的「键: 值」行。

    三类结构分别处理：
    - 标量 / 短字符串 → 直接展示
    - 字典 → 递归一层，跳过超长值与嵌套容器
    - 列表 → 若是「问题/记录」类字典，取其可读字段；否则只报条数
    """
    rows: list[str] = []

    def walk(value: Any, prefix: str, depth: int) -> None:
        if len(rows) >= limit or depth > 2:
            return
        if isinstance(value, dict):
            for key, inner in list(value.items())[:20]:
                # 下划线开头是工具内部字段（如 workflow 节点的 ``_df`` 表格转储，
                # 单个节点 4~8 KB）：它既没信息量又会占满展示位与上下文。
                if str(key).startswith("_") or str(key) in _SKIP_KEYS:
                    continue
                walk(inner, f"{prefix}{key}." if prefix else f"{key}.", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            items = list(value)
            if not items:
                return
            label = prefix.rstrip(".")
            if isinstance(items[0], dict):
                # 记录清单：取「标识 + 关键数值」组合，比只报裸列名/条数有用得多。
                # 例如 outlier 的 {"column": "DepDelay", "outlier_count": 1347945}
                # 应显示成「DepDelay（outlier_count=1347945）」而非光秃秃的「DepDelay」——
                # 后者让模型只能回「异常值数量本次未获取」。
                texts: list[str] = []
                for item in items[:_MAX_LIST_ITEMS]:
                    ident = next(
                        (str(item.get(f)) for f in ("column", "name", "label", "feature", "title")
                         if item.get(f) not in (None, "")),
                        "",
                    )
                    # ★ 筛选条件必须带上**比较符**。
                    #   只取 column + value 会显示成「credit_amount（value=5000）」，
                    #   模型据此回「筛选条件：credit_amount = 5000」——
                    #   把「大于」写成了「等于」，筛选语义整个变了。
                    op = item.get("op")
                    if ident and op is not None and item.get("value") is not None:
                        texts.append(f"{ident} {_op_symbol(op)} {_fmt_scalar(item['value'])}")
                        continue
                    nums: list[str] = []
                    for f in ("outlier_count", "count", "missing", "missing_count", "mean", "std", "min", "max", "value", "match_coverage", "null_count"):
                        v = item.get(f)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            nums.append(f"{f}={_fmt_scalar(v)}")
                    if ident and nums:
                        texts.append(f"{ident}（{'，'.join(nums)}）")
                    elif ident:
                        texts.append(ident)
                    else:
                        # 兜底 1：原字符串字段提取
                        raw = next((str(item.get(f)) for f in ("message", "check", "summary")
                                    if item.get(f) not in (None, "")), "")
                        if raw:
                            texts.append(raw)
                        else:
                            # 兜底 2：把整条记录压成「键=值」串。
                            # 聚合 / 筛选 / 预览回执的就是这种裸记录
                            # （{"purpose": "car", "credit_amount_mean": 1234.5}），
                            # 没有 column/outlier_count 这类约定键 —— 少了这一步，
                            # 用户拿到的答案只能是「结果共 8 行」而看不到任何数值。
                            parts: list[str] = []
                            for key, inner in list(item.items())[:6]:
                                if str(key).startswith("_"):
                                    continue
                                if isinstance(inner, bool):
                                    parts.append(f"{key}={'是' if inner else '否'}")
                                elif isinstance(inner, (int, float)):
                                    parts.append(f"{key}={_fmt_scalar(inner)}")
                                elif isinstance(inner, str) and inner.strip() and len(inner) <= 30:
                                    parts.append(f"{key}={inner.strip()}")
                            if parts:
                                texts.append("，".join(parts))
                if texts:
                    label_text = label or "明细"
                    rows.append(
                        f"{label_text}：{_join_items(texts, len(items), sep='；')}"
                    )
                else:
                    rows.append(f"{label or '明细'}：共 {len(items)} 条")
            else:
                rows.append(f"{label or '取值'}：{_fmt_list(items)}")
            return
        if isinstance(value, bool) or isinstance(value, (int, float)):
            rows.append(f"{prefix.rstrip('.') or '值'}：{_fmt_scalar(value)}")
            return
        if isinstance(value, str):
            text = " ".join(value.split())
            if not text:
                return
            label = prefix.rstrip(".") or "值"
            if len(text) <= 80:
                rows.append(f"{label}：{text}")
            else:
                rows.append(f"{label}：{text[:_MAX_FACT_VALUE_CHARS]}…")

    # 相关性矩阵先压成「最强的几对」再走通用遍历 —— 否则 49 个裸数字
    # 会把事实条数吃光，模型一个系数都看不到。
    walked = data
    if isinstance(data, dict) and isinstance(data.get("matrix"), dict):
        rows.extend(_correlation_pairs(data["matrix"]))
        walked = {k: v for k, v in data.items()
                  if k not in ("matrix",) and str(k) not in _SKIP_KEYS}

    walk(walked, "", 0)
    return rows[:limit]


def _read_metrics(metrics: Any) -> list[str]:
    """把数值指标翻成人能用的判断。

    只解读**阈值明确**的指标。RMSE、MAE 这类与量纲强相关的指标不解读 ——
    0.8 的 RMSE 在「预测房价（万元）」和「预测耗时（秒）」里含义完全不同，
    硬给一个「偏高/偏低」就是编造。
    """
    if not isinstance(metrics, dict):
        return []
    out: list[str] = []
    silhouette = metrics.get("silhouette")
    if isinstance(silhouette, (int, float)):
        if silhouette < _SILHOUETTE_WEAK:
            out.append(
                f"轮廓系数 {silhouette:.4f} 偏低（低于 {_SILHOUETTE_WEAK}）："
                "当前特征与簇数下几乎没有可解释的簇结构，这个聚类结果不足以支撑结论"
            )
        elif silhouette < _SILHOUETTE_FAIR:
            out.append(
                f"轮廓系数 {silhouette:.4f}：簇结构一般，"
                "可尝试调整簇数，或先做特征选择与标准化再评估"
            )
        else:
            out.append(f"轮廓系数 {silhouette:.4f}：簇结构较清晰")
    r2 = metrics.get("r2")
    if isinstance(r2, (int, float)) and r2 < _R2_WEAK:
        out.append(
            f"R² {r2:.4f} 偏低：模型解释的方差不足一半，"
            "建议先确认特征是否覆盖关键信息，再考虑换用非线性模型"
        )
    f1 = metrics.get("f1")
    if isinstance(f1, (int, float)) and f1 < _F1_WEAK:
        out.append(
            f"F1 {f1:.4f} 偏低：先看正负样本是否严重不平衡，"
            "再考虑调整决策阈值或补充特征"
        )
    return out


def _failed(steps: list[tuple[PlaybookStep, Any]]) -> list[tuple[PlaybookStep, Any]]:
    """本轮**没做成**的步骤。失败不能悄悄过去，更不能包装成计划。"""
    return [
        (step, result)
        for step, result in steps
        if result is not None and not getattr(result, "success", True)
    ]


def _trained(steps: list[tuple[PlaybookStep, Any]]) -> bool:
    """本轮是否真的训出了模型（detect_task 跑过不等于训过）。"""
    return any(
        step.tool == "ml.train" and getattr(result, "success", False)
        for step, result in steps
    )


def _advice(steps: list[tuple[PlaybookStep, Any]]) -> list[str]:
    """从工具结果里总结出「这说明什么 / 下一步做什么」。

    三类信号值得单独说：
    - **指标**：见 :func:`_read_metrics`
    - **没法定**：``needs_target`` —— 本次没有建模，必须告诉用户，
      否则他以为模型已经跑完了
    - **规模**：百万行以上提醒抽样
    """
    lines: list[str] = []
    # 「目标列未确定」这句话只在**真的没训**的时候才成立。
    # 典型链路是 detect_task(needs_target=True) → 用户补 target → ml.train 成功，
    # 此时再把那句提示塞进答案，就会出现「本次没有训练任何模型」与下面一串
    # 指标并存的自我矛盾 —— 用户第一眼看到的是「没训」，于是以为指标是假的。
    trained = _trained(steps)
    for _step, result in steps:
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        lines.extend(_read_metrics(data.get("metrics")))
        if data.get("needs_target") and not trained:
            candidates = data.get("target_candidates") or {}
            regression = "、".join(str(c) for c in (candidates.get("regression") or [])[:5]) or "无"
            classification = (
                "、".join(str(c) for c in (candidates.get("classification") or [])[:5]) or "无"
            )
            lines.append(
                "目标列未确定，本次**没有**训练任何模型："
                f"请先指定要预测的列（回归候选 {regression}；分类候选 {classification}）再建模"
            )
        rows = data.get("row_count") or data.get("rows")
        if isinstance(rows, int) and rows >= _LARGE_ROW_COUNT:
            lines.append(
                f"数据规模 {rows:,} 行：全量建模耗时较长，建议先抽样验证方案，确认有效后再全量执行"
            )

    # 去重但保序：同一次运行里 detect 与 train 可能都带候选列
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique[:_MAX_ADVICE]


def _chart_notice(steps: list[tuple[PlaybookStep, Any]]) -> str:
    """图表自检话术。

    ★ 历史缺陷：用户点名「画个直方图」，答案把每个分箱的计数描述了一遍
      （「最低箱是 [8.1, 15.2)，计数 284」），但**图从头到尾没有出现过**。
      文字描述不是图 —— 用户要的是能一眼看懂的分布，不是一串数字。

    出图了就必须说清楚「图已经在对话框里」并给出可调整的维度；
    没出图就必须明说「当前环境无法显示」并指向导出/预览，
    **绝不能假装已经生成**。
    """
    rendered = 0
    attempted = False
    for _step, result in steps:
        meta = getattr(result, "metadata", None)
        if not isinstance(meta, dict):
            continue
        chart = meta.get("chart")
        if not isinstance(chart, dict):
            continue
        attempted = True
        if chart.get("rendered"):
            rendered += 1
    if not attempted:
        return ""
    if rendered:
        return "图表已渲染在对话框中，如需调整分箱数或颜色，请告诉我。"
    return "当前环境无法直接显示图片，请点击 [导出/预览] 按钮查看。"


def render_template(user_request: str, steps: list[tuple[PlaybookStep, Any]]) -> str:
    """模板渲染。``steps`` 是 (步骤定义, ToolResult) 的列表。"""
    lines: list[str] = []
    for step, result in steps:
        if result is None:
            continue
        lines.append(f"**{step.title}**")
        summary = (getattr(result, "summary", "") or "").strip()
        if summary:
            lines.append(summary)
        if getattr(result, "success", False):
            facts = _collect_facts(getattr(result, "data", None))
            for fact in facts:
                lines.append(f"- {fact}")
        else:
            errors = getattr(result, "errors", None) or []
            lines.append(f"- 未成功：{errors[0] if errors else '未知原因'}")
        for warning in (getattr(result, "warnings", None) or [])[:3]:
            lines.append(f"- 提示：{warning}")
        lines.append("")
    advice = _advice(steps)
    if advice:
        lines.append("**解读与下一步**")
        for item in advice:
            lines.append(f"- {item}")
        lines.append("")
    notice = _chart_notice(steps)
    if notice:
        lines.append(notice)
        lines.append("")
    if not lines:
        return "本次没有产生可展示的结果。"
    return "\n".join(lines).strip()


def _facts_for_llm(steps: list[tuple[PlaybookStep, Any]]) -> str:
    """给 LLM 的事实摘要：只取 success 的摘要与关键字段，并卡字符上限。"""
    chunks: list[str] = []
    for step, result in steps:
        if result is None:
            continue
        head = f"[{step.tool}] {step.title}"
        if getattr(result, "success", False):
            summary = (getattr(result, "summary", "") or "").strip()
            # 每条 8 条的老上限会把双侧结果（left/right）的右半边整段切掉，
            # 模型只能回答「未获取」—— 这里与模板渲染同口径。
            facts = _collect_facts(getattr(result, "data", None))
            body = summary or "（无摘要）"
            if facts:
                body += "\n" + "\n".join(facts)
        else:
            errors = getattr(result, "errors", None) or []
            body = f"失败：{errors[0] if errors else '未知原因'}"
        chunks.append(f"{head}\n{body}")
    advice = _advice(steps)
    if advice:
        # 解读也要喂给模型：否则润色后的答案会把「训练成功」讲得更顺，
        # 却仍然不提 silhouette 0.13 意味着什么。
        chunks.append("[解读与下一步]\n" + "\n".join(advice))
    if _trained(steps):
        # ★ 事实优先级：detect_task 的结论是**训练前**的中间态。
        # 用户补了目标列、ml.train 也真的跑完之后，模型仍会读到
        # 「未识别到 target / 任务为 clustering」，于是写出
        # 「未获取目标列确认信息，请显式指定 target」—— 而用户三秒前刚指定过。
        chunks.append(
            "[事实优先级] 本次已成功训练模型（见 ml.train）。"
            "ml.detect_task 中的「未识别到目标列 / 任务为聚类」是训练前的中间结论，"
            "已被用户补充的目标列与本次实际训练的任务覆盖；"
            "不要据此说「没有训练」「未指定目标列」或「任务类型不一致」。"
        )
    if _failed(steps):
        # ★ 与上面同类的另一个方向：失败被**包装成计划**。
        #   实测「先清洗再训练模型」的工作流创建失败（ml.train 缺 target_column），
        #   模型却答成「清洗与训练工作流可按以下顺序搭建：1… 2… 3…」，
        #   只字不提这一步根本没建成 —— 用户以为流程已经搭好了。
        chunks.append(
            "[事实优先级] 本次有步骤没有成功："
            + "、".join(f"{s.tool}（{s.title}）" for s, _ in _failed(steps))
            + "。回答必须明确告诉用户这一步没做成以及原因，"
            "不要把它包装成「可以按以下顺序搭建」「建议下一步」这类计划式描述。"
        )
    text = "\n\n".join(chunks)
    if len(text) > _FACTS_MAX_CHARS:
        text = text[:_FACTS_MAX_CHARS] + "\n…（已截断）"
    return text


def render(
    user_request: str,
    steps: list[tuple[PlaybookStep, Any]],
    *,
    provider: LLMProvider | None = None,
    usage: Any | None = None,
    timeout: float = 30.0,
    #: 当前数据集的真实列名。给了就在 prompt 里钉死可用范围，
    #: 否则模型会用训练语料里的列名给建议（实测给出过 credit_amount）。
    columns: list[str] | None = None,
) -> tuple[str, AnswerSource]:
    """渲染最终答案。LLM 不可用时静默退回模板渲染。

    返回 ``(答案文本, 答案来源)``。来源必须如实标注：
    模板渲染不能标成大模型回答。
    """
    template = render_template(user_request, steps)
    if provider is None:
        return template, ANSWER_SOURCE_RULE

    # 用户请求先过一遍回指剥离：「筛选后还剩多少行」里的「筛选」不是本轮诉求
    # （见 intents.strip_result_reference）。带着它喂给模型，模型会发现手里
    # 没有筛选这一步的事实，于是开头先写一句「无法算出」，再报出真实行数。
    request_for_llm = strip_result_reference(user_request)

    system = (
        "你是小洛实验室的数据分析助手。请基于下面给出的【事实摘要】用中文回答用户的问题。\n"
        "硬性要求：\n"
        "1. 只使用事实摘要中出现的数据，禁止编造任何数字、列名或结论；\n"
        "2. 回答控制在 400 字以内，用简洁的短句与要点，不要复述工具名；\n"
        # ★ 下面这条是实测出来的：模型会原样写出「本次未获取该信息。原因：事实摘要中
        # 只包含…」—— 把我们的内部数据结构与缺口直接甩给用户，还顺带让他自己去重跑。
        # 用户要的是结论，不是「我们内部缺了什么」的清单。
        "3. 禁止出现「事实摘要」「本次未获取」「未获取该信息」这类内部口径，"
        "也不要逐条罗列没算出来的东西；摘要里没有的内容直接不写；\n"
        # ★ 同一类问题的另一个变体：模型会用「本次只读取了…没有执行任何…操作」
        #   描述自己刚才干了什么，然后才开始给数字。实测「筛选后还剩多少行」得到
        #   「筛选后的行数目前无法给出：本次只读取了数据集基本信息，没有执行任何
        #   筛选操作」—— 前半句说给不出、后半句又把 473 行写出来了，自相矛盾。
        #   用户不关心系统内部跑了哪一步，只关心答案。
        "4. 禁止描述系统自己做了什么或没做什么（例如「本次只…」「没有执行…操作」"
        "「我目前没有…的结果」）：直接给结论；\n"
        # ★ 同一类伤的最深的一次：用户问「筛选后还剩多少行」，模型第一句写
        #   「筛选条件还没给，无法算出剩余行数」，第二句才写「当前数据集共 164 行」。
        #   两句自相矛盾，而且用户第一眼看到的是「我白问了」。
        #   正确姿势：先把手里的事实说完，缺的那部分留到最后一句，用引导句式带过。
        "5. 第一句必须是已拿到的事实或数字，不要用「无法 / 不能 / 还没 / 尚未 / 没有」开头；"
        "确实缺少的信息放在最后一句，用「如果你想…可以说…」的句式带过；\n"
        "6. 版本号写成 v8 这种简短形式，不要写「version 8」；\n"
        "7. 缺的内容不要让用户「重新执行 / 重新运行」，改成告诉他下一句该怎么说"
        "（例如：再说一句「画 age 与 credit_amount 的散点图」）；\n"
        "8. 如果某个步骤失败了，必须明确写出哪一步没做成以及原因，"
        "不要把它包装成「可以按以下顺序…」「建议下一步…」这类计划式描述；\n"
        # ★ 实测：用户只问「还剩多少行」，答案把 v1…v9 九个版本的行数全列了一遍。
        #   数字是真的，但没人要看版本年表。
        "9. 用户没问版本历史时，不要罗列各个历史版本的行数，只说当前版本。\n"
        # ★ 实测：答案是「tenure_months 的直方图」，末尾却建议「画 credit_amount
        #   的直方图」—— credit_amount 根本不在这个数据集里，是模型从训练语料
        #   里带出来的（第十轮）。用户照着说一句，得到的只有一次失败。
        #   把真实列名摆进 prompt，模型就没有编造的余地。
        + (
            f"10. 数据集里真实存在的列只有：{'、'.join(columns)}；"
            "任何举例、建议里出现的列名都必须来自这份清单，清单外的列名一律不许写。"
            if columns else ""
        )
    )
    user = f"用户请求：{request_for_llm}\n\n【事实摘要】\n{_facts_for_llm(steps)}"
    try:
        response = provider.chat(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            temperature=0.2,
            max_tokens=900,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        # 润色失败就用模板答案：用户拿到的是真实工具结果，只是文笔差一点。
        logger.warning("LLM 答案润色失败，退回模板渲染：%s", exc)
        return template, ANSWER_SOURCE_RULE

    content = (response.content or "").strip()
    if not content:
        return template, ANSWER_SOURCE_RULE

    if usage is not None:
        usage.llm_calls += 1
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens

    # ★ 图表自检话术必须**确定性地**接在答案后面。放进 prompt 让模型自己写，
    #   它会在 400 字预算里把这句挤掉，或者写成「如上图所示」之类无法验证的话。
    notice = _chart_notice(steps)
    if notice and notice not in content:
        content = f"{content}\n\n{notice}"
    return content, answer_source_llm(response.model or getattr(provider, "model", ""))


def render_chat(
    user_request: str,
    *,
    provider: LLMProvider | None = None,
    usage: Any | None = None,
    history: list[dict[str, str]] | None = None,
    context: str = "",
    timeout: float = 30.0,
) -> tuple[str, AnswerSource]:
    """纯对话渲染。无 LLM 时给一条诚实的说明，不假装回答。

    :param history: 会话里已经发生过的对话（``[{"role": ..., "content": ...}]``）
    :param context: 本次可用的事实（数据集概况、上一次分析的结论）

    这两项不是锦上添花。没有它们，用户追问「你推荐哪个目标列」时，
    模型手里只有这八个字，只能回答「请补充候选列有哪些」——
    而那些候选列上一轮就摆在屏幕上。
    """
    if provider is None:
        return (
            "我没有理解你想做的具体分析。可以直接说要做的事，例如：\n"
            "- 看看这个数据集的质量\n"
            "- 分析 age 列的分布\n"
            "- 计算各字段的相关性\n"
            "- 训练一个预测模型\n"
            "- 生成一份分析报告\n\n"
            "当前未配置大模型 API Key，我只能处理上述明确的分析指令。",
            ANSWER_SOURCE_RULE,
        )

    system = (
        "你是小洛实验室的助手。用户的问题不涉及具体数据分析操作，"
        "请结合【会话上下文】与【对话历史】用简洁的中文回答。\n"
        "硬性要求：\n"
        "1. 用户追问上一轮结论时，必须基于历史里已有的真实结论作答；\n"
        "2. 上下文与历史里没有的信息，直接说不知道，不要编造列名或数字，"
        "也不要写「本次未获取该信息」这类内部口径；\n"
        "3. 若问题确实需要再跑一次数据才能回答，用一句话说明需要用户补充什么，"
        "不要罗列编号选项清单，也不要写「我目前没有…的结果」「本次只…」这类"
        "描述系统自身状态的句子；\n"
        # ★ 最危险的一类幻觉：用户要「小提琴图」，这句话掉进了对话路径，
        #   模型回的是「小提琴图我直接按 Churn 分组画（能同时看出…）」——
        #   它根本没调用任何工具，却把图画说得跟真的一样（Telco 压测实测）。
        #   对话路径**没有工具结果**，因此不许出现任何「已做」的措辞。
        "3.1 你这次没有调用任何工具，所以绝不能出现「我画了 / 已生成 / 我直接按…画 / "
        "我帮你算了 / 已跑完」这类宣称已执行的说法；"
        "做不到就直说暂不支持，并给出一条用户可以直接照说的替代指令；\n"
        # ★ 实测：用户说「帮我把这个数据集删掉」，对话路径答「我这边无法直接执行」。
        #   「我能不能」是系统内部的事，用户要的是「那我该去哪儿点」。
        "4. 做不到的事直接说清该去哪里做或该怎么说，不要写「我无法 / 我不能 / "
        "我这边没有权限」这类描述自身能力的句子；\n"
        "5. 回答控制在 300 字以内；\n"
        # ★ 实测：用户问「哪些字段可能存在共线性」，对话路径回了一堆分析以后
        #   以「你希望我对哪些字段做这个计算？」收尾；「客户价值矩阵」以
        #   「请告诉我用中位数还是四分位切分」收尾。两句都把本可以由系统定下来的
        #   默认值推回给用户（Telco 压测实测）。设计/分析类问题要给**结论**，
        #   真正需要人拍板时才提问，且必须同时给出你推荐的默认值。
        "6. 不要以「请告诉我 / 你希望我 / 需要你补充」这类反问结尾："
        "需要选参数时，直接按一个合理的默认值给出结论，并补一句"
        "「我按 X 做的，要改成 Y 说一声」；确实必须人来定的，"
        "也要先给结论再问，并附上你的推荐值。"
    )
    messages: list[LLMMessage] = [LLMMessage(role="system", content=system)]
    if context.strip():
        messages.append(LLMMessage(role="system", content=f"【会话上下文】\n{context.strip()}"))
    for item in (history or [])[-_CHAT_HISTORY_MESSAGES:]:
        role = str(item.get("role") or "")
        content = " ".join(str(item.get("content") or "").split())
        if role in ("user", "assistant") and content:
            messages.append(LLMMessage(role=role, content=content[:_CHAT_HISTORY_ITEM_CHARS]))
    messages.append(LLMMessage(role="user", content=user_request))
    try:
        response = provider.chat(messages, temperature=0.4, max_tokens=600, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("对话渲染失败：%s", exc)
        return (
            "抱歉，当前无法调用大模型来回答这个问题。"
            "如果你要做数据分析，可以直接下达明确指令（如「检查数据质量」）；"
            "如需自由对话，请先在设置页配置大模型 API Key。",
            ANSWER_SOURCE_RULE,
        )

    content = (response.content or "").strip()
    if not content:
        return "模型返回了空回答，请换个说法再试一次。", ANSWER_SOURCE_RULE

    if usage is not None:
        usage.llm_calls += 1
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens

    return content, answer_source_llm(response.model or getattr(provider, "model", ""))
