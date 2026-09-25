"""平台自带能力的**闲聊应答** —— 只在「远程大模型不可用」时启用。

为什么需要它
------------
`runtime._direct_chat()` 原先在 `llm is None` 时只回一句
「当前尚未配置可用的大模型。」；而数据链路 `_compose_answer()` **是有本地兜底的**
（`已完成 N 个真实数据工具步骤。` + 逐条工具摘要）。

两条链路不对称的后果：设置页关掉「启用远程 API 大模型」之后，**数据请求仍然完全可用，
闲聊却彻底熄火** —— 这与设置页「关闭后 Agent 改用平台自带模型」的承诺不符。
更糟的是那句文案本身在关掉开关时是**事实错误**：`PUT /settings/llm/remote` 只切开关、
凭据原样保留，用户看到「尚未配置」会跑去重填 Key（见 `no_llm_notice()`）。

设计边界（刻意收窄，不要扩成聊天机器人）
----------------------------------------
1. **只覆盖答案确定的意图**：问候 / 自述 / 能力询问 / 使用入口。
   这些问题的正确答案是确定的，用模板表达不算「假装会说话」。
2. **拿不准就返回 `None`**，由上层用 `no_llm_notice()` 说明现状。
   **不猜、不瞎聊** —— 一旦开始对任意句子胡编，就不如老实说「需要远程模型」。
3. **能力清单从 `TOOL_REGISTRY` 动态生成**，不硬编码任何工具名。
   平台增删工具后这段话自动跟随，与「本地 Router 标签空间动态派生」是同一个原则。
4. **只在 `llm is None` 时被调用**：远程大模型开启时，`_direct_chat` 仍旧走 LLM，
   行为**完全不变**（这一点由 `tests/test_local_chat.py` 钉住）。

已知取舍
--------
这仍是**规则表**，不产生新信息：同一个问题每次都得到同一段话。
它是「关掉远程后平台不要像死的一样」的最低成本方案，不是本地生成模型。
论文里应写成**受限的本地应答能力**，而不是「平台自带大模型」。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DOMAIN_LABELS",
    "local_reply",
    "local_result_summary",
    "no_llm_notice",
    "tool_overview",
]

# 「谢谢」「再见」混在 `AgentRuntime.GREETINGS` 里（那张表是给 chat/agent 分流用的），
# 这里按应答语义拆开。
_THANKS = {"谢谢", "感谢", "多谢", "辛苦了", "thanks", "thank you", "thx"}
_FAREWELL = {"再见", "拜拜", "bye", "goodbye", "下次见", "回头聊"}
# 身份询问同时也是问候语表中的词，必须先判身份（应答内容差别很大）。
_IDENTITY_PATTERNS = (
    "你是谁", "你叫什么", "你是什么", "介绍一下你", "自我介绍", "你是哪个", "谁开发的", "你是人还是",
)
_CAPABILITY_PATTERNS = (
    "你能做什么", "你会做什么", "能做什么", "会做什么", "有什么功能", "有哪些功能",
    "能帮我做什么", "能干什么", "支持什么", "支持哪些", "都会啥", "会啥",
)
_USAGE_PATTERNS = (
    "怎么用", "如何使用", "使用方法", "使用说明", "怎么用你", "帮助", "help", "教我", "怎么开始",
)

# 能力域的中文名。键取自 `contract.Intent` 的取值（动态派生的能力域）。
DOMAIN_LABELS: dict[str, str] = {
    "dataset": "数据接入与查看",
    "data_transform": "数据清洗与转换",
    "eda": "探索性分析（EDA）",
    "ml": "机器学习建模",
    "workflow": "流程编排",
    "report": "报告生成",
    "other": "其他",
}
_DOMAIN_ORDER = ("dataset", "data_transform", "eda", "ml", "workflow", "report", "other")

# 每个能力域最多举几个工具名做示例（全列出来会把消息撑爆）
_EXAMPLES_PER_DOMAIN = 3

_MODE_NOTE = "（当前运行在**平台自带能力**模式：远程大模型已停用，以下回复来自内置规则，不是语言模型生成的）"

# ★ 「远程开着但这一次调用失败」与「远程被主动停用」是两件事：
# 复用 `_MODE_NOTE` 会在欠费 / 超时场景下告诉用户「远程已停用」—— 那是假的，
# 用户会跑去设置页找开关，而真实原因是余额 / 网络。文案必须分开。
_DEGRADED_NOTE = "（本次远程大模型调用失败，已退回**平台自带能力**：以下回复来自内置规则，不是语言模型生成的）"


def _note(*, degraded: bool = False) -> str:
    return _DEGRADED_NOTE if degraded else _MODE_NOTE


# ---------------------------------------------------------------------------
# 动态能力清单
# ---------------------------------------------------------------------------


def tool_overview() -> tuple[int, list[tuple[str, list[str]]]]:
    """返回 `(工具总数, [(能力域中文名, 该域工具名列表), ...])`，全部实时读自注册表。

    读不到注册表时返回 `(0, [])` —— 调用方负责优雅退化，绝不在这里抛异常。
    """
    try:
        from app.local_router import contract as C

        names = list(C.tool_label_space())
        buckets: dict[str, list[str]] = {}
        for name in names:
            intent = C.intent_of_tool(name)
            buckets.setdefault(intent.value if intent is not None else "other", []).append(name)
    except Exception as exc:  # noqa: BLE001 — 能力清单拿不到不该让闲聊也失败
        logger.warning("读取工具注册表失败，能力清单降级：%s", exc)
        return 0, []

    ordered = [(DOMAIN_LABELS.get(key, key), sorted(buckets[key]))
               for key in _DOMAIN_ORDER if key in buckets]
    return len(names), ordered


def _capability_brief() -> str:
    """「我能做这些」的正文；注册表不可用时给一句诚实说明而不是空表。"""
    total, groups = tool_overview()
    if total == 0:
        return ("工具注册表当前不可用，我暂时列不出能力清单。"
                "可以稍后重试，或直接描述你的需求我试着执行。")
    lines = []
    for label, names in groups:
        shown = "、".join(names[:_EXAMPLES_PER_DOMAIN])
        more = f" 等 {len(names)} 个" if len(names) > _EXAMPLES_PER_DOMAIN else ""
        lines.append(f"• {label}：{shown}{more}")
    return (f"我能调度平台**真实注册**的 {total} 个数据工具，按能力域分成 {len(groups)} 类：\n"
            + "\n".join(lines)
            + "\n\n这份清单是**实时读自工具注册表**的，平台增删工具会自动跟随，不是写死的。")


# ---------------------------------------------------------------------------
# 各意图的应答
# ---------------------------------------------------------------------------


def _thanks_reply(*, degraded: bool = False) -> str:
    return "不客气。有数据上的需求随时说 —— 比如「看看这批数据的分布」「检查一下数据质量」。\n" + _note(degraded=degraded)


def _farewell_reply(*, degraded: bool = False) -> str:
    return "再见！需要的时候再来找我。\n" + _note(degraded=degraded)


def _greeting_reply(*, degraded: bool = False) -> str:
    lead = (
        "你好！我是小洛实验室的 AI 助手。\n\n这次远程大模型没能调用成功，下面这段是平台内置规则给的；"
        if degraded
        else "你好！我是小洛实验室的 AI 助手。\n\n虽然现在跑在平台自带能力模式下（对话回复是内置规则），"
    )
    return (lead
            + "**数据分析这条路是通的**："
              "你把需求说清楚，我就用平台真实的工具去执行，再把结果汇总给你。\n\n"
              "可以试试：「看看这批数据的分布」或「检查一下数据质量」。\n" + _note(degraded=degraded))


def _identity_reply(*, degraded: bool = False) -> str:
    lead = (
        "我是小洛实验室的 AI 助手。这一次远程大模型调用失败，所以现在按**平台自带能力**回答 —— "
        if degraded
        else "我是小洛实验室的 AI 助手，跑在**平台自带能力**模式下 —— "
    )
    tail = (
        "对话回复来自内置规则，只有数据分析由平台真实工具完成。\n\n"
        if degraded
        else "意思是我现在不调用远程大模型，对话回复来自内置规则，只有数据分析由平台真实工具完成。\n\n"
    )
    return lead + tail + _capability_brief()


def _capability_reply(*, degraded: bool = False) -> str:
    return ("我现在能做的事，按能力域列给你：\n\n"
            + _capability_brief()
            + "\n\n把需求说清楚我就直接执行（例如「把缺失值清洗掉」「训练一个分类模型」）。"
              "复杂或需要多轮澄清的需求，建议到「设置 → AI 服务」重新开启远程大模型。\n"
            + _note(degraded=degraded))


def _usage_reply(*, degraded: bool = False) -> str:
    return ("平台用法，三条路径：\n\n"
            "1. **直接说需求**（最常用）：「看看这批数据的分布」「把缺失值清洗掉」"
            "「训练一个分类模型，目标列是 label」—— 我用真实工具执行后汇总结果给你。\n"
            "2. **先绑数据集再提问**：在「数据」里上传或选中数据集，会话里绑定它；"
            "这样「看看分布」这类省略主语的话也能直接执行，否则我会先反问你用哪个数据集。\n"
            "3. **固定下来重复跑**：处理链稳定后用「流程」画布搭好，之后一键运行；"
            "想练手可以去「学习中心」，那里有带判分的用例。\n\n"
            + _note(degraded=degraded))


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def local_reply(utterance: str, *, degraded: bool = False) -> str | None:
    """命中确定性意图则返回本地应答；**拿不准返回 `None`**（由上层说明现状）。

    判定顺序即优先级：身份询问在问候语之前（「你是谁」同时在两张表里，
    而自述比一句「你好」有用得多）。

    `degraded=True` 表示这次是「远程调用失败后的降级」而不是「远程本来就关着」，
    两种场景的结论性注释不同（见 `_note`）—— 混用会让用户在欠费时看到
    「远程已停用」这种与事实不符的说明。
    """
    text = (utterance or "").strip()
    if not text:
        return None
    low = text.lower()

    if low in _THANKS:
        return _thanks_reply(degraded=degraded)
    if low in _FAREWELL:
        return _farewell_reply(degraded=degraded)
    if any(pattern in low for pattern in _IDENTITY_PATTERNS):
        return _identity_reply(degraded=degraded)
    if any(pattern in low for pattern in _CAPABILITY_PATTERNS):
        return _capability_reply(degraded=degraded)
    if any(pattern in low for pattern in _USAGE_PATTERNS):
        return _usage_reply(degraded=degraded)

    # 复用 `_route` 那张问候语表，避免两处词表各自漂移（延迟 import 以免循环依赖）
    from app.agent.runtime.runtime import AgentRuntime

    if low in {word.lower() for word in AgentRuntime.GREETINGS}:
        return _greeting_reply(degraded=degraded)

    return None


def _fmt_num(value: Any) -> str:
    """安全地把数值格式化成短字符串（NaN / None / 超长尾巴都兜住）。"""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "—"
    if abs(f) >= 1e6 or (abs(f) > 0 and abs(f) < 1e-4):
        return f"{f:.3g}"
    return f"{f:,.4g}"


def _render_distribution_overview(data: Any) -> str:
    """从 ``eda.distribution_overview`` 的真实 ``data`` 生成一段本地叙述。

    这是**统一**的数据驱动渲染：读结构化字段（numeric_columns / categorical_columns /
    signals），不针对某个特定数据集硬编码，也不逐列堆数字 —— 给结论式摘要。
    """
    if not isinstance(data, dict):
        return ""
    numeric = data.get("numeric_columns") or []
    categorical = data.get("categorical_columns") or []
    lines: list[str] = []
    lines.append(f"这份数据共 {len(numeric)} 个数值列、{len(categorical)} 个类别列。")

    if numeric:
        lines.append("数值列分布概况：")
        for item in numeric[:8]:
            name = item.get("column", "?")
            mean = _fmt_num(item.get("mean"))
            median = _fmt_num(item.get("median"))
            skew = item.get("skew_direction", "")
            missing = item.get("missing", 0)
            lines.append(
                f"  • {name}：均值 {mean}，中位数 {median}，{skew}"
                + (f"，缺失 {missing} 条" if missing else "")
            )
    if categorical:
        lines.append("类别列概况：")
        for item in categorical[:8]:
            name = item.get("column", "?")
            uniq = item.get("unique_count", "?")
            top = (item.get("top_categories") or [])[:3]
            top_str = "、".join(
                f"{t.get('value')}({t.get('count')})" for t in top
            ) if top else "无有效值"
            lines.append(f"  • {name}：{uniq} 个唯一值，Top：{top_str}")
    return "\n".join(lines)


def _render_describe(data: Any) -> str:
    """``eda.describe`` → 行数、数值列量级与离散度、缺失最多的列。

    只给**结论式摘要**（哪几列值得看、哪里有缺失），不逐列堆全部统计量。
    """
    if not isinstance(data, dict):
        return ""
    columns = data.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return ""
    lines = [f"共 {data.get('row_count', '?')} 行、{len(columns)} 列。"]
    missing: list[tuple[str, int]] = []
    numeric: list[tuple[str, Any, Any]] = []
    for item in columns:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column", "?"))
        miss = int(item.get("missing_count") or 0)
        if miss:
            missing.append((name, miss))
        if "mean" in item and item.get("mean") is not None:
            numeric.append((name, item.get("mean"), item.get("std")))
    if numeric:
        lines.append("数值列量级（前 8 列，均值 ± 标准差）：")
        for name, mean, std in numeric[:8]:
            lines.append(f"  • {name}：{_fmt_num(mean)} ± {_fmt_num(std)}")
    if missing:
        missing.sort(key=lambda kv: -kv[1])
        shown = "、".join(f"{n}（{m} 条）" for n, m in missing[:6])
        lines.append(f"存在缺失的列：{shown}。")
    else:
        lines.append("未发现缺失值。")
    return "\n".join(lines)


def _render_outlier(data: Any) -> str:
    """``eda.outlier`` → 哪些列有异常、异常占比多少。

    历史缺陷：`_extract_signals` 判的是顶层 `outliers` / `outlier_count` 键，
    而本分析器只返回 `{"method", "columns"}`，异常数嵌在 `columns[i]` 里 ——
    信号永远产不出来。这里直接读真实结构，不依赖信号。
    """
    if not isinstance(data, dict):
        return ""
    columns = data.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return ""
    rows: list[tuple[str, int, float]] = []
    skipped = 0
    for item in columns:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            skipped += 1
            continue
        count = int(item.get("outlier_count") or 0)
        ratio = float(item.get("outlier_ratio") or 0.0)
        if count:
            rows.append((str(item.get("column", "?")), count, ratio))
    if not rows:
        tail = f"（{skipped} 列因空列/常数列已跳过）" if skipped else ""
        return f"按 {data.get('method', '默认')} 方法检测：各数值列均未发现异常值{tail}。"
    rows.sort(key=lambda r: -r[1])
    lines = [f"按 {data.get('method', '默认')} 方法检测，{len(rows)} 个数值列存在异常值："]
    for name, count, ratio in rows[:8]:
        lines.append(f"  • {name}：{count} 条（占 {ratio * 100:.2f}%）")
    if skipped:
        lines.append(f"另有 {skipped} 列因空列/常数列跳过。")
    return "\n".join(lines)


def _render_correlation(data: Any) -> str:
    """``eda.correlation`` → 强相关对（|r| ≥ 0.8，与 report.generate 同口径）。

    矩阵本身不适合直接给人看，「哪两列强相关」才是结论。
    """
    if not isinstance(data, dict):
        return ""
    matrix = data.get("matrix")
    if not isinstance(matrix, dict) or not matrix:
        return ""
    pairs: list[tuple[str, str, float]] = []
    cols = list(matrix)
    for i, a in enumerate(cols):
        row = matrix.get(a)
        if not isinstance(row, dict):
            continue
        for b in cols[i + 1:]:
            r = row.get(b)
            if r is None:
                continue
            try:
                value = float(r)
            except (TypeError, ValueError):
                continue
            if abs(value) >= 0.8:
                pairs.append((str(a), str(b), value))
    if not pairs:
        return f"按 {data.get('method', '默认')} 方法计算：未发现 |r| ≥ 0.8 的强相关列对。"
    pairs.sort(key=lambda p: -abs(p[2]))
    lines = [f"按 {data.get('method', '默认')} 方法，|r| ≥ 0.8 的强相关列对："]
    for a, b, r in pairs[:8]:
        kind = "正相关" if r > 0 else "负相关"
        lines.append(f"  • {a} ↔ {b}：r = {r:.3f}（{kind}）")
    return "\n".join(lines)


#: 工具名 → 结构化渲染器。
#:
#: 历史缺陷：这里**只有一个** `eda.distribution_overview` 的渲染器，其余 35 个工具
#: 全部退化成「• 工具名：静态 description」—— 而 EDA 工具的 summary 就是类描述常量，
#: 于是「统计一下数据」「看看异常值」跑完之后，用户看到的是一句工具介绍，
#: 完全没有分析结论（表现为「执行了工具，但没有结论」）。
#: 按工具名分派是刻意的：不同工具的数据结构没有共同 schema，硬写一套通用
#: 渲染器只能得到一堆无意义的键名。拿不到结构的仍走 summary 兜底。
_RESULT_RENDERERS: dict[str, Any] = {
    "eda.distribution_overview": _render_distribution_overview,
    "eda.describe": _render_describe,
    "eda.outlier": _render_outlier,
    "eda.correlation": _render_correlation,
}


def local_result_summary(tool_calls: list[Any]) -> str:
    """无远程 LLM 时，基于**真实 ToolResult.data** 生成最终答案。

    渲染原则：按工具名分派到结构化渲染器，只读数据里的**通用字段**给结论式摘要，
    不为每个数据集硬编码。拿不到可识别结构时退回到逐条工具 summary（最低兜底），
    但绝不只丢一句「已完成 N 个步骤」就把用户打发了。

    ``tool_calls`` 是 ``ToolCallRecord`` 列表（有 ``.tool`` / ``.status`` /
    ``.result``），由 ``runtime._compose_answer`` 传入。
    """
    ok_calls = [c for c in tool_calls if getattr(c, "status", "") == "ok" and getattr(c, "result", None) is not None]
    if not ok_calls:
        return "任务没有产生有效的数据处理结果。"

    parts: list[str] = []
    for c in ok_calls:
        data = getattr(c.result, "data", None)
        renderer = _RESULT_RENDERERS.get(getattr(c, "tool", ""))
        rendered = renderer(data) if renderer is not None else ""
        if rendered:
            parts.append(rendered)
        else:
            summary = (c.result.summary or "").strip()
            parts.append(f"• {c.tool}：{summary}" if summary else f"• {c.tool} 已执行。")

    body = "\n\n".join(parts)
    # 附带真实 signals（机器可读的下一步信号），让用户知道「接下来可以深入什么」。
    signals: list[str] = []
    for c in ok_calls:
        signals.extend(list(getattr(c.result, "signals", []) or []))
    if signals:
        body += "\n\n进一步提示（数据驱动）：" + "、".join(dict.fromkeys(signals))
    return body


def no_llm_notice(reason: str = "") -> str:
    """远程大模型不可用时的**说明性文案** —— 必须区分三种完全不同的处境。

    原先一律写「当前尚未配置可用的大模型。」，但关掉设置页开关时凭据是**原样保留**的
    （`PUT /settings/llm/remote` 只切开关）。几种状态下用户该做的事完全不同：

    - 已配置但停用       ⇒ 去设置页**重新打开开关**（不需要重填 Key）
    - 已配置、调用失败   ⇒ 看 `reason`（欠费 / 超时 / 5xx …）决定是充值还是重试
    - 从未配置           ⇒ 需要**填 API Key**

    合并成一句会让第一种用户白折腾一遍，并且怀疑自己弄丢了配置。

    `reason` 非空即表示「开关开着、凭据也在，但这一次调用失败了」——
    这是欠费 / 限流 / 网络故障时用户唯一能看到的原因，必须原样带出去。
    """
    try:
        from app.core.config import settings

        key_set = bool(settings.LLM_API_KEY)
        remote_on = bool(settings.LLM_REMOTE_ENABLED)
    except Exception:  # noqa: BLE001
        key_set, remote_on = False, True

    tail = ("在停用期间能做的事：**数据类请求不受影响** —— 你说需求，我用平台真实工具执行。\n"
            "试试说：「看看这批数据的分布」。\n\n"
            "需要自由对话时，到「设置 → AI 服务」重新打开开关即可（**凭据保留，不用重填**）。")
    # 「远程这一侧这次没走通」的原因行：欠费、超时、5xx 都从这里透出给用户。
    reason_line = f"远程大模型**这一次调用失败**了：{reason}\n" if reason else ""

    if reason:
        return (reason_line
                + "闲聊回复需要大模型生成，所以这一段没有真正的回答；\n\n"
                + "**数据类请求不受影响** —— 你说需求，我用平台真实工具执行，"
                  "只是最后的自然语言汇总会退回内置规则。\n"
                  "可以到「设置 → AI 服务」点一次「测试连接」确认连通性与账户余额。")
    if key_set and not remote_on:
        return ("远程大模型**已停用**（设置 → AI 服务里的总开关），凭据仍然保留着。\n"
                "闲聊回复需要大模型生成，所以这一块暂时没有输出；\n\n" + tail)
    if key_set and remote_on:
        # 理论上不该走到这里（开关开着且有 Key ⇒ 上层会拿到 Provider）；
        # 真出现了说明 Provider 构造被别处拦下，如实说明而不是假装是配置问题。
        return ("大模型凭据已配置、开关也开着，但这一次没能取到可用的大模型。\n"
                "可以先在「设置 → AI 服务」点一次「测试连接」确认连通性；\n"
                "在此之前，数据类请求仍然可以正常执行。")
    return ("当前**尚未配置**大模型。\n"
            "到「设置 → AI 服务」填 API Key 就能接入远程大模型；不配置也能做数据分析，\n\n" + tail)
