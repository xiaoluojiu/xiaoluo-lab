"""reports/narrator.py —— 让 LLM 基于**真实工具结果**撰写报告正文。

背景：此前报告的文本完全来自 ReportGenerator 的模板句，Agent 只做「调用 → 落盘」，
报告短小、没有分析语言，用户感知不到 Agent 的参与感（"报告文件短小、不像 AI 写的"）。
图表其实已经生成（histogram / heatmap / scatter / bar / qq / area 的内联 SVG），
但因为没有被叙述引用，价值没有被表达出来。

本模块的做法：
1. `build_facts()` 把结构化报告压成一份**事实摘要**（不含 SVG / 原始数据表，省 Token）；
2. LLM 只在这份事实之上写作，输出 JSON
   {"executive_summary", "sections": {标题: 段落}, "conclusions": [...], "recommendations": [...]}；
3. `apply_narration()` 把结果并回 Report：原文（含表格与图表）保留，
   在每个小节追加分析性正文，并补一个「Agent 分析综述」小节与结论/建议。

任何 LLM 失败都静默降级为纯模板报告，绝不阻塞报告生成。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.llm.base import LLMMessage, LLMProvider
from app.core.config import settings
from app.reports.numbering import (
    CHAPTERS,
    chapter_heading,
    refresh_chapter_status,
    strip_section_number,
)

#: 计划表标题 -> 带序号的最终标题。LLM 新增的小节若撞上计划表章节，
#: 必须用计划表的标题（含统一序号），否则会与「未生成章节说明」自相矛盾。
_PLAN_TITLE_TO_HEADING = {title: chapter_heading(key) for key, title in CHAPTERS}

logger = logging.getLogger(__name__)

# 剥离 markdown 代码围栏（```json ... ```）
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# 事实摘要里不要出现的字段（体积大、对写作无帮助）
_DROP_KEYS = {"svg", "data", "matrix", "boxes", "value_counts", "raw"}

# 叙述不完整的重试次数（每次把 max_tokens 放大一倍）
_MAX_NARRATION_ATTEMPTS = 3

# JSON 语法残留：以 `": "` / `,` / `]` / `}` 这类 JSON 标点开头的片段。
# 只在「开头的标点确实是 JSON 残留」时匹配，正常中文散文不会被误伤
# （正文以 `"` 开头但紧跟着文字时不匹配）。
_LEAD_JSON_RESIDUE_RE = re.compile(r'^[\s"\']*(?::|,|\]|\})[\s"\',:\[\]{}]*')


def _repair_inner_quotes(text: str) -> str:
    """转义字符串值内部**未转义**的 ASCII 双引号。

    真实事故（2026-09-23，航空公司延误报告）：LLM 写出

        {"sections": {"二、数据质量": "...把模型训练成"只会预测准点"的退化模型..."}}

    键值结构完全正确，只有内层两个引号没转义。``json.loads`` 当场抛
    ``Expecting ',' delimiter``，而既有的 ``_truncate_salvage`` 只擅长修补「被
    max_tokens 截断」的 JSON（按字符回退 + 补闭合符号），对「长度完整、仅内层引号
    失配」毫无办法且会把好字符砍掉。结果是整份叙述退回「原文当综述」，
    **用户看到的报告正文就是原始 JSON**，还派生出以 `": "` 开头的假章节。

    判定规则：字符串内遇到的引号，**只有当其后第一个非空白字符是 `:`、`,`、`}`、
    `]` 或到达结尾时**才算该字符串的结束引号；否则它是内层引号，必须转义。
    中文引号（“”）与单引号不参与判定，原样保留。
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            # 已转义的序列整体跳过（\" \\ \n \uXXXX），不要再看里面的字符
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch != '"':
            out.append(ch)
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j >= n or text[j] in ":,}]":
            out.append('"')
            in_string = False
        else:
            out.append('\\"')
        i += 1
    return "".join(out)


def _looks_like_json(text: str) -> bool:
    """是否是 JSON 结构（用于决定能否走「当散文用」的降级路径）。"""
    return text.lstrip().startswith(("{", "["))


def _looks_like_raw_json_block(text: str) -> bool:
    """整段看起来就是一段原始 JSON（键值结构明显），绝不能展示给用户。"""
    head = text.lstrip()
    return head.startswith(("{", "[")) and '":' in head[:400] and '"' in head[:400]


def _scrub_narration_text(text: str) -> str:
    """兜底清洗：原始 JSON 与 JSON 标点残留一律不许进入用户可见的正文。

    即使上游修复全部失效，这一层也保证报告里不会出现 `{"executive_summary":` 或
    以 `": "` 开头的片段。
    """
    if not text:
        return ""
    stripped = text.strip()
    if _looks_like_raw_json_block(stripped):
        return ""
    return _LEAD_JSON_RESIDUE_RE.sub("", stripped).strip()



def _missing_parts(narration: dict[str, Any], headings: list[str] | None) -> list[str]:
    """检查 LLM 叙述是否覆盖了所有必须产物，返回缺失项说明。

    判定故意放宽：章节正文允许 LLM 用略微不同的标题（按序号前缀匹配），
    但「所有章节都被覆盖 / 有结论 / 有建议」是硬要求。
    """
    missing: list[str] = []
    if not str(narration.get("executive_summary") or "").strip():
        missing.append("executive_summary")
    provided = {str(k) for k in (narration.get("sections") or {})}
    for heading in headings or []:
        if heading not in provided and not _same_section(heading, provided):
            missing.append(f"章节正文：{heading}")
    if len([c for c in (narration.get("conclusions") or []) if str(c).strip()]) < 4:
        missing.append("conclusions(≥4)")
    if len([c for c in (narration.get("recommendations") or []) if str(c).strip()]) < 3:
        missing.append("recommendations(≥3)")
    return missing


def _same_section(heading: str, provided: set[str]) -> bool:
    """『一、数据概览』与『数据概览』视为同一节（LLM 常省略序号前缀）。"""
    core = strip_section_number(heading)
    return any(strip_section_number(p) == core for p in provided)


# 结论/建议可能被 LLM 塞进 sections 里（而不是顶层）。真实事故：
#   sections = {..., "conclusions": [...], "recommendations": [...]}，顶层两者皆无。
# 后果有两层：① _missing_parts 判定「缺 conclusions/recommendations」→ 白重试；
# ② 它们被当成两个名为 conclusions/recommendations 的**章节**，
# 正文是 Python 列表的 repr（"['数据规模 10000000 行 …']"）。
_LIST_SECTION_KEYS = {
    "conclusions", "conclusion", "recommendations", "recommendation",
    "suggestions", "suggestion", "summary_bullets", "结论", "建议", "结论与建议",
}


def _as_prose(value: Any) -> str:
    """把章节值统一成正文文本；列表按段落拼接，其他非字符串一律不生成 repr。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "\n\n".join(parts)
    return ""


def _normalize_sections(narration: dict[str, Any]) -> dict[str, Any]:
    """LLM 有时把 sections 写成 [{heading, content}] 数组，统一归一成 {标题: 正文}。

    归一后还要做两件「结构纠偏」，都是真实踩过的：
    1. 把误放进 sections 的 conclusions / recommendations **提升**到顶层；
    2. 丢掉不是字符串（也不是字符串列表）的 section 值 —— 否则 str(list)
       会把 Python repr 写进用户可见的正文。
    """
    raw = narration.get("sections")
    if isinstance(raw, list):
        mapping: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                heading = str(item.get("heading") or item.get("title") or "").strip()
                body = _as_prose(item.get("content") or item.get("text") or item.get("paragraph"))
                if heading:
                    mapping[heading] = body
            elif isinstance(item, str) and item.strip():
                mapping[f"段落 {len(mapping) + 1}"] = item.strip()
        narration["sections"] = mapping
    elif isinstance(raw, dict):
        narration["sections"] = {str(k): _as_prose(v) for k, v in raw.items()}
    else:
        narration["sections"] = {}

    sections = narration["sections"]
    for key in list(sections):
        normalized = str(key).strip().lower()
        if normalized not in _LIST_SECTION_KEYS:
            continue
        # 顶层已有就不覆盖（顶层优先，那是模型按 schema 写对的部分）
        if not narration.get(key):
            narration[key] = _split_bullets(sections[key])
        del sections[key]
    # 清掉正文为空的章节，避免报告里出现「有标题、没内容」的空壳
    narration["sections"] = {k: v for k, v in sections.items() if str(v).strip()}
    return narration


def _split_bullets(text: str) -> list[str]:
    """把「段落拼成的字符串」拆回条目列表（提升误嵌套键时用）。"""
    items = [line.strip(" -·•\t") for line in str(text).splitlines()]
    return [i for i in items if i]


def _chart_fact(chart: dict[str, Any]) -> dict[str, Any]:
    """把一张图表压成 LLM 可叙述的极简事实（标题 + 关键数值摘要）。

    普通图只给 title（让 LLM 能引用「如图〈xxx〉所示」）；趋势图（line）附带
    trend_summary（首/末/均值/方向），让 LLM 能基于真实数值解读趋势，避免
    「贴了图却写无法判断」的图文脱节（回归）。
    """
    fact: dict[str, Any] = {"type": chart.get("type"), "title": chart.get("title")}
    data = chart.get("data") or {}
    if isinstance(data, dict) and data.get("trend_summary"):
        fact["trend"] = data["trend_summary"]
    return fact


def build_facts(report: Any, *, max_chars: int | None = None) -> dict[str, Any]:
    """把 Report 压成 LLM 可消费的事实摘要（纯标量 + 少量列表）。"""
    limit = max_chars or settings.AGENT_REPORT_NARRATION_MAX_CHARS
    payload: dict[str, Any] = {
        "title": report.get("title") if isinstance(report, dict) else getattr(report, "title", ""),
    }
    dataset = report.get("dataset") if isinstance(report, dict) else getattr(report, "dataset", {})
    payload["dataset"] = _shrink(dataset, 900)

    sections = report.get("sections") if isinstance(report, dict) else [
        s.to_dict() for s in getattr(report, "sections", [])
    ]
    facts_sections = []
    for sec in sections or []:
        item: dict[str, Any] = {"heading": sec.get("heading")}
        content = str(sec.get("content") or "")
        if content:
            item["template_text"] = content[:900]
        tables = []
        for table in sec.get("tables") or []:
            rows = [[_scalar(c) for c in row] for row in (table.get("rows") or [])[:15]]
            tables.append({"title": table.get("title"), "headers": table.get("headers"), "rows": rows})
        if tables:
            item["tables"] = tables
        charts = [
            _chart_fact(c) for c in (sec.get("charts") or [])
        ]
        if charts:
            item["charts"] = charts
        facts_sections.append(item)
    payload["sections"] = facts_sections

    charts = report.get("charts") if isinstance(report, dict) else getattr(report, "charts", [])
    payload["chart_titles"] = [c.get("title") for c in (charts or []) if isinstance(c, dict)]
    experiments = report.get("experiments") if isinstance(report, dict) else getattr(report, "experiments", [])
    payload["experiments"] = _shrink(experiments, 1200)
    conclusions = report.get("conclusions") if isinstance(report, dict) else getattr(report, "conclusions", [])
    payload["existing_conclusions"] = [str(c) for c in (conclusions or [])][:8]

    return _fit_limit(payload, limit)


def _fit_limit(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    """逐级降级裁剪事实摘要，直到 JSON 长度不超过 limit。

    历史缺陷：原实现把裁剪后的字符串赋给了局部变量 text 就 `return payload`，
    返回的是**未裁剪**的对象，导致 AGENT_REPORT_NARRATION_MAX_CHARS 形同虚设；
    大报告的事实摘要会直接击穿 LLM 上下文（外层表现为叙述静默失败、报告退回模板）。

    降级顺序按「体积大且对写作价值低」优先：
    章节表格 → 模板正文截短 → 去掉模板正文 → 实验/已有结论/图表标题/dataset → 尾部章节。
    """
    def size_of(data: dict[str, Any]) -> int:
        return len(json.dumps(data, ensure_ascii=False, default=str))

    if size_of(payload) <= limit:
        return payload
    payload["truncated"] = True

    sections = payload.get("sections")
    if not isinstance(sections, list):
        return payload

    for degrade in (
        lambda: [sec.pop("tables", None) for sec in sections],
        lambda: [sec.__setitem__("template_text", str(sec.get("template_text") or "")[:240]) for sec in sections],
        lambda: [sec.pop("template_text", None) for sec in sections],
    ):
        degrade()
        if size_of(payload) <= limit:
            return payload

    for key in ("experiments", "existing_conclusions", "chart_titles", "dataset"):
        payload.pop(key, None)
        if size_of(payload) <= limit:
            return payload

    # 最后才牺牲章节：从尾部开始丢弃，至少保留第一节能让 LLM 有依据可写
    while len(sections) > 1 and size_of(payload) > limit:
        payload["sections"] = sections = sections[:-1]
    return payload


def default_provider() -> LLMProvider | None:
    """进程级共享 LLM Provider（用于非 Agent-Run 上下文的报告叙述）。"""
    try:
        from app.api.deps import get_llm_provider
        return get_llm_provider()
    except Exception as exc:  # noqa: BLE001
        logger.debug("未配置可用的 LLM Provider：%s", exc)
        return None


def narrate_report(report_payload: dict[str, Any], provider: LLMProvider | None, *, user_request: str = "") -> dict[str, Any] | None:
    """调用 LLM 撰写报告叙述。失败返回 None（调用方走模板兜底）。"""
    if provider is None or not settings.AGENT_REPORT_NARRATION:
        return None
    facts = build_facts(report_payload)
    headings = [str(s.get("heading")) for s in (facts.get("sections") or [])]
    system = (
        "你是小洛实验室的数据科学家助手，现在要为一份数据分析报告撰写正文。"
        "你只能使用「事实摘要」中出现的数值与字段，严禁编造没有出现过的列、指标或结论；"
        "没有依据时写『样本/指标不足，暂无法判断』。"
        "写作要求：专业、具体、有信息量，面向要做决策的人；每个小节 2-4 段，每段 40-120 字，"
        "解释**数字意味着什么**、有什么风险、下一步该做什么。"
        "必须引用图表标题（例如『如图〈xxx 分布直方图〉所示』），把图表和文字串起来。"
        "全文控制在 1500 字以内。段落之间用 \\n 分隔，正文内不要用 Markdown 标题/表格/列表符号。"
        "输出**纯 JSON 对象**，不要任何解释文字，形如："
        '{"executive_summary": str, "sections": {小节标题: 正文}, "conclusions": [str], "recommendations": [str]}。'
        f"硬性要求：\"sections\" 必须覆盖下面列出的**每一个** heading，一个都不能少：{headings}；"
        "\"conclusions\" 至少 4 条，\"recommendations\" 至少 3 条，都要结合上面的真实数值。"
        # 数字口径红线（防章节间数据矛盾）：LLM 复述结构化数字时极易张冠李戴，
        # 例如把 fare_amount 的异常值数量安到 total_amount 头上。因此：
        "【数字口径红线，必须遵守】"
        "1. 每个数值必须与其字段名严格一一对应，从事实摘要原样照抄，禁止把 A 字段的数字写成 B 字段；"
        "2. 同一字段的同一指标，全篇只能出现一个数值，综述、正文、结论必须一致；"
        "3. 拿不准某个数字属于哪个字段时，就**省略该数字**、只写定性结论，绝不要猜；"
        "4. 异常值/缺失值的『数量 + 占比』要成对出现且与字段绑定，禁止只写数量不写占比或反之；"
        "5. 结论与建议里避免罗列大段数字，数字交给正文表格承载，你只提炼『意味着什么』。"
    )
    user = (
        f"用户原始诉求：{user_request or '对该数据集做全面分析并形成报告'}\n\n"
        f"事实摘要：\n{json.dumps(facts, ensure_ascii=False, default=str)}"
    )
    messages = [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]
    base_tokens = min(int(settings.AGENT_LLM_MAX_OUTPUT_TOKENS or 4096), 6000)

    # JSON 被 max_tokens 截断时，前面救回来的章节能合并，但结论/建议/最后一个章节会静默丢失
    # （曾表现为：报告的「探索性分析」正文与全部 LLM 结论都不见了）。
    # 因此这里做「完整性校验 + 放大预算重试」，只有不达标才多花 Token。
    narration: dict[str, Any] | None = None
    for attempt in range(_MAX_NARRATION_ATTEMPTS):
        content = ""
        max_tokens = min(base_tokens * (attempt + 1), 16384)
        # 优先走 JSON Output 模式（DeepSeek / OpenAI 兼容协议均支持），拿到的必然是合法 JSON；
        # 不支持的网关会报错，个别情况下也会返回空 content，两者都降级为普通对话 + 容错解析。
        for extra in ({"response_format": {"type": "json_object"}}, {}):
            try:
                response = provider.chat(messages, max_tokens=max_tokens, **extra)
                candidate = (response.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("报告叙述 LLM 调用失败（%s）：%s", "json_object" if extra else "plain", exc)
                continue
            if candidate:
                content = candidate
                break
            logger.warning("报告叙述 LLM 返回空内容（%s），换用普通对话重试", "json_object" if extra else "plain")
        if not content:
            continue
        narration = _parse_narration(content, headings)
        if isinstance(narration, dict):
            narration = _normalize_sections(narration)
        missing = [] if narration is None else _missing_parts(narration, headings)
        if narration and not missing:
            return narration
        logger.warning(
            "报告叙述不完整（第 %s 次尝试，max_tokens=%s），缺失：%s；放大预算重试",
            attempt + 1, max_tokens, missing or "无法解析",
        )
    return narration


def apply_narration(
    report_payload: dict[str, Any],
    narration: dict[str, Any] | None,
    *,
    user_request: str = "",
) -> dict[str, Any]:
    """把 LLM 叙述并回报告 payload（原地修改并返回）。"""
    if not narration:
        return report_payload
    # 幂等归一：调用方可能传入未经 _normalize_sections 的 narration
    # （例如直接构造 narration 的其他入口/测试），这里再兜一次结构纠偏。
    narration = _normalize_sections(dict(narration))
    sections = report_payload.setdefault("sections", [])
    known_headings = [str(s.get("heading")) for s in sections if isinstance(s, dict)]
    by_heading = {str(s.get("heading")): s for s in sections if isinstance(s, dict)}

    exec_summary = _scrub_narration_text(str(narration.get("executive_summary") or ""))
    if exec_summary:
        sections.insert(
            0,
            {
                "heading": "Agent 分析综述",
                "content": exec_summary,
                "tables": [],
                "charts": [],
            },
        )
        by_heading["Agent 分析综述"] = sections[0]

    for heading, text in (narration.get("sections") or {}).items():
        # 兜底清洗：任何情况下都不让原始 JSON / JSON 标点残留 / 列表 repr 进入正文
        body = _scrub_narration_text(_as_prose(text))
        if not body:
            continue
        heading_text = str(heading)
        target = by_heading.get(heading_text)
        if target is None:
            # LLM 常省略「一、」这类序号前缀，不能因为标题字面对不上就把整段正文丢掉
            for known in known_headings:
                if _same_section(known, {heading_text}):
                    target = by_heading[known]
                    break
        if target is None:
            # 新增小节的标题一律去掉序号前缀：序号由 app.reports.numbering 的计划表
            # 统一分配。若放任 LLM 自带序号，会出现「四、建模建议」与计划表里的
            # 「四、建模与评估」撞号（真实事故：报告章节号一/二/三/六）。
            new_heading = strip_section_number(heading_text) or heading_text
            new_heading = _PLAN_TITLE_TO_HEADING.get(new_heading, new_heading)
            target = {"heading": new_heading, "content": "", "tables": [], "charts": []}
            sections.append(target)
            by_heading[new_heading] = target
            known_headings.append(new_heading)
        original = str(target.get("content") or "").strip()
        target["content"] = f"{original}\n\n{body}" if original else body

    conclusions = [
        c for c in (_scrub_narration_text(str(x)) for x in (narration.get("conclusions") or [])) if c
    ]
    recommendations = [
        c for c in (_scrub_narration_text(str(x)) for x in (narration.get("recommendations") or [])) if c
    ]
    existing = list(report_payload.get("conclusions") or [])
    # 保留自动归纳的客观结论（质量分、任务结论），再叠加 LLM 结论与建议
    merged = existing + [c for c in conclusions if c not in existing]
    if recommendations:
        merged.extend(f"建议：{r}" for r in recommendations)
    report_payload["conclusions"] = merged
    meta = report_payload.setdefault("metadata", {})
    meta["narrated_by"] = "agent-llm"
    if user_request:
        meta["user_request"] = user_request
    if recommendations:
        meta["recommendations"] = recommendations
    # 叙述可能补齐了原本未生成的章节，章节计划状态必须同步（否则自相矛盾）
    refresh_chapter_status(report_payload)
    return report_payload


def narrate_and_apply(
    report_payload: dict[str, Any],
    provider: LLMProvider | None,
    *,
    user_request: str = "",
) -> dict[str, Any]:
    """一步到位：生成叙述并写回报告（内部已做异常兜底）。"""
    try:
        narration = narrate_report(report_payload, provider, user_request=user_request)
    except Exception as exc:  # noqa: BLE001
        logger.warning("报告叙述生成异常：%s", exc)
        narration = None
    return apply_narration(report_payload, narration, user_request=user_request)


# ----------------------------------------------------------------------

def _parse_narration(text: str, headings: list[str] | None = None) -> dict[str, Any] | None:
    """容错解析 LLM 的叙述输出。

    LLM 常见的四种「看起来像 JSON 但解析不了」的输出：
    1. 用 ```json 围栏包裹；
    2. 段落里出现**真实换行**（默认 json 解析器会因控制字符报错）→ strict=False；
    3. 被 max_tokens 截断，JSON 未闭合 → 逐步回退补齐闭合符号抢救；
    4. **字符串值里有未转义的 ASCII 双引号**（本次事故）→ `_repair_inner_quotes`。

    关键约定：**原始 JSON 绝不能当正文用**。既有的降级做法（把原文塞进
    executive_summary + 用 `text.find(标题)` 在 JSON 里切片）会把
    `{"executive_summary": ...}` 和 `": "…` 直接写进报告，是本次事故的放大器。
    因此这里改成：JSON 结构但解析失败 ⇒ 只做结构抢救；抢不回来就**放弃本次叙述**
    （报告退回模板正文），而不是把 JSON 展示给用户。只有**非 JSON 的长文**才走
    「整段当综述 + 按标题切段」的降级路径。
    """
    data = _loads_lenient(text)
    if isinstance(data, dict):
        return data
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
    if not stripped:
        return None
    if _looks_like_json(stripped):
        salvaged = _salvage_json_pairs(stripped)
        if salvaged:
            logger.warning("报告叙述 JSON 损坏，已按键值结构抢救出 %s 个字段", len(salvaged))
            return salvaged
        logger.warning("报告叙述是损坏的 JSON 且无法抢救，放弃本次叙述（报告退回模板正文）")
        return None
    logger.warning("报告叙述非 JSON 输出，降级为整段综述")
    return {
        "executive_summary": _scrub_narration_text(stripped[:6000]),
        "sections": _guess_sections(stripped, headings or []),
    }


# 「短键」的定义：不带换行、长度受限的 JSON 键名（避免把正文里的引号当键）
_JSON_KEY_RE = re.compile(r'"([^"\\\n]{1,40})"\s*:\s*"')


def _salvage_json_pairs(text: str) -> dict[str, Any] | None:
    """结构抢救：先整体修引号，再逐键取值，最后退到只抓已知字段。

    返回 None 表示「抢不回来」，由调用方决定放弃本次叙述。
    """
    for candidate in (text, _repair_inner_quotes(text)):
        try:
            value = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    repaired = _repair_inner_quotes(text)
    out: dict[str, Any] = {}
    # 逐对抽取「键: 字符串值」。值一律用 _repair_inner_quotes 同款规则找结束引号，
    # 这样内层未转义引号不会把值截断。
    for match in _JSON_KEY_RE.finditer(repaired):
        key = match.group(1)
        start = match.end()
        end = _find_string_end(repaired, start)
        if end == -1:
            continue
        raw = repaired[start:end]
        try:
            out[key] = json.loads(f'"{raw}"', strict=False)
        except json.JSONDecodeError:
            out[key] = raw.replace('\\"', '"')
    if not out:
        return None
    sections = out.pop("sections", None)
    if isinstance(sections, str):
        # sections 被抽成字符串（整体仍是坏的）：不勉强还原，交给模板正文
        sections = None
    if sections is None:
        # 只抢救出扁平字段（executive_summary / conclusions / recommendations）
        # 时，章节正文缺失会让 _missing_parts 判定不完整并触发重试，这是期望行为。
        sections = {}
    out["sections"] = sections if isinstance(sections, dict) else {}
    return out if (out.get("executive_summary") or sections) else None


def _find_string_end(text: str, start: int) -> int:
    """从 start 起找到字符串值的结束引号位置（同 _repair_inner_quotes 的判定规则）。"""
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ":,}]":
                return i
        i += 1
    return -1


def _loads_lenient(text: str) -> Any:
    candidates: list[str] = []
    for match in [None, *_FENCE_RE.finditer(text)]:
        base = text.strip() if match is None else match.group(1)
        if not base:
            continue
        candidates.append(base)
        # 未转义内层引号是最高频的 LLM JSON 缺陷，先修它再谈截断抢救
        repaired = _repair_inner_quotes(base)
        if repaired != base:
            candidates.append(repaired)
    for candidate in candidates:
        for raw in (candidate, _truncate_salvage(candidate)):
            if not raw:
                continue
            try:
                value = json.loads(raw, strict=False)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, dict):
                    return value
            decoder = json.JSONDecoder(strict=False)
            for start_char in ("{", "["):
                start = raw.find(start_char)
                while start != -1:
                    try:
                        value, _end = decoder.raw_decode(raw[start:])
                    except json.JSONDecodeError:
                        start = raw.find(start_char, start + 1)
                    else:
                        if isinstance(value, dict):
                            return value
                        start = raw.find(start_char, start + 1)
    return None


def _truncate_salvage(text: str) -> str:
    """对被截断的 JSON 逐步追加闭合符号，尽量抢救出可用对象。"""
    for _ in range(200):
        try:
            value = json.loads(text, strict=False)
        except json.JSONDecodeError as exc:
            # 还在字符串内部：截掉最后一个字符再闭合
            open_quotes = text.count('"') % 2 == 1
            text = text[:-1] if open_quotes else text
            if exc.pos and exc.pos < len(text) and text[exc.pos] in ("}", "]", ","):
                text = text[: exc.pos]
            text = (text.rstrip().rstrip(",") + ('"' if open_quotes else "") + "}")
            continue
        else:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else ""
    return ""


def _guess_sections(text: str, headings: list[str]) -> dict[str, str]:
    """把非结构化长文按小节标题切成段落映射（尽力而为）。

    只在**非 JSON 的散文**上使用。切片结果仍要过一遍 `_scrub_narration_text`：
    标题本身可能出现在引号或 JSON 键里，切出来就会带上 `": "` 这类残留。
    """
    if not headings:
        return {}
    positions = [(h, text.find(h)) for h in headings if text.find(h) != -1]
    positions.sort(key=lambda x: x[1])
    out: dict[str, str] = {}
    for idx, (heading, pos) in enumerate(positions):
        end = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
        body = text[pos + len(heading): end]
        cleaned = _scrub_narration_text(body)
        if cleaned:
            out[heading] = cleaned
    return out


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)[:200]
    return value


def _shrink(value: Any, max_chars: int) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return value
    if len(text) <= max_chars:
        return _drop_big_keys(value)
    return {"_truncated_preview": text[:max_chars]}


def _drop_big_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_big_keys(v) for k, v in value.items() if k not in _DROP_KEYS}
    if isinstance(value, list):
        return [_drop_big_keys(v) for v in value]
    return value
