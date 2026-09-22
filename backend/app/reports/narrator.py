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

logger = logging.getLogger(__name__)

# 剥离 markdown 代码围栏（```json ... ```）
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# 事实摘要里不要出现的字段（体积大、对写作无帮助）
_DROP_KEYS = {"svg", "data", "matrix", "boxes", "value_counts", "raw"}

# 叙述不完整的重试次数（每次把 max_tokens 放大一倍）
_MAX_NARRATION_ATTEMPTS = 3


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
    core = re.sub(r"^[一二三四五六七八九十]+、\s*", "", str(heading)).strip()
    return any(re.sub(r"^[一二三四五六七八九十]+、\s*", "", p).strip() == core for p in provided)


def _normalize_sections(narration: dict[str, Any]) -> dict[str, Any]:
    """LLM 有时把 sections 写成 [{heading, content}] 数组，统一归一成 {标题: 正文}。"""
    raw = narration.get("sections")
    if isinstance(raw, list):
        mapping: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                heading = str(item.get("heading") or item.get("title") or "").strip()
                body = str(item.get("content") or item.get("text") or item.get("paragraph") or "").strip()
                if heading:
                    mapping[heading] = body
            elif isinstance(item, str) and item.strip():
                mapping[f"段落 {len(mapping) + 1}"] = item.strip()
        narration["sections"] = mapping
    elif not isinstance(raw, dict):
        narration["sections"] = {}
    return narration


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
    sections = report_payload.setdefault("sections", [])
    known_headings = [str(s.get("heading")) for s in sections if isinstance(s, dict)]
    by_heading = {str(s.get("heading")): s for s in sections if isinstance(s, dict)}

    exec_summary = str(narration.get("executive_summary") or "").strip()
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
        body = str(text or "").strip()
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
            target = {"heading": heading_text, "content": "", "tables": [], "charts": []}
            sections.append(target)
            by_heading[heading_text] = target
            known_headings.append(heading_text)
        original = str(target.get("content") or "").strip()
        target["content"] = f"{original}\n\n{body}" if original else body

    conclusions = [str(c).strip() for c in (narration.get("conclusions") or []) if str(c).strip()]
    recommendations = [str(c).strip() for c in (narration.get("recommendations") or []) if str(c).strip()]
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

    LLM 常见的三种「看起来像 JSON 但解析不了」的输出：
    1. 用 ```json 围栏包裹；
    2. 段落里出现**真实换行**（默认 json 解析器会因控制字符报错）→ strict=False；
    3. 被 max_tokens 截断，JSON 未闭合 → 逐步回退补齐闭合符号抢救。
    """
    data = _loads_lenient(text)
    if isinstance(data, dict):
        return data
    # 完整内容救不回来时，至少把文字当成综述，不让整段 LLM 输出白白浪费
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
    if stripped:
        logger.warning("报告叙述非 JSON 输出，降级为整段综述")
        return {"executive_summary": stripped[:6000], "sections": _guess_sections(stripped, headings or [])}
    return None


def _loads_lenient(text: str) -> Any:
    candidates = [text.strip()]
    candidates.extend(match.group(1) for match in _FENCE_RE.finditer(text))
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
    """把非结构化长文按小节标题切成段落映射（尽力而为）。"""
    if not headings:
        return {}
    positions = [(h, text.find(h)) for h in headings if text.find(h) != -1]
    positions.sort(key=lambda x: x[1])
    out: dict[str, str] = {}
    for idx, (heading, pos) in enumerate(positions):
        end = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
        out[heading] = text[pos + len(heading): end].strip("：:\n ")
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
