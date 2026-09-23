"""报告章节编号与「未生成章节」标注。

真实缺陷（来自航空公司延误数据集的 Agent 报告）：
章节序号在生成器里写死「一、数据概览 … 四、建模与评估」，在三个导出器里又各自
写死「五、关联实验」「六、结论」。当某一章因为缺少上游结果而没生成时——例如
Agent 只跑了质量检查与 EDA、没有执行 ``ml.train``——报告正文就会出现
**「一 / 二 / 三 / 六」这种跳号，且没有任何说明**。读者无法分辨这是
「该章确实没有」还是「编号写错了」，在毕业论文/正式实验报告场景里是硬伤。

本模块把章节计划收敛到唯一一处，并提供三件事：

1. ``chapter_heading(key)`` —— 按计划表取「序号 + 标题」，生成器与导出器共用，
   杜绝同一章在不同导出格式里编号不一致；
2. ``strip_section_number(heading)`` —— 去掉任意来源标题上的序号前缀。
   LLM 叙述常把「数据概览」写成「一、数据概览」或「1. 数据概览」，
   序号由渲染层统一负责，内容层不该自带（否则会出现「四、建模建议」
   与计划表里的「四、建模与评估」撞号）；
3. ``chapter_status(obj)`` —— 算出完整章节计划里每一章是否生成、未生成的原因，
   供报告显式标注「本章未生成：因为…」，而不是静默跳号。
"""

from __future__ import annotations

import re
from typing import Any

#: 章节计划：(key, 中文标题)。顺序即报告章节顺序，序号由此推导。
CHAPTERS: tuple[tuple[str, str], ...] = (
    ("overview", "数据概览"),
    ("quality", "数据质量"),
    ("eda", "探索性分析"),
    ("ml", "建模与评估"),
    ("experiments", "关联实验"),
    ("conclusions", "结论"),
)

_CN_DIGITS = "零一二三四五六七八九"

#: 未生成章节的默认原因（具体、可操作，而不是「无数据」这种废话）。
DEFAULT_REASONS: dict[str, str] = {
    "overview": "未获取到数据集信息。",
    "quality": "本次分析未执行数据质量检查（dataset.quality），无法给出质量得分与问题明细。",
    "eda": "本次分析未执行探索性分析（eda.describe / eda.correlation / eda.outlier），本章没有可写的统计结果。",
    "ml": "本次分析未关联到已完成的建模结果：没有执行 ml.train / ml.evaluate，也没有可用的 Experiment / Run。",
    "experiments": "本次分析未产生 Experiment / Run 记录，因此没有可关联的实验。",
    "conclusions": "本次分析未产生结论条目。",
}

#: 标题上的序号前缀：一、 / 1. / （一） / 第一章 / 第 2 节 等。
_SECTION_NUMBER_RE = re.compile(
    r"^\s*[（(]?\s*(?:第\s*)?([一二三四五六七八九十百零〇]|[0-9]{1,2})"
    r"(?:\s*[、.．,，:：)）]|\s*[章节部分])"
)


def cn_number(n: int) -> str:
    """1 -> 一，10 -> 十，11 -> 十一，21 -> 二十一。

    章节数不超过 99；超出范围退回阿拉伯数字（宁可不美化，也不能算错）。
    """
    if n <= 0:
        return str(n)
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        return "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    if n < 100:
        tens, ones = divmod(n, 10)
        return _CN_DIGITS[tens] + "十" + (_CN_DIGITS[ones] if ones else "")
    return str(n)


def chapter_index(key: str) -> int:
    """章节 key -> 从 1 开始的序号；未知 key 返回 0。"""
    for i, (k, _) in enumerate(CHAPTERS):
        if k == key:
            return i + 1
    return 0


def chapter_title(key: str) -> str:
    for k, title in CHAPTERS:
        if k == key:
            return title
    return key


def chapter_heading(key: str) -> str:
    """章节 key -> 「一、数据概览」这样的最终标题。"""
    idx = chapter_index(key)
    if idx <= 0:
        return chapter_title(key)
    return f"{cn_number(idx)}、{chapter_title(key)}"


def strip_section_number(heading: Any) -> str:
    """去掉标题上的序号前缀，返回纯标题。

    ``strip_section_number("一、数据概览") == "数据概览"``
    ``strip_section_number("1. Overview") == "Overview"``
    ``strip_section_number("一般性说明") == "一般性说明"``  # 无分隔符，不动
    """
    text = str(heading or "").strip()
    if not text:
        return ""
    return _SECTION_NUMBER_RE.sub("", text, count=1).strip()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _heading_set(sections: Any) -> set[str]:
    out: set[str] = set()
    for sec in sections or []:
        heading = sec.get("heading") if isinstance(sec, dict) else getattr(sec, "heading", None)
        core = strip_section_number(heading)
        if core:
            out.add(core)
    return out


def chapter_status(obj: Any) -> list[dict[str, Any]]:
    """返回完整章节计划及每章的生成状态。

    优先采用元数据里已记录的 ``chapters``（生成器写入，原因更具体）；
    缺失/非法时按报告实际内容推断，保证外部传入的历史报告也能正确渲染。

    :param obj: ``Report`` 实例或报告 dict。
    """
    sections = _get(obj, "sections", []) or []
    experiments = _get(obj, "experiments", []) or []
    conclusions = _get(obj, "conclusions", []) or []
    metadata = _get(obj, "metadata", {}) or {}

    recorded = metadata.get("chapters") if isinstance(metadata, dict) else None
    if isinstance(recorded, list) and recorded:
        plan: list[dict[str, Any]] = []
        for item in recorded:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "")
            if not key:
                continue
            plan.append(
                {
                    "key": key,
                    "index": chapter_index(key),
                    "heading": chapter_heading(key),
                    "generated": bool(item.get("generated")),
                    "reason": (
                        None
                        if item.get("generated")
                        else str(item.get("reason") or DEFAULT_REASONS.get(key) or "本章未生成。")
                    ),
                }
            )
        if plan:
            return plan

    headings = _heading_set(sections)
    generated = {
        "overview": True,  # 概览是所有报告的第一章，生成器无条件产出
        "quality": chapter_title("quality") in headings,
        "eda": chapter_title("eda") in headings,
        "ml": chapter_title("ml") in headings,
        "experiments": bool(experiments),
        "conclusions": bool(conclusions),
    }
    return [
        {
            "key": key,
            "index": chapter_index(key),
            "heading": chapter_heading(key),
            "generated": generated.get(key, False),
            "reason": None if generated.get(key) else DEFAULT_REASONS.get(key, "本章未生成。"),
        }
        for key, _ in CHAPTERS
    ]


def missing_chapters(obj: Any) -> list[dict[str, Any]]:
    """只返回未生成的章节（含原因），渲染「未生成章节说明」用。"""
    return [item for item in chapter_status(obj) if not item["generated"]]


def refresh_chapter_status(obj: Any) -> list[dict[str, Any]]:
    """LLM 叙述追加/补齐章节后，重算计划状态并写回 ``metadata["chapters"]``。

    必须重算的原因：叙述可能补齐了原本未生成的章节（例如上游没跑 ml.train，
    但 LLM 依据上下文写了一段「建模与评估」）。此时若不同步状态，报告正文里
    会同时出现该章内容与「该章未生成」的说明，自相矛盾。

    :param obj: 报告 dict（``Report`` 实例亦可，但只读）。返回重算后的计划。
    """
    if not isinstance(obj, dict):
        return []
    sections = obj.get("sections") or []
    headings = _heading_set(sections)
    metadata = obj.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return []
    recorded = metadata.get("chapters")
    plan: list[dict[str, Any]] = []
    for key, title in CHAPTERS:
        was = next(
            (r for r in (recorded or []) if isinstance(r, dict) and str(r.get("key")) == key),
            None,
        )
        generated = title in headings or bool(was and was.get("generated"))
        plan.append(
            {
                "key": key,
                "index": chapter_index(key),
                "heading": chapter_heading(key),
                "generated": generated,
                "reason": None if generated else DEFAULT_REASONS.get(key, "本章未生成。"),
            }
        )
    metadata["chapters"] = plan
    return plan
