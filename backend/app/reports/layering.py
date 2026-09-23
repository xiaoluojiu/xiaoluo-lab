"""Reports · 报告分层与决策化（第六层改造）。

改造前的问题
------------
报告列了 20 个数字，但没说「所以呢」：结论 7 条、建议 4 条，内容与正文
高度重复，没有优先级，结尾也没回答「下一步该跑什么实验」。

本模块把报告的结论区标准化为三层，每层回答一个**不同**的问题：

============  ==============================  ==================================
事实层 fact     数据说了什么                    描述性统计、指标数字、分布形态
判断层 judgment 这意味着什么                    方法局限、风险提示、可信度
行动层 action   下一步做什么                    带优先级的建议（必须/建议/可选）
============  ==============================  ==================================

三条硬约束
----------
1. **每条发现对应一条或多条行动**——没有对应行动的「发现」只是陈述，
   不进结论区；
2. **行动必须带优先级**，且同优先级内按层序排列，不平铺 10 条；
3. **重复必须去掉**：同一件事在正文和结论里各说一遍等于没说，
   :func:`dedupe` 归一化后做包含度比较，保留信息量更大的那一条。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PRIORITIES",
    "ReportItem",
    "ReportLayers",
    "build_layers",
    "classify_layer",
    "dedupe",
    "normalize",
    "render_markdown",
]

#: 优先级词表（与特征工程建议器共用同一套）
PRIORITIES: tuple[str, ...] = ("必须做", "建议做", "可选做")

_PUNCT_RE = re.compile(r"[\s，。；、：,.;:！!？?（）()《》\"'“”‘’\-—_/\\|]+")

#: 强动作词。刻意**不含**「需」「应」「做」这类弱词——它们大量出现在判断句里
#: （「需注意…」「应结合业务判断…」），把它们算成行动会把判断层清空。
_ACTION_HINTS = (
    "建议", "请", "下一步", "重跑", "清洗", "删除", "补齐", "指定", "改用",
    "调整", "调优", "复核", "补充", "做特征工程", "先处理", "后再", "换用", "落地",
)
_JUDGMENT_HINTS = (
    "可能", "注意", "局限", "不代表", "风险", "谨慎", "无法", "不一定",
    "不适用于", "待确认", "疑似", "可疑", "应结合",
)
_MUST_HINTS = ("必须", "阻塞", "不能", "撑爆", "错误", "失败", "泄漏")
_OPTIONAL_HINTS = ("可选", "锦上添花", "视情况")

#: 显式优先级前缀（如「必须做：…」）——命中即判为行动层
_PRIORITY_PREFIX_RE = re.compile(r"^\s*(必须做|建议做|可选做)\s*[:：]?")


def normalize(text: str) -> str:
    """归一化：去掉空白与标点，用于去重比较。"""
    return _PUNCT_RE.sub("", str(text or "")).lower()


def dedupe(texts: list[str], *, min_len: int = 8) -> list[str]:
    """保序去重。

    判定重复的两条规则：
    * 归一化后完全相同；
    * 归一化后一条**包含**另一条 —— 保留更长的那条（信息量更大）。

    为什么不只判相等：结论里「数据存在 12 个质量问题」与
    「数据存在 12 个质量问题，其中 3 个需在建模前处理」是同一件事的两种写法，
    只判相等会两条都留，读者看到的就是「重复」。
    """
    kept: list[str] = []
    for text in texts:
        raw = str(text or "").strip()
        if not raw:
            continue
        key = normalize(raw)
        if not key:
            continue
        replaced = False
        for i, existing in enumerate(kept):
            ekey = normalize(existing)
            if key == ekey:
                if len(raw) > len(existing):
                    kept[i] = raw
                replaced = True
                break
            if len(key) >= min_len and len(ekey) >= min_len:
                if key in ekey:
                    replaced = True
                    break
                if ekey in key:
                    kept[i] = raw
                    replaced = True
                    break
        if not replaced:
            kept.append(raw)
    return kept


def classify_layer(text: str) -> str:
    """按措辞判定一句话属于哪一层（事实 / 判断 / 行动）。

    顺序很关键：**先看优先级前缀，再看强动作词，最后才看判断词**。
    反过来会把「需结合 MAE 判断」这种判断句误判成行动，也会把
    「必须做：…」这种行动句漏成事实。
    """
    raw = str(text or "")
    if _PRIORITY_PREFIX_RE.match(raw):
        return "action"
    if any(h in raw for h in _ACTION_HINTS):
        return "action"
    if any(h in raw for h in _JUDGMENT_HINTS):
        return "judgment"
    return "fact"


def _priority_of(text: str) -> str:
    raw = str(text or "")
    matched = _PRIORITY_PREFIX_RE.match(raw)
    if matched:
        return matched.group(1)
    if any(h in raw for h in _MUST_HINTS):
        return "必须做"
    if any(h in raw for h in _OPTIONAL_HINTS):
        return "可选做"
    return "建议做"


@dataclass
class ReportItem:
    text: str
    layer: str = "fact"
    priority: str = "建议做"
    #: 支撑该条的事实（可为空），用于「发现 → 行动」的对应关系
    evidence: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "layer": self.layer,
            "priority": self.priority, "evidence": list(self.evidence),
            "source": self.source,
        }


@dataclass
class ReportLayers:
    facts: list[ReportItem] = field(default_factory=list)
    judgments: list[ReportItem] = field(default_factory=list)
    actions: list[ReportItem] = field(default_factory=list)
    #: 结尾必须回答的「下一步该跑什么实验」
    next_action: str = ""

    def all_items(self) -> list[ReportItem]:
        return list(self.facts) + list(self.judgments) + list(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [i.to_dict() for i in self.facts],
            "judgments": [i.to_dict() for i in self.judgments],
            "actions": [i.to_dict() for i in self.actions],
            "next_action": self.next_action,
        }


def build_layers(
    items: list[str] | list[dict[str, Any]],
    *,
    next_action: str = "",
    include_actionless_facts: bool = True,
) -> ReportLayers:
    """把一串平铺的结论/建议组织成三层。

    Args:
        items: 字符串或 ``{text, layer?, priority?, evidence?, source?}``。
        next_action: 报告结尾的「下一步」；为空时从行动层取优先级最高的一条兜底。
        include_actionless_facts: 是否保留没有对应行动的事实（默认保留，
            事实层是判断层的依据，删了会导致结论无法复核）。
    """
    layers = ReportLayers()
    raw_items: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            raw_items.append((text, item))
        else:
            text = str(item or "").strip()
            if text:
                raw_items.append((text, {}))

    # 先去重再分层：重复项分层后会在同一层里出现两次，更显眼
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    for text, meta in raw_items:
        seen.setdefault(normalize(text), (text, meta))
    kept_texts = dedupe([t for t, _ in seen.values()])
    kept_meta = {normalize(t): m for t, m in seen.values()}

    for text in kept_texts:
        meta = kept_meta.get(normalize(text), {})
        layer = str(meta.get("layer") or classify_layer(text))
        priority = str(meta.get("priority") or _priority_of(text))
        evidence = [str(e) for e in (meta.get("evidence") or []) if e]
        entry = ReportItem(
            text=text, layer=layer, priority=priority,
            evidence=evidence, source=str(meta.get("source") or ""),
        )
        if layer == "action":
            layers.actions.append(entry)
        elif layer == "judgment":
            layers.judgments.append(entry)
        else:
            if include_actionless_facts:
                layers.facts.append(entry)

    layers.actions.sort(key=lambda a: (PRIORITIES.index(a.priority) if a.priority in PRIORITIES else 9))
    if not next_action and layers.actions:
        next_action = layers.actions[0].text
    layers.next_action = next_action
    return layers


def render_markdown(layers: ReportLayers, *, title: str = "结论与建议") -> str:
    """把三层渲染成 Markdown（事实 / 判断 / 行动 + 下一步）。"""
    lines: list[str] = [f"### {title}", ""]
    if layers.facts:
        lines.append("**事实（数据说了什么）**")
        lines.extend(f"- {i.text}" for i in layers.facts)
        lines.append("")
    if layers.judgments:
        lines.append("**判断（这意味着什么）**")
        lines.extend(f"- {i.text}" for i in layers.judgments)
        lines.append("")
    if layers.actions:
        lines.append("**行动（下一步做什么）**")
        for priority in PRIORITIES:
            group = [i for i in layers.actions if i.priority == priority]
            if not group:
                continue
            lines.append(f"- **{priority}**")
            lines.extend(f"  - {i.text}" for i in group)
        lines.append("")
    if layers.next_action:
        lines.append(f"**下一步**：{layers.next_action}")
    return "\n".join(lines).rstrip()
