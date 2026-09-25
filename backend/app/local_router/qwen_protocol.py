"""Qwen Router 的**协议层**：提示词口径 + 输出解析 + 折叠成现有契约。

这一层是「Qwen 不参与执行」这条边界的落点：它只做**翻译**，不产生任何动作。
翻译产物是 :class:`~app.local_router.contract.RouterDecision` —— 与词法模型
走的是同一个契约对象，下游 Planner / Tool Registry / Permission 完全感知不到
这次判定是神经网络给的还是 TF-IDF 给的。

提示词必须与训练口径**逐字一致**
--------------------------------
训练语料（`data/training/processed/train.jsonl`）的 system prompt 由
`data/training/scripts/generate_dataset.py:sys_prompt()` 生成。模型是在那个
形态上收敛的，推理时少一句「可用数据集：…」就等于换了一个分布。**这里不是
可以自由发挥的 prompt engineering 区域** —— 修改 :data:`SYSTEM_PROMPT` 或
上下文拼接格式，等于让已训练的 LoRA 失效。

关于「没有第二套 Intent」
------------------------
:class:`QwenMode` 是**模型的输出词表**（和词法模型的 `chat` / `call::<tool>`
标签同一性质），不是新的意图体系。模型输出的 `intent` 字段值域被刻意限定为
:class:`~app.local_router.contract.Intent` 的既有成员，因此解析时零映射成本；
最终写进决策的 intent **一律由 tool 反查得出**（见 :func:`to_router_decision`），
与 :func:`app.local_router.router.route_request` 里词法路径的写法完全一致
—— 工具是硬事实，模型自述的意图只是软判断。

为什么 clarification 会被折成 escalate
--------------------------------------
现有契约里「反问」的语义是**「工具已定但缺必填槽位」**（`ask::<tool>`），
没有工具的反问在 :class:`RouterDecision` 里无法表达：
`decision_to_route()` 在 `tool is None` 时直接返回 `chat`，missing 会被丢掉。
而训练语料里的 clarification 样本（611 条）大多**不带 tool**。

与其在契约里硬塞一个表达不出来的状态，不如诚实地转 escalate(ambiguous)：
平台本来就有比 Router 更完备的澄清通道（`app/agent/clarify.py` 与 Pre-flight
的 `_await_clarification`），交给它们比在这里伪造一个 tool 更正确。
原始 missing 字段会留在 :class:`QwenDecision` 里进 trace，供离线分析。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.local_router.contract import (
    EscalationReason,
    Intent,
    RouterDecision,
    RouterRequest,
    intent_of_tool,
    known_params,
    tool_label_space,
)

__all__ = [
    "QWEN_MODES",
    "QwenDecision",
    "QwenMode",
    "SYSTEM_PROMPT",
    "build_messages",
    "parse_decision",
    "to_router_decision",
]

#: 与训练语料**逐字一致**的 system prompt。改动前请先重训。
SYSTEM_PROMPT = (
    "你是小洛实验室（Xiaoluo Lab）的本地数据分析 Agent Router，负责把用户的自然语言请求"
    "转化为结构化决策：判定意图（intent）、选择工具（tool）或规划多步工作流（steps）、"
    "在缺少必要参数时反问（clarification）、在平台能力之外或过于复杂时升级"
    "（escalate/unsupported）。你只输出一段 JSON 决策，不要输出任何解释性自然语言。"
    "工具与参数以平台注册表为准，禁止编造不存在的工具或参数。"
)


class QwenMode(str, Enum):
    """模型输出的 6 个 mode。**这是模型的标签词表，不是平台的 Intent。**

    与 :class:`~app.local_router.contract.Intent` 的关系：Intent 回答「属于哪个能力域」
    （dataset / eda / ml / …），mode 回答「这一步该怎么走」（直接执行 / 编排 / 闲聊 /
    反问 / 升级）。两者正交，且 Intent 最终由 tool 反查得出，不采信模型自述。
    """

    CHAT = "chat"
    COMMAND = "command"
    WORKFLOW = "workflow"
    CLARIFICATION = "clarification"
    ESCALATE = "escalate"
    UNSUPPORTED = "unsupported"


QWEN_MODES: frozenset[str] = frozenset(mode.value for mode in QwenMode)

#: 训练语料里的 `escalate_reason` 比现有契约多一类 `complex_task`（43 条）。
#: 现有五类原因是给**结构信号**用的（连词密度、外部词表…），而 complex_task 说的是
#: 「这件事太复杂」，语义上最接近 multi_step。这里做收窄映射而不是扩展枚举 ——
#: 平台侧只有一个「升级」动作，多一类 reason 只会让统计口径更难对齐。
#: 原始值保留在 `QwenDecision.raw_reason` 里，离线分析仍可区分。
_REASON_MAP: dict[str, EscalationReason] = {
    "multi_step": EscalationReason.MULTI_STEP,
    "complex_task": EscalationReason.MULTI_STEP,
    "ambiguous": EscalationReason.AMBIGUOUS,
    "out_of_scope": EscalationReason.OUT_OF_SCOPE,
    "param_dependency": EscalationReason.PARAM_DEPENDENCY,
    "conflict": EscalationReason.CONFLICT,
}


@dataclass
class QwenDecision:
    """模型输出的强类型镜像。

    刻意**不是** `RouterDecision`：它保留模型原始语义（含 `steps` 与 `raw_reason`），
    这些字段在契约里无处安放，但对离线分析（「模型到底想干什么」）很有价值。
    折叠成契约对象的动作只在 :func:`to_router_decision` 里发生一次。
    """

    mode: QwenMode
    confidence: float = 0.0
    intent: Intent | None = None
    tool: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    escalate_reason: EscalationReason | None = None
    #: 模型原文里的 reason（可能是 `complex_task` 这类平台没有的值）。
    raw_reason: str | None = None
    #: 模型自述的 intent 原文，用于离线评估「模型意图判断」这项独立能力。
    raw_intent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 4),
            "intent": self.intent.value if self.intent else None,
            "tool": self.tool,
            "params": dict(self.params),
            "steps": list(self.steps),
            "missing": list(self.missing),
            "escalate_reason": self.escalate_reason.value if self.escalate_reason else None,
            "raw_reason": self.raw_reason,
            "raw_intent": self.raw_intent,
        }


# ---------------------------------------------------------------------------
# 提示词构建
# ---------------------------------------------------------------------------


def _dataset_items(request: Any) -> list[tuple[str, int]]:
    """从 RouterRequest 取出 `(name, id)` 列表；缺字段时退化成只有 id。"""
    raw = list(getattr(request, "available_datasets", None) or [])
    items: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            items.append((str(item.get("name") or ""), int(item["id"])))
        except (KeyError, TypeError, ValueError):
            continue
    return items


def build_messages(request: Any, *, max_datasets: int = 30) -> list[dict[str, str]]:
    """按训练口径构造 `[system, user]`。

    `max_datasets` 截断数据集清单：训练语料里是 50 个，但线上可能有成百上千，
    全量注入会把 prefill 拖到几百毫秒并稀释注意力。截断是**必要的工程妥协**，
    已知会造成轻微分布漂移，靠 shadow trace 观察。
    """
    utterance = str(getattr(request, "utterance", "") or "")
    bound_id = getattr(request, "bound_dataset_id", None)
    bound_name = getattr(request, "bound_dataset_name", None)
    items = _dataset_items(request)

    system = SYSTEM_PROMPT
    if bound_id is not None:
        # 训练语料里这一句只在「有绑定」时出现；未绑定时**不写**（而不是写「未绑定」）。
        # 词法模型那边需要双向编码，是因为它没有别的信号源；神经模型有 `available_datasets`
        # 可以做区分，照抄双向编码反而会让模型见到训练时没见过的句式。
        name = bound_name or next((n for n, i in items if i == bound_id), "")
        system += f"\n当前会话上下文：已绑定数据集 {name}（dataset_id={bound_id}）。"
    if items:
        shown = items[: max(0, int(max_datasets))]
        system += "\n可用数据集：" + "、".join(f"{name}(id={did})" for name, did in shown) + "。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": utterance},
    ]


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从生成文本里抠出第一段 JSON 对象。

    用 `raw_decode` 而不是「找首尾花括号」：模型有时会先说一句废话再输出 JSON，
    有时会在 JSON 后面继续编，`raw_decode` 只认**第一个合法对象**，两头都稳。
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _as_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    # 模型偶尔会吐 0~100 而不是 0~1；统一收进 [0, 1]，异常值按 0（不可信）处理。
    if score > 1.0:
        score = score / 100.0 if score <= 100.0 else 1.0
    return max(0.0, min(1.0, score))


def parse_decision(raw: str) -> QwenDecision | None:
    """把模型输出解析成 :class:`QwenDecision`；任何不合法都返回 None（= 解析失败）。

    「解析失败」与「解析出一个 escalate」是两件事：前者触发降级（换词法模型），
    后者是模型的**有效判定**（走升级）。区分它们的依据就是这里的返回值。
    """
    payload = _extract_json_object(raw or "")
    if payload is None:
        return None

    raw_mode = str(payload.get("mode") or "").strip().lower()
    if raw_mode not in QWEN_MODES:
        return None
    mode = QwenMode(raw_mode)

    raw_intent = payload.get("intent")
    intent: Intent | None = None
    if isinstance(raw_intent, str):
        try:
            intent = Intent(raw_intent.strip().lower())
        except ValueError:
            intent = None

    raw_reason = payload.get("escalate_reason")
    reason_text = str(raw_reason).strip().lower() if isinstance(raw_reason, str) else None
    reason = _REASON_MAP.get(reason_text or "")

    params = payload.get("params")
    steps = payload.get("steps")
    missing = payload.get("missing")

    return QwenDecision(
        mode=mode,
        confidence=_as_confidence(payload.get("confidence")),
        intent=intent,
        tool=payload.get("tool") if isinstance(payload.get("tool"), str) else None,
        params=params if isinstance(params, dict) else {},
        steps=[s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else [],
        missing=[str(m) for m in missing] if isinstance(missing, list) else [],
        escalate_reason=reason,
        raw_reason=reason_text,
        raw_intent=str(raw_intent) if isinstance(raw_intent, str) else None,
    )


# ---------------------------------------------------------------------------
# 折叠成现有契约
# ---------------------------------------------------------------------------


def _known_tool(name: str | None, label_space: frozenset[str]) -> str | None:
    """工具必须在**当前**注册表里。

    这是「模型不许发明平台没有的能力」的硬闸门：LoRA 是在 35 个工具上训的，
    平台后来新增/改名都会让它吐出过时名字 —— 与其执行一个不存在的东西，
    不如返回 None 让整次解析降级。
    """
    if isinstance(name, str) and name in label_space:
        return name
    return None


def _safe_params(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """只保留 schema 声明过的参数键，丢掉模型编造的。"""
    declared = known_params(tool)
    if not declared:
        return {}
    return {k: v for k, v in params.items() if k in declared}


def _apply_dataset_guard(
    params: dict[str, Any], request: Any, allowed_ids: set[int]
) -> dict[str, Any]:
    """**dataset_id 防幻觉校正** —— 这一条必须做，不能信模型。

    训练语料里 `params.dataset_id` 出现了 2235 次，模型是照着「可用数据集清单」
    学出来的映射。线上数据集一变，它会非常自信地吐出训练时记住的 id
    （例如数据集 34，而你的库里根本没有 34）—— 一个不存在的 id 会一路走到
    SQL 查询才炸，那是**最难排查**的一类故障。

    规则：
    * 会话已绑定 ⇒ 无条件用绑定值覆盖（用户的当前上下文优先于模型的猜测）；
    * 未绑定但模型给了 id ⇒ 只在该 id 确实存在于可见数据集里时才采纳，否则删除，
      让下游 Planner / 反问规则去补（它们有真实数据可查）。
    """
    if "dataset_id" not in params:
        return params
    bound = getattr(request, "bound_dataset_id", None)
    if bound is not None:
        params["dataset_id"] = bound
        return params
    value = params.get("dataset_id")
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        params.pop("dataset_id", None)
        return params
    # allowed_ids 为空 = 调用方没提供清单 ⇒ 无从校验，同样不采信。
    if allowed_ids and candidate in allowed_ids:
        params["dataset_id"] = candidate
    else:
        params.pop("dataset_id", None)
    return params


def to_router_decision(
    decision: QwenDecision,
    request: Any = None,
) -> RouterDecision | None:
    """把模型判定折叠成 :class:`RouterDecision`。

    返回 None 表示**这次判定不可用**（工具不在注册表等），调用方据此降级到词法模型。
    intent 一律由 tool 反查（与词法路径同写法），模型自述的 intent 只进 trace。
    """
    label_space = frozenset(tool_label_space())
    allowed_ids = {did for _name, did in _dataset_items(request)}
    confidence = decision.confidence

    # --- 闲聊：不需要任何工具 ---
    if decision.mode == QwenMode.CHAT:
        return RouterDecision(intent=Intent.CHAT, tool=None, confidence=confidence)

    # --- 超出平台能力 / 主动升级 ---
    if decision.mode == QwenMode.UNSUPPORTED:
        return RouterDecision(
            escalate=True,
            escalate_reason=decision.escalate_reason or EscalationReason.OUT_OF_SCOPE,
            confidence=confidence,
        )
    if decision.mode == QwenMode.ESCALATE:
        return RouterDecision(
            escalate=True,
            escalate_reason=decision.escalate_reason or EscalationReason.AMBIGUOUS,
            confidence=confidence,
        )

    # --- 反问：见模块 docstring「为什么 clarification 会被折成 escalate」 ---
    if decision.mode == QwenMode.CLARIFICATION:
        tool = _known_tool(decision.tool, label_space)
        if tool is not None:
            # 模型难得地给了工具 ⇒ 契约能表达，按「工具已定但缺槽位」处理。
            return RouterDecision(
                intent=intent_of_tool(tool) or Intent.CHAT,
                tool=tool,
                params=_apply_dataset_guard(
                    _safe_params(tool, decision.params), request, allowed_ids
                ),
                missing=[m for m in decision.missing if m],
                confidence=confidence,
            )
        return RouterDecision(
            escalate=True,
            escalate_reason=EscalationReason.AMBIGUOUS,
            confidence=confidence,
        )

    # --- 多步编排：取首步工具 ---
    # 与 runtime 的「本地识别首步 + signals 驱动后续」是同一套机制
    # （`_local_direct_plan` → `_run_dynamic(first_tool=...)`）。模型给的 steps 完整
    # 保留在 QwenDecision 里；若首步不合法才退回通用的 workflow 构建工具。
    if decision.mode == QwenMode.WORKFLOW:
        first: str | None = None
        for step in decision.steps:
            candidate = _known_tool(step.get("tool") if isinstance(step.get("tool"), str) else None,
                                    label_space)
            if candidate is not None:
                first = candidate
                break
        tool = first or _known_tool("workflow.build_and_run", label_space)
        if tool is None:
            return None
        arguments = decision.steps[0].get("arguments") if decision.steps else None
        params = arguments if isinstance(arguments, dict) else {}
        return RouterDecision(
            intent=intent_of_tool(tool) or Intent.WORKFLOW,
            tool=tool,
            params=_apply_dataset_guard(_safe_params(tool, params), request, allowed_ids),
            confidence=confidence,
        )

    # --- 单步工具调用 ---
    tool = _known_tool(decision.tool, label_space)
    if tool is None:
        return None
    return RouterDecision(
        intent=intent_of_tool(tool) or Intent.CHAT,
        tool=tool,
        params=_apply_dataset_guard(
            _safe_params(tool, decision.params), request, allowed_ids
        ),
        confidence=confidence,
    )


def request_of(value: Any) -> RouterRequest:
    """把 dict / RouterRequest 归一成 RouterRequest（供调用方复用）。"""
    if isinstance(value, RouterRequest):
        return value
    if not isinstance(value, dict):
        return RouterRequest(utterance=str(value))
    allowed = set(RouterRequest.model_fields)
    return RouterRequest(**{k: v for k, v in value.items() if k in allowed})
