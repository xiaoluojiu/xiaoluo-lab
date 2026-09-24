"""「这条回答到底是谁给的」——答案来源（provenance）标记。

为什么需要它
------------
开启远程 API 之后，界面上那条助手消息可能有三种完全不同的来路，而且**长得一模一样**：

1. **远程大模型生成** —— 普通对话走 `_direct_chat` 的 LLM 分支，
   工具结果汇总走 `_compose_answer` 的 LLM 分支；
2. **平台内置规则生成** —— 远程未启用时的确定性应答（`local_chat`）
   或内置兜底汇总（`_compose_answer` 的 fallback）；
3. **远程调用失败后降级** —— 开关开着、凭据也在，但这一次请求没成功，
   于是仍然落到规则兜底。

第 3 条正是「开了 API 也不知道到底是谁回答的」的根源：它和第 1 条在界面上无从区分，
用户只能靠猜。本模块给每次运行打一个**机器可读**的 `answer_source`，随 SSE 的
`completed` 事件与 `GET /runs/{id}` 一起下发，界面据此明示来源。

设计边界
--------
- 只标记**事实**（这段文字由谁产出），不做质量评价、不做打分。
- 判定依据是「运行时这一次到底有没有走通 LLM」，而不是配置项。
  配置说开了、但实际调用抛异常 ⇒ 标 `LLM_ERROR_FALLBACK`，不谎报成远程生成。
- 模型名只作展示信息附带，不参与判定。
"""

from __future__ import annotations

from typing import Any

# --- 来源取值 ---------------------------------------------------------------
# remote_*  = 远程大模型（语言模型）真的产出了这段文字
# platform_*= 平台内置规则产出了这段文字（没有语言模型参与）
# 以外的 `llm_error_fallback` = 本想用远程但没走通，仍然由规则产出

REMOTE_LLM_CHAT = "remote_llm_chat"
REMOTE_LLM_SUMMARY = "remote_llm_summary"
PLATFORM_RULES_CHAT = "platform_rules_chat"
PLATFORM_RULES_NOTICE = "platform_rules_notice"
PLATFORM_RULES_SUMMARY = "platform_rules_summary"
LLM_ERROR_FALLBACK = "llm_error_fallback"
NO_ANSWER = "no_answer"

__all__ = [
    "LLM_ERROR_FALLBACK",
    "NO_ANSWER",
    "PLATFORM_RULES_CHAT",
    "PLATFORM_RULES_NOTICE",
    "PLATFORM_RULES_SUMMARY",
    "REMOTE_LLM_CHAT",
    "REMOTE_LLM_SUMMARY",
    "describe",
    "llm_model_name",
]

# source -> (短标签, 一句话说明, 是否由语言模型生成)
_META: dict[str, tuple[str, str, bool]] = {
    REMOTE_LLM_CHAT: ("远程大模型 · 对话", "这条回复由远程大模型直接生成（普通对话链路，未调用数据工具）。", True),
    REMOTE_LLM_SUMMARY: ("远程大模型 · 汇总", "工具结果由远程大模型汇总成自然语言；数据本身来自平台真实工具。", True),
    PLATFORM_RULES_CHAT: ("平台内置规则 · 对话", "远程大模型未启用，这条回复来自平台内置规则模板，不是语言模型生成的。", False),
    PLATFORM_RULES_NOTICE: ("平台内置规则 · 说明", "内置规则没有覆盖这个问法，回复是一段当前状态说明。", False),
    PLATFORM_RULES_SUMMARY: ("平台内置规则 · 汇总", "远程大模型未启用，工具结果由平台内置规则拼装成回答。", False),
    # ★ 最容易误导用户的一种：开关开着，但这次没走通。必须显式标出来。
    LLM_ERROR_FALLBACK: ("远程调用失败 · 已降级", "本次本应由远程大模型生成，但调用失败，已退回平台内置规则。", False),
    # ★ 两种「没有回答」都落在这里：运行真的失败了；或远程失败后内置规则也没覆盖这个问法
    # （界面上那段文字只是**状态说明**，不是对用户问题的回答）。
    # 两者的共同点是都不能标成「远程生成」或「已降级」——降级必须真的产出了替代回答。
    NO_ANSWER: ("未产出回答", "这次运行没有产出回答（失败 / 被拒绝 / 被取消 / 远程调用失败且内置规则未覆盖）。", False),
}


def llm_model_name() -> str:
    """当前配置的大模型名，**仅用于展示**；缺失时返回空串（不参与任何判定）。"""
    from app.core.config import settings

    return str(getattr(settings, "LLM_MODEL", "") or "")


def describe(source: str) -> dict[str, Any]:
    """把 `answer_source` 展开成界面可直接渲染的结构。

    未知 / 空值统一落到 `NO_ANSWER`，避免前端为每个取值补兜底分支。
    """
    key = source if source in _META else NO_ANSWER
    label, detail, by_llm = _META[key]
    # 模型名是纯展示信息：拿不到就留空，**绝不能因为它把来源标记整条链路带崩**
    # （来源标记属于可观测性，一旦抛异常会让 completed 事件都发不出去）。
    try:
        model = llm_model_name()
    except Exception:  # noqa: BLE001
        model = ""
    return {
        "source": key,
        "label": label,
        "detail": detail,
        "by_llm": by_llm,
        # 只有「确实由远程模型产出」时才带模型名，避免降级场景里出现误导性的模型标识。
        "model": model if (by_llm and model) else "",
    }
