"""钉住「平台自带能力」的闲聊应答（`app/agent/local_chat.py`）。

为什么值得单独钉
----------------
这条链路只在**关掉远程大模型**时才走到，而它恰恰是最容易被反向改坏的地方：

| 不变式 | 为什么 |
| --- | --- |
| **远程开启时行为完全不变** | `_direct_chat` 的 LLM 分支一个字都不能动，否则「有 LLM vs 无 LLM」对照实验失去意义 |
| 拿不准的句子必须返回 `None` | 一旦对任意句子都开始回话，就从「受限应答」退化成劣质聊天机器人 |
| 能力清单必须**实时读注册表** | 硬编码工具名会在平台增删工具后静默过期（与 Router 标签空间同一类风险） |
| 停用 / 未配置必须**分开表述** | 合并成「尚未配置」会误导用户去重填 Key（本次修复的起因） |
| 任何意图都不报错 | 闲聊兜底自身抛异常会让 `_direct_chat` 崩掉，比不回话严重得多 |

全部用例毫秒级、不启服务、不联网。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent import local_chat as L
from app.agent.local_chat import local_reply, no_llm_notice, tool_overview
from app.agent.runtime.models import AgentRun, AgentSession, RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.core.config import settings
from app.local_router import contract as C


# ---------------------------------------------------------------------------
# 一、确定性意图
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", ["你好", "您好", "嗨", "hello", "hi", "早上好", "在吗"])
def test_greeting_gets_local_reply(utterance):
    reply = local_reply(utterance)
    assert reply, f"{utterance!r} 应命中问候应答"
    assert "平台自带能力" in reply
    assert "尚未配置" not in reply, "问候语不该收到「未配置」文案"
    assert "看看这批数据的分布" in reply, "应答应给出可执行的下一步示例"


@pytest.mark.parametrize("utterance", ["谢谢", "感谢", "辛苦了"])
def test_thanks(utterance):
    assert "不客气" in (local_reply(utterance) or "")


@pytest.mark.parametrize("utterance", ["再见", "拜拜"])
def test_farewell(utterance):
    assert "再见" in (local_reply(utterance) or "")


@pytest.mark.parametrize("utterance", ["你是谁", "你叫什么", "介绍一下你自己"])
def test_identity_beats_generic_greeting(utterance):
    """「你是谁」同时在 `AgentRuntime.GREETINGS` 里，但自述比一句「你好」有用得多。"""
    reply = local_reply(utterance) or ""
    assert "小洛实验室" in reply
    assert "AI 助手" in reply
    assert "工具" in reply, "自述应带上真实能力清单，而不是空话"


def test_capability_reply_lists_real_registry_count():
    reply = local_reply("你能做什么") or ""
    total, groups = tool_overview()
    assert total == len(C.tool_label_space()) > 0
    assert f"{total} 个数据工具" in reply
    assert len(groups) >= 1
    for label, names in groups:
        assert label in reply
        assert names[0] in reply, f"{label} 域的首个工具名应出现在应答里"
    assert "实时读自工具注册表" in reply


def test_usage_reply_mentions_binding_dataset():
    reply = local_reply("这个平台怎么用") or ""
    assert "绑" in reply and "数据集" in reply
    assert "流程" in reply and "学习中心" in reply


@pytest.mark.parametrize("utterance", ["今天天气不错", "讲个笑话", "帮我写首诗", ""])
def test_unknown_intent_returns_none(utterance):
    """**最关键的一条**：拿不准就交回上层，绝不硬聊。"""
    assert local_reply(utterance) is None


# ---------------------------------------------------------------------------
# 二、能力清单必须动态（不许硬编码工具名）
# ---------------------------------------------------------------------------


def test_tool_overview_follows_registry(monkeypatch):
    monkeypatch.setattr(C, "tool_label_space", lambda: ["fake.alpha", "fake.beta"])
    monkeypatch.setattr(C, "intent_of_tool", lambda name: None)
    total, groups = tool_overview()
    assert total == 2
    assert groups == [(L.DOMAIN_LABELS["other"], ["fake.alpha", "fake.beta"])]

    reply = L._capability_reply()
    assert "2 个数据工具" in reply
    assert "fake.alpha" in reply
    assert "eda.distribution" not in reply, "旧工具名不该残留 —— 说明清单是动态生成的"


def test_tool_overview_degrades_without_registry(monkeypatch):
    def _boom():
        raise RuntimeError("注册表炸了")

    monkeypatch.setattr(C, "tool_label_space", _boom)
    total, groups = tool_overview()
    assert (total, groups) == (0, [])
    reply = local_reply("你能做什么") or ""
    assert "暂时列不出能力清单" in reply, "注册表不可用时应诚实说明，而不是抛异常或给空表"


def test_capability_examples_are_capped(monkeypatch):
    """工具多了不能把消息撑爆：每域只举例，其余折叠成「等 N 个」。"""
    monkeypatch.setattr(C, "tool_label_space", lambda: [f"eda.t{i}" for i in range(9)])

    from app.local_router.contract import Intent

    monkeypatch.setattr(C, "intent_of_tool", lambda name: Intent.EDA)
    reply = L._capability_reply()
    assert "等 9 个" in reply
    assert sum(reply.count(f"eda.t{i}") for i in range(9)) == L._EXAMPLES_PER_DOMAIN


# ---------------------------------------------------------------------------
# 三、说明性文案：停用 / 未配置 必须区分
# ---------------------------------------------------------------------------


def test_notice_when_key_present_but_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-test", raising=False)
    monkeypatch.setattr(settings, "LLM_REMOTE_ENABLED", False, raising=False)
    notice = no_llm_notice()
    assert "已停用" in notice
    assert "尚未配置" not in notice, "凭据还在，说「尚未配置」是事实错误"
    assert "不用重填" in notice, "必须告诉用户不需要重新填 Key"
    assert "数据类请求不受影响" in notice


def test_notice_when_key_absent(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "LLM_REMOTE_ENABLED", True, raising=False)
    notice = no_llm_notice()
    assert "尚未配置" in notice
    assert "API Key" in notice


def test_notice_when_key_present_and_enabled(monkeypatch):
    """开关开着、Key 也有却仍走到这里 ⇒ 如实说明是「这次没取到」，不要赖成配置问题。"""
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-test", raising=False)
    monkeypatch.setattr(settings, "LLM_REMOTE_ENABLED", True, raising=False)
    notice = no_llm_notice()
    assert "没能取到" in notice
    assert "测试连接" in notice


# ---------------------------------------------------------------------------
# 四、接线：只有 llm 为 None 时才轮到它
# ---------------------------------------------------------------------------


def _run(text: str = "你好") -> AgentRun:
    return AgentRun(id="r-t", session_id="s-t", user_id="u-t", user_request=text)


def _session() -> AgentSession:
    return AgentSession(id="s-t", user_id="u-t")


def _stub(**attrs) -> SimpleNamespace:
    """`_direct_chat` 只用到 `self.llm` / `self._emit` / `self._emit_usage`，不必构造整个 AgentRuntime。

    `_emit_usage` 是后来加的「Token 账本快照」钩子；缺了它会让这里所有用例
    AttributeError —— 属于桩件漏字段，不是产品代码回归。
    """
    base = {"llm": None, "_emit": lambda *args, **kwargs: None, "_emit_usage": lambda *args, **kwargs: None}
    base.update(attrs)  # 允许用例覆盖 llm（SimpleNamespace(llm=None, **{"llm": llm}) 会重复关键字）
    return SimpleNamespace(**base)


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list] = []

    def chat(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


def test_direct_chat_without_llm_uses_local_reply():
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(), run, session, None)
    assert run.status == RunStatus.COMPLETED
    assert "平台自带能力" in run.final_answer
    assert "尚未配置" not in run.final_answer
    assert session.history[-1]["content"] == run.final_answer


def test_direct_chat_without_llm_falls_back_to_notice():
    """拿不准的句子 ⇒ 说明性文案。

    断言**与 `no_llm_notice()` 逐字一致**，而不是硬写某个分支的片段：本机 `.env`
    里 `LLM_API_KEY` 是否配置会决定文案走哪个分支，写死片段会让用例随环境红绿。
    """
    run, session = _run("今天天气不错"), _session()
    AgentRuntime._direct_chat(_stub(), run, session, None)
    assert run.final_answer == no_llm_notice()
    assert "数据类请求" in run.final_answer, "三种状态都要告诉用户：数据链路还能用"
    assert "平台自带能力" not in run.final_answer, "拿不准时不该给出像闲聊应答的话"


def test_direct_chat_greeting_survives_remote_disabled(monkeypatch):
    """**本次缺陷的回归用例**：设置页关掉远程大模型后说「你好」。

    修复前回「当前尚未配置可用的大模型。」—— 而凭据其实原样保留着，
    用户会误以为配置丢了。修复后应命中平台自带能力的问候应答。
    """
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-test", raising=False)
    monkeypatch.setattr(settings, "LLM_REMOTE_ENABLED", False, raising=False)
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(), run, session, None)
    assert "尚未配置" not in run.final_answer
    assert "平台自带能力" in run.final_answer
    assert "看看这批数据的分布" in run.final_answer, "应给出可执行的下一步"


def test_llm_path_is_untouched(monkeypatch):
    """**远程大模型开启时行为完全不变** —— 本地应答绝不能插手。

    用「一旦被调用就断言失败」的方式钉住，而不是只看输出：这样即使将来有人把
    `local_reply` 挪到分支之外、又被 LLM 的正常输出掩盖，也会立刻失败。
    """
    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("llm 可用时不应调用本地应答")

    monkeypatch.setattr(L, "local_reply", _must_not_be_called)
    monkeypatch.setattr(L, "no_llm_notice", _must_not_be_called)

    llm = _FakeLLM("你好！我是小洛实验室的 AI 助手。")
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(llm=llm), run, session, None)

    assert len(llm.calls) == 1, "应走 LLM 分支且只调一次"
    assert run.final_answer == "你好！我是小洛实验室的 AI 助手。"


def test_llm_path_falls_back_to_llm_default_when_content_empty():
    """LLM 返回空内容时的既有兜底也不能被本地应答顶替（保持原行为）。"""
    llm = _FakeLLM("")
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(llm=llm), run, session, None)
    assert run.final_answer == "我在。有什么可以帮你？"
