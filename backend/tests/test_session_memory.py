"""会话记忆：AI 说过的话要留下来，追问时要带着上下文。

★ 事故一（历史丢失）：在 AI 实验室聊完，去别的模块逛一圈回来，
助手的回复全没了，只剩用户自己的提问。
根因：``session.history`` 是前端重建对话的**唯一数据源**，
而引擎从头到尾只往里写过用户消息 —— 助手那一半从来没写。

★ 事故二（割裂感）：追问「你推荐哪个目标列」时，模型回答
「没有看到你的数据字段和业务目标」。那些字段上一轮就摆在屏幕上，
只是对话路径压根没把它们带进 prompt。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.engine import AgentEngine
from app.agent.models import RunStatus
from app.agent.store import AgentStore
from app.agent.llm import LLMMessage


# ---------------------------------------------------------------- 助手消息入历史


def test_assistant_reply_is_recorded(engine: AgentEngine, store: AgentStore, bound_session):
    """★ 跑完一轮之后，会话历史里必须有助手这一条。"""
    run = store.create_run(bound_session.id, "有哪些数据集")
    engine.run(run, bound_session)

    session = store.get_session(bound_session.id)
    roles = [item["role"] for item in session.history]
    assert roles.count("assistant") >= 1
    assert session.history[-1]["role"] == "assistant"
    assert session.history[-1]["content"] == run.final_answer.strip()


def test_history_survives_reloading_the_session(engine: AgentEngine, store: AgentStore, bound_session):
    """前端「回到 AI 实验室」= 重新拉 session；历史必须还在。"""
    run = store.create_run(bound_session.id, "有哪些数据集")
    engine.run(run, bound_session)

    reloaded = store.get_session(bound_session.id)
    assert any(item["role"] == "assistant" for item in reloaded.history)


def test_failed_run_records_why(engine: AgentEngine, store: AgentStore, session):
    """失败也要留一句话：用户不该只看到自己那句孤零零的提问。"""
    run = store.create_run(session.id, "看看数据质量")
    run.error = "缺少数据集，无法执行"
    engine._remember_answer(run, RunStatus.FAILED)

    reloaded = store.get_session(session.id)
    assert reloaded.history[-1] == {"role": "assistant", "content": "缺少数据集，无法执行"}


def test_empty_answer_is_not_recorded(engine: AgentEngine, store: AgentStore, session):
    """没话说就别写一条空的助手消息进去。"""
    before = len(store.get_session(session.id).history)
    run = store.create_run(session.id, "你好")
    run.final_answer = "   "
    engine._remember_answer(run, RunStatus.COMPLETED)
    assert len(store.get_session(session.id).history) == before


# ---------------------------------------------------------------- 对话上下文


def test_chat_history_drops_the_current_request(engine: AgentEngine, store: AgentStore, session):
    """本轮请求已在历史末尾，再带一次用户的话就会出现两遍。"""
    run = store.create_run(session.id, "你好")
    store.append_history(session.id, "user", "你好")
    session = store.get_session(session.id)

    assert [item["content"] for item in engine._chat_history(session, run)] == []


def test_chat_history_keeps_earlier_turns(engine: AgentEngine, store: AgentStore, session):
    store.append_history(session.id, "user", "看看质量")
    store.append_history(session.id, "assistant", "缺失值 12 处")
    store.append_history(session.id, "user", "你好")
    run = store.create_run(session.id, "你好")
    session = store.get_session(session.id)

    history = engine._chat_history(session, run)
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[-1]["content"] == "缺失值 12 处"


def test_chat_context_carries_the_previous_conclusion(engine: AgentEngine, store: AgentStore, bound_session):
    """★ 上一轮跑出来的结论必须进上下文，否则追问只能得到「没看到你的数据」。"""
    first = store.create_run(bound_session.id, "有哪些数据集")
    engine.run(first, bound_session)

    second = store.create_run(bound_session.id, "你推荐哪个目标列")
    session = store.get_session(bound_session.id)
    context = engine._chat_context(session, second)

    assert "上一次分析的结论" in context
    # 结论文本进了上下文（换行被压成空格，因此按片段比对）
    assert "列出可用数据集" in context


class _CapturingProvider:
    """把模型真正看到的 messages 记录下来。"""

    def __init__(self) -> None:
        self.messages: list[LLMMessage] = []
        self.model = "fake-model"

    def chat(self, messages, *args: Any, **kwargs: Any):
        self.messages = list(messages)

        class _Resp:
            content = "推荐 fare_amount"
            model = "fake-model"
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

        return _Resp()


@pytest.fixture()
def chat_engine(store: AgentStore) -> tuple[AgentEngine, _CapturingProvider]:
    provider = _CapturingProvider()
    return AgentEngine(store, llm=provider), provider


def test_chat_prompt_includes_context_and_history(
    chat_engine: tuple[AgentEngine, _CapturingProvider], store: AgentStore, bound_session
):
    """★ 端到端：追问时 prompt 里必须同时有会话上下文与历史对话。"""
    engine, provider = chat_engine
    first = store.create_run(bound_session.id, "有哪些数据集")
    engine.run(first, bound_session)

    second = store.create_run(bound_session.id, "你推荐哪个目标列")
    engine.run(second, bound_session)

    seen = "\n".join(m.content for m in provider.messages)
    assert "【会话上下文】" in seen
    assert "上一次分析的结论" in seen
    # 历史里上一轮的助手回复也要在
    assert any(m.role == "assistant" for m in provider.messages)
    assert second.status is RunStatus.COMPLETED
