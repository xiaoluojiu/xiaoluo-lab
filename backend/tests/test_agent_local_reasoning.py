"""Phase 3 本地推理模型接口（LocalReasoningProvider）单元测试。

钉住三条不变式，全部离线、毫秒级、不加载任何真实权重：

1. **接口即真实**：本地模型不可用时，``chat`` 显式抛 ``LLMException(fallbackable=True)``，
   绝不静默返回一个空内容或伪造置信度 —— 上层据此决定「升级远程 / 降级规则」。
2. **能力声明诚实**：本地推理 Provider 只声明 ``chat`` / ``reasoning``，不声明
   ``tool_calling`` / ``structured_output`` 等能力（None = 未知），Agent 不应假设可用。
3. **测试替身可用**：``FakeLocalReasoningProvider`` 置 ``is_available()`` 为 True 并
   回放预设响应，用于在「本地推理可用」前提下验证决策链路。
"""

from __future__ import annotations

import pytest

from app.agent.llm.base import LLMException, LLMMessage, is_fallbackable_error
from app.agent.llm.local import (
    FakeLocalReasoningProvider,
    LocalReasoningProvider,
)


def test_unavailable_provider_is_never_silently_available() -> None:
    """基类 LocalReasoningProvider 尚未接入后端，is_available 恒为 False。"""
    p = LocalReasoningProvider()
    assert p.is_available() is False


def test_unavailable_provider_chat_raises_fallbackable() -> None:
    """不可用时不静默吞错：抛可降级 LLMException。"""
    p = LocalReasoningProvider(model_name="qwen")
    with pytest.raises(LLMException) as exc_info:
        p.chat([LLMMessage(role="user", content="看看这批数据分布")])
    assert exc_info.value.fallbackable is True
    # 这是「本地这一侧不可用」，属于可降级错误（与远程 LLMException 同口径）。
    assert is_fallbackable_error(exc_info.value) is True


def test_unavailable_chat_does_not_count_as_a_call() -> None:
    """不可用即抛错，不应累计 usage（没有发生真实推理）。"""
    p = LocalReasoningProvider()
    with pytest.raises(LLMException):
        p.chat([LLMMessage(role="user", content="x")])
    snap = p.usage_snapshot()
    assert snap["calls"] == 0
    assert snap["total_tokens"] == 0


def test_capabilities_declare_only_chat_and_reasoning() -> None:
    """能力声明诚实：只声明 chat/reasoning，不声称支持工具调用/结构化输出。"""
    p = LocalReasoningProvider()
    cap = p.capabilities
    assert cap.chat is True
    assert cap.reasoning is True
    assert cap.tool_calling is not True
    assert cap.structured_output is not True
    assert cap.json_mode is not True


def test_fake_provider_is_available_and_replays() -> None:
    """测试替身：is_available=True，回放预设响应。"""
    p = FakeLocalReasoningProvider(responses=["本地推理：应检查数据质量"])
    assert p.is_available() is True
    resp = p.chat([LLMMessage(role="user", content="检查一下数据质量")])
    assert resp.content == "本地推理：应检查数据质量"
    assert resp.model == "qwen-fake"
    assert len(p.calls) == 1


def test_fake_provider_records_usage() -> None:
    """测试替身同样走 begin_call / record_usage，usage 可观测。"""
    p = FakeLocalReasoningProvider(responses=["ok"])
    p.chat([LLMMessage(role="user", content="hi")])
    assert p.usage_snapshot()["calls"] == 1


def test_fake_provider_fallback_content_when_queue_empty() -> None:
    """响应队列耗尽时回退到 [local-reasoning] + 最后一条用户消息，而非空串。"""
    p = FakeLocalReasoningProvider()
    resp = p.chat([LLMMessage(role="user", content="你好")])
    assert "你好" in resp.content
