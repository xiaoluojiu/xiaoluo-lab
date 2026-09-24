"""LLM Provider 模块：厂商无关的对话与结构化输出接口。

包含：
- ``LLMProvider``（抽象基类）与 ``LLMMessage`` / ``LLMResponse``；
- ``OpenAICompatibleProvider``（远程 OpenAI-compatible 网关）；
- ``LocalReasoningProvider``（本地生成式推理模型接口，Phase 3 预留）；
- ``MockLLM``（测试替身）。
"""

from app.agent.llm.base import (
    LLMException,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    fallback_allowed,
    is_fallbackable_error,
)
from app.agent.llm.local import LocalReasoningProvider, FakeLocalReasoningProvider
from app.agent.llm.mock import MockLLM

__all__ = [
    "LLMException",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "fallback_allowed",
    "is_fallbackable_error",
    "LocalReasoningProvider",
    "FakeLocalReasoningProvider",
    "MockLLM",
]
