"""LLM 模型能力描述。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

CapabilityValue = bool | None

@dataclass(frozen=True)
class LLMCapabilities:
    """以模型能力而非厂商名称驱动 Agent 策略。None 表示未知。"""
    chat: CapabilityValue = True
    streaming: CapabilityValue = None
    tool_calling: CapabilityValue = None
    structured_output: CapabilityValue = None
    json_mode: CapabilityValue = None
    vision: CapabilityValue = None
    reasoning: CapabilityValue = None
    context_window: int | None = None
    max_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def supports(self, capability: str) -> bool:
        return getattr(self, capability, None) is True


DEFAULT_OPENAI_COMPATIBLE_CAPABILITIES = LLMCapabilities()
