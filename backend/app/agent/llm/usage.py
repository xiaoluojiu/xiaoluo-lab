"""Provider-agnostic LLM usage 统计。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False
    provider: str = ""
    model: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None, *, provider: str = "", model: str = "") -> "LLMUsage":
        raw = raw or {}
        input_tokens = int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0)
        output_tokens = int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0)
        total = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
        estimated = not bool(raw.get("total_tokens") or raw.get("prompt_tokens") or raw.get("input_tokens"))
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cached_input_tokens=int(raw.get("cached_tokens", raw.get("cache_read_input_tokens", 0)) or 0),
            reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
            estimated=estimated,
            provider=provider,
            model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
