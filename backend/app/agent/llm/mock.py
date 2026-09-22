"""MockLLM：用于测试 / 开发 / CI，不调用真实 API。"""

from __future__ import annotations

from typing import Any

from app.agent.llm.base import LLMMessage, LLMProvider, LLMResponse


class MockLLM(LLMProvider):
    name = "mock"

    def __init__(
        self,
        responses: list[str] | None = None,
        structured_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._chat_queue = list(responses or [])
        self._structured_queue = list(structured_responses or [])
        self.calls: list[list[dict[str, str]]] = []
        self.structured_calls: list[type] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.begin_call()
        self.calls.append([m.to_dict() for m in messages])
        if self._chat_queue:
            content = self._chat_queue.pop(0)
        else:
            last_user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            content = f"[mock] {last_user}"
        response = LLMResponse(content=content, model="mock-1")
        self.record_usage(response.usage)
        return response

    def structured_output(
        self,
        messages: list[LLMMessage],
        schema: type,
        *,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.begin_call()
        self.calls.append([m.to_dict() for m in messages])
        self.structured_calls.append(schema)
        if self._structured_queue:
            item = self._structured_queue.pop(0)
            result = schema.model_validate(item).model_dump()
        else:
            result = schema.model_construct().model_dump() if hasattr(schema, "model_construct") else {}
        self.record_usage({})
        return result

    def assert_chat_count(self, expected: int) -> None:
        if len(self.calls) != expected:
            raise AssertionError(f"期望 {expected} 次 chat 调用，实际 {len(self.calls)} 次")
