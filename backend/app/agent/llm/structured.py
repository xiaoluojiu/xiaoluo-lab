"""Prompt 120：结构化输出解析。

处理 JSON / Pydantic / Schema validation / Retry。
非法输出（无法解析或校验失败）抛 StructuredOutputError，
绝不进入 Agent 执行环节。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent.llm.base import LLMException, LLMMessage, LLMProvider

# 剥离 markdown 代码围栏（```json ... ```）
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class StructuredOutputError(LLMException):
    """LLM 输出无法解析或未通过 Schema 校验。"""

    http_status = 502
    default_code = "LLM_STRUCTURED_OUTPUT_ERROR"
    default_message = "LLM structured output invalid"


def extract_json(text: str) -> Any:
    """从文本中提取第一个 JSON 值（对象或数组）。

    支持直接 JSON、markdown 围栏包裹、以及前后夹杂说明文字的情况。
    """
    if not text or not text.strip():
        raise StructuredOutputError("LLM 输出为空")
    candidates: list[str] = [text.strip()]
    candidates.extend(match.group(1) for match in _FENCE_RE.finditer(text))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 扫描式解析：从第一个 { 或 [ 开始尝试 raw_decode（容忍前后夹杂文字）
        for start_char in ("{", "["):
            start = candidate.find(start_char)
            while start != -1:
                try:
                    value, _end = decoder.raw_decode(candidate[start:])
                    return value
                except json.JSONDecodeError:
                    start = candidate.find(start_char, start + 1)
    raise StructuredOutputError("LLM 输出中未找到合法 JSON")


def parse_structured(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """解析并校验结构化输出；失败抛 StructuredOutputError。"""
    data = extract_json(text)
    try:
        model = schema.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            "LLM 输出未通过 Schema 校验",
            details={"errors": exc.errors(include_url=False)[:10]},
        ) from exc
    return model.model_dump(mode="json")


def structured_output_with_retry(
    provider: LLMProvider,
    messages: list[LLMMessage],
    schema: type[BaseModel],
    *,
    max_attempts: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """调用 LLM 并解析结构化输出；失败时把错误信息回传给模型重试。"""
    convo = list(messages)
    last_error: StructuredOutputError | None = None
    for _attempt in range(1, max_attempts + 1):
        response = provider.chat(convo, **kwargs)
        try:
            return parse_structured(response.content, schema)
        except StructuredOutputError as exc:
            last_error = exc
            convo = convo + [
                LLMMessage(role="assistant", content=response.content),
                LLMMessage(
                    role="user",
                    content=(
                        "你的输出不符合要求，请只输出一个符合规范的 JSON 对象，"
                        f"不要包含任何其他文字。问题：{exc.message}"
                    ),
                ),
            ]
    raise StructuredOutputError(
        f"重试 {max_attempts} 次后仍无法获得合法结构化输出",
        details={"last_error": last_error.message if last_error else None},
    )
