"""LLM 响应解析：推理型模型的 content 空值兜底。

背景：deepseek-flash / deepseek-reasoner 这类推理型模型把主要输出放在
``reasoning_content``，小 max_tokens 下 ``content`` 会为空。旧解析只读
``content``，于是得到空回答并静默退回模板（用户看到「回答总是那几句」）。
"""

from __future__ import annotations

from app.agent.llm import LLMMessage, OpenAICompatibleProvider, _thinking_payload


class _FakeResp:
    status_code = 200

    def __init__(self, data: dict, text: str = ""):
        self._data = data
        self.text = text

    def json(self) -> dict:
        return self._data


def _resp(message: dict) -> _FakeResp:
    return _FakeResp(
        {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
    )


def test_content_is_used_when_present():
    parsed = OpenAICompatibleProvider._parse(
        _resp({"role": "assistant", "content": "42", "reasoning_content": "thinking..."})
    )
    assert parsed.content == "42"


def test_empty_content_falls_back_to_reasoning_content():
    """推理型模型 content 为空时必须回退，不能静默返回空回答。"""
    parsed = OpenAICompatibleProvider._parse(
        _resp({"role": "assistant", "content": "", "reasoning_content": "答案是 42"})
    )
    assert parsed.content == "答案是 42"


def test_both_empty_returns_empty_not_crash():
    parsed = OpenAICompatibleProvider._parse(_resp({"role": "assistant", "content": ""}))
    assert parsed.content == ""


def test_usage_is_parsed():
    parsed = OpenAICompatibleProvider._parse(
        _resp({"role": "assistant", "content": "x"})
    )
    assert parsed.usage.input_tokens == 5
    assert parsed.usage.output_tokens == 10


# ---------------------------------------------------------------- 思考模式


def test_deepseek_requests_thinking_disabled():
    """deepseek 必须显式关闭 thinking。

    默认开启时模型把 token 花在思维链上，小 max_tokens 下 content 为空，
    旧逻辑回退 reasoning_content ⇒ 用户拿到的是模型自言自语，不是答案。
    """
    payload = _thinking_payload("deepseek-flash", "https://api.deepseek.com")
    assert payload == {"thinking": {"type": "disabled"}}


def test_non_deepseek_gets_no_thinking_field():
    """给不支持该字段的厂商发过去会变成一次不重试的 400，绝不能乱发。"""
    assert _thinking_payload("qwen-plus", "https://dashscope.aliyuncs.com") == {}
    assert _thinking_payload("gpt-4o-mini", "https://api.openai.com/v1") == {}


def test_chat_payload_carries_thinking_flag():
    """构造出来的请求体必须真的带上这个字段，否则等于没改。"""
    captured: dict = {}

    class _Client:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 - httpx 的签名
            captured.update(json)
            return _FakeResp({"choices": [{"message": {"content": "ok"}}], "usage": {}})

        def close(self):
            pass

    provider = OpenAICompatibleProvider(
        "https://api.deepseek.com", "deepseek-flash", "k", client=_Client()
    )
    provider.chat([LLMMessage(role="user", content="hi")])
    assert captured.get("thinking") == {"type": "disabled"}


def test_thinking_enabled_keeps_provider_default():
    """显式打开时必须尊重调用方：有些人就是要拿思维链。"""
    captured: dict = {}

    class _Client:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
            captured.update(json)
            return _FakeResp({"choices": [{"message": {"content": "ok"}}], "usage": {}})

        def close(self):
            pass

    provider = OpenAICompatibleProvider(
        "https://api.deepseek.com", "deepseek-flash", "k", client=_Client(), thinking="enabled"
    )
    provider.chat([LLMMessage(role="user", content="hi")])
    assert "thinking" not in captured
