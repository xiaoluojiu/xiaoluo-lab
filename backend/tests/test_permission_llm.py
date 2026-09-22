"""Phase 6（Prompt 114-120）Permission 与 LLM 测试。"""

from __future__ import annotations

import httpx
import pytest
from app.agent.llm.base import LLMException, LLMMessage
from app.agent.llm.mock import MockLLM
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.agent.llm.structured import (
    StructuredOutputError,
    extract_json,
    parse_structured,
    structured_output_with_retry,
)
from app.agent.permission.manager import PermissionManager
from app.agent.permission.models import ROLE_PERMISSIONS, Decision, Permission
from app.agent.permission.rules import (
    DEFAULT_TOOL_RISKS,
    RiskLevel,
    risk_requires_confirmation,
)
from app.tools.builtin import TOOL_REGISTRY
from app.tools.context import ToolExecutionContext
from pydantic import BaseModel


def ctx_with(*permissions: Permission, dataset_ids: set[int] | None = None):
    return ToolExecutionContext(
        user_id="tester",
        permissions=set(permissions),
        dataset_ids=dataset_ids or set(),
    )


# ----------------------------------------------------------------------
# Prompt 114: Permission 模型
# ----------------------------------------------------------------------
def test_permission_enum_complete():
    expected = {
        "read_data", "analyze_data", "modify_data", "create_version",
        "train_model", "export_data", "export_report", "execute_workflow",
        "delete_data",
    }
    assert {p.value for p in Permission} == expected


def test_role_presets():
    assert Permission.DELETE_DATA in ROLE_PERMISSIONS["admin"]
    assert Permission.DELETE_DATA not in ROLE_PERMISSIONS["analyst"]
    assert ROLE_PERMISSIONS["viewer"] == {
        Permission.READ_DATA,
        Permission.ANALYZE_DATA,
    }


# ----------------------------------------------------------------------
# Prompt 115: Risk Level 与映射
# ----------------------------------------------------------------------
def test_risk_levels_and_confirmation():
    assert not risk_requires_confirmation(RiskLevel.LOW)
    assert not risk_requires_confirmation(RiskLevel.MEDIUM)
    assert risk_requires_confirmation(RiskLevel.HIGH)
    assert risk_requires_confirmation(RiskLevel.CRITICAL)


def test_tool_risk_mapping_covers_builtin_tools():
    for name in TOOL_REGISTRY.names():
        assert name in DEFAULT_TOOL_RISKS, f"工具 {name} 缺少风险映射"
    assert DEFAULT_TOOL_RISKS["dataset.list"] == RiskLevel.LOW
    assert DEFAULT_TOOL_RISKS["data.merge"] == RiskLevel.HIGH


# ----------------------------------------------------------------------
# Prompt 116: PermissionManager
# ----------------------------------------------------------------------
def test_manager_allow():
    tool = TOOL_REGISTRY.get("dataset.list")
    decision = PermissionManager().check(
        "u1", tool, ctx_with(Permission.READ_DATA)
    )
    assert decision.decision == Decision.ALLOW


def test_manager_deny_missing_permission():
    tool = TOOL_REGISTRY.get("dataset.profile")  # analyze_data
    decision = PermissionManager().check(
        "u1", tool, ctx_with(Permission.READ_DATA)
    )
    assert decision.decision == Decision.DENY
    assert "缺少权限" in decision.reason


def test_manager_deny_dataset_scope():
    tool = TOOL_REGISTRY.get("dataset.preview")
    decision = PermissionManager().check(
        "u1",
        tool,
        ctx_with(Permission.READ_DATA, dataset_ids={7}),
        params={"dataset_id": 1},
    )
    assert decision.decision == Decision.DENY
    assert "允许范围" in decision.reason


def test_manager_require_confirmation_for_high_risk():
    tool = TOOL_REGISTRY.get("ml.train")
    decision = PermissionManager().check(
        "u1", tool, ctx_with(Permission.TRAIN_MODEL)
    )
    assert decision.decision == Decision.REQUIRE_CONFIRMATION


def test_agent_cannot_bypass_manager():
    """ToolRegistry.execute 必须经过 PermissionManager：无权限直接拒绝。"""
    tool = TOOL_REGISTRY.get("data.merge")
    assert tool.risk_level in ("high", RiskLevel.HIGH)
    with pytest.raises(Exception, match="缺少权限"):
        TOOL_REGISTRY.execute(
            "data.merge", {}, ctx_with(Permission.READ_DATA), None
        )


# ----------------------------------------------------------------------
# Prompt 117: LLMProvider 接口
# ----------------------------------------------------------------------
def test_provider_interface_and_structured_output_dispatch():
    class Answer(BaseModel):
        value: int

    llm = MockLLM(structured_responses=[{"value": 3}])
    result = llm.structured_output(
        [LLMMessage(role="user", content="给个数")], Answer
    )
    assert result == {"value": 3}
    assert len(llm.calls) == 1


# ----------------------------------------------------------------------
# Prompt 118: OpenAI-compatible Provider
# ----------------------------------------------------------------------
def test_openai_provider_requires_api_key(monkeypatch):
    for var in ("XIAOLUO_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    provider = OpenAICompatibleProvider(
        "https://api.example.com/v1", model="test-model"
    )
    with pytest.raises(LLMException, match="API Key"):
        provider.chat([LLMMessage(role="user", content="hi")])


def test_openai_provider_chat_success(monkeypatch):
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers})
        resp = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "你好！"}}],
                "model": "test-model",
                "usage": {"total_tokens": 10},
            },
            request=httpx.Request("POST", url),
        )
        return resp

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OpenAICompatibleProvider(
        "https://api.example.com/v1/", model="test-model", api_key="sk-test"
    )
    resp = provider.chat(
        [LLMMessage(role="user", content="hi")], temperature=0.2
    )
    assert resp.content == "你好！"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["temperature"] == 0.2
    # API Key 不应出现在请求体中（只在 Header）
    assert "sk-test" not in str(captured["json"])


def test_openai_provider_http_error(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            500, text="boom", request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OpenAICompatibleProvider(
        "https://api.example.com/v1", model="m", api_key="sk"
    )
    with pytest.raises(LLMException, match="500"):
        provider.chat([LLMMessage(role="user", content="hi")])


# ----------------------------------------------------------------------
# Prompt 119: MockLLM
# ----------------------------------------------------------------------
def test_mock_llm_queue_and_echo():
    llm = MockLLM(responses=["第一次回复"])
    first = llm.chat([LLMMessage(role="user", content="q1")])
    assert first.content == "第一次回复"
    second = llm.chat([LLMMessage(role="user", content="q2")])
    assert second.content == "[mock] q2"
    llm.assert_chat_count(2)
    with pytest.raises(AssertionError):
        llm.assert_chat_count(3)


# ----------------------------------------------------------------------
# Prompt 120: 结构化输出解析
# ----------------------------------------------------------------------
class Step(BaseModel):
    action: str
    args: dict = {}


def test_extract_json_direct_and_fenced():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('说明文字 {"a": [1, 2]} 尾部') == {"a": [1, 2]}
    assert extract_json('[1, 2, 3]') == [1, 2, 3]
    with pytest.raises(StructuredOutputError):
        extract_json("完全没有 JSON")
    with pytest.raises(StructuredOutputError):
        extract_json("")


def test_parse_structured_validation():
    result = parse_structured('{"action": "filter", "args": {"x": 1}}', Step)
    assert result == {"action": "filter", "args": {"x": 1}}
    with pytest.raises(StructuredOutputError, match="Schema"):
        parse_structured('{"wrong": true}', Step)


def test_structured_output_retry_success():
    class Out(BaseModel):
        action: str

    llm = MockLLM(
        responses=["这不是 JSON", '{"action": "filter"} 多余文字'],
    )
    result = structured_output_with_retry(
        llm, [LLMMessage(role="user", content="做点什么")], Out, max_attempts=3
    )
    assert result == {"action": "filter"}
    assert len(llm.calls) == 2  # 第一次失败后重试了一次


def test_structured_output_retry_exhausted():
    class Out(BaseModel):
        action: str

    llm = MockLLM(responses=["坏输出 1", "坏输出 2", "坏输出 3"])
    with pytest.raises(StructuredOutputError, match="重试"):
        structured_output_with_retry(
            llm, [LLMMessage(role="user", content="x")], Out, max_attempts=3
        )
    assert len(llm.calls) == 3
