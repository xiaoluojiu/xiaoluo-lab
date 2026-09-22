"""Prompt 005 测试：配置系统。"""

from __future__ import annotations

import json

from app.core.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.APP_NAME == "xiaoluo-lab"
    assert s.APP_ENV == "dev"
    assert s.LOG_LEVEL == "INFO"
    assert s.DATABASE_URL.startswith("sqlite")


def test_env_override():
    s = Settings(_env_file=None, APP_ENV="prod", PORT=9999)
    assert s.APP_ENV == "prod"
    assert s.PORT == 9999


def test_masked_summary_never_contains_api_key():
    s = Settings(_env_file=None, LLM_API_KEY="sk-very-secret-key")
    summary = json.dumps(s.masked_summary())
    assert "sk-very-secret-key" not in summary
    assert s.masked_summary()["llm_api_key_set"] is True


def test_mask_url_with_credentials():
    masked = Settings._mask_url("postgresql://user:pass123@localhost:5432/lab")
    assert "pass123" not in masked
    assert masked.startswith("postgresql://user:***@")


def test_remote_llm_available_requires_both_switch_and_key():
    """总开关与 API Key 必须**同时**满足才算可用。

    设置页那个「启用远程 API 大模型」开关靠这一条生效：关掉开关时，
    即使进程里还留着 API Key，也不得构造远程 Provider。
    """
    off = Settings(_env_file=None, LLM_API_KEY="sk-x", LLM_REMOTE_ENABLED=False)
    assert off.remote_llm_available() is False
    no_key = Settings(_env_file=None, LLM_API_KEY="", LLM_REMOTE_ENABLED=True)
    assert no_key.remote_llm_available() is False
    both = Settings(_env_file=None, LLM_API_KEY="sk-x", LLM_REMOTE_ENABLED=True)
    assert both.remote_llm_available() is True

    # 开关状态必须回给前端（否则界面无法显示「远程 API 已停用」）。
    assert off.llm_model_summary()["remote_enabled"] is False
    assert both.llm_model_summary()["remote_enabled"] is True
    # 关掉开关不清凭据：重新开启不需要重填 Key。
    assert off.llm_model_summary()["api_key_set"] is True


def test_get_llm_provider_respects_remote_switch(monkeypatch):
    """关掉总开关 ⇒ Agent 拿不到远程 Provider（退回规则规划器）。"""
    from app.api.deps import get_llm_provider
    from app.core.config import settings as runtime_settings

    monkeypatch.setattr(runtime_settings, "LLM_API_KEY", "sk-x")
    monkeypatch.setattr(runtime_settings, "LLM_REMOTE_ENABLED", False)
    assert get_llm_provider() is None

    monkeypatch.setattr(runtime_settings, "LLM_REMOTE_ENABLED", True)
    assert get_llm_provider() is not None
