"""大模型连接配置的落盘与恢复。

★ 事故：用户每次启动平台都要重新在设置页点一次「应用到 Agent」，
否则 API 就显示没接入。浏览器那份配置还在，问题出在后端**没有地方记住** ——
``PUT /settings/llm`` 只改进程内存，重启就回到 .env。
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.core.runtime_settings import (
    clear_llm_overrides,
    config_path,
    has_persisted,
    load_llm_overrides,
    save_llm_overrides,
)


class _FakeSettings:
    """只带落盘白名单里那几个键的替身。"""

    def __init__(self, **kwargs: object) -> None:
        self.LLM_PROVIDER = "openai_compatible"
        self.LLM_BASE_URL = ""
        self.LLM_MODEL = ""
        self.LLM_API_KEY = ""
        self.LLM_CONTEXT_WINDOW = 8000
        self.LLM_MAX_OUTPUT_TOKENS = 900
        self.LLM_REMOTE_ENABLED = False
        self.SECRET_NOT_PERSISTED = "database-url"
        self.__dict__.update(kwargs)


def _saved(tmp_path: pathlib.Path, **kwargs: object) -> _FakeSettings:
    settings = _FakeSettings(**kwargs)
    assert save_llm_overrides(settings, tmp_path) is True
    return settings


# ---------------------------------------------------------------- 写入与恢复


def test_roundtrip_restores_key_across_restart(tmp_path: pathlib.Path):
    """重启 = 新建一个 settings 实例；Key 必须能从盘上回来。"""
    _saved(tmp_path, LLM_BASE_URL="https://api.example.com/v1",
           LLM_MODEL="deepseek-chat", LLM_API_KEY="sk-test", LLM_REMOTE_ENABLED=True)

    fresh = _FakeSettings()
    assert fresh.LLM_API_KEY == ""
    assert load_llm_overrides(fresh, tmp_path) is True

    assert fresh.LLM_BASE_URL == "https://api.example.com/v1"
    assert fresh.LLM_MODEL == "deepseek-chat"
    assert fresh.LLM_API_KEY == "sk-test"
    assert fresh.LLM_REMOTE_ENABLED is True


def test_has_persisted_reflects_file(tmp_path: pathlib.Path):
    """设置页靠它显示「重启后是否还在」。"""
    assert has_persisted(tmp_path) is False
    _saved(tmp_path, LLM_MODEL="m")
    assert has_persisted(tmp_path) is True
    assert clear_llm_overrides(tmp_path) is True
    assert has_persisted(tmp_path) is False


def test_no_file_means_env_wins(tmp_path: pathlib.Path):
    """没有落盘文件时绝不能把字段清成空 —— 那会盖掉 .env 里的配置。"""
    fresh = _FakeSettings(LLM_MODEL="from-env", LLM_API_KEY="sk-env")
    assert load_llm_overrides(fresh, tmp_path) is False
    assert fresh.LLM_MODEL == "from-env"
    assert fresh.LLM_API_KEY == "sk-env"


# ---------------------------------------------------------------- 白名单


def test_only_whitelisted_keys_are_persisted(tmp_path: pathlib.Path):
    """整个 Settings dump 下去会把数据库 URL、上传上限一起固化 —— 必须白名单。"""
    _saved(tmp_path, LLM_MODEL="m", SECRET_NOT_PERSISTED="leaked")
    payload = json.loads(config_path(tmp_path).read_text(encoding="utf-8"))
    assert "SECRET_NOT_PERSISTED" not in payload
    assert "LLM_MODEL" in payload


def test_unknown_keys_in_file_are_ignored(tmp_path: pathlib.Path):
    """盘上的文件是用户可编辑的：出现陌生键时忽略，而不是崩掉。"""
    config_path(tmp_path).write_text(
        json.dumps({"LLM_MODEL": "m", "EVIL": "rm -rf"}), encoding="utf-8"
    )
    fresh = _FakeSettings()
    assert load_llm_overrides(fresh, tmp_path) is True
    assert fresh.LLM_MODEL == "m"
    assert not hasattr(fresh, "EVIL")


# ---------------------------------------------------------------- 容错


def test_corrupted_file_does_not_break_startup(tmp_path: pathlib.Path):
    """坏文件不能让进程起不来 —— 落盘是便利功能，不是启动依赖。"""
    config_path(tmp_path).write_text("{ 这不是 JSON", encoding="utf-8")
    fresh = _FakeSettings(LLM_MODEL="from-env")
    assert load_llm_overrides(fresh, tmp_path) is False
    assert fresh.LLM_MODEL == "from-env"


def test_wrong_type_file_is_ignored(tmp_path: pathlib.Path):
    config_path(tmp_path).write_text('["not", "a", "dict"]', encoding="utf-8")
    assert load_llm_overrides(_FakeSettings(), tmp_path) is False


def test_clear_on_missing_file_is_not_an_error(tmp_path: pathlib.Path):
    assert clear_llm_overrides(tmp_path) is True


def test_atomic_write_leaves_no_temp_files(tmp_path: pathlib.Path):
    """写失败会留下 .llm-*.tmp 残片；正常路径下一个都不该有。"""
    _saved(tmp_path, LLM_MODEL="m")
    assert list(tmp_path.glob(".llm-*")) == []
    assert [p.name for p in tmp_path.iterdir()] == ["llm_settings.json"]


# ---------------------------------------------------------------- HTTP 闭环


def test_saving_via_api_persists_for_the_next_start(client, settings_sandbox):
    """★ 端到端：设置页「保存并应用」之后，重启（新进程读盘）要能自己恢复。

    这一步缺失时的症状就是用户那句「每次启动平台都要重新应用一遍」。
    """
    resp = client.put("/api/v1/settings/llm", json={
        "provider_type": "openai_compatible",
        "base_url": "https://api.example.com/v1",
        "model": "deepseek-chat",
        "api_key": "sk-from-settings-page",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["api_key_set"] is True
    assert data["persisted"] is True

    # 模拟一次重启：新建一个空白 settings，只做启动时那一次回填
    from app.core.config import Settings
    from app.core.runtime_settings import load_llm_overrides

    restarted = Settings()
    assert load_llm_overrides(restarted, settings_sandbox) is True
    # 落盘值必须盖过 .env：用户在界面上填的那一版才是他要用的
    assert restarted.LLM_API_KEY == "sk-from-settings-page"
    assert restarted.LLM_MODEL == "deepseek-chat"


@pytest.fixture()
def settings_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    """把落盘位置与进程内 LLM 字段隔离出来，用完还原。"""
    from app.core import config as config_module
    from app.core.runtime_settings import clear_llm_overrides

    keys = ("DATA_ROOT", "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")
    saved = {key: getattr(config_module.settings, key) for key in keys}
    monkeypatch.setattr(config_module.settings, "DATA_ROOT", str(tmp_path))
    yield tmp_path
    for key, value in saved.items():
        setattr(config_module.settings, key, value)
    clear_llm_overrides(tmp_path)
