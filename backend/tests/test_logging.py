"""Prompt 006 测试：日志系统。"""

from __future__ import annotations

import logging

from app.core import logging as lab_logging


def test_redact_sensitive_fields():
    data = {
        "llm_api_key": "sk-abc",
        "access_token": "t-1",
        "nested": {"password": "p"},
        "name": "ok",
    }
    out = lab_logging.redact(data)
    assert out["llm_api_key"] == "***"
    assert out["access_token"] == "***"
    assert out["nested"]["password"] == "***"
    assert out["name"] == "ok"


def test_formatter_contains_time_module_request_id():
    formatter = lab_logging.LabLogFormatter()
    record = logging.LogRecord(
        name="mymodule", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    lab_logging.set_request_id("rid-1")
    line = formatter.format(record)
    assert "mymodule" in line
    assert "rid-1" in line
    assert "hello world" in line


def test_formatter_json_mode():
    import json

    formatter = lab_logging.LabLogFormatter(json_mode=True)
    record = logging.LogRecord(
        name="m", level=logging.INFO, pathname="", lineno=0,
        msg="hi", args=None, exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hi"
    assert {"time", "level", "module", "request_id"} <= set(payload)


def test_setup_logging_levels():
    lab_logging.setup_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    lab_logging.setup_logging("INFO")
    assert root.level == logging.INFO


def test_setup_logging_writes_to_file(tmp_path):
    """日志必须落盘（此前只写 stdout，服务重启后错误无从溯源——回归）。"""
    lab_logging.setup_logging("INFO", log_dir=str(tmp_path))
    logger = logging.getLogger("test_file_log")
    logger.error("落盘测试")
    for h in logging.getLogger().handlers:
        h.flush()
    files = list(tmp_path.glob("*.log"))
    assert files, "应生成日志文件"
    content = files[0].read_text(encoding="utf-8")
    assert "落盘测试" in content
    # 恢复默认（避免污染后续测试的日志目录）
    lab_logging.setup_logging("INFO")


def test_redact_params_sanitizes_sensitive_keys():
    """工具参数脱敏：api_key 等敏感键的值不能原样进日志。"""
    from app.tools.registry import _redact_params
    text = _redact_params({"dataset_id": 5, "api_key": "sk-secret", "ok": "fine"})
    assert "sk-secret" not in text
    assert "dataset_id" in text
