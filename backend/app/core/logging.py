"""统一日志系统。

- 输出：时间 | 级别 | 模块 | request_id | 消息
- 异常自动包含 traceback
- 敏感字段（API Key / Token / 密码）自动脱敏
- 不记录完整用户数据（调用方只传必要的结构化字段）
- 预留 JSON 日志能力（json_mode=True）
- **同时落盘**：控制台 + 日志文件（logs/ 下按天轮转），解决「Agent 工具执行失败后
  无法溯源」的问题——此前日志只写 stdout，服务重启/后台运行即丢失。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# 日志文件目录：默认 backend/logs/（相对 backend 运行目录解析，与 DATA_ROOT 一致）。
# 可通过环境变量 XIAOLUO_LOG_DIR 覆盖。
DEFAULT_LOG_DIR = Path(os.environ.get("XIAOLUO_LOG_DIR", "./logs"))

# 日志文件保留天数（按天轮转，旧文件自动清理）
LOG_BACKUP_DAYS = int(os.environ.get("XIAOLUO_LOG_BACKUP_DAYS", "14"))

# 单文件最大字节数（触发 size 轮转的兜底，正常按天转）
LOG_MAX_BYTES = int(os.environ.get("XIAOLUO_LOG_MAX_BYTES", str(10 * 1024 * 1024)))

# 含这些子串的字段视为敏感信息
SENSITIVE_MARKS = ("key", "token", "secret", "password", "authorization")

# logging.LogRecord 的保留属性，不属于用户 extra
_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


def redact(value: Any) -> Any:
    """递归脱敏：字典中命中敏感字段名的值替换为 ***。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str) and v and any(m in k.lower() for m in SENSITIVE_MARKS):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class LabLogFormatter(logging.Formatter):
    """控制台 / JSON 双模式格式器。"""

    def __init__(self, json_mode: bool = False) -> None:
        super().__init__()
        self.json_mode = json_mode  # 为未来 JSON 日志预留

    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: redact(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if self.json_mode:
            payload = {
                "time": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
                "level": record.levelname,
                "module": record.name,
                "request_id": request_id_var.get(),
                "message": record.getMessage(),
                **extra,
            }
            line = json.dumps(payload, ensure_ascii=False, default=str)
        else:
            line = (
                f"{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}"
                f" | {record.levelname:<7} | {record.name}"
                f" | {request_id_var.get()} | {record.getMessage()}"
            )
            if extra:
                line += f" | {extra}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", json_mode: bool = False, *, log_dir: str | Path | None = None) -> None:
    """初始化根日志器；应用启动时调用一次。

    同时挂两个 handler：控制台（stdout）+ 按天轮转的文件（logs/xiaoluo.log）。
    文件 handler 失败（如目录不可写）时静默降级为仅控制台，绝不影响应用启动。
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(LabLogFormatter(json_mode=json_mode))
    root.addHandler(console)

    # 文件（落盘，按天轮转，保留 LOG_BACKUP_DAYS 天）
    try:
        directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            directory / "xiaoluo.log",
            when="midnight",
            backupCount=LOG_BACKUP_DAYS,
            encoding="utf-8",
        )
        file_handler.setFormatter(LabLogFormatter(json_mode=False))
        file_handler.suffix = "%Y-%m-%d.log"
        root.addHandler(file_handler)
    except Exception:  # noqa: BLE001 - 落盘失败不阻断启动
        root.warning("日志文件 handler 初始化失败，仅输出到控制台", exc_info=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(rid: str) -> None:
    request_id_var.set(rid)


def get_request_id() -> str:
    return request_id_var.get()
