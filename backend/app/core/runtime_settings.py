"""运行期配置落盘：让「设置页改过的大模型配置」跨进程重启存活。

为什么需要它
------------
``Settings`` 的值有四个来源（构造参数 > 环境变量 > .env > 默认值），
但**没有一个是用户在界面上填的**。设置页 ``PUT /settings/llm`` 改的是
进程内存里的字段，进程一重启就回到 .env —— 而几乎没有用户会去写 .env。

表现就是用户反馈的那句：「每次我启动平台都要重新在设置里把配置应用到
Agent 操作一遍，不然 API 就显示没接入」。这不是前端没记住（浏览器那份还在），
而是后端**没有地方记住**。

范围与边界
----------
- 只持久化「大模型连接」这一组键。Agent 策略类配置（预算、上下文分区）仍按
  「进程内可调、重启回默认」处理 —— 它们是调参旋钮，不该悄悄变成永久值。
- 落盘位置 ``{DATA_ROOT}/llm_settings.json``：与数据库、数据集同目录，
  备份/迁移时一起走。
- 文件权限收紧到 0600。API Key 在这里是明文的（与 .env 同级风险），
  收紧权限是能做的最后一道；真正安全的方案是密钥管理服务，不在本次范围。
- **读写失败一律不影响启动**：落盘是便利功能，不是启动依赖。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 落盘文件名（放在 DATA_ROOT 下）
_FILE_NAME = "llm_settings.json"

#: 允许落盘的键。新增键必须同时在这里登记 —— 白名单而不是「把整个 Settings
#: dump 下去」，后者会把数据库 URL、上传上限这些不该被界面改动的值也固化掉。
_PERSISTED_KEYS: tuple[str, ...] = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_CONTEXT_WINDOW",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_REMOTE_ENABLED",
)


def config_path(root: Path | str) -> Path:
    return Path(root) / _FILE_NAME


def has_persisted(root: Path | str) -> bool:
    """是否存在落盘配置（供设置页显示「重启后是否还在」）。"""
    return config_path(root).is_file()


def load_llm_overrides(target: Any, root: Path | str) -> bool:
    """把落盘值回填进 settings 实例。

    文件不存在时什么都不做 —— 此时完全按环境变量 / .env 走，
    新装环境和 CI 不受影响。

    :return: 是否真的回填过（False 表示没有落盘配置，或读到了坏文件）
    """
    path = config_path(root)
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # 坏文件不能让进程起不来：按 .env 继续，并把原因留在日志里
        logger.warning("读取落盘的大模型配置失败（%s），按环境变量 / .env 继续：%s", path, exc)
        return False
    if not isinstance(raw, dict):
        logger.warning("落盘的大模型配置格式不正确（%s），已忽略", path)
        return False

    applied = 0
    for key in _PERSISTED_KEYS:
        if key in raw and hasattr(target, key):
            setattr(target, key, raw[key])
            applied += 1
    if applied:
        logger.info(
            "已从 %s 恢复大模型连接配置（model=%s，远程开关=%s）",
            path, getattr(target, "LLM_MODEL", ""), getattr(target, "LLM_REMOTE_ENABLED", ""),
        )
    return applied > 0


def save_llm_overrides(source: Any, root: Path | str) -> bool:
    """把当前 settings 里的大模型连接配置写盘。原子写 + 权限收紧。"""
    payload = {key: getattr(source, key, None) for key in _PERSISTED_KEYS}
    path = config_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".llm-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            # 半截文件不能留在磁盘上：下次启动会读到一个截断的 JSON
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        _restrict(path)
        return True
    except OSError as exc:
        logger.warning("持久化大模型配置失败（%s）：%s", path, exc)
        return False


def clear_llm_overrides(root: Path | str) -> bool:
    """删除落盘配置（恢复默认：只认 .env 与环境变量）。"""
    path = config_path(root)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning("删除落盘的大模型配置失败（%s）：%s", path, exc)
        return False


def _restrict(path: Path) -> None:
    """收紧为仅属主可读写。Windows 上 chmod 语义有限，失败不影响功能。"""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
