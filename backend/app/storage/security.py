r"""Storage 安全检查。

目标：任何 Storage 操作都不能逃逸存储根目录（DATA_ROOT）。
检查项：
- 路径穿越（.. / ..\）
- 绝对路径（/ 开头、盘符）
- 符号链接逃逸（resolve 后必须仍在根目录内）
- 非法文件名（Windows 保留名、非法字符、控制字符）
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from app.core.exceptions import StorageException

# Windows 保留设备名
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Windows 非法文件名字符 + 控制字符
_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def validate_key(key: str) -> str:
    """校验并规范化存储 key（POSIX 风格相对路径），非法时抛出 StorageException。"""
    if not key or not key.strip():
        raise StorageException("empty storage key", code="INVALID_PATH", details={"key": key})

    normalized = key.strip().replace("\\", "/")

    if normalized.startswith("/") or _DRIVE_RE.match(normalized):
        raise StorageException(
            "absolute path is not allowed",
            code="INVALID_PATH",
            details={"key": key},
        )

    segments = normalized.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise StorageException(
                "path traversal or empty segment is not allowed",
                code="INVALID_PATH",
                details={"key": key},
            )
        if _ILLEGAL_CHARS.search(segment):
            raise StorageException(
                "illegal characters in path",
                code="INVALID_PATH",
                details={"key": key},
            )
        if segment.lower() in _RESERVED_NAMES:
            raise StorageException(
                "reserved device name is not allowed",
                code="INVALID_PATH",
                details={"key": key},
            )

    return str(PurePosixPath(normalized))


def ensure_inside_root(root: Path, target: Path) -> Path:
    """确保目标路径（resolve 后，解析符号链接）仍在 root 内，返回解析后的目标路径。"""
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError:
        raise StorageException(
            "path escapes storage root",
            code="PATH_ESCAPE",
            details={"target": str(target)},
        ) from None
    return resolved_target
