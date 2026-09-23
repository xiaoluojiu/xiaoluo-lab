"""连接器凭据加密。

为什么必须加密
--------------
外部数据库口令是本平台唯一「为了访问别人家数据而必须长期持有的秘密」。
把它明文写进平台自己的 SQLite，等于把一次平台备份泄漏放大成
「所有下游库的账号一起泄漏」。所以口令只以密文落库，出参一律掩码
（前端永远拿不回原文，只能重填）。

设计与取舍
----------
- 算法用 ``cryptography`` 的 **Fernet**（AES-128-CBC + HMAC-SHA256，
  带认证的对称加密）。不自造加密原语：口令存储是最不适合发挥创造力的地方。
- 密钥来源优先级：显式配置 ``CONNECTOR_SECRET_KEY`` → 持久化密钥文件
  ``{MODEL_ROOT}/connector_secret.key``（首次生成，权限收紧到 0600）。
  显式配置是为了让多实例部署共用同一密钥；自动生成只适合单机。
- ``cryptography`` 缺失时**不静默降级为明文**：需要保存口令的调用会收到一个
  明确的、可操作的错误（``CONNECTOR_CRYPTO_UNAVAILABLE``）。能明文跑通的
  保险做法只会让人以为它是加密的。
- 密文自带版本前缀 ``enc:v1:``，便于将来换算法时区分历史数据。
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger("app.connectors.crypto")

CIPHER_PREFIX = "enc:v1:"
_MASK = "********"


class ConnectorCryptoUnavailable(AppException):
    """加密依赖缺失（需要保存口令但无法安全存储）。"""

    http_status = 501
    default_code = "CONNECTOR_CRYPTO_UNAVAILABLE"
    default_message = "凭据加密不可用：请安装 cryptography 依赖"


def crypto_available() -> bool:
    """``cryptography`` 是否可用。"""
    try:
        import cryptography.fernet  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def mask_secret(value: str | None) -> str:
    """掩码展示：有值返回固定星号串，无值返回空串。

    刻意不返回「前几位 + 星号」：那仍然泄漏口令前缀长度与首字符，
    而前端本来也不需要它——只关心「填过了没有」。
    """
    return _MASK if value else ""


def _derive_key(raw: str) -> bytes:
    """把任意字符串密钥规整为 Fernet 要求的 32 字节 urlsafe base64。

    用 sha256 派生而不是要求用户自己去生成 base64，是为了让
    ``CONNECTOR_SECRET_KEY=whatever-strong-passphrase`` 也能安全使用。
    """
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _default_key_path() -> Path:
    return settings.model_root_path / "connector_secret.key"


def _load_or_create_file_key(path: Path | None = None) -> bytes:
    """从密钥文件读取；不存在则生成并收紧权限。"""
    target = path or _default_key_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        raw = target.read_text(encoding="utf-8").strip()
        if raw:
            return base64.urlsafe_b64decode(raw.encode("ascii"))

    key = base64.urlsafe_b64encode(os.urandom(32))
    target.write_text(key.decode("ascii"), encoding="utf-8")

    # 尽力收紧权限：Windows 上 chmod 语义有限，失败不应阻断启动。
    try:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - 平台差异
        logger.warning("无法收紧密钥文件权限（不影响功能）：%s", target)

    logger.info("已生成连接器密钥文件：%s", target)
    return key


class CredentialCipher:
    """口令的对称加解密。"""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key
        self._fernet = None

    # ---- 构造 ----

    @classmethod
    def from_settings(cls) -> CredentialCipher:
        """按配置解析密钥来源。"""
        raw = (settings.CONNECTOR_SECRET_KEY or "").strip()
        if raw:
            return cls(_derive_key(raw))
        return cls(_load_or_create_file_key())

    # ---- 能力 ----

    @property
    def available(self) -> bool:
        return crypto_available() and self._key is not None

    def _fernet_instance(self) -> Any:
        if self._fernet is not None:
            return self._fernet
        if not self.available:
            raise ConnectorCryptoUnavailable(
                "无法加密连接器口令：请安装 cryptography（pip install cryptography），"
                "或使用无口令的连接方式（如 SQLite 文件）。",
                details={"install": "cryptography"},
            )
        from cryptography.fernet import Fernet

        self._fernet = Fernet(self._key)
        return self._fernet

    # ---- 读写 ----

    def encrypt(self, plaintext: str | None) -> str:
        """加密口令；空值返回空串（表示「无口令」，不是「未设置」）。"""
        if not plaintext:
            return ""
        token = self._fernet_instance().encrypt(plaintext.encode("utf-8"))
        return f"{CIPHER_PREFIX}{token.decode('ascii')}"

    def decrypt(self, stored: str | None) -> str:
        """解密口令。历史/异常数据解密失败时返回空串并记日志，不让整条链路崩掉。"""
        if not stored:
            return ""
        if not stored.startswith(CIPHER_PREFIX):
            # 兼容：非本前缀的内容一律视为不可用，绝不当作明文使用。
            logger.error("连接器口令格式无法识别，已忽略（不做明文回退）")
            return ""
        token = stored[len(CIPHER_PREFIX):].encode("ascii")
        try:
            return self._fernet_instance().decrypt(token).decode("utf-8")
        except Exception:  # noqa: BLE001 - 密钥轮换 / 数据损坏
            logger.exception("连接器口令解密失败（可能是密钥已变更）")
            return ""


_DEFAULT_CIPHER: CredentialCipher | None = None


def get_cipher() -> CredentialCipher:
    """进程级默认加密器（避免每次请求都读密钥文件）。"""
    global _DEFAULT_CIPHER
    if _DEFAULT_CIPHER is None:
        _DEFAULT_CIPHER = CredentialCipher.from_settings()
    return _DEFAULT_CIPHER


def reset_cipher() -> None:
    """测试辅助：丢弃缓存的密钥（切换 MODEL_ROOT 后需要重新读取）。"""
    global _DEFAULT_CIPHER
    _DEFAULT_CIPHER = None
