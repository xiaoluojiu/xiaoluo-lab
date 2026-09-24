"""OpenAI-compatible Provider。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import httpx

from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, LLMResponse
from app.agent.llm.capabilities import DEFAULT_OPENAI_COMPATIBLE_CAPABILITIES, LLMCapabilities
from app.core.config import settings

logger = logging.getLogger(__name__)

API_KEY_ENV_VARS = ("XIAOLUO_LLM_API_KEY", "OPENAI_API_KEY")

# 推理型模型耗尽输出预算时的重试次数与预算上限
_MAX_REASONING_RETRIES = 2
_MAX_RETRY_TOKENS = 16384

# 网络/服务端错误的指数退避重试（区别于「推理耗尽」的放大预算重试）。
# 只对「可重试」错误重试：连接失败 / 超时 / 5xx / 429；4xx（鉴权/参数错误）不重试。
_MAX_NETWORK_RETRIES = 2
_RETRY_BACKOFF_BASE_SECONDS = 1.0


def _is_retryable(exc: httpx.HTTPError) -> bool:
    """判断一个 HTTP 错误是否值得重试（瞬时故障 vs 确定性失败）。"""
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)):
        return True
    return False


def _is_retryable_status(status: int) -> bool:
    """**仅按状态码**判断是否值得重试。

    注意这里刻意不接受响应体：429 既可能是「限流，稍后再试」，也可能是
    「账户余额不足」——后者重试一万次也不会成功，只会把用户的额度探测
    变成一次 DDoS。配额 / 计费类错误由 `_is_quota_error()` 先行拦截，
    走不到这个判断。
    """
    return status >= 500 or status == 429


# 配额 / 计费类错误的特征词。各家网关的措辞不统一（code / type / message 都可能带），
# 因此统一小写后做子串匹配；宁可多认领一个（不重试的代价只是一次请求失败），
# 也不能漏掉（漏掉会变成无意义的重试风暴）。
_QUOTA_HINTS = (
    "insufficient_quota",
    "insufficient quota",
    "quota",
    "billing",
    "balance",
    "credit",
    "payment",
    "arrears",
    "out of balance",
    "exceeded your current quota",
    "欠费",
    "余额不足",
    "账户余额",
)


def _error_envelope(body: str) -> dict[str, Any]:
    """取 OpenAI 兼容协议的错误信封 `{"error": {...}}`；取不到就返回空字典。

    入参是**响应体文本**而不是 Response 对象：`_is_quota_error` 与
    `_quota_message` 都要复用它，而它们拿到的只有已经截断过的 body。
    """
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    error = data.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str):
        return {"message": error}
    return {}


def _is_quota_error(status: int, body: str) -> bool:
    """这次失败是不是「账户没钱了」这类**重试永远不会成功**的错误。"""
    if status == 402:  # Payment Required：部分网关用它表示欠费
        return True
    if status not in (400, 401, 403, 429):
        # 5xx 是服务端自己的问题，不属于配额；其余 4xx 多为参数/鉴权，
        # 也按「不重试」处理，但语义不是欠费，不在这里认领。
        return False
    blob = body.lower()
    return any(hint in blob for hint in _QUOTA_HINTS)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """尊重服务端给的 `Retry-After`（秒 / HTTP 日期），取不到返回 None。"""
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return None


def _quota_message(status: int, body: str) -> str:
    """把网关那句英文错误翻成用户能行动的中文，并原样附上原始信息。

    「Insufficient Balance」对用户的含义是「去充值或换 Key」，
    直接把它塞进「暂时无法完成对话请求：…」里等于什么都没说。
    """
    envelope = _error_envelope(body)
    provider_msg = str(envelope.get("message") or envelope.get("code") or "").strip()
    if not provider_msg:
        provider_msg = body.strip()[:200]
    return f"大模型账户配额/余额不足（HTTP {status}）" + (f"：{provider_msg}" if provider_msg else "")


class _CircuitBreaker:
    """进程级熔断器：连续失败达到阈值后短路一段时间，避免 LLM 抖动期持续打请求。"""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                if time.monotonic() - self._opened_at < self.cooldown_seconds:
                    raise LLMException(
                        "LLM 服务熔断中：连续多次失败，请稍后再试",
                        details={"cooldown_seconds": self.cooldown_seconds},
                    )
                # 冷却期已过，重置为半开状态（允许下一次请求试探）
                self._opened_at = None
                self._consecutive_failures = 0

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = time.monotonic()


# 按 base_url 隔离的熔断器（不同网关互不影响）。
_CIRCUIT_BREAKERS: dict[str, _CircuitBreaker] = {}
_CB_LOCK = threading.Lock()


def _breaker_for(base_url: str) -> _CircuitBreaker:
    with _CB_LOCK:
        if base_url not in _CIRCUIT_BREAKERS:
            _CIRCUIT_BREAKERS[base_url] = _CircuitBreaker()
        return _CIRCUIT_BREAKERS[base_url]


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容 Provider；DeepSeek/Qwen/Kimi/GLM/网关/本地服务均可复用。"""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        capabilities: LLMCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or next((os.environ[v] for v in API_KEY_ENV_VARS if os.environ.get(v)), None)
        self.timeout = timeout
        self.capabilities = capabilities or DEFAULT_OPENAI_COMPATIBLE_CAPABILITIES
        self._breaker = _breaker_for(self.base_url)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMException("LLM API Key 未配置（应通过 backend 环境变量或构造参数提供）")
        self.begin_call()
        payload: dict[str, Any] = {"model": self.model, "messages": [m.to_dict() for m in messages], **kwargs}
        if temperature is not None:
            payload["temperature"] = temperature
        effective_max_tokens = max_tokens if max_tokens is not None else settings.AGENT_LLM_MAX_OUTPUT_TOKENS
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens

        # 推理型模型（如 deepseek-flash / reasoner）会把 max_tokens 先花在思维链上：
        # 预算不够时 finish_reason='length'、content 为空串，正文全在 reasoning_content 里。
        # 直接返回空内容会让上层以为"调用成功但没有输出"，表现为计划/叙述静默失效。
        # 这里识别该情形并放大预算重试，最多放大到 MAX_RETRY_TOKENS。
        attempt_budget = effective_max_tokens
        for attempt in range(_MAX_REASONING_RETRIES + 1):
            if attempt_budget is not None:
                payload["max_tokens"] = int(attempt_budget)
            data = self._request(payload)
            content = self._content_of(data)
            finish = self._finish_reason_of(data)
            reasoning = self._reasoning_of(data)
            if content or not (reasoning or "") or finish != "length":
                break
            next_budget = min(int(attempt_budget or 0) * 2, _MAX_RETRY_TOKENS)
            if next_budget <= int(attempt_budget or 0):
                break
            logger.warning(
                "LLM 输出被推理内容耗尽（finish_reason=length，content 为空），"
                "提高 max_tokens %s → %s 重试（第 %s 次）",
                attempt_budget, next_budget, attempt + 1,
            )
            attempt_budget = next_budget

        usage = data.get("usage") or {}
        self.record_usage(usage)
        return LLMResponse(content=content or "", model=data.get("model", self.model), usage=usage, raw=data)

    # ------------------------------------------------------------------
    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._breaker.before_call()
        last_exc: Exception | None = None
        for attempt in range(_MAX_NETWORK_RETRIES + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < _MAX_NETWORK_RETRIES:
                    delay = _RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        "LLM 请求失败（%s），第 %s 次重试，%.1fs 后重试：%s",
                        type(exc).__name__, attempt + 1, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                break

            if resp.status_code >= 400:
                body = resp.text[:500]
                # ★ 欠费 / 配额耗尽必须**立即**转成一次「可降级的 LLMException」：
                # 它和 429 限流共用状态码，但重试永远等不到成功，只会把一次失败
                # 放大成 _MAX_NETWORK_RETRIES+1 次请求 + 指数退避的空等。
                if _is_quota_error(resp.status_code, body):
                    self._breaker.record_failure()
                    raise LLMException(
                        _quota_message(resp.status_code, body),
                        details={"status": resp.status_code, "body": body},
                    )
                last_exc = LLMException(
                    f"LLM 返回错误状态 {resp.status_code}",
                    details={"body": body},
                )
                if _is_retryable_status(resp.status_code) and attempt < _MAX_NETWORK_RETRIES:
                    # 服务端给了 Retry-After 就以它为准；否则才用指数退避。
                    delay = _retry_after_seconds(resp)
                    if delay is None:
                        delay = _RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        "LLM 返回状态 %s，第 %s 次重试，%.1fs 后重试",
                        resp.status_code, attempt + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                break

            self._breaker.record_success()
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMException("LLM 响应不是合法 JSON", details={"body": resp.text[:300]}) from exc

        # 走到这里说明重试耗尽或遇到不可重试错误。
        self._breaker.record_failure()
        if isinstance(last_exc, LLMException):
            raise last_exc
        raise LLMException(f"LLM 请求失败：{last_exc}") from last_exc

    @staticmethod
    def _message_of(data: dict[str, Any]) -> dict[str, Any]:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMException("LLM 响应格式异常（缺少 choices[0].message）") from exc
        return message or {}

    @classmethod
    def _content_of(cls, data: dict[str, Any]) -> str:
        return (cls._message_of(data).get("content") or "").strip()

    @classmethod
    def _reasoning_of(cls, data: dict[str, Any]) -> str:
        return (cls._message_of(data).get("reasoning_content") or "").strip()

    @staticmethod
    def _finish_reason_of(data: dict[str, Any]) -> str:
        try:
            return str(data["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, TypeError):
            return ""
