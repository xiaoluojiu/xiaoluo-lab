"""LLM 客户端：OpenAI 兼容协议的单一实现。

设计取舍
--------
整个平台**只有一个** LLM 调用点。

旧架构把它拆成 base / capabilities / local / mock / openai_compatible /
structured / usage 七个子模块，新增一个调用场景要同时改好几处，
且「本地生成模型」「结构化输出」各有各的降级分支。这里收敛为一个文件：
一个 Provider 类、一个 chat 方法、一个 usage 记录。

健壮性（三条硬约束，缺一不可）
------------------------------
1. **显式超时**：默认 30 秒，调用方可按场景收紧（连通性测试传 20 秒）。
2. **有界重试**：默认 2 次，指数退避 + 抖动。**只对可重试错误重试** ——
   连接失败 / 超时 / 429 / 5xx。把 401 / 403 / 400 也放进重试，
   只会让「Key 填错了」变成「卡 30 秒后报超时」，排查成本翻倍。
3. **降级由调用方决定**：本模块只负责如实抛出 ``LLMException``，
   绝不伪造回答。「LLM 不可用时改用规则回答」的策略在 Agent 引擎里，不在这里。

凭据
----
API Key 只从入参读取（最终来源是环境变量或 .env），
绝不写进日志、异常 details 或 ``capability_snapshot()``。
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from app.core.exceptions import AppException

logger = logging.getLogger(__name__)

#: 可重试的 HTTP 状态。这个集合刻意很小 —— 重试是把故障变慢，不是把故障变没。
_RETRIABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 2
_BACKOFF_BASE = 0.5


def _thinking_payload(model: str, base_url: str) -> dict[str, Any]:
    """推理型厂商的「关闭思考模式」请求体；不支持的厂商返回空 dict。

    为什么要显式关闭
    ----------------
    ``deepseek-flash`` / ``deepseek-reasoner`` 默认开启 thinking：
    模型把思维链放进 ``reasoning_content``，最终答案放进 ``content``。
    在 ``max_tokens`` 较小的抽取/润色场景里，token 常常**全被思维链吃掉**，
    于是 ``content`` 是空字符串。此时若按旧逻辑回退到 ``reasoning_content``，
    用户看到的最终答案就是一段模型的自言自语
    （实测出现过以「我们需要回答中文，基于事实摘要。用户要求全面分析：」开头的回答）。

    官方支持在请求体里关闭：``{"thinking": {"type": "disabled"}}``。
    只认 deepseek（模型名或 base_url 命中）—— 给不支持该字段的厂商发过去
    会变成一次 400，而这个 400 不重试、直接暴露，排查成本极高。
    """
    text = f"{model or ''} {base_url or ''}".lower()
    if "deepseek" not in text:
        return {}
    return {"thinking": {"type": "disabled"}}


class LLMException(AppException):
    """LLM 调用失败。

    ``details`` 里带 HTTP 状态与**响应体片段**，供设置页展示厂商返回的真实原因。
    响应体可能含敏感内容，因此只取前 2000 字符，且绝不携带请求头。
    """

    http_status = 502
    default_code = "LLM_CALL_FAILED"
    default_message = "大模型调用失败"


@dataclass(frozen=True)
class LLMMessage:
    """一条对话消息。"""

    role: str
    content: str


@dataclass(frozen=True)
class LLMUsage:
    """单次调用的 Token 用量。厂商未返回时回退为 0（不做假估算）。

    真实用量以厂商返回的 ``usage`` 为准；本地估算只用于**展示节省量**，
    不能当作计费依据，因此 :meth:`LLMResponse.estimated` 显式区分二者。
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResponse:
    """单次调用结果。

    :attr:`from_reasoning` 标明 ``content`` 是不是**退而求其次**拿到的思维链。
    这个标记不是装饰：上层要靠它判断「这次调用其实没拿到正经答案」。
    少了它，重试保护会被自己骗过去（见 ``_request_with_retry`` 的注释）。
    """

    content: str
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: True = content 是思维链回退产物，不是面向用户的答案
    from_reasoning: bool = False


class LLMProvider(ABC):
    """Provider 抽象。新增厂商只需实现 :meth:`chat`。"""

    name: str = "unknown"
    model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """发起一次对话。失败抛 :class:`LLMException`。"""

    def capability_snapshot(self) -> dict[str, Any]:
        """可安全展示的能力摘要（不含凭据）。"""
        return {"provider_type": self.name, "model": self.model}


def _sleep_backoff(attempt: int) -> None:
    """指数退避 + 抖动。抖动是为了避免多个并发调用同时重试形成尖峰。"""
    delay = _BACKOFF_BASE * (2**attempt)
    time.sleep(delay * (1.0 + random.uniform(-0.3, 0.3)))  # noqa: S311 - 抖动不需要密码学随机


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions 兼容协议（httpx 直连，不绑定厂商）。"""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        client: httpx.Client | None = None,
        thinking: str | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        # 凭据只存实例字段，不参与 repr / 日志 / 快照
        self._api_key = api_key or ""
        self.timeout = float(timeout) if timeout else _DEFAULT_TIMEOUT
        self.max_retries = max(0, int(max_retries))
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        # None = 自动（推理型厂商关闭思考）；"enabled" = 保留厂商默认（要思维链时用）
        self.thinking = thinking
        self._client = client

    # ---- 公开接口 ----------------------------------------------------
    @property
    def api_key_set(self) -> bool:
        return bool(self._api_key)

    def capability_snapshot(self) -> dict[str, Any]:
        return {
            "provider_type": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "api_key_set": self.api_key_set,
        }

    def with_thinking(self, thinking: str | None) -> "OpenAICompatibleProvider":
        """返回一份**仅 thinking 设置不同**的副本。

        为什么默认关、但又要留一个切换口
        --------------------------------
        默认关闭是为了防「思维链吃掉全部 max_tokens，content 变空」——
        实测过用户看到以「我们需要回答中文，基于事实摘要」开头的自言自语。

        但反过来，**多步数值计算恰恰需要推理空间**。实测 10 个工资求平均：
        - 关闭 thinking：14400.00 / 14740.00 / 14218.18（三个不同轮次，全错，真值 15440）
        - 开启 thinking：15440.00（对）

        所以正确策略不是「全局开」或「全局关」，而是**按任务开**：
        普通问答关掉（省 token、防污染），遇到聚合计算打开（要算得准）。
        本方法让调用方做这个切换，而不是把策略硬编码在 Provider 里。

        副本共享同一个 httpx 客户端，不额外占连接。
        """
        return OpenAICompatibleProvider(
            self.base_url,
            self.model,
            self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            client=self._client,
            thinking=thinking,
        )

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        if not self.model:
            raise LLMException("未配置模型名", code="LLM_MODEL_MISSING")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if response_format:
            payload["response_format"] = response_format
        if self.thinking != "enabled":
            payload.update(_thinking_payload(self.model, self.base_url))

        return self._request_with_retry(payload, timeout=float(timeout) if timeout else self.timeout)

    # ---- 内部 --------------------------------------------------------
    def _client_(self) -> httpx.Client:
        return self._client or httpx.Client()

    def _request_with_retry(self, payload: dict[str, Any], *, timeout: float) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if attempt:
                _sleep_backoff(attempt - 1)
            try:
                client = self._client_()
                owns_client = self._client is None
                try:
                    response = client.post(url, json=payload, headers=headers, timeout=timeout)
                finally:
                    if owns_client:
                        client.close()

                status = response.status_code
                if status in _RETRIABLE_STATUS:
                    last_exc = self._http_error(status, response.text)
                    logger.warning(
                        "LLM 返回可重试状态 %s（第 %s 次）：%s",
                        status, attempt + 1, _clip(response.text),
                    )
                    continue
                if status >= 400:
                    # 4xx 业务错误（401 Key 错 / 403 无权限 / 400 参数错）不重试：
                    # 重试只会把「立刻能看懂的原因」拖成一次超时。
                    raise self._http_error(status, response.text)
                parsed = self._parse(response)
                # 兜底：content 为空说明模型把 token 全花在思维链上了。
                # 关掉 thinking 再要一次 —— 直接返回空会让上层静默退回模板，
                # 用户看到的是「回答总是那几句」而不是「这次没答上来」。
                # ★ 这里必须同时看 ``from_reasoning``。
                #   只判断 ``content`` 为空是不够的：``_parse`` 已经把思维链填进了 content，
                #   于是 content 非空、这个分支永远不成立，「关闭 thinking 重试」的保护
                #   形同虚设 —— 用户拿到的就是一段模型的自言自语。
                #   实测：穷举类问题开启推理链后推理过程吃掉全部额度，
                #   M19/M20 的回答直接变成思维链（相关性从 25 掉到 5）。
                if (not parsed.content.strip() or parsed.from_reasoning) and not payload.get("thinking"):
                    disabled = _thinking_payload(self.model, self.base_url)
                    if disabled:
                        logger.warning(
                            "LLM 未返回正经 content（第 %s 次），关闭 thinking 重试", attempt + 1
                        )
                        payload = {**payload, **disabled}
                        continue
                return parsed
            except LLMException:
                # 业务异常直接上抛，不参与重试（下面的分支只处理网络/解析类错误）。
                raise
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("LLM 调用超时（第 %s 次，timeout=%.1fs）", attempt + 1, timeout)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("LLM 连接失败（第 %s 次）：%s", attempt + 1, exc)
            except ValueError as exc:
                # 响应不是合法 JSON：网关返回 HTML 错误页时常见，重试无意义
                raise LLMException(f"大模型返回了无法解析的响应：{exc}") from exc

        raise LLMException(
            f"大模型调用失败（已重试 {self.max_retries} 次）：{last_exc}",
            details={"retries": self.max_retries, "timeout": timeout},
        ) from last_exc

    @staticmethod
    def _http_error(status: int, body: str) -> LLMException:
        # 只带状态码与截断后的响应体；headers 含 Authorization，绝不入 details。
        return LLMException(
            f"大模型服务返回 HTTP {status}",
            code="LLM_HTTP_ERROR",
            details={"status": status, "body": body[:2000]},
        )

    @staticmethod
    def _parse(response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMException(f"大模型返回了非 JSON 响应：{exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMException("大模型返回了空的 choices", details={"body": _clip(response.text)})
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        from_reasoning = False
        if not content:
            content = message.get("reasoning_content") or ""
            if content:
                from_reasoning = True
                logger.warning("LLM 返回空 content，已回退到 reasoning_content（思维链）")
        raw_usage = data.get("usage") or {}
        usage = LLMUsage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
        )
        return LLMResponse(
            content=str(content),
            model=str(data.get("model") or ""),
            usage=usage,
            from_reasoning=from_reasoning,
        )


def _clip(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def build_default_provider() -> OpenAICompatibleProvider | None:
    """按当前配置构造 Provider；不可用返回 None（调用方负责降级）。

    「总开关关闭」与「未配置 Key」都返回 None —— 两者的处置完全相同，
    不需要在调用方再区分。
    """
    from app.core.config import settings

    if not settings.remote_llm_available():
        return None
    return OpenAICompatibleProvider(
        settings.LLM_BASE_URL,
        settings.LLM_MODEL,
        settings.LLM_API_KEY,
        context_window=settings.LLM_CONTEXT_WINDOW,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    )
