# AI 实验室核心运行问题修复（P0 稳定性 + 容错闭环 + 回归测试）

> 本轮为**稳定性修复与容错闭环修复**，不是新功能开发。范围严格控制在：Effect 生命周期、SSE 不被误杀、本地兜底闭环、Planner 降级、429/quota 识别、来源语义、回归测试。
> 未重写 Agent Runtime / AI 页面 / 状态管理框架 / 网络层，未引入 Redis / Celery / 消息队列 / 第二套执行引擎。

---

## ① 根因

### P0-1　Effect 生命周期错误（AI 不回应 / 运行消失 / SSE 中断 / Inspector 页签锁死概览）

| # | 根因 | 位置 | 表现 |
|---|---|---|---|
| 1 | 会话镜像 Effect 的依赖数组里放了 `onRunRestored`，而 `AI/index.tsx` 传的是**内联箭头函数** `() => setPanelOpen(true)`。每次 render 都产生新引用 ⇒ Effect 每次 render 都重跑一次 | `useAgentRun.ts` 会话镜像 Effect + `pages/AI/index.tsx` | Effect 体里的 `stopPolling() / abortStream() / clear() / setRun(null)` 被反复执行 ⇒ **运行凭空消失、SSE 被自己 abort、请求发出去没有响应** |
| 2 | 同一 Effect 体里还无条件 `setInspectorTab("overview")`；另有 `refreshRun()` 与 `applyEffects()` 两条路径也会改写页签 | `useAgentRun.ts` | 用户切到「活动 / 工具链 / 产物」后，任意一次 rerender（收到 SSE 事件、usage 更新、父组件渲染）就被打回「概览」⇒ **页签锁死** |
| 3 | `startPolling()` 的 `await getRun(...)` 之后没有代次校验 | `useAgentRun.ts` | 快速切会话时，旧会话的轮询结果后到并覆盖新会话状态 ⇒ 状态回退 |

**修复后的纪律**：会话镜像 Effect 的依赖数组**只允许 `switchToken`**（真正切会话/新建/回退时才变）；所有回调与 `lastRunId` 一律通过 ref 读取最新值；「会话切换」与「恢复完成后通知页面」两个职责分离。

### P0-2　普通聊天远程失败没有真的降级（报错文案 + `LLM_ERROR_FALLBACK` 自相矛盾）

`_direct_chat()` 的 `except` 分支捕获远程异常后，**直接返回硬编码的「暂时无法完成对话请求」**，却把 `run.answer_source` 标成 `LLM_ERROR_FALLBACK`。结果是：用户看到的是一句报错，系统记录的却是「已降级为本地回答」——来源在撒谎，兜底闭环根本没走。

**修复后**：远程失败 → 先判 `fallback_allowed(exc)` → 真的调 `local_reply(utterance, degraded=True)` → 有回答标 `LLM_ERROR_FALLBACK`，没有回答标 `NO_ANSWER` 并给 `no_llm_notice(reason=真实原因)`。

### P1　容错与配置层面的四个缺口

| # | 缺口 | 修复 |
|---|---|---|
| 1 | `build_plan_resilient()` 只捕获 `PlanInvalidError`，`LLMException` 直接冒泡 ⇒ 数据任务在远程挂掉时整体失败，永远到不了 Rule Planner | 捕获 `(PlanInvalidError, LLMException)`，按「候选工具集 → 全量工具集」两轮后落 Rule Planner，置 `planner_fallback=True` |
| 2 | `_is_retryable_status()` 只有 `status >= 500 or status == 429` ⇒ 欠费型 429 会傻傻重试到超时 | 解析 `error.code/type/message` 与 HTTP 402：quota / billing / insufficient_quota / balance / credit 直接转可降级的 `LLMException` **一次都不重试**；真瞬时限流才有限重试并遵守 `Retry-After` |
| 3 | `AGENT_ALLOW_MODEL_FALLBACK` 默认 `False`，且 `settings.py` 里被**写死**成 `False`，前端只能显示不能写 ⇒ 假配置 | 默认改为 `True`；接口新增 `allow_model_fallback` 可写；planner / runtime 真正读取它作为总闸 |
| 4 | 缺少回归测试，上述行为没有任何东西兜住 | 后端 +22 项、前端 +7 项，含 React 生命周期与 SSE abort 计数 |

---

## ② 修改文件列表

**后端（11）**
- `backend/app/agent/llm/base.py`
- `backend/app/agent/llm/openai_compatible.py`
- `backend/app/agent/planner/models.py`
- `backend/app/agent/planner/planner.py`
- `backend/app/agent/runtime/runtime.py`
- `backend/app/agent/local_chat.py`
- `backend/app/agent/answer_source.py`
- `backend/app/core/config.py`
- `backend/app/api/v1/settings.py`
- `backend/tests/test_agent_llm_fallback.py`
- `backend/tests/test_answer_source.py`

**前端（11）**
- `frontend/src/features/agent/hooks/useAgentRun.ts`
- `frontend/src/pages/AI/index.tsx`
- `frontend/src/lib/agentEvents.ts`
- `frontend/src/api/settings.ts`
- `frontend/src/pages/Settings/AgentModelPanel.tsx`
- `frontend/tests/useAgentRun.test.ts`
- `frontend/tests/dom.mjs`
- `frontend/tests/bootstrap.mjs`
- `frontend/tests/resolve-hook.mjs`
- `frontend/tests/agentEvents.test.ts`
- `frontend/package.json`

---

## ③ 每个改动文件的完整修改后代码

> 以下为**修改后文件的完整内容**，不是 diff。


### 1. `backend/app/agent/llm/base.py`

- 新增 `LLMException.fallbackable`：区分「可降级的远程故障」与「不该降级的程序 bug / 权限错误」。
- 新增 `_BUG_ERRORS`（TypeError / AttributeError / NameError / ImportError / NotImplementedError / AssertionError / RecursionError …）判定清单。
- 新增 `is_fallbackable_error(exc)`（错误分类 SSOT）与 `fallback_allowed(exc)`（总闸 × 错误分类）。

````python
"""LLMProvider 接口与 Token 预算追踪。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from app.agent.llm.capabilities import LLMCapabilities
from app.agent.llm.usage import LLMUsage
from app.core.config import settings
from app.core.exceptions import AppException


class LLMException(AppException):
    """LLM 调用 / 解析失败。

    `fallbackable` 是这次失败**是否允许退回平台内置规则**的显式声明：
    Provider 把「连接失败 / 超时 / 4xx / 5xx / 配额不足 / 响应格式异常」统一封装成
    本异常，这些都属于「远程这一侧这次不可用」，默认允许降级；只有 Provider
    自己判定「降级没有意义」（例如熔断期短路）时才显式置 False。
    """

    http_status = 502
    default_code = "LLM_ERROR"
    default_message = "LLM provider error"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: Any = None,
        fallbackable: bool = True,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.fallbackable = fallbackable


# 「这类异常几乎一定是代码缺陷，不是远程不可用」——出现在这里说明调用栈写错了，
# 拿内置规则去兜底只会把 bug 伪装成一次降级成功的回答。
_BUG_ERRORS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    UnboundLocalError,
    SyntaxError,
    IndentationError,
    ImportError,
    NotImplementedError,
    AssertionError,
    RecursionError,
)


def is_fallbackable_error(exc: BaseException) -> bool:
    """这次失败是否允许「远程 LLM → 平台内置规则」降级。

    判据是**错误发生在哪一侧**，而不是笼统地「出错了就兜底」：

    - `LLMException`：Provider 统一封装的远程侧失败（连接 / 超时 / 429 / 5xx /
      配额 / 结构化输出 / 响应格式），默认可降级；
    - 其它 `AppException`（工具执行 / 权限 / 数据集 / 参数校验）：不是「远程不可用」，
      兜底没有意义，**不降级**；
    - 代码缺陷类异常（`TypeError` 等）：不降级，必须暴露出来；
    - 其余（如测试替身或未封装的 `RuntimeError("连接超时")`）：只在
      `llm.chat()` 这一次调用的边界内出现，语义仍是「远程这次没产出内容」，
      允许降级。

    ★ 这条函数只在 LLM 调用边界被调用。工具执行 / 校验 / 持久化发生在别的调用栈，
    那里的异常照常向上抛 —— 不会因为「兜底」被吞成一段似是而非的回答。
    """
    if isinstance(exc, LLMException):
        return bool(getattr(exc, "fallbackable", True))
    if isinstance(exc, AppException):
        return False
    if isinstance(exc, _BUG_ERRORS):
        return False
    return True


def fallback_allowed(exc: BaseException) -> bool:
    """总闸 `AGENT_ALLOW_MODEL_FALLBACK` 与「这次错误是否可降级」的合取。

    单独拆出来是为了让「开关」与「错误分类」两件事都能被单独验证：
    开关关着 ⇒ 任何远程失败都直接失败；开关开着也只降级**该降的那几类**。
    """
    from app.core.config import settings

    return bool(settings.AGENT_ALLOW_MODEL_FALLBACK) and is_fallbackable_error(exc)


@dataclass
class LLMMessage:
    """对话消息。role: system / user / assistant。"""
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """一次对话调用的返回。"""
    content: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """LLM 提供方抽象；Agent 只依赖能力与统一 usage，不依赖具体厂商。"""

    name: str = "base"
    capabilities = LLMCapabilities()

    def __init__(self) -> None:
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._usage_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
            f"llm_usage_sink_{id(self)}", default=None
        )
        self._budget_checker: ContextVar[Callable[[], None] | None] = ContextVar(
            f"llm_budget_checker_{id(self)}", default=None
        )

    @contextmanager
    def capture_usage(
        self,
        sink: Callable[[dict[str, Any]], None],
        *,
        budget_checker: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        """把当前调用上下文中的真实 usage 转发给一个 Agent Run，并使用 Run 级预算。"""
        usage_token = self._usage_sink.set(sink)
        budget_token = self._budget_checker.set(budget_checker)
        try:
            yield
        finally:
            self._budget_checker.reset(budget_token)
            self._usage_sink.reset(usage_token)

    def begin_call(self) -> None:
        """真实请求前检查当前 Run 的调用预算；无 Run 时才使用 Provider 累计预算。"""
        checker = self._budget_checker.get()
        if checker is not None:
            checker()
            self._usage["calls"] += 1
            return
        if self._usage["calls"] >= settings.AGENT_LLM_MAX_CALLS:
            raise LLMException(
                "Agent LLM 调用次数已达到预算上限",
                details={"max_calls": settings.AGENT_LLM_MAX_CALLS},
            )
        if self._usage["total_tokens"] >= settings.AGENT_LLM_MAX_TOTAL_TOKENS:
            raise LLMException(
                "Agent LLM Token 预算已耗尽",
                details={"max_total_tokens": settings.AGENT_LLM_MAX_TOTAL_TOKENS},
            )
        self._usage["calls"] += 1

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """累计 Provider usage，同时把本次真实 usage 通知当前 Agent Run。"""
        normalized = LLMUsage.from_raw(usage, provider=self.name).to_dict()
        self._usage["prompt_tokens"] += normalized["input_tokens"]
        self._usage["completion_tokens"] += normalized["output_tokens"]
        self._usage["total_tokens"] += normalized["total_tokens"]
        sink = self._usage_sink.get()
        if sink is not None:
            sink(normalized)

    def usage_snapshot(self) -> dict[str, int]:
        """返回当前 Provider 的累计使用量（用于诊断，不作为 Run 账本）。"""
        return dict(self._usage)

    def capability_snapshot(self) -> dict[str, Any]:
        return self.capabilities.to_dict()

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """对话补全。"""

    def structured_output(
        self,
        messages: list[LLMMessage],
        schema: type,
        *,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """带重试的结构化输出。"""
        from app.agent.llm.structured import structured_output_with_retry
        return structured_output_with_retry(self, messages, schema, max_attempts=max_attempts, **kwargs)
````

### 2. `backend/app/agent/llm/openai_compatible.py`

- 新增配额类错误识别：`_QUOTA_HINTS`（insufficient_quota / quota / billing / balance / credit / payment / arrears / 欠费 / 余额不足 / 账户余额）+ HTTP 402。
- 新增 `_error_envelope(body)` 解析 OpenAI 兼容协议 `{"error": {...}}`，从 `code` / `type` / `message` 三个字段一起嗅探。
- 配额/欠费类 429 直接抛不可重试、可降级的 `LLMException`，一次都不重试；真正的瞬时限流才有限重试，并且优先遵守 `Retry-After`，没有则指数退避。

````python
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
````

### 3. `backend/app/agent/planner/models.py`

- `AgentPlan` 新增 `planner_fallback: bool` 字段，标记「这份计划是规则规划器降级产出的」，供运行时与前端展示真实来源。

````python
"""Prompt 123：PlanStep / AgentPlan。

计划是纯数据结构：goal + steps。
每个 PlanStep 只声明：执行哪个 tool、参数、期望输出、所需权限。
Plan 不携带任何执行能力 —— 执行只能发生在 AgentExecutor。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 工具名规范：category.action（小写字母数字下划线 + 一个点）
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class PlanStep(BaseModel):
    """计划中的一步。"""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    permission: str = ""

    @field_validator("tool")
    @classmethod
    def _tool_name_valid(cls, value: str) -> str:
        value = value.strip()
        if not TOOL_NAME_RE.match(value):
            raise ValueError(f"非法工具名 {value!r}（必须形如 category.action）")
        return value

    @field_validator("arguments")
    @classmethod
    def _arguments_is_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("arguments 必须是对象")
        # 禁止把可执行代码塞进参数（防御式：参数只能是数据）
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("arguments 的键必须是字符串")
            if callable(item) or isinstance(item, (type,)):
                raise ValueError(f"参数 {key!r} 不能是可执行对象")
        return value


class AgentPlan(BaseModel):
    """Agent 计划。"""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    notes: str = ""  # Replanner 分析备注等
    retry: bool = False  # Replanner 显式标记：本计划首步是否为失败步骤的原样重试
    cache_hit: bool = False  # Planner 标记：本计划是否来自 plan cache（不参与 LLM 输出解析）
    planner_fallback: bool = False  # Planner 标记：本计划是否由「规则规划器」降级产出（远程规划失败）

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("goal 不能为空")
        return value.strip()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
````

### 4. `backend/app/agent/planner/planner.py`

- `build_plan_resilient()` 原来只捕获 `PlanInvalidError`，远程 LLM 抛 `LLMException` 时直接冒泡 ⇒ 数据分析任务整体失败。
- 改为捕获 `(PlanInvalidError, LLMException)`，按「候选工具集 → 全量工具集」两轮尝试，最终落到 Rule Planner 并置 `plan.planner_fallback = True`。
- 新增 `_fallback_allowed(exc)` = 总闸 `AGENT_ALLOW_MODEL_FALLBACK` × 错误分类；程序 bug 与不可降级异常原样抛出，绝不静默吞掉。
- 一次 run 内远程最多尝试 2 次（两轮 scope），规则规划是终点，不存在循环 fallback。

````python
"""Agent Planner。"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

from pydantic import ValidationError

from app.agent.context.budget import ContextBudget
from app.agent.context.models import AgentContext
from app.agent.intent import Intent, wants_modeling
from app.agent.llm.base import LLMException, LLMProvider, is_fallbackable_error
from app.agent.planner.models import AgentPlan, PlanStep
from app.core.config import settings
from app.core.exceptions import AgentException

logger = logging.getLogger(__name__)


class PlanInvalidError(AgentException):
    """计划非法。"""
    http_status = 400
    default_code = "AGENT_PLAN_INVALID"
    default_message = "Agent plan invalid"


class AgentPlanner:
    """只负责规划，不读取数据、不执行工具。"""

    def __init__(self, llm: LLMProvider | None = None, *, max_steps: int = 6) -> None:
        self.llm = llm
        self.max_steps = max_steps
        # 进程级共享（deps 注入单例）后存在多线程并发访问，plan cache 必须加锁。
        self._plan_cache: OrderedDict[str, AgentPlan] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.plan_cache_hits = 0
        self.plan_cache_misses = 0

    def build_plan(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> AgentPlan:
        if self.llm is not None:
            cache_key = self._cache_key(user_request, context, tools)
            if settings.AGENT_ENABLE_PLAN_CACHE:
                cached = self._cache_get(cache_key)
                if cached is not None:
                    with self._cache_lock:
                        self.plan_cache_hits += 1
                    cached.cache_hit = True
                    return self._validate(cached.model_copy(deep=True), tools, context)
                with self._cache_lock:
                    self.plan_cache_misses += 1
            plan = self._llm_plan(user_request, context, tools)
            plan = self._validate(plan, tools, context)
            if settings.AGENT_ENABLE_PLAN_CACHE:
                self._cache_put(cache_key, plan)
            return plan
        return self._validate(self._rule_plan(user_request, context), tools, context)

    def build_plan_resilient(
        self,
        user_request: str,
        context: AgentContext,
        tools: list[dict[str, Any]],
        *,
        all_tools: list[dict[str, Any]] | None = None,
    ) -> AgentPlan:
        """带降级的规划入口：**远程规划不可用时必须真的降到规则规划**。

        历史缺陷：这里只捕获 `PlanInvalidError`，而远程规划最常见的失败是
        `LLMException`（连接失败 / 超时 / 429 / 5xx / 配额不足 / 结构化输出解析失败）。
        于是「账户欠费」这类场景下整次数据分析请求直接失败，界面上还显示
        「工具调用失败」—— 用户完全看不出真实原因是远程大模型不可用。

        现在的三级降级：
        1. 候选工具集规划；
        2. 失败则放开到全量工具集重新规划（工具已注册 ≠ 被召回，消除 PlanInvalidError）；
        3. 再失败则用规则规划兜底，保证数据分析请求至少能跑出真实结果。

        ★ 只允许**可降级**的失败走到规则规划：
          - `PlanInvalidError`（LLM 给的计划不可用）
          - `LLMException`（远程这一侧不可用）
        程序 bug（`TypeError` 等）、工具执行异常、权限异常**照常向上抛** ——
        用规则规划去兜一个代码缺陷，只会把 bug 伪装成「降级成功」。

        ★ 一次 `build_plan_resilient` 最多走到规则规划一次，且规则计划是终点
        （不会再回头试远程），因此不存在「远程 ↔ 规则」的降级循环。
        """
        if self.llm is None:
            # 本来就没有远程模型 ⇒ 规则规划是正常路径，谈不上「降级」。
            return self._validate(self.rule_plan(user_request, context), all_tools or tools, context)

        scopes: list[tuple[list[dict[str, Any]], str]] = [(tools, "候选工具集")]
        if all_tools and len(all_tools) > len(tools):
            scopes.append((all_tools, "全量工具集"))

        last_error: BaseException | None = None
        for scope, label in scopes:
            try:
                return self.build_plan(user_request, context, scope)
            # 只认领「远程规划没产出可用计划」这一类；其它异常原样抛出。
            except (PlanInvalidError, LLMException) as exc:
                last_error = exc
                logger.warning("远程规划失败（%s）：%s：%s", label, type(exc).__name__, exc)
                if not self._fallback_allowed(exc):
                    raise

        # 三级：规则规划兜底。一次调用最多到这里一次，且是终点。
        plan = self._validate(self.rule_plan(user_request, context), all_tools or tools, context)
        plan.planner_fallback = True
        logger.warning(
            "远程规划不可用，已降级到规则规划（原因：%s）",
            type(last_error).__name__ if last_error else "未知",
        )
        return plan

    def _fallback_allowed(self, exc: BaseException) -> bool:
        """这次规划失败是否允许降到规则规划器。

        两道门：总闸 `AGENT_ALLOW_MODEL_FALLBACK`（关着就一律直接失败）
        ×「这次错误是不是远程不可用」（见 `is_fallbackable_error`）。
        """
        from app.core.config import settings

        if not settings.AGENT_ALLOW_MODEL_FALLBACK:
            return False
        return isinstance(exc, PlanInvalidError) or is_fallbackable_error(exc)

    def rule_plan(self, user_request: str, context: AgentContext) -> AgentPlan:
        """公开的规则规划入口（降级兜底时使用）。"""
        return self._rule_plan(user_request, context)

    def _cache_get(self, key: str) -> AgentPlan | None:
        with self._cache_lock:
            cached = self._plan_cache.get(key)
            if cached is not None:
                self._plan_cache.move_to_end(key)
            return cached

    def _cache_put(self, key: str, plan: AgentPlan) -> None:
        with self._cache_lock:
            self._plan_cache[key] = plan.model_copy(deep=True)
            self._plan_cache.move_to_end(key)
            while len(self._plan_cache) > settings.AGENT_PLAN_CACHE_MAX_ITEMS:
                self._plan_cache.popitem(last=False)

    def _context_budget(self) -> ContextBudget:
        return ContextBudget(
            max_chars=settings.AGENT_CONTEXT_MAX_CHARS,
            user_request=settings.AGENT_CONTEXT_USER_REQUEST_CHARS,
            dataset=settings.AGENT_CONTEXT_DATASET_CHARS,
            task=settings.AGENT_CONTEXT_TASK_CHARS,
            permissions=settings.AGENT_CONTEXT_PERMISSION_CHARS,
            tools=settings.AGENT_CONTEXT_TOOL_CHARS,
            history=settings.AGENT_CONTEXT_HISTORY_CHARS,
            history_messages=settings.AGENT_CONTEXT_HISTORY_MESSAGES,
        )

    def _cache_key(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> str:
        # 缓存键刻意排除会话历史：对同一请求 + 同一工具集 + 同一数据集，
        # 最优计划不变；把历史掺进 key 会导致缓存永远 miss。
        tool_fingerprint = json.dumps(
            [{"name": t.get("name"), "schema": t.get("input_schema", {})} for t in tools],
            ensure_ascii=False, sort_keys=True, default=str,
        )
        provider = getattr(self.llm, "name", "unknown")
        model = getattr(self.llm, "model", "")
        dataset_ids = ",".join(str(x) for x in sorted(context.dataset_ids()))
        raw = "\n".join((provider, model, str(self.max_steps), tool_fingerprint, user_request.strip(), dataset_ids))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_plan(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> AgentPlan:
        text = user_request.lower()
        # 关键词判定统一走 app.agent.intent（唯一真源）：这里再抄一份关键词，
        # 就会出现「规则规划认得、LLM 提示词不认得」的口径漂移。
        from app.agent.intent import hits

        _hits = hits(text)
        plan_intent = []
        if _hits.get(Intent.REPORT):
            plan_intent.append("report.generate")
        if _hits.get(Intent.WORKFLOW):
            plan_intent.append("workflow.build_and_run")
        intent_hint = ""
        if plan_intent:
            intent_hint = f"本次请求的意图工具已被识别为：{'、'.join(plan_intent)}，计划中应当包含它们。"

        system = (
            "你是小洛实验室的数据 Agent Planner。你的职责只有一件事：根据用户请求制定真实工具执行计划。"
            "严禁读取数据、严禁假装已经分析、严禁直接给出数据结论。"
            "你只能使用下面提供的候选工具，不能创造工具名。"
            "需要 dataset_id 的工具，如果上下文只有一个关联数据集，必须填写该 ID；多个数据集时不要猜。"
            "普通聊天不需要工具时输出空 steps。"
            "跨步骤传值必须用结构化引用 {{stepN.字段}}（N 从 1 开始，等于步骤序号），例如 {{step5.target}}、{{step3.workflow_id}}；"
            "禁止把上一步的内容用自然语言复述成参数。"
            + intent_hint +
            "【数据分析任务的标准交付路径】用户要求分析/智能分析/全面分析时，计划必须覆盖："
            "① 读取结构 dataset.inspect（必要时 dataset.schema）；"
            "② 数据质量 dataset.quality；"
            "③ 统计画像 dataset.profile；"
            "④ 分布与关系 eda.describe / eda.correlation；"
            "需要画图时用 eda.visualize（其 column/x/y/columns 必须来自 dataset.schema / dataset.profile 的"
            "真实字段名，不确定就不要画）。"
            "★★ report.generate 的唯一判定标准（不要与其它规则冲突，也不要默认加）："
            "只有用户**明确要求产出报告文件**时（说了「报告 / 汇报 / PDF / 导出」）才把它加进计划，"
            "且只加一次、放在最后。用户只要求「分析 / 统计 / 总结 / 结论 / 洞察 / 给建议」时"
            "**不要**加 report.generate —— 那些要的是一段回答，不是一份文件。"
            "workflow 相关请求请用 workflow.build_and_run 一步完成（内部已包含创建与执行），不要拆成 workflow.create + workflow.run。"
            "建模请求固定顺序：dataset.inspect → dataset.profile → ml.detect_task(infer_target=true, goal=用户诉求原文) → ml.prepare(target={{stepN.target}}) → ml.train(target={{stepN.target}})。"
            "ml.detect_task 会自己按「命名约定 → 诉求语义（goal 与数据集名称）→ 排除日历/时间/标识列后的唯一候选」"
            "推断目标列，并把依据写在 reasons / target_source 里 —— 所以 goal 一定要填用户诉求原文"
            "（例如「预测出发延误」），推断质量取决于它。"
            "只有当 ml.detect_task 回传 needs_target=true（确实无法唯一确定）时，才需要你从 target_candidates 中"
            "依据用户诉求选定目标列，改用 ml.train(target=选定的列, model=\"auto\") 重试；绝不要在监督任务上用聚类代替。"
            "若计划里同时有 ml.prepare / ml.train，直接引用 {{stepN.target}}（N=ml.detect_task 的步号）；"
            "ml.train 自身也具备同一套推断能力，未给 target 时会自动推断并回传 target_inferred。"
            "【信息不足时用 agent.clarify，不要猜】目标列不明、任务类型与列类型矛盾时，"
            "用 agent.clarify 提出结构化问题并把候选列写进 options（禁止自由生成问题文本、"
            "禁止在监督任务上用聚类顶替）；后续步骤用 {{stepN.answer}} 引用用户回答。"
            "确实能从命名约定或诉求语义唯一确定目标列时，不要反问，直接填。"
            "模型选择优先用 model=\"auto\"（按任务类型自动选），只有用户明确要求具体算法时才写模型名。"
            "每个步骤的 expected_output 用不超过 20 字的一句话概括，不要写长句。"
            "输出 JSON：{\"goal\": str, \"steps\": [{\"tool\": str, \"arguments\": object, \"expected_output\": str, \"permission\": str}]}。"
        )
        context_text = context.to_prompt_text(max_chars=settings.AGENT_CONTEXT_MAX_CHARS, budget=self._context_budget())
        tool_specs = json.dumps(
            [
                {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {}),
                    "permission": t.get("permission", ""),
                }
                for t in tools
            ], ensure_ascii=False, default=str,
        )
        user = f"候选工具：\n{tool_specs}\n\n上下文（仅元数据，不含原始数据）：\n{context_text}\n\n用户请求：{user_request}\n最多 {self.max_steps} 步。"
        from app.agent.llm.base import LLMMessage
        data = self.llm.structured_output(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            AgentPlan,
            max_tokens=settings.AGENT_LLM_MAX_OUTPUT_TOKENS,
        )
        try:
            return AgentPlan.model_validate(data)
        except ValidationError as exc:
            raise PlanInvalidError("LLM 计划未通过校验", details={"errors": exc.errors(include_url=False)[:5]}) from exc

    def _rule_plan(self, user_request: str, context: AgentContext) -> AgentPlan:
        text = user_request.lower()
        ds_ids = context.dataset_ids()
        ds_id = ds_ids[0] if len(ds_ids) == 1 else None
        steps: list[PlanStep] = []

        def _args(extra: dict[str, Any] | None = None) -> dict[str, Any]:
            args: dict[str, Any] = {}
            if ds_id is not None:
                args["dataset_id"] = ds_id
            if extra:
                args.update(extra)
            return args

        # 关键词判定统一走 app.agent.intent（与运行时路由、候选工具注入共用一份）。
        # 改之前这里有一套自己的关键词，漏改一处就表现为「走错分支」。
        # ★ 细分说法（质量 / 全面分析）也只能在 intent.py 里定义一份，
        #   本文件不出现任何字面关键词表 —— 否则又变成第二个真源。
        from app.agent.intent import (
            COMPREHENSIVE_KEYWORDS,
            QUALITY_KEYWORDS,
            hits,
            model_hint,
            wants_merge,
        )

        intent_hits = hits(text)

        def _matched(intent: Intent) -> list[str]:
            return intent_hits.get(intent) or []

        wants_train = wants_modeling(text)
        wants_quality = any(k in text for k in QUALITY_KEYWORDS)
        # 「相关」本身就在 EDA 关键词里：从命中的词里取，而不是再抄一遍关键词。
        wants_corr = "相关" in _matched(Intent.EDA)
        wants_workflow = bool(intent_hits.get(Intent.WORKFLOW))
        wants_merge = wants_merge(text)  # noqa: F811 - 同名覆盖为布尔意图标记
        wants_report = bool(intent_hits.get(Intent.REPORT))
        wants_comprehensive = any(k in text for k in COMPREHENSIVE_KEYWORDS)
        wants_eda = bool(intent_hits.get(Intent.EDA))
        # 「评估 / 指标」同属 ML 关键词，复用命中结果而不是另起一份。
        wants_evaluate = any(k in _matched(Intent.ML) for k in ("评估", "效果", "指标", "准确率"))
        # DATA_TRANSFORM 的细分：只挑「不需要列名就能确定」的操作（见下方分支说明）
        transform_hits = _matched(Intent.DATA_TRANSFORM)
        wants_dedupe = any(k in transform_hits for k in ("去重", "重复"))
        wants_missing = any(k in transform_hits for k in ("缺失", "填充")) or any(
            k in text for k in ("空值", "missing", "null")
        )
        wants_column_op = any(k in transform_hits for k in ("筛选", "聚合", "排序"))
        # 注意：这里用到的工具必须在 ContextBuilder.with_tools 的确定性注入集合里，
        # 否则规则规划会引用未被注入的候选工具而被 _validate 拒绝。
        if not ds_ids:
            return AgentPlan(goal=f"需要数据集才能执行：{user_request[:80]}", steps=[])

        if wants_merge and len(ds_ids) >= 2:
            # 多表关联：keys 不填，由 data.merge 按同名列自动推断并写进执行结果的 plan 里。
            # 规划阶段拿不到列名（ContextBuilder 明确禁止读 Schema），以前只能硬猜或整条失败。
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments={"dataset_id": ds_ids[0]}),
                PlanStep(tool="data.merge", arguments={
                    "left_dataset_id": ds_ids[0],
                    "right_dataset_id": ds_ids[1],
                    "join_type": "left" if any(k in text for k in ("左", "保留左", "left")) else "inner",
                }),
            ])
            return AgentPlan(goal=f"关联多张表：{user_request[:80]}", steps=steps[: self.max_steps])

        if wants_workflow:
            # 一条「读数据 → 质量 → 统计 → 汇总」的编排：build_and_run 内部完成创建与执行
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="workflow.build_and_run", arguments={
                    "name": user_request[:24] or "Agent 自动编排",
                    "nodes": [
                        {"id": "n1", "type": "dataset.read", "config": {"dataset_id": ds_id}},
                        {"id": "n2", "type": "data.quality_check", "config": {}},
                        {"id": "n3", "type": "data.statistics", "config": {}},
                        {"id": "n4", "type": "report.summary", "config": {}},
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2"},
                        {"source": "n1", "target": "n3"},
                        {"source": "n2", "target": "n4"},
                        {"source": "n3", "target": "n4"},
                    ],
                }),
            ])
            # 不再固定追加 report.generate：workflow 里的 report.summary 节点已经产出汇总，
            # 用户没要报告文件时不该再落一份 PDF（与全局报告判定保持同一条规则）。
            return AgentPlan(goal=f"编排并执行工作流：{user_request[:80]}", steps=steps[: self.max_steps])

        if wants_train:
            # goal 带上用户诉求原文：ml.detect_task 靠它做语义匹配选目标列
            # （规划阶段看不到列名，诉求是唯一能带过去的语义线索）。
            _goal = user_request[:200]
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="dataset.schema", arguments=_args()),
                PlanStep(tool="dataset.profile", arguments=_args()),
                PlanStep(
                    tool="ml.detect_task",
                    arguments=_args({"infer_target": True, "goal": _goal}),
                ),
                PlanStep(tool="ml.prepare", arguments=_args({"target": "{{step4.target}}"})),
            ])
            # target 不再靠猜：ml.detect_task（第 4 步）会按命名约定与数据集名称推断目标列，
            # ml.train 通过结构化引用消费其输出。
            # ★ model 用 auto 而不是硬编一个监督模型：用户说「选择适合的机器学习模型」
            # （请求里既没有「回归」也没有「分类」）时，旧写法一律给 logistic_regression，
            # 与 detect_task 判出的任务不匹配，被 ml.train 静默换成 kmeans —— 真实事故里
            # 一条延误回归请求最终跑成了聚类。auto 交给任务类型决定，并在结果里留痕。
            model = model_hint(text)
            train_args = _args({"model": model, "target": "{{step4.target}}", "goal": _goal})
            steps.append(PlanStep(tool="ml.train", arguments=train_args))
            # 「训练并评估」是常见说法：训练步自带指标，但显式要求评估时应补 ml.evaluate，
            # 这样链路才闭环（此前规则规划只训练不评估，用户看到的结果少一截）。
            if wants_evaluate:
                steps.append(
                    PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step%d.run_id}}" % len(steps)})
                )
        elif wants_dedupe or wants_missing or wants_column_op:
            # ---- DATA_TRANSFORM：规则规划的确定性路径 -------------------------
            # 去重 / 缺失值处理**不需要列名**（data.clean 对全表生效），可以真实执行。
            # 筛选 / 聚合 / 排序必须知道列名，而规划阶段拿不到 schema（ContextBuilder
            # 明确禁止读数据）：这里绝不臆造列名填进 conditions / group_by —— 那必然
            # 失败，且错误信息会把用户引向「列名填错了」而不是「还没拿到列名」。
            # 改为先取 schema + profile，把真实列名交到结果里，由用户或 LLM 接着做。
            steps.append(PlanStep(tool="dataset.inspect", arguments=_args()))
            if wants_dedupe or wants_missing:
                clean_args: dict[str, Any] = {}
                if wants_missing:
                    # strategy 必须显式给：data.clean 不填会被判「缺少参数」。
                    # 用 `drop`（删掉含空值的行）而不是 mean/median ——
                    # 后者在字符串列上会直接报「只适用于数值列」，而规则规划器
                    # 看不到列类型，猜错了就是一次必然失败的运行。
                    clean_args["missing"] = {"strategy": "drop"}
                if wants_dedupe:
                    clean_args["deduplicate"] = {"keep": "first"}
                steps.append(PlanStep(tool="data.clean", arguments=_args(clean_args)))
            else:
                steps.extend([
                    PlanStep(tool="dataset.schema", arguments=_args()),
                    PlanStep(tool="dataset.profile", arguments=_args()),
                ])
        elif wants_comprehensive:
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="dataset.schema", arguments=_args()),
                PlanStep(tool="dataset.quality", arguments=_args()),
                PlanStep(tool="dataset.profile", arguments=_args()),
            ])
            if wants_corr:
                steps.append(PlanStep(tool="eda.correlation", arguments=_args()))
        elif wants_quality:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="dataset.quality", arguments=_args())])
        elif wants_eda:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="eda.describe", arguments=_args())])
            if wants_corr:
                steps.append(PlanStep(tool="eda.correlation", arguments=_args()))
        else:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="dataset.profile", arguments=_args())])

        # 报告只在用户**明确要求**时才生成。
        #
        # 历史行为：任何数据分析分支都无条件 append report.generate。于是
        # 「帮我看看有没有缺失值」也会顺带产出一份 PDF 写进报告中心 —— 过度交付；
        # 无 LLM 模式下这条固定尾巴还让「最小可用分析链路」永远背着一次重 IO。
        # 判定复用既有 intent 真源（`app.agent.intent` 的 REPORT 关键词），不新增关键词表。
        if wants_report and not any(s.tool == "report.generate" for s in steps):
            steps.append(PlanStep(tool="report.generate", arguments=_args()))
        return AgentPlan(goal=f"完成用户请求：{user_request[:80]}", steps=steps[: self.max_steps])

    def _validate(self, plan: AgentPlan, tools: list[dict[str, Any]], context: AgentContext) -> AgentPlan:
        known = {t["name"]: t for t in tools}
        unknown = [s.tool for s in plan.steps if s.tool not in known]
        if unknown:
            raise PlanInvalidError("计划引用了未被工具检索选中的工具", details={"unknown_tools": unknown, "available": sorted(known)})

        dataset_ids = context.dataset_ids()
        if len(dataset_ids) == 1:
            dataset_id = dataset_ids[0]
            for step in plan.steps:
                schema = known[step.tool].get("input_schema") or {}
                properties = schema.get("properties") or {}
                required = schema.get("required") or []
                if "dataset_id" in properties and "dataset_id" in required and "dataset_id" not in step.arguments:
                    step.arguments["dataset_id"] = dataset_id
        elif len(dataset_ids) > 1:
            for step in plan.steps:
                schema = known[step.tool].get("input_schema") or {}
                required = schema.get("required") or []
                if "dataset_id" in required and "dataset_id" not in step.arguments:
                    raise PlanInvalidError("当前会话关联多个数据集，请明确指定要操作的数据集", details={"tool": step.tool, "dataset_ids": dataset_ids})

        # 步数超限不再整体判非法（LLM 常略超上限，直接失败会让整次运行空手而归）；
        # 截断到上限并保留前缀，保证「先产出结果」而不是「什么都不做」。
        if len(plan.steps) > self.max_steps:
            # report.generate 是交付物，超限时也不能被截掉
            tail_keep: list = []
            for step in reversed(plan.steps):
                if step.tool == "report.generate":
                    tail_keep.insert(0, step)
                    break
            plan.steps = plan.steps[: max(0, self.max_steps - len(tail_keep))] + tail_keep
        return plan
````

### 5. `backend/app/agent/runtime/runtime.py`

- P0-2 根因：`_direct_chat()` 捕获异常后直接返回硬编码的「暂时无法完成对话请求」，却把来源标成 `LLM_ERROR_FALLBACK` ⇒ 用户看到报错文案、系统却声称已降级。
- 改为：可降级则真的调用 `local_reply(..., degraded=True)` 生成回答并标 `LLM_ERROR_FALLBACK`；本地答不上则 `no_llm_notice(reason=...)` 并标 `NO_ANSWER`。
- `_compose_answer()` 走同一套判定；远程返回空内容时若总闸关闭则抛 `LLMException` 而不是假装降级。
- `run()` 新增 `except LLMException` 分支，保证远程故障落为 `FAILED` 且事件流闭合。
- `plan_ready` 事件 payload 透出 `planner_fallback`，前端据此提示。

````python
"""Agent Runtime：对话路由 -> 元数据上下文 -> 工具检索 -> 规划 -> 工具执行 -> 结果汇总。"""
from __future__ import annotations
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
from app.agent.answer_source import (
    LLM_ERROR_FALLBACK,
    NO_ANSWER,
    PLATFORM_RULES_CHAT,
    PLATFORM_RULES_NOTICE,
    PLATFORM_RULES_SUMMARY,
    REMOTE_LLM_CHAT,
    REMOTE_LLM_SUMMARY,
    describe,
)
from app.agent.clarify import ANSWERS_KEY
from app.agent.context.builder import ContextBuilder
from app.agent.context.models import AgentContext
from app.agent.executor.executor import AgentExecutor, ToolCallRecord
from app.agent.intent import GREETINGS, INTENT_KEYWORDS, classify, explain as explain_intent
from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, fallback_allowed
from app.agent.permission.models import ROLE_PERMISSIONS
from app.local_router.contract import Intent
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner, PlanInvalidError
from app.agent.planner.replanner import AgentLimitExceeded, ReplanLimits, Replanner
from app.agent.preflight import PreflightInput, run_preflight
from app.agent.runtime.models import AgentEvent, AgentRun, AgentSession, AgentStore, RunStatus
from app.agent.validator.validator import AgentResultValidator
from app.core.config import settings
from app.core.exceptions import AgentException, ValidationException
from app.data_engine.service import DataEngineService
from app.tools.base import ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.registry import TOOL_REGISTRY, ToolRegistry

EventCallback = Callable[[AgentEvent], None]

# 一次运行允许的事件数上限（终态事件不计），见 `_emit` 里的说明。
MAX_EVENTS_PER_RUN = 2000
_TERMINAL_EVENTS = frozenset({"completed", "failed"})

class AgentRuntime:
    def __init__(self, data_engine: DataEngineService, *, experiment_service: Any | None = None, db: Any | None = None, llm: LLMProvider | None = None, store: AgentStore | None = None, registry: ToolRegistry | None = None, planner: AgentPlanner | None = None, validator: AgentResultValidator | None = None, replanner: Replanner | None = None, limits: ReplanLimits | None = None) -> None:
        self.data_engine = data_engine; self.experiment_service = experiment_service; self.db = db; self.llm = llm; self.store = store or AgentStore()
        from app.tools.builtin import register_builtin_tools
        register_builtin_tools()
        # 未显式传入 planner 时必须沿用配置的步数上限：AgentPlanner 默认 max_steps=6，
        # 直接构造会把「建模 + 评估 + 生成报告」这类长链路计划从尾部截断（计划里只剩到
        # ml.train，评估与报告步骤被静默丢弃），而 deps.py 走的是配置值 12，
        # 于是「服务端跑正常、脚本/测试里跑被截断」这类不一致很难定位。
        default_planner = AgentPlanner(llm, max_steps=settings.AGENT_MAX_STEPS)
        self.registry = registry or TOOL_REGISTRY; self.planner = planner or default_planner; self.validator = validator or AgentResultValidator(); self.replanner = replanner or Replanner(limits); self.limits = limits or ReplanLimits(); self.executor = AgentExecutor(self.registry)

    def _usage_scope(self, run: AgentRun):
        if self.llm is None: return nullcontext()
        return self.llm.capture_usage(run.token_ledger.record_usage, budget_checker=run.token_ledger.check_budget)

    def create_session(self, *, user_id: str = "anonymous", title: str = "", dataset_ids: list[int] | None = None) -> AgentSession:
        ids = list(dict.fromkeys(int(x) for x in (dataset_ids or []) if int(x) > 0))
        return self.store.add_session(AgentSession(id=self.store.next_session_id(), user_id=user_id, title=title or "新会话", dataset_ids=ids))
    def list_sessions(self, user_id: str | None = None, *, include_archived: bool = True) -> list[AgentSession]: return self.store.list_sessions(user_id, include_archived=include_archived)
    def get_session(self, session_id: str) -> AgentSession: return self.store.get_session(session_id)
    def set_session_archived(self, session_id: str, archived: bool) -> AgentSession: return self.store.set_session_archived(session_id, archived)
    def delete_session(self, session_id: str) -> AgentSession: return self.store.delete_session(session_id)
    def get_run(self, run_id: str) -> AgentRun: return self.store.get_run(run_id)

    def run(self, session: AgentSession, user_request: str, *, role: str = "analyst", confirmed: bool = False, plan_override: AgentPlan | None = None, on_event: EventCallback | None = None) -> AgentRun:
        if not user_request or not user_request.strip(): raise ValidationException("用户请求不能为空")
        run = AgentRun(id=self.store.next_run_id(), session_id=session.id, user_id=session.user_id, user_request=user_request.strip())
        self.store.add_run(run)  # add_run 内部已把 run.id 幂等挂到 session.run_ids
        if run.id not in session.run_ids: session.run_ids.append(run.id)
        session.history.append({"role":"user","content":run.user_request}); run.started_at=time.time()
        try:
            with self._usage_scope(run):
                self._set_status(run, RunStatus.PLANNING)
                mode, route_reason, intent = self._route(
                    run.user_request, plan_override=plan_override, has_datasets=bool(session.dataset_ids)
                )
                self._emit(run,"route",{"mode":mode,"reason":route_reason,"intent":explain_intent(intent)},on_event)
                # 开局就发一次账本快照：预算上限是「这次会不会被截断」的关键信息，
                # 只在运行结束后才给出的话，用户无法在过程中判断还能不能继续问。
                self._emit_usage(run,on_event)
                if plan_override is None: self._trace_route(run, session, mode, route_reason)
                if plan_override is None and mode == "chat":
                    self._emit(run,"chat",{"stage":"direct_chat"},on_event); self._direct_chat(run,session,on_event); return run
                self._agent_turn(run, session, intent, confirmed=confirmed, plan_override=plan_override, on_event=on_event)
        except AgentLimitExceeded as exc: self._fail(run,str(exc),on_event)
        except AgentException as exc: self._fail(run,exc.message,on_event)
        # LLM 失败是「远程不可用」，不是「Agent 内部异常」：错误信息要原样给出去，
        # 不能被包成一句 `Agent 运行异常：...` 之后再让用户猜到底是哪里坏了。
        except LLMException as exc: self._fail(run,exc.message,on_event)
        except Exception as exc: self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally: run.finished_at=time.time(); self._trace_outcome(run, session)
        return run

    def _agent_turn(self, run: AgentRun, session: AgentSession, intent, *, confirmed: bool, plan_override: AgentPlan | None, on_event: EventCallback | None) -> None:
        """路由之后、给出结果之前的完整一段（上下文 → Pre-flight → 规划 → 执行）。

        单独抽出来的原因：用户回答反问后要**从 Pre-flight 处继续**（重新规划），
        而不是从头再走一次路由；抽出来才能让「首次运行」与「回答后继续」
        共用同一段代码，避免两处逻辑漂移。
        """
        context=self._build_context(run,session); self._emit(run,"planning",{"stage":"context_ready","data_access":"metadata_only"},on_event)
        # ★ 空数据集是**正常业务状态**：建了数据集还没导入数据时，给一句明确提示就结束，
        # 不进 Pre-flight、不规划、更不把它当成异常把整次运行打成 failed。
        if plan_override is None:
            empty = self._datasets_without_version(context)
            if empty:
                self._no_data_notice(run, session, empty, on_event)
                return
        if plan_override is None and settings.AGENT_PREFLIGHT_ENABLED:
            pre=self._preflight(run,session,intent,context)
            run.preflight=pre.to_dict()
            self._emit(run,"preflight",{"outcome":str(pre.outcome),"checks_run":pre.checks_run,"findings":[f.to_dict() for f in pre.findings],"clarifications":[c.to_dict() for c in pre.clarifications],"resolved":pre.resolved},on_event)
            if pre.needs_user:
                self._await_clarification(run,pre,on_event)
                return
            # Pre-flight 顺带确认下来的事实（如推断出的目标列）直接交给规划器，
            # 避免规划阶段再猜一遍。
            if pre.resolved:
                context.task_context = {**(context.task_context or {}), "preflight": dict(pre.resolved)}
        all_tools=self.registry.list(); self._fill_tools(context,all_tools); candidate_tools=self._candidate_tools(context,all_tools)
        retrieval=context.tool_context.get("retrieval_scores") or {}
        self._emit(run,"planning",{"stage":"tools_retrieved","count":len(candidate_tools),"tools":[t.get("name") for t in candidate_tools],"retrieval":retrieval},on_event)
        plan=self._build_plan(run, context, candidate_tools, all_tools, plan_override)
        run.plan=plan.to_dict(); self._emit(run,"planning",{"stage":"plan_ready","goal":plan.goal,"steps":len(plan.steps),"cache_hit":bool(getattr(plan,"cache_hit",False)),"planner_fallback":bool(getattr(plan,"planner_fallback",False))},on_event)
        # 规划器是本次运行的第一笔真实开销（除非命中 Plan Cache），这里立刻反映到账本上。
        self._emit_usage(run,on_event)
        if not plan.steps: self._direct_chat(run,session,on_event); return
        run.status=RunStatus.RUNNING; self._run_plan(run,session,context,self._role_of(run),plan,offset=0,attempts={},confirmed=confirmed,on_event=on_event)

    def _build_plan(self, run: AgentRun, context: AgentContext, candidate_tools: list[dict[str, Any]], all_tools: list[dict[str, Any]], plan_override: AgentPlan | None) -> AgentPlan:
        if plan_override is not None: return plan_override
        return self.planner.build_plan_resilient(run.user_request, context, candidate_tools, all_tools=all_tools)

    def _datasets_without_version(self, context: AgentContext) -> list[str]:
        """会话里「还没有任何数据版本」的数据集名称。

        ``has_version`` 缺失时按「有数据」处理：老版本持久化的上下文没有这个键，
        默认成「没数据」会把正常会话全部拦下来。
        """
        names: list[str] = []
        for brief in (context.dataset_context or {}).values():
            if not isinstance(brief, dict) or brief.get("has_version", True):
                continue
            name = str(brief.get("name") or "").strip()
            names.append(name or f"数据集 {brief.get('dataset_id', '?')}")
        return names

    def _no_data_notice(self, run: AgentRun, session: AgentSession, datasets: list[str], on_event: EventCallback | None) -> None:
        """明确告诉用户「这份数据集还没有数据」，并给出下一步该做什么。

        错误文案要回答三件事：发生了什么 / 为什么 / 怎么办。一句
        ``dataset has no versions`` 三件都没说清。
        """
        joined = "、".join(datasets)
        answer = (
            f"当前数据集（{joined}）还没有数据版本，暂时无法分析。\n"
            "请先导入数据：在「数据集」页上传 CSV / Excel，或用数据库连接器抽取，"
            "导入成功后版本时间线会出现 v1，再到这里让我分析。"
        )
        run.answer_source = PLATFORM_RULES_NOTICE
        run.final_answer = answer
        run.status = RunStatus.COMPLETED
        run.finished_at = run.finished_at or time.time()
        session.history.append({"role": "assistant", "content": answer})
        self._emit(
            run, "completed",
            {"final_answer": answer, "mode": "notice", "reason": "dataset_has_no_version",
             "datasets": datasets, "answer_source": describe(run.answer_source)},
            on_event,
        )

    # ---- Pre-flight（第一层改造） ----------------------------------------
    def _preflight(self, run: AgentRun, session: AgentSession, intent, context: AgentContext):
        """开工前的确定性检查。拿不到列信息的检查会自动跳过（宁可不问，不乱问）。"""
        columns: list[dict[str, Any]] = []
        if settings.AGENT_PREFLIGHT_READ_SCHEMA and len(session.dataset_ids) == 1:
            columns = self._column_schema(session.dataset_ids[0])
        meta: dict[int, dict[str, Any]] = {}
        for key, value in (context.dataset_context or {}).items():
            try: meta[int(key)] = dict(value)
            except (TypeError, ValueError): continue
        return run_preflight(
            PreflightInput(
                user_request=run.user_request,
                intent=intent,
                dataset_ids=list(session.dataset_ids),
                dataset_meta=meta,
                columns=columns,
                answers=dict(run.clarification_answers),
            )
        )

    def _column_schema(self, dataset_id: int) -> list[dict[str, Any]]:
        """只读列结构（列名 + 类型），**不加载任何数据行**。

        Pre-flight 里刻意不用 `dataset.schema`（它要算每列唯一值，千万行表上
        是几十秒级开销），只读 Parquet 的 schema——列名与 dtype 已足够完成
        「目标列是否明确 / 任务类型是否矛盾」这两项判定。
        """
        try:
            return list(self.data_engine.column_schema(dataset_id) or [])
        except Exception:  # noqa: BLE001 - 拿不到列信息就跳过依赖它的检查
            return []

    def _await_clarification(self, run: AgentRun, pre, on_event: EventCallback | None) -> None:
        """把 Pre-flight 的反问挂到运行上，转入等待态。"""
        question = pre.first_question()
        payload = question.to_dict() if question else {
            "code": "preflight.unknown", "question": "需要补充信息才能继续", "options": [],
        }
        payload["outcome"] = str(pre.outcome)
        payload["checks_run"] = pre.checks_run
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = payload
        self._emit(run, "clarification", {**payload, "stage": "clarification_required"}, on_event)
        self.store.persist(force=True)

    def answer_clarification(self, run_id: str, answer: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """用户回答反问：把答案记进账本，然后从 Pre-flight / 原步骤继续。

        与 :meth:`resume` 的分工：``resume`` 处理「高风险操作授权」，
        这里处理「信息补全」。两者都会让运行脱离等待态，但语义不同。
        """
        # 空回答不是「接受默认值」，而是没答：直接往下走会用一个空串去填参数，
        # 得到看似成功实则答非所问的结果。要默认值就显式把 default 发回来。
        if not str(answer or "").strip():
            raise ValidationException("澄清回答不能为空")
        # ★ 与 confirm 同口径：原子领取，同一条反问只被消费一次。
        run, pending, code = self.store.claim_clarification(run_id, str(answer))
        if pending is None:
            raise ValidationException("该运行不在等待澄清状态（回答已被受理或运行已结束）")
        step_index = pending.get("step_index")
        session = self.store.get_session(run.session_id)
        intent = classify(run.user_request, has_datasets=bool(session.dataset_ids))
        self._emit(run, "clarification", {"stage": "answered", "code": code, "answer": str(answer)}, on_event)
        # 计划内反问（agent.clarify）→ 从原步骤继续；Pre-flight 反问 → 重新规划
        continue_from = int(step_index) if step_index is not None else None
        try:
            with self._usage_scope(run):
                run.status = RunStatus.RUNNING
                if continue_from is not None and run.plan:
                    continue_plan = AgentPlan(
                        goal=(run.plan or {}).get("goal", "继续执行"),
                        steps=self._plan_from_dict(run.plan).steps[continue_from:],
                    )
                    context = self._build_context(run, session)
                    self._run_plan(run, session, context, self._role_of(run), continue_plan,
                                   offset=continue_from, attempts={continue_from: 1},
                                   confirmed=False, on_event=on_event)
                else:
                    run.status = RunStatus.PLANNING
                    self._agent_turn(run, session, intent, confirmed=False, plan_override=None, on_event=on_event)
        except AgentLimitExceeded as exc: self._fail(run, str(exc), on_event)
        except AgentException as exc: self._fail(run, exc.message, on_event)
        except Exception as exc: self._fail(run, f"Agent 运行异常：{exc}", on_event)
        finally: run.finished_at = time.time()
        return run

    # 关键词表已迁到 `app.agent.intent`（唯一真源）：
    # 路由、候选工具注入、规则规划三处共用同一份，避免「改了两处漏了一处」
    # 造成的「工具已注册但 Agent 不调用」。这里保留别名仅为向后兼容。
    GREETINGS = GREETINGS
    DATA_TERMS = tuple(k for kws in INTENT_KEYWORDS.values() for k in kws)

    def _route(self, text: str, *, plan_override: AgentPlan | None = None, has_datasets: bool = False) -> tuple[str, str, Any]:
        """对话/工具路由。返回 ``(mode, reason, intent_decision)``。

        判定本身完全交给 :func:`app.agent.intent.classify`；这里只做
        「Intent → 走聊天还是走工具」的映射，不再维护第二套关键词。
        """
        if plan_override is not None:
            return "agent", "服务端指定计划（plan_override），直接进入工具流程", classify(text, has_datasets=has_datasets)
        decision = classify(text, has_datasets=has_datasets)
        if decision.value == Intent.CHAT:
            return "chat", decision.reasons[0] if decision.reasons else "按普通对话处理", decision
        return "agent", decision.reasons[0] if decision.reasons else "进入工具流程", decision

    def _needs_data_tools(self, text: str) -> bool:
        return self._route(text)[0] == "agent"

    # ---- 本地 Router shadow 埋点 -------------------------------------------------
    # 默认关闭（settings.LOCAL_ROUTER_MODE="off"），打开后**只记录、不改变行为**：
    # 即使模型缺失、推理抛异常、写盘失败，本次运行的路径与结果都完全不受影响。
    # 实现与数据形状见 app/local_router/trace.py。
    def _trace_route(self, run: AgentRun, session: AgentSession, mode: str, reason: str) -> None:
        try:
            from app.local_router.contract import RouterRequest
            from app.local_router.trace import shadow_route

            shadow_route(
                run_id=run.id, session_id=session.id,
                # 只喂路由发生时可**零成本**拿到的信号：available_columns 需要解析
                # Parquet 才能得到，shadow 阶段不为此付出代价（离线口径差异见 trace.py 文档）。
                request=RouterRequest(
                    utterance=run.user_request,
                    bound_dataset_id=(session.dataset_ids[0] if session.dataset_ids else None),
                ),
                rules_mode=mode, rules_reason=reason,
            )
        except Exception:
            pass

    def _trace_outcome(self, run: AgentRun, session: AgentSession) -> None:
        try:
            from app.local_router.trace import record_outcome

            record_outcome(
                run_id=run.id, session_id=session.id, status=str(run.status),
                planned_tools=[str(step.get("tool")) for step in ((run.plan or {}).get("steps") or [])],
                executed_tools=[str(record.tool) for record in run.tool_calls],
                error=run.error or "", elapsed=round(run.elapsed(), 3),
            )
        except Exception:
            pass

    def _direct_chat(self,run:AgentRun,session:AgentSession,on_event:EventCallback|None)->None:
        # 无远程大模型时，先试**平台自带能力的确定性应答**（问候 / 自述 / 能力清单 / 用法），
        # 拿不准才退回说明性文案。这样闲聊链路与数据链路（`_compose_answer` 的兜底）对称，
        # 「关掉远程」不再等于闲聊熄火 —— 见 app/agent/local_chat.py 的设计边界。
        if self.llm is None:
            from app.agent.local_chat import local_reply,no_llm_notice
            reply=local_reply(run.user_request)
            # ★ 区分「规则真的答了」与「规则答不了、只给了段状态说明」——
            # 两者对用户的含义完全不同：前者是答案，后者是「这里需要开远程」。
            run.answer_source=PLATFORM_RULES_CHAT if reply is not None else PLATFORM_RULES_NOTICE
            answer=reply or no_llm_notice()
        else:
            history=session.history[:-1][-settings.AGENT_CONTEXT_HISTORY_MESSAGES:]
            messages=[LLMMessage(role="system",content="你是小洛实验室的 AI 助手。当前是普通对话，不调用数据工具，不声称执行过任何数据处理。自然回答用户；只有明确的数据处理/分析需求才进入 Agent 工具流程。")]
            for item in history:
                if item.get("role") in {"user","assistant","system"} and item.get("content"): messages.append(LLMMessage(role=item["role"],content=str(item["content"])))
            messages.append(LLMMessage(role="user",content=run.user_request))
            try:
                answer=self.llm.chat(messages).content or "我在。有什么可以帮你？"
                run.answer_source=REMOTE_LLM_CHAT
            except Exception as exc:  # noqa: BLE001 — 只在 LLM 调用边界内捕获
                # ★ 「远程没走通」有三种完全不同的结局，不能一律塞一句「暂时无法完成对话请求」：
                #   ① 允许降级且内置规则答得上 ⇒ 真的给出本地回答，标 `LLM_ERROR_FALLBACK`；
                #   ② 允许降级但规则答不上 ⇒ 如实给一段状态说明，标 `NO_ANSWER`（这次没答上问题）；
                #   ③ 不允许降级（开关关着 / 这次错误不该兜底）⇒ 原样抛出，让这次运行如实失败。
                # 历史缺陷：三种一律走成「固定错误文案 + LLM_ERROR_FALLBACK」，
                # 既没真的兜底，又把一个纯报错标成了「已降级到规则」。
                if not fallback_allowed(exc):
                    raise
                from app.agent.local_chat import local_reply,no_llm_notice
                reply=local_reply(run.user_request,degraded=True)
                if reply is not None:
                    answer=reply
                    run.answer_source=LLM_ERROR_FALLBACK
                else:
                    answer=no_llm_notice(reason=str(exc))
                    run.answer_source=NO_ANSWER
        run.final_answer=answer; run.status=RunStatus.COMPLETED; session.history.append({"role":"assistant","content":answer}); self._emit_usage(run,on_event); self._emit(run,"completed",{"final_answer":answer,"mode":"chat","answer_source":describe(run.answer_source)},on_event)

    def _candidate_tools(self,context:AgentContext,all_tools:list[dict[str,Any]])->list[dict[str,Any]]:
        names={x.get("name") for x in context.tool_context.get("tools",[])}; return [t for t in all_tools if t.get("name") in names]

    def resume(self,session:AgentSession,run_id:str,*,on_event:EventCallback|None=None)->AgentRun:
        # ★ 原子领取：并发的两次 confirm 只有一个能拿到 pending，另一个被明确拒绝。
        # 不能依赖前端的 `confirming` 标记 —— 它只在单个标签页内有效，
        # 挡不住两个标签 / 网络重放 / 客户端重试。
        run, pending = self.store.claim_confirmation(run_id)
        if pending is None: raise ValidationException("该运行不在等待确认状态（授权已被领取或运行已结束）")
        step_index=int(pending["step_index"]); self._emit(run,"permission",{"stage":"confirmed","tool":pending["call"].tool,"step_index":step_index},on_event)
        plan=self._plan_from_dict(run.plan or {}); continue_plan=AgentPlan(goal=plan.goal,steps=plan.steps[step_index:])
        try:
            with self._usage_scope(run):
                # ★ 授权范围 = 用户刚才批准的那一步（``confirmed_step_index``）。
                # 用户确认的是弹窗里那一条「工具 + 参数」，不是整份计划；把 confirmed
                # 一路传给后续步骤等于一次确认放行整条链路，第二个高风险工具会被静默执行。
                context=self._build_context(run,session); self._run_plan(run,session,context,self._role_of(run),continue_plan,offset=step_index,attempts={step_index:1},confirmed=True,confirmed_step_index=step_index,on_event=on_event)
        except AgentLimitExceeded as exc:self._fail(run,str(exc),on_event)
        except AgentException as exc:self._fail(run,exc.message,on_event)
        except Exception as exc:self._fail(run,f"Agent 运行异常：{exc}",on_event)
        finally:run.finished_at=time.time()
        return run

    #: 两个等待态各自的终止文案。共用一条 deny 路径，但**不能共用一句文案**：
    #: 「拒绝授权」和「不回答澄清」对用户的含义不同，混在一起会让界面显示错误原因。
    _DENY_MESSAGES = {
        RunStatus.WAITING_CONFIRMATION: "用户拒绝授权，运行终止",
        RunStatus.WAITING_CLARIFICATION: "用户未回答澄清问题，运行终止",
    }

    def deny(self, run_id: str, *, on_event: EventCallback | None = None) -> AgentRun:
        """放弃等待：终止运行并释放会话（否则 run 永远停在等待态，会话被 409 锁死）。

        覆盖两种等待态，且**只清自己那一份载荷**：

            WAITING_CONFIRMATION  -> pending_confirmation（问「这个高风险操作做不做」）
            WAITING_CLARIFICATION -> pending_clarification（问「信息不全，请补一个参数」）

        早先的实现用 ``pending_confirmation is None`` 做统一门禁，于是澄清态
        （载荷在 pending_clarification 里）被判定为「无需处理」直接返回——
        用户点取消/拒绝后运行毫无反应，会话永久锁死。
        """
        # ★ 同样走原子领取：并发的两次 deny 只有一个真正终止运行，
        # 另一个拿到 kind="" 直接返回（幂等），不会把已失败的运行再写一遍。
        run, kind = self.store.claim_wait(run_id)
        if not kind:
            return run

        self._fail(run, self._DENY_MESSAGES[run.status], on_event)
        self.store.persist(force=True)
        return run

    def cancel(self, run_id: str) -> AgentRun:
        """请求取消运行：设置标记，_run_plan 在当前步骤结束后停止。

        等待态下取消等价于放弃等待（确认 → 拒绝授权，澄清 → 不回答），
        必须真的终止，否则会话一直被 ACTIVE_STATUSES 锁住。
        """
        run = self.store.get_run(run_id)
        if run.status in (RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RUNNING):
            run.cancel_requested = True
            self.store.persist(force=True)
        elif run.status in (RunStatus.WAITING_CONFIRMATION, RunStatus.WAITING_CLARIFICATION):
            run = self.deny(run_id)
        return run

    def _run_plan(self,run:AgentRun,session:AgentSession,context:AgentContext,role:str,plan:AgentPlan,*,offset:int,attempts:dict[int,int],confirmed:bool=False,confirmed_step_index:int|None=None,on_event:EventCallback|None)->None:
        """执行计划；失败即交给 `Replanner` 决定「重试 / 跳过 / 终止」。

        ★★ 授权范围（P0）：``confirmed`` 只在 ``confirmed_step_index`` 指定的那一步生效。

        历史缺陷：``resume()`` 把 ``confirmed=True`` 传进 ``_run_plan`` 后，这个布尔值会
        随着循环一路传给**每一个后续步骤**。于是「Step1 data.clean(HIGH) → 用户确认」之后，
        Step2 ml.train(HIGH)、Step3 workflow.run(HIGH) 全部被静默放行 —— 一次确认 = 放行整条
        高风险链路，而用户以为自己只批准了弹窗里那一个工具。

        现在：一次确认最多授权**一次**对应的高风险调用；后续再次遇到 HIGH / CRITICAL 会重新进入
        WAITING_CONFIRMATION。``confirmed_step_index is None`` 表示调用方显式授权整轮执行，
        仅用于服务端/测试直接调用（HTTP API 不接受客户端传 confirmed）。

        ★★ 授权是**一次性凭据**，不是状态开关（2026-09-24 P0）
        ``authorized_step`` 在本轮执行里只存在到「那一步真正被执行一次」为止，用完即置空。
        否则会出现：确认 → 高风险工具执行失败 → Replanner 判「瞬时故障、重试同一步」→
        同一步以 ``abs_idx == confirmed_step_index`` 再次命中授权 → 未经用户同意又执行一次。
        用户点的是弹窗里那一次操作，失败后重试属于**新的一次授权请求**，必须重新弹窗。
        （此前侥幸没出事，只是因为 ``resume()`` 预置 ``attempts={step_index:1}`` 让重试在第
        一次失败后就被跳过 —— 那是记账口径的巧合，不是授权语义的保证。）


        ★★ 死循环防线（2026-09-22 r-22 P0，实测过同一条路径产生 199987 条 replanning 事件）
        原实现里有三处叠加的漏洞，任意一处都足以让运行永不结束：

        1. **「参数解析失败」这条分支不记账** —— 只把 `attempts.get(abs_idx,0)+1` 传进
           replanner，从不写回 `attempts`。于是 `MAX_ATTEMPTS_PER_STEP` 永远看到 attempt=1，
           永远判定「还能再试一次」。
        2. 同一条分支**无条件 `offset=abs_idx+1`**，即使 replanner 返回的是「重试同一步」
           （`retry=True`）。步号一路 +1、计划却原封不动 ⇒ 同一个失败步骤以**全新下标**
           被反复重试，第 1 条的计数自然永远归零。
        3. `assert_limits`（工具调用数 / 超时 / token）**只在执行工具前调用**，解析失败的
           迭代完全不过熔断 ⇒ 连 180s 超时都救不了。

        现在的三层保护：**尝试数写回** + **重试不前进下标** + **每次迭代过熔断与重规划总闸**。
        """
        tool_ctx=self._tool_context(session,role,user_request=run.user_request)
        # 本轮已回答的反问注入工具上下文（保持 `_tool_context` 原签名不变：
        # 它是既有测试替身的重写点，改签名会连带打断它们）。
        # ★ 必须**赋值**而不是 ``setdefault``：``_tool_context`` 总是预置一个空 dict，
        # setdefault 因此永远不生效，工具拿到的 answers 恒为空 —— 用户回答后
        # agent.clarify 会原样再问一遍，形成「回答 → 继续 → 再问」的死循环。
        extra = getattr(tool_ctx, "extra", None)
        if isinstance(extra, dict):
            extra[ANSWERS_KEY] = dict(run.clarification_answers or {})
        services=self._services(); current=plan; permanent_failure=""; total_steps=max(len(plan.steps),1); replans=0
        # 一次性授权凭据：不是「本轮已确认」这个状态，而是「还剩这一次没用掉」。
        # None 表示调用方授权整轮（服务端/测试直调），不参与消费。
        authorized_step: int | None = confirmed_step_index
        while current.steps:
            # ① 重规划总闸：与单步尝试上限互补，任何原因的循环到这里都会被截断。
            #    阈值只在 ``Replanner.assert_replan_budget`` 定义一处——早先这里内联了
            #    一份同样的判断，两边一旦漂移就会出现「测试过的闸不是线上那个」。
            try:
                self.replanner.assert_replan_budget(replans)
            except AgentLimitExceeded as exc:
                permanent_failure = str(exc); break
            replanned=False
            for i,raw_step in enumerate(current.steps):
                if run.cancel_requested:
                    self._fail(run,"运行已被用户取消",on_event); return
                abs_idx=offset+i
                # ② 熔断：**每次迭代**都过，不只是执行工具前那一次
                self.replanner.assert_limits(tool_calls=run.tool_call_count+1,elapsed_seconds=run.elapsed(),total_tokens=run.token_ledger.actual_total_tokens,max_total_tokens=settings.AGENT_LLM_MAX_TOTAL_TOKENS)
                try: step=self._resolve_step_arguments(raw_step,run,session,context)
                except (ValidationException,PlanInvalidError) as exc:
                    errors=[getattr(exc,"message",str(exc))]
                    # ③ 解析失败同样算一次尝试并**写回**，否则单步上限形同虚设
                    attempt_no=attempts.get(abs_idx,0)+1; attempts[abs_idx]=attempt_no
                    replan=self.replanner.replan(current,failed_step_index=i,errors=errors,attempts=attempt_no,base_index=offset); replans+=1
                    self._emit(run,"replanning",{"step_index":abs_idx,"notes":replan.notes,"remaining":len(replan.steps),"retry":bool(getattr(replan,"retry",False)),"attempt":attempt_no},on_event)
                    if not replan.steps: permanent_failure=f"第 {abs_idx} 步失败：{'；'.join(errors)}"; break
                    # ④ 与执行失败分支同口径：要重试就留在原地，要跳过才前进
                    retrying=bool(getattr(replan,"retry",False)); offset=abs_idx if retrying else abs_idx+1; current=replan; replanned=True; break
                attempt_no=attempts.get(abs_idx,0)+1; attempts[abs_idx]=attempt_no
                self._emit(run,"tool_call",{"step_index":abs_idx,"tool":step.tool,"arguments":step.arguments,"progress":min(90,max(5,round((run.tool_call_count/max(total_steps,1))*90)))},on_event)
                # 授权只落到被用户确认的那一步；其余步骤按 PermissionManager 的裁决走
                # （HIGH / CRITICAL 会再次进入 WAITING_CONFIRMATION）。
                step_confirmed=confirmed and (confirmed_step_index is None or abs_idx==authorized_step)
                if step_confirmed and confirmed_step_index is not None:
                    # 凭据用掉即失效：同一高风险步骤再次执行（重试 / 重规划）必须重新确认。
                    authorized_step=None
                record=self.executor.execute_step(step,tool_ctx,services,confirmed=step_confirmed,attempt=attempt_no,step_index=abs_idx); run.tool_calls.append(record)
                if record.status=="needs_confirmation":
                    run.status=RunStatus.WAITING_CONFIRMATION; run.pending_confirmation={"call":record,"step_index":abs_idx}; self._emit(run,"permission",{"stage":"confirmation_required","tool":step.tool,"reason":record.error,"step_index":abs_idx},on_event); return
                if record.status=="needs_clarification":
                    # 计划执行中的结构化反问（agent.clarify）：记下步号，
                    # 用户回答后从这一步继续，前面的重型步骤不重跑。
                    #
                    # ★ 载荷必须落在 ``pending_clarification``。曾经错写成
                    # ``pending_confirmation``，而 ``answer_clarification`` 只读
                    # ``pending_clarification`` —— 结果是计划内反问**永远无法被回答**，
                    # 运行卡在 WAITING_CLARIFICATION 直到进程重启（会话期间一直 409）。
                    clarification=record.clarification.to_dict() if hasattr(record.clarification,"to_dict") else {}
                    run.status=RunStatus.WAITING_CLARIFICATION
                    run.pending_clarification={**clarification,"step_index":abs_idx,"tool":step.tool}
                    self._emit(run,"clarification",{**clarification,"stage":"clarification_required","tool":step.tool,"step_index":abs_idx},on_event)
                    self.store.persist(force=True); return
                if record.status=="denied": permanent_failure=f"第 {abs_idx} 步被拒绝：{record.error}"; break
                result_payload=self._result_for_llm(run,record); self._emit(run,"tool_result",{"step_index":abs_idx,"tool":step.tool,"status":record.status,"summary":record.result.summary if record.result else record.error,"llm_result":result_payload,"progress":min(95,max(5,round(((abs_idx+1)/max(total_steps,1))*90)))},on_event)
                # `_result_for_llm` 刚把「结果压缩省下多少 Token」记进账本，这里立刻同步给界面。
                self._emit_usage(run,on_event)
                if record.status=="ok":
                    validation=self.validator.validate(step,record.result,self._tool_output_schema(step.tool)); self._emit(run,"validation",{"step_index":abs_idx,"tool":step.tool,"valid":validation.valid,"errors":validation.errors,"warnings":validation.warnings},on_event)
                    if validation.valid: continue
                    errors=validation.errors
                else:
                    errors=[record.error] if record.error else ["工具执行失败"]; self._emit(run,"validation",{"step_index":abs_idx,"tool":step.tool,"valid":False,"errors":errors},on_event)
                replan=self.replanner.replan(current,failed_step_index=i,errors=errors,attempts=attempt_no,base_index=offset); replans+=1; self._emit(run,"replanning",{"step_index":abs_idx,"notes":replan.notes,"remaining":len(replan.steps),"retry":bool(getattr(replan,"retry",False)),"attempt":attempt_no},on_event)
                if not replan.steps: permanent_failure=f"第 {abs_idx} 步失败且无剩余步骤：{'；'.join(errors)}"; break
                retrying=bool(getattr(replan,"retry",False)); offset=abs_idx if retrying else abs_idx+1; current=replan; replanned=True; break
            if permanent_failure or not replanned: break
        if permanent_failure:self._fail(run,permanent_failure,on_event)
        else:self._complete(run,session,on_event)

    def _resolve_step_arguments(self,step:PlanStep,run:AgentRun,session:AgentSession,context:AgentContext)->PlanStep:
        tool=self.registry.get(step.tool); schema=tool.input_schema or {}
        # 先解析 {{stepN.field}} 结构化依赖（后续工具消费前一步输出），再做必填补全
        from app.agent.runtime.step_resolution import resolve_arguments
        step=resolve_arguments(step,run,session,context,schema)
        args=dict(step.arguments or {}); required=list(schema.get("required") or [])
        if "dataset_id" in required:
            if args.get("dataset_id") is None:
                ids=context.dataset_ids()
                if len(ids)==1: args["dataset_id"]=ids[0]
                elif not ids: raise ValidationException(f"工具 {step.tool} 需要 dataset_id，但当前会话没有关联数据集。")
                else: raise ValidationException(f"工具 {step.tool} 需要明确的 dataset_id，但当前会话关联了多个数据集。")
            try: args["dataset_id"]=int(args["dataset_id"])
            except (TypeError,ValueError) as exc: raise ValidationException(f"工具 {step.tool} 的 dataset_id 必须是整数。") from exc
            if session.dataset_ids and args["dataset_id"] not in session.dataset_ids: raise ValidationException(f"工具 {step.tool} 指定的数据集不在当前会话关联范围内。")
        missing=[name for name in required if args.get(name) in (None,"")]
        if missing: raise ValidationException(f"工具 {step.tool} 缺少必要参数：{', '.join(missing)}")
        return PlanStep(tool=step.tool,arguments=args,expected_output=step.expected_output,permission=step.permission)

    def _result_for_llm(self,run:AgentRun,record:ToolCallRecord,*,compact:bool=False)->dict[str,Any]|None:
        if record.result is None:return None
        if settings.AGENT_ENABLE_RESULT_COMPRESSION:
            view=record.result.for_llm(max_items=10 if compact else 16,max_chars=1600 if compact else 2600)
        else:view=record.result.to_dict()
        token_meta=view.get("_token") if isinstance(view,dict) else None
        if isinstance(token_meta,dict): run.token_ledger.record_result_saving(int(token_meta.get("estimated_saved",0) or 0))
        if isinstance(view,dict): view.pop("_token",None)
        return view

    def _build_context(self,run:AgentRun,session:AgentSession)->AgentContext: return ContextBuilder(self.data_engine).build(run.user_request,dataset_ids=session.dataset_ids,role=self._role_of(run),history=session.history[:-1])
    def _fill_tools(self,context:AgentContext,tools:list[dict[str,Any]])->None: ContextBuilder(self.data_engine).with_tools(context,tools,registry=self.registry)
    def _role_of(self,run:AgentRun)->str:return "analyst"
    def _tool_context(self, session: AgentSession, role: str, *, user_request: str = "") -> ToolExecutionContext:
        # `extra["user_request"]`：工具的「语义兜底输入」。
        # 没有它，ml.detect_task 这类需要理解用户诉求的工具就只能指望 LLM 记得把
        # 诉求原文抄进参数里 —— 那是把正确性押在提示词遵从度上（且规划器本来
        # 也拿不到列名）。工具读这个字段即可获得稳定的意图来源。
        # `extra[ANSWERS_KEY]`：本轮已回答的结构化反问（第一层），
        # agent.clarify 靠它判断「这个问题是否已经答过」。
        permissions=ROLE_PERMISSIONS.get(role,ROLE_PERMISSIONS["viewer"])
        return ToolExecutionContext(
            user_id=session.user_id,
            session_id=session.id,
            # 会话关联了哪些数据集就是哪些：空列表 ⇒ set()（一个都不许访问），
            # 而不是 None（不限制）。Agent 场景下不存在「无限制访问全部数据集」的语义，
            # 用户没选数据集时应由 Pre-flight / 反问来处理，而不是静默读任意数据。
            dataset_ids=set(session.dataset_ids),
            permissions=set(permissions),
            extra={
                "user_request": str(user_request or ""),
                ANSWERS_KEY: {},
            },
        )
    def _services(self)->ToolServices:
        # connector_service 按需构造（依赖同一个 Session 与 DatasetService）：
        # 连接器导入必须复用 DatasetService 的版本链路，不能自建一套存储。
        from app.connectors.service import ConnectorService
        return ToolServices(dataset_service=self.data_engine.dataset_service,data_engine_service=self.data_engine,experiment_service=self.experiment_service,connector_service=ConnectorService(self.db,self.data_engine.dataset_service),db=self.db)
    def _tool_output_schema(self,tool_name:str)->dict[str,Any]|None:
        try:return self.registry.get(tool_name).output_schema
        except Exception:return None
    def _plan_from_dict(self,data:dict[str,Any])->AgentPlan: return AgentPlan(goal=data.get("goal","继续执行"),steps=[PlanStep(tool=s["tool"],arguments=s.get("arguments") or {},expected_output=s.get("expected_output") or "",permission=s.get("permission") or "") for s in data.get("steps",[])])
    def _set_status(self,run:AgentRun,status:RunStatus)->None:run.status=status
    def _emit(self,run:AgentRun,event_type:str,payload:dict[str,Any],on_event:EventCallback|None)->None:
        # ★ 事件数硬上限：r-3 一次死循环往同一次运行里写了 **199987 条事件**（store 撑到 51MB，
        # 且每次持久化都要整体重写一遍）。重规划总闸理论上已经拦住这一类循环，但这条是
        # 「不管什么原因」的最后护栏 —— 宁可让这一次运行失败，也不能把进程和磁盘拖死。
        # 终态事件（completed / failed）永远放行，否则连失败原因都发不出去。
        if event_type not in _TERMINAL_EVENTS and len(run.events)>=MAX_EVENTS_PER_RUN:
            raise AgentLimitExceeded(f"单次运行的事件数达到上限 {MAX_EVENTS_PER_RUN}，判定为异常循环，已终止")
        event=AgentEvent(seq=len(run.events)+1,run_id=run.id,type=event_type,payload=payload); run.events.append(event)
        if on_event:on_event(event)
    def _fail(self,run:AgentRun,message:str,on_event:EventCallback|None)->None:
        # 失败/取消/拒绝都不产生回答，来源如实标成「未产出回答」而不是留空让前端猜。
        if not run.answer_source: run.answer_source=NO_ANSWER
        run.error=message;run.status=RunStatus.FAILED;run.finished_at=run.finished_at or time.time();self._emit_usage(run,on_event);self._emit(run,"failed",{"error":message,"answer_source":describe(run.answer_source)},on_event)
    def _complete(self,run:AgentRun,session:AgentSession,on_event:EventCallback|None)->None:run.final_answer=self._compose_answer(run);run.status=RunStatus.COMPLETED;run.finished_at=run.finished_at or time.time();session.history.append({"role":"assistant","content":run.final_answer});self._emit_usage(run,on_event);self._emit(run,"completed",{"final_answer":run.final_answer,"mode":"agent","answer_source":describe(run.answer_source)},on_event)
    def _emit_usage(self,run:AgentRun,on_event:EventCallback|None)->None:
        """发一帧 Token 账本快照（`usage` 事件）。

        为什么单独发一类事件而不是塞进每个事件里：账本在**运行过程中**就在变
        （规划器调用、结果压缩节省、Plan Cache 命中），而界面原先只在结束后从
        `GET /runs/{id}` 拿一次总数 —— 长任务（或像 r-22 那种卡住的任务）期间
        用户完全看不到「已经花了多少 / 省了多少」。
        这里只发快照、不改任何行为；写盘节流由上层 persist 控制，
        故意**不**加进 `agent_runtime.py` 的落盘白名单，避免多出的写放大。
        """
        try: self._emit(run,"usage",{"token_usage":run.token_ledger.to_dict()},on_event)
        except Exception: pass  # noqa: BLE001 — 账本上报失败绝不能影响主流程
    def _compose_answer(self,run:AgentRun)->str:
        views=[];summaries=[]
        for call in run.tool_calls:
            if call.status!="ok" or call.result is None:continue
            summaries.append(f"{call.tool}：{call.result.summary}");view=self._result_for_llm(run,call,compact=True)
            if view is not None:views.append({"tool":call.tool,"result":view})
        fallback=f"已完成 {len(summaries)} 个真实数据工具步骤。\n"+"\n".join(f"{i}. {s}" for i,s in enumerate(summaries,1)) if summaries else "任务没有产生有效的数据处理结果。"
        # 无远程模型 ⇒ 汇总也是规则拼的，如实标注（数据本身仍是真实工具跑出来的）。
        if self.llm is None:
            run.answer_source=PLATFORM_RULES_SUMMARY
            return fallback
        system=("你是小洛实验室的数据分析助手。基于真实工具结果回答用户的原始请求。输出结构：先给 3-5 条结论式关键发现，再列数据质量问题与风险，最后给可操作建议。只引用对结论有支撑的关键数值，禁止逐项复述工具结果里的全部数字；总长度控制在 400 字以内。不得声称做过没有工具记录的操作。")
        try:
            response=self.llm.chat([LLMMessage(role="system",content=system),LLMMessage(role="user",content=f"用户原始请求：{run.user_request}\n\n工具结果（已压缩）：\n{json.dumps(views,ensure_ascii=False,default=str) or '（无有效工具结果）'}")])
        except Exception as exc:  # noqa: BLE001 — 只在 LLM 调用边界内捕获
            # ★ 与闲聊链路同口径：走了 LLM 但没走通 ⇒ 标降级，不算「远程生成」。
            # 但降级必须是被允许的（总闸 × 错误分类），否则如实失败——
            # 「工具都跑成功了，只是汇总时模型挂了」仍然是一次失败的运行，
            # 不该被静默包装成「已降级」蒙混过去。
            if not fallback_allowed(exc):
                raise
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        if not (response.content or "").strip():
            # 模型返回空串也算「这次没拿到模型输出」，不能标成远程生成。
            if not settings.AGENT_ALLOW_MODEL_FALLBACK:
                raise LLMException("远程大模型返回空内容")
            run.answer_source=LLM_ERROR_FALLBACK
            return fallback
        run.answer_source=REMOTE_LLM_SUMMARY
        return response.content
````

### 6. `backend/app/agent/local_chat.py`

- 新增 `_DEGRADED_NOTE` 与 `_note(*, degraded: bool)`：把「远程大模型已停用」和「远程大模型这一次调用失败」区分开，不再对用户撒谎。
- 6 个私有应答函数与 `local_reply()` 统一增加 `degraded` 关键字参数（默认 False，兼容既有调用与测试）。
- `no_llm_notice(reason)` 在 reason 非空时把真实失败原因写进文案。

````python
"""平台自带能力的**闲聊应答** —— 只在「远程大模型不可用」时启用。

为什么需要它
------------
`runtime._direct_chat()` 原先在 `llm is None` 时只回一句
「当前尚未配置可用的大模型。」；而数据链路 `_compose_answer()` **是有本地兜底的**
（`已完成 N 个真实数据工具步骤。` + 逐条工具摘要）。

两条链路不对称的后果：设置页关掉「启用远程 API 大模型」之后，**数据请求仍然完全可用，
闲聊却彻底熄火** —— 这与设置页「关闭后 Agent 改用平台自带模型」的承诺不符。
更糟的是那句文案本身在关掉开关时是**事实错误**：`PUT /settings/llm/remote` 只切开关、
凭据原样保留，用户看到「尚未配置」会跑去重填 Key（见 `no_llm_notice()`）。

设计边界（刻意收窄，不要扩成聊天机器人）
----------------------------------------
1. **只覆盖答案确定的意图**：问候 / 自述 / 能力询问 / 使用入口。
   这些问题的正确答案是确定的，用模板表达不算「假装会说话」。
2. **拿不准就返回 `None`**，由上层用 `no_llm_notice()` 说明现状。
   **不猜、不瞎聊** —— 一旦开始对任意句子胡编，就不如老实说「需要远程模型」。
3. **能力清单从 `TOOL_REGISTRY` 动态生成**，不硬编码任何工具名。
   平台增删工具后这段话自动跟随，与「本地 Router 标签空间动态派生」是同一个原则。
4. **只在 `llm is None` 时被调用**：远程大模型开启时，`_direct_chat` 仍旧走 LLM，
   行为**完全不变**（这一点由 `tests/test_local_chat.py` 钉住）。

已知取舍
--------
这仍是**规则表**，不产生新信息：同一个问题每次都得到同一段话。
它是「关掉远程后平台不要像死的一样」的最低成本方案，不是本地生成模型。
论文里应写成**受限的本地应答能力**，而不是「平台自带大模型」。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "DOMAIN_LABELS",
    "local_reply",
    "no_llm_notice",
    "tool_overview",
]

# 「谢谢」「再见」混在 `AgentRuntime.GREETINGS` 里（那张表是给 chat/agent 分流用的），
# 这里按应答语义拆开。
_THANKS = {"谢谢", "感谢", "多谢", "辛苦了", "thanks", "thank you", "thx"}
_FAREWELL = {"再见", "拜拜", "bye", "goodbye", "下次见", "回头聊"}
# 身份询问同时也是问候语表中的词，必须先判身份（应答内容差别很大）。
_IDENTITY_PATTERNS = (
    "你是谁", "你叫什么", "你是什么", "介绍一下你", "自我介绍", "你是哪个", "谁开发的", "你是人还是",
)
_CAPABILITY_PATTERNS = (
    "你能做什么", "你会做什么", "能做什么", "会做什么", "有什么功能", "有哪些功能",
    "能帮我做什么", "能干什么", "支持什么", "支持哪些", "都会啥", "会啥",
)
_USAGE_PATTERNS = (
    "怎么用", "如何使用", "使用方法", "使用说明", "怎么用你", "帮助", "help", "教我", "怎么开始",
)

# 能力域的中文名。键取自 `contract.Intent` 的取值（动态派生的能力域）。
DOMAIN_LABELS: dict[str, str] = {
    "dataset": "数据接入与查看",
    "data_transform": "数据清洗与转换",
    "eda": "探索性分析（EDA）",
    "ml": "机器学习建模",
    "workflow": "流程编排",
    "report": "报告生成",
    "other": "其他",
}
_DOMAIN_ORDER = ("dataset", "data_transform", "eda", "ml", "workflow", "report", "other")

# 每个能力域最多举几个工具名做示例（全列出来会把消息撑爆）
_EXAMPLES_PER_DOMAIN = 3

_MODE_NOTE = "（当前运行在**平台自带能力**模式：远程大模型已停用，以下回复来自内置规则，不是语言模型生成的）"

# ★ 「远程开着但这一次调用失败」与「远程被主动停用」是两件事：
# 复用 `_MODE_NOTE` 会在欠费 / 超时场景下告诉用户「远程已停用」—— 那是假的，
# 用户会跑去设置页找开关，而真实原因是余额 / 网络。文案必须分开。
_DEGRADED_NOTE = "（本次远程大模型调用失败，已退回**平台自带能力**：以下回复来自内置规则，不是语言模型生成的）"


def _note(*, degraded: bool = False) -> str:
    return _DEGRADED_NOTE if degraded else _MODE_NOTE


# ---------------------------------------------------------------------------
# 动态能力清单
# ---------------------------------------------------------------------------


def tool_overview() -> tuple[int, list[tuple[str, list[str]]]]:
    """返回 `(工具总数, [(能力域中文名, 该域工具名列表), ...])`，全部实时读自注册表。

    读不到注册表时返回 `(0, [])` —— 调用方负责优雅退化，绝不在这里抛异常。
    """
    try:
        from app.local_router import contract as C

        names = list(C.tool_label_space())
        buckets: dict[str, list[str]] = {}
        for name in names:
            intent = C.intent_of_tool(name)
            buckets.setdefault(intent.value if intent is not None else "other", []).append(name)
    except Exception as exc:  # noqa: BLE001 — 能力清单拿不到不该让闲聊也失败
        logger.warning("读取工具注册表失败，能力清单降级：%s", exc)
        return 0, []

    ordered = [(DOMAIN_LABELS.get(key, key), sorted(buckets[key]))
               for key in _DOMAIN_ORDER if key in buckets]
    return len(names), ordered


def _capability_brief() -> str:
    """「我能做这些」的正文；注册表不可用时给一句诚实说明而不是空表。"""
    total, groups = tool_overview()
    if total == 0:
        return ("工具注册表当前不可用，我暂时列不出能力清单。"
                "可以稍后重试，或直接描述你的需求我试着执行。")
    lines = []
    for label, names in groups:
        shown = "、".join(names[:_EXAMPLES_PER_DOMAIN])
        more = f" 等 {len(names)} 个" if len(names) > _EXAMPLES_PER_DOMAIN else ""
        lines.append(f"• {label}：{shown}{more}")
    return (f"我能调度平台**真实注册**的 {total} 个数据工具，按能力域分成 {len(groups)} 类：\n"
            + "\n".join(lines)
            + "\n\n这份清单是**实时读自工具注册表**的，平台增删工具会自动跟随，不是写死的。")


# ---------------------------------------------------------------------------
# 各意图的应答
# ---------------------------------------------------------------------------


def _thanks_reply(*, degraded: bool = False) -> str:
    return "不客气。有数据上的需求随时说 —— 比如「看看这批数据的分布」「检查一下数据质量」。\n" + _note(degraded=degraded)


def _farewell_reply(*, degraded: bool = False) -> str:
    return "再见！需要的时候再来找我。\n" + _note(degraded=degraded)


def _greeting_reply(*, degraded: bool = False) -> str:
    lead = (
        "你好！我是小洛实验室的 AI 助手。\n\n这次远程大模型没能调用成功，下面这段是平台内置规则给的；"
        if degraded
        else "你好！我是小洛实验室的 AI 助手。\n\n虽然现在跑在平台自带能力模式下（对话回复是内置规则），"
    )
    return (lead
            + "**数据分析这条路是通的**："
              "你把需求说清楚，我就用平台真实的工具去执行，再把结果汇总给你。\n\n"
              "可以试试：「看看这批数据的分布」或「检查一下数据质量」。\n" + _note(degraded=degraded))


def _identity_reply(*, degraded: bool = False) -> str:
    lead = (
        "我是小洛实验室的 AI 助手。这一次远程大模型调用失败，所以现在按**平台自带能力**回答 —— "
        if degraded
        else "我是小洛实验室的 AI 助手，跑在**平台自带能力**模式下 —— "
    )
    tail = (
        "对话回复来自内置规则，只有数据分析由平台真实工具完成。\n\n"
        if degraded
        else "意思是我现在不调用远程大模型，对话回复来自内置规则，只有数据分析由平台真实工具完成。\n\n"
    )
    return lead + tail + _capability_brief()


def _capability_reply(*, degraded: bool = False) -> str:
    return ("我现在能做的事，按能力域列给你：\n\n"
            + _capability_brief()
            + "\n\n把需求说清楚我就直接执行（例如「把缺失值清洗掉」「训练一个分类模型」）。"
              "复杂或需要多轮澄清的需求，建议到「设置 → AI 服务」重新开启远程大模型。\n"
            + _note(degraded=degraded))


def _usage_reply(*, degraded: bool = False) -> str:
    return ("平台用法，三条路径：\n\n"
            "1. **直接说需求**（最常用）：「看看这批数据的分布」「把缺失值清洗掉」"
            "「训练一个分类模型，目标列是 label」—— 我用真实工具执行后汇总结果给你。\n"
            "2. **先绑数据集再提问**：在「数据」里上传或选中数据集，会话里绑定它；"
            "这样「看看分布」这类省略主语的话也能直接执行，否则我会先反问你用哪个数据集。\n"
            "3. **固定下来重复跑**：处理链稳定后用「流程」画布搭好，之后一键运行；"
            "想练手可以去「学习中心」，那里有带判分的用例。\n\n"
            + _note(degraded=degraded))


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def local_reply(utterance: str, *, degraded: bool = False) -> str | None:
    """命中确定性意图则返回本地应答；**拿不准返回 `None`**（由上层说明现状）。

    判定顺序即优先级：身份询问在问候语之前（「你是谁」同时在两张表里，
    而自述比一句「你好」有用得多）。

    `degraded=True` 表示这次是「远程调用失败后的降级」而不是「远程本来就关着」，
    两种场景的结论性注释不同（见 `_note`）—— 混用会让用户在欠费时看到
    「远程已停用」这种与事实不符的说明。
    """
    text = (utterance or "").strip()
    if not text:
        return None
    low = text.lower()

    if low in _THANKS:
        return _thanks_reply(degraded=degraded)
    if low in _FAREWELL:
        return _farewell_reply(degraded=degraded)
    if any(pattern in low for pattern in _IDENTITY_PATTERNS):
        return _identity_reply(degraded=degraded)
    if any(pattern in low for pattern in _CAPABILITY_PATTERNS):
        return _capability_reply(degraded=degraded)
    if any(pattern in low for pattern in _USAGE_PATTERNS):
        return _usage_reply(degraded=degraded)

    # 复用 `_route` 那张问候语表，避免两处词表各自漂移（延迟 import 以免循环依赖）
    from app.agent.runtime.runtime import AgentRuntime

    if low in {word.lower() for word in AgentRuntime.GREETINGS}:
        return _greeting_reply(degraded=degraded)

    return None


def no_llm_notice(reason: str = "") -> str:
    """远程大模型不可用时的**说明性文案** —— 必须区分三种完全不同的处境。

    原先一律写「当前尚未配置可用的大模型。」，但关掉设置页开关时凭据是**原样保留**的
    （`PUT /settings/llm/remote` 只切开关）。几种状态下用户该做的事完全不同：

    - 已配置但停用       ⇒ 去设置页**重新打开开关**（不需要重填 Key）
    - 已配置、调用失败   ⇒ 看 `reason`（欠费 / 超时 / 5xx …）决定是充值还是重试
    - 从未配置           ⇒ 需要**填 API Key**

    合并成一句会让第一种用户白折腾一遍，并且怀疑自己弄丢了配置。

    `reason` 非空即表示「开关开着、凭据也在，但这一次调用失败了」——
    这是欠费 / 限流 / 网络故障时用户唯一能看到的原因，必须原样带出去。
    """
    try:
        from app.core.config import settings

        key_set = bool(settings.LLM_API_KEY)
        remote_on = bool(settings.LLM_REMOTE_ENABLED)
    except Exception:  # noqa: BLE001
        key_set, remote_on = False, True

    tail = ("在停用期间能做的事：**数据类请求不受影响** —— 你说需求，我用平台真实工具执行。\n"
            "试试说：「看看这批数据的分布」。\n\n"
            "需要自由对话时，到「设置 → AI 服务」重新打开开关即可（**凭据保留，不用重填**）。")
    # 「远程这一侧这次没走通」的原因行：欠费、超时、5xx 都从这里透出给用户。
    reason_line = f"远程大模型**这一次调用失败**了：{reason}\n" if reason else ""

    if reason:
        return (reason_line
                + "闲聊回复需要大模型生成，所以这一段没有真正的回答；\n\n"
                + "**数据类请求不受影响** —— 你说需求，我用平台真实工具执行，"
                  "只是最后的自然语言汇总会退回内置规则。\n"
                  "可以到「设置 → AI 服务」点一次「测试连接」确认连通性与账户余额。")
    if key_set and not remote_on:
        return ("远程大模型**已停用**（设置 → AI 服务里的总开关），凭据仍然保留着。\n"
                "闲聊回复需要大模型生成，所以这一块暂时没有输出；\n\n" + tail)
    if key_set and remote_on:
        # 理论上不该走到这里（开关开着且有 Key ⇒ 上层会拿到 Provider）；
        # 真出现了说明 Provider 构造被别处拦下，如实说明而不是假装是配置问题。
        return ("大模型凭据已配置、开关也开着，但这一次没能取到可用的大模型。\n"
                "可以先在「设置 → AI 服务」点一次「测试连接」确认连通性；\n"
                "在此之前，数据类请求仍然可以正常执行。")
    return ("当前**尚未配置**大模型。\n"
            "到「设置 → AI 服务」填 API Key 就能接入远程大模型；不配置也能做数据分析，\n\n" + tail)
````

### 7. `backend/app/agent/answer_source.py`

- `NO_ANSWER` 的说明补全为「失败 / 被拒绝 / 被取消 / 远程调用失败且内置规则未覆盖」，作为来源语义的一部分而不是兜底垃圾值。

````python
"""「这条回答到底是谁给的」——答案来源（provenance）标记。

为什么需要它
------------
开启远程 API 之后，界面上那条助手消息可能有三种完全不同的来路，而且**长得一模一样**：

1. **远程大模型生成** —— 普通对话走 `_direct_chat` 的 LLM 分支，
   工具结果汇总走 `_compose_answer` 的 LLM 分支；
2. **平台内置规则生成** —— 远程未启用时的确定性应答（`local_chat`）
   或内置兜底汇总（`_compose_answer` 的 fallback）；
3. **远程调用失败后降级** —— 开关开着、凭据也在，但这一次请求没成功，
   于是仍然落到规则兜底。

第 3 条正是「开了 API 也不知道到底是谁回答的」的根源：它和第 1 条在界面上无从区分，
用户只能靠猜。本模块给每次运行打一个**机器可读**的 `answer_source`，随 SSE 的
`completed` 事件与 `GET /runs/{id}` 一起下发，界面据此明示来源。

设计边界
--------
- 只标记**事实**（这段文字由谁产出），不做质量评价、不做打分。
- 判定依据是「运行时这一次到底有没有走通 LLM」，而不是配置项。
  配置说开了、但实际调用抛异常 ⇒ 标 `LLM_ERROR_FALLBACK`，不谎报成远程生成。
- 模型名只作展示信息附带，不参与判定。
"""

from __future__ import annotations

from typing import Any

# --- 来源取值 ---------------------------------------------------------------
# remote_*  = 远程大模型（语言模型）真的产出了这段文字
# platform_*= 平台内置规则产出了这段文字（没有语言模型参与）
# 以外的 `llm_error_fallback` = 本想用远程但没走通，仍然由规则产出

REMOTE_LLM_CHAT = "remote_llm_chat"
REMOTE_LLM_SUMMARY = "remote_llm_summary"
PLATFORM_RULES_CHAT = "platform_rules_chat"
PLATFORM_RULES_NOTICE = "platform_rules_notice"
PLATFORM_RULES_SUMMARY = "platform_rules_summary"
LLM_ERROR_FALLBACK = "llm_error_fallback"
NO_ANSWER = "no_answer"

__all__ = [
    "LLM_ERROR_FALLBACK",
    "NO_ANSWER",
    "PLATFORM_RULES_CHAT",
    "PLATFORM_RULES_NOTICE",
    "PLATFORM_RULES_SUMMARY",
    "REMOTE_LLM_CHAT",
    "REMOTE_LLM_SUMMARY",
    "describe",
    "llm_model_name",
]

# source -> (短标签, 一句话说明, 是否由语言模型生成)
_META: dict[str, tuple[str, str, bool]] = {
    REMOTE_LLM_CHAT: ("远程大模型 · 对话", "这条回复由远程大模型直接生成（普通对话链路，未调用数据工具）。", True),
    REMOTE_LLM_SUMMARY: ("远程大模型 · 汇总", "工具结果由远程大模型汇总成自然语言；数据本身来自平台真实工具。", True),
    PLATFORM_RULES_CHAT: ("平台内置规则 · 对话", "远程大模型未启用，这条回复来自平台内置规则模板，不是语言模型生成的。", False),
    PLATFORM_RULES_NOTICE: ("平台内置规则 · 说明", "内置规则没有覆盖这个问法，回复是一段当前状态说明。", False),
    PLATFORM_RULES_SUMMARY: ("平台内置规则 · 汇总", "远程大模型未启用，工具结果由平台内置规则拼装成回答。", False),
    # ★ 最容易误导用户的一种：开关开着，但这次没走通。必须显式标出来。
    LLM_ERROR_FALLBACK: ("远程调用失败 · 已降级", "本次本应由远程大模型生成，但调用失败，已退回平台内置规则。", False),
    # ★ 两种「没有回答」都落在这里：运行真的失败了；或远程失败后内置规则也没覆盖这个问法
    # （界面上那段文字只是**状态说明**，不是对用户问题的回答）。
    # 两者的共同点是都不能标成「远程生成」或「已降级」——降级必须真的产出了替代回答。
    NO_ANSWER: ("未产出回答", "这次运行没有产出回答（失败 / 被拒绝 / 被取消 / 远程调用失败且内置规则未覆盖）。", False),
}


def llm_model_name() -> str:
    """当前配置的大模型名，**仅用于展示**；缺失时返回空串（不参与任何判定）。"""
    from app.core.config import settings

    return str(getattr(settings, "LLM_MODEL", "") or "")


def describe(source: str) -> dict[str, Any]:
    """把 `answer_source` 展开成界面可直接渲染的结构。

    未知 / 空值统一落到 `NO_ANSWER`，避免前端为每个取值补兜底分支。
    """
    key = source if source in _META else NO_ANSWER
    label, detail, by_llm = _META[key]
    # 模型名是纯展示信息：拿不到就留空，**绝不能因为它把来源标记整条链路带崩**
    # （来源标记属于可观测性，一旦抛异常会让 completed 事件都发不出去）。
    try:
        model = llm_model_name()
    except Exception:  # noqa: BLE001
        model = ""
    return {
        "source": key,
        "label": label,
        "detail": detail,
        "by_llm": by_llm,
        # 只有「确实由远程模型产出」时才带模型名，避免降级场景里出现误导性的模型标识。
        "model": model if (by_llm and model) else "",
    }
````

### 8. `backend/app/core/config.py`

- `AGENT_ALLOW_MODEL_FALLBACK` 由 `False` 改为 `True`：它现在是真总闸（planner 与 runtime 都真正读取），平台推荐默认开启本地兜底。

````python
"""项目配置。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# env_file 必须锚定 BACKEND_ROOT：pydantic-settings 按**进程 cwd**解析相对路径，
# 从项目根目录（而非 backend/）启动时会读不到 backend/.env， silently 退化为空 LLM Key，
# 表现为「Agent 不调用工具 / 不写报告」。同时保留仓库根 .env 作为兜底。
_ENV_FILES = (BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env")


class Settings(BaseSettings):
    """应用运行配置。"""

    model_config = SettingsConfigDict(
        env_file=tuple(str(p) for p in _ENV_FILES),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    APP_NAME: str = "xiaoluo-lab"
    APP_ENV: str = "dev"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./data/xiaoluo.db"
    DATA_ROOT: str = "./data"
    MODEL_ROOT: str = "./models"
    LOG_LEVEL: str = "INFO"
    PDF_FONT_PATH: str | None = None

    # =========================================================
    # HTTP 边车能力（跨域 / 压缩 / 限流）
    # =========================================================
    # 允许的跨域来源，逗号分隔。留空 ⇒ **不放松同源策略**
    # （开发时前端走 Vite proxy 同源；容器部署才需要显式放开前端域名）。
    # 生产示例：CORS_ALLOW_ORIGINS=https://lab.example.com
    CORS_ALLOW_ORIGINS: str = ""
    # 跨域是否携带凭据（Cookie）。放开前 Web 前端必须有可信的抗 CSRF 措施。
    # 本项目用 Bearer-less 的本地单租户模型，默认 False。
    CORS_ALLOW_CREDENTIALS: bool = False
    # 响应体压缩。报告 / 数据集列表这类 JSON 动辄几百 KB，压缩收益明显；
    # SSE 流式响应在中间件里自动跳过（见 core/middleware.py）。
    GZIP_ENABLED: bool = True
    GZIP_MINIMUM_SIZE: int = 1024
    # 轻量限流：只保护「会消耗 LLM token / 触发重任务」的端点（见 RATE_LIMITED_PATHS
    # 常量），不限制静态浏览。按客户端 IP + 路径的滑动窗口计数。
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 60
    # 被判定为同一客户端时是否信任 X-Forwarded-For 的最左一跳。
    # 反向代理后才可打开；直接暴露端口时打开等于可被伪造绕过。
    RATE_LIMIT_TRUST_X_FORWARDED_FOR: bool = False

    # LLM：默认走 OpenAI-compatible 协议，不绑定具体厂商。
    # 一次 Agent Run 只使用当前选定的一个 Provider + Model；切换通过设置完成。
    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_CONTEXT_WINDOW: int | None = None
    LLM_MAX_OUTPUT_TOKENS: int | None = None
    # 「是否启用远程 API 大模型」总开关（设置页可切换）。
    # 关闭后不再构造远程 Provider ⇒ Agent 走平台自带的规则规划器，
    # 用于测试平台自带的小模型 / 做「有 LLM vs 无 LLM」的对照实验。
    # 与 LLM_API_KEY 刻意解耦：关闭时凭据仍保留在进程内，重新开启无需重填。
    LLM_REMOTE_ENABLED: bool = True

    # 本地 Router（本地小模型路由；实现见 app/local_router/）
    # 三档模式：
    #   off    = 完全关闭：不加载模型、不写 trace（默认值，零风险）
    #   shadow = **只记录不改变行为**：把「本地 Router 会怎么判」与「线上实际怎么走」
    #            并排写进 trace，用于积累真人测试数据、验证离线指标是否可复现
    #   guard  = 预留档：低置信/结构不合格时交由本地 Router 决定（尚未启用，
    #            打开前必须先跑 analyze_fusion.py 的覆盖率-准确率曲线选阈值）
    LOCAL_ROUTER_MODE: str = "off"
    # 门控阈值。仅 guard 档生效；默认 0.0 表示与离线评测口径一致（不门控）。
    LOCAL_ROUTER_CONFIDENCE_THRESHOLD: float = 0.0
    # 产物目录。留空 ⇒ `{MODEL_ROOT}/local_router/`。
    LOCAL_ROUTER_MODEL_DIR: str = ""
    # trace 单文件上限（字节），超出后轮转为 `shadow.jsonl.1`（只保留一代，避免无限增长）。
    LOCAL_ROUTER_TRACE_MAX_BYTES: int = 5 * 1024 * 1024

    # Agent / Token-aware 上下文控制
    AGENT_CONTEXT_MAX_CHARS: int = 12000
    # ★ token 上限（0 = 关闭，退回纯字符口径）。
    # 字符预算表达不了真实开销：中文约 1 字≈1 token，英文约 4 字符≈1 token。
    # 只按字符卡，中文场景会把 LLM 输入窗口悄悄吃满；加上这一档之后
    # 「字符」与「token」双重约束，谁先到按谁截断。
    # 默认 6000 约为 AGENT_LLM_MAX_INPUT_TOKENS(24000) 的四分之一，
    # 给系统提示词与工具清单留出足够空间。
    AGENT_CONTEXT_MAX_TOKENS: int = 6000
    AGENT_CONTEXT_USER_REQUEST_CHARS: int = 1200
    AGENT_CONTEXT_DATASET_CHARS: int = 1600
    AGENT_CONTEXT_TASK_CHARS: int = 1200
    AGENT_CONTEXT_PERMISSION_CHARS: int = 400
    AGENT_CONTEXT_TOOL_CHARS: int = 3000
    AGENT_CONTEXT_HISTORY_CHARS: int = 1600
    AGENT_CONTEXT_HISTORY_MESSAGES: int = 6
    AGENT_DATASET_CACHE_ENABLED: bool = True
    AGENT_DATASET_CACHE_MAX_ITEMS: int = 32
    # 注意：这些只是「缺省值」。实际生效值以 backend/.env 为准（12 / 16 / 0.08 / 60000）。
    AGENT_MAX_STEPS: int = 12
    AGENT_LLM_MAX_CALLS: int = 12
    AGENT_LLM_MAX_INPUT_TOKENS: int = 24000
    AGENT_LLM_MAX_OUTPUT_TOKENS: int = 4096
    AGENT_LLM_MAX_TOTAL_TOKENS: int = 60000
    # ★ 真正的总闸：控制「远程 LLM 失败后是否允许退回平台内置规则」。
    # True  = 允许降级（推荐，也是平台默认）：欠费 / 超时 / 5xx 时数据分析仍能跑出真实结果；
    # False = 远程失败就直接让这次运行失败，不做任何「看起来答上了」的包装。
    # 它必须真的被读取（runtime._direct_chat / _compose_answer / planner._fallback_allowed），
    # 否则就只是一个显示在设置页上的假开关。
    AGENT_ALLOW_MODEL_FALLBACK: bool = True
    AGENT_ENABLE_TOOL_RETRIEVAL: bool = True
    AGENT_TOOL_RETRIEVAL_TOP_K: int = 16
    AGENT_TOOL_RETRIEVAL_MIN_SCORE: float = 0.08
    AGENT_ENABLE_RESULT_COMPRESSION: bool = True
    AGENT_ENABLE_PLAN_CACHE: bool = True
    AGENT_PLAN_CACHE_MAX_ITEMS: int = 64
    # 第一层改造：规划前的 Pre-flight 检查。关闭后行为与改造前一致（直接规划）。
    AGENT_PREFLIGHT_ENABLED: bool = True
    # Pre-flight 是否读取列结构（只读 Parquet schema，不加载数据行）。
    # 关掉后「目标列是否明确」「任务类型是否矛盾」两项检查会自动跳过。
    AGENT_PREFLIGHT_READ_SCHEMA: bool = True

    # =========================================================
    # 数据规模与吞吐（大数据接入）
    # =========================================================
    # 单文件上传上限。默认 2 GiB；历史值曾是硬编码 100 MB，与「大数据平台」
    # 定位不符。放在配置里是为了让部署方按磁盘/内存实际容量调整，
    # 而不是改代码常量。
    MAX_UPLOAD_SIZE_BYTES: int = 2 * 1024 * 1024 * 1024
    # 上传时分块读取的块大小（流式，内存占用与块大小同阶）。
    UPLOAD_CHUNK_SIZE_BYTES: int = 4 * 1024 * 1024
    # 是否启用流式入库（scan_* + sink_parquet）。关闭后退回「整表物化再写」，
    # 便于在排查 Polars 流式引擎差异时做 A/B。
    INGEST_STREAMING_ENABLED: bool = True
    # 流式入库的目标行组大小（Parquet row group，单位=行）。行组越小 →
    # 后续投影/谓词下推的粒度和并发越好，但文件元数据开销越大。
    INGEST_ROW_GROUP_ROWS: int = 262144
    # CSV/NDJSON 的 schema 推断采样行数；推断失败时回退为全字符串列（不报错）。
    INGEST_SCHEMA_INFER_ROWS: int = 50000
    # 物化兜底路径（xlsx / arff / 标准 JSON）允许的最大体量：超过则明确拒绝，
    # 而不是把进程 OOM 掉再报 500。
    INGEST_MAX_MATERIALIZE_BYTES: int = 1024 * 1024 * 1024
    # ★ 分块入库的单块字节数。这是「数据规模上限」的真正旋钮：
    # 实测 Polars 的 scan_csv → sink_parquet 在 1.44 上并不真正流式（峰值 ≈ 全量物化，
    # 见 docs/大数据规模优化与吞吐提升方案.md 的基准表），因此超大文本文件改走
    # 「按字节切块 → 逐块解析 → 增量写行组」，内存 ≈ 块大小，与文件总体积无关。
    # 实测：32 MB 块能把「数据翻倍时的内存倍率」压到 0.97×；调到 64 MB 反而升到 1.20×
    # （pyarrow 行组缓冲放大），因此除非有明确实测依据，不要轻易调大。
    INGEST_CHUNK_BYTES: int = 32 * 1024 * 1024
    # ★ 自适应分界的文件大小。两条路径的实测取舍：
    #   - sink（scan_csv → sink_parquet）：边际吞吐约 730 MB/s，但峰值内存 ≈ 3× 文件体积；
    #   - chunked（分块）：内存恒定约 400 MB，但边际吞吐约 120 MB/s（少了 Polars 的并行压缩）。
    # 因此「小到装得下就用快的，大到装不下就用有界的」。设为 0 表示一律分块
    # （内存最省的部署），设为极大值表示一律走 sink（内存充裕的专用机）。
    INGEST_STREAMING_THRESHOLD_BYTES: int = 256 * 1024 * 1024
    # 版本快照是否用 scan_parquet（懒执行 + 投影/谓词下推）替代全量解码。
    DATASET_LAZY_SCAN_ENABLED: bool = True

    # =========================================================
    # 机器学习：内存治理（大数据集训练的硬约束）
    # =========================================================
    # ★ 单次训练的最大样本数（0 = 不限制）。超过时**随机抽样**并在结果里
    # 显式告警。原因：sklearn 的估计器几乎都要求稠密 numpy 矩阵，
    # 10,000,000 行 × 767 列的 one-hot 结果 = 57.1 GiB，必然 OOM
    # （实测 numpy._core._exceptions._ArrayMemoryError）。
    # 抽样是有损的，所以绝不静默进行 —— 结果里会带 sampled 标记与原始行数。
    ML_MAX_TRAIN_ROWS: int = 200_000
    # ★ 稠密特征矩阵的内存预算（字节）。预处理输出超过它时，给出**可操作的
    # 中文报错**（提示改用 ordinal 编码 / 调小 ML_MAX_TRAIN_ROWS / 关闭抽样前先扩内存），
    # 而不是让 numpy 抛 "Unable to allocate 57.1 GiB"。这是抽样之外的兜底安全网：
    # 即使调用方把 ML_MAX_TRAIN_ROWS 设为 0，也不会把进程打挂。
    ML_MAX_DENSE_BYTES: int = 2 * 1024 * 1024 * 1024
    # one-hot 单列的最大类别数。超过时把低频类别合并为一个「其他」列
    # （sklearn 的 max_categories）。Origin/Dest 这类 300 量级的高基数列
    # 会让特征数暴涨，既是内存问题也是统计问题。
    # 设为 0 表示不合并（保留旧行为，仅建议在小基数数据上使用）。
    ML_ONEHOT_MAX_CATEGORIES: int = 50
    # 轮廓系数（silhouette）的采样上限。它的复杂度是 O(n²)，
    # 在千万行上既算不完也算不下，必须采样。
    ML_MAX_SILHOUETTE_SAMPLES: int = 20_000

    # =========================================================
    # 数据库连接器（拓展功能）
    # =========================================================
    # 连接器口令的加密密钥（Fernet，32 字节 urlsafe base64）。
    # 留空 ⇒ 首次启动自动生成并写入 {MODEL_ROOT}/connector_secret.key（权限 0600）。
    # 生产环境应显式注入，避免多实例各自生成不同密钥导致解不开。
    CONNECTOR_SECRET_KEY: str = ""
    # 单次连接/查询超时（秒）与连接池大小。
    CONNECTOR_POOL_SIZE: int = 5
    CONNECTOR_CONNECT_TIMEOUT_SECONDS: int = 10
    CONNECTOR_STATEMENT_TIMEOUT_SECONDS: int = 300
    # 抽取批次大小（行）。这是「常量内存」的关键旋钮：内存占用 ≈ 批大小 × 行宽。
    CONNECTOR_BATCH_ROWS: int = 50000
    # 单次抽取的最大行数上限（0 表示不限制）。防呆：避免误抽一张 10 亿行表
    # 把磁盘写满。
    CONNECTOR_MAX_ROWS: int = 0
    # 预览行数上限与保存连接器数量上限（防呆）。
    CONNECTOR_PREVIEW_ROWS: int = 200
    CONNECTOR_MAX_CONNECTORS: int = 100
    # 允许的方言白名单（逗号分隔）。默认只放开「零外部依赖」的方言，
    # Postgres / MySQL 需先安装对应驱动再放开。
    CONNECTOR_ALLOWED_DIALECTS: str = "sqlite,duckdb,postgresql,mysql"

    # 版本快照缓存（数据分析模块）。
    # DatasetVersion 不可变，因此缓存永不失效，只需 LRU 淘汰；
    # 一轮 EDA 浏览可把同一份 Parquet 的解码次数从 8~10 次降到 1 次。
    DATASET_FRAME_CACHE_ENABLED: bool = True
    DATASET_FRAME_CACHE_MAX_ITEMS: int = 16
    DATASET_FRAME_CACHE_MAX_BYTES: int = 512 * 1024 * 1024  # 512 MB
    # 单个分析请求的软超时（秒）：超过后重分析端点降级/拒绝，避免线程被长期占用。
    ANALYSIS_SOFT_TIMEOUT_SECONDS: float = 60.0
    # 相关性矩阵的输入行数抽样上限（超出则抽样，并在结果中标注 sampled）。
    ANALYSIS_CORRELATION_MAX_ROWS: int = 5000
    # 相关性矩阵的列数上限（超出则截断，并在结果中标注 truncated）。
    ANALYSIS_CORRELATION_MAX_COLUMNS: int = 30

    # 报告增强
    AGENT_REPORT_NARRATION: bool = True          # 用 LLM 基于真实工具结果撰写报告章节正文
    AGENT_REPORT_NARRATION_MAX_CHARS: int = 24000  # 喂给 LLM 的事实摘要上限
    # SSE 在「等待用户确认」期间保持连接的最长时间（秒）
    AGENT_SSE_CONFIRM_WAIT_SECONDS: float = 900.0
    # 通知 SSE：连接最长存活时间（秒）。到点主动断开，由 EventSource 自动重连，
    # 避免长连接的 goroutine/任务在服务端无限堆积。
    NOTIFICATION_SSE_MAX_SECONDS: float = 1800.0
    # 通知 SSE：服务端检查版本号变化的间隔（秒）。
    # 这是「一次进程内整数比较」，成本远低于让每个客户端各自拉一遍完整列表。
    NOTIFICATION_SSE_INTERVAL_SECONDS: float = 3.0

    @property
    def data_root_path(self) -> Path:
        return self._resolve_path(self.DATA_ROOT)

    @property
    def allowed_connector_dialects(self) -> tuple[str, ...]:
        """连接器允许的方言白名单（小写、去空、去重，保持声明顺序）。"""
        raw = [item.strip().lower() for item in (self.CONNECTOR_ALLOWED_DIALECTS or "").split(",")]
        seen: dict[str, None] = {}
        for item in raw:
            if item:
                seen.setdefault(item, None)
        return tuple(seen)

    def upload_limits(self) -> dict[str, Any]:
        """上传限额摘要（供前端展示，避免前端硬编码 100 MB）。"""
        return {
            "max_size_bytes": int(self.MAX_UPLOAD_SIZE_BYTES),
            "chunk_size_bytes": int(self.UPLOAD_CHUNK_SIZE_BYTES),
            "streaming_ingest": bool(self.INGEST_STREAMING_ENABLED),
        }

    @property
    def model_root_path(self) -> Path:
        return self._resolve_path(self.MODEL_ROOT)

    @property
    def database_url(self) -> str:
        """实际使用的数据库 URL（相对 sqlite 路径锚定 BACKEND_ROOT）。

        DATA_ROOT / MODEL_ROOT 已锚定 BACKEND_ROOT，但 DATABASE_URL 若写成相对路径
        （默认 ``sqlite:///./data/xiaoluo.db``），SQLAlchemy 会按**进程 cwd** 解析：
        从项目根目录启动时会连到根目录的空库，所有数据接口报 ``no such table``。
        这里统一按 BACKEND_ROOT 解析，保证任意启动目录都指向同一个库。
        """
        return self._resolve_database_url(self.DATABASE_URL)

    @staticmethod
    def _resolve_database_url(url: str) -> str:
        """仅重写「相对路径」形式的 sqlite URL，绝对路径与内存库保持原样。"""
        if not url.startswith("sqlite") or "///" not in url:
            return url
        scheme, raw = url.split("///", 1)
        # sqlite:////abs/path（绝对路径）与 sqlite:///:memory:（内存库）原样返回
        if not raw or raw.startswith("/") or raw.startswith(":"):
            return url
        return f"{scheme}///{(BACKEND_ROOT / raw).resolve().as_posix()}"

    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

    def agent_context_summary(self) -> dict[str, Any]:
        return {
            "context_max_chars": self.AGENT_CONTEXT_MAX_CHARS,
            "context_max_tokens": self.AGENT_CONTEXT_MAX_TOKENS,
            "context_sections": {
                "user_request": self.AGENT_CONTEXT_USER_REQUEST_CHARS,
                "dataset": self.AGENT_CONTEXT_DATASET_CHARS,
                "task": self.AGENT_CONTEXT_TASK_CHARS,
                "permissions": self.AGENT_CONTEXT_PERMISSION_CHARS,
                "tools": self.AGENT_CONTEXT_TOOL_CHARS,
                "history": self.AGENT_CONTEXT_HISTORY_CHARS,
            },
            "history_messages": self.AGENT_CONTEXT_HISTORY_MESSAGES,
            "dataset_cache": {"enabled": self.AGENT_DATASET_CACHE_ENABLED, "max_items": self.AGENT_DATASET_CACHE_MAX_ITEMS},
            "max_steps": self.AGENT_MAX_STEPS,
            "llm_budget": {
                "max_calls": self.AGENT_LLM_MAX_CALLS,
                "max_input_tokens": self.AGENT_LLM_MAX_INPUT_TOKENS,
                "max_output_tokens": self.AGENT_LLM_MAX_OUTPUT_TOKENS,
                "max_total_tokens": self.AGENT_LLM_MAX_TOTAL_TOKENS,
            },
            "agent_policy": {
                "allow_model_fallback": self.AGENT_ALLOW_MODEL_FALLBACK,
                "enable_tool_retrieval": self.AGENT_ENABLE_TOOL_RETRIEVAL,
                "tool_retrieval_top_k": self.AGENT_TOOL_RETRIEVAL_TOP_K,
                "tool_retrieval_min_score": self.AGENT_TOOL_RETRIEVAL_MIN_SCORE,
                "enable_result_compression": self.AGENT_ENABLE_RESULT_COMPRESSION,
                "enable_plan_cache": self.AGENT_ENABLE_PLAN_CACHE,
                "plan_cache_max_items": self.AGENT_PLAN_CACHE_MAX_ITEMS,
                "preflight": {
                    "enabled": self.AGENT_PREFLIGHT_ENABLED,
                    "read_schema": self.AGENT_PREFLIGHT_READ_SCHEMA,
                },
            },
        }

    def remote_llm_available(self) -> bool:
        """远程 API 大模型是否可用＝总开关打开 **且** 已配置 API Key。

        所有构造远程 Provider 的地方都必须先问这里，否则「关闭开关」会被
        某条自己拼 Provider 的分支绕过（历史上 workflow 的 ai.analyze 就是这样）。
        """
        return bool(self.LLM_REMOTE_ENABLED and self.LLM_API_KEY)

    def llm_model_summary(self) -> dict[str, Any]:
        """返回可展示的模型策略；绝不返回 API Key。"""
        return {
            "provider_type": self.LLM_PROVIDER,
            "base_url": self.LLM_BASE_URL,
            "model": self.LLM_MODEL,
            "context_window": self.LLM_CONTEXT_WINDOW,
            "max_output_tokens": self.LLM_MAX_OUTPUT_TOKENS,
            "api_key_set": bool(self.LLM_API_KEY),
            # 总开关状态：前端据此显示「远程 API 已停用」，并决定是否采纳上面的配置。
            "remote_enabled": bool(self.LLM_REMOTE_ENABLED),
        }

    def local_router_summary(self) -> dict[str, Any]:
        """本地 Router 的可展示状态。`active=False` 表示链路完全未被触碰。"""
        mode = (self.LOCAL_ROUTER_MODE or "off").strip().lower()
        return {
            "mode": mode,
            "active": mode != "off",
            "confidence_threshold": self.LOCAL_ROUTER_CONFIDENCE_THRESHOLD,
            "model_dir": self.LOCAL_ROUTER_MODEL_DIR or str(self.model_root_path / "local_router"),
            "trace_max_bytes": self.LOCAL_ROUTER_TRACE_MAX_BYTES,
        }

    def masked_summary(self) -> dict[str, Any]:
        return {
            "app_name": self.APP_NAME,
            "app_env": self.APP_ENV,
            "debug": self.DEBUG,
            "database_url": self._mask_url(self.DATABASE_URL),
            "data_root": str(self.data_root_path),
            "model_root": str(self.model_root_path),
            "log_level": self.LOG_LEVEL,
            "llm_provider": self.LLM_PROVIDER,
            "llm_base_url": self.LLM_BASE_URL,
            "llm_model": self.LLM_MODEL,
            "llm_api_key_set": bool(self.LLM_API_KEY),
            "agent": self.agent_context_summary(),
            "local_router": self.local_router_summary(),
        }

    @property
    def cors_allowed_origins(self) -> list[str]:
        """跨域白名单（去空去重，保留声明顺序）。

        留空返回空列表 ⇒ 调用方不加 CORSMiddleware，等价于「只接受同源请求」。
        这样默认部署不会因为手滑配了 ``*`` 而把写接口暴露给任意站点。
        """
        raw = [item.strip() for item in (self.CORS_ALLOW_ORIGINS or "").split(",")]
        seen: dict[str, None] = {}
        for item in raw:
            if item:
                seen.setdefault(item, None)
        return list(seen)

    @staticmethod
    def _mask_url(url: str) -> str:
        if "://" not in url or "@" not in url:
            return url
        scheme, rest = url.split("://", 1)
        credentials, host = rest.rsplit("@", 1)
        if ":" not in credentials:
            return url
        username = credentials.split(":", 1)[0]
        return f"{scheme}://{username}:***@{host}"


# =====================================================================
# 工具召回的类目关键词表（ToolRegistry 语义检索用）
# =====================================================================
# 放在配置里而不是塞进检索函数体，是为了让「补一个关键词」不需要改
# retrieve_with_scores 的逻辑，也便于对不同语种分别维护。
#
# category -> 关键词列表。匹配时对 query 做子串包含判断，因此
#   - 中文词尽量写完整说法（"相关性"、"外部数据源"），不要写单字（会误命中很广）；
#   - 英文词写小写形式（query 进入检索前已 lower）。
# 注意：工具的 name / description / category 本身也参与匹配（见 registry.py），
# 这里只是给「描述里没写到、但用户会这么说」的说法兜底。
TOOL_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "data": (
        # 中文
        "清洗", "过滤", "筛选", "转换", "聚合", "合并", "去重", "排序", "填充", "处理", "修改",
        # 英文
        "filter rows", "filter row", "filtering", "clean data", "cleaning", "transform",
        "aggregate", "aggregation", "group by", "groupby", "deduplicate", "dedupe",
        "drop duplicates", "fill missing", "impute", "sort by", "reshape",
    ),
    "dataset": (
        "数据集", "数据集列表", "预览", "查看数据", "字段", "结构", "质量", "缺失",
        "重复", "版本", "样本", "画像", "统计",
        "dataset list", "datasets", "preview", "schema", "column names", "head rows",
        "profile", "missing values", "null count", "column types", "dtypes",
    ),
    "eda": (
        "分析", "探索", "分布", "相关", "相关性", "异常", "离群", "可视化", "图表", "趋势", "统计",
        "eda", "exploratory", "distribution", "histogram", "correlation", "outlier",
        "scatter", "describe", "plot", "chart", "summary stats",
    ),
    "ml": (
        "训练", "模型", "预测", "分类", "回归", "评估", "特征", "机器学习", "解释", "对比",
        "train", "training", "model", "predict", "prediction", "classify", "classification",
        "regress", "regression", "evaluate", "evaluation", "feature importance", "compare models",
    ),
    "workflow": (
        "workflow", "工作流", "流程", "编排", "节点", "运行流程", "流水线", "pipeline",
        "dag", "orchestration", "run pipeline",
    ),
    "report": (
        "报告", "导出报告", "实验报告", "pdf", "html", "markdown", "汇报", "结果文档",
        "export", "write report", "summarize findings",
    ),
    "connector": (
        "连接器", "数据库", "外部数据源", "导入数据", "mysql", "postgres", "postgresql", "sqlite",
        "sql server", "oracle", "duckdb", "数据接入",
        "import table", "external database", "connect to db", "read sql",
    ),
}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
````

### 9. `backend/app/api/v1/settings.py`

- `AgentSettingsUpdateRequest` 新增 `allow_model_fallback` 字段，去掉「接口里写死 `settings.AGENT_ALLOW_MODEL_FALLBACK = False`」的假配置行为。

````python
"""系统设置 API。"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app import __version__
from app.agent.llm.base import LLMException, LLMMessage
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.api.deps import get_storage_service
from app.core.config import settings
from app.schemas.common import ApiResponse
from app.storage.service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


class LLMTestRequest(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""


class LLMUpdateRequest(BaseModel):
    provider_type: str = "openai_compatible"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)


class AgentSettingsUpdateRequest(BaseModel):
    context_max_chars: int = Field(ge=1000, le=100000)
    context_sections: dict[str, int] = Field(default_factory=dict)
    history_messages: int = Field(ge=0, le=50)
    dataset_cache_enabled: bool
    dataset_cache_max_items: int = Field(ge=1, le=500)
    max_calls: int = Field(ge=1, le=50)
    max_input_tokens: int = Field(ge=256, le=200000)
    max_output_tokens: int = Field(ge=128, le=100000)
    max_total_tokens: int = Field(ge=512, le=300000)
    enable_tool_retrieval: bool
    tool_retrieval_top_k: int = Field(ge=1, le=30)
    tool_retrieval_min_score: float = Field(ge=0, le=1)
    enable_result_compression: bool
    enable_plan_cache: bool
    plan_cache_max_items: int = Field(ge=1, le=500)
    #: 是否允许「远程大模型调用失败 → 退回平台内置规则」。
    #: 可选项：前端未传时保留进程内当前值（与 context_window 等字段同口径），
    #: 这样老客户端不会因为缺字段就把开关打回默认。
    allow_model_fallback: bool | None = None


@router.get("/agent", response_model=ApiResponse[dict[str, Any]])
def agent_settings() -> ApiResponse[dict[str, Any]]:
    return ApiResponse[dict[str, Any]](data=settings.agent_context_summary())


@router.get("/local_router", response_model=ApiResponse[dict[str, Any]])
def local_router_settings() -> ApiResponse[dict[str, Any]]:
    """本地 Router 的只读状态（模式 / 产物路径 / 是否已加载）。

    只读是刻意的：模式切换会**改变线上路由行为**，不该由一个 HTTP 请求顺手完成
    （设置页可切换的那些参数都只影响上下文预算，不影响决策路径）。
    切档请改 `backend/.env` 的 `LOCAL_ROUTER_MODE` 后重启。

    这里附带「产物是否已就位/是否过期」，因为最常见的困惑是
    「明明打开了 shadow 却没有新数据」—— 那通常是模型不存在或平台工具清单已变。
    """
    summary = settings.local_router_summary()
    try:
        from app.local_router.model import artifact_path, get_model

        path = artifact_path()
        model = get_model() if summary["active"] else None
        summary |= {
            "artifact": str(path),
            "artifact_exists": path.exists(),
            "artifact_loaded": model is not None,
            "artifact_staleness": model.staleness() if model is not None else None,
        }
        if summary["active"] and model is None and path.exists():
            # 文件在却加载不了 ⇒ 过期或结构不符，这类静默失效必须显式暴露。
            summary["artifact_staleness"] = summary["artifact_staleness"] or (
                "产物存在但未能加载（可能已过期，请重跑 scripts/router/train_runtime_l1.py）"
            )
    except Exception as exc:  # noqa: BLE001 — 状态查询不该因模型层异常而 500
        summary["probe_error"] = f"{type(exc).__name__}: {exc}"
    return ApiResponse[dict[str, Any]](data=summary)


@router.put("/agent", response_model=ApiResponse[dict[str, Any]])
def update_agent_settings(body: AgentSettingsUpdateRequest) -> ApiResponse[dict[str, Any]]:
    """更新当前进程中的 Agent 策略；重启后仍以 .env/默认配置为准。"""
    section = body.context_sections
    required_sections = (
        "user_request", "dataset", "task", "permissions", "tools", "history"
    )
    for key in required_sections:
        value = int(section.get(key, 0))
        if value < 80 or value > body.context_max_chars:
            raise ValueError(f"上下文分区 {key} 必须在 80 到 context_max_chars 之间")
        setattr(settings, f"AGENT_CONTEXT_{'PERMISSION' if key == 'permissions' else key.upper()}_CHARS", value)

    settings.AGENT_CONTEXT_MAX_CHARS = body.context_max_chars
    settings.AGENT_CONTEXT_HISTORY_MESSAGES = body.history_messages
    settings.AGENT_DATASET_CACHE_ENABLED = body.dataset_cache_enabled
    settings.AGENT_DATASET_CACHE_MAX_ITEMS = body.dataset_cache_max_items
    settings.AGENT_LLM_MAX_CALLS = body.max_calls
    settings.AGENT_LLM_MAX_INPUT_TOKENS = body.max_input_tokens
    settings.AGENT_LLM_MAX_OUTPUT_TOKENS = body.max_output_tokens
    settings.AGENT_LLM_MAX_TOTAL_TOKENS = body.max_total_tokens
    # ★ 历史缺陷：这里写死成 False，于是设置页上那个「模型兜底」指示器永远显示关，
    # 用户改不动，而代码里也没有任何一处真的读它 —— 一个纯粹装饰性的配置。
    # 现在它真的控制「远程失败是否降级」，且未显式传入时**保留原值**。
    if body.allow_model_fallback is not None:
        settings.AGENT_ALLOW_MODEL_FALLBACK = bool(body.allow_model_fallback)
    settings.AGENT_ENABLE_TOOL_RETRIEVAL = body.enable_tool_retrieval
    settings.AGENT_TOOL_RETRIEVAL_TOP_K = body.tool_retrieval_top_k
    settings.AGENT_TOOL_RETRIEVAL_MIN_SCORE = body.tool_retrieval_min_score
    settings.AGENT_ENABLE_RESULT_COMPRESSION = body.enable_result_compression
    settings.AGENT_ENABLE_PLAN_CACHE = body.enable_plan_cache
    settings.AGENT_PLAN_CACHE_MAX_ITEMS = body.plan_cache_max_items
    return ApiResponse[dict[str, Any]](data=settings.agent_context_summary())


@router.get("/llm", response_model=ApiResponse[dict[str, Any]])
def llm_settings() -> ApiResponse[dict[str, Any]]:
    return ApiResponse[dict[str, Any]](data=settings.llm_model_summary())


@router.put("/llm", response_model=ApiResponse[dict[str, Any]])
def update_llm_settings(body: LLMUpdateRequest) -> ApiResponse[dict[str, Any]]:
    """更新当前进程的默认 Provider/Model；一次 Agent Run 仍只使用这一组配置。

    未提供的字段一律**保留原值**（与 api_key 的既有语义保持一致）：
    旧的实现无条件赋值 `settings.LLM_CONTEXT_WINDOW = body.context_window`，
    而调用方（如设置页的「应用到 Agent」）通常只带 base_url / model / api_key，
    于是 Pydantic 默认的 None 会把已配置的 context_window 与 max_output_tokens
    静默清空——表现为「同步一次模型，输出上限就丢了」。
    """
    settings.LLM_PROVIDER = body.provider_type.strip() or "openai_compatible"
    settings.LLM_BASE_URL = body.base_url.strip()
    settings.LLM_MODEL = body.model.strip()
    if body.context_window is not None:
        settings.LLM_CONTEXT_WINDOW = body.context_window
    if body.max_output_tokens is not None:
        settings.LLM_MAX_OUTPUT_TOKENS = body.max_output_tokens
    if body.api_key.strip():
        settings.LLM_API_KEY = body.api_key.strip()
    return ApiResponse[dict[str, Any]](data=settings.llm_model_summary())


class LLMRemoteToggleRequest(BaseModel):
    enabled: bool


@router.put("/llm/remote", response_model=ApiResponse[dict[str, Any]])
def update_llm_remote(body: LLMRemoteToggleRequest) -> ApiResponse[dict[str, Any]]:
    """启用 / 停用「远程 API 大模型」总开关（设置页那个开关）。

    关闭后所有构造远程 Provider 的入口（`deps.get_llm_provider` 与 workflow 的
    `ai.analyze`）都会拒绝构造 ⇒ Agent 走平台自带的规则规划器，用于测试平台自带小模型。

    **只切开关、不动凭据**：`LLM_API_KEY` / base_url / model 原样保留，重新开启立即生效。
    这样既不会出现「关一次就要重填 Key」，也不会和 PUT /llm 的条件赋值语义打架。
    """
    settings.LLM_REMOTE_ENABLED = body.enabled
    return ApiResponse[dict[str, Any]](data=settings.llm_model_summary())


def _provider_error_message(exc: Exception) -> str:
    """把 Provider 返回的安全错误信息带回设置页，避免只显示一个无意义的 HTTP 状态码。"""
    if not isinstance(exc, LLMException):
        return str(exc)
    details = exc.details if isinstance(exc.details, dict) else {}
    raw_body = details.get("body")
    if not raw_body:
        return exc.message
    try:
        payload = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            provider_message = error.get("message") or error.get("detail") or error.get("code")
            if provider_message:
                return f"{exc.message}：{provider_message}"
        provider_message = payload.get("message") or payload.get("detail")
        if provider_message:
            return f"{exc.message}：{provider_message}"
    return f"{exc.message}：{str(raw_body)[:300]}"


@router.post("/llm/test", response_model=ApiResponse[dict[str, Any]])
def test_llm_connection(body: LLMTestRequest) -> ApiResponse[dict[str, Any]]:
    base_url = body.base_url.strip() or settings.LLM_BASE_URL
    model = body.model.strip() or settings.LLM_MODEL
    api_key = body.api_key.strip() or settings.LLM_API_KEY
    if not api_key:
        return ApiResponse[dict[str, Any]](data={"ok": False, "message": "未提供 API Key"})
    provider = OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key, timeout=20)
    try:
        response = provider.chat([LLMMessage(role="user", content="只回复 OK")], temperature=0, max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        # 连通性测试按约定始终返回 200 + ok:false（前端要展示原因），
        # 但异常必须留痕，否则网关/证书/DNS 类故障在服务端完全不可见。
        logger.warning("LLM 连通性测试失败（base_url=%s model=%s）：%s", base_url, model, exc, exc_info=True)
        return ApiResponse[dict[str, Any]](data={"ok": False, "message": f"连接失败：{_provider_error_message(exc)}"})
    return ApiResponse[dict[str, Any]](data={"ok": True, "message": "连接成功", "model": response.model or model, "capabilities": provider.capability_snapshot()})


@router.get("/storage", response_model=ApiResponse[dict[str, Any]])
def storage_summary(storage: StorageService = Depends(get_storage_service)) -> ApiResponse[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    total_size = 0
    total_files = 0
    for meta in storage.list(""):
        total_size += int(meta.size)
        total_files += 1
        group = meta.key.split("/", 1)[0] if "/" in meta.key else "其他"
        item = groups.setdefault(group, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(meta.size)
    return ApiResponse[dict[str, Any]](data={"app": settings.APP_NAME, "version": __version__, "data_root": str(settings.data_root_path), "total_files": total_files, "total_bytes": total_size, "groups": groups})
````

### 10. `backend/tests/test_agent_llm_fallback.py`

- 新增 22 项回归测试，覆盖第十一节要求的全部后端场景（普通聊天兜底 / no_answer / 数据任务 Rule Planner 接管 / quota 429 只请求一次 / transient 429 有限重试 / Answer Source 不撒谎）。

````python
"""远程大模型「失败之后到底发生了什么」的兜底闭环回归测试。

为什么值得单独一个文件
----------------------
开关开着、凭据也在，但这一次调用没成功 —— 这是线上最常见、也最容易做假的一类失败
（欠费 / 限流 / 超时 / 网关 5xx）。它暴露过四个彼此独立的问题：

1. **闲聊兜底是假的**：`_direct_chat` 只是把异常文本拼成
   「暂时无法完成对话请求：…」，然后标 `LLM_ERROR_FALLBACK`。用户既没拿到回答，
   界面还告诉他「已降级到规则」。
2. **规划兜底是假的**：`build_plan_resilient` 只捕获 `PlanInvalidError`，
   而远程失败抛的是 `LLMException` ⇒ 欠费时整次数据分析直接失败。
3. **重试策略不分成因**：429 一律重试，于是「余额不足」被重试 3 次 + 指数退避空等。
4. **来源会撒谎**：见上第 1 条。

本文件钉住修复后的四条不变式，全部离线、毫秒级、**不调用任何真实大模型**。
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from app.agent import answer_source as A
from app.agent.context.builder import ContextBuilder
from app.agent.llm import openai_compatible as O
from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, is_fallbackable_error
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.agent.planner.planner import AgentPlanner
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.core.config import settings
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FailingLLM(LLMProvider):
    """永远失败的远程模型：模拟欠费 / 限流 / 超时 / 网关 5xx。"""

    name = "failing"

    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__()
        self._exc = exc or LLMException("账户余额不足")
        self.chat_calls = 0
        self.structured_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        self.chat_calls += 1
        self.begin_call()
        raise self._exc

    def structured_output(self, messages, schema, *, max_attempts=3, **kwargs):
        self.structured_calls += 1
        self.begin_call()
        raise self._exc


class OkLLM(LLMProvider):
    """永远成功的远程模型（用于对照「真的由模型产出」那一格）。"""

    name = "ok"

    def __init__(self, content: str = "远程模型给出的回答") -> None:
        super().__init__()
        self._content = content
        self.chat_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        self.chat_calls += 1
        self.begin_call()
        from app.agent.llm.base import LLMResponse

        return LLMResponse(content=self._content, model="ok-1")


def _frame(rows: int = 60) -> pl.DataFrame:
    half = max(rows // 2, 1)
    return pl.DataFrame(
        {
            "x1": [float(i + 1) for i in range(rows)],
            "x2": [float(rows - i) for i in range(rows)],
            "target": ["pos" if i < half else "neg" for i in range(rows)],
        }
    )


@pytest.fixture()
def env(db, storage):
    """数据引擎 + 实验 + 已带版本的数据集（60 行，Pre-flight 不会因样本量反问）。"""
    ds = DatasetService(db, storage)
    dataset = ds.create("toy", "玩具数据")
    ds.create_version(dataset.id, _frame())
    return {
        "db": db,
        "ds": ds,
        "engine": DataEngineService(ds),
        "exp": ExperimentService(db, ds),
        "dataset_id": dataset.id,
    }


def _runtime(env, llm) -> AgentRuntime:
    """构造带指定 LLM 的 Runtime；规划器同步注入（与 deps.get_agent_runtime 同口径）。"""
    return AgentRuntime(
        env["engine"],
        experiment_service=env["exp"],
        db=env["db"],
        llm=llm,
        planner=AgentPlanner(llm, max_steps=settings.AGENT_MAX_STEPS),
    )


# ---------------------------------------------------------------------------
# 一、普通聊天：远程失败后必须真的走本地兜底
# ---------------------------------------------------------------------------


def test_chat_falls_back_to_local_reply_when_remote_fails(env, monkeypatch):
    """★ P0：远程开着但调用失败 ⇒ 真的走 `local_reply()`，不是把异常文本当回答。"""
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "你好")

    assert run.status == RunStatus.COMPLETED, "降级成功就是完成，不该被打成失败"
    assert run.answer_source == A.LLM_ERROR_FALLBACK
    assert A.describe(run.answer_source)["by_llm"] is False
    # 真兜底：拿到的是本地应答，不是「暂时无法完成对话请求：<异常文本>」
    assert "暂时无法完成对话请求" not in run.final_answer
    assert run.final_answer and "AI 助手" in run.final_answer
    # 文案必须说「这一次调用失败」，不能谎称「远程已停用」（用户会跑去翻开关）
    assert "远程大模型已停用" not in run.final_answer


def test_chat_falls_back_to_notice_when_local_cannot_answer(env, monkeypatch):
    """★ P0：内置规则也答不上时，给的是**状态说明**，来源如实标 `NO_ANSWER`。

    「绝不能无响应」+「绝不能假装远程」：文字照给，但绝不标成
    `LLM_ERROR_FALLBACK`（没有真的降级成功）或 `REMOTE_LLM_*`。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    from app.agent.local_chat import no_llm_notice

    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "讲个笑话")

    assert run.status == RunStatus.COMPLETED
    assert run.final_answer == no_llm_notice(reason="账户余额不足")
    assert "账户余额不足" in run.final_answer, "真实失败原因必须透出给用户"
    assert run.answer_source == A.NO_ANSWER
    assert A.describe(run.answer_source)["by_llm"] is False


def test_chat_fails_when_fallback_is_disabled(env, monkeypatch):
    """`AGENT_ALLOW_MODEL_FALLBACK=False` ⇒ 远程失败就如实失败，不做任何包装。"""
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", False)
    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "你好")

    assert run.status == RunStatus.FAILED
    assert run.answer_source == A.NO_ANSWER
    assert not run.final_answer


def test_program_bug_is_not_masked_as_fallback(env, monkeypatch):
    """★ 不该盲目兜底的一类：`TypeError` 是代码缺陷，不是「远程不可用」。

    用规则应答去兜一个程序 bug，界面上会显示「降级成功」，
    于是真正的缺陷永远不会被发现。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(TypeError("unexpected keyword argument"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "你好")

    assert run.status == RunStatus.FAILED
    assert run.answer_source == A.NO_ANSWER
    assert is_fallbackable_error(TypeError("x")) is False


# ---------------------------------------------------------------------------
# 二、数据分析：远程规划失败后 Rule Planner 必须真的接管
# ---------------------------------------------------------------------------


def test_data_task_falls_back_to_rule_planner_and_executes(env, monkeypatch):
    """★ P0：欠费时数据分析仍然要跑出**真实工具结果**，而不是整次运行失败。"""
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(LLMException("Insufficient Balance"))
    rt = _runtime(env, llm)
    session = rt.create_session(dataset_ids=[env["dataset_id"]])

    run = rt.run(session, "检查一下数据质量")

    assert run.status == RunStatus.COMPLETED, run.error
    executed = [call.tool for call in run.tool_calls]
    assert "dataset.quality" in executed, f"规则规划器应接管并执行 dataset.quality，实际：{executed}"
    assert run.plan and run.plan.get("planner_fallback") is True
    # 数据是真的跑出来的；汇总时模型又挂了 ⇒ 汇总标降级，但 `by_llm` 必须是 False
    assert run.answer_source in (A.LLM_ERROR_FALLBACK, A.PLATFORM_RULES_SUMMARY)
    assert A.describe(run.answer_source)["by_llm"] is False
    assert run.final_answer


def test_build_plan_resilient_falls_back_only_once(env, monkeypatch):
    """★ 一次规划只允许切到 Rule Planner 一次；规则计划是终点，不回头再试远程。

    「禁止无限循环」的可观测证据：远程规划最多被调用 2 次（候选工具集 + 全量工具集），
    之后无论规则规划结果如何都不会再发起远程调用。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(LLMException("连接超时"))
    planner = AgentPlanner(llm, max_steps=settings.AGENT_MAX_STEPS)
    context = ContextBuilder(env["engine"]).build("检查一下数据质量", dataset_ids=[env["dataset_id"]])

    plan = planner.build_plan_resilient("检查一下数据质量", context, [], all_tools=TOOL_REGISTRY.list())

    assert plan.planner_fallback is True
    assert llm.structured_calls <= 2, f"远程规划只应尝试有限次，实际 {llm.structured_calls} 次"
    assert any(step.tool == "dataset.quality" for step in plan.steps)


def test_planning_fails_when_fallback_is_disabled(env, monkeypatch):
    """总闸关着 ⇒ 远程规划失败直接抛，不允许偷偷换成规则计划。"""
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", False)
    llm = FailingLLM(LLMException("连接超时"))
    planner = AgentPlanner(llm, max_steps=settings.AGENT_MAX_STEPS)
    context = ContextBuilder(env["engine"]).build("检查一下数据质量", dataset_ids=[env["dataset_id"]])

    with pytest.raises(LLMException):
        planner.build_plan_resilient("检查一下数据质量", context, [], all_tools=TOOL_REGISTRY.list())


# ---------------------------------------------------------------------------
# 三、429 / 配额：重试策略必须分成因
# ---------------------------------------------------------------------------


class _Resp:
    """最小 httpx.Response 替身。"""

    def __init__(self, status_code: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}
        self._json: Any = None
        import json as _json

        try:
            self._json = _json.loads(body)
        except ValueError:
            self._json = None

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("not json")
        return self._json


def test_quota_429_is_not_retried(monkeypatch):
    """★ P1：余额不足的 429 **一次都不重试** —— 重试一万次也不会成功。

    顺带验证它转成的是「可降级的 LLMException」：上层据此走本地兜底。
    """
    calls: list[int] = []

    def fake_post(url, **kwargs):
        calls.append(1)
        return _Resp(429, '{"error":{"code":"insufficient_quota","message":"Insufficient Balance"}}')

    monkeypatch.setattr(O.httpx, "post", fake_post)
    # base_url 唯一：熔断器按 base_url 隔离，避免与其它用例互相拖累
    provider = OpenAICompatibleProvider("http://quota.test", "m", "k")

    with pytest.raises(LLMException) as excinfo:
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1, f"配额类错误只应请求一次，实际 {len(calls)} 次"
    assert "余额" in excinfo.value.message or "配额" in excinfo.value.message
    assert excinfo.value.fallbackable is True


@pytest.mark.parametrize(
    "status,body",
    [
        (429, '{"error":{"code":"billing_hard_limit_reached"}}'),
        (429, '{"error":{"type":"insufficient_quota"}}'),
        (402, '{"error":{"message":"Insufficient Balance"}}'),
        (429, '{"error":{"message":"您的账户余额不足"}}'),
    ],
)
def test_quota_signatures_are_all_recognized(monkeypatch, status, body):
    """各家网关措辞不统一：code / type / message / HTTP 402 都要认得出来。"""
    calls: list[int] = []
    monkeypatch.setattr(O.httpx, "post", lambda url, **kw: (calls.append(1), _Resp(status, body))[1])
    provider = OpenAICompatibleProvider(f"http://quota-{status}-{abs(hash(body))}.test", "m", "k")

    with pytest.raises(LLMException):
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1


def test_transient_429_retries_then_succeeds(monkeypatch):
    """★ P1：真正的瞬时限流要**有限重试**，并且用服务端给的 `Retry-After`。"""
    monkeypatch.setattr(O, "_RETRY_BACKOFF_BASE_SECONDS", 0.0)  # 测试里不真睡
    statuses = [429, 429, 200]
    delays: list[float] = []
    real_sleep = O.time.sleep
    monkeypatch.setattr(O.time, "sleep", lambda s: delays.append(s))

    def fake_post(url, **kwargs):
        status = statuses.pop(0)
        if status == 200:
            return _Resp(200, '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}')
        return _Resp(429, '{"error":{"message":"Rate limit reached"}}', headers={"Retry-After": "0"})

    monkeypatch.setattr(O.httpx, "post", fake_post)
    provider = OpenAICompatibleProvider("http://ratelimit.test", "m", "k")

    response = provider.chat([LLMMessage(role="user", content="hi")])

    assert response.content == "ok"
    assert len(delays) == 2, f"两次失败 ⇒ 等待两次，实际 {len(delays)}"
    assert all(d == 0.0 for d in delays), "`Retry-After: 0` 应被优先采用而不是指数退避"
    assert real_sleep is not None


def test_retry_after_is_preferred_over_backoff(monkeypatch):
    """服务端明确给了 `Retry-After` 时以它为准（不再自己拍一个指数退避）。"""
    resp = _Resp(429, "{}", headers={"Retry-After": "7"})
    assert O._retry_after_seconds(resp) == 7.0
    assert O._retry_after_seconds(_Resp(429, "{}", headers={})) is None


def test_unretryable_4xx_is_not_retried(monkeypatch):
    """鉴权 / 参数错误重试没有意义：只请求一次。"""
    calls: list[int] = []
    monkeypatch.setattr(O.httpx, "post", lambda url, **kw: (calls.append(1), _Resp(401, '{"error":{"message":"invalid api key"}}'))[1])
    provider = OpenAICompatibleProvider("http://auth.test", "m", "k")

    with pytest.raises(LLMException):
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 四、Answer Source 不许撒谎（四种处境的完整对照）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,utterance,llm_factory,expect_source,expect_by_llm",
    [
        ("远程正常", "你好", lambda: OkLLM(), A.REMOTE_LLM_CHAT, True),
        ("远程失败+本地答得上", "你好", lambda: FailingLLM(LLMException("余额不足")), A.LLM_ERROR_FALLBACK, False),
        # 同一个问法换个 LLM 状态，来源必须跟着变 —— 这是「来源不许撒谎」最硬的一条
        ("远程失败+本地答不上", "讲个笑话", lambda: FailingLLM(LLMException("余额不足")), A.NO_ANSWER, False),
        ("远程未启用+本地答得上", "你好", lambda: None, A.PLATFORM_RULES_CHAT, False),
        ("远程未启用+本地答不上", "讲个笑话", lambda: None, A.PLATFORM_RULES_NOTICE, False),
    ],
)
def test_answer_source_never_lies(env, monkeypatch, label, utterance, llm_factory, expect_source, expect_by_llm):
    """★ 唯一判据是「这一次到底有没有走通远程」，不是配置项。

    `远程失败+本地答不上` 那一格最关键：它**不许**标 `LLM_ERROR_FALLBACK`
    （没有真的降级成功），也不许标 `REMOTE_*`（不是模型生成的）。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = llm_factory()
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, utterance)

    assert run.answer_source == expect_source, f"[{label}] 来源标错"
    described = A.describe(run.answer_source)
    assert described["by_llm"] is expect_by_llm, f"[{label}] by_llm 标错"
    if not expect_by_llm:
        assert described["model"] == "", f"[{label}] 非模型产出不许挂模型名"


def test_no_answer_case_still_answers_the_user():
    """`NO_ANSWER` 是「没有回答用户的问题」，不等于「界面上什么都不显示」。

    界面仍会收到一段如实的状态说明（含真实失败原因），只是来源标成未产出回答。
    """
    from app.agent.local_chat import no_llm_notice

    notice = no_llm_notice(reason="Insufficient Balance")
    assert "Insufficient Balance" in notice
    assert notice


def test_fallback_flag_default_is_on():
    """平台推荐默认开启：远程挂了也要能用，而不是让用户对着失败发呆。

    ★ 这里**绝不能**用 `importlib.reload(app.core.config)`：reload 会换掉模块里的
    `settings` 单例，而其它模块早已持有旧对象的引用（`from app.core.config import settings`
    在不同时刻 import 会拿到不同对象），于是后续用例的 monkeypatch 全部打空，
    表现为「改了 A 处、B 处用例挂」—— 用类直接取默认值即可，不碰进程内单例。
    """
    from app.core.config import Settings

    assert Settings(_env_file=None).AGENT_ALLOW_MODEL_FALLBACK is True
````

### 11. `backend/tests/test_answer_source.py`

- 原用例断言的是「报错文案 + LLM_ERROR_FALLBACK」这一错误行为，本轮按新的正确语义更新断言。

````python
"""钉住「这条回答到底是谁给的」——答案来源标记（`app/agent/answer_source.py`）。

为什么值得单独钉
----------------
用户的原话是「我不知道在开了 API 的情况下我的回答究竟是谁回答的」。这句话背后是
**三种在界面上长得一模一样、但含义完全不同**的来路：

| 场景 | 真实来源 | 应标记 |
| --- | --- | --- |
| 远程开着，调用成功 | 远程大模型 | `REMOTE_LLM_*` |
| 远程关着 | 平台内置规则 | `PLATFORM_RULES_*` |
| **远程开着，但这一次调用失败** | **仍是内置规则兜底** | `LLM_ERROR_FALLBACK` |

第三行是唯一「看起来像模型答的、其实是规则答的」的情况，也是最容易误判的一行。
本文件的核心用例就是钉死它：**凡是没走通 LLM 的分支，一律不许标成 `by_llm=True`**。

其余不变式：
- `describe()` 对未知/空值必须优雅退化（`NO_ANSWER`），不能让前端为未知值崩溃；
- 模型名只在「确实由模型产出」时附带 —— 降级时不带，避免界面显示误导性的模型标识；
- 来源标记必须随 `run.summary()` 下发，否则前端拿不到。

全部用例毫秒级、不启服务、不联网、**不调用任何真实大模型**。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent import answer_source as A
from app.agent.runtime.models import AgentRun, AgentSession, AgentTokenLedger
from app.agent.runtime.runtime import AgentRuntime


# ---------------------------------------------------------------------------
# 一、describe() 的契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,by_llm",
    [
        (A.REMOTE_LLM_CHAT, True),
        (A.REMOTE_LLM_SUMMARY, True),
        (A.PLATFORM_RULES_CHAT, False),
        (A.PLATFORM_RULES_NOTICE, False),
        (A.PLATFORM_RULES_SUMMARY, False),
        (A.LLM_ERROR_FALLBACK, False),
        (A.NO_ANSWER, False),
    ],
)
def test_describe_marks_by_llm_correctly(source, by_llm):
    d = A.describe(source)
    assert d["source"] == source
    assert d["by_llm"] is by_llm
    assert d["label"] and d["detail"], "每个来源都要有可展示的标签与说明"


@pytest.mark.parametrize("source", ["", None, "totally-unknown", 123])
def test_describe_degrades_for_unknown(source):
    """未知 / 空值统一落到 NO_ANSWER，避免前端为每个取值补兜底。"""
    d = A.describe(source)
    assert d["source"] == A.NO_ANSWER
    assert d["by_llm"] is False
    assert d["model"] == ""


def test_model_name_only_attached_when_llm_really_answered(monkeypatch):
    """降级场景绝不许挂模型名 —— 那会让用户以为「DeepSeek 答的这句」。"""
    monkeypatch.setattr(A, "llm_model_name", lambda: "deepseek-chat")
    assert A.describe(A.REMOTE_LLM_CHAT)["model"] == "deepseek-chat"
    assert A.describe(A.LLM_ERROR_FALLBACK)["model"] == ""
    assert A.describe(A.PLATFORM_RULES_CHAT)["model"] == ""


def test_describe_survives_settings_failure(monkeypatch):
    """模型名读不到只影响展示，不能让整条链路挂掉。"""
    def _boom():
        raise RuntimeError("配置不可用")

    monkeypatch.setattr(A, "llm_model_name", _boom)
    assert A.describe(A.REMOTE_LLM_CHAT)["model"] == ""
    assert A.describe(A.REMOTE_LLM_CHAT)["by_llm"] is True


# ---------------------------------------------------------------------------
# 二、闲聊链路 `_direct_chat`
# ---------------------------------------------------------------------------


def _run(text: str = "你好") -> AgentRun:
    return AgentRun(id="r-src", session_id="s-src", user_id="u-src", user_request=text)


def _session() -> AgentSession:
    return AgentSession(id="s-src", user_id="u-src")


class _FakeLLM:
    def __init__(self, content: str = "远程回答", *, raise_exc: Exception | None = None) -> None:
        self.content = content
        self._raise = raise_exc
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return SimpleNamespace(content=self.content)


def _stub(llm=None) -> SimpleNamespace:
    return SimpleNamespace(
        llm=llm,
        _emit=lambda *a, **k: None,
        _emit_usage=lambda *a, **k: None,
    )


def test_chat_without_llm_is_marked_as_platform_rules():
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(), run, session, None)
    assert run.answer_source == A.PLATFORM_RULES_CHAT
    assert A.describe(run.answer_source)["by_llm"] is False


def test_chat_without_llm_and_unknown_intent_is_notice():
    """规则答不了时给出的是「状态说明」，不是答案 —— 两者必须能区分。"""
    run, session = _run("讲个笑话"), _session()
    AgentRuntime._direct_chat(_stub(), run, session, None)
    assert run.answer_source == A.PLATFORM_RULES_NOTICE


def test_chat_with_llm_is_marked_as_remote():
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(_FakeLLM()), run, session, None)
    assert run.answer_source == A.REMOTE_LLM_CHAT
    assert A.describe(run.answer_source)["by_llm"] is True


def test_chat_with_llm_error_is_marked_as_degraded():
    """★ 本次透明化缺陷的核心用例：开关开着、凭据也在，但这一次调用失败了。

    历史缺陷：这条路径只是把异常文本拼成一句「暂时无法完成对话请求：…」就完了，
    然后把来源标成 `LLM_ERROR_FALLBACK` —— 既**没有真的兜底**（用户拿到的只是一句报错），
    又假装自己「已降级到规则」。

    现在：远程失败必须真的走 `local_reply()`，拿得到本地回答才算降级；
    来源照旧是 `LLM_ERROR_FALLBACK`（`by_llm=False`），但回答内容必须是真的本地应答。
    """
    run, session = _run("你好"), _session()
    AgentRuntime._direct_chat(_stub(_FakeLLM(raise_exc=RuntimeError("连接超时"))), run, session, None)
    assert run.answer_source == A.LLM_ERROR_FALLBACK
    assert A.describe(run.answer_source)["by_llm"] is False
    # 「降级」必须真的产出了替代回答，而不是把异常文本当成回答。
    assert "暂时无法完成对话请求" not in run.final_answer
    assert run.final_answer and "AI 助手" in run.final_answer
    # 文案必须说明是「这一次调用失败」，不能谎称「远程已停用」。
    assert "远程大模型已停用" not in run.final_answer


# ---------------------------------------------------------------------------
# 三、工具结果汇总链路 `_compose_answer`
# ---------------------------------------------------------------------------


def _run_with_tool_call() -> AgentRun:
    run = _run("看看这批数据的分布")
    run.tool_calls = [SimpleNamespace(status="ok", tool="eda.describe", result=SimpleNamespace(summary="3 列 / 100 行"))]
    return run


def _compose_stub(llm=None) -> SimpleNamespace:
    return SimpleNamespace(
        llm=llm,
        _emit=lambda *a, **k: None,
        _emit_usage=lambda *a, **k: None,
        # 汇总只关心压缩后的视图，这里给 None（等价于「无有效视图」）即可，
        # 不影响来源判定。
        _result_for_llm=lambda run, call, compact=False: None,
    )


def test_summary_without_llm_is_marked_as_platform_rules():
    run = _run_with_tool_call()
    answer = AgentRuntime._compose_answer(_compose_stub(), run)
    assert run.answer_source == A.PLATFORM_RULES_SUMMARY
    assert A.describe(run.answer_source)["by_llm"] is False
    assert "真实数据工具步骤" in answer, "规则汇总仍应交代数据是真的跑出来的"


def test_summary_with_llm_is_marked_as_remote():
    run = _run_with_tool_call()
    answer = AgentRuntime._compose_answer(_compose_stub(_FakeLLM("远程汇总的结论")), run)
    assert run.answer_source == A.REMOTE_LLM_SUMMARY
    assert answer == "远程汇总的结论"


def test_summary_with_llm_error_is_marked_as_degraded():
    """★ 同上：工具跑成功了、但汇总时大模型挂了 ⇒ 这段总结是规则拼的。"""
    run = _run_with_tool_call()
    answer = AgentRuntime._compose_answer(_compose_stub(_FakeLLM(raise_exc=RuntimeError("429"))), run)
    assert run.answer_source == A.LLM_ERROR_FALLBACK
    assert A.describe(run.answer_source)["by_llm"] is False
    assert "真实数据工具步骤" in answer


def test_summary_with_empty_llm_content_is_degraded():
    """模型返回空串也算「这次没拿到模型输出」，不能算远程生成。"""
    run = _run_with_tool_call()
    AgentRuntime._compose_answer(_compose_stub(_FakeLLM("")), run)
    assert run.answer_source == A.LLM_ERROR_FALLBACK


# ---------------------------------------------------------------------------
# 四、对外下发：summary() 必须带上来源
# ---------------------------------------------------------------------------


def test_summary_exposes_answer_source():
    run = _run()
    run.answer_source = A.REMOTE_LLM_SUMMARY
    data = run.summary()
    assert data["answer_source"]["source"] == A.REMOTE_LLM_SUMMARY
    assert data["answer_source"]["by_llm"] is True
    assert "label" in data["answer_source"]


def test_summary_of_unmarked_run_is_no_answer():
    """还没产出回答的运行（失败 / 取消）如实标成「未产出回答」，不留空让前端猜。"""
    run = _run()
    assert run.summary()["answer_source"]["source"] == A.NO_ANSWER


def test_failed_run_is_marked_no_answer():
    run = _run()
    events: list[str] = []
    AgentRuntime._fail(
        SimpleNamespace(_emit=lambda r, t, p, e: events.append(t), _emit_usage=lambda *a, **k: None),
        run,
        "第 3 步失败",
        None,
    )
    assert run.answer_source == A.NO_ANSWER
    assert A.describe(run.answer_source)["by_llm"] is False


# ---------------------------------------------------------------------------
# 五、Token 账本快照（`usage` 事件）：运行过程中就要能看到花了多少 / 省了多少
# ---------------------------------------------------------------------------


def test_emit_usage_publishes_a_snapshot_without_touching_behaviour():
    run = _run()
    run.token_ledger = AgentTokenLedger(llm_calls=2, actual_input_tokens=100, actual_output_tokens=40, actual_total_tokens=140, estimated_result_saved_tokens=900)
    captured: list[tuple[str, dict]] = []
    AgentRuntime._emit_usage(
        SimpleNamespace(_emit=lambda r, t, p, e: captured.append((t, p))),
        run,
        None,
    )
    assert [t for t, _ in captured] == ["usage"]
    payload = captured[0][1]["token_usage"]
    assert payload["actual"]["total_tokens"] == 140
    assert payload["optimization"]["estimated_saved_tokens"] == 900
    assert payload["budget"]["max_total_tokens"] >= 0


def test_emit_usage_never_breaks_the_run():
    """账本上报属于可观测性增强：即使抛异常也不能把主流程带崩。"""
    run = _run()

    def _boom(*_a, **_k):
        raise RuntimeError("事件流断了")

    # 不应抛出
    AgentRuntime._emit_usage(SimpleNamespace(_emit=_boom), run, None)
````

### 12. `frontend/src/features/agent/hooks/useAgentRun.ts`

- P0-1 根因一：会话镜像 Effect 依赖数组里放了 `onRunRestored`，而页面传的是内联箭头函数 ⇒ 每次 render 引用都变 ⇒ Effect 每次重跑 ⇒ `stopPolling() / abortStream() / clear() / setRun(null) / setInspectorTab('overview')`。
- 依赖数组收敛为 `[switchToken]`（只有真正切会话才变），回调经 `onRunRestoredRef` 读取最新值；`lastRunId` 同样改为 ref 读取。
- P0-1 根因二：页签被 `refreshRun()` / `applyEffects()` 无条件改写。新增 `tabPinnedRef` + `setInspectorTab`（用户手动选，置 pin）与 `autoInspectorTab`（自动切，pin 住则不动）两条路径。
- `startPolling()` 增加代次校验 `seq !== seqRef.current` 后立即停轮询，防止旧会话的轮询结果覆盖新会话状态。

````ts
/**
 * Agent 运行的全部状态与流程：发送、SSE、轮询、授权、取消。
 *
 * 页面不再持有这些状态，只负责布局；本 Hook 对外的契约是
 * 「给一组状态 + 几个动作」，不含任何 JSX。
 *
 * 与旧版相比的关键收敛：
 * - `activeRunId` 改为派生值（最后一个事件的 run_id ?? run.id），不再是需要在
 *   切会话时手工清空的 state —— 旧版漏清就会出现「对新会话点了旧 run 的允许」。
 * - 切会话的去重从 `switchSeqRef`（每个 await 后手工比对）改成 effect 的
 *   `cancelled` 闭包，少一类「忘了比对」的隐患。
 * - 进度 / 阶段的计算从 3 处各写一遍收敛为 `progressOfRun` / `stageOfRun`。
 *
 * ★ 会话镜像 Effect 的依赖纪律（本轮修的两个 P0 都出在这里）：
 * - 依赖数组里**只允许出现 `switchToken`**。页面传进来的 `onRunRestored` 是内联箭头
 *   函数，每次 render 都是新引用；一旦进依赖数组，effect 就会在每次 render 重跑，
 *   执行 `stopPolling() / abortStream() / clear() / setRun(null) / setInspectorTab("overview")`
 *   —— 表现为「AI 不回应、运行消失、SSE 被自己掐断、Inspector 页签锁死在概览」。
 * - 因此回调一律走 ref（`onRunRestoredRef` / `lastRunIdRef`），「会话切换」与
 *   「恢复完成后通知页面」两件事拆开：前者由 effect 负责，后者只是成功后的一次调用。
 * - `abortStream()` 只允许出现在三处：会话切换、组件卸载、新请求顶替旧请求。
 *   SSE 收到事件 / usage / progress / stage / busy 变化 / 普通 rerender 都不得中断它。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelRun,
  confirmRun,
  denyRun,
  getRun,
  sendMessage,
  type AgentToolInfo,
} from "../../../api/agent";
import { shouldAutoConfirm } from "../../../lib/toolPermissions";
import { toolDisplayName } from "../../../store/aiLab";
import type { AgentRun, ChatMessage, InspectorTab, PermissionRequest } from "../../../types/agent";
import { useAgentEvents, type EventEffects } from "./useAgentEvents";
import { useAgentUsage } from "./useAgentUsage";

const STATUS_LABEL: Record<string, string> = {
  pending: "待运行",
  planning: "规划中",
  running: "执行中",
  waiting_confirmation: "等待确认",
  completed: "已完成",
  failed: "失败",
};

/** 轮询失败上限：后端不可达 / run 被删除（404）时不能 2s 一次无限重试。 */
const MAX_POLL_FAILURES = 8;

function stageOfRun(full: AgentRun): string {
  if (full.status === "completed") return "任务完成";
  if (full.status === "failed") return "任务失败";
  return STATUS_LABEL[full.status] ?? full.status;
}

function progressOfRun(full: AgentRun): number {
  if (full.status === "completed" || full.status === "failed") return 100;
  const total = full.plan?.steps?.length || 0;
  const done = full.tool_call_count || 0;
  if (total) return Math.min(95, Math.round((done / total) * 100));
  return Math.min(90, 10 + (full.events?.length ?? 0) * 5);
}

function errText(e: unknown, fallback: string): string {
  return e instanceof Error ? e.message : fallback;
}

export interface UseAgentRunOptions {
  sessionId: string | null;
  /** 当前会话最后一次运行的 id；切会话时用它在面板里回填上一次执行。 */
  lastRunId: string | null;
  /** 会话切换计数（由 useAiSession 提供）：只有显式切换 / 新建 / 删除回退才 +1。 */
  switchToken: number;
  datasetIds: number[];
  tools: AgentToolInfo[];
  /** 没有会话时自动建一个，返回会话 id；失败返回 null。 */
  ensureSession: (title: string) => Promise<string | null>;
  appendMessage: (message: ChatMessage) => void;
  onError: (message: string | null) => void;
  onNotice: (message: string | null) => void;
  /** 切会话时成功回填了上一次运行——页面据此自动展开运行面板。 */
  onRunRestored?: () => void;
}

export function useAgentRun(opts: UseAgentRunOptions) {
  const { sessionId, lastRunId, switchToken, datasetIds, tools, ensureSession, appendMessage, onError, onNotice, onRunRestored } = opts;

  const [run, setRun] = useState<AgentRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("等待任务");
  const [permission, setPermission] = useState<PermissionRequest | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [inspectorTab, setInspectorTabState] = useState<InspectorTab>("overview");
  /**
   * 用户是否**手动**切过 Inspector 页签。
   *
   * 历史缺陷：`tool_call` 事件（切到「活动」）与 `refreshRun()`（按有无工具调用切到
   * 「工具链 / 活动」）都会在用户不知情时改写页签。用户切到「Token」看用量，
   * 下一个工具调用就把他弹回「活动」—— 页签永远停不住。
   * 现在：用户点过一次之后，自动切换一律让位，直到新一轮任务 / 切会话才复位。
   */
  const tabPinnedRef = useRef(false);

  const { events, setEvents, push, clear } = useAgentEvents(tools);
  const { usage, live, history, resetHistory } = useAgentUsage(events, run, busy);

  const pollRef = useRef<number | null>(null);
  /** SSE 流的取消句柄：离开页面 / 切会话时主动中断，避免后端 tail 线程挂到超时上限。 */
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false);
  /** 已自动放行过的授权请求（run_id:step_index:tool），避免同一请求被无限自动确认。 */
  const autoAllowedRef = useRef<Set<string>>(new Set());
  /** 会话代次：任何 await 之后比对，不是最新一代就丢弃结果。 */
  const seqRef = useRef(0);

  // ★ 页面传进来的回调每次 render 都是新引用。放进 effect 依赖数组就会让
  // 「会话镜像 effect」每次 render 重跑（= 掐断 SSE + 清空面板），
  // 所以这里一律用 ref 持有最新值，依赖数组里只留真正的变化源。
  const onRunRestoredRef = useRef(onRunRestored);
  onRunRestoredRef.current = onRunRestored;
  const lastRunIdRef = useRef(lastRunId);
  lastRunIdRef.current = lastRunId;

  /** 用户手动切页签：置上「用户已选择」标记，之后不再被自动切换覆盖。 */
  const setInspectorTab = useCallback((tab: InspectorTab) => {
    tabPinnedRef.current = true;
    setInspectorTabState(tab);
  }, []);

  /** 自动切页签（`tool_call` 事件 / 刷新运行详情）：用户没手动选过时才生效。 */
  const autoInspectorTab = useCallback((tab: InspectorTab) => {
    if (tabPinnedRef.current) return;
    setInspectorTabState(tab);
  }, []);

  // 运行中 SSE 事件已经带 run_id；流结束后才有 run 对象。二者取其一即可，
  // 不再单独维护一个需要在切会话时清空的 activeRunId。
  const activeRunId = events.length ? events[events.length - 1].run_id : (run?.id ?? null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const abortStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (runId: string) => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      const seq = seqRef.current;
      let failures = 0;
      // 后台标签页暂停轮询：`document.hidden` 为 true 时跳过本次请求，回到前台自动恢复。
      pollRef.current = window.setInterval(async () => {
        if (document.hidden) return;
        try {
          const full = await getRun(runId);
          // ★ 轮询跨越了 await：这期间可能已经切了会话。把旧会话的结果写回界面
          // 会把「新会话 + 旧运行」混在一起（`activeRunId` 也会跟着指向旧 run）。
          if (seq !== seqRef.current) {
            stopPolling();
            return;
          }
          failures = 0;
          setRun(full);
          setEvents(full.events ?? []);
          setProgress(progressOfRun(full));
          setStage(stageOfRun(full));
          if (full.pending_confirmation) setPermission(full.pending_confirmation);
          if (full.status !== "pending" && full.status !== "planning" && full.status !== "running") {
            stopPolling();
          }
        } catch (e) {
          failures += 1;
          if (failures >= MAX_POLL_FAILURES) {
            stopPolling();
            setStage("状态获取失败");
            onError(`无法获取运行状态（${errText(e, "网络错误")}），请刷新页面后重试。`);
          }
        }
      }, 2000);
    },
    [setEvents, onError, stopPolling],
  );

  /** 拉取单次运行详情并落到面板（SSE onDone 之后使用）。 */
  const refreshRun = useCallback(
    async (runId: string) => {
      const seq = seqRef.current;
      try {
        const full = await getRun(runId);
        if (seq !== seqRef.current) return;
        setRun(full);
        setEvents(full.events ?? []);
        setProgress(progressOfRun(full));
        setStage(stageOfRun(full));
        if (full.pending_confirmation) setPermission(full.pending_confirmation);
        // ★ 只做「用户没手动选过」时的自动定位，绝不把用户正在看的页签打回去。
        autoInspectorTab(full.tool_calls.length ? "chain" : "activity");
      } catch {
        /* 流已结束，轮询会兜底 */
      }
    },
    [setEvents, autoInspectorTab],
  );

  /** 把一条事件的增量应用到界面。授权请求在这里触发自动放行判定。 */
  const applyEffects = useCallback(
    (fx: EventEffects, runId: string) => {
      if (fx.stage) setStage(fx.stage);
      if (fx.progress !== undefined) setProgress(fx.progress);
      if (fx.notice) onNotice(fx.notice);
      // ★ 收到 SSE 事件**不得**打断流，也不得把用户手动选中的页签改写掉。
      if (fx.tab) autoInspectorTab(fx.tab);
      if (fx.append) appendMessage(fx.append);
      if (!fx.permission) return;
      const req = fx.permission;
      const autoKey = `${runId}:${req.step_index}:${req.tool}`;
      // 设置页把该工具设为「自动放行」时直接替用户确认；仍失败则回退为手动弹窗。
      if (shouldAutoConfirm(req.tool, tools) && !autoAllowedRef.current.has(autoKey)) {
        autoAllowedRef.current.add(autoKey);
        void autoAllow(runId, req);
        return;
      }
      setPermission(req);
      setStage("等待确认");
      setProgress(95);
    },
    // autoAllow 在下方定义，此处读取的是最近一次渲染的闭包；它只用 ref 与 setter，行为稳定。
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools, appendMessage, onNotice, autoInspectorTab],
  );

  /** 设置页已把该工具设为「自动放行」时的免确认路径。失败回退为手动授权弹窗。 */
  const autoAllow = useCallback(
    async (runId: string, req: PermissionRequest) => {
      const seq = seqRef.current;
      try {
        const resumed = await confirmRun(runId);
        if (seq !== seqRef.current) return;
        setRun(resumed);
        setEvents(resumed.events ?? []);
        setPermission(null);
        onNotice(`已按设置自动放行「${toolDisplayName(req.tool, tools)}」。`);
        if (resumed.final_answer) {
          appendMessage({
            role: "assistant",
            content: resumed.final_answer,
            source: resumed.answer_source ?? null,
          });
        }
        if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) {
          setPermission(resumed.pending_confirmation);
        }
        startPolling(runId);
      } catch (e) {
        if (seq !== seqRef.current) return;
        setPermission(req);
        onError(errText(e, "自动放行失败，请手动确认"));
      }
    },
    [tools, appendMessage, onNotice, onError, setEvents, startPolling],
  );

  const send = useCallback(
    async (content: string) => {
      if (!content.trim() || sendingRef.current) return;
      sendingRef.current = true;
      // 先建会话再动运行状态：ensureSession 会写入 sessionId，
      // 若把 setProgress(5) 放在它之前，切会话 effect 有可能在之后把进度清零。
      setBusy(true);
      onError(null);
      // 新一轮任务：上一轮的「用户手动选过页签」只属于上一轮，这里复位，
      // 让本轮的工具调用能自动定位到「活动 / 工具链」。
      tabPinnedRef.current = false;
      let sid = sessionId;
      if (!sid) {
        sid = await ensureSession(content.slice(0, 30) || "数据分析会话");
        if (!sid) {
          setBusy(false);
          sendingRef.current = false;
          return;
        }
      }
      // 发送即开启一条新 SSE 流：先中断上一条（正常情况下 busy 会挡住并发，这里是防御性兜底）。
      abortStream();
      const controller = new AbortController();
      abortRef.current = controller;
      appendMessage({ role: "user", content });
      setProgress(5);
      setStage("理解任务");
      clear();
      onNotice(null);
      setPermission(null);
      let runId: string | null = null;
      try {
        await sendMessage(sid, {
          content,
          stream: true,
          datasetIds,
          signal: controller.signal,
          onEvent: (ev) => {
            runId = ev.run_id;
            applyEffects(push(ev), ev.run_id);
          },
          onDone: () => {
            setBusy(false);
            sendingRef.current = false;
            if (runId) {
              void refreshRun(runId);
              startPolling(runId);
            }
          },
        });
      } catch (e) {
        // abort 导致的异常是「正常中断」（切会话 / 卸载 / 新请求顶替），不应当成发送失败报错。
        const aborted = controller.signal.aborted || (e instanceof DOMException && e.name === "AbortError");
        if (!aborted) onError(errText(e, "发送失败"));
        setBusy(false);
        sendingRef.current = false;
        if (runId && !aborted) startPolling(runId);
      }
    },
    [
      sessionId,
      datasetIds,
      ensureSession,
      appendMessage,
      onError,
      onNotice,
      abortStream,
      clear,
      push,
      applyEffects,
      refreshRun,
      startPolling,
    ],
  );

  /**
   * 确认授权。
   *
   * 历史缺陷：这里原来是 `if (!run || busy) return;` —— SSE 流还没结束时 run 仍为 null、
   * busy 为 true，于是点击「允许」必然静默 return：不发请求、不报错、弹窗也不关，
   * 而「拒绝」用的是事件里已经赋值的 activeRunId，所以看起来只有拒绝能点。
   * 现在与 deny / stop 统一取 `activeRunId`，并且不再用发送锁阻塞授权请求。
   */
  const allow = useCallback(async () => {
    const target = activeRunId;
    if (!target || confirming) return;
    const seq = seqRef.current;
    setConfirming(true);
    onError(null);
    try {
      const resumed = await confirmRun(target);
      if (seq !== seqRef.current) return; // 授权期间已切会话，丢弃结果
      setRun(resumed);
      setEvents(resumed.events ?? []);
      setPermission(null);
      onNotice(null);
      setStage(stageOfRun(resumed));
      if (resumed.final_answer) {
        appendMessage({ role: "assistant", content: resumed.final_answer });
      }
      if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) {
        setPermission(resumed.pending_confirmation);
      }
      startPolling(target);
    } catch (e) {
      // 失败时保留弹窗，让用户可以重试或直接拒绝，而不是把授权请求丢掉。
      if (seq === seqRef.current) onError(errText(e, "确认失败"));
    } finally {
      setConfirming(false);
    }
  }, [activeRunId, confirming, appendMessage, onError, onNotice, setEvents, startPolling]);

  const deny = useCallback(() => {
    // 拒绝必须通知后端终止 run，否则 run 卡在 WAITING_CONFIRMATION，会话被 409 锁死。
    setPermission(null);
    appendMessage({ role: "assistant", content: "已拒绝该高风险操作的授权，对应步骤不会执行。" });
    const target = activeRunId;
    if (!target) return;
    const seq = seqRef.current;
    void denyRun(target)
      .then((denied) => {
        if (seq !== seqRef.current) return;
        setRun(denied);
        setEvents(denied.events ?? []);
      })
      .catch(() => {
        /* 后端不可达时保留本地提示，轮询会兜底同步 */
      });
  }, [activeRunId, appendMessage, setEvents]);

  /**
   * 请求取消当前运行。
   * 后端只在「步骤边界」检查 cancel_requested，长步骤（训练 / 报告生成 / 大模型响应）
   * 期间不会立刻停止，因此补上轮询兜底，让进度条继续反映真实状态。
   */
  const stop = useCallback(async () => {
    const target = activeRunId;
    if (!target) return;
    setStage("正在取消…");
    onError(null);
    try {
      await cancelRun(target);
      setStage("已请求取消，将在当前步骤结束后停止");
      startPolling(target);
    } catch (e) {
      onError(errText(e, "取消失败"));
      setStage("取消失败");
    }
  }, [activeRunId, onError, startPolling]);

  /**
   * 会话镜像：切会话 / 新建 / 删除后回退时，中断上一条流与轮询、清空本轮状态，
   * 再按会话最后一次运行回填面板。
   *
   * ★★ 依赖数组里**只有 `switchToken`**，这是本轮 P0 修复的关键。
   *
   * 历史缺陷（两个独立成因，表现都是「AI 不回应 / 运行消失 / SSE 中断 / 页签锁死」）：
   * 1. `onRunRestored` 是页面传进来的**内联箭头函数**，每次 render 都是新引用。
   *    它在依赖数组里 ⇒ effect 每次 render 重跑 ⇒ 每次都 `abortStream()`。
   *    SSE 刚建立就在下一次 rerender（收到第一条事件就会触发）被自己掐断。
   * 2. `lastRunId` 也在依赖数组里：它派生自 `activeSession.run_ids[-1]`，
   *    会话列表刷新 / 新 run 写入都会让它变化 ⇒ 同样触发一整轮清理。
   *
   * 现在：
   * - 回调与 `lastRunId` 都走 ref，effect 只在**会话真正切换**时执行；
   * - 「会话切换」与「恢复完成后通知页面」拆开：通知只是恢复成功分支里的一次调用；
   * - `switchToken` 的语义保持不变（只有显式切换 / 新建 / 删除回退才 +1），
   *   `send()` 内部自动建会话依旧不算切换，不会打断刚收到的事件。
   */
  useEffect(() => {
    let cancelled = false;
    seqRef.current += 1;
    stopPolling();
    abortStream();
    autoAllowedRef.current.clear();
    clear();
    setRun(null);
    setPermission(null);
    setProgress(0);
    setStage("等待任务");
    tabPinnedRef.current = false;
    setInspectorTabState("overview");
    resetHistory();
    const runId = lastRunIdRef.current;
    if (!runId) {
      return () => {
        cancelled = true;
        stopPolling();
        abortStream();
      };
    }
    void (async () => {
      try {
        const full = await getRun(runId);
        if (cancelled) return;
        setRun(full);
        setEvents(full.events ?? []);
        setProgress(progressOfRun(full));
        setStage(stageOfRun(full));
        if (full.pending_confirmation) setPermission(full.pending_confirmation);
        setInspectorTabState(full.tool_calls.length ? "chain" : "activity");
        // 恢复成功才通知页面（展开面板）。这是「通知」，不是「切换」——
        // 它不参与任何清理逻辑，也不会反过来让 effect 再跑一次。
        onRunRestoredRef.current?.();
      } catch {
        if (!cancelled) {
          clear();
          setRun(null);
        }
      }
    })();
    return () => {
      cancelled = true;
      stopPolling();
      abortStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 见上方说明：依赖只允许 switchToken
  }, [switchToken]);

  // 卸载时收尾：中断 SSE，避免后端 tail 线程挂到超时上限。
  useEffect(
    () => () => {
      stopPolling();
      abortStream();
    },
    [stopPolling, abortStream],
  );

  return {
    run,
    events,
    busy,
    progress,
    stage,
    permission,
    confirming,
    inspectorTab,
    setInspectorTab,
    activeRunId,
    usage,
    live,
    history,
    send,
    allow,
    deny,
    stop,
  };
}
````

### 13. `frontend/src/pages/AI/index.tsx`

- `onRunRestored` 由内联箭头函数改为 `useCallback(() => setPanelOpen(true), [])`，引用恒定——这是让上面 Effect 依赖纪律真正生效的前提。

````tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./ai-lab.css";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { ChatPanel } from "../../features/agent/ChatPanel";
import { PermissionRequest } from "../../features/agent/PermissionRequest";
import { SmartAnalysisButton } from "../../features/agent/SmartAnalysisButton";
import { RunInspector } from "../../features/agent/RunInspector";
import { SessionSidebar } from "../../features/agent/SessionSidebar";
import { useAiSession, sessionTitle } from "../../features/agent/hooks/useAiSession";
import { useAgentRun } from "../../features/agent/hooks/useAgentRun";
import { generateReport } from "../../api/reports";
import { useAiLab } from "../../store/aiLab";
import type { AgentSession, InspectorTab } from "../../types/agent";

const QUICK_ACTIONS = [
  { title: "检查数据质量", description: "缺失值、重复值、字段类型与异常概览", prompt: "请检查当前关联的数据集质量，给出缺失值、重复值、字段类型和明显异常的摘要。" },
  { title: "探索数据", description: "生成关键统计，并指出值得进一步分析的变量", prompt: "请对当前关联的数据集做一次探索性分析，给出关键统计、变量关系和最值得继续分析的问题。" },
  { title: "设计机器学习实验", description: "根据数据与目标提出可执行的建模方案", prompt: "请根据当前数据集设计一个机器学习实验方案，说明目标变量、特征、候选模型、评价指标和下一步执行建议。" },
];

/**
 * AI 实验室页面：**只负责布局**。
 *
 * 会话域（列表 / 归档 / 删除 / 消息）在 useAiSession，运行域（发送 / SSE / 轮询 /
 * 授权 / 用量）在 useAgentRun；页面只把两者接起来，并持有纯界面状态
 * （面板展开、列表折叠、二次确认弹窗、错误与提示条）。
 */
export default function AI() {
  const tools = useAiLab((s) => s.tools);
  const navigate = useNavigate();

  // 页面级反馈条：会话域与运行域共用同一处展示，故提升到布局层。
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // 运行面板：空闲默认收起，任务启动 / 切到带运行的会话时自动展开；用户可随时手动收起。
  const [panelOpen, setPanelOpen] = useState(false);
  // 运行面板高度档位：默认 / 放大（面板原高度偏小，长工具链与用量表看不全）。
  const [panelSize, setPanelSize] = useState<"normal" | "large">("normal");
  // 会话列表：宽屏常开；窄屏（<1280）可折叠成一行入口。
  const [listOpen, setListOpen] = useState(true);
  const [generatingReport, setGeneratingReport] = useState(false);
  // 删除会话的二次确认目标（走统一 ConfirmDialog）。
  const [pendingDelete, setPendingDelete] = useState<AgentSession | null>(null);
  // 批量删除 / 清空：{ ids, all }，all=true 表示「清空全部」。
  const [pendingBulk, setPendingBulk] = useState<{ ids: string[]; all: boolean } | null>(null);

  const session = useAiSession({ onError: setError, onNotice: setNotice });
  const run = useAgentRun({
    sessionId: session.sessionId,
    lastRunId: session.lastRunId,
    switchToken: session.switchToken,
    datasetIds: session.selectedDatasets,
    tools,
    ensureSession: session.ensureSession,
    appendMessage: session.appendMessage,
    onError: setError,
    onNotice: setNotice,
    // 切到带历史运行的会话时回填成功 ⇒ 展开运行面板（与原行为一致）。
    //
    // ★ 必须 `useCallback(..., [])`：内联箭头函数每次 render 都是新引用，
    // 而它是 `useAgentRun` 里「会话镜像 Effect」的输入。引用不稳定 ⇒
    // 那个 effect 每次 render 重跑 ⇒ 执行 `abortStream() / clear() / setRun(null)`
    // ⇒ 表现为「刚发出去的消息没有回应、运行面板里的运行消失、SSE 被前端自己掐断、
    // Inspector 页签被锁死在概览」。
    onRunRestored: useCallback(() => setPanelOpen(true), []),
  });

  const displayMessages = useMemo(
    () => session.messages.filter((m, i, a) => i === 0 || m.content !== a[i - 1].content || m.role !== a[i - 1].role),
    [session.messages],
  );

  /**
   * 面板自动展开：只在「任务开始」这一次转变上触发（busy 由 false → true），
   * 并在出现待授权弹窗时强制展开。
   * 历史缺陷：原来依赖 [busy, permission] 且无条件 setPanelOpen(true)，
   * 导致运行期间用户点「收起运行面板」后，只要 permission 变化就被再次弹开，
   * 与注释里「用户可随时手动收起」的承诺不符。
   */
  const panelUserCollapsedRef = useRef(false);
  const prevBusyRef = useRef(false);
  useEffect(() => {
    if (run.busy && !prevBusyRef.current) panelUserCollapsedRef.current = false; // 新一轮任务开始：重置收起意图
    if (!run.busy) panelUserCollapsedRef.current = false;                         // 任务结束：复位
    if (run.permission) { setPanelOpen(true); return; }                           // 授权弹窗必须可见
    if (run.busy && !panelUserCollapsedRef.current) setPanelOpen(true);
    prevBusyRef.current = run.busy;
  }, [run.busy, run.permission]);

  /** 生成报告后直接跳到报告详情路由，不再整页刷新跳转列表页。 */
  async function generateDataReport() {
    const datasetId = session.selectedDatasets[0];
    if (!datasetId) return setError("请先在上方选择一个数据集");
    setGeneratingReport(true);
    setError(null);
    try {
      const report = await generateReport({
        dataset_id: datasetId,
        title: "AI 数据分析报告",
        include_quality: true,
        include_eda: true,
        include_ml: true,
        conclusions: run.run?.final_answer ? [run.run.final_answer] : undefined,
      });
      const key = typeof report?.metadata?.report_key === "string" ? report.metadata.report_key : "";
      setNotice("报告已生成，正在打开…");
      if (key) navigate(`/reports/${encodeURIComponent(key)}`, { state: { report } });
      else navigate("/reports", { state: { report } });
    } catch (e) {
      setError(e instanceof Error ? e.message : "报告生成失败");
    } finally {
      setGeneratingReport(false);
    }
  }

  const emptyState = (
    <div className="ai-start-panel">
      <h2>从一个问题开始你的数据实验</h2>
      <p className="muted">{session.selectedDatasets.length ? "选择一个起点快速开始，或直接在下方输入你的问题。" : "先在上方选择数据集，即可解锁快捷操作；也可以直接输入问题对话。"}</p>
      <div className="ai-action-grid">
        {QUICK_ACTIONS.map((action) => (
          <button key={action.title} className="ai-action-card" type="button" disabled={run.busy || !session.selectedDatasets.length} onClick={() => void run.send(action.prompt)}>
            <strong>{action.title}</strong>
            <span>{action.description}</span>
            <em>开始 →</em>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="ai-lab-page">
      <PageHeader
        title="AI 实验室"
        description="用对话驱动数据分析。"
        actions={<Link className="btn" to="/reports">报告中心 →</Link>}
      />

      {/* 上下文条：数据集选择 + 关联状态 + 授权提示合并为一行，原左栏摘要卡已并入此处。 */}
      <div className="ai-context-bar">
        <div className="ai-context-main">
          <span className="ai-context-title">数据上下文</span>
          <DatasetSelector value={session.selectedDatasets} onChange={session.setSelectedDatasets} multi compact showLabel={false} />
          <span className={`ai-chip${session.selectedDatasets.length ? "" : " warn"}`}>{session.selectedDatasets.length ? `已关联 ${session.selectedDatasets.length} 个数据集` : "未选择数据集"}</span>
          {run.permission && <span className="ai-chip danger">待授权确认</span>}
        </div>
        <div className="ai-context-actions">
          <SmartAnalysisButton disabled={run.busy || !session.selectedDatasets.length} busy={run.busy} onClick={() => void run.send("请对当前关联的数据集做一次智能分析：先检查数据质量，再给出关键统计与问题摘要。")} />
        </div>
      </div>

      {/* panel-large：面板放大时同步压低对话区，让两者仍在同一屏内可见（整页仍可下滑）。 */}
      <div className={`ai-workspace${panelOpen ? " panel-open" : ""}${panelOpen && panelSize === "large" ? " panel-large" : ""}`}>
        <SessionSidebar
          sessions={session.sessions}
          activeId={session.sessionId}
          listOpen={listOpen}
          onToggleList={() => setListOpen((v) => !v)}
          deletingId={session.deletingSessionId}
          bulkBusy={session.bulkDeleting}
          selectedIds={session.selectedVisible}
          allIds={session.allIds}
          allSelected={session.allSelected}
          someSelected={session.someSelected}
          onToggleSelect={session.toggleSelect}
          onToggleSelectAll={session.toggleSelectAll}
          onInvertSelect={session.invertSelect}
          onNew={() => void session.newSession()}
          onOpen={session.activate}
          onToggleArchive={(s) => void session.toggleArchive(s)}
          onArchiveSelected={(ids) => void session.archiveSelected(ids)}
          onRequestDelete={setPendingDelete}
          onRequestBulk={(ids, all) => setPendingBulk({ ids, all })}
        />

        {/* 主列：上对话、下运行面板（运行面板宽度与对话一致，便于观察长文本与工具链）。 */}
        <div className="ai-main-column">
          <section className="card ai-chat-column">
            <div className="ai-chat-header">
              <div>
                <h3>{session.sessionId ? "当前实验" : "开始一次 AI 分析"}</h3>
                <div className="muted ai-chat-sub">{session.selectedDatasets.length ? `已关联 ${session.selectedDatasets.length} 个数据集` : "先选择数据集，也可以直接创建会话。"}</div>
              </div>
              <div className="ai-chat-actions">
                {run.busy && <button className="btn" type="button" onClick={() => void run.stop()} title="在当前步骤结束后停止运行">停止</button>}
                <button className={`btn ai-panel-toggle${panelOpen ? " active" : ""}`} type="button" aria-pressed={panelOpen} onClick={() => { panelUserCollapsedRef.current = panelOpen; setPanelOpen((v) => !v); }}>
                  {panelOpen ? "收起运行面板" : "展开运行面板"}{(run.busy || run.permission) && <span className="ai-panel-dot" aria-hidden="true" />}
                </button>
                <button className="btn primary" type="button" disabled={!session.selectedDatasets.length || generatingReport} onClick={() => void generateDataReport()}>{generatingReport ? "报告生成中..." : "生成报告"}</button>
              </div>
            </div>
            {notice && <div className="ai-route-notice" role="status"><span>ℹ️</span><span style={{ flex: 1 }}>{notice}</span><button type="button" className="btn" style={{ padding: "2px 10px" }} onClick={() => setNotice(null)}>知道了</button></div>}
            <ChatPanel messages={displayMessages} busy={run.busy} onSend={(content) => void run.send(content)} showEmptyState={false} emptyState={emptyState} />
            {error && <div className="badge failed ai-error">{error}</div>}
          </section>

          {/* 运行面板：位于对话正下方，宽度与主区域一致，支持展开 / 收起。 */}
          {panelOpen && (
            <RunInspector
              run={run.run}
              events={run.events}
              busy={run.busy}
              progress={run.progress}
              stage={run.stage}
              tab={run.inspectorTab}
              onTabChange={(tab: InspectorTab) => run.setInspectorTab(tab)}
              usage={run.usage}
              live={run.live}
              history={run.history}
              tools={tools}
              large={panelSize === "large"}
              onToggleSize={() => setPanelSize((v) => (v === "large" ? "normal" : "large"))}
              onCollapse={() => { panelUserCollapsedRef.current = true; setPanelOpen(false); }}
            />
          )}
        </div>
      </div>
      <PermissionRequest request={run.permission} busy={run.confirming} onAllow={() => void run.allow()} onDeny={() => void run.deny()} />
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除这个会话？"
        message={
          <>
            会话「{pendingDelete ? sessionTitle(pendingDelete) : ""}」下的
            {pendingDelete?.run_ids?.length ?? 0} 条运行记录会一并删除，且无法恢复。
          </>
        }
        confirmText="删除会话"
        danger
        onConfirm={() => { if (pendingDelete) void session.removeSessions([pendingDelete.id]); setPendingDelete(null); }}
        onCancel={() => setPendingDelete(null)}
      />
      <ConfirmDialog
        open={pendingBulk !== null}
        title={pendingBulk?.all ? "清空全部会话？" : `删除选中的 ${pendingBulk?.ids.length ?? 0} 个会话？`}
        message={
          pendingBulk?.all
            ? <>将删除全部 {pendingBulk.ids.length} 个会话及其运行记录，且无法恢复。正在运行中的会话会跳过。</>
            : <>选中的 {pendingBulk?.ids.length ?? 0} 个会话及其运行记录会被删除，且无法恢复。正在运行中的会话会跳过。</>
        }
        confirmText={pendingBulk?.all ? "清空全部" : "批量删除"}
        danger
        onConfirm={() => { if (pendingBulk) void session.removeSessions(pendingBulk.ids); setPendingBulk(null); }}
        onCancel={() => setPendingBulk(null)}
      />
    </div>
  );
}
````

### 14. `frontend/src/lib/agentEvents.ts`

- `planning.plan_ready` 且 `planner_fallback === true` 时产出明确 notice：「远程规划不可用，已改用平台内置规则规划（执行结果仍是真实工具跑出来的）。」

````ts
/**
 * SSE 事件 → 界面状态增量的**纯翻译层**（零 React / 零 Store 依赖，可被 Node 单测）。
 *
 * 放在 lib/ 是为了能被测试直接导入：原先这段 if/else 埋在 300 行的 send() 里，
 * 进度回退、自动放行、校验提示这几处历史 bug 只能靠肉眼看界面回归。
 * 它只做翻译，不碰网络、不做异步；副作用（自动放行、起轮询）在 useAgentRun。
 */
import type { AgentEvent, ChatMessage, InspectorTab, PermissionRequest } from "../types/agent";

const STAGE_LABEL: Record<string, string> = {
  context_ready: "准备上下文",
  tools_retrieved: "检索工具",
  plan_ready: "执行计划",
  direct_chat: "普通对话",
};

/** 一条事件对界面的增量影响；未出现的字段表示「不改」。 */
export interface EventEffects {
  stage?: string;
  /** 定值，或「与当前值取较大者」——进度条只允许前进，回退只在重新规划时显式发生。 */
  progress?: number | ((prev: number) => number);
  notice?: string;
  tab?: InspectorTab;
  permission?: PermissionRequest;
  append?: ChatMessage;
}

/**
 * @param label 工具名 → 中文名。由调用方注入（组件层传 `(n) => toolDisplayName(n, tools)`），
 *              这样本文件不必依赖 Store / React。
 */
export function eventEffects(ev: AgentEvent, label: (tool: string) => string = (tool) => tool): EventEffects {
  const p = ev.payload ?? {};
  switch (ev.type) {
    case "route": {
      const chat = String(p.mode ?? "chat") === "chat";
      const reason = String(p.reason ?? "");
      return {
        stage: `${chat ? "对话模式" : "工具模式"} · ${reason}`,
        progress: (prev) => Math.max(prev, chat ? 40 : 8),
      };
    }
    case "planning": {
      const s = String(p.stage ?? "planning");
      const bump = s === "plan_ready" ? 25 : s === "tools_retrieved" ? 15 : 8;
      // 远程规划不可用、已降级到平台内置规则：这是「数据仍能跑出来、但计划不是大模型定的」
      // 的关键事实，必须让用户看见，而不是让他在结果里自己猜。
      const degraded = s === "plan_ready" && p.planner_fallback === true;
      return {
        stage: STAGE_LABEL[s] ?? s,
        progress: (prev) => Math.max(prev, bump),
        notice: degraded ? "远程规划不可用，已改用平台内置规则规划（执行结果仍是真实工具跑出来的）。" : undefined,
      };
    }
    case "tool_call":
      return {
        tab: "activity",
        stage: `执行工具：${label(String(p.tool ?? "tool"))}`,
        progress: (prev) => Math.max(prev, Math.min(90, prev + 8)),
      };
    case "tool_result":
      return { stage: `工具完成：${label(String(p.tool ?? "tool"))}` };
    case "validation": {
      if (p.valid !== false) return {};
      const errs = (p.errors as string[] | undefined)?.join("；") || "未知原因";
      return { notice: `第 ${Number(p.step_index ?? 0) + 1} 步校验失败：${errs}` };
    }
    case "replanning": {
      const notes = String(p.notes ?? "");
      // 重试时进度显式回退，让用户感知到「卡住后重来」，而不是停在同一个数字上。
      return {
        stage: p.retry ? `重试中 · ${notes}` : `重新规划 · ${notes}`,
        progress: (prev) => Math.max(5, prev - 15),
        notice: notes || undefined,
      };
    }
    case "permission": {
      if (!p.tool) return {};
      return {
        permission: {
          tool: String(p.tool),
          arguments: (p.arguments as Record<string, unknown>) ?? {},
          reason: String(p.reason ?? ""),
          step_index: Number(p.step_index ?? 0),
        },
      };
    }
    case "completed":
      return {
        progress: 100,
        stage: "任务完成",
        append:
          typeof p.final_answer === "string"
            ? {
                role: "assistant",
                content: p.final_answer,
                source: (p.answer_source as ChatMessage["source"]) ?? null,
              }
            : undefined,
      };
    case "failed":
      return {
        progress: 100,
        stage: "任务失败",
        append: { role: "assistant", content: `执行失败：${String(p.error ?? "未知错误")}` },
      };
    default:
      // usage 不在这里处理：Token 账本由 useAgentUsage 直接从事件列表派生。
      return {};
  }
}
````

### 15. `frontend/src/api/settings.ts`

- `updateAgentSettings` 新增可选 `allow_model_fallback`，让前端能真正写后端这个总闸。

````ts
/** 设置与系统状态 API。 */
import { client, unwrap } from "./client";

export interface LlmTestResult { ok: boolean; message: string; model?: string; capabilities?: Record<string, unknown> }
export interface LlmModelSettings { provider_type: string; base_url: string; model: string; context_window: number | null; max_output_tokens: number | null; api_key_set: boolean; remote_enabled: boolean }
export interface AgentSettings {
  context_max_chars: number;
  context_sections: { user_request: number; dataset: number; task: number; permissions: number; tools: number; history: number };
  history_messages: number;
  dataset_cache: { enabled: boolean; max_items: number };
  llm_budget: { max_calls: number; max_input_tokens: number; max_output_tokens: number; max_total_tokens: number };
  agent_policy: {
    allow_model_fallback: boolean;
    enable_tool_retrieval: boolean;
    tool_retrieval_top_k: number;
    tool_retrieval_min_score: number;
    enable_result_compression: boolean;
    enable_plan_cache: boolean;
    plan_cache_max_items: number;
  };
}
export interface StorageSummary { app: string; version: string; data_root: string; total_files: number; total_bytes: number; groups: Record<string, { files: number; bytes: number }> }

export function testLlmConnection(body: { base_url?: string; model?: string; api_key?: string }) { return unwrap<LlmTestResult>(client.post("/settings/llm/test", body)); }
export function getLlmSettings() { return unwrap<LlmModelSettings>(client.get("/settings/llm")); }
export function updateLlmSettings(body: { provider_type: string; base_url: string; model: string; api_key?: string }) { return unwrap<LlmModelSettings>(client.put("/settings/llm", body)); }
/**
 * 启用 / 停用「远程 API 大模型」总开关。
 *
 * 只切开关，不动凭据（base_url / model / api_key 原样保留）：关闭后 Agent 不再构造远程
 * Provider，改走平台自带的规则规划器；重新开启立即生效，不需要重填 Key。
 */
export function setRemoteLlm(enabled: boolean) { return unwrap<LlmModelSettings>(client.put("/settings/llm/remote", { enabled })); }
export function getAgentSettings() { return unwrap<AgentSettings>(client.get("/settings/agent")); }
export function updateAgentSettings(body: {
  context_max_chars: number;
  context_sections: AgentSettings["context_sections"];
  history_messages: number;
  dataset_cache_enabled: boolean;
  dataset_cache_max_items: number;
  max_calls: number;
  max_input_tokens: number;
  max_output_tokens: number;
  max_total_tokens: number;
  enable_tool_retrieval: boolean;
  tool_retrieval_top_k: number;
  tool_retrieval_min_score: number;
  enable_result_compression: boolean;
  enable_plan_cache: boolean;
  plan_cache_max_items: number;
  /**
   * 远程大模型调用失败后是否允许退回平台内置规则。
   *
   * ★ 这是**真开关**：后端 `runtime._direct_chat` / `_compose_answer` /
   * `planner._fallback_allowed` 都会读它。关掉意味着「远程挂了就如实失败」，
   * 不再是界面上那个仅供展示的假指示器。
   */
  allow_model_fallback?: boolean;
}) { return unwrap<AgentSettings>(client.put("/settings/agent", body)); }
export function getStorageSummary() { return unwrap<StorageSummary>(client.get("/settings/storage")); }
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
````

### 16. `frontend/src/pages/Settings/AgentModelPanel.tsx`

- 保存时透传 `allow_model_fallback`；策略区新增「模型兜底（远程失败降级）」勾选项。

````tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { getAgentSettings, getLlmSettings, updateAgentSettings, type AgentSettings, type LlmModelSettings } from "../../api/settings";
import { listTools, type AgentToolInfo } from "../../api/agent";
import { readBrowserLlm } from "../../lib/browserLlm";
import { applyLlmToBackend, backendModelFingerprint, browserModelFingerprint } from "../../lib/llmSync";
import { numberOrPrevious } from "../../lib/numberInput";
import { InfoHint } from "../../components/InfoHint";
import { ToolPermissionList } from "./ToolPermissionList";

type Props = { onToast?: (text: string) => void };

export default function AgentModelPanel({ onToast }: Props) {
  const [model, setModel] = useState<LlmModelSettings | null>(null);
  const [tools, setTools] = useState<AgentToolInfo[]>([]);
  const [saving, setSaving] = useState(false);
  const [syncingModel, setSyncingModel] = useState(false);
  const [draft, setDraft] = useState<AgentSettings | null>(null);
  const [browserModel, setBrowserModel] = useState(() => readBrowserLlm().model);
  const [loadError, setLoadError] = useState("");

  /**
   * onToast 用 ref 持有，effect 依赖保持空数组。
   *
   * 历史问题：effect 依赖写的是 [onToast]，而调用方一旦传入内联箭头函数，
   * 父组件每次重渲染都会产生新引用 → effect 重跑 → 重复三次网络请求
   * （getLlmSettings + getAgentSettings + listTools）并重复触发自动同步。
   */
  const onToastRef = useRef(onToast);
  useEffect(() => { onToastRef.current = onToast; }, [onToast]);
  const toast = useCallback((text: string) => onToastRef.current?.(text), []);

  /** 首次加载：读后端模型策略 + Agent 策略 + 工具目录，然后尝试一次静默同步。 */
  useEffect(() => {
    let cancelled = false;
    void Promise.all([getLlmSettings(), getAgentSettings(), listTools()])
      .then(([m, a, t]) => {
        if (cancelled) return;
        setModel(m);
        setDraft(a);
        setTools(t);
        setLoadError("");
        void autoSyncModel(m);
      })
      .catch(() => {
        if (!cancelled) setLoadError("无法读取 Agent 模型策略，请检查后端服务是否可用。");
      });
    return () => { cancelled = true; };
    // autoSyncModel 是本组件内的稳定逻辑（只依赖 ref 与 setState），无需进依赖数组
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function patch(partial: Partial<AgentSettings>) {
    setDraft((current) => current ? ({ ...current, ...partial }) : current);
  }

  function patchPolicy(partial: Partial<AgentSettings["agent_policy"]>) {
    setDraft((current) => current ? ({ ...current, agent_policy: { ...current.agent_policy, ...partial } }) : current);
  }

  function patchSection(key: keyof AgentSettings["context_sections"], value: number) {
    setDraft((current) => current ? ({
      ...current,
      context_sections: { ...current.context_sections, [key]: value },
    }) : current);
  }

  /**
   * 把浏览器里的 AI 配置应用到后端。
   *
   * 历史问题：旧实现无条件写入，于是「改了表单没保存 → 切走 → 切回来」会用
   * 浏览器里的旧值反复覆盖后端全局凭据。现在统一走 lib/llmSync，
   * 内部先比较 base_url + model 指纹，已一致就直接返回不写。
   *
   * @param auto true = 进入页面的静默同步（不弹提示、不显示 loading）
   * @param serverState 调用方已取到的后端摘要，避免重复请求
   */
  async function syncModel(auto = false, serverState: LlmModelSettings | null = null) {
    if (!auto) setSyncingModel(true);
    const result = await applyLlmToBackend({ force: !auto, serverState });
    if (!auto) setSyncingModel(false);

    switch (result.status) {
      case "applied":
        setModel(result.model);
        setBrowserModel(readBrowserLlm().model);
        toast("当前 AI 配置已应用到 Agent 后端");
        return true;
      case "already-synced":
        setModel(result.model);
        if (!auto) toast("当前 AI 配置与后端一致，无需重复同步");
        return true;
      case "not-configured":
        if (!auto) toast("请先在上方 AI 服务中填写并保存 Base URL、模型名和 API Key");
        return false;
      case "failed":
        if (!auto) toast(result.message);
        return false;
    }
  }

  /** 进入设置页时静默同步一次，避免用户忘记点同步导致后端仍用旧配置。 */
  async function autoSyncModel(serverState: LlmModelSettings | null) {
    const synced = await syncModel(true, serverState);
    if (synced) toast("已自动将当前 AI 配置应用到 Agent");
  }

  async function save() {
    if (!draft) return;
    setSaving(true);
    try {
      const saved = await updateAgentSettings({
        context_max_chars: draft.context_max_chars,
        context_sections: draft.context_sections,
        history_messages: draft.history_messages,
        dataset_cache_enabled: draft.dataset_cache.enabled,
        dataset_cache_max_items: draft.dataset_cache.max_items,
        max_calls: draft.llm_budget.max_calls,
        max_input_tokens: draft.llm_budget.max_input_tokens,
        max_output_tokens: draft.llm_budget.max_output_tokens,
        max_total_tokens: draft.llm_budget.max_total_tokens,
        enable_tool_retrieval: draft.agent_policy.enable_tool_retrieval,
        tool_retrieval_top_k: draft.agent_policy.tool_retrieval_top_k,
        tool_retrieval_min_score: draft.agent_policy.tool_retrieval_min_score,
        enable_result_compression: draft.agent_policy.enable_result_compression,
        enable_plan_cache: draft.agent_policy.enable_plan_cache,
        plan_cache_max_items: draft.agent_policy.plan_cache_max_items,
        // 真开关：关掉 ⇒ 远程失败直接失败；开启 ⇒ 退回平台内置规则。
        allow_model_fallback: draft.agent_policy.allow_model_fallback,
      });
      setDraft(saved);
      toast("Agent 设置已保存（当前后端进程生效）");
    } catch (error) {
      toast(error instanceof Error ? error.message : "Agent 设置保存失败");
    } finally {
      setSaving(false);
    }
  }

  /** 浏览器配置与后端生效配置是否指向同一组 base_url + model。 */
  const backendFp = backendModelFingerprint(model);
  const browserFp = browserModelFingerprint();
  const missingKey = Boolean(model && !model.api_key_set);
  const synced = Boolean(backendFp && browserFp && backendFp === browserFp && !missingKey);
  const browser = readBrowserLlm();

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            Agent 模型能力
            <InfoHint label="Agent 模型能力说明">
              一次 Agent 任务只使用当前选定的一个模型，不会自动切换或回退；「后端生效模型」是 Agent 实际调用的，
              「浏览器待应用模型」需在「AI 服务」中填写并应用后才生效。
            </InfoHint>
          </h3>
        </div>
        <span className={`badge ${synced ? "success" : "failed"}`}>
          {synced ? "已同步" : "未同步"}
        </span>
      </div>

      {loadError && <p className="settings-note settings-note-warn">{loadError}</p>}

      {model ? <>
        {/* 前后端两个状态并列显示：只显示一个「当前模型」正是旧版误导用户的根源。 */}
        <div className="settings-stat-grid">
          <div>
            <small>后端生效模型</small>
            <strong>{model.model || "—"}</strong>
          </div>
          <div>
            <small>浏览器待应用模型</small>
            <strong>{browser.model || "未填写"}</strong>
          </div>
        </div>

        {(missingKey || !synced) && (
          <p className="settings-note settings-note-warn">
            {missingKey && browser.api_key
              ? "后端尚未收到 API Key，Agent 会退化为规则规划器。点击下方按钮应用配置。"
              : "浏览器配置与后端生效配置不一致，点下方按钮将其应用到 Agent。"}
          </p>
        )}
      </> : !loadError && <p className="settings-empty">读取模型策略...</p>}

      <div className="settings-actions">
        <button className="btn primary" disabled={syncingModel} onClick={() => void syncModel()}>
          {syncingModel ? "同步中..." : "将浏览器配置应用到 Agent"}
        </button>
      </div>
    </section>

    <ToolPermissionList tools={tools} title="Agent 工具目录与授权" />

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            Agent 预算与策略
            <InfoHint label="Agent 预算与策略生效范围">
              这里的开关和数值会立即作用于当前后端进程，重启后仍以后端 .env / 默认配置为准。
            </InfoHint>
          </h3>
        </div>
        <button className="btn primary" disabled={saving || !draft} onClick={() => void save()}>
          {saving ? "保存中..." : "保存 Agent 设置"}
        </button>
      </div>
      {draft ? <>
        <div className="settings-form-grid">
          <NumberField label="最大 LLM 调用次数" min={1} max={50} value={draft.llm_budget.max_calls}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_calls: v } })} />
          <NumberField label="最大上下文字符数" min={1000} max={100000} value={draft.context_max_chars}
            onChange={(v) => patch({ context_max_chars: v })} />
          <NumberField label="输入 Token 上限" min={256} value={draft.llm_budget.max_input_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_input_tokens: v } })} />
          <NumberField label="输出 Token 上限" min={128} value={draft.llm_budget.max_output_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_output_tokens: v } })} />
          <NumberField label="总 Token 上限" min={512} value={draft.llm_budget.max_total_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_total_tokens: v } })} />
          <NumberField label="历史消息数量" min={0} max={50} value={draft.history_messages}
            onChange={(v) => patch({ history_messages: v })} />
        </div>

        <div className="settings-panel-heading" style={{ marginTop: 22 }}>
        <div>
          <h4 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            上下文分区预算
            <InfoHint label="上下文分区预算说明">
              分别控制用户请求、数据集、任务、权限、工具和历史进入 Agent 上下文的最大字符数。
            </InfoHint>
          </h4>
        </div>
        </div>
        <div className="settings-form-grid">
          <NumberField label="用户请求" min={80} value={draft.context_sections.user_request}
            onChange={(v) => patchSection("user_request", v)} />
          <NumberField label="数据集上下文" min={80} value={draft.context_sections.dataset}
            onChange={(v) => patchSection("dataset", v)} />
          <NumberField label="任务上下文" min={80} value={draft.context_sections.task}
            onChange={(v) => patchSection("task", v)} />
          <NumberField label="权限信息" min={80} value={draft.context_sections.permissions}
            onChange={(v) => patchSection("permissions", v)} />
          <NumberField label="工具描述" min={80} value={draft.context_sections.tools}
            onChange={(v) => patchSection("tools", v)} />
          <NumberField label="历史上下文" min={80} value={draft.context_sections.history}
            onChange={(v) => patchSection("history", v)} />
        </div>

        <div className="settings-status-list" style={{ marginTop: "var(--space-4)" }}>
          <label title="开启：远程大模型失败（欠费 / 超时 / 5xx）时退回平台内置规则，数据分析仍能跑出真实结果。关闭：远程失败就直接失败。"><span>模型兜底（远程失败降级）</span>
            <input type="checkbox" checked={draft.agent_policy.allow_model_fallback}
              onChange={(e) => patchPolicy({ allow_model_fallback: e.target.checked })} /></label>
          <label><span>Tool Retrieval</span>
            <input type="checkbox" checked={draft.agent_policy.enable_tool_retrieval}
              onChange={(e) => patchPolicy({ enable_tool_retrieval: e.target.checked })} /></label>
          <label><span>工具结果压缩</span>
            <input type="checkbox" checked={draft.agent_policy.enable_result_compression}
              onChange={(e) => patchPolicy({ enable_result_compression: e.target.checked })} /></label>
          <label><span>Plan Cache</span>
            <input type="checkbox" checked={draft.agent_policy.enable_plan_cache}
              onChange={(e) => patchPolicy({ enable_plan_cache: e.target.checked })} /></label>
          <label><span>数据集上下文缓存</span>
            <input type="checkbox" checked={draft.dataset_cache.enabled}
              onChange={(e) => patch({ dataset_cache: { ...draft.dataset_cache, enabled: e.target.checked } })} /></label>
        </div>
        <div className="settings-form-grid" style={{ marginTop: "var(--space-4)" }}>
          <NumberField label="工具召回数量" min={1} max={30} value={draft.agent_policy.tool_retrieval_top_k}
            onChange={(v) => patchPolicy({ tool_retrieval_top_k: v })} />
          <NumberField label="召回最低相关度" min={0} max={1} step={0.01} value={draft.agent_policy.tool_retrieval_min_score}
            onChange={(v) => patchPolicy({ tool_retrieval_min_score: v })} />
          <NumberField label="Plan Cache 容量" min={1} max={500} value={draft.agent_policy.plan_cache_max_items}
            onChange={(v) => patchPolicy({ plan_cache_max_items: v })} />
          <NumberField label="数据集缓存数量" min={1} max={500} value={draft.dataset_cache.max_items}
            onChange={(v) => patch({ dataset_cache: { ...draft.dataset_cache, max_items: v } })} />
        </div>
      </> : <p className="settings-empty">读取 Agent 策略...</p>}
    </section>
  </>;
}

/**
 * 数字输入框：清空时保持上一个有效值，而不是把空串变成 0。
 *
 * 历史问题：直接 `Number(e.target.value)` 时，用户清空输入框会得到 0，
 * 绕过 HTML 的 min 约束（min 只参与表单校验，不约束 React 受控值），
 * 提交后被后端 Pydantic 的 ge= 约束拦下返回 422——用户看到的是报错而非即时的输入反馈。
 */
function NumberField({
  label, value, onChange, min, max, step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="field">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(numberOrPrevious(e.target.value, value))}
      />
    </label>
  );
}
````

### 17. `frontend/tests/useAgentRun.test.ts`

- 新增 6 项 React 生命周期回归测试（页签不被 rerender 打回 / 用户手动选择优先 / SSE 收事件不 abort / 只有切会话才清理 / 卸载才中断 / lastRunId 变化不清理）。

````ts
/**
 * `useAgentRun` 生命周期回归测试：本轮修的两个 P0 都出在 React Effect 上。
 *
 * 为什么必须真渲染才能测
 * ----------------------
 * 这两个缺陷**不是**纯函数算错，而是「Effect 在不该跑的时候跑了」：
 *
 * 1. `onRunRestored` 是页面传进来的内联箭头函数，每次 render 都是新引用；
 *    它在「会话镜像 Effect」的依赖数组里 ⇒ effect 每次 render 重跑 ⇒
 *    执行 `stopPolling() / abortStream() / clear() / setRun(null) / setInspectorTab("overview")`。
 *    用户看到的就是：AI 不回应、运行消失、SSE 自己断、Inspector 页签锁死在概览。
 * 2. 页签被 `refreshRun()` / `tool_call` 事件无条件改写，用户手动选的页签停不住。
 *
 * 依赖数组的稳定性只能靠「真的 render 一次看 effect 有没有重跑」来证伪，
 * 所以这里用 jsdom + react-dom 真实挂载（基础设施见 tests/dom.mjs），
 * 全程离线：`fetch` 与 `getRun` 都由本地替身接管，不发任何真实请求。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { installDom, mountPoint } from "./dom.mjs";

installDom();

// DOM 就绪之后才加载 react-dom（它在挂载时才需要 document，顺序颠倒会拿不到容器）
const React = (await import("react")).default;
const { createRoot } = await import("react-dom/client");
const { act } = await import("react-dom/test-utils");
const { useAgentRun } = await import("../src/features/agent/hooks/useAgentRun.ts");
type RunApi = ReturnType<typeof useAgentRun>;

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

// ---------------------------------------------------------------------------
// 替身：SSE 流
// ---------------------------------------------------------------------------

interface StreamCtl {
  push(type: string, payload: Record<string, unknown>): void;
  events: number;
  aborts: number;
}

/** 一个可控的 SSE 流：测试自己决定什么时候往下推事件。 */
function installSseFetch(): StreamCtl {
  const encoder = new TextEncoder();
  const ctl: StreamCtl & { controller?: ReadableStreamDefaultController } = {
    push(type, payload) {
      const frame = `event: ${type}\ndata: ${JSON.stringify({ seq: 1, run_id: "r-1", type, payload, created_at: 0 })}\n\n`;
      ctl.controller?.enqueue(encoder.encode(frame));
    },
    events: 0,
    aborts: 0,
  };

  (globalThis as unknown as { fetch: unknown }).fetch = async (_url: string, init: { signal?: AbortSignal } = {}) => {
    init.signal?.addEventListener("abort", () => {
      ctl.aborts += 1;
    });
    const body = new ReadableStream({
      start(controller) {
        ctl.controller = controller;
      },
    });
    return { ok: true, status: 200, body } as unknown as Response;
  };
  return ctl;
}

// ---------------------------------------------------------------------------
// 挂载装置
// ---------------------------------------------------------------------------

interface HarnessProps {
  sessionId: string | null;
  lastRunId: string | null;
  switchToken: number;
  tools: never[];
}

function mountHarness(initial: HarnessProps) {
  const api: { current: RunApi | null } = { current: null };
  let props = initial;
  // 计数：会话镜像 Effect 若在不该跑的时候跑了，这里一定会被加一。
  const counters = { restored: 0, renders: 0, errors: [] as string[] };

  function Harness(p: HarnessProps) {
    counters.renders += 1;
    const run = useAgentRun({
      sessionId: p.sessionId,
      lastRunId: p.lastRunId,
      switchToken: p.switchToken,
      datasetIds: [],
      tools: p.tools,
      ensureSession: async () => "s-new",
      appendMessage: () => undefined,
      onError: (message) => {
        if (message) counters.errors.push(message);
      },
      onNotice: () => undefined,
      // ★ 刻意写成内联箭头函数：真实页面就是这样传的，每次 render 都是新引用。
      // 修复前这一条足以让「会话镜像 Effect」每次 render 重跑。
      onRunRestored: () => {
        counters.restored += 1;
      },
    });
    api.current = run;
    return null;
  }

  const container = mountPoint();
  const root = createRoot(container);

  const render = async (patch: Partial<HarnessProps> = {}) => {
    props = { ...props, ...patch };
    await act(async () => {
      root.render(React.createElement(Harness, props));
      await flush();
    });
  };

  return {
    api,
    counters,
    render,
    state: () => api.current as RunApi,
    unmount: async () => {
      await act(async () => {
        root.unmount();
        await flush();
      });
      container.remove();
    },
  };
}

/** 监听 `AbortController.abort()` —— 「SSE 被谁掐断」的唯一直接证据。 */
function watchAborts() {
  type AbortProto = { abort: (...args: unknown[]) => void };
  const proto = (globalThis as unknown as { AbortController: { prototype: AbortProto } }).AbortController.prototype;
  const original = proto.abort;
  let count = 0;
  proto.abort = function patched(this: unknown, ...args: unknown[]) {
    count += 1;
    return original.apply(this, args);
  };
  return {
    get count() {
      return count;
    },
    restore() {
      proto.abort = original;
    },
  };
}

const BASE: HarnessProps = { sessionId: "s-1", lastRunId: null, switchToken: 0, tools: [] };

// ---------------------------------------------------------------------------
// 一、Inspector 页签：用户选过之后不许被任何自动逻辑打回
// ---------------------------------------------------------------------------

test("Inspector 页签不被 rerender 打回 overview", async () => {
  const aborts = watchAborts();
  const h = await mountHarness(BASE);
  await h.render(); // 首次挂载

  assert.equal(h.state().inspectorTab, "overview");

  // 用户点「Token」看用量
  await act(async () => {
    h.state().setInspectorTab("token");
    await flush();
  });
  assert.equal(h.state().inspectorTab, "token");

  // 连续 rerender（模拟收到 SSE / usage / progress / stage / busy 变化后的重渲染）
  for (let i = 0; i < 3; i += 1) await h.render();

  assert.equal(h.state().inspectorTab, "token", "普通 rerender 不许改写用户选中的页签");
  // 会话镜像 Effect 没重跑 ⇒ 没有清理、没有恢复回调
  assert.equal(h.counters.restored, 0, "lastRunId 为空时不该触发恢复回调");
  assert.equal(h.state().run, null, "effect 重跑会把 run 清空为 null，这里必须保持");
  assert.equal(aborts.count, 0, "没有切会话就不该有任何 abort");

  aborts.restore();
  await h.unmount();
});

test("tool_call 事件尊重用户已选中的页签（未选过时仍自动定位）", async () => {
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  // 未手动选过 ⇒ 自动定位仍然生效
  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("tool_call", { tool: "dataset.quality", status: "ok" });
    await flush();
  });
  assert.equal(h.state().inspectorTab, "activity", "未手动选择时 tool_call 应自动切到活动");

  // 用户切到「预算」
  await act(async () => {
    h.state().setInspectorTab("budget");
    await flush();
  });
  await act(async () => {
    stream.push("tool_call", { tool: "dataset.profile", status: "ok" });
    await flush();
  });
  assert.equal(h.state().inspectorTab, "budget", "用户选过之后不许被 tool_call 弹回活动");

  await h.unmount();
});

// ---------------------------------------------------------------------------
// 二、SSE 不许被前端自己误杀
// ---------------------------------------------------------------------------

test("SSE 收到事件不触发 abort", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  assert.equal(h.state().busy, true, "发送后应处于忙碌态");

  // 连续推事件：每一次都会触发 setState ⇒ rerender。修复前这就是掐断流的时刻。
  for (const [type, payload] of [
    ["route", { mode: "agent", reason: "要分析" }],
    ["planning", { stage: "plan_ready", steps: 2 }],
    ["usage", { token_usage: { actual: { total_tokens: 10 } } }],
    ["tool_call", { tool: "dataset.quality", status: "ok" }],
  ] as const) {
    await act(async () => {
      stream.push(type, payload);
      await flush();
    });
  }

  assert.ok(h.state().events.length >= 4, `事件应被接收，实际 ${h.state().events.length} 条`);
  assert.equal(stream.aborts, 0, "SSE 收到事件不得触发 abort");
  assert.equal(aborts.count, 0, "整个过程中的 rerender 都不许 abort");
  assert.equal(h.state().busy, true, "流还在进行，busy 不该被清掉");
  assert.deepEqual(h.counters.errors, [], "不该有发送失败之类的报错");

  aborts.restore();
  await h.unmount();
});

// ---------------------------------------------------------------------------
// 三、只有切会话 / 卸载 / 新请求顶替才允许清理旧流
// ---------------------------------------------------------------------------

test("切会话才允许清理旧流（并中断上一条 SSE）", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("route", { mode: "agent", reason: "要分析" });
    await flush();
  });
  assert.equal(h.state().events.length, 1);

  // 切会话：switchToken +1
  await h.render({ switchToken: 1 });

  assert.equal(stream.aborts, 1, "切会话必须中断上一条 SSE（且只中断一次）");
  assert.equal(aborts.count, 1);
  assert.equal(h.state().events.length, 0, "切会话后事件应被清空");
  assert.equal(h.state().run, null);
  assert.equal(h.state().inspectorTab, "overview", "切会话是真正的重置，页签回到概览");

  aborts.restore();
  await h.unmount();
});

test("卸载时才中断 SSE，卸载前不中断", async () => {
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();
  await act(async () => {
    void h.state().send("hi");
    await flush();
  });

  assert.equal(stream.aborts, 0);
  await h.unmount();
  assert.equal(stream.aborts, 1, "卸载必须中断 SSE，避免后端 tail 线程挂到超时上限");
});

test("lastRunId 变化（未切会话）不清理当前运行", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();
  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("route", { mode: "agent", reason: "要分析" });
    await flush();
  });

  // 会话列表刷新把 lastRunId 换了（switchToken 没变）——修复前这也会触发整轮清理
  await h.render({ lastRunId: "r-1" });

  assert.equal(stream.aborts, 0, "lastRunId 变化不等于切会话，不许中断流");
  assert.equal(h.state().events.length, 1, "当前运行的事件必须保留");

  aborts.restore();
  await h.unmount();
});
````

### 18. `frontend/tests/dom.mjs`

- 新增最小 jsdom 挂载基础设施（唯一新增 devDependency 就是 jsdom），不引入 testing-library / vitest / jest。

````js
/**
 * 最小 DOM 环境：`npm test` 里跑 React 生命周期测试所必需的那一点基础设施。
 *
 * 为什么不引一整套测试框架：需要验证的三件事（Inspector 页签不被 rerender 打回、
 * SSE 不被误杀、切会话才允许清理旧流）**都是 React Effect 的生命周期行为**，
 * 纯函数测试根本碰不到。要真跑 Effect 就得有 DOM + 真实渲染器。
 *
 * 于是这里只补「最小必要」的一层：
 *   jsdom（唯一的外部依赖，devDependency） + react-dom/client 自带的 act。
 * 不引入 vitest / jest / @testing-library —— 它们带来的都是这一层之上的封装。
 */

import { JSDOM } from "jsdom";

/** React 只关心这些全局；逐个搬过去，Node 已有的（如 AbortController）不覆盖。 */
const BRIDGED_KEYS = [
  "HTMLElement",
  "HTMLInputElement",
  "Element",
  "Node",
  "Event",
  "CustomEvent",
  "MouseEvent",
  "KeyboardEvent",
  "DocumentFragment",
  "Text",
  "DOMParser",
  "getComputedStyle",
  "requestAnimationFrame",
  "cancelAnimationFrame",
  "localStorage",
  "sessionStorage",
  "XMLHttpRequest",
];

function define(target, key, value) {
  try {
    Object.defineProperty(target, key, { value, writable: true, configurable: true });
  } catch {
    /* 只读全局（如 Node 的 navigator）跳过即可，React 不依赖它 */
  }
}

export function installDom() {
  if (globalThis.document) return globalThis.document;

  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: "http://localhost/",
    pretendToBeVisual: true,
  });

  define(globalThis, "window", dom.window);
  define(globalThis, "document", dom.window.document);
  define(globalThis, "navigator", dom.window.navigator);
  for (const key of BRIDGED_KEYS) {
    if (globalThis[key] === undefined && dom.window[key] !== undefined) {
      define(globalThis, key, dom.window[key]);
    }
  }
  // React 18 的 act 要求显式声明「现在是测试环境」，否则会一直告警并跳过 effect 刷新。
  define(globalThis, "IS_REACT_ACT_ENVIRONMENT", true);
  return dom.window.document;
}

/** 造一个挂载点，用完由调用方 unmount。 */
export function mountPoint() {
  const el = globalThis.document.createElement("div");
  globalThis.document.body.appendChild(el);
  return el;
}
````

### 19. `frontend/tests/bootstrap.mjs`

- 预置 `globalThis.__VITE_ENV__`，供 resolve-hook 替换 `import.meta.env`。

````js
/**
 * `npm test` 的引导文件：注册解决「省略扩展名」的解析钩子。
 *
 * Node 没有「--loader 直接给 hook」的开关，必须先 import 一段 JS，
 * 由它调用 `module.register()` 把钩子挂到专门的 hooks 线程上 ——
 * 这也是为什么这里只有一个 register 调用，钩子本体在 resolve-hook.mjs。
 *
 * `globalThis.__VITE_ENV__` 是给 resolve-hook 的 `load` 钩子用的替身：
 * 源码里的 `import.meta.env` 会被替换成它（详见 resolve-hook.mjs 的说明）。
 * 必须在 register **之前** 定义，因为被测试的模块可能在钩子安装的同时就被加载。
 */

import { register } from "node:module";

globalThis.__VITE_ENV__ = {
  DEV: false,
  PROD: true,
  MODE: "test",
  VITE_API_BASE_URL: "",
};

register("./resolve-hook.mjs", import.meta.url);
````

### 20. `frontend/tests/resolve-hook.mjs`

- 新增 `load()` 钩子：把 `import.meta.env` 替换为 `globalThis.__VITE_ENV__`；`patchSource()` 同时兼容 string 与 Uint8Array 两种 source 形态。

````js
/**
 * Node 测试用的模块解析钩子。
 *
 * 背景：项目源码用 Vite 的省略式相对导入（`from "./nodeSpecs"`），
 * 而 Node 的 ESM 解析器要求完整扩展名（`./nodeSpecs.ts`），
 * 于是 `node --experimental-strip-types --test` 一碰到有内部依赖的模块
 * 就抛 ERR_MODULE_NOT_FOUND。
 *
 * 与其把源码里的 import 全部补扩展名（要动 tsconfig 的
 * allowImportingTsExtensions，还会影响 Vite 之外的工具链），
 * 不如在测试入口挂一个解析钩子 —— 只作用于 `npm test`，不碰主构建。
 */

import { existsSync } from "node:fs";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** 依次尝试的后缀；目录场景补 index.ts。 */
const EXTENSIONS = [".ts", ".tsx", ".mts", "/index.ts"];

/** 判断 specifier 是否「看起来已经有扩展名」（取最后一段路径看有没有点）。 */
function hasExtension(specifier) {
  const last = specifier.split("/").pop() ?? "";
  return last.includes(".");
}

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith(".") && !hasExtension(specifier) && context.parentURL) {
    const parentDir = dirname(fileURLToPath(context.parentURL));
    for (const extension of EXTENSIONS) {
      const candidate = resolvePath(parentDir, `${specifier}${extension}`);
      if (existsSync(candidate)) {
        return nextResolve(pathToFileURL(candidate).href, context);
      }
    }
  }
  // 其余情况（含三方包）原样交给 Node 默认解析。
  return nextResolve(specifier, context);
}

/**
 * `import.meta.env` 是 **Vite 注入**的编译期常量，Node 里不存在
 * （`import.meta` 只有 url / dirname / filename / resolve）。
 * 因此任何 import 到 `api/client.ts` 的模块在测试里都会
 * `TypeError: Cannot read properties of undefined (reading 'DEV')`。
 *
 * 与其给源码加一堆 `import.meta.env?.` 的防御（那是为测试污染产品代码），
 * 不如在钩子这一层把它替换成一个全局对象 —— 只作用于 `npm test`。
 * 常量的取值由 bootstrap.mjs 在 register 之前写好。
 */
const NEEDLE = "import.meta.env";
const REPLACEMENT = "globalThis.__VITE_ENV__";
const decoder = new TextDecoder();
const encoder = new TextEncoder();

/** Node 22 对 `.ts` 返回的 `source` 是 **Uint8Array**（format=`module-typescript`），
 *  普通 `.js` 才是 string。两种都要处理，否则替换根本不会发生。 */
function patchSource(source) {
  if (typeof source === "string") {
    return source.includes(NEEDLE) ? source.replaceAll(NEEDLE, REPLACEMENT) : null;
  }
  if (source instanceof Uint8Array) {
    const text = decoder.decode(source);
    return text.includes(NEEDLE) ? encoder.encode(text.replaceAll(NEEDLE, REPLACEMENT)) : null;
  }
  return null;
}

export async function load(url, context, nextLoad) {
  const loaded = await nextLoad(url, context);
  const patched = loaded.source ? patchSource(loaded.source) : null;
  return patched === null ? loaded : { ...loaded, source: patched };
}
````

### 21. `frontend/tests/agentEvents.test.ts`

- 新增「远程规划降级到内置规则时必须显式告知用户」用例。

````ts
/** SSE 事件 → 界面增量（lib/agentEvents）单元测试。
 *
 * 为什么值得测：这段映射里埋过三类真实 bug ——
 *   1. 重试时进度不回退，用户看不到「卡住后重来」；
 *   2. 授权事件的字段没做归一，缺字段时弹窗拿不到 tool / step_index；
 *   3. 校验失败事件没有给出「第几步 + 原因」，只在界面上无声跳过。
 * 放在 lib/ 下就是为了能用 Node 原生 runner 直接跑（无 DOM 依赖）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { eventEffects } from "../src/lib/agentEvents.ts";
import type { AgentEvent } from "../src/types/agent.ts";

function ev(type: AgentEvent["type"], payload: Record<string, unknown>): AgentEvent {
  return { seq: 1, run_id: "run-1", type, payload, created_at: 0 };
}

/** progress 可能是定值或 updater，统一取值。 */
function progressOf(fx: ReturnType<typeof eventEffects>, prev: number): number {
  if (fx.progress === undefined) return prev;
  return typeof fx.progress === "function" ? fx.progress(prev) : fx.progress;
}

const label = (tool: string) => (tool === "ml.train" ? "训练模型" : tool);

test("route：对话模式与工具模式给出不同起步进度", () => {
  assert.equal(eventEffects(ev("route", { mode: "chat", reason: "寒暄" })).stage, "对话模式 · 寒暄");
  assert.equal(progressOf(eventEffects(ev("route", { mode: "chat", reason: "寒暄" })), 5), 40);
  assert.equal(progressOf(eventEffects(ev("route", { mode: "agent", reason: "要分析" })), 5), 8);
});

test("planning：plan_ready 推进到 25，未知阶段回落到 8", () => {
  assert.equal(progressOf(eventEffects(ev("planning", { stage: "plan_ready" })), 5), 25);
  assert.equal(eventEffects(ev("planning", { stage: "unknown_stage" })).stage, "unknown_stage");
  assert.equal(progressOf(eventEffects(ev("planning", { stage: "unknown_stage" })), 5), 8);
});

test("tool_call：切到活动页签并推进进度，且不超过 90", () => {
  const fx = eventEffects(ev("tool_call", { tool: "ml.train" }), label);
  assert.equal(fx.tab, "activity");
  assert.equal(fx.stage, "执行工具：训练模型");
  assert.equal(progressOf(fx, 10), 18);
  assert.equal(progressOf(eventEffects(ev("tool_call", { tool: "ml.train" }), label), 88), 90);
});

test("replanning：进度显式回退且有下限 5（历史 bug：曾停在原数字不动）", () => {
  const fx = eventEffects(ev("replanning", { retry: true, notes: "工具超时" }), label);
  assert.equal(fx.stage, "重试中 · 工具超时");
  assert.equal(fx.notice, "工具超时");
  assert.equal(progressOf(fx, 60), 45);
  assert.equal(progressOf(eventEffects(ev("replanning", {})), 10), 5);
});

test("validation：失败给出「第几步 + 原因」，通过时不打扰用户", () => {
  const failed = eventEffects(ev("validation", { valid: false, step_index: 2, errors: ["缺 target", "缺 features"] }));
  assert.equal(failed.notice, "第 3 步校验失败：缺 target；缺 features");
  // step_index 缺失时按第 1 步算，不能出现 NaN
  assert.match(eventEffects(ev("validation", { valid: false })).notice ?? "", /第 1 步/);
  assert.equal(eventEffects(ev("validation", { valid: true })).notice, undefined);
});

test("permission：字段归一，缺字段时给出可用默认值", () => {
  const fx = eventEffects(ev("permission", { tool: "data.write", reason: "会写库", step_index: 3, arguments: { a: 1 } }));
  assert.deepEqual(fx.permission, { tool: "data.write", arguments: { a: 1 }, reason: "会写库", step_index: 3 });
  const minimal = eventEffects(ev("permission", { tool: "data.write" }));
  assert.deepEqual(minimal.permission, { tool: "data.write", arguments: {}, reason: "", step_index: 0 });
  // 没有 tool 名时不产生授权请求：否则会弹出一个无法确认的空弹窗
  assert.equal(eventEffects(ev("permission", {})).permission, undefined);
});

test("completed / failed：把最终答案与失败原因追加为助手消息", () => {
  const done = eventEffects(ev("completed", { final_answer: "分析完成", answer_source: { source: "remote", label: "远程", detail: "d", by_llm: true, model: "m" } }));
  assert.equal(done.append?.role, "assistant");
  assert.equal(done.append?.content, "分析完成");
  assert.equal(done.append?.source?.by_llm, true);
  assert.equal(progressOf(done, 12), 100);
  const failed = eventEffects(ev("failed", { error: "工具不存在" }));
  assert.equal(failed.append?.content, "执行失败：工具不存在");
});

test("usage 事件不产生界面增量：Token 账本由 useAgentUsage 从事件列表派生", () => {
  assert.deepEqual(eventEffects(ev("usage", { token_usage: { llm_calls: 1 } })), {});
});

test("planning：远程规划降级到内置规则时必须显式告知用户", () => {
  // 「数据还是真的，但计划不是大模型定的」——不说出来用户会误以为是模型分析的结果
  const degraded = eventEffects(ev("planning", { stage: "plan_ready", planner_fallback: true }));
  assert.match(String(degraded.notice), /内置规则/);
  // 没有降级标记时不该弹提示，否则每次规划都打扰一次
  assert.equal(eventEffects(ev("planning", { stage: "plan_ready" })).notice, undefined);
  assert.equal(eventEffects(ev("planning", { stage: "plan_ready", planner_fallback: false })).notice, undefined);
});
````

### 22. `frontend/package.json`

- devDependencies 新增 `jsdom@25`（仅为让 React 生命周期测试能在 node:test 里真实挂载组件）。

````json
{
  "name": "xiaoluo-lab-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "repository": {
    "type": "git",
    "url": "https://github.com/xiaoluojiu/xiaoluo-lab"
  },
  "scripts": {
    "dev": "vite",
    "build": "npm run typecheck && vite build",
    "preview": "vite preview",
    "typecheck": "tsc --noEmit",
    "test": "node --experimental-strip-types --import ./tests/bootstrap.mjs --test tests/*.test.ts"
  },
  "dependencies": {
    "axios": "^1.7.7",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2",
    "recharts": "^2.12.7",
    "zustand": "^4.5.5"
  },
  "devDependencies": {
    "@types/react": "^18.3.8",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "jsdom": "^25.0.1",
    "typescript": "^5.6.2",
    "vite": "^5.4.7"
  }
}
````

---

## ④ 测试结果

### 后端 pytest

```
$ cd backend && ./.venv/Scripts/python.exe -m pytest -p no:randomly -q --no-header --tb=line -rf
收集用例：1187
EXIT = 0    FAILED = 0    ERROR = 0
```

本次新增 / 受影响的用例：

| 文件 | 用例数 | 结果 |
|---|---|---|
| `tests/test_agent_llm_fallback.py`（新） | 22 | 全绿 |
| `tests/test_answer_source.py` | 全量 | 全绿 |
| `tests/test_local_chat.py` | 全量 | 全绿 |
| `tests/test_config.py` | 全量 | 全绿 |
| `tests/test_agent.py` | 40 | 全绿 |
| `tests/test_agent_e2e.py` | 19 | 全绿 |

过程中真实发生过、并已修掉的问题（不甩锅给环境）：

1. **`_capability_reply() missing 1 required keyword-only argument: 'degraded'`** —— 给 `local_chat.py` 的 6 个私有应答函数与 `_note()` 的 `degraded` 参数补默认值 `False`，保持对既有调用兼容。
2. **`test_chat_with_llm_error_is_marked_as_degraded` 失败** —— 该用例原本断言的正是「报错文案 + `LLM_ERROR_FALLBACK`」这一错误行为，按新语义更新断言。
3. **`useAgentRun.ts(167,15): TS2304 Cannot find name 'seq'`** —— `startPolling()` 里补 `const seq = seqRef.current;`。
4. **前端测试崩溃 `Cannot read properties of undefined (reading 'DEV')`** —— Node 的 `load()` 钩子对 `.ts` 返回的 `source` 是 `Uint8Array`（不是 string），字符串替换没生效；改为同时处理两种形态，并在 bootstrap 里预置 `globalThis.__VITE_ENV__`。
5. **`test_answer_source_never_lies` 的「远程失败+本地答不上」格断言不成立** —— 该格与「本地能答」格复用了同一个问法「你好」（本地规则能覆盖），改为参数化新增 `utterance`，「答不上」两格改用「讲个笑话」。
6. **全量 pytest 出现 5 个连带失败** —— 根因是 `test_fallback_flag_default_is_on` 里用了 `importlib.reload(app.core.config)`：reload 换掉了模块里的 `settings` 单例，后续用例的 monkeypatch 全部打空（表现为「改 A 处、B 处挂」）。改为 `Settings(_env_file=None).AGENT_ALLOW_MODEL_FALLBACK is True`，不 reload。
7. **回归验证**：把 Effect 依赖改回 `[switchToken, lastRunId, onRunRestored]` 后，新前端测试立刻触发 React `Maximum update depth exceeded` 无限更新循环 —— 这是 P0-1 的硬证据，随后恢复为 `[switchToken]`。

### 前端

```
$ cd frontend && npm run typecheck   → EXIT=0（0 error）
$ cd frontend && npm run test        → # tests 61  # pass 61  # fail 0
$ cd frontend && npm run build       → ✓ built in 5.73s   EXIT=0
```

前端测试由 54 → 61（新增 `useAgentRun` 6 项 + `agentEvents` 1 项）。

---

## ⑤ 修改前后行为对比

| 场景 | 修改前 | 修改后 |
|---|---|---|
| **A. 远程正常** | `remote_llm_chat`，但一次 rerender 就可能把运行和 SSE 干掉 | `remote_llm_chat`，运行稳定、SSE 持续、页签保持 |
| **B. 欠费 / quota 429 + 普通聊天** | 无脑重试到超时，最后返回「暂时无法完成对话请求」且标 `LLM_ERROR_FALLBACK`（撒谎） | 一次请求即判定不可重试 → `local_reply(degraded=True)` → `llm_error_fallback`，`{"source":"llm_error_fallback","by_llm":false}` |
| **C. 欠费 + 本地答不上** | 同上，一句报错 | `no_llm_notice(reason=真实原因)` → `no_answer` |
| **D. 欠费 + 数据分析任务** | `LLMException` 冒泡 ⇒ 整条 run 失败，永远到不了 Rule Planner | 两轮远程尝试失败后 Rule Planner 接管 → 真实执行 `dataset.quality` 并返回真实结果，`plan.planner_fallback=True`，前端显式告知「已改用平台内置规则规划」 |
| **E. Inspector 页签** | 切到「活动/工具链」后，任意 rerender 被打回「概览」 | 用户手动选择后 pin 住；自动切换只在未 pin 时生效 |
| **F. SSE 生命周期** | 父组件 rerender 就 `controller.abort()` | 只有「切换会话 / 卸载页面 / 用户显式停止 / 新请求顶替旧请求」四种情况才 abort |
| **G. 切会话后的状态** | 旧会话轮询结果后到，覆盖新会话状态（状态回退） | `startPolling` 带代次校验，代次不符立即停轮询 |
| **H. `AGENT_ALLOW_MODEL_FALLBACK`** | 默认 False 且接口写死 False，形同虚设 | 默认 True，前后端可写，planner/runtime 真正作为总闸读取；`False` 时远程失败直接失败不假装降级 |
| **I. 程序 bug（TypeError 等）** | 与远程故障一样被兜底吞掉，问题被掩盖 | `is_fallbackable_error()` 判定为不可降级，原样抛出 |
