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
