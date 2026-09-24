"""LocalReasoningProvider —— 本地生成式推理模型接口（为未来接入 Qwen 等预留）。

定位（Phase 3）
--------------
为未来接入 Qwen 等**本地生成式推理模型**保留路径。当前只提供一个最小可用实现，
不建设复杂的模型服务平台。

设计约束
--------
1. **复用现有抽象**：直接继承 ``LLMProvider``，与 ``OpenAICompatibleProvider`` 平级，
   不另起一套「推理模型」抽象。本地推理模型对外仍是「一个 LLMProvider」。
2. **接口即真实**：本地模型要么真正被加载、真正被调用，要么**诚实地不可用**。
   不把「接了个接口」当成「模型真正参与运行」——本地模型不可用时，``chat`` 抛
   ``LLMException(..., fallbackable=True)``，交由上层决策是否升级远程 / 降级规则。
3. **不静默吞错**：加载失败 / 未配置 / 推理异常一律显式抛出，不进 reasons 里
   含糊其辞，不返回一个伪造的置信度。

为什么本地推理模型单独保留一个 Provider，而不是塞进 ``OpenAICompatibleProvider``
------------------------------------------------------------------------------
``OpenAICompatibleProvider`` 依赖一个 **HTTP 网关**（base_url + api_key + 熔断重试），
而本地推理模型（Qwen 经 llama.cpp / vLLM / transformers 加载）是**进程内或本地 socket**
的推理后端：
- 它没有 api_key、没有 HTTP 限流 / 欠费语义；
- 它的「不可用」是「模型文件缺失 / 显存不足 / 权重未下载」，不是「远程 502」；
- 它需要显式的「可用性探测」与「延迟加载」，而不是每次调用都发 HTTP。

因此单独保留 ``LocalReasoningProvider`` 这个最小接口，未来接入真实后端时在此实现
真正的推理调用；当前实现是**诚实不可用**占位（``is_available()`` 恒为 False），
并附带一个可用于测试的 ``FakeLocalReasoningProvider``（测试替身，非生产路径）。
"""

from __future__ import annotations

from typing import Any

from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, LLMResponse
from app.agent.llm.capabilities import LLMCapabilities

# 本地推理模型的默认能力描述：能做对话与推理，但**未声明**结构化输出 / 工具调用 /
# JSON 模式等能力（None = 未知，Agent 不应假设其可用）。
_LOCAL_REASONING_CAPABILITIES = LLMCapabilities(
    chat=True,
    reasoning=True,
    streaming=None,
    tool_calling=None,
    structured_output=None,
    json_mode=None,
    vision=None,
)


class LocalReasoningProvider(LLMProvider):
    """本地生成式推理模型的最小接口（Qwen 等未来接入点）。

    当前为**诚实不可用**占位：``is_available()`` 恒为 ``False``，``chat`` 抛可降级的
    ``LLMException``。接入真实后端时，在子类中实现 ``is_available()`` 与 ``chat``。
    """

    name = "local_reasoning"

    def __init__(
        self,
        *,
        model_name: str = "qwen",
        model_path: str | None = None,
        capabilities: LLMCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.capabilities = capabilities or _LOCAL_REASONING_CAPABILITIES
        self._load_error: str | None = None

    def is_available(self) -> bool:
        """本地推理后端当前是否可用。

        基类实现恒为 ``False``（尚未接入真实后端）。子类接入后应返回真实的
        可用性判断（模型文件存在 / 后端进程可达 / 显存加载成功）。
        """
        return False

    def _require_available(self) -> None:
        if not self.is_available():
            raise LLMException(
                f"本地推理模型不可用（model={self.model_name}）：尚未接入本地生成式后端。"
                "本地模型不可用时由上层决策升级远程或降级规则，而非静默吞掉。",
                details={"model": self.model_name, "available": False},
                fallbackable=True,
            )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._require_available()
        # 子类接入真实后端后在此实现真正的推理调用；基类实现不会走到这里。
        raise LLMException(
            f"本地推理模型（{self.model_name}）尚未实现推理调用",
            fallbackable=True,
        )


class FakeLocalReasoningProvider(LocalReasoningProvider):
    """测试替身：模拟一个「可用」的本地推理模型，回放预设响应。

    仅用于单元测试 / 本地开发，**不用于生产**。它把 ``is_available()`` 置为 True，
    并回放 ``responses`` 队列里的预设文本，让上层可以在「本地推理可用」的前提下
    验证决策链路，而不依赖真实模型权重。
    """

    name = "fake_local_reasoning"

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        model_name: str = "qwen-fake",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def is_available(self) -> bool:
        return True

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._require_available()
        self.begin_call()
        self.calls.append([m.to_dict() for m in messages])
        if self._responses:
            content = self._responses.pop(0)
        else:
            last_user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            content = f"[local-reasoning] {last_user}"
        response = LLMResponse(content=content, model=self.model_name)
        self.record_usage(response.usage)
        return response


__all__ = [
    "LocalReasoningProvider",
    "FakeLocalReasoningProvider",
]
