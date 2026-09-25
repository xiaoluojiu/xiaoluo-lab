"""本地 Router 的**神经侧**：Qwen3-0.6B + LoRA 的加载与推理。

与 :mod:`app.local_router.model`（TF-IDF + LinearSVC 词法模型）的关系
-------------------------------------------------------------------
两者是**同一层（L1 工具/意图选择）的两种实现**，不是两个层：

    L0 升级规则 → L1（Qwen 神经 ‖ 词法）→ 反问规则

Qwen 命中就用 Qwen，Qwen 不可用（依赖没装 / 权重缺失 / 加载失败 / 推理超时 /
输出解析失败）就退回词法模型 —— 见 :mod:`app.local_router.router` 的合成顺序。

为什么依赖是**可选的**
----------------------
torch + transformers + peft 装下来约 2GB，而本项目的默认部署形态是「零本地
模型也能完整跑」。所以这里做成 `pip install -e ".[router]"` 的可选 extra：
**没装不是错误，只是这一层不存在**。这是与 `model.py` 的 `get_model()` 返回
None 一致的失败哲学 —— 宁可降级，绝不猜，也绝不让请求失败。

为什么推理必须有硬超时
----------------------
`app/api/v1/agent.py` 的 SSE worker 要等 `AgentRuntime.run()` 返回才会把终止
标记放进队列。run() 一旦阻塞在模型推理上，`event: done` 永远不会发出，前端
就一直转圈到 120 秒超时文案。这里是整条链路上**唯一可能无限阻塞**的新增点，
因此超时不是优化项，而是正确性依赖。

线程模型
--------
加载与推理都在调用方线程里做（模型只加载一次，用锁保护）；超时通过在
**守护线程**里跑 generate + `join(timeout)` 实现 —— Python 无法强制杀死线程，
超时只是「放弃等待」，那条线程会自己跑完然后退出。代价可控：0.6B 模型、
160 token 上限，最坏情况是几十毫秒的无效 GPU 占用。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "QWEN_SOURCE",
    "QwenPrediction",
    "QwenRouterModel",
    "available",
    "get_model",
    "reset_model_cache",
    "unavailable_reason",
]

#: 决策来源标记。与 `DecisionSource.LOCAL_ROUTER`（词法）区分开，
#: 这样 DecisionTrace 能回答「这条是 Qwen 判的还是词法判的」。
QWEN_SOURCE = "local_qwen"


# ---------------------------------------------------------------------------
# 可选依赖：导入失败只记一次，不刷日志
# ---------------------------------------------------------------------------

_IMPORT_LOCK = threading.Lock()
_BACKEND: tuple[Any, ...] | None = None
_IMPORT_ERROR: str | None = None


def _import_backend() -> tuple[Any, ...] | None:
    """延迟导入 torch / transformers / peft。返回 None 表示依赖未安装。"""
    global _BACKEND, _IMPORT_ERROR
    with _IMPORT_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        if _IMPORT_ERROR is not None:
            return None
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
            from peft import PeftModel  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 — 缺依赖是预期状态，不是异常
            _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
            logger.info(
                "本地 Router 未启用 Qwen 神经路由（缺少可选依赖）：%s。"
                "执行 pip install -e '.[router]' 可启用；当前自动使用词法模型。",
                _IMPORT_ERROR,
            )
            return None
        _BACKEND = (torch, AutoModelForCausalLM, AutoTokenizer, PeftModel)
        return _BACKEND


def available() -> bool:
    """神经路由在当前进程是否具备运行条件（依赖已装）。"""
    return _import_backend() is not None


def unavailable_reason() -> str | None:
    """不可用的原因；可用则返回 None。用于排障与设置页展示。"""
    if _import_backend() is not None:
        return None
    return _IMPORT_ERROR or "依赖未安装"


# ---------------------------------------------------------------------------
# 超时包装
# ---------------------------------------------------------------------------


def _run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> tuple[Any, str | None]:
    """在守护线程里跑 `fn`，超时则放弃等待。返回 `(结果, 错误原因)`。"""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — 必须连 KeyboardInterrupt 一起兜住
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="qwen-router-infer")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return None, f"推理超时（>{timeout_s:.2f}s）"
    if "error" in box:
        return None, f"{type(box['error']).__name__}: {box['error']}"
    return box.get("value"), None


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


@dataclass
class QwenPrediction:
    """一次推理的结果。

    `payload` 是**已解析**的 JSON dict；解析失败时整个 `QwenPrediction` 不存在
    （返回 None），由上层走降级 —— 半解析的中间态没有意义，留着只会误导。
    """

    payload: dict[str, Any]
    raw: str
    confidence: float
    elapsed_ms: float


@dataclass
class QwenRouterModel:
    """加载好的 Qwen + LoRA。只承载「给 messages，回一段文本」这一件事。"""

    tokenizer: Any
    model: Any
    device: str = "cpu"
    adapter_path: str = ""
    base_model: str = ""

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        """按训练时的 chat 模板生成。贪心解码 —— Router 不该有随机性。"""
        torch = _import_backend()[0]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 只解出新生成的部分，避免把 prompt 又解一遍
        prompt_len = int(inputs["input_ids"].shape[-1])
        return str(
            self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        )

    def predict(
        self, messages: list[dict[str, str]], *, max_new_tokens: int, timeout_s: float
    ) -> tuple[str, str | None]:
        """带超时的生成。返回 `(文本, 错误原因)`。"""
        return _run_with_timeout(
            lambda: self.generate(messages, max_new_tokens=max_new_tokens), timeout_s
        )


def _pick_device(torch: Any, configured: str) -> str:
    """auto ⇒ CUDA 可用则用 CUDA；显式 cuda 但不可用时退回 CPU（而不是报错）。"""
    if configured == "cpu":
        return "cpu"
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _load() -> QwenRouterModel | None:
    """真正加载模型。任何一步失败都返回 None（调用方只记录一次原因）。"""
    backend = _import_backend()
    if backend is None:
        return None
    torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = backend

    from app.core.config import settings

    adapter = settings.local_router_adapter_path()
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA 适配器不存在：{adapter}")

    base = settings.local_router_base_model()
    device = _pick_device(torch, settings.local_router_device())

    # tokenizer 与 adapter 同目录：训练时是把 tokenizer 一起存进 adapter 目录的，
    # 从 base 目录读也能用，但会多一次 1.5GB 目录的扫描。
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)

    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    except TypeError:  # transformers < 4.56 只认 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    # 显式传 base 之外的 adapter 目录；`adapter_config.json` 里的
    # base_model_name_or_path 指向训练机绝对路径，这里必须覆盖它，
    # 否则换机器后 peft 会去读一个不存在的路径。
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    if device == "cuda":
        model.to("cuda")

    return QwenRouterModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        adapter_path=str(adapter),
        base_model=str(base),
    )


_MODEL_LOCK = threading.Lock()
_MODEL: QwenRouterModel | None = None
#: 记住失败原因：模型缺失时每条请求都会问一次，不记住会把日志淹掉。
_MODEL_FAILURE: str | None = None
_MODEL_FAILURE_AT: float = 0.0
#: 失败后隔多久才允许再试一次。
#: 「永久记住失败」看起来省事，但最常见的失败其实是**权重还没放好** ——
#: 用户先起了服务、后拷模型，那次失败不该让他必须重启进程才生效。
#: 60 秒的节流既能挡住每条请求重试（加载要几秒，挡不住会拖垮吞吐），
#: 又能在用户放好文件后自动恢复。
_RETRY_AFTER_S = 60.0


def get_model(*, refresh: bool = False) -> QwenRouterModel | None:
    """取进程级共享模型（懒加载 + 锁 + 失败记忆 + 节流重试）。取不到返回 None。"""
    global _MODEL, _MODEL_FAILURE, _MODEL_FAILURE_AT
    with _MODEL_LOCK:
        if _MODEL is not None and not refresh:
            return _MODEL
        if (
            _MODEL_FAILURE is not None
            and not refresh
            and (time.monotonic() - _MODEL_FAILURE_AT) < _RETRY_AFTER_S
        ):
            return None
        started = time.perf_counter()
        try:
            model = _load()
        except Exception as exc:  # noqa: BLE001 — 加载失败不该拖垮请求链路
            _MODEL_FAILURE = f"{type(exc).__name__}: {exc}"
            _MODEL_FAILURE_AT = time.monotonic()
            logger.warning(
                "本地 Router 的 Qwen 模型加载失败（将自动退回词法模型，%d 秒后重试）：%s",
                int(_RETRY_AFTER_S),
                _MODEL_FAILURE,
            )
            return None
        _MODEL, _MODEL_FAILURE = model, None
        logger.info(
            "本地 Router Qwen 模型已加载：adapter=%s base=%s device=%s，耗时 %.2fs",
            model.adapter_path,
            model.base_model,
            model.device,
            time.perf_counter() - started,
        )
        return _MODEL


def is_loaded() -> bool:
    """模型是否**已经**加载过。刻意不触发加载 —— 供状态查询接口使用。"""
    return _MODEL is not None


def device_of() -> str | None:
    """已加载模型的设备；未加载返回 None。"""
    return _MODEL.device if _MODEL is not None else None


def reset_model_cache() -> None:
    """仅供测试使用：清掉缓存与失败记忆，让下一次 `get_model()` 重新加载。"""
    global _MODEL, _MODEL_FAILURE
    with _MODEL_LOCK:
        _MODEL, _MODEL_FAILURE = None, None


def adapter_dir() -> Path:
    """当前配置解析出的适配器目录（供日志与设置页展示）。"""
    from app.core.config import settings

    return settings.local_router_adapter_path()
