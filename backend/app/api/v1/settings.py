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

    # Qwen 神经路由的可用性探测。**刻意不在这里触发加载**（加载要几秒，
    # 一次状态查询不该付出这个代价），只回答三件事：依赖装没装、权重在不在、
    # 现在是否已加载。真正的加载留给第一次真实请求。
    try:
        from app.local_router import qwen as qwen_module

        summary |= {
            "qwen_deps_installed": qwen_module.available(),
            "qwen_deps_error": qwen_module.unavailable_reason(),
            "qwen_loaded": qwen_module.is_loaded(),
            "qwen_device": qwen_module.device_of(),
        }
    except Exception as exc:  # noqa: BLE001
        summary["qwen_probe_error"] = f"{type(exc).__name__}: {exc}"
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
