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
    AGENT_ALLOW_MODEL_FALLBACK: bool = False
    AGENT_ENABLE_TOOL_RETRIEVAL: bool = True
    AGENT_TOOL_RETRIEVAL_TOP_K: int = 16
    AGENT_TOOL_RETRIEVAL_MIN_SCORE: float = 0.08
    AGENT_ENABLE_RESULT_COMPRESSION: bool = True
    AGENT_ENABLE_PLAN_CACHE: bool = True
    AGENT_PLAN_CACHE_MAX_ITEMS: int = 64

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

    @property
    def data_root_path(self) -> Path:
        return self._resolve_path(self.DATA_ROOT)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
