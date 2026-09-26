"""项目配置。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]

#: 仓库根（backend/ 的上一级）。本地模型权重默认落在 `{PROJECT_ROOT}/models/`：
#: 与 `backend/models/`（运行时产物）分开，也避开 `backend/app/models/`（SQLAlchemy 源码）。
PROJECT_ROOT = BACKEND_ROOT.parent

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

    # 本地生成式推理模型（Phase 3 预留；实现见 app/agent/llm/local.py）。
    # 当前仅保留接口与最小实现，未接入真实后端（Qwen 等），默认关闭。
    # enabled=False 时不构造 LocalReasoningProvider，Agent 不依赖本地生成模型。
    LOCAL_REASONING_MODEL_ENABLED: bool = False
    # 未来接入时使用的模型名（如 qwen / qwen2.5-7b-instruct）。
    LOCAL_REASONING_MODEL_NAME: str = "qwen"
    # 未来接入时使用的模型权重路径（留空表示尚未指定）。
    LOCAL_REASONING_MODEL_PATH: str = ""

    # 本地 Router（本地小模型路由；实现见 app/local_router/）
    # 三档模式：
    #   off    = 完全关闭：不加载模型、不写 trace（默认值，零风险）
    #   shadow = **只记录不改变行为**：把「本地 Router 会怎么判」与「线上实际怎么走」
    #            并排写进 trace，用于积累真人测试数据、验证离线指标是否可复现
    #   active = 本地 Router 正式作为第一层意图路由器参与决策
    #   guard  = 历史预留档，从未启用。读到它按 active 处理并打一次告警，
    #            避免已有 .env 悄悄退化成 off。
    LOCAL_ROUTER_MODE: str = "off"
    # 门控阈值。默认 0.0 表示与离线评测口径一致（不门控）。
    LOCAL_ROUTER_CONFIDENCE_THRESHOLD: float = 0.0
    # 产物目录（词法 L1 的 pkl）。留空 ⇒ `{MODEL_ROOT}/local_router/`。
    LOCAL_ROUTER_MODEL_DIR: str = ""
    # trace 单文件上限（字节），超出后轮转为 `shadow.jsonl.1`（只保留一代，避免无限增长）。
    LOCAL_ROUTER_TRACE_MAX_BYTES: int = 5 * 1024 * 1024

    # ---- Qwen3-0.6B + LoRA 神经路由（可选依赖；未安装 / 加载失败均自动降级）----
    # LoRA 适配器目录。**同时支持相对路径与绝对路径**：相对路径依次按
    # `{PROJECT_ROOT}`、`{BACKEND_ROOT}` 解析，绝对路径原样使用。
    # 留空 ⇒ 自动探测 `{PROJECT_ROOT}/models/qwen3-lora-xiaoluo-router`，
    #         再探测 `{MODEL_ROOT}/qwen3-lora-xiaoluo-router`。
    # 开源分发时请把适配器放进上述任一位置——默认值里**不含任何本机绝对路径**。
    LOCAL_ROUTER_MODEL_PATH: str = ""
    # 基座模型：本地目录（相对/绝对均可），或 HuggingFace repo id（如 `Qwen/Qwen3-0.6B`）。
    # 当取值不像本地路径时按 repo id 走 HF 下载。留空 ⇒ 探测
    # `{PROJECT_ROOT}/models/Qwen3-0.6B`，仍不存在才回落到 repo id。
    LOCAL_ROUTER_BASE_MODEL: str = ""
    # 推理设备：auto（CUDA 可用则 CUDA，否则 CPU）/ cuda / cpu。
    LOCAL_ROUTER_DEVICE: str = "auto"
    # ★ 单次推理硬超时（毫秒）。这是「SSE 不挂死」的命门：
    # `AgentRuntime.run()` 若阻塞，`app/api/v1/agent.py` 的 SSE worker 就永远不会
    # 把终止标记放进队列 ⇒ 流不结束、前端转圈到底。超时一律按「模型不可用」降级。
    LOCAL_ROUTER_TIMEOUT_MS: int = 800
    # 生成上限。Router 只输出一段短 JSON，160 token 足够；多生成纯属拖慢自己，
    # 还会放大「模型开始说废话」导致 JSON 解析失败的概率。
    LOCAL_ROUTER_MAX_NEW_TOKENS: int = 160
    # 注入 system prompt 的数据集清单条数上限。训练语料里模型见到的是完整清单，
    # 但线上数据集可能有成百上千个，全量注入会把 prefill 拖到几百毫秒并稀释注意力。
    LOCAL_ROUTER_MAX_CONTEXT_DATASETS: int = 30

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
    # 已废弃（统一 Agent Loop 无 plan cache）：保留字段仅为兼容前端设置页与场景脚本，
    # 不再被任何运行期代码消费。
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

    # ---- 本地 Router：Qwen 神经路由的路径与档位解析 -------------------------
    # 全部做成「探测 + 回落」而不是「直接拼路径」，是为了让开源用户什么都不配也能
    # 找到 `{项目根}/models/` 下的权重，同时又允许绝对路径指向任意位置。

    #: 适配器 / 基座在 `models/` 下的默认目录名。
    _DEFAULT_ROUTER_ADAPTER_DIRNAME = "qwen3-lora-xiaoluo-router"
    _DEFAULT_ROUTER_BASE_DIRNAME = "Qwen3-0.6B"
    #: 本地找不到基座时的 HuggingFace 回落源。
    _DEFAULT_ROUTER_BASE_REPO = "Qwen/Qwen3-0.6B"

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        """区分「本地路径」与「HuggingFace repo id」。

        repo id 的形态是 `Qwen/Qwen3-0.6B`——含斜杠但不含盘符、不以 `.` 开头。
        因此判据是**盘符或前导点**，而不是「有没有斜杠」。
        """
        return (
            value.startswith((".", "/", "\\"))
            or (len(value) >= 2 and value[1] == ":")
            or value.startswith("~")
        )

    @staticmethod
    def _resolve_model_path(value: str) -> Path:
        """相对路径按 `{PROJECT_ROOT}` → `{BACKEND_ROOT}` 依次探测；绝对路径原样返回。

        找不到时返回**按项目根解析出的那个路径**（而不是 None），调用方据此
        得到一条可读的「模型不存在于 X」日志，而不是一句含糊的「配置有误」。
        """
        raw = (value or "").strip()
        path = Path(raw)
        if path.is_absolute():
            return path
        for base in (PROJECT_ROOT, BACKEND_ROOT):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (PROJECT_ROOT / path).resolve()

    @property
    def local_router_mode(self) -> str:
        """归一化档位。

        `guard` 是历史预留档（从未启用），语义被 `active` 覆盖；直接折成 `active`，
        否则老 .env 会在升级后**静默退化成 off** —— 那是最难排查的一类故障。
        未知值一律按 `off` 处理：宁可不动，也不要猜。
        """
        raw = (self.LOCAL_ROUTER_MODE or "off").strip().lower()
        if raw == "guard":
            return "active"
        return raw if raw in ("off", "shadow", "active") else "off"

    def local_router_adapter_path(self) -> Path:
        """LoRA 适配器目录。配置优先，未配置则按默认位置探测。"""
        if (self.LOCAL_ROUTER_MODEL_PATH or "").strip():
            return self._resolve_model_path(self.LOCAL_ROUTER_MODEL_PATH)
        for base in (PROJECT_ROOT, self.model_root_path):
            candidate = base / self._DEFAULT_ROUTER_ADAPTER_DIRNAME
            if candidate.exists():
                return candidate
        return PROJECT_ROOT / "models" / self._DEFAULT_ROUTER_ADAPTER_DIRNAME

    def local_router_base_model(self) -> str:
        """基座模型：本地目录路径或 HuggingFace repo id。"""
        raw = (self.LOCAL_ROUTER_BASE_MODEL or "").strip()
        if raw:
            return str(self._resolve_model_path(raw)) if self._looks_like_path(raw) else raw
        for base in (PROJECT_ROOT, self.model_root_path):
            candidate = base / self._DEFAULT_ROUTER_BASE_DIRNAME
            if candidate.exists():
                return str(candidate)
        return self._DEFAULT_ROUTER_BASE_REPO

    def local_router_device(self) -> str:
        raw = (self.LOCAL_ROUTER_DEVICE or "auto").strip().lower()
        return raw if raw in ("auto", "cuda", "cpu") else "auto"

    def local_router_summary(self) -> dict[str, Any]:
        """本地 Router 的可展示状态。`active=False` 表示链路完全未被触碰。"""
        mode = self.local_router_mode
        adapter = self.local_router_adapter_path()
        return {
            "mode": mode,
            "active": mode != "off",
            "confidence_threshold": self.LOCAL_ROUTER_CONFIDENCE_THRESHOLD,
            "model_dir": self.LOCAL_ROUTER_MODEL_DIR or str(self.model_root_path / "local_router"),
            "trace_max_bytes": self.LOCAL_ROUTER_TRACE_MAX_BYTES,
            # Qwen 神经路由：路径只用于排障展示，不含任何凭据。
            "qwen_adapter_path": str(adapter),
            "qwen_adapter_present": adapter.exists(),
            "qwen_base_model": self.local_router_base_model(),
            "qwen_device": self.local_router_device(),
            "qwen_timeout_ms": int(self.LOCAL_ROUTER_TIMEOUT_MS),
            "qwen_max_new_tokens": int(self.LOCAL_ROUTER_MAX_NEW_TOKENS),
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
