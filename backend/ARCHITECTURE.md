# 小洛实验室 · 后端架构分析与优化报告

> **分析范围**：仅 `backend/` 目录（未修改 `frontend/` 及仓库其它目录）。
> **生成日期**：2026-09-21
> **代码规模**：`app/` 下 133 个 `.py` 文件，约 20,067 行；`tests/` 32 个测试文件；`migrations/` 1 个版本脚本。
> **方法**：AST 静态扫描 + 全仓符号引用统计 + 逐文件人工复核（未运行业务服务，只做导入与纯函数级验证）。
> **变更声明**：本轮所有代码改动均保持对外接口与既有行为不变，最小化且可回滚，详见 [第 6 节](#6-本轮已完成的优化改动原因--影响范围--风险--回滚)。

---

## 快速索引

| 章节 | 内容 | 何时看 |
| --- | --- | --- |
| [1. 总览](#1-总览技术栈与分层) | 技术栈、规模、分层图 | 第一次接触后端 |
| [2. 目录结构与模块职责](#2-目录结构与模块职责) | 完整目录树 + 每个包的职责 + 入口文件 | 找"某功能在哪个文件" |
| [3. 核心业务流程与调用链](#3-核心业务流程与调用链) | 7 条主链路（数据集/加工/EDA/ML/报告/Agent/Workflow） | 排查链路问题、改功能前评估影响 |
| [4. 问题清单](#4-问题清单) | 严重级别 + 文件:行号 + 修复建议 | 排期修 bug / 安全评审 |
| [5. 冗余清单](#5-冗余清单) | 重复实现、未抽取逻辑、死代码、未声明依赖 | 重构、减负 |
| [6. 已完成的优化](#6-本轮已完成的优化改动原因--影响范围--风险--回滚) | 本轮改了什么、为什么、风险、怎么回滚 | 代码评审 / 回滚 |
| [7. 后续建议](#7-后续建议按优先级) | 未做的改进项与优先级 | 排下个迭代 |
| [8. 附录](#8-附录分析方法与工具约束) | 分析手段、环境约束、易错点备忘 | 复现本次分析 |

**按角色速查**

- 想了解"请求怎么进来" → [2.3 入口文件](#23-入口文件) → [3.1 数据集主链路](#31-数据集上传--版本化)
- 想了解"Agent 怎么跑" → [3.6 Agent Turn](#36-agent-turn含-sse-与授权确认闭环)
- 想了解"Workflow 怎么跑" → [3.7 Workflow DAG](#37-workflow-dag-执行)
- 只想看"现在最该修什么" → [4.2 待处理 P0](#42-待处理-p0未修需决策)

---

## 1. 总览：技术栈与分层

### 1.1 技术栈

| 层 | 选型 | 说明 |
| --- | --- | --- |
| Web 框架 | FastAPI 0.115+ | 唯一 ASGI 入口 `app/main.py`；统一前缀 `/api/v1` |
| 校验 / 配置 | Pydantic v2 + pydantic-settings | `app/core/config.py`（`Settings`，按 `BACKEND_ROOT` 锚定 `.env`） |
| ORM / DB | SQLAlchemy 2.0 + SQLite + Alembic | `app/models/*`；库文件 `backend/data/xiaoluo.db` |
| 数据计算 | Polars + PyArrow | 列式计算，`pl.DataFrame` 是领域层通用数据协议 |
| 机器学习 | scikit-learn（+ numpy） | `app/ml_engine/*`，模型落盘 `data/experiments/<id>/runs/<rid>/` |
| 报告 | 自研 generator + 内联 SVG + fpdf2 | `app/reports/*`（图表为纯 SVG，无 matplotlib 依赖） |
| 存储抽象 | `app/storage/*`（local 实现） | 文件与产物统一走 StorageService |
| 日志 / 异常 | `app/core/logging.py` + `app/core/exceptions.py` | `redact()` 脱敏；业务异常统一 `AppException` 族 |

依赖声明（`pyproject.toml`）与真实导入存在差异，见 [5.4](#54-依赖与声明不一致)。

### 1.2 分层关系

```
┌──────────────────────────────────────────────────────────────┐
│  接入层  app/main.py  →  app/api/v1/*（12 个 router）          │  HTTP/SSE、参数校验、响应包装
└───────────────┬──────────────────────────────────────────────┘
                │  Depends
┌───────────────▼──────────────────────────────────────────────┐
│  依赖装配  app/api/deps.py                                     │  进程级单例：AGENT_STORE、WORKFLOW_SERVICE、Planner 缓存
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│  服务层（领域门面）                                            │
│  services/dataset_service · services/file_service              │  元数据 + 版本 + 文件落盘
│  data_engine/service.py（DataEngineService）                    │  数据加工统一门面
│  experiments/service.py（ExperimentService）                    │  训练/推理/对比
│  workflow/service.py（WorkflowService）                         │  DAG 生命周期
│  agent/runtime/*（AgentRuntime）                                │  Agent Turn 编排
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│  领域能力层                                                    │
│  analysis.py（质量/画像/EDA/可视化数据）· ml_engine/*           │
│  data_engine/{loaders,operations,merge,merge_multi}            │
│  reports/*（生成/叙述/导出）· tools/*（Agent 工具集）           │
│  workflow/{executor,runners,validator,state}                    │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│  数据访问 / 基础设施                                           │
│  models/*（SQLAlchemy）· core/database.py（Session）           │
│  storage/*（对象存储抽象）· core/{config,logging,middleware}    │
└──────────────────────────────────────────────────────────────┘
```

**分层约定与现状偏差**

| 约定 | 现状 |
| --- | --- |
| 路由层不写业务逻辑 | ✅ 基本遵守（路由只做参数校验 + 调 service + 包装 `ApiResponse`） |
| 领域层不依赖 FastAPI | ✅ `data_engine` / `ml_engine` / `analysis` 均无 FastAPI 导入 |
| 依赖方向自上而下 | ⚠️ 存在倒置：`app/analysis.py:22-23` 依赖 `app/data_engine/{exceptions,json_utils}`，而 `app/data_engine/service.py:32` 又依赖 `app/analysis` → 双向依赖（详见 [4.3-⑤](#43-待处理-p1)） |
| 异常统一 | ✅ `core/exceptions.py` 定义 8 类业务异常，由 `core/middleware.py` 的 ASGI 中间件统一兜底（未用 `add_exception_handler`） |

---

## 2. 目录结构与模块职责

### 2.1 顶层

```
backend/
├── app/                    # 应用源码（133 个 .py）
├── alembic.ini + alembic/  # 空目录，实际迁移在 migrations/
├── migrations/             # Alembic 迁移（versions/0001_initial_schema.py）
├── data/                   # 运行时数据：xiaoluo.db / datasets / experiments / reports / demo
├── exports/                # 导出产物样例
├── scripts/                # 演示数据与实验脚本（非服务运行时依赖）
├── tests/                  # pytest 用例（32 个）
├── Dockerfile
├── pyproject.toml          # 依赖 + ruff + pytest 配置
└── .env                    # LLM Key、Agent 预算等（core/config.py 按 BACKEND_ROOT 锚定读取）
```

### 2.2 `app/` 目录树与职责

```
app/
├── main.py                       ★ 唯一 ASGI 入口；注册 RequestContextMiddleware + api_router；/api/v1/health
├── __init__.py                   __version__
│
├── api/
│   ├── deps.py                   ★ 依赖装配与进程级单例（AGENT_STORE / WORKFLOW_SERVICE / Planner 缓存）
│   └── v1/
│       ├── __init__.py           ★ api_router（prefix=/api/v1），注册 12 个 router
│       ├── datasets.py           数据集 CRUD、版本列表
│       ├── dataset_analysis.py   preview / schema / profile / quality
│       ├── files.py              上传、列表、下载、删除
│       ├── processing.py         清洗/筛选/转换/聚合（生成新版本）
│       ├── merge.py              关联分析、预览、执行合并
│       ├── eda.py                探索性分析与 /visualize（图表数据）
│       ├── experiments.py        实验 CRUD + 训练/推理（含独立 ml_router）
│       ├── workflow.py           Workflow CRUD / run / cancel / clone
│       ├── agent.py              Agent 会话、运行、SSE、confirm/deny/cancel、trace
│       ├── reports.py            报告生成 / 导出 / 已保存报告管理
│       ├── settings.py           Agent 参数、LLM 配置与连通性测试、存储概览
│       └── _serializers.py       路由层共用序列化助手
│
├── core/
│   ├── config.py                 ★ Settings（env 锚定、database_url 相对路径解析、Agent 预算、LLM 配置）
│   ├── database.py               Engine / SessionLocal / get_db / Base
│   ├── exceptions.py             8 类业务异常（AppException 族）
│   ├── logging.py                结构化日志 + redact 脱敏 + request_id
│   └── middleware.py             RequestContextMiddleware（异常兜底、请求上下文）
│
├── models/                       SQLAlchemy ORM：dataset / dataset_version / file /
│                                 experiment / experiment_run / operation / base
├── schemas/                      Pydantic 出入参：common(ApiResponse/Pagination) / dataset / file
│
├── services/
│   ├── dataset_service.py        数据集与版本：创建/加载/落盘/最新版本
│   └── file_service.py           文件存储：上传流/读取/删除/文件名净化
│
├── storage/
│   ├── base.py / local.py        存储后端实现（本地文件系统）
│   ├── security.py               ★ 路径穿越防护（key 校验与规范化）
│   └── service.py                StorageService 门面（list/get/put/delete）
│
├── data_engine/
│   ├── service.py                ★ DataEngineService 门面（依赖 DatasetService）
│   ├── loaders.py                CSV/Excel/JSON/Parquet 加载，REGISTRY
│   ├── operations.py             OPERATION_REGISTRY：清洗/筛选/转换/聚合（1323 行，最大文件）
│   ├── param_validation.py       操作参数校验
│   ├── merge/                    合并子域：executor / key_analyzer / plan / report /
│   │                             schema_mapper / validator
│   ├── merge_multi.py            多表合并编排
│   ├── json_utils.py             json_safe 等 JSON 兜底
│   ├── exceptions.py             DataEngineException / TransformError / SchemaError…
│   └── base.py                   DataEngineBase 抽象契约（⚠️ 未被引用，见 5.3）
│
├── analysis.py                   统一只读分析：schema/profile/quality/EDA/可视化数据（1307 行）
│
├── ml_engine/
│   ├── base.py                   ModelAdapter 抽象（fit/predict 编排）
│   ├── classification.py / regression.py / clustering.py / dimensionality.py
│   ├── preprocessing.py          列预处理 Pipeline
│   ├── evaluation.py             指标 + 直方图/分桶
│   ├── explainability.py         特征重要性
│   ├── registry.py               模型注册表
│   └── exceptions.py             MLEngineException
│
├── experiments/
│   ├── service.py                ExperimentService：训练/推理/对比/持久化
│   └── comparator.py             实验对比
│
├── reports/
│   ├── models.py                 Report / ReportSection 数据结构
│   ├── generator.py              报告生成器（模板正文 + 表格 + 图表）
│   ├── report_charts.py          必保图表策划（分布/热力图/散点）
│   ├── chart_svg.py              纯 SVG 渲染（800×450，无 matplotlib）
│   ├── narrator.py               LLM 叙述：事实摘要 → JSON → 合并回报告
│   └── markdown.py / html.py / pdf.py   三种导出
│
├── workflow/
│   ├── service.py                ★ WorkflowService（内存仓库 + run/cancel/clone）
│   ├── executor.py               DAG 执行器（校验 → 拓扑排序 → runner）
│   ├── runners.py                build_default_runners / build_agent_semantic_runners
│   ├── validator.py / state.py / models.py
│
├── tools/                        Agent 工具集（注册到 ToolRegistry 供 LLM 调用）
│   ├── registry.py               工具注册与检索（retrieve_with_scores）
│   ├── base.py / context.py / result.py   工具基类/上下文/结果压缩
│   ├── builtin.py                内置工具装配
│   ├── data_tools.py / dataset_tools.py / eda_tools.py / ml_tools.py /
│   ├── report_tools.py / workflow_tools.py
│
└── agent/
    ├── runtime/
    │   ├── agent_runtime.py      AgentRuntime 门面（事件增量持久化 + 授权范围）
    │   ├── runtime.py            核心编排：_route / _run_plan / resume / deny / cancel
    │   ├── models.py             AgentStore（RLock + JSON 持久化）/ AgentRun / AgentSession / TokenLedger
    │   └── step_resolution.py    {{stepN.field}} 结构化依赖解析
    ├── planner/                  models / planner（计划生成 + 缓存）/ replanner（失败重规划）
    ├── context/                  builder（上下文构建）/ budget（预算）/ cache / models
    ├── permission/               manager / models / rules（风险分级与授权判定）
    ├── executor/                 executor（工具执行与重试记录）
    ├── validator/                validator / models（结果校验）
    └── llm/
        ├── base.py / capabilities.py / usage.py
        ├── openai_compatible.py  OpenAI 兼容协议（DeepSeek 等）
        ├── structured.py         JSON 模式输出与容错解析
        └── mock.py               离线/测试用 Mock
```

### 2.3 入口文件

| 入口 | 文件 | 说明 |
| --- | --- | --- |
| ASGI 应用 | `app/main.py` | 创建 `FastAPI`、注册 `RequestContextMiddleware`、`include_router(api_router)`、定义 `/api/v1/health` |
| 路由汇总 | `app/api/v1/__init__.py` | `api_router`（prefix `/api/v1`），注册 12 个 router（注意 `experiments.router` 与 `experiments.ml_router` 是**两个** router） |
| 依赖装配 | `app/api/deps.py` | 进程级单例 `AGENT_STORE`、`WORKFLOW_SERVICE`；`_shared_planner()`（带锁 + 上限 8 的缓存）；`get_db` 复用 |
| 配置 | `app/core/config.py` | `Settings`；`database_url` 属性把相对 sqlite 路径按 `BACKEND_ROOT` 解析（避免从项目根启动连到空库） |
| 迁移 | `migrations/env.py` + `alembic.ini` | 复用 `Settings.database_url`（勿改回直接用 `settings.DATABASE_URL`） |
| 脚本 | `scripts/demo_data.py`、`scripts/experiments/*` | 演示数据与实验脚本，非运行时依赖 |

### 2.4 进程级单例（影响并发与部署）

| 单例 | 位置 | 风险 |
| --- | --- | --- |
| `AGENT_STORE = AgentStore()` | `api/deps.py:23` | 会话/运行存内存 + `data/agent_store.json`；**多 worker 部署下不共享** |
| `WORKFLOW_SERVICE = WorkflowService(...)` | `api/deps.py:24` | 工作流与运行句柄全在内存；**多 worker 下 run/cancel 可能落到不同进程** |
| `_RESERVED_SESSIONS` | `api/v1/agent.py:54` | Agent 会话预留集合（堵住 `_ensure_idle` 与 worker 之间的竞态） |
| `_PLANNERS` 缓存 | `api/deps.py:29` | 有锁 + 上限 8，✅ 无问题 |

> 结论：当前架构**隐含要求单 worker**。若用 `uvicorn --workers > 1`，Agent SSE / confirm 与 Workflow run/cancel 会出现"请求落到另一个进程"的诡异失效。

---

## 3. 核心业务流程与调用链

### 3.1 数据集上传 → 版本化

```
POST /files/upload            files.py → FileService.upload_stream(_sanitize_filename)
POST /datasets                datasets.py → DatasetService.create
                                          → FileService.read → REGISTRY.load(扩展名匹配)
                                          → DatasetService.create_version(df)  # 写 parquet + 版本行
GET  /datasets                → _to_response(dataset) → latest_version（每条一次查询，N+1，见 4.3-④）
GET  /datasets/{id}/preview   → DatasetService.load_version → df.head
```

**不变式**：数据修改一律产生**新版本**，原版本永不覆盖（`data/datasets/<id>/v00000N.parquet`）。

### 3.2 数据加工

```
POST /processing/clean | filter | transform | aggregate
        → DataEngineService.对应方法
        → param_validation / OPERATION_REGISTRY 调度（data_engine/operations.py）
        → 新 df → DatasetService.create_version（新版本）

POST /merge/analyze-keys → key_analyzer
POST /merge/preview      → MergePlan + 预览
POST /merge/execute      → MergeExecutor → schema_mapper → 报告 → 新版本
```

### 3.3 EDA 与可视化

```
GET/POST /eda/*          → analysis.py（DescriptiveAnalyzer / CorrelationAnalyzer / EdaModule 族）
GET  /eda/visualize      → VisualizationBuilder.builders[chart]（analysis.py）
                         → 返回图表数据；渲染在前端 recharts
```

> ⚠️ 新增图表类型必须同时改 5 处（后端 builders、`/visualize` 的正则 pattern 与查询参数、`eda_tools.py` 的 `input_schema` enum、前端 `analysis.ts` 类型、前端 `VisualizationPanel.tsx`），漏一处即"图表用不了/被 422 拒绝"。

### 3.4 ML 实验

```
POST /experiments            → ExperimentService.create
POST /experiments/{id}/run   → preprocessing(Pipeline) → ModelAdapter.fit（ml_engine/*）
                             → evaluation 指标 → 落盘 model.pkl / pipeline.pkl
                             → ExperimentRun 行（status/metrics/artifacts）
POST /experiments/{id}/predict → 加载 pipeline+model → transform → predict
                             → 分类任务追加 predict_proba（本轮已限制为 preview 行，见 6.7）
GET  /experiments/compare    → comparator.py
```

### 3.5 报告生成（含图表与 LLM 叙述）

```
POST /reports/generate
   → version_row + load_version → df
   → ReportGenerator.generate(df, charts=build_report_charts(df))
        · 必保三件套：分布直方图 / 相关系数热力图 / 相关性散点图
        · 散点图不可用 |r|>=阈值 卡掉（orders 数据 r 仅 0.10），取最强相关对并标注 r
   → narrator.narrate_report(report_payload, provider)
        · build_facts()  → 事实摘要（本轮新增 _fit_limit 强制上限，见 6.3）
        · LLM 输出 JSON  → _missing_parts 完整性校验 → 放大预算重试（最多 3 次）
        · apply_narration() 并回报告（保留原表格/图表，追加分析正文）
   → storage 保存 JSON（data/reports/<key>.json）

POST /reports/export   → _report_from_dict（本轮改为逐字段取值，见 6.5）
                       → export_markdown / export_html / export_pdf
```

### 3.6 Agent Turn（含 SSE 与授权确认闭环）

```
POST /agent/sessions/{id}/messages  （api/v1/agent.py: post_message）
  1. _ensure_idle(session)            # 存在活动 run → 409
  2. _try_reserve(session_id)         # 进程内预留集合，堵住"检查→worker 创建"的竞态
  3a. stream=false → runtime.run(...) → store.persist(force=True) → 返回 summary
  3b. stream=true  → _sse_live_run()
        · 起 daemon 线程执行 runtime.run(...)，事件经 queue 推流
        · run 停在 WAITING_CONFIRMATION 时 SSE **保持打开**并持续 tail run.events
          （AGENT_SSE_CONFIRM_WAIT_SECONDS，默认 900s；本轮新增 tail_stop 断连检测）
        · 用户确认 → POST /agent/runs/{id}/confirm → runtime.resume() 同步执行
          → 新事件继续经同一条 SSE 推出（这是"点确认不卡死"的关键）
        · 终态/断连 → 结束流

runtime.run 内部：
  _route → ContextBuilder → _fill_tools/retrieve_with_scores
        → AgentPlanner.build_plan_resilient
        → _run_plan → _resolve_step_arguments（{{stepN.field}}）
        → executor.execute_step → registry.execute
        → PermissionManager.check（高风险 → WAITING_CONFIRMATION）
        → tool.execute → validator.validate → 失败则 replanner.replan
```

**授权语义**：一次确认放行**整份计划**内所有受控工具（不是逐个工具二次确认），越权由 PermissionManager 与 deny/cancel 兜底。

### 3.7 Workflow DAG 执行

```
POST /workflows           → WorkflowService.create
                             _build_workflow → _ensure_valid → validator.validate_workflow
POST /workflows/{id}/run  → WorkflowService.run（生成本次 run_id，登记句柄）
                             → WorkflowExecutor.execute
                                 validate_workflow → _topo_order → 逐节点 runner
                                 （runners.py: build_default_runners）
                             → 节点边界检查 is_cancelled → cancel_event
POST /workflows/runs/{id}/cancel → handle.cancel_event.set()
```

**数据传递**：链式 DAG（`dataset.read → quality_check → statistics → report.summary`）中分析节点不产出 df，靠 `_remember_df(ctx, df)` 写共享上下文 + `_pull_df(upstream, ctx)` 兜底读取；工具边界由 `sanitize_output` 剥离 `_df`。

---

## 4. 问题清单

### 4.0 严重级别定义

| 级别 | 含义 |
| --- | --- |
| **P0** | 必然崩溃 / 数据被覆盖 / 可被外部利用的安全问题，需立即处理 |
| **P1** | 特定输入或并发下出错、资源浪费明显、错误信息误导，应尽快处理 |
| **P2** | 可维护性与卫生问题，不修不影响运行 |

### 4.1 已修复（本轮，共 11 项）

详见 [第 6 节](#6-本轮已完成的优化改动原因--影响范围--风险--回滚)。摘要：

| # | 级别 | 位置 | 一句话 |
| --- | --- | --- | --- |
| 1 | P0 | `api/v1/agent.py:5` | 补 `import json`（trace 接口必崩） |
| 2 | P0 | `workflow/service.py:42` | 工作流仓库加锁（并发 id 重复覆盖） |
| 3 | P1 | `reports/narrator.py:128` | 事实摘要截断真正生效 |
| 4 | P1 | `api/v1/files.py:18` | 下载响应头注入防护 |
| 5 | P1 | `api/v1/reports.py:51` | 报告小节解包改为字段级取值 |
| 6 | P1 | `analysis.py:384` | 单样本 std 返回 None 导致 TypeError |
| 7 | P1 | `experiments/service.py:439` | 概率矩阵全量物化 |
| 8 | P1 | `agent/runtime/models.py` | 读取无锁 + 逐事件全量落盘 |
| 9 | P1 | `api/v1/agent.py:98` | SSE tail 无断连检测 |
| 10 | P2 | `api/v1/settings.py:148` | 连通性测试吞异常 |
| 11 | P2 | 2 处 | 删除未使用 import |

### 4.2 待处理 P0（未修，需决策）

| # | 位置 | 问题 | 影响 | 建议 |
| --- | --- | --- | --- | --- |
| ① | `api/v1/settings.py:98-110` | `PUT /api/v1/settings/llm` **无任何鉴权**即可改写进程级 `settings.LLM_BASE_URL / LLM_MODEL / LLM_API_KEY` | 任何人可把全平台 LLM 流量劫持到自己的网关（凭据外泄 + 提示注入 + 结果篡改） | 最小改动：新增可选 `ADMIN_API_TOKEN` 配置，设置后要求 `X-Admin-Token` 头；未设置时行为不变。更彻底：引入统一鉴权依赖 |
| ② | 全局 | 后端**完全没有认证/授权体系**（全仓无 `HTTPBearer` / token 依赖） | 所有写接口（删数据集、删实验、改设置）对内网任何人开放 | 至少在网关/反向代理层加鉴权；服务不要绑定 `0.0.0.0` 暴露公网；内部部署走 127.0.0.1 + Vite proxy |
| ③ | `api/deps.py:23-24`、`api/v1/agent.py:54` | `AGENT_STORE` / `WORKFLOW_SERVICE` / `_RESERVED_SESSIONS` 为**进程内**状态 | 多 worker 部署时 SSE 与 confirm、run 与 cancel 可能落在不同进程 → 静默失效 | 部署文档固定 `--workers 1`；或改用 Redis/DB 共享状态 |

> 注：`app/main.py` **未配置 CORS**（当前靠 Vite 代理同域规避）。这是安全的默认值，请勿直接改成 `allow_origins=["*"]`；确需跨域时白名单前端源。

### 4.3 待处理 P1

| # | 位置 | 问题 | 影响 | 建议 |
| --- | --- | --- | --- | --- |
| ① | `agent/runtime/models.py:129-135` | `AgentStore` 每次持久化都把**全部会话 + 全部运行 + 全部事件**序列化重写（本轮已做 0.5s 节流，但仍是 O(全量)） | 数据量增长后单次写入可达数 MB，tail 期间抖动 | 迁移 SQLite（已有 ORM）或按 run 分片/增量追加 |
| ② | `api/v1/agent.py:126-143` | SSE tail 最长 900s（`AGENT_SSE_CONFIRM_WAIT_SECONDS`）、裸 `threading.Thread(daemon=True)`、**无并发上限** | 大量挂起连接会耗尽线程；客户端全部断开后线程仍存活至超时 | 线程池 + 每会话并发上限 + 缩短默认等待（如 300s）并配合前端重连 |
| ③ | `api/v1/settings.py:99-110` | 直接改写全局 pydantic `settings` 对象 | 进程内全局可变状态：无并发保护、重启即丢失、与 `.env` 不一致（UI 显示"已配置"但重启回退） | 引入显式"运行时覆盖层"，读取时标注来源（`env` / `runtime`），并提供"写回 .env"开关 |
| ④ | `api/v1/datasets.py:22-41` | 列表接口每条数据集再查一次 `latest_version`（N+1） | 数据集数量增长后线性变慢 | 一次批量查询版本（`WHERE dataset_id IN (...)`）后内存映射 |
| ⑤ | `analysis.py:22-23` ↔ `data_engine/service.py:32` | 分层倒置：`analysis` 依赖 `data_engine.exceptions/json_utils`，而 `data_engine.service` 又依赖 `analysis` | 双向依赖；未来任一方结构调整都会引发循环导入 | 抽 `app/common/{errors.py, json_utils.py}`，两侧改为依赖公共层 |
| ⑥ | `api/v1/reports.py:46-47` | `ExportRequest.report: dict[str, Any]` 无大小/深度上限 | 超大 body 会造成内存与 CPU 飙高（导出 PDF 尤其明显） | 字段级限制（如 sections ≤ 200、总字符 ≤ 2MB），超限返回 422 |
| ⑦ | `api/v1/files.py` upload | 上传未限制大小 | 大文件打满内存（`FileService.upload_stream` 仍会读进内存） | 增加 `MAX_UPLOAD_BYTES` 配置与流式落盘 |
| ⑧ | `api/v1/eda.py` 等 | 部分路由 `except Exception` 后返回 200 + 错误文案（与 `/settings/llm/test` 同类问题） | 监控无法区分成功与失败 | 统一策略：业务可预期失败用 4xx；仅"连通性探测"类接口允许 200+ok:false，但必须 `logger.warning`（本轮已对 settings 补齐） |
| ⑨ | `experiments/service.py` | 训练/推理无超时与样本量上限 | 大数据集训练会长时间独占请求线程 | 加 `max_rows` 采样上限与训练超时（后台任务化更佳） |

### 4.4 待处理 P2（卫生问题）

- `app/data_engine/base.py` 整个模块未被引用（抽象契约无实现方）。
- 若干模型/字段命名与 DB 列名差异（`schemas/dataset.py:43` `schema` 字段触发 Pydantic 告警）。
- `tests/pip_install.log` 等临时产物被提交进仓库。
- `data/`、`exports/` 直接落在项目目录（建议纳入配置与备份策略，非缺陷）。

---

## 5. 冗余清单

### 5.1 重复实现（同一逻辑多份）

| 逻辑 | 位置 | 说明 |
| --- | --- | --- |
| JSON 安全化 | `experiments/service.py:48 _jsonable`、`ml_engine/evaluation.py:172 _jsonable`、`data_engine/json_utils.py:13 json_safe` | 3 份近似实现，行为细节（NaN/Decimal/时间处理）可能不一致 |
| 文本/对象截断 | `tools/result.py:9 _compact_value`、`reports/narrator.py:406 _shrink`、`agent/context/models.py:21 clip_obj` | 3 份，策略不同（max_items / max_chars / 递归深度） |
| JSON 提取（去代码围栏） | `agent/llm/structured.py:19 _FENCE_RE` 与 `reports/narrator.py:31 _FENCE_RE` | **完全同值**的正则各写一份；解析实现也有两份（`structured.py:30` vs `narrator.py:303`，后者多了"闭合抢救"逻辑） |
| 直方图分桶 | `analysis.py:1013 histogram` 与 `ml_engine/evaluation.py:153 _histogram` | 两套分桶口径 |
| `_make_estimator` | `ml_engine/base.py:53`（abstract）、`classification.py:26`、`regression.py:24`、`dimensionality.py:33` | 三个子类的实现都是同一段 `{**default_params, **params}` 合并；可在基类提供默认实现 |
| 「get_version_row + load_version」两段式 | `eda.py:33`、`dataset_analysis.py:26`、`reports.py:69`、`merge.py:74`、`data_engine/service.py`（4 处）、`tools/*`（5 处）等 **约 13 处** | 建议抽 `DatasetService.load(dataset_id, version=None) -> (row, df)` |

### 5.2 可复用但未抽取

- 「高风险工具授权判定 + 待确认负载构造」散落在 `agent/permission/*` 与 `agent/runtime/runtime.py`，可收敛成一个 `AuthorizationService`。
- 「图表规格 → SVG」只在 `reports/chart_svg.py`，但 `analysis.py` 的可视化数据构造与之耦合（新增图表类型要改 5 处，见 3.3）。
- 「工具结果压缩/预算记账」在 `tools/result.py`、`agent/context/budget.py`、`agent/runtime/models.py` 三处各算一遍。

### 5.3 死代码 / 未引用（已用符号引用统计复核，均为"仅定义处出现"）

| 位置 | 符号 | 备注 |
| --- | --- | --- |
| `data_engine/base.py:20` | `DataEngineBase`（整个模块） | 无实现方、无引用 |
| `data_engine/exceptions.py:26` | `SchemaError` | 未使用 |
| `data_engine/json_utils.py:84` | `scalar_json_safe` | 未使用 |
| `data_engine/loaders.py:88 / :98` | `read_bytes_or_path` / `loader_error` | 未使用 |
| `experiments/service.py:318` | `_default_preprocessing` | 未使用 |
| `agent/runtime/models.py:15` | `EVENT_TYPES` | 未使用（`agent_runtime.py` 另写了一份集合字面量） |
| `agent/runtime/models.py:25` | `AgentTokenLedger.record_context_saving` | 未使用（有 `record_result_saving`/`record_cache_hit` 在用） |
| `agent/runtime/runtime.py:88` | `_needs_data_tools` | 未使用 |
| `agent/executor/executor.py:35` | `ToolCallRecord.to_llm_dict` | 未使用 |
| `tools/registry.py:57` | `ToolRegistry.retrieve` | 未使用（在用的是 `retrieve_with_scores`） |
| `reports/chart_svg.py:56` | `to_svg_data_uri` | 未使用（Markdown 导出走的是内联 `<img>` 拼接） |
| `core/logging.py:105` | `get_request_id` | 未使用 |
| `schemas/dataset.py:116` | `DatasetListResponse` | 未使用 |
| `schemas/file.py:27` | `FileListResponse` | 未使用 |
| `tools/workflow_tools.py:9`、`agent/runtime/step_resolution.py:11` | 未使用 import | ✅ 本轮已删除 |

**不要误删（有引用，只是不在 AST 名字层面）**

- `agent/llm/mock.py` `MockLLM`：测试与脚本引用。
- `agent/permission/rules.py` `DEFAULT_TOOL_RISKS`：`tests/test_permission_llm.py` 断言使用。
- `core/exceptions.py` `PermissionDeniedException` / `ToolExecutionException`：仅测试引用，属异常公共面，建议保留。
- `workflow/state.py`、`tools/builtin.py`、`agent/validator/models.py`：均在用。
- `storage/security.py` 的路径穿越防护：在用且完备，**勿改**。

### 5.4 依赖与声明不一致

- `numpy` 被 `ml_engine/{evaluation,explainability,preprocessing}.py` **直接 import**，但未在 `pyproject.toml` 的 `dependencies` 中声明（目前靠 scikit-learn 传递依赖）。建议显式声明，避免依赖树变化后直接 ImportError。
- AST 全仓扫描：**未使用顶层 import 为 0 处**（删除上述 2 处后）。

---

## 6. 本轮已完成的优化（改动原因 / 影响范围 / 风险 / 回滚）

> 全部改动均满足：不新增/删除接口，不改变正常路径的返回值，不改变数据库结构。

### 6.1 补 `import json`（P0）

- **文件**：`app/api/v1/agent.py:5`
- **原因**：`_trace_markdown()` 使用 `json.dumps` 处理工具参数与事件负载，但模块从未导入 `json`（原 `:120` 处用 `__import__("json")` 应急）。只要 run 带有 `tool_calls` 或 `events` —— 即**几乎所有 run**，访问 `GET /agent/runs/{id}/trace?format=md` 就会抛 `NameError` → 500。
- **改动**：顶部加 `import json`；同时把 `:120` 的 `__import__("json")` 换成 `json`（行为完全一致，去掉特例写法）。
- **影响**：仅 trace 导出接口从"必然 500"变为可用。
- **风险**：无。
- **回滚**：删除该 import 行 + 还原 `__import__` 写法。

### 6.2 WorkflowService 加锁（P0）

- **文件**：`app/workflow/service.py:42`（新增 `_wf_lock = threading.RLock()`），覆盖 `:59 create`、`:76 list`、`:106 update`、`:112 delete`、`:174 clone`；`run` 的 run_id 生成并入 `_run_lock` 临界区。
- **原因**：`_next_id` 自增与 `_workflows` 写入此前**完全无锁**（`_run_lock` 只保护 `_runs`），而 `api/deps.py:24` 的 `WORKFLOW_SERVICE` 是**进程级单例且被两个 router 共用**。并发创建会算出相同 id 并互相覆盖；`run_id` 由 `len(self._runs)+1` 推导却在锁外读取 → 并发 run 拿到相同 run_id，`cancel` 会作用到被覆盖的句柄上而失效。
- **影响**：仅并发正确性；单线程行为不变。
- **风险**：极低。用 `RLock` 且临界区内只做字典操作，不嵌套获取其它锁，不存在死锁路径。
- **验证**：50 线程并发 `create` + 30 线程并发 `clone` → 80 个 id 全唯一，0 异常。
- **回滚**：移除 `with self._wf_lock:` 块（保留 `run` 中的 `_run_lock` 合并即可单独回退）。

### 6.3 事实摘要截断真正生效（P1）

- **文件**：`app/reports/narrator.py:128`（新增 `_fit_limit()`），`build_facts()` 末尾改为 `return _fit_limit(payload, limit)`。
- **原因**：原实现把裁剪后的字符串赋给局部变量 `text` 就 `return payload` —— 返回的是**未裁剪**的对象，`AGENT_REPORT_NARRATION_MAX_CHARS`（默认 24000）形同虚设。大报告的事实摘要会击穿 LLM 上下文，表现为叙述静默失败、报告退回模板。
- **改动**：逐级降级裁剪 —— 章节表格 → 模板正文截短至 240 字 → 去掉模板正文 → 去掉 experiments/existing_conclusions/chart_titles/dataset → 从尾部丢弃章节（至少保留 1 节），并在超限时置 `truncated: True`。
- **影响**：只有超限的报告会拿到更精简的事实摘要；未超限报告**逐字节不变**。降级后 `headings` 随之变少，`_missing_parts` 与 `apply_narration` 仍按同一份 headings 工作，未覆盖的章节保留模板正文（原有行为）。
- **风险**：极端小的 `max_chars` 会让 LLM 可依据的信息变少（原本是直接失败），属预期内的降级。
- **验证**：`limit=24000/6000/1500/300` 四档实测长度分别为 14790/4350/1467/239，均不超限。
- **回滚**：把 `return _fit_limit(payload, limit)` 还原为原 4 行。

### 6.4 下载响应头注入防护（P1）

- **文件**：`app/api/v1/files.py:18`（新增 `_content_disposition()`），`:99` 的下载接口改用它。
- **原因**：`original_name` 是用户可控输入（`_sanitize_filename` 只作用于落盘的 `stored_name`）。原代码 `f'attachment; filename="{file.original_name}"'` 遇到文件名里的 `"` 或 CRLF 会截断/注入响应头。
- **改动**：剔除 `"` `\` 与所有控制字符生成 ASCII 安全名，同时按 RFC 5987 输出 `filename*=UTF-8''<percent-encoded>` 保留中文原名。
- **影响**：下载文件名对正常文件保持一致（中文名通过 `filename*` 正确还原，比原先更符合标准）。
- **风险**：极低。
- **验证**：`a"b\r\nX-Evil: 1.csv` → `filename="a_b__X-Evil__1.csv"; filename*=UTF-8''a_b__X-Evil%3A%201.csv`；`报表 2026.csv` → `filename*=UTF-8''%E6%8A%A5%E8%A1%A8%202026.csv`；空名 → `download`。
- **回滚**：还原为原 f-string。

### 6.5 报告小节解包改为字段级取值（P1）

- **文件**：`app/api/v1/reports.py:51`（新增 `_section_from_dict()`），`_report_from_dict()` 调用它。
- **原因**：`ReportSection(**s)` 直接解包外部输入 —— `sections` 里出现非 dict 元素、或带 `ReportSection` 不认识的键，就抛 `TypeError` 并被兜底层转成 500，前端只看到"导出失败"。
- **改动**：按 `heading/content/tables/charts` 逐字段取值，`tables`/`charts` 过滤非 dict 元素；非法小节抛 `ValidationException`（422）并回带问题内容的截断快照。
- **影响**：合法输入行为不变；非法输入从 500 变为可读的 422。
- **风险**：低。若上游曾依赖"额外键被透传"——不可能，因为原本就会崩溃。
- **回滚**：还原 `[ReportSection(**s) for s in ...]`。

### 6.6 单样本标准差导致 TypeError（P1）

- **文件**：`app/analysis.py:384`（`compute_outlier_bounds`，zscore 分支）
- **原因**：Polars 的 `std(ddof=1)` 在只有 1 个非空样本时返回 `None`，`float(None)` 抛 `TypeError`，整个质量检查失败。
- **改动**：`std_raw = clean.std(); std = float(std_raw) if std_raw is not None else 0.0`，随后原有的 `std == 0` 分支自然走"常数列"处理。
- **影响**：仅修复崩溃路径；多样本行为逐值不变。
- **风险**：极低。
- **回滚**：还原为 `float(clean.std())`。

### 6.7 分类推理概率矩阵全量物化（P1）

- **文件**：`app/experiments/service.py:439`
- **原因**：`probabilities = proba.to_dicts()` 为**全表**构造 Python dict，而下面只有前 `preview` 行被写进 payload，`limit` 参数对它完全无效 → 百万行数据集会瞬间构造百万个 dict。
- **改动**：把 `rows/preview` 的计算提到概率获取之前，`probabilities = proba.head(preview).to_dicts()`。
- **影响**：返回内容不变；内存与耗时随 `limit` 生效。
- **风险**：极低（polars DataFrame 具备 `.head`）。
- **回滚**：还原为 `proba.to_dicts()`。

### 6.8 AgentStore 读取加锁 + 持久化节流（P1）

- **文件**：`app/agent/runtime/models.py`（`get_session`、`get_run`、`list_sessions` 持锁；`persist(*, force=False)` 节流 0.5s）、`app/agent/runtime/agent_runtime.py:27`、`app/agent/runtime/runtime.py`（deny/cancel 强制落盘）、`app/api/v1/agent.py:143/226/250`
- **原因**：
  1. `get_session/get_run/list_sessions` 不持锁；`list_sessions` 无锁遍历 `dict.values()` 时若另一线程正在增删会话，会抛 `RuntimeError: dictionary changed size during iteration` → 500。
  2. `agent_runtime.py` 在**每个事件**都调 `persist()`，而 `_persist_locked()` 每次把全部会话+运行+事件全量序列化重写，一次运行会产生数十次 O(全量) 写盘。
- **改动**：读取加 `RLock`（已有锁，可重入，无死锁风险）；`persist()` 增加 `force` 参数与 0.5s 最小间隔，**终态事件（completed/failed）与 cancel/deny/confirm/会话变更一律 `force=True` 立即落盘**。
- **影响**：运行中的中间态落盘频率降低（重启时未终态运行本来也会被标记为"中断"并保留过程），终态数据不丢。
- **风险**：低。代价是进程被 `kill -9` 时可能丢失最后 0.5s 内的中间事件（终态不受影响）。
- **回滚**：`persist()` 去掉 force 参数判断；三处读取去掉 `with self._lock`。

### 6.9 SSE tail 断连检测（P1）

- **文件**：`app/api/v1/agent.py:98`（`tail_stop`）、worker 循环条件、`generate()` 的异常处理
- **原因**：SSE tail 最长挂 `AGENT_SSE_CONFIRM_WAIT_SECONDS`（默认 900s），且完全没有断连检测 —— 浏览器关闭/刷新后线程仍存活到超时，堆积后耗尽线程。
- **改动**：`generate()` 在流被关闭（`GeneratorExit`）或发送异常时置位 `tail_stop`，worker 的下一次循环立即退出。
- **影响**：对保持连接的客户端行为完全不变。
- **风险**：低。仅新增一个 `threading.Event` 的读取；不触碰运行本身，`resume/confirm` 路径不受影响。
- **回滚**：删除 `tail_stop` 判断与 `generate()` 的 try/except。

### 6.10 LLM 连通性测试异常留痕（P2）

- **文件**：`app/api/v1/settings.py:148`
- **原因**：`except Exception` 把 500 吞成 200（该接口按设计返回 `ok:false`，可接受），但异常**完全不留日志**，网关/证书/DNS 类故障在服务端不可见。
- **改动**：补 `logger.warning(..., exc_info=True)`（带 base_url/model，不含 key）。
- **影响**：仅日志。
- **风险**：无。

### 6.11 删除 2 处未使用 import（P2）

- `app/tools/workflow_tools.py:9`（`json_safe`）、`app/agent/runtime/step_resolution.py:11`（`ToolCallRecord`）
- 影响：无（全仓 AST 扫描确认零引用）。删除后全仓未使用 import 数为 0。

**验证汇总**：11 个改动文件全部通过 `py_compile`；`app.*` 相关模块导入冒烟通过（无 ImportError）；三处纯逻辑修复（截断上限、响应头、并发 id 唯一性）已用脚本实测通过。

---

## 7. 后续建议（按优先级）

1. **补最小鉴权**（P0-①/②）：先给 `PUT /settings/*` 这类"改全局凭据"的接口加可选管理员令牌，再逐步把写接口纳入统一鉴权依赖。
2. **明确单 worker 约束**（P0-③）：在启动脚本/部署文档中固定 `uvicorn --workers 1`，并在配置里加显式注释；长期方案是把 Agent/Workflow 状态外置。
3. **AgentStore 持久化换实现**（P1-①）：优先迁到 SQLite（ORM 已具备），或按 run 分文件 + 增量追加。
4. **抽公共层解双向依赖**（P1-⑤）：`app/common/{errors,json_utils}`，顺带消灭 3 份 `json_safe/_jsonable`。
5. **合并重复实现**（P2）：JSON 安全化 1 份、截断策略 1 份、`_FENCE_RE` 与 JSON 解析 1 份、直方图分桶 1 份、`_make_estimator` 基类默认实现、抽 `DatasetService.load(dataset_id, version)` 消灭 13 处两段式。
6. **清理死代码**（P2）：按 5.3 表逐项删除（每项独立提交，便于回滚）。
7. **补齐上传/导出/训练的体积与超时上限**（P1-⑥⑦⑨）。
8. **依赖声明补齐** `numpy`（5.4）；可选引入 `deptry`/`ruff --select F401` 进 CI，防止死代码回流。

---

## 8. 附录：分析方法与工具约束

**分析手段**

1. 目录树与规模统计（node 脚本遍历，排除 `__pycache__` / `.venv`）。
2. 未使用 import：AST 扫描（顶层 `Import`/`ImportFrom` × 全仓 `Name`/`Attribute` 引用）。
3. 死代码候选：顶层定义 × 全仓符号引用统计，再逐个 `grep` 人工复核（排除"仅测试引用"与"动态分发"）。
4. 缺陷定位：`Read` + `Grep` 逐文件复核子代理产出的事实清单，并对每个结论给出可复现的文件:行号。
5. 修复验证：`py_compile` 全量编译 + 模块导入冒烟 + 三处纯逻辑的脚本实测。

**环境约束与易错点（下次直接照做）**

- 本机 Bash 缺 coreutils（`ls/cat/head/grep/dirname` 不可用），PowerShell 工具常返回空 stdout；**可靠途径是 node 绝对路径 + Glob/Grep/Read**，需要看长输出时先写入临时文件再 `Read`。
- 跑 Python 必须用 `backend/.venv/Scripts/python.exe`（系统 Python 缺 polars，会导致"端口监听成功但 app 导入失败、请求悬挂到超时"）。
- 数据库路径是相对路径，`Settings.database_url` 已按 `BACKEND_ROOT` 解析；**不要改回直接用 `settings.DATABASE_URL`**，否则从项目根启动会连到空库。
- 改 `backend/.env`（LLM Key、Agent 预算）后**必须重启后端**才生效。
- 仓库 git 已损坏（`fatal: bad object HEAD`），无法用 git 做基线对比，改动请自行留注释便于回滚。
