# 变更日志

本文件记录对外可见的重要变更。格式参考 Keep a Changelog：
`新增 / 修复 / 优化 / 文档 / 破坏性变更`。日期为合并日期。

## [Unreleased] — Agent 后端彻底重构

**旧 Agent 架构整体删除，不做兼容式修补。** 目标是一个简单、清晰、可扩展、
低 Token 消耗的 Data Analysis Agent。

### 破坏性变更

- 移除 `app/agent/` 下的 `runtime/`、`loop.py`、`state.py`、`context/`、
  `permission/{manager,models,rules}`、`executor/`、`validator/`、`trace/`，
  以及 `app/local_router/`（本地神经路由，Qwen3-0.6B + LoRA）。
- 移除 SSE 事件类型 `preflight` 与 `replanning`（随「失败即重规划」一起删除）。
- 移除 `GET /agent/runs/{id}/events`（前端未使用）。
- **对前端无影响**：所有端点路径、SSE 事件名、`AgentRun` 字段名、
  `RunStatus` / `ToolCallStatus` 枚举值、`capabilities` 三层结构全部保持原样。

### 新增

- **扁平 9 模块替代深层分包**（`app/agent/`，约 2 600 行 → 旧架构约 5 500 行）：
  `engine` / `intents` / `playbooks` / `slots` / `answer` / `models` / `store` /
  `permission` / `llm` / `channels`。
- **规则意图路由**（`intents.py`）：关键词打分（强 +3 / 弱 +1 / 阈值 3），
  零 LLM、零网络、纯函数，26 个意图。
- **固定 playbook**（`playbooks.py`）：意图 → 写死的工具链，支持
  `carry`（步骤间传参）、`fallback_tool`（缺参数时退化到粗粒度工具）、`skippable`、
  `ask_if`（上一步跑成功了但结论里有必须人来定的选择时，挂起反问而不是猜）。
- **零 Token 默认路径**：未配置 `LLM_API_KEY` 时路由、执行、答案渲染全部照常完成，
  `token_usage.llm_calls == 0`。LLM 只在「规则读不出必填参数」与「答案润色」
  两处可选介入，且受 `MAX_LLM_CALLS=4` 限制。
- **有界执行**：`MAX_STEPS=8` / `MAX_LLM_CALLS=4` / `WALL_CLOCK_SECONDS=300` /
  单 run 事件上限 500。超界如实失败，不重规划。
- **存储三条防线**（`store.py`）：原子写（`.tmp` + `os.replace`）、
  损坏自愈（改名 `.corrupt.<ts>` 后空存储启动）、事件硬性封顶。
  新增**监听者机制**，SSE 实时取事件而不是轮询 store。
- **测试套件**（`backend/tests/`，242 个用例）：覆盖路由、playbook、权限、
  槽位（含可微调参数）、渲染、存储、引擎、确认闭环、未决反问、会话记忆、
  配置落盘、HTTP/SSE 契约。全程 `LLM_REMOTE_ENABLED=false`，
  **不发起任何真实模型调用**。

### 修复

- **高风险操作既不给确认机会、确认后又必然失败**（P0，两个独立成因叠加）：
  1. `ToolRegistry.execute` 先判 `decision.denied` 再判 `needs_confirmation`。
     而 NEEDS_CONFIRMATION 的 `allowed` 同样是 False，于是高风险工具被当成
     「权限不足」抛 `ToolPermissionError` —— 用户点完「允许」之后，
     仍以「需要你确认后才会执行」失败一次。现在判定顺序改为先确认后拒绝，
     并把 `PermissionDecision.denied` 的语义收敛为「被规则挡死，不是等确认」。
  2. `Tool.describe` 原样下发类属性 `requires_confirmation`，而几乎没有工具显式
     声明过它：`ml.train` 明明写着 `risk_level = "high"`，对外却自称
     「低风险、可自动放行」。前端据此默认自动放行，用户**连确认弹窗都见不到**。
     现在自描述下发 `requires_confirmation = 显式开关 or 高风险等级`，
     与裁决器共用同一个 `risk_needs_confirmation()`。
- **「没法定」被当成结论，链路自动挑默认值开跑**（P0，答非所问）：
  用户问「探索这两个数据集间的联系，并选用合适的机器学习分析」，
  拿到的却是 2,964,624 行上跑出来的 kmeans（8 簇、轮廓系数 0.1316）。四个成因：

  1. **没有能回答「两个数据集有什么联系」的工具**。请求里既没有「合并」
     也没有「相关性」，唯一命中的强关键词是「机器学习」，整句被吸进
     `Intent.ML_TRAIN`。新增只读工具 `dataset.relation`（共同字段 /
     类型是否一致 / 值域重叠度 / 推荐关联键 / 注意事项）与 `Intent.RELATION`
     及其 playbook（`dataset.inspect → dataset.relation → ml.detect_task`，
     不开训）；`RELATION` 在优先级表里排在建模意图之前。
  2. **复合诉求里没执行的那一半被静默丢弃**。`RouteResult` 新增 `secondary`，
     路由事件里明说「本次只执行主意图」。
  3. **`ml.detect_task` 推断不出目标列时仍返回 `success=True, task="clustering"`**，
     一个结论形状的回答，回答的其实是一个不可判定的问题。现在它回传
     `needs_target=True`（以及 `row_count`、候选列），`ml.train` 也据此失败
     而不是硬跑；`PlaybookStep` 新增 `ask_if` / `ask_slot` / `ask_options_key`，
     链路在开训前**挂起问用户**，并把候选列按「适合回归 / 适合分类」分组
     作为可点选项返回。
  4. **`carry` 把过期的任务类型带到了下一步**。目标列是后补的，
     `task` 必须重新判定 —— 带着 `clustering` 去训新选的数值列会把回归跑成聚类。
     现在 `ml.train` 只 carry `target`。
- **答案只列数字、不给判断**：`silhouette = 0.1316` 原样出现在「训练成功」下面，
  用户无从判断好坏。答案渲染新增「解读与下一步」：轮廓系数 / R² / F1 阈值解读、
  「目标列未确定 ⇒ 本次没有训练任何模型」、百万行以上提醒先抽样验证。
  解读同时进入 LLM 润色的事实摘要。
- **引擎纵深防御**：注册表若仍抛 `ToolConfirmationRequired`，一律回到挂起等确认，
  绝不转成「执行失败」—— 高风险操作最坏的表现就是不给用户选择权。
- **权限枚举与工具层取值不一致**（严重）：`Permission` 定义的是 `write_data`，
  而工具层声明的是 `modify_data` / `analyze_data` / `create_version`。
  结果是除 `dataset.list` 外几乎所有工具都被判 DENY，而 Agent 只是「正常失败」，
  极不显眼。现在枚举以工具层实际取值为准，并在裁决处做字符串归一化。
- **风险等级写成裸字符串**：5 个工具写 `risk_level = "high"`，
  触发 `AttributeError: 'str' object has no attribute 'needs_confirmation'`。
  同样在裁决处归一化，无法识别的取值一律按 MEDIUM 处理。
- **会话未绑定数据集时把工具打崩**：`dataset_id` 为 None 直接传进工具导致
  `KeyError`。现在把工具 schema 声明的 `dataset_id` / `left_dataset_id`
  视作隐式必填，缺了就反问用户；退化到 `fallback_tool` 后会重新判缺。
- **澄清答案键名不匹配**：`slot.run_id` 与 `run_id` 两种写法都要认，
  否则同一个问题会被反复追问。
- 路由关键词补漏：「预览前10行」「看看整体概览」「看看这个数据集的质量」「新增一列」
  此前会掉进对话兜底；同时移除 OVERVIEW 的弱关键词「分析」，
  它太泛，会把「分析一下相关性」从 CORRELATION 手里抢走。

### 修复（用户报障三连：配置不持久 / 历史丢失 / 不会调参数）

- **每次启动平台都要重新「应用到 Agent」**（P1）：`PUT /settings/llm` 只改进程内存，
  进程一重启就退回 `.env`，于是 API Key 显示「未配置」，用户被迫重填一遍。
  新增 `app/core/runtime_settings.py`：把大模型连接配置（白名单 7 个键）原子写到
  `{DATA_ROOT}/llm_settings.json`（`mkstemp + fsync + os.replace`，权限 0600），
  启动时由 `config._restore_persisted_llm()` 回填。读写失败一律不挡启动；
  落盘值优先于 `.env`（界面上填的那版才是用户要用的）。`GET /settings/llm`
  新增 `persisted` 字段供设置页显示「重启后是否还在」。
- **去别的模块逛一圈回来，助手的回复全没了**（P1）：`session.history` 是前端重建
  对话的唯一数据源（``useAiSession.activate`` 直接把它映射成消息列表），
  而引擎从头到尾只写过用户消息。现在 `_finish()` 前由 `_remember_answer()` 把
  助手这一轮的话（完成用 `final_answer`、失败用 `error`）写进历史。
- **追问时模型答「没有看到你的数据字段」**（P1，割裂感）：对话路径此前是
  「无上下文的一问一答」，模型手里只有当前这八个字。现在 `render_chat` 带上
  `history`（最近 8 条真实对话）与 `context`（绑定数据集的名称与列名 +
  本会话上一次分析的结论），并要求只能基于这些事实作答。
- **字段名被截没了**：喂给模型的事实摘要里，列表类字段固定只取前 5 条，
  关系分析列出 20 个同名字段时模型只看到 5 个，于是写出
  「共有 3 个同名字段（具体字段名本次未列出）」。改为按**字符预算**收口
  （最多 20 项 / 800 字符），事实摘要总预算 3000 → 6000 字符，
  并让 `_collect_facts` 认得 `column` / `feature` 这类字段名键。
- **用户点名的模型与超参一个都不生效**（P1，「只会调工具不会调参数」）：
  `model` / `params` 从来不是槽位，抽出来了也会被 `defaults={"model": "auto"}`
  盖掉；LLM 抽取时只拿到 `["model"]` 这种光秃秃的键名，看不到 enum，只能填 `auto`。
  三层修复：
  1. `PlaybookStep` 新增 `tuned_slots`（可微调槽位）—— 抽到就覆盖 `defaults`，
     抽不到用默认值，**绝不反问**（「请问用什么模型」是废话）。
     `ml.train` 声明 `tuned_slots=("model", "params")`。
  2. 规则抽取认模型族名（随机森林 / 逻辑回归 / KNN / KMeans / DBSCAN …）
     与超参说法（「分成 5 簇」→ `n_clusters=5`、「200 棵树」→ `n_estimators=200`）；
     `ml.train` 侧新增 `_MODEL_FAMILIES`，按任务把族名解析成注册模型名，
     该任务下不适用就明确失败，不静默退回默认模型。
  3. `llm_extract` 新增 `tool_schema` 入参，把工具 `input_schema` 的 enum
     与描述写进 prompt（`_slot_help`），模型才知道这里能填什么。
  `ml.train` 的 schema 同步补全：`model` 给出完整 enum（auto + 族名 + 注册模型名），
  `params` 写明各模型真实接受的超参名与示例。
- **「预测 survived」「用随机森林」「做聚类」掉进对话兜底**：`ML_TRAIN` 关键词表里
  只有「预测模型」这种组合词命中，「预测」只是弱关键词（+1，够不到阈值 3）。
  现在「预测」与常见算法名、聚类说法一并算强信号。

### 文档

- `backend/ARCHITECTURE.md`：重写 2.2 模块树与 3.6 Agent Turn，附新旧架构减法对照。
- `README.md`：更新 Agent 链路、意图路由说明、扩展指南与已知限制。

---

## [Unreleased] — 2026-09-23 · 上线前收尾（清垃圾 + 补短板 + 修隐患）

### 新增

- **生产级 HTTP 中间件**（`app/core/middleware.py`）
  - CORS：仅在 `CORS_ALLOW_ORIGINS` 非空时挂载，默认不放开同源策略。
  - GZip：自实现纯 ASGI 版本，**跳过 SSE 与二进制**，只对「已知长度且 ≥1 KB」的
    响应压缩。Starlette 自带版本会把逐 token 推送缓冲成块，因此没有直接用。
  - 限流：固定窗口，只保护昂贵写端点（对话 / 报告生成 / 训练 / 连接器导入），
    GET 不走限流。可通过 `RATE_LIMIT_ENABLED=false` 关闭。
- **配置体检**（`check_runtime_configuration`）：启动时把「能跑但跑不好」的问题
  写进告警日志——`APP_ENV=prod` 缺 `LLM_API_KEY`、生产开 `DEBUG`、CORS 配 `*`、
  生产仍用 SQLite。
- **通知推送通道**（`GET /api/v1/notifications/stream`）：SSE 只在版本号变化时推事件，
  替代前端「每 15 秒拉一次完整列表」。前端 `lib/notificationFeed.ts` 实现
  SSE 优先 + 指数退避轮询兜底（15s→30s→60s→120s），后台标签页不发请求。
- **上下文 token 预算**（`app/agent/context/tokens.py`）：在字符预算之外增加 token 上限
  （`AGENT_CONTEXT_MAX_TOKENS`，默认 6000）。中文 1 字≈1 token、英文 4 字符≈1 token，
  只按字符卡会让中文场景悄悄吃满 LLM 输入窗口；设为 0 可退回旧行为。
- **报告列表元数据层**（`app/reports/saved.py`）：HTTP 接口与 Agent 工具的写路径
  统一走它，保存时落 `.meta.json` 轻量副本。
- 文档：`SECURITY.md`、`CONTRIBUTING.md`、`CHANGELOG.md`、`LICENSE`（MIT）。
- 前端单测：`frontend/tests/toolLabel.test.ts`（Node 自带 test runner，零新增依赖）。

### 修复

- **`agent.clarify` 未登记风险等级**：`DEFAULT_TOOL_RISKS` 缺条目，
  PermissionManager 无法确定是否需要人工确认。新增覆盖测试后当场暴露。
- **`except ... as exc` 闭包延迟引用导致的 `NameError`**
  （`app/api/v1/agent.py` SSE worker）：失败原因被写进 lambda 的 f-string，
  而 except 块结束时 Python 已 `del exc`，于是「报告失败原因」这行代码自己抛错、
  前端只能看到连接中断。改为在块内立即取值。
- **`app/ml_engine/metric_advisor.py` 引用未导入的 `Finding`**（ruff F821）：
  因模块启用了 `from __future__ import annotations` 而未在运行时炸出，属于潜伏缺陷。
- **Agent 工具生成的报告没有元数据副本** ⇒ 列表接口退回「读整篇正文」的慢路径；
  现在两条写路径共用 `app.reports.saved.save_report`。

### 优化

- `GET /api/v1/reports/saved`：由「读全部正文 + JSON 解析」改为
  **只读元数据副本 + 目录签名缓存**，并去掉重复的第二次目录扫描与逐个 `stat`。
  100 份报告（正文各约 10 KB SVG）实测：328.9ms → 首次 <200ms，命中缓存后 <1ms。
- SQLite 加 `PRAGMA journal_mode=WAL / synchronous=NORMAL / busy_timeout=5000 / foreign_keys=ON`，
  避免写事务把读请求锁死（`database is locked`）。
- `ToolRegistry` 的类目关键词表移到 `app/core/config.py::TOOL_CATEGORY_HINTS`，
  并补英文说法；英文提问（如 `filter rows`）现在能正确召回 `data.filter`。
- 前端工具目录缓存加 **5 分钟 TTL** 与 `force` 手动刷新通道，避免后端重启后
  一直拿着旧清单；拉取失败不再清空已有清单。
- `toolDisplayName` 首句截断加 30 字上限（超出补 `…`），并对英文句号做安全的首句切分。

### 文档

- README：补齐目录结构（`connectors` / `learning` / `notifications` / `settings` /
  `quality` / `local_router` / `reports/saved.py` 等），新增「状态与已知限制」章节，
  明确 `local_router` 为**实验性模块、默认关闭**。
- `docker-compose.yml`：访问地址注释修正为 `http://localhost:8080`（与 `8080:80` 映射一致），
  并补充跨域配置项说明。
- `.env.example`：补充 CORS / GZip / 限流 / token 预算 / LOCAL_ROUTER_MODE 等新配置。

## 历史基线

- `9621be3 标准化整理`：完成度整理、文档与目录归并。
- `3d69347 init: 完工`：初始版本合入。
