# 变更日志

本文件记录对外可见的重要变更。格式参考 Keep a Changelog：
`新增 / 修复 / 优化 / 文档 / 破坏性变更`。日期为合并日期。

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
