# 数据库连接器模块设计方案

> 目标读者：后续接手本模块的开发/部署者。
> 定位：扩展中心的「数据库连接器」——把外部数据库的表接进平台，
> 让数据规模不再受「本地文件体积」限制。

---

## 1. 为什么需要它

上一节（[`大数据规模优化与吞吐提升方案.md`](./大数据规模优化与吞吐提升方案.md)）把
**单文件上限**从内存瓶颈里解放出来了，但仍有两条硬边界：

1. **文件形态**：生产数据往往不落地成文件，而是躺在 MySQL / PostgreSQL 的库里。
   让用户先 `mysqldump` 导出再上传，既笨重又容易把磁盘写满。
2. **文件体积**：GB 级 CSV 走「分块入库」虽然内存有界，但仍需先把文件搬上服务器。

连接器的作用是**把数据源从「文件」扩展到「数据库」**：平台直连业务库，
按批抽取、增量落成 Parquet 快照，之后的全部分析链路（EDA / 特征 / 建模 / 报告）
与文件导入完全一致。

---

## 2. 功能范围

### 2.1 支持的能力

| 能力 | 说明 |
| --- | --- |
| 连接管理 | 新建 / 编辑 / 删除 / 列表，凭据加密落库 |
| 试连 | 保存前先试（不落库），保存后也可随时重测 |
| 方言目录 | 前端下拉框数据源，含**驱动是否已安装**的实时状态 |
| 表结构内省 | 列 schema / 主键 / 可空 / 类型 |
| 数据预览 | 抽样若干行，确认「连的是这张表」 |
| 抽取导入 | 整表 / 选列 / `WHERE` / `ORDER BY` → 生成新的数据集版本 |
| Agent 工具 | 4 个工具，让 Agent 能自主查询与导入 |

### 2.2 方言支持矩阵

`app/connectors/dialects.py` 是唯一的方言真相来源（URL 构造、驱动依赖、
标识符引用、内省 SQL、是否支持服务端游标）。

| 方言 | 默认白名单 | 驱动 | 抽取方式 | 说明 |
| --- | --- | --- | --- | --- |
| `sqlite` | ✅ | 内置 | 服务端游标 | 零外部依赖，用于本地/测试 |
| `postgresql` | ✅ | `psycopg[binary]` | 服务端游标 | |
| `mysql` | ✅ | `pymysql` | 服务端游标 | |
| `mssql` | ❌ | 未安装 | LIMIT/OFFSET | 需装 `pyodbc` 并放开白名单 |
| `oracle` | ❌ | 未安装 | LIMIT/OFFSET | 需装 `oracledb` 并放开白名单 |

> 白名单由 `CONNECTOR_ALLOWED_DIALECTS` 控制，默认只放开**已装驱动**的三个。
> 未启用方言在目录里仍然可见（`enabled=false`），前端会灰掉并提示装什么驱动 ——
> 比「下拉框里没有这个选项」更利于排查。
>
> `connector_secret.key` / 驱动缺失都是**运行期可判断**的，因此目录接口不做缓存。

---

## 3. 架构

```
frontend/src/pages/Extensions/
  ├─ index.tsx              扩展中心外壳（连接器面板 + 路线图）
  └─ ConnectorPanel.tsx     连接器面板：列表/新建/试连/选表/预览/导入
        │
frontend/src/api/connectors.ts   11 个 API 封装 + 类型（详情用列表接口刷新，未单独封装 GET /{id}）
        │
        ▼  HTTP /api/v1/connectors
backend/app/api/v1/connectors.py  12 个端点
        │
backend/app/connectors/service.py     ConnectorService（编排 + 状态）
        ├─ crypto.py     凭据加解密（Fernet）
        ├─ dialects.py   方言注册表 + URL 构造 + 内省 SQL
        └─ extract.py    分块抽取引擎（三种策略 + 增量写 Parquet）
        │
backend/app/models/connector.py   DbConnector（ORM）
backend/migrations/versions/0004_db_connectors.py
        │
backend/app/tools/connector_tools.py  4 个 Agent 工具
```

### 3.1 连接器记录

`DbConnector` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `name` / `dialect` | 展示名与方言 key |
| `host` / `port` / `database` / `schema_name` | 连接坐标 |
| `username` / `password_enc` | **口令只存密文** |
| `options_json` | 方言特定的额外参数 |
| `last_status` / `last_error` / `last_checked_at` | 最近一次连接结果（`unknown` / `ok` / `error`） |
| `dataset_id` / `last_import_json` | 最近一次导入目标与统计 |

### 3.2 与数据集的关系

连接器**不新建一套存储**，而是复用数据集版本机制：

```
抽取 ──► 临时 Parquet ──► storage.promote() 原子落位 ──► 新 DatasetVersion
```

因此导入后的数据天然享有：不可变版本、投影/谓词下推、以及全部分析与建模能力。
这是刻意的设计选择 —— 连接器是**数据入口**，不是第二套数据模型。

---

## 4. 抽取引擎（`extract.py`）

### 4.1 三种策略，自动选择

| 策略 | 触发条件 | 适用 |
| --- | --- | --- |
| `server-side-cursor` | 方言支持（PG / MySQL / SQLite） | **首选**。驱动侧按批取回，内存与页深无关 |
| `keyset` | 显式给出 `keyset_column` | 大表且**页深很大**时优于 OFFSET |
| `paged` | 兜底（`LIMIT`/`OFFSET`） | 无游标 + 无 keyset 列 |

为什么 `keyset` 值得单独实现：`LIMIT 50000 OFFSET 4000000` 要求数据库**先扫过**
前 400 万行再丢弃。页越深越慢，而 keyset（`WHERE id > :last ORDER BY id LIMIT n`）
的每页成本恒定。默认不启用，因为需要用户指定一个**唯一且非空**的列。

### 4.2 常量内存与增量落盘

抽取不是「查完再写」，而是**边取边写**：

```
for batch in iter_batches(...):
    frame ──► arrow ──► ParquetWriter.write_table(row_group_size=...)
```

内存 ≈ 批大小 × 行宽，由 `CONNECTOR_BATCH_ROWS`（默认 5 万行）控制。
与入库引擎共用 `INGEST_ROW_GROUP_ROWS`，保证两边的行组粒度一致。

### 4.3 Schema 一致性

各批的 arrow schema 由**第一批固定**；后续批若不一致则显式 `cast` 到首批 schema。
没有这一步，pyarrow 会因 schema 漂移而拒绝写入 —— 而数据库返回类型随数据变化
（如某批某列全为 NULL 导致类型推断不同）是真实会发生的。

### 4.4 已知陷阱（都踩过并修掉）

1. **keyset 首页不能带游标条件。** 首页还没有「上一页最后的值」，
   若生成 `WHERE key > :_keyset_value` 就会**声明了绑定参数却不传值**，
   驱动直接报错。首页必须走无游标分支。
2. **分页路径的异常必须归一。** 底层驱动抛的异常种类繁多，
   不归一成 `ConnectorExtractError` 就会漏成 500，排查时看不到 SQL。
3. **`batch_size` 的下界校验。** 过小的批会导致往返次数爆炸；
   schema 层已加 `ge` 约束。

---

## 5. 安全设计

### 5.1 凭据加密

- 算法：`cryptography` 的 **Fernet**（AES-128-CBC + HMAC-SHA256）。
- 密钥来源：① 配置项 `CONNECTOR_SECRET_KEY`（生产应显式注入，
  避免多实例各自生成不同密钥导致互相解不开）；② 留空则首次启动自动生成
  `{MODEL_ROOT}/connector_secret.key`，权限收紧到 **0600**。
- **口令只以密文形式落库**（`password_enc`），接口回显一律走 `mask_secret()`。
- `crypto_available()` 供上层判断依赖是否就绪 —— `cryptography` 缺失时
  给出明确的可安装提示，而不是抛一个难懂的 ImportError。

> ⚠️ 部署提醒：`connector_secret.key` 与数据库一同备份。
> 丢了它，已保存的连接器口令无法恢复，只能重填。

### 5.2 只读与最小权限

- 抽取只生成 `SELECT`（`build_select` 是唯一的 SQL 构造入口）。
- `where` / `order_by` 是用户输入，**不做拼接**，而是作为 SQL 片段交给
  SQLAlchemy 的 `text()` 并配合参数绑定；标识符（表名/列名）按方言的
  引用规则加引号，防止注入与大小写歧义。
- 平台侧建议：给连接器使用的数据库账号**只授 `SELECT`**。

### 5.3 防呆

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `CONNECTOR_MAX_ROWS` | 0（不限） | 单次抽取行数上限，避免误抽十亿行表把磁盘写满 |
| `CONNECTOR_PREVIEW_ROWS` | 200 | 预览行数上限 |
| `CONNECTOR_MAX_CONNECTORS` | 100 | 连接器数量上限 |
| `CONNECTOR_CONNECT_TIMEOUT_SECONDS` | 10 | 连接超时 |
| `CONNECTOR_STATEMENT_TIMEOUT_SECONDS` | 300 | 语句超时 |
| `CONNECTOR_POOL_SIZE` | 5 | 连接池大小（试连 / 预览等高频短查询走池） |
| `NullPool` | — | **抽取**场景用 `NullPool`（`open_engine(..., pooled=False)` 默认）。抽取是「一次一个短生命周期连接」，池化既不提速，又会让远端超时后留下的坏连接长期占着 |

---

## 6. REST API

前缀 `/api/v1/connectors`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/catalog` | 方言目录 + 驱动安装状态（前端下拉数据源） |
| `GET` | `` | 连接器列表（分页） |
| `POST` | `` | 新建 |
| `POST` | `/test` | **试连（不落库）** |
| `GET` | `/{id}` | 详情 |
| `PATCH` | `/{id}` | 修改 |
| `DELETE` | `/{id}` | 删除 |
| `POST` | `/{id}/test` | 重测已保存的连接器 |
| `GET` | `/{id}/tables` | 列表 / 模式列表 |
| `GET` | `/{id}/tables/{table}/columns` | 表结构内省 |
| `POST` | `/{id}/preview` | 抽样预览 |
| `POST` | `/{id}/import` | 抽取导入为数据集版本 |

**设计约定：连接失败是正常业务结果，不是 HTTP 错误。**
`/test` 系列返回 `{ok: false, message: "..."}` 与 HTTP 200 —— 因为「连不上」
是用户需要看到并据此修改配置的正常反馈，用 4xx/5xx 会让前端把「握手失败」
和「接口挂了」混在一起，也让错误提示难以定制。
`/import` 则相反：失败即异常（`ConnectorExtractError`），因为那是真的执行失败。

---

## 7. Agent 工具

| 工具 | 风险等级 | 作用 |
| --- | --- | --- |
| `connector.list` | LOW | 列出已保存连接器 |
| `connector.tables` | LOW | 列表 / 表结构内省 |
| `connector.preview` | LOW | 抽样预览 |
| `connector.import` | **HIGH** | 抽取导入（写数据） |

风险等级必须登记在 `app/agent/permission/rules.py` 的 `DEFAULT_TOOL_RISKS`。
**漏登记不会在启动时报错，而是在 Agent 真正调用该工具、走到权限裁决时才失败** ——
有一条测试专门钉这个不变量：`test_services_expose_connector_tool_risk_mapping`。

`ToolServices` 新增 `connector_service` 字段，由 `AgentRuntime._services()` 注入。

---

## 8. 部署清单

```bash
# 1) 装依赖（已写入 backend/pyproject.toml）
pip install "cryptography>=42" "psycopg[binary]>=3.1" "pymysql>=1.1"

# 2) 迁移建表
alembic upgrade head        # 0004_db_connectors

# 3) 生产环境显式注入密钥（否则每实例各自生成）
export CONNECTOR_SECRET_KEY="<你的密钥>"

# 4) 若要放开 mssql / oracle
#    装驱动后改配置：CONNECTOR_ALLOWED_DIALECTS=sqlite,duckdb,postgresql,mysql,mssql,oracle
```

**不装 `cryptography` 会怎样**：连接器功能整体不可用（口令无法加密），
`crypto_available()` 返回 False，目录接口与新建接口都会给出「请安装 xxx」的提示。
`sqlite` 例外地不需要额外驱动，因此可在无加密依赖时用于本地验证
（但仍推荐装上，否则口令是明文风险）。

---

## 9. 验证

`backend/tests/test_connectors.py`（含 SQLite 端到端）：

| 用例 | 覆盖点 |
| --- | --- |
| 端到端抽取 → 数据集版本 | 主链路可用、行数/schema 正确 |
| 方言白名单 | 未放开方言被拒，且错误信息可读 |
| keyset 分页 | 首页不带游标、多页不重不漏 |
| 分页异常归一 | 底层异常 → `ConnectorExtractError`，且带 SQL |
| 凭据加密往返 | 密文不含明文、可解回原值、`mask_secret` 不泄漏 |
| 工具风险等级 | 4 个工具均已登记（见 §7） |

**为什么用 SQLite 做端到端**：它是唯一零外部依赖的方言，CI 里必然可用。
PostgreSQL / MySQL 的差异主要落在 URL 构造与游标选项上，这两块由方言注册表的
单元测试覆盖，不必起真实服务。

### 9.1 ⚠️ 测试全绿 ≠ 生产可用：必须单独确认迁移已执行

`tests/conftest.py` 每个用例都会新建**内存库**并 `Base.metadata.create_all()`，
所以 `db_connectors` 表总是「存在」的 —— 这意味着**测试永远发现不了
「生产库没跑迁移」**。实测就踩到过：测试全绿，但 `backend/data/xiaoluo.db`
的 `alembic_version` 仍停在 `0003_custom_cards`，连接器接口一调就是
`no such table: db_connectors`。

部署/接手时按这两步自证：

```bash
# ① 迁移到位了吗
./.venv/Scripts/python.exe -m alembic upgrade head
# ② 生产库里真的有这张表吗（别只看迁移日志）
./.venv/Scripts/python.exe -c "
import sqlite3; c=sqlite3.connect('data/xiaoluo.db')
print([r[0] for r in c.execute('select * from alembic_version')])
print([r[0] for r in c.execute(\"select name from sqlite_master where name='db_connectors'\")])"
```

**隔离式实时冒烟**（推荐做法）：把 `DATABASE_URL` / `DATA_ROOT` / `MODEL_ROOT`
指到临时目录，跑一个 `TestClient` 脚本走完
`catalog → create → test → tables → columns → preview → import → delete`，
最后直接把导入产物 Parquet 读回来和源数据逐值比对。这样既验证了**真实 HTTP 栈
+ 真实存储链路**，又不会往生产库/生产数据集里留东西。

实测结论（SQLite 源表，含中文、逗号、内嵌引号、NULL）：

| 环节 | 结果 |
| --- | --- |
| `catalog` | sqlite / postgresql / mysql 驱动就绪；mssql / oracle 未装 |
| `test` | `ok=true`，`server_version=3.42.0` |
| `tables` / `columns` | 正确列出，主键识别正确 |
| `preview` | `WHERE` 下推进 SQL，中文与 NULL 正确 |
| `import` | `strategy=keyset`，`row_count=5`，写盘 `datasets/1/v000001.parquet` |
| 回读比对 | `matches_source=true`（中文、`折扣,促销`、`含"引号"`、NULL 全部保真） |
| `delete` | 200，列表清空，**数据集保留**（符合「删配置不动数据」的约定） |

---

## 10. 已知限制与后续方向

1. **无增量同步。** 目前每次导入都是全量抽取，生成新版本。
   后续可基于 `keyset_column` + 水位线做增量追加（连接器表已有 `last_import_json`
   可存水位）。
2. **无 SQL 查询模式。** 现在只能选表 + `WHERE`。开放自定义 SQL 会带来注入面，
   需要单独设计（只读校验 + 权限隔离 + 超时）。
3. **`mssql` / `oracle` 未做真机验证。** 方言注册表已就位，但内省 SQL 与
   分页语法没有在真实实例上跑过 —— 放开前必须补验证。
4. **抽取任务不异步。** 大表导入会长时间占用请求。
   应改为后台任务 + 进度上报（`CONNECTOR_STATEMENT_TIMEOUT_SECONDS` 现在是兜底）。
5. **无并发抽取限流。** 多个大表同时导入会争抢数据库连接与本地磁盘 I/O。
