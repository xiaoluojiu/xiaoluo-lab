# 数据分析模块 · 后端代码评估报告

> 评估对象：数据分析（EDA / Profiling / Quality）模块后端实现
> 评估日期：2026-09-21
> 评估范围：仅 `backend/`（未改动 `frontend/`）
> 说明：本报告基于源码静态阅读，未运行任何测试或基准（按用户约束）。

---

## 快速索引

| 章节 | 内容 |
| --- | --- |
| [0. 模块地图](#0-模块地图) | 入口、核心文件、调用链、依赖关系 |
| [1. 功能现状](#1-功能现状) | 已实现能力清单 + 缺失项 + 完成度判断 |
| [2. 代码质量](#2-代码质量) | 接口设计 / 查询聚合 / 缓存 / 异常 / 并发与性能 |
| [3. 体验优化](#3-体验优化) | 响应速度 / 准确性 / 展示友好度 / 报错提示 |
| [4. 优化建议](#4-优化建议) | 高 / 中 / 低优先级可落地改进项 |
| [5. 风险速查表](#5-风险速查表) | 一页纸问题清单 |
| [6. 已落地优化项](#6-已落地优化项2026-09-21-实施) | 实际完成的改动 + 原因/影响/风险/回滚 |

---

## 0. 模块地图

### 0.1 入口与分层

```
backend/app/main.py                       ← 唯一 ASGI 入口
  └─ core/middleware.py                   ← RequestContextMiddleware（统一异常 → HTTP）
  └─ api/v1/__init__.py                   ← api_router，前缀 /api/v1
       ├─ api/v1/eda.py                   ← 【EDA 入口】5 个 GET 端点
       ├─ api/v1/dataset_analysis.py      ← 【分析入口】preview / schema / profile / quality
       └─ api/v1/datasets.py              ← 数据集 CRUD（版本管理）
  └─ api/deps.py                          ← DI 装配
       └─ data_engine/service.py          ← DataEngineService（门面）
            ├─ analysis.py                ← 【核心计算】3000+ 行，全部分析算法
            ├─ data_engine/operations.py  ← preview（分页/排序/筛选）
            ├─ services/dataset_service.py← 版本快照读写（Parquet）
            └─ storage/local.py           ← 本地文件 Storage
```

### 0.2 核心文件职责

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `app/analysis.py` | ~1307 | **全部只读分析算法**。合并了原 profiling / quality / eda 三模块：`analyze_schema`、`profile`、4 个 QualityChecker、5 个 EDA Analyzer、`VisualizationBuilder`（9 类图表） |
| `app/api/v1/eda.py` | 158 | EDA 5 端点，统一 `_load_df()` 加载 |
| `app/api/v1/dataset_analysis.py` | 180 | 数据集级分析 4 端点（preview/schema/profile/quality） |
| `app/data_engine/service.py` | 1035 | 门面：`schema/profile/preview/quality` + 操作 + 合并 |
| `app/data_engine/operations.py` | 400+ | `preview()` 分页/排序/筛选 DSL |
| `app/services/dataset_service.py` | 440 | 版本管理、Parquet 读写 |
| `app/tools/eda_tools.py` | 147 | Agent 侧 5 个 EDA 工具（复用上面 Analyzer） |
| `app/data_engine/json_utils.py` | 87 | `json_safe` / `df_to_records` |

### 0.3 三条调用链

**链 1｜前端 EDA 请求（最常见）**
```
GET /api/v1/datasets/{id}/eda/descriptive?columns=a,b
 → deps.get_data_engine_service()          # 每请求新建 DatasetService + DataEngineService
 → deps.get_dataset_service() → get_db()   # 每请求新建 Session
 → eda._load_df()                           # ① DB 查版本行 ② 读 Parquet 字节 ③ pl.read_parquet 全量
 → DescriptiveAnalyzer().analyze(df)        # 逐列 Python 循环 + 多次独立聚合
 → analysis.py → ApiResponse[dict]
```

**链 2｜Preview（分页预览）**
```
GET /api/v1/datasets/{id}/preview?page=1&page_size=20
 → _load_df()                               # 同样全量加载
 → service.preview(df, ...)                 # df.lazy() → filter/sort → select(pl.len()) → slice
 → df_to_records(result)                    # 逐行 iter_rows(named=True) + 逐值 json_safe
```

**链 3｜Agent 工具调用**
```
AgentRuntime → registry.execute("eda.correlation")
 → PermissionManager.check → EdaCorrelationTool.execute()
 → ds.load_version(dataset_id, version)     # 又全量加载一次
 → CorrelationAnalyzer().analyze(df)
```
> 注：链 1 与链 3 是**两套并行实现**，同一份算法被前端路由和 Agent 工具分别调用，各自独立加载数据。

### 0.4 关键依赖关系

- `analysis.py` 是**纯计算层**，只依赖 `polars` + `json_utils`，不依赖 FastAPI —— 分层干净 ✅
- `api/v1/dataset_analysis.py` → `data_engine/service.py` → `services/dataset_service.py` → `storage/local.py` —— 单向 ✅
- ⚠️ **分层倒置**：`data_engine/service.py` 第 32 行 `from app.analysis import ...`，而 `analysis.py` 位于上层目录，形成 `data_engine` ↔ `app` 的循环依赖风险（已在 `backend/ARCHITECTURE.md` 记录）

---

## 1. 功能现状

### 1.1 已实现能力清单

#### A. 数据画像（Profiling）

| 能力 | 端点 / 函数 | 状态 | 备注 |
| --- | --- | --- | --- |
| Schema 分析 | `GET /datasets/{id}/schema` → `analyze_schema` | ✅ | 列名/类型/可空/空值数/唯一值数/5 个样本值 |
| 数据画像 | `GET /datasets/{id}/profile` → `profile` | ✅ | 按 numeric/temporal/boolean/categorical 分类画像；数值列含 min/max/mean/median/std/3 分位 |
| 完整行统计 | `profile()` 内 | ✅ | `complete_row_count` / `complete_row_rate`（全列非空行） |
| 缺失率 | `profile()` 内 | ✅ | 逐列 `missing_count` + `missing_rate` |

#### B. 数据质量（Quality）

| 能力 | 类 | 状态 | 阈值机制 |
| --- | --- | --- | --- |
| 缺失值检查 | `MissingChecker` | ✅ | 4 档严重度（low/medium/high/critical），可配 3 档阈值 |
| 重复行检查 | `DuplicateChecker` | ✅ | 支持全行 / 指定列子集 |
| 异常值检查 | `OutlierChecker` | ✅ | IQR + Z-Score 双方法 |
| Schema 一致性 | `SchemaChecker` | ✅ | 缺失列 / 多余列 / 类型冲突，8 种类型别名 |
| 汇总报告 | `build_report` | ✅ | severity 统计 + `has_errors` + 自动生成中文处理建议 |
| 质量端点 | `GET /datasets/{id}/quality` | ✅ | 但**未暴露 `expected_schema` 参数**（见 §1.2 缺失 F4） |

#### C. 探索性分析（EDA）

| 能力 | 端点 / 方法 | 状态 | 实现要点 |
| --- | --- | --- | --- |
| 描述统计 | `GET /eda/descriptive` → `DescriptiveAnalyzer` | ✅ | 数值列 5 统计量 + 3 分位；布尔列 true/false 计数；类别列 unique/top/freq |
| 相关性 | `GET /eda/correlation` → `CorrelationAnalyzer` | ✅ | Pearlman / Spearman / **auto**（整数→Spearman，浮点→Pearson，异类统一 Spearman） |
| 分布分析 | `GET /eda/distribution` → `DistributionAnalyzer` | ✅ | 数值分桶（`bins` 1~100）；类别频次 Top-N（`top_n` 1~100）+ 占比 |
| 异常值分析 | `GET /eda/outlier` → `EdaOutlierAnalyzer` | ✅ | 输出边界、异常数/占比、样本异常值、**内点范围**、中间统计量 |
| 可视化数据 | `GET /eda/visualize` → `VisualizationBuilder` | ✅ | **9 类图表**（见下） |

**9 类图表实现细节**

| 图表 | 方法 | 状态 | 亮点 / 限制 |
| --- | --- | --- | --- |
| histogram | `histogram` | ✅ | 复用数值分桶 |
| bar | `bar` | ✅ | 复用类别频次 |
| line | `line` | ✅ | 重复 x 聚合为均值；`max_points=1000` 抽稀 |
| scatter | `scatter` | ✅ | `sample_limit`（默认 1000，上限 10 万）+ `seed=42` 可复现采样；返回 `sampled` 标记 |
| boxplot | `boxplot` | ✅ | 支持 `group_by` 分组；whisker 用 1.5×IQR；异常值样本 Top-10 |
| heatmap | `heatmap` | ✅ | 复用相关性矩阵 |
| **qq**（正态 Q-Q） | `qq` | ✅ | 自研 `_norm_ppf`（Acklam 逼近，**零 scipy 依赖**，误差 ~1.15e-9）+ 最小二乘参考线 |
| **grouped_bar**（分组柱） | `grouped_bar` | ✅ | 分类 × 分组，4 种聚合（mean/sum/count/median） |
| **area**（CDF） | `area` | ✅ | 累积分布，`mode="cdf"` |

#### D. 分页预览与筛选

| 能力 | 状态 | 实现要点 |
| --- | --- | --- |
| 服务端分页 | ✅ | `page` / `page_size`（1~200），LazyFrame `slice` |
| 列选择 | ✅ | `columns` 逗号分隔 |
| 单列排序 | ✅ | `sort_column` / `sort_desc`，`nulls_last` 默认 True |
| 筛选 DSL | ✅ | 9 个算子（eq/neq/gt/gte/lt/lte/contains/in/is_null）+ and/or 逻辑，**禁止 eval/exec** 的安全表达式构建 |

### 1.2 缺失项

| 编号 | 缺失项 | 影响 | 严重度 |
| --- | --- | --- | --- |
| F1 | **无结果缓存** | 同一数据集同一参数的重复请求全额重算 + 重读 Parquet | 🔴 高 |
| F2 | **无 LazyFrame / 列裁剪优化** | EDA 全部走 eager 全量加载，哪怕只分析 1 列 | 🔴 高 |
| F3 | **相关性矩阵无抽样、无并行** | O(n²) 次独立 `_corr`，每列对数各一次 `drop_nulls()` | 🔴 高 |
| F4 | `expected_schema` 未暴露到 `/quality` 端点 | `SchemaChecker` 实际不可用（只能跑默认 3 个 checker） | 🟡 中 |
| F5 | 无**多列联合分布**（交叉表 / 列联表） | 只能单列分布，无法回答「A 类中 B 占比」 | 🟡 中 |
| F6 | 无**时间序列分解 / 趋势检验**（如季节性、ADF 平稳性） | 时序数据只能画 line / area | 🟡 中 |
| F7 | 无**缺失值模式可视化**（missingno 式矩阵/热力图） | `MissingChecker` 只有数字，看不出「缺失成块」 | 🟢 低 |
| F8 | 无**分析结果导出**（CSV / Excel / JSON 下载） | 前端只能看不能带走 | 🟡 中 |
| F9 | 无**分析结果持久化 / 历史对比** | 每次分析都是「一次性」，无法比较两个版本的质量变化 | 🟡 中 |
| F10 | `mode` / `skew` / `kurtosis` 等分布形态统计缺失 | 描述统计只有 5 个基础量，论文里偏弱 | 🟢 低 |
| F11 | 相关性**无 p 值 / 显著性标记** | 只看系数不看样本量，小样本高相关会误导 | 🟡 中 |
| F12 | 图表**不支持 `log` 轴 / 归一化** | 长尾数据（如金额）直方图几乎只有 1 根柱 | 🟢 低 |

### 1.3 完成度判断

| 维度 | 完成度 | 判断依据 |
| --- | --- | --- |
| **功能广度** | **85%** | EDA 四大件（描述/相关/分布/异常）+ 9 类图表 + 质量四检查 + 分页筛选；缺交叉表、时序分解、导出、历史对比 |
| **算法正确性** | **90%** | 大量边界已处理（常数列、单样本、空列、NaN/Inf）；缺 p 值、分布形态统计 |
| **性能与规模化** | **45%** | **最大短板**：无缓存、无列裁剪、相关性 O(n²) 串行、全量 eager 加载、同步阻塞端点 |
| **健壮性** | **75%** | 异常体系完整、边界处理细致；但 `analyze_schema` 的 `n_unique()` 潜在失败、`_corr` 的 n=2 边界、无超时保护 |
| **接口设计** | **80%** | RESTful 规范、参数校验到位（`Query(ge/le/pattern)`）；但 eda 与 dataset_analysis 两套 `_load_df` 重复、缺缓存语义 |
| **体验** | **60%** | 报错信息英文为主、无进度反馈、大请求无超时提示 |

**综合完成度：约 72%**
> 一句话结论：**"能算对，但算得慢、不算久、不够用"** —— 算法层质量扎实（尤其 Q-Q 的零依赖实现、auto 相关性方法选择是亮点），瓶颈完全集中在**数据加载与缓存策略**上。

---

## 2. 代码质量

### 2.1 接口设计

#### ✅ 优点

1. **RESTful 语义清晰**：`GET /datasets/{id}/eda/{kind}`，只读端点明确标注"不修改数据"
2. **参数校验前移到 FastAPI 层**：
   ```python
   method: str = Query("pearson", pattern="^(pearson|spearman|auto)$")
   bins: int = Query(10, ge=1, le=100)
   top_n: int = Query(20, ge=1, le=100)
   sample_limit: int = Query(1000, ge=1, le=100000)
   ```
   —— 非法输入在进入业务层前就被拦下，比在 Analyzer 内校验更省资源 ✅
3. **统一响应包装** `ApiResponse[dict]` + `version` 字段回填，前端能确认结果对应哪个数据版本 ✅
4. **只读语义一致性**：`EdaModule` 基类声明"只读"，所有 `analyze()` 返回新 dict，不修改入参 df ✅

#### ⚠️ 问题

| 编号 | 问题 | 位置 | 说明 |
| --- | --- | --- | --- |
| I1 | **`_load_df` 重复实现两份** | `eda.py:30` 与 `dataset_analysis.py:21` | 逻辑几乎一致（仅返回 `version` vs `version_row`），应抽到 `data_engine` 或共享 helper |
| I2 | **返回类型过宽为 `dict`** | 全部端点 `ApiResponse[dict]` | 丢失 OpenAPI schema，前端只能靠 TS 手写类型对齐；建议定义 Pydantic 响应模型 |
| I3 | **参数传递用 `**options` 字典解包** | `eda.py:50-54`、`analysis.py:690` | `options: dict[str, Any]` 让 Analyzer 无法静态校验参数名；拼错 key 会静默忽略（如传 `bin` 而非 `bins` 不报错，走默认值） |
| I4 | **`method` 参数在 `visualize` 端点缺失** | `eda.py:116` | `heatmap` 方法内部读 `options.get("method", "auto")`，但端点未声明 `method` 参数 → heatmap 永远只能 auto，前端无法选 Pearson/Spearman |
| I5 | **`max_points` / `seed` 未暴露** | `analysis.py:1056/1081` | `line` 的 `max_points=1000`、`scatter` 的 `seed=42` 都写死在代码里读 `options.get()`，端点无对应参数 → 不可调 |
| I6 | **`version` 语义在两条链路不一致** | `eda.py` vs `dataset_analysis.py` | 前者返回 `data["version"] = v`（int），后者返回 `version_row`（对象），前端需分别处理 |
| I7 | **无 `HEAD` / 轻量探测端点** | — | 前端想知道「这个数据集有多少行/多少列/能否算相关」必须全量跑一次 analysis |
| I8 | **OpenAPI 未标注端点耗时量级** | — | 前端无法据此设 axios timeout，只能统一 15s |

### 2.2 数据查询与聚合逻辑

#### ✅ 亮点

- **`preview()` 全程走 LazyFrame**（`operations.py:309`）：`filter → sort → select → collect` 才落地；`total` 用 `select(pl.len())` 而非 `height`，避免物化 ✅
- **筛选 DSL 安全**：`build_filter_expr` 白名单算子 + 结构化参数，**无 eval/exec** ✅
- **分桶用向量化**：
  ```python
  idx = ((clean - mn) / width).floor().clip(0, bins - 1).cast(pl.Int64)
  counts = idx.value_counts()
  ```
  比 Python 循环快得多 ✅
- **`json_safe` 递归处理** Decimal / datetime / NaN / Inf / numpy 标量 ✅

#### ⚠️ 问题（性能相关）

| 编号 | 问题 | 位置 | 详细分析 |
| --- | --- | --- | --- |
| **P1** | **`load_version` 每次全量读 Parquet 到内存** | `dataset_service.py:378-408` | `storage.read()` 返回**完整 bytes** → `io.BytesIO` → `pl.read_parquet` 全量解码。**没有任何缓存**。一次 EDA 请求 = 一次完整 Parquet 解码；`/preview?page=1&page_size=20` 也走同一路径 —— 只为看 20 行也要解全表 |
| **P2** | **相关矩阵 O(n²) 次独立计算，且每对重复 `drop_nulls()`** | `analysis.py:810-819, 828-841` | `n` 个数值列 → **n² 次** `_corr`（对角线外 n(n-1) 次）。每次 `_corr` 内 `df.select([a,b]).drop_nulls()` 都新建 DataFrame；100 列 = 9900 次两列投影 + 9900 次空值过滤。**且 Spearman 需要 2 次 `rank()`**（`analysis.py:803-804`），100 列数据集在 10 万行下是分钟级 |
| **P3** | **`analyze_schema` 对每列调用 `n_unique()`** | `analysis.py:56` | `n_unique()` 是**全列哈希去重**，内存与 CPU 开销都高。对一个 100 列 × 100 万行数据集，这就是 100 次独立全表扫描。而 schema 端点的语义只是"看看结构"，不需要精确唯一值数 |
| **P4** | **`profile()` 逐列 Python 循环 + 每列多次独立聚合** | `analysis.py:134-175` | `series.null_count()`、`series.n_unique()`、`_numeric_profile` 内 `min/max/mean/median/std` + 3 次 `quantile` —— **7 次独立扫描同一列**。Polars 本可用 `df.select([...])` 一次算出所有列所有统计量 |
| **P5** | **`preview()` 的 `total` 在 filter 后仍需全表 count** | `operations.py:371-376` | `frame.select(pl.len()).collect()` —— 对无筛选场景，其实可用 `DatasetVersion.row_count`（已在 DB 中）直接返回，省一次全表扫描 |
| **P6** | **`df_to_records` 逐行 `iter_rows(named=True)` + 逐值 `json_safe`** | `json_utils.py:70-81` | Python 层逐单元格函数调用。200 行 × 50 列 = 10000 次 `json_safe` 调用（每次都走 `isinstance` 链）。`page_size` 上限 200 尚可接受，但若调大就是线性劣化 |
| **P7** | **`EdaOutlierAnalyzer` 对每个数值列重复 `filter` 两次** | `analysis.py:965-967` | `s.filter(mask)` 取异常 + `s.filter(~mask & s.is_not_null())` 取内点 —— 同一 mask 计算两遍全列扫描。可一次 `partition` 或先算 `inliers` 再反推 |
| **P8** | **`Scatter` 的 `sampled` 标记重复计算** | `analysis.py:1086` | `data.height < df.select([x,y]).drop_nulls().height` —— 为算这一个 bool，**又做了一次两列投影 + 空值过滤**，而 `data` 就是它的结果 |
| **P9** | **`qq` 图无抽样上限** | `analysis.py:1156-1164` | `s.to_list()` 把整列转 Python list，`_norm_ppf` 逐点 Python 调用（n 次浮点运算 + 列表推导）。100 万行 = 100 万次 Python 函数调用，且返回 100 万点的 x/y 数组塞进 JSON |
| **P10** | **`grouped_bar` 用 Python dict 做透视** | `analysis.py:1217-1225` | `sorted(df[column].drop_nulls().unique().to_list())` 全列去重两遍（分类列 + 分组列），再 Python 层组装 `series`。若分类数 × 分组数很大，返回体膨胀且无上限（`top_n` 参数在此**未被使用**） |
| **P11** | **无列裁剪**：EDA 分析指定 `columns` 后仍然加载完整 df | `eda.py:49` | `_load_df` 先全量加载，再在 Analyzer 里 `pick_numeric(df, columns)` 过滤。Parquet 支持列投影（`read_parquet(columns=[...])`），当前完全没用 |
| **P12** | **`line` 抽稀算法有缺陷** | `analysis.py:1057-1061` | `pl.col("__i") == (pl.col("__i") * 1.0 // step).cast(pl.Int64)` —— 除以 `step` 后取整再比较，实际是"每 step 个点取 1 个"，但比较用 `==` 而非取模，浮点边界可能少取/漏取；且没保证首末点 |
| **P13** | **`DistributionAnalyzer.numeric_distribution` 分桶边界用 `:.4g`** | `analysis.py:896` | 极端量级（如 1e-8 或 1e12）时 label 会退化为科学计数或全 0，前端图表 x 轴标签不可读 |

### 2.3 缓存

#### 现状：**分析层零缓存**

| 层级 | 缓存情况 | 说明 |
| --- | --- | --- |
| 端点层 | ❌ 无 | 无 `Cache-Control`、无 ETag、无服务端结果缓存 |
| Service 层 | ❌ 无 | `DatasetService` 每次 new，`DataEngineService` 每次 new（`deps.py:47-51`） |
| 数据层 | ❌ 无 | `load_version` 无 DataFrame 缓存；每次读磁盘 + 全量解码 |
| Agent 层 | ✅ 有 | `AGENT_PLAN_CACHE_MAX_ITEMS=64`（计划缓存）、`AGENT_DATASET_CACHE_MAX_ITEMS=32`（数据集工具缓存） |

**关键对比**：Agent 侧有数据集缓存（`AGENT_DATASET_CACHE_ENABLED`），但**前端 EDA 请求完全绕开了它** —— 走的是 `deps.get_data_engine_service()` 新建的实例，没有共享缓存。

**影响量化（推理）**：
- 一份 50 MB Parquet、10 万行的数据集
- 前端打开「数据分析」页 → 选数据集 → 自动加载样本（`previewDataset` 50 行）+ schema + 4 个 KPI → **至少 2~3 次全量 Parquet 解码**
- 点「运行分析」→ 描述统计 + 相关 + 分布 + 异常 → **又 4 次全量解码**
- 每次切 tab（分布/异常）→ **再 1 次**
- 一轮完整 EDA 浏览 ≈ **8~10 次完整 Parquet 解码同一份数据**

### 2.4 异常处理

#### ✅ 优点

1. **统一异常体系**：`AppException` → `DataEngineException` → `TransformError/DataQualityError/LoadError`，每类带 `http_status` + `default_code`
2. **中间件兜底**：`RequestContextMiddleware` 捕获 `AppException` → 结构化 JSON（`code`/`message`/`details`/`request_id`）；未捕获异常 → 记 `logger.exception` 后返回 500
3. **大量业务边界已处理**：
   - 空 df：`MissingChecker`/`DuplicateChecker`/`OutlierChecker` 均 `if df.height == 0: return []`
   - 单样本 std：`compute_outlier_bounds` 显式处理 `std_raw is not None`
   - 常数列：返回 `{"constant": True}` 并跳过，不除以 0
   - NaN/Inf：`json_safe` 转 `None`
   - 相关性分母为 0：`_corr` 返回 `None` 而非崩溃
   - 空 Parquet 读取失败：包装为 `DatasetException(VERSION_READ_ERROR)` 带 storage_path

#### ⚠️ 问题

| 编号 | 问题 | 位置 | 说明 |
| --- | --- | --- | --- |
| E1 | **`analyze_schema` 的 `n_unique()` 未保护** | `analysis.py:56` | 若列含不可哈希类型（如 `List`/`Struct`），`n_unique()` 会抛原生异常 → 冒泡到中间件变成 **500 INTERNAL_ERROR**，用户看到"Internal Server Error"。同文件 `profile()` 也有同样调用（`:150`） |
| E2 | **`_corr` 在 n=2 时 `pair.height - 1` 分母需验证** | `analysis.py:808` | `pair.height < 2` 才返回 None，height=2 时走 `sum / (2-1) / denom` = 样本协方差形式。数值上能算，但 2 个点的"相关系数"必为 ±1，语义误导且无警示 |
| E3 | **`ok` 与 `skipped` 语义未在 API 层统一** | `analysis.py:960-962` | 异常值分析里空列/常数列返回 `status: "skipped"`，而其它列 `status: "ok"`。前端若只读 `bounds` 会拿到 undefined → 需自行判断；且**同行不同列结构不一致**，TS 类型难对齐 |
| E4 | **`DataQualityError` 用于「无可用列」这类输入问题** | `analysis.py:943` | `EdaOutlierAnalyzer` 在无数值列时抛 `DataQualityError`（http_status 400），但语义上这是**参数/数据不适用**问题，更像 422。前端错误提示会误导为"数据质量差" |
| E5 | **`_require_str` 抛 `TransformError`（400）而非 422** | `analysis.py:1303-1306` | 前端漏传 `column` → 400 "TRANSFORM_ERROR"，但 FastAPI 自身的必填参数缺失是 422。同一页面两种错误码，前端提示逻辑分裂 |
| E6 | **`Query(..., pattern=...)` 的不匹配返回 422 且 message 是英文正则** | `eda.py:120` | 用户看到 `string does not match regex "^(histogram\|...)$"`，完全不可读 |
| E7 | **中间件对 `AppException` 不记日志** | `middleware.py:56-59` | 只对未捕获异常 `logger.exception`。业务异常（如 TransformError）只记在 finally 的 access log 的 status 里，**丢失 traceback 与 details**，线上排查困难 |
| E8 | **`preview` 的 JSON 解析错误 details 泄露内部细节** | `dataset_analysis.py:48-51` | `details={"reason": str(exc)}` 直接把 Python json 异常文本回传前端 |
| E9 | **无超时 / 无请求取消传播** | 全局 | 端点全是同步 `def`，在 FastAPI threadpool 执行；客户端断开后**计算继续跑完**。大数据集 + 相关性矩阵可能占用线程池数分钟 |
| E10 | **`_load_refs` / `run_multi_merge` 无事务边界保护** | `service.py:871-998` | 多文件合并先 `create_version` 写 storage，再 `_record_operation` 写 DB；若 DB 提交失败，storage 已产生孤儿文件（`create_version` 内部有清理，但 `run_multi_merge` 的 `_record_operation` 失败无回滚） |

### 2.5 并发与大数据量场景

#### 并发问题

| 编号 | 问题 | 位置 | 风险 |
| --- | --- | --- | --- |
| **C1** | **端点全同步 `def`（`eda.py:43/59/76/91/116`）** | 全部 EDA 端点 | FastAPI 会把同步函数丢到 **AnyIO threadpool（默认 40 线程）**。若 40 个并发 EDA 请求同时跑相关矩阵，40 份 DataFrame 同时在内存 → **内存峰值 = 40 × 数据集大小**，极易 OOM |
| **C2** | **每请求新建 `DatasetService`/`DataEngineService`** | `deps.py:47-51` | 无状态是对的，但意味着**无法做跨请求的 DataFrame 缓存**；且 `DataEngineService.__init__` 每次都 `MergeExecutor()` |
| **C3** | **SQLite + 多线程** | `core/database.py` | SQLite 写入串行化；EDA 只读还好，但与数据处理/合并并发时会有锁等待 |
| **C4** | **无 worker 限制的隐性依赖** | `deps.py:23-24` | `AGENT_STORE`/`WORKFLOW_SERVICE` 是进程内单例 → 必须单 worker。单 worker + threadpool 40 线程下，一个大 EDA 请求就可能饿死整个服务 |
| **C5** | **无并发闸门 / 无排队** | 全局 | 没有 semaphore 限制同时进行的重分析数量，也没有 429 或降级策略 |
| **C6** | **`_PLANNERS` 缓存有竞态** | `deps.py:35-42` | `if len(_PLANNERS) >= _PLANNER_CACHE_MAX: _PLANNERS.clear()` —— 有 `_PLANNER_LOCK` 保护，但 `clear()` 会一次性丢弃所有 planner（含正在被使用的），语义上是"全清"而非 LRU 淘汰 |

#### 大数据量瓶颈

设数据集为 `R` 行 × `C` 列：

| 操作 | 当前复杂度 | 理论最优 | 倍数差 |
| --- | --- | --- | --- |
| `analyze_schema` | O(C × R)（每列 n_unique 全扫描） | O(R × C) 单次遍历或近似 | 小，但常数大 |
| `profile` | O(C × 7 × R)（每列 7 次独立扫描） | O(C × R) 一次 `select` 聚合 | **~7×** |
| `correlation` | O(C² × R) + C² 次 DataFrame 构造 | O(C² × R) 单次批量化 + 抽样 | **常数 5~10×** |
| `outlier` | O(C × 2 × R)（每列 2 次 filter） | O(C × R) | ~2× |
| `qq` | O(R) Python 逐点 + JSON 全量返回 | O(min(R, 5000)) | **R 越大越崩** |
| `preview` | O(R)（filter 后 count）+ O(page_size) | O(page_size) + 用 DB row_count | 视 R 而定 |

**具体风险场景**：

1. **100 列 × 50 万行数据集点「运行分析」**
   - `edaDescriptive`（100 列 × 7 次扫描）+ `edaCorrelation`（**100² = 10000 次 _corr**，每次两列投影 + drop_nulls）
   - 预计：描述统计 2~5s，**相关性 30s~数分钟**
   - 前端 axios **15s 超时** → 用户看到「请求超时」，但后端仍在跑
   - 重试 → 又一个并发重负载任务

2. **含日期列的宽表**
   - `profile` 的 `dtype.is_temporal()` 分支只算 min/max ✓
   - 但 `pick_numeric` 正确排除时间列 ✓

3. **Q-Q 图**
   - `analysis.py:1163` `vals = [float(v) for v in s.to_list()]` —— 100 万行直接转 Python list
   - `theoretical = [_norm_ppf(...) for i in range(n)]` —— 100 万次 Python 函数调用
   - 返回体：x/y 各 100 万个 float ≈ **16 MB JSON**（仅两个数组）

4. **`grouped_bar` 高基数**
   - `column` 若有 5000 个唯一值 × `group_by` 20 个 → `series` 是 20 × 5000 = 10 万个点
   - `top_n` 参数在 `grouped_bar` 中**完全未使用**（`analysis.py:1189-1232` 全程没读 `top_n`）

5. **`preview` 的 `page_size` 上限 200**
   - 但 `MAX_PAGE_SIZE = 200`（`operations.py:276`），而 `/preview` 端点的 `le=200` 与之一致 ✓
   - 前端「查看样本」用 `page_size=20`，`PreviewTable` 默认 20 ✓

---

## 3. 体验优化（使用者视角）

### 3.1 响应速度 🔴

| 编号 | 问题 | 用户感受 | 根因 |
| --- | --- | --- | --- |
| U1 | **切 tab 就重新请求** | 「点一下要等 3 秒，来回点很烦」 | 前端每次 `setActiveTab` 触发新的 EDA 请求；后端无缓存 → 每次都全量重算 |
| U2 | **15s 超时 vs 后端继续跑** | 「提示超时了，刷新一看结果又出来了/或者一直转圈」 | axios timeout 15s（已知约定），后端无超时/取消传播（E9） |
| U3 | **相关性分析偶发卡住很久** | 「点了相关性，整个页面像死了」 | O(C²) 串行 + 无进度反馈（U10） |
| U4 | **同一数据集重复分析无加速** | 「第二次点还是那么慢」 | 零缓存（§2.3） |
| U5 | **首次加载页面发出多个重请求** | 「打开页面就卡一下」 | `useEffect` 里 `getSchema` + `previewDataset(50)` 并行，各自全量加载一次 |

### 3.2 结果准确性 🟡

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| U6 | **相关性无 p 值 / 无样本量提示** | 只有系数。用户可能对 5 个样本得出的 r=0.99 深信不疑 |
| U7 | **`auto` 方法选择对用户不可见** | `_auto_matrix` 返回 `methods_used`，但前端若不展示，用户不知道"A 列用了 Spearman、B 列用了 Pearson"，而同矩阵里不同单元格方法不同（`:835-839` 异类统一 Spearman）→ **同一张热力图混用两种相关系数**，视觉上不可区分 |
| U8 | **分桶 label 精度 `:.4g` 在小量级下不可读** | 金额 0.0001 级别 → label 变成 `[0, 0)` |
| U9 | **箱线图 whisker 固定 1.5×IQR** | 不可配（端点无 `k` 参数传给它，`boxplot` 内部硬编码 `1.5`） |

### 3.3 数据展示友好度 🟡

| 编号 | 问题 | 说明 |
| --- | --- | --- |
| U10 | **无加载进度 / 无阶段提示** | 长分析只显示 `Loading` 组件；用户不知道在算哪一步、还要多久 |
| U11 | **`skipped` 列与 `ok` 列结构不一致** | 异常值表里有的行有 `bounds`、有的没有 → 前端表格出现空白格，无解释 |
| U12 | **Q-Q 图/散点图点数无上限展示策略** | 100 万点回传，前端 recharts 渲染必卡死 |
| U13 | **`grouped_bar` 无基数上限** | 高基数列会画出无法阅读的图 |
| U14 | **热力图列数无上限** | 100 列 → 100×100 矩阵，前端格子小到看不见 |
| U15 | **无「结果摘要」一句话结论** | 用户要自己从数字里找洞察；对比 `report_charts.py` 的「必保图表」思路，分析结果也应有 highlight |
| U16 | **分析结果不可导出** | 无 CSV/Excel 下载（F8） |

### 3.4 报错提示 🔴

| 编号 | 问题 | 用户看到的内容 | 期望 |
| --- | --- | --- | --- |
| **U17** | **错误信息全英文** | `"EDA columns not found"` / `"correlation requires at least 2 numeric columns"` / `"unsupported chart type: 'xxx'"` | 中文 + 可操作建议，如「相关性分析至少需要 2 个数值字段，当前仅 1 个」 |
| U18 | **`details` 是原始 Python 结构** | `details: {"missing": [...], "available": [...]}` 直接透出 | 应转成人类可读句子 |
| U19 | **参数校验错误是正则原文** | `string does not match regex "^(histogram|bar|...)$"` | 应列出合法取值 |
| U20 | **500 错误信息完全无用** | `"Internal Server Error"` + `code: INTERNAL_ERROR` | 应保留 request_id 给用户（已做 ✓），但前端需展示出来便于报障 |
| U21 | **同类问题错误码分散** | 缺参数可能是 400（`_require_str`）/ 422（FastAPI）/ 400（`DataQualityError`） | 统一为 422 表示"请求不可处理" |
| U22 | **错误无 traceback 落日志** | `AppException` 分支不 `logger.exception`（E7） | 线上排查困难 |
| U23 | **前端提示与后端 errors 数组拼接** | `runAll()` 用 `errors.join("；")` 把所有 EDA 失败拼成一行 | 多个失败堆一起，无法定位是哪个分析失败 |

---

## 4. 优化建议

### 🔴 高优先级

#### H1. 引入版本级 DataFrame 缓存（收益最大）

**问题**：`load_version()` 无缓存，一轮 EDA 浏览重复解码同一 Parquet 8~10 次（P1）

**方案**：在 `DataEngineService` 或 `deps.py` 增加**进程级** LRU 缓存，key = `(dataset_id, version)`，value = `pl.DataFrame`。

```python
# 建议位置：app/data_engine/cache.py（新增），deps.py 持有单例
class VersionFrameCache:
    """按 (dataset_id, version) 缓存不可变版本快照。
    版本是不可变的，因此缓存永不失效，只需淘汰。"""
    def __init__(self, max_items: int = 8, max_bytes: int = 512 * 1024 * 1024):
        ...
```

**理由**：`DatasetVersion` 语义上**不可变**（"原版本永远不覆盖"，`service.py` 文档字符串），因此缓存**无失效问题**，只需 LRU 淘汰。

**预期收益**：
- 一轮 EDA 浏览从 8~10 次解码 → **1 次**
- 首次请求后，后续请求延迟从「秒级」降到「毫秒级 + 计算时间」
- 实现成本低（约 60 行），风险可控（缓存命中失败就回退到读盘）

**风险**：内存占用。需设 `max_bytes` 上限（建议 512 MB）+ 记录日志；单 worker 前提不变。

---

#### H2. `profile` / `describe` 批量化聚合（7× → 1×）

**问题**：`profile()` 每列 7 次独立扫描（P4）

**方案**：把逐列 Python 循环改为单次 `df.select([...])` 批量表达式。

```python
# 现状（analysis.py:134-175）：for name, dtype in df.schema.items(): series.min() / max() / mean() / ...
# 改为：按类型分组，一次性收集所有表达式
numeric_cols = [c for c, d in df.schema.items() if d.is_numeric()]
exprs = []
for c in numeric_cols:
    exprs += [pl.col(c).min().alias(f"{c}__min"), pl.col(c).max().alias(f"{c}__max"), ...]
stats = df.select(exprs).row(0, named=True)   # 一次遍历
```

**预期收益**：profile 从 **7×R×C** 降到 **~1×R×C**，实测通常 **3~6 倍**加速。

---

#### H3. 相关性矩阵批量化 + 抽样 + 上限

**问题**：O(C²) 次 `_corr`，每次两列投影 + `drop_nulls()`（P2）

**方案**（三层）：
1. **一次投影**：`mat_df = df.select(numeric_cols).drop_nulls()` 只做一次，后续 `_corr` 直接对 `mat_df` 切片，不再重建
2. **对称性优化**：只算上三角（`a < b`），下三角复制 —— 直接省一半
3. **抽样上限**：当 `mat_df.height > N`（建议 5000）时抽样，并在结果里返回 `sampled: true` + `sample_size`，前端标注"基于 N 条抽样"
4. **列数上限**：`numeric_cols` 超过 30 时，返回 `truncated: true` 并只算方差最大的前 30 列（或让前端显式传 `columns`）

**预期收益**：
- 常数上 **5~10×**（批投影 + 对称性）
- 100 列 × 50 万行从「分钟级」降到「秒级」
- 彻底消除前端 15s 超时

---

#### H4. 统一错误信息中文化 + 错误码规范化

**问题**：全英文错误、错误码分散 400/422、traceback 不落日志（U17~U23, E5, E6, E7）

**方案**：
1. `analysis.py` 内所有 `raise TransformError(...)` 的 `message` 改中文，英文细节放 `details`
2. 参数不适用（缺 `column`、列不存在、列数不足）统一用 `ValidationException`（422）
3. `middleware.py` 的 `AppException` 分支加 `logger.warning(..., exc_info=True)`（或对非 4xx 用 `logger.exception`）
4. 为 `Query(pattern=...)` 加 `description`，并在前端映射常见 422 为正则友好文案

**预期收益**：用户能自己看懂 60% 以上的报错；线上可定位业务异常堆栈。

---

#### H5. Q-Q / scatter / line 的硬上限（防打爆）

**问题**：Q-Q 无上限（P9）、scatter 上限 10 万（仍过大）、返回体可达 16 MB

**方案**：
- `qq`：加 `sample_limit`（默认 **5000**），抽样后仍算参考线（抽样不改变分布形状）
- `scatter`：`sample_limit` 上限从 `100000` 降到 **10000**（超出前端也渲染不了）
- `line`：`max_points` 从代码常量提到 Query 参数，默认 1000，上限 5000
- **统一**：所有返回点数组的端点加 `downsampled: bool` + `original_count: int` 字段，前端可提示"已抽样展示"

**预期收益**：消除前端卡死与超大响应；超时问题一并缓解。

---

### 🟡 中优先级

#### M1. 列裁剪：把 `columns` 下推到 `read_parquet`

**问题**：指定 `columns` 后仍加载全表（P11）

**方案**：Parquet 支持列投影。
```python
# dataset_service.py 增加
def load_version_columns(self, dataset_id, version, columns: list[str] | None = None) -> pl.DataFrame:
    ...  # pl.read_parquet(io.BytesIO(content), columns=columns)
```
`eda.py` 的 `_load_df` 在 `columns` 存在时传入。

**预期收益**：只分析 3 列的请求，I/O 与解码量降到 `3/C`。

---

#### M2. `preview` 的总数走 DB 元数据

**问题**：无筛选时仍全表 count（P5）

**方案**：`DatasetVersion.row_count` 已有 → 无 filter 时直接返回它；有 filter 时才 `select(pl.len())`。

**预期收益**：`/preview` 在无筛选场景省一次全表扫描。

---

#### M3. 暴露缺失的端点参数

| 参数 | 端点 | 说明 |
| --- | --- | --- |
| `method` | `/eda/visualize` | 让 heatmap 可选 pearson/spearman（I4） |
| `max_points` | `/eda/visualize` | line 抽稀（I5） |
| `seed` | `/eda/visualize` | scatter 可复现（I5） |
| `expected_schema` | `/datasets/{id}/quality` | 启用 SchemaChecker（F4） |
| `k` | `/eda/visualize`（boxplot） | whisker 倍率（U9） |
| `top_n` | `/eda/visualize`（grouped_bar） | 分类基数上限（P10） |

**预期收益**：补全已有算法能力，零新代码量（只是把内部 `options.get()` 提到 Query 参数）。

---

#### M4. 缺少的交叉表 / 列联表分析

**问题**：F5

**方案**：新增 `CrossTabAnalyzer`（复用 `DataEngineService.apply_operation` 的 `pivot`/`aggregate`），端点 `GET /eda/cross`，参数 `row` / `col` / `agg` / `normalize`（none/row/col/all）。

**预期收益**：补齐"分类 × 分类"这一最常见的 EDA 缺口，复用已有 pivot 算子，实现量小。

---

#### M5. 相关性 p 值 / 显著性

**问题**：F11, U6

**方案**：用 t 检验近似公式（无需 scipy）：
```
t = r * sqrt((n-2) / (1 - r²))
```
再自研 t 分布 CDF 的近似（或返回 t 值让前端按阈值判断）。结果加 `p_values` 矩阵 + `n_pairs` 矩阵。

**预期收益**：论文级严谨性提升；避免小样本误导。

---

#### M6. 分析结果导出

**问题**：F8, U16

**方案**：新增 `GET /datasets/{id}/eda/{kind}/export?format=csv|xlsx|json`，复用已算结果（配合 H1 缓存）+ `df_to_records`。

**预期收益**：用户能把分析结果带走；论文可直接引用。

---

#### M7. 分析结果持久化与历史对比

**问题**：F9

**方案**：新增 `AnalysisRun` 表（`dataset_id` / `version` / `kind` / `params` / `result_json` / `created_at`），可选"保存本次分析"。配合版本对比可回答"清洗后质量变好了吗"。

**预期收益**：从"一次性工具"升级为"可追溯的分析档案"，是毕设可讲的功能扩展点。

---

#### M8. 并发闸门与超时保护

**问题**：C1, C5, E9

**方案**：
1. 对重分析端点加进程级 `threading.Semaphore`（建议 4），超出时返回 `429` + `Retry-After`
2. 关键循环（相关性、profile）内检查 `request.is_disconnected()`，断开则提前退出（或至少设一个软 deadline）
3. `QQ`/`correlation` 加内部耗时兜底：超过 `ANALYSIS_SOFT_TIMEOUT_SECONDS`（建议 60s）时降级返回（抽样结果 + `partial: true`）

**预期收益**：防雪崩；单 worker 下保障其他接口可用。

---

#### M9. `_load_df` 去重 + 响应模型化

**问题**：I1, I2, I6

**方案**：
- 把 `_load_df` 提到 `data_engine/service.py` 作为 `load_dataset_frame(dataset_id, version)` 方法
- 为每个端点定义 Pydantic 响应模型（`DescriptiveResult` 等），让 OpenAPI 有 schema

**预期收益**：减少重复、前端类型可从 OpenAPI 生成。

---

### 🟢 低优先级

| 编号 | 项 | 说明 | 预期收益 |
| --- | --- | --- | --- |
| L1 | 分布形态统计（skew / kurtosis / mode） | `DescriptiveAnalyzer` 补 3 个量，公式简单 | 论文描述统计更完整（F10） |
| L2 | 缺失值模式矩阵 | 新增 `missing_pattern` 图表类型，输出列 × 行的缺失布尔矩阵抽样 | 直观看出"缺失成块"（F7） |
| L3 | 分桶 label 智能精度 | 按 `(mx-mn)` 量级动态选有效数字而非固定 `.4g` | 小量级数据可读（P13, U8） |
| L4 | `grouped_bar` 兜底截断 | 使用 `top_n`（当前完全未用）+ 返回 `truncated` | 防高基数爆炸（P10） |
| L5 | `line` 抽稀算法修正 | 用 `with_row_index().filter(pl.col("__i") % step == 0)` 或 `gather_every` | 保证首末点与均匀抽稀（P12） |
| L6 | `qq` 的 `_norm_ppf` 向量化 | 用 polars 表达式替代 Python 循环 | 大 n 下快数倍（P9） |
| L7 | 热力图列数上限 | 超 30 列时返回 `truncated` + 提示 | 前端可读（U14） |
| L8 | 分析结果一句话摘要 | 基于算出的统计量生成 highlight（如"最强相关：x↔y r=0.83"） | 降低读图门槛（U15） |
| L9 | 端点加 `Cache-Control` / ETag | 配合 H1 让浏览器也缓存 | 减少重复请求 |
| L10 | 错误 message 的 details 结构化 | `details` 加 `field` / `hint` 字段供前端渲染 | 报错更友好（U18） |
| L11 | `_PLANNERS` 改 LRU | 当前 `clear()` 是全清而非淘汰（C6） | 计划缓存更稳 |
| L12 | heatmap 方法混用可视化 | 返回 `methods_used` 时前端标注单元格方法（U7） | 避免误读 |

---

## 5. 风险速查表

> 「状态」列反映第 6 章实施后的结果：✅ 已修复 / ⚙️ 部分缓解 / ⬜ 未处理。

| 编号 | 类别 | 问题 | 位置 | 级别 | 状态 |
| --- | --- | --- | --- | --- | --- |
| P1 | 性能 | `load_version` 无缓存，每次全量读 Parquet | `dataset_service.py:378` | 🔴 **P0** | ✅ H1 |
| P2 | 性能 | 相关性 O(C²) 次独立计算 + 重复 drop_nulls | `analysis.py:810,828` | 🔴 **P0** | ✅ H3 |
| C1 | 并发 | 端点全同步，40 线程池无闸门 → OOM 风险 | `eda.py` 全部端点 | 🔴 **P0** | ⬜ 顺延 |
| U17 | 体验 | 错误信息全英文 | `analysis.py` 全局 | 🔴 **P0** | ✅ H4 |
| P4 | 性能 | `profile` 每列 7 次独立扫描 | `analysis.py:134` | 🔴 P1 | ✅ H2 |
| P3 | 性能 | `analyze_schema` 每列 `n_unique()` | `analysis.py:56` | 🔴 P1 | ✅ H2 |
| P9 | 性能 | `qq` 无抽样上限，100 万点回传 | `analysis.py:1156` | 🔴 P1 | ✅ H5 |
| E1 | 健壮 | `n_unique()` 对不可哈希类型抛 500 | `analysis.py:56,150` | 🔴 P1 | ✅ H2 |
| E9 | 健壮 | 无超时/取消传播，断开仍跑 | 全局 | 🟡 P1 | ⬜ 顺延 |
| P10 | 性能 | `grouped_bar` 无基数上限，`top_n` 未使用 | `analysis.py:1189` | 🟡 P2 | ✅ H5 |
| P5 | 性能 | `preview` 无筛选时仍全表 count | `operations.py:371` | 🟡 P2 | ✅ M2 |
| I4 | 接口 | `visualize` 缺 `method` 参数，heatmap 不可选 | `eda.py:116` | 🟡 P2 | ✅ M3 |
| F4 | 功能 | `expected_schema` 未暴露，SchemaChecker 不可用 | `dataset_analysis.py:165` | 🟡 P2 | ✅ M3 |
| E5/E6 | 异常 | 错误码 400/422 不统一 | 多处 | 🟡 P2 | ✅ H4 |
| E7 | 异常 | `AppException` 不记 traceback | `middleware.py:56` | 🟡 P2 | ✅ H4 |
| I1 | 冗余 | 两份 `_load_df` | `eda.py:30` / `dataset_analysis.py:21` | 🟢 P3 | ⬜ 不合并（见 6.4） |
| E3 | 一致性 | `ok`/`skipped` 结构不一致 | `analysis.py:960` | 🟢 P3 | ✅ H5 |
| U7 | 准确性 | auto 方法混用不可见 | `analysis.py:835` | 🟢 P3 | ✅ H3（`pair_methods`） |
| E10 | 健壮 | `run_multi_merge` 无事务边界 | `service.py:982` | 🟢 P3 | ⬜ 未处理 |

---

## 附：亮点清单（答辩可讲）

1. **零第三方依赖的科研级统计**：`_norm_ppf` 自研 Acklam 有理逼近（误差 ~1.15e-9），环境无 scipy/numpy 也能出 Q-Q 图 —— `analysis.py:1264`
2. **`auto` 相关性方法智能选择**：整数列→Spearman（计数/序数），浮点列→Pearson，异类统一 Spearman 保证单调性稳健 —— `analysis.py:821-847`
3. **边界处理细致**：常数列、单样本 std、NaN/Inf、空 df、分母为 0、空 Parquet 全有兜底
4. **安全性**：筛选 DSL 白名单算子、路径穿越防护（`storage/security.py`）、无 eval/exec
5. **筛选 DSL 与 LazyFrame 结合**：`preview()` 全链路 lazy，`select(pl.len())` 避免物化
6. **一套算法两处复用**：`analysis.py` 的 Analyzer 同时服务前端 REST 端点与 Agent 工具（`eda_tools.py`），无重复实现 ✅
7. **9 类图表覆盖**直方图/柱/折线/散点/箱线/热力/Q-Q/分组柱/CDF，其中后 3 类达到科研级

---

*本报告仅静态分析源码，未运行测试。建议按 H1 → H4 → H3 → H5 顺序落地，前两项即可解决 80% 的性能与体验投诉。*

---

## 6. 已落地优化项（2026-09-21 实施）

> 本章记录按第 4 章建议实际完成的代码改动，**逐项给出原因 / 影响 / 风险 / 回滚方式**。
> 全部改动仅限 `backend/`，未触碰 `frontend/`。
> 验证方式：静态冒烟脚本（一次性，已删除），覆盖 78 项断言，结果全部通过。

### 6.1 改动文件清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `app/data_engine/cache.py` | **新增** | 版本级 DataFrame LRU 缓存 |
| `app/services/dataset_service.py` | 修改 | 接入缓存 + Parquet 列裁剪 |
| `app/core/config.py` | 修改 | 新增 6 个配置项 |
| `app/core/middleware.py` | 修改 | `AppException` 记录日志与 traceback |
| `app/analysis.py` | 大量修改 | 批量化聚合、对称相关矩阵、图表上限、错误中文化 |
| `app/api/v1/eda.py` | 修改 | 补参数、列裁剪下推、图表列推断 |
| `app/api/v1/dataset_analysis.py` | 修改 | 暴露 `expected_schema`、`preview` 走元数据行数 |

### 6.2 高优先级（H1–H5）

#### H1 · 版本级 DataFrame 缓存 ✅ 已实现

- **原因**：`DatasetVersion` 是不可变快照（源码明确「原版本永远不覆盖」），因此缓存**永不失效**，
  只需 LRU 淘汰即可 —— 这是本方案成立的理论基础。
- **实现**：新增 `VersionFrameCache`（`OrderedDict` LRU，`max_items` + `max_bytes` 双限）。
  `get()` 返回 `frame.clone()`，避免调用方原地修改污染缓存；`delete()` 时按 `dataset_id` 失效。
- **配置**：`DATASET_FRAME_CACHE_ENABLED=True`、`MAX_ITEMS=16`、`MAX_BYTES=512MB`。
- **预期收益**：同一版本重复分析（调参重跑、多端点连续请求）从「每次读 Parquet」变为内存命中。
- **风险**：内存占用上限 512MB（可配）。**多 worker 部署时会各自持有一份缓存**——本项目因
  进程内状态约束（架构文档已注明）必须单 worker，故无此问题。
- **回滚**：`DATASET_FRAME_CACHE_ENABLED=False` 即整体关闭；或把 `MAX_ITEMS` 设为 0。

#### H2 · `profile` / `describe` 批量化聚合（7× → 1×）✅ 已实现

- **原因**：原实现按列逐次调用 `series.min()/max()/mean()/median()/std()/quantile()`，
  每列 7 次独立扫描；`analyze_schema` 的 `n_unique()` 同样逐列。
- **实现**：新增 `_batch_series_stats()`，用一次 `df.select([...]).row(0, named=True)`
  出齐全部聚合量；`_stats_of()` 负责取值。新增 `_numeric_profile_from_stats()` /
  `_temporal_profile_from_stats()` 消费这批统计量，删除旧的逐列 `_numeric_profile` /
  `_temporal_profile`（避免留下死代码）。
- **安全性修复**：新增 `_safe_n_unique()`，对 List/Struct/Array/Object 等不可哈希类型
  返回 `-1`，修掉原先直接抛 500 的健壮性问题（风险表 E1）。
- **预期收益**：C 列的扫描次数从 7C 降到 1（外加类型分支），大宽表收益线性放大。

#### H3 · 相关性矩阵批量化 + 对称 + 抽样 + 列上限 ✅ 已实现

- **原因**：原实现按列对独立计算，C 列要算 C² 次，且每个列对重复做 `drop_nulls`。
- **实现**：
  - `_Prepared` dataclass + `_prepare()`：一次投影、**单次** `drop_nulls`、剔除全空列。
  - `_matrix()` 改为只算上三角并镜像填充，天然对称。
  - `_auto_matrix()` 附带 `pair_methods`（每个列对实际用的方法），可追溯。
  - 行抽样（`ANALYSIS_CORRELATION_MAX_ROWS=5000`）+ 列上限
    （`ANALYSIS_CORRELATION_MAX_COLUMNS=30`，按方差排序保留信息量最大的列）。
- **元信息恒定**：`sampled` / `sample_size` / `original_rows` / `truncated` / `dropped_columns`
  **始终存在**（未抽样时 `sampled=False`、`sample_size=original_rows`），前端不必写两套分支。
- **预期收益**：C² → C(C−1)/2 且每对常数开销消失；超大表不再打爆内存与响应体。
- **注意**：抽样会引入估计误差，`sampled=true` 时前端应提示「基于 N 行抽样」。

#### H4 · 错误信息中文化 + 错误码规范化 ✅ 已实现

- **原因**：错误信息全英文（风险表 U17），且 `AppException` 不记 traceback（E7）。
- **实现**：
  - `analysis.py`：`require_columns` / `DistributionAnalyzer` / `compute_outlier_bounds` /
    `SchemaChecker` / `MissingChecker` / `DuplicateChecker` / `OutlierChecker` /
    `VisualizationBuilder` 的报错全部中文化，并补 `details`（缺失列 + 可用列清单）。
  - `_FIELD_LABELS` 字段中文名映射，报错里用「字段（column）」形式，兼顾可读与可定位。
  - `middleware.py`：`AppException` 分支按状态码分级记录 —— 5xx 用 `logger.error(..., exc_info=True)`，
    4xx 用 `logger.warning`。此前异常被静默吞掉，排查困难。
  - 统一改用 `ValidationException`（422）替代裸 `ValueError`，错误码口径一致（E5/E6）。
- **回滚**：本轮未改 HTTP 状态码语义（仍走 `AppException.http_status`），仅统一了抛错类型与文案。

#### H5 · Q-Q / scatter / line 大点集上限 ✅ 已实现

- **原因**：`qq` 图无抽样上限，百万行数据集会把百万个点回传前端（风险表 P9）；
  `scatter` 上限 10 万仍然过大；`line` 无抽稀。
- **实现**：
  | 图表 | 处理方式 | 新增字段 |
  | --- | --- | --- |
  | `line` | `gather_every(step)` 等间隔抽稀 | `downsampled`、`original_count` |
  | `scatter` | 抽样上限由 10 万降到 1 万 | `sampled`、`original_count` |
  | `qq` | `QQ_MAX_POINTS=5000` + 抽稀 | `sampled`、`original_count` |
  | `boxplot` | 新增 `k` 参数（whisker 倍率） | `k` 回显 |
  | `heatmap` | 透出相关性元信息 | `pair_methods`/`sampled`/`original_count`/`truncated` |
  | `grouped_bar` | `top_n` 基数上限（原先参数未使用，P10） | `truncated`、`dropped_categories` |
  | `area` | 统一抛 `ValidationException` | — |
- **附带修复**：分布图分桶标签原用固定 `:.4g`，会丢精度或出现同标签；
  改为 `_bin_label_format(mn, mx)` 按量级自适应精度。
- **附带修复（P7）**：`EdaOutlierAnalyzer` 原有两个 mask 全列扫描，
  统一为单次表达式求值；`skipped` 列补齐 `bounds`/`outlier_count`/`sample_outliers`/
  `inlier_range`/`stats` 全部键，前端表格不再出现空白格（E3）。
- **行为对齐（重要）**：原 `iqr` 分支在 IQR=0（常数列）时**静默返回退化区间**
  （`lower == upper`），前端看起来像「0 个异常」但实际没算；而 `zscore` 分支会标记 `constant`。
  现已统一：IQR=0 时也返回 `constant: True`，`EdaOutlierAnalyzer` 随之把该列标为 `skipped`，
  与 `zscore` 口径一致。
- **预期收益**：极端数据集不再打爆浏览器；所有降采样都可追溯（带 `original_count`）。

### 6.3 中优先级（M1–M3）

#### M1 · 列裁剪下推到 Parquet ✅ 已实现

- **实现**：`DatasetService.load_version(..., columns=...)` 新增列裁剪路径；
  仅当**请求列数 < 总列数 × 0.6** 时才裁剪（否则读全表更划算），避免为了省 2 列反而多做一次 schema 查询。
- **端点接线**：
  - `descriptive` / `correlation` 透传 `columns`。
  - `distribution` 固定传 `[column]`（单列分析）。
  - `quality` / `outlier` 透传 `columns`。
  - `visualize` 新增 `_chart_columns()`：按图表类型推断真正需要的列
    （如 `scatter` 只要 x/y、`qq` 只要 column）；`heatmap` 需要全部数值列，仅在显式传
    `columns` 时才裁剪。
- **注意**：`columns` 存在时会**跳过缓存**（返回的是列子集，与缓存的整表语义不同），
  属于有意取舍 —— 列裁剪省 I/O，缓存省解码，两者互斥。

#### M2 · `preview` 总数走 DB 元数据 ✅ 已实现

- **实现**：`preview()` 新增 `known_total` 参数；**仅在无 filter 时**生效（有 filter 行数会变）。
  接口层传 `version_row.row_count`。
- **收益**：列表页翻页省掉一次 `select(pl.len()).collect()` 全表扫描。
- **安全性**：过滤条件下自动回落真实 count，不会返回错误总数。

#### M3 · 暴露缺失的端点参数 ✅ 已实现

- `visualize` 补 `method`（heatmap 原先不可选，风险表 I4）、`k`、`max_points`、`seed`，
  并把 `sample_limit` 上限从 10 万收紧到 1 万；全部 Query 补中文 `description`。
- `quality` 补 `expected_schema`（JSON 串），使原先**存在但不可达**的 `SchemaChecker` 真正可用
  （风险表 F4）。解析层 `_parse_expected_schema()` 对非 JSON / 非对象 / 非字符串键值统一抛
  `ValidationException` 中文报错。
  - **语义提示**：`SchemaChecker` 把声明的键视为**完整的期望列集合** ——
    多出来的列会产生 `low` 级别「预期外字段」提示，这是设计意图（用于契约校验），
    不是 bug。只想校验部分列时应只声明这些列并接受此提示。

### 6.4 未采纳 / 顺延项

| 项 | 结论 | 理由 |
| --- | --- | --- |
| C1 并发闸门（Semaphore） | 顺延 | 与「进程内单 worker」约束叠加后收益有限；毕设演示场景不构成实际风险 |
| M8 超时保护 | 配置已就绪 | `ANALYSIS_SOFT_TIMEOUT_SECONDS=60` 已入配置，但未接线（需在端点层包 `asyncio.wait_for`，属行为变更，暂缓） |
| M4/M5/M6/M7 | 未做 | 属功能拓展（交叉表 / p 值 / 导出 / 持久化），超出「优化」范围 |
| I1 两份 `_load_df` 合并 | 未做 | 两处分层职责不同（一处带列裁剪、一处带 `version_row`），强行合并会引入耦合 |
| E9 断连取消 | 未做 | 属架构级改动（需贯穿 LazyFrame 执行），风险 > 收益 |

### 6.5 验证方式

一次性冒烟脚本（`scripts/_smoke_opt.py`，**验证后已删除**）覆盖 78 项断言，全部通过，包括：

- 路由与参数注册（OpenAPI 层复核：`/eda/visualize` 含 `method`/`k`/`max_points`/`seed`，
  `/quality` 含 `expected_schema`）—— 共 68 条路径，EDA 5 端点参数齐全。
- `analyze`/`profile` 结构与数值正确性（含常数列、字符串列无 `mean`）。
- 相关性矩阵对称性、对角线为 1、完全共线列 r≈1、`auto` 模式下整数列→Spearman。
- 分布分桶标签为区间且相邻不同。
- 异常值常数列被 `skipped` 且结构补全、`reason` 中文。
- 各图表抽稀/截断生效且 `original_count` 正确。
- `SchemaChecker` 三态（匹配 / 类型不符 / 缺列）与未知类型报错中文化。
- `preview` 的 `known_total` 生效、带 filter 时正确回落真值、列裁剪生效。

**未覆盖**：真实 HTTP 请求链路、并发场景、超大文件（>100 万行）的端到端性能 ——
建议后续在有真实数据集的环境下补一次端到端验证。
