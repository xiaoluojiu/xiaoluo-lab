# Agent 平台架构重构：审查报告 + 标准化设计 + 改造清单

> 面向「 airlines 出发延误预测（OpenML 42728，1000 万行 × 10 列，目标 DepDelay 回归）」暴露的
> 10 个问题，做架构级归因与标准化重构。本文既是**改造前审查报告**，也是**未来扩展的标准参考**。

---

## 零、一句话结论

平台架构（Agent / Workflow / Experiment 三层分离、Tool Registry + Permission + Validator）
是对的，问题出在**每个模块的防错能力不够**，且**同类逻辑在不同模块里各写了一份**。

10 个问题归到底是三个矛盾：

| 矛盾 | 表现 | 解法（本次落地） |
| --- | --- | --- |
| 默认值 vs 意图 | 未指定目标列 → 默认 clustering | 把默认值改为**触发澄清**（Pre-flight + 反问协议） |
| 通用方法 vs 领域适配 | IQR 在非负字段上给出负下界 | 方法选择依赖**字段语义元数据**（ColumnSemantics） |
| 描述性输出 vs 决策性输出 | 报告 20 个数字，没有「所以呢」 | 报告分**事实 / 判断 / 行动**三层 |

---

## 一、改造前审查：设计意图 vs 实际实现

### 1.1 Agent 层（第一层）

| 模块 | 设计意图 | 实际实现 | 差距 |
| --- | --- | --- | --- |
| `agent/runtime/runtime.py` | 路由 → 上下文 → 规划 → 执行 | 有路由，无「开工前检查」 | **没有任何环节在动手前问「你要预测什么」**，目标列不明时一路走到 KMeans |
| 关键词路由 | 一处维护 | `_route` / `ContextBuilder.with_tools` / `Planner._rule_plan` **三份关键词表** | 改两处漏一处 ⇒ 「工具已注册但 Agent 不调用」 |
| `RunStatus` | 状态机 | 只有 `waiting_confirmation`（授权） | 缺「等用户补信息」这一态 |
| LLM 反问 | 由 `clarify` 工具统一发起 | 无此工具，LLM 只能自由生成问题文本 | 问题不可机读、不可渲染、不可追踪 |

### 1.2 Quality 层（第二层）

| 模块 | 设计意图 | 实际实现 | 差距 |
| --- | --- | --- | --- |
| `analysis.compute_outlier_bounds` | 算异常值边界 | 返回 `(lower, upper, info)` 裸三元组 | 调用方**无从判断方法是否适用**；`[-16.5, 19.5]` 照样输出 |
| 字段元数据 | 驱动方法选择 | 只有 dtype 与 `classify_columns` 的布尔判定 | 无「非负 / 长尾 / 零膨胀 / 高基数 / 伪时间」这些**决定方法适用性**的维度 |
| 目标列保护 | 目标列不做清洗 | 无 | 目标列的极端值被当成脏数据建议清洗 |
| 业务口径 | 用户可定义「什么是异常」 | 无入口 | 统计口径一家独大 |

### 1.3 ML Engine 层（第三、四层）

| 模块 | 设计意图 | 实际实现 | 差距 |
| --- | --- | --- | --- |
| 特征工程 | 可注册、可复用 | **没有注册表**；各处在 `if 列名 == 'xxx'` | HHMM 伪时间未解析、高基数未编码、共线性未提示 |
| 高基数列 | 可控 | `OneHotEncoder` 无 `max_categories` | 10 列 → 767 列 → `ArrayMemoryError: 57.1 GiB` |
| 评估指标 | 适配任务 | 固定 `mae/mse/rmse/r2` | 零膨胀长尾目标上不适配；无分层误差、无业务指标 |
| 指标解释 | Agent 能解释 | 无 | `r2=0.02` 被读成「模型没用」 |

### 1.4 Experiments 层（第五层）

| 模块 | 设计意图 | 实际实现 | 差距 |
| --- | --- | --- | --- |
| 实验记录 | 可解读 | 只有 `status` + `metrics` | 失败实验 #29 无原因、无修复建议；成功实验 #30 定位模糊 |
| 实验对比 | 输出结论 | `comparator` 只输出指标最优与参数 diff | 不回答「哪个改动带来了提升」 |

### 1.5 Reports 层（第六层）

| 模块 | 设计意图 | 实际实现 | 差距 |
| --- | --- | --- | --- |
| 结论区 | 决策化 | 平铺 7 条结论 + 4 条建议，与正文重复 | 无优先级、无「发现→行动」对应、结尾无「下一步」 |
| 方法局限 | 显式说明 | 无 | 读者不知道「这个结论在什么情况下不成立」 |

### 1.6 贯穿性的重复代码（架构级问题）

| 重复 | 出现处 |
| --- | --- |
| 「用户想干什么」的关键词表 | runtime / ContextBuilder / planner（3 处） |
| 键值注册表的实现 | ToolRegistry / ModelRegistry / DataEngine `OPERATION_REGISTRY`（3 处，报错口径各异） |
| 「字段分类」的判定 | `analysis.classify_columns` / `target_inference`（2 处，各写一份分词与规则） |
| 「注册表 + 旁路元数据字典」 | DataEngine 的 `OPERATION_REGISTRY` + `OPERATION_METADATA` |

---

## 二、标准化内核（本次重构的地基）

四层共用，新增能力一律从这里长出来。

### 2.1 统一决策契约 `app/core/contracts.py`

任何「程序替用户做的判断」都必须是这个形状：

```python
Decision(value, source, confidence, reasons, alternatives, findings, evidence)
```

* `value is None` ⇒ **没有结论**，调用方必须查 `needs_clarification`，禁止拿 None 硬往下走；
* `confidence` 与 `source` 成对出现——没有来源的置信度没有意义；
* `findings: list[Finding]` 带 `code` + `severity` + `evidence` + `suggestion`；
* `Severity` 四档直接对应「反问三档输出」：`BLOCK`（硬阻断）/ `CLARIFY`（反问）/ `WARN`（软警告）/ `INFO`（标注）。

`ml_engine.target_inference.TargetGuess` 是改造前**唯一**自带证据链的判定；
新契约把它推广到全部六层。

### 2.2 统一注册表协议 `app/core/registry.py`

```python
class Registry(Generic[T]):
    register / unregister / get / try_get / keys / values / items / describe_all
    _conflict_error(key)   # 子类定制，沿用本层既有异常体系
    _missing_error(key)
```

三张表现有注册表已改为继承它（`ToolRegistry`、`ModelRegistry`、新增的
`FEATURE_OP_REGISTRY`），**删除了三份重复的 dict 管理代码**。
`describe_all()` 让元数据回到注册项自己身上（消除 DataEngine 那种
「注册表 + 旁路元数据字典」的双份结构）。

### 2.3 统一意图判定 `app/agent/intent.py`

`classify(text) -> Decision[Intent]`，`Intent` 复用 `app.local_router.contract` 的封闭枚举
（能力域词表只有一份）。三处关键词表合并成一份 `INTENT_KEYWORDS`。

### 2.4 字段语义标准 `app/quality/semantics.py`

```python
ColumnSemantics(role, domain, shape, cardinality, null_ratio, is_target, pseudo_time_format, ...)
```

* `role`：continuous / categorical / temporal / identifier / constant / text / target
* `domain`：non_negative / positive / signed / unit —— **决定统计口径能不能直接用**
* `shape`：normal / long_tail / zero_inflated / bimodal / uniform —— **决定指标与异常值口径**
* `pseudo_time_format`：HHMM 伪数值时间（判据：整值 + 落在 [100,2359] + 分钟位 <60 + 存在 59→00 跳变）

它是 Quality / 特征工程 / 指标推荐**唯一的方法适配输入**。

---

## 三、六层改造落地情况

### 第一层：任务理解与意图澄清 —— 已完成

| 交付 | 位置 |
| --- | --- |
| Pre-flight Check（可注册检查项） | `app/agent/preflight.py` |
| 反问模板库（按 code 渲染） | `app/agent/clarify.py` |
| `agent.clarify` 工具（LLM 反问唯一通道） | 同上，已注册进 ToolRegistry |
| `RunStatus.WAITING_CLARIFICATION` + `clarification` 事件 | `app/agent/runtime/models.py` |
| `POST /agent/runs/{id}/clarify` | `app/api/v1/agent.py` |

* 检查项走注册表：`_c_dataset_required` / `_c_target_resolution` / `_c_task_conflict` / `_c_scale`；
* 目标列解析优先级与 `ml_engine.target_inference` 一致（显式 > 命名约定 > 诉求语义 > 列候选），
  但**只读列名与 dtype**（`DataEngineService.column_schema`，Parquet schema，不加载数据行）；
* 输出三档：`PROCEED` / `CLARIFY` / `BLOCK`；反问带 `options` 与 `default`，机读可渲染；
* 用户回答后**从 Pre-flight 处重新规划**（或从发起反问的那一步继续），不重跑已完成的重型步骤。

### 第二层：数据质量判定的方法适配 —— 已完成

| 交付 | 位置 |
| --- | --- |
| 字段语义标准 | `app/quality/semantics.py` |
| 异常值方法适配（声明式方法表） | `app/quality/outlier_strategy.py` |
| 接线 | `analysis.OutlierChecker` / `build_report` / `DataEngineService.quality` / `dataset.quality` 工具 |

* 方法表 `METHOD_SPECS`：`business_rule`(100) > `low_frequency`(90) > `quantile`(60) >
  `mad`(50) > `iqr`(40) > `zscore`(30) > `none`(0)；每个方法自己声明 `guard`；
* 请求方法不适用 ⇒ 产出 `quality.method_mismatch` 警告并自动改选，原因写进 message；
* **目标列保护**：`target` 传入后该列 `method=none`、`cleaning_suggested=False`；
* **业务口径优先**：`BusinessRule`（区间 / 枚举，不支持字符串求值）优先于一切统计口径；
* 兼容性：`adaptive=False` 为默认，判定数字与改造前一致，只**增加**适用性标注，
  不静默改变既有报告。

### 第三层：特征工程自动化与注册 —— 已完成

| 交付 | 位置 |
| --- | --- |
| 操作注册表 + 执行计划 | `app/ml_engine/feature/registry.py` |
| 内置操作 | `app/ml_engine/feature/operations.py` |
| 建议器 | `app/ml_engine/feature/advisor.py` |

* 操作：`time.parse_hhmm` / `time.cyclic` / `encode.frequency` / `encode.target` /
  `aggregate.group_stat` / `collinearity.flag`；
* **泄漏硬约束**：`leaky=True` 的操作拿不到 `folds` 时**拒绝执行**并返回可操作原因；
* 执行产物是 `FeaturePlan`（表达式 + 小表 join），不物化「行 × 列」的中间大表；
* 建议器输出三档优先级（`必须` / `建议` / `可选`），与报告层共用同一套词表。

### 第四层：评估指标推荐与分层 —— 已完成

| 交付 | 位置 |
| --- | --- |
| 指标推荐器 | `app/ml_engine/metric_advisor.py` |

* 声明式 `METRIC_SPECS`：长尾/零膨胀 ⇒ MAE + 分位数损失 + 分层 MAE，并对 RMSE/R² 出
  `metric.rmse_misleading` 警告；不平衡分类 ⇒ AUC + 召回率；
* `stratified_regression_report`：按目标分位数分段报误差；
* `business_metric_report`：业务阈值命中率（如「延误 > 15 分钟」）与统计指标并列；
* `explain_metric`：给 Agent 的指标解释，挡住「r² 低 ⇒ 模型没用」的误读。

### 第五层：实验记录结构化 —— 核心逻辑已完成，落库为后续项

| 交付 | 位置 |
| --- | --- |
| 结构化叙述 + 失败归因 + 结论归因 | `app/experiments/structured.py` |
| `GET /experiments/{id}/narrative` | `app/api/v1/experiments.py` |

* `ExperimentNarrative(hypothesis, conclusion, next_action, failure_reason)`；
* `classify_failure`：7 类真实失败码（target_in_excluded / missing_target /
  model_param_mismatch / memory_overflow / stratify_on_regression / column_not_found /
  silently_clustered），每类带可操作修复建议；
* `conclude`：从对比里读出「哪个改动带来了提升」（只报**差异参数**，不罗列全部参数）。

> **预留口径**：后续给 `experiments` 表加 `hypothesis / conclusion / next_action /
> failure_reason` 四个字段时，直接沿用 `REQUIRED_STRUCTURED_FIELDS` 里的字段名与语义，
> 不要另起一套命名（需要 alembic 迁移 + 生产库 upgrade，本次未做）。

### 第六层：报告分层与决策化 —— 已完成

| 交付 | 位置 |
| --- | --- |
| 三层结构 + 去重 + 优先级 + 下一步 | `app/reports/layering.py` |
| 接线 | `reports/generator.py`（`metadata["layers"]` / `metadata["next_action"]`） |

* `fact / judgment / action` 三层，各回答一个不同的问题；
* `dedupe`：归一化后做**包含度**比较，保留信息量更大的那条（不只是判相等）；
* 行动按 `必须做 > 建议做 > 可选做` 排序，不平铺；
* 报告结尾**必须**有 `next_action`（`ReportGenerator._next_action` 按「有无阻塞性质量问题 /
  有无指标」给出可执行的下一步）。

---

## 四、改造前后对照（同一数据集的同类场景）

| 场景 | 改造前 | 改造后 |
| --- | --- | --- |
| 建模请求未指定目标列 | 默认 clustering，实跑 KMeans | Pre-flight 解析：数据集名含「延误」⇒ 自动定 `DepDelay`；否则 `BLOCK` 并给出候选（日历列不排第一位） |
| 声明「分类」目标为连续列 | 静默训练 | `BLOCK`：`task_type_conflict` + 建议改 regression |
| `DepDelay` 的 IQR | 边界 `[-16.5, 19.5]`，建议清洗 | 目标列 `method=none`，只描述不清洗；非负字段自动改分位数口径并标注原因 |
| `Distance` 的 IQR | 边界 `[-636.5, 1896]` 照常输出 | `quality.method_mismatch`：非负字段 IQR 下界易为负，改用 quantile；有业务口径时业务口径优先 |
| `CRSDepTime` | 当连续数值入模 | 识别为 HHMM，建议 `time.parse_hhmm`（必须）+ 循环编码 |
| `Origin`(369) | one-hot 展开成 369 列 | 建议 `encode.frequency`（必须，列数 +1）；目标编码需 oof |
| 指标 | mae/mse/rmse/r2 | MAE + 分位数损失 + 分层 MAE + 业务命中率，并对 RMSE/R² 标注局限 |
| 失败实验 #29 | 只有 `failed` | `[target_in_excluded] 目标列被列入排除清单：把目标列从 excluded_columns 里移除…` |
| 报告结论 | 7 条结论 + 4 条建议，重复 | 三层 + 去重 + 优先级 + 「下一步：…」 |

---

## 五、验收与回归

新增 `backend/tests/test_standardization_governance.py`（80 项，纯逻辑，不调 LLM），
按六层分组钉住上述标准行为：

```
TestStandardKernel      (10)  # 统一契约 + 统一注册表
TestIntent              ( 6)  # 意图分类唯一真源
TestPreflight           ( 8)  # 第一层
TestClarify             ( 8)  # 第一层
TestSemantics           ( 7)  # 第二层
TestOutlierStrategy     ( 8)  # 第二层
TestFeatureOps          (11)  # 第三层
TestMetricAdvisor       ( 7)  # 第四层
TestExperimentNarrative ( 6)  # 第五层
TestReportLayering      ( 9)  # 第六层
```

跑法（排除会真实调用 LLM 的文件）：

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header -p no:randomly -p no:warnings -rf \
  --ignore=tests/test_agent.py --ignore=tests/test_permission_llm.py \
  --ignore=tests/test_phase10_integration.py --ignore=tests/test_phase9_benchmark.py \
  --ignore=tests/test_phase9_security.py --ignore=tests/test_local_chat.py
```

已知既有失败（与本次无关）：`tests/test_data_center.py` 的 3 项（`app/analysis.py`，
改造前即存在）。

---

## 六、给未来扩展的标准（照抄即可）

1. **新增一种「程序判断」** → 返回 `Decision`，带 `source` + `confidence` + `reasons`；
   不确定就返回 `None` + `Severity.CLARIFY`，**不要给默认值**。
2. **新增一张注册表** → 继承 `app.core.registry.Registry`，只实现两个错误钩子；
   元数据放注册项的 `describe()`，不要另建旁路字典。
3. **新增一种「方法」**（异常值 / 特征工程 / 指标）→ 往对应的声明式表里加一条记录
   （`METHOD_SPECS` / `FEATURE_OP_REGISTRY` / `METRIC_SPECS`），**不改调度代码**。
4. **新增一种用户可补的信息** → 加一个 Pre-flight 检查项 + 一个 `clarify` 模板
   （按 `code` 配对），前端自动能渲染，无需改协议。
5. **新增一种失败** → 往 `FAILURE_PATTERNS` 加一条（正则 + 修复建议）。
6. **任何「按目标取值数」做的判定** → 必须先问任务类型（回归被分层切分是两个真实事故的来源）。
7. **任何两条调用路径（工具层 / HTTP 层）里的同一逻辑** → 先合并成一份再改。
8. **派生数字必须能从主表算回去**；比值从计数算，格式化成百分比前不要提前舍位。
