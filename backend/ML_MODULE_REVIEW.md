# 机器学习模块 · 后端代码深度审查报告

> 评估对象：`backend/app/ml_engine/`、`backend/app/experiments/`、`backend/app/tools/ml_tools.py`、`backend/app/api/v1/experiments.py`
> 评估日期：2026-09-21
> 评估方法：源码静态阅读 + 既有测试用例（`tests/test_ml_*.py`）交叉验证；未运行训练/预测基准（按审查范围）。
> 结论口径：每个问题给出「存在 / 缺失」判断 + 最小修复思路，并标注严重程度（🔴 高 / 🟡 中 / 🟢 低）。

---

## 0. 模块地图（先建立共识）

### 0.1 调用链

```
前端 / Agent
  ├─ POST /api/v1/ml/train      → experiments.py:train      → ExperimentService.create + run   【训练链路】
  ├─ POST /api/v1/ml/predict    → experiments.py:predict    → ExperimentService.predict        【推理链路】
  ├─ GET  /api/v1/ml/explain/{id}→ experiments.py:explain   → ExperimentService.explain        【解释链路】
  ├─ /experiments/*             → 实验 CRUD / run / compare
  └─ Agent 工具 ml.detect_task / ml.prepare / ml.train / ml.predict / ml.evaluate / ml.compare / ml.explain
        └─ tools/ml_tools.py → ExperimentService（同一实现，Agent 与 API 共用）

ExperimentService.run → _execute:
  1) _load_data(exp)                      加载数据集版本快照（Polars）
  2) 特征排除 excluded_columns             （存于 preprocessing JSON，非独立列）
  3) build_pipeline(pp_cfg, X)             构造 sklearn ColumnTransformer 管道（仅配置，未 fit）
  4) _build_model(exp)                     注册表取适配器，seed 注入 random_state
  5) 训练集 fit_transform / 测试集 transform（防泄漏：拟合只在训练集）
  6) model.fit / model.predict / evaluate_*(真实 sklearn 指标)
  7) 落库 model.pkl + pipeline.pkl 到 Storage，写 metrics / artifacts
```

### 0.2 核心文件职责

| 文件 | 职责 |
| --- | --- |
| `ml_engine/base.py` | `ModelAdapter` 统一接口：fit/predict/predict_proba/evaluate/to_bytes/from_bytes |
| `ml_engine/registry.py` | `MODEL_REGISTRY`，11 个内置模型（4 分类 / 4 回归 / 2 聚类 / 1 PCA） |
| `ml_engine/preprocessing.py` | `PreprocessingPipeline`：缺失填补 / 编码 / 缩放，sklearn `ColumnTransformer` |
| `ml_engine/classification.py` `regression.py` `clustering.py` `dimensionality.py` | 各模型的 sklearn 适配器 |
| `ml_engine/evaluation.py` | accuracy/precision/recall/f1/roc_auc、mae/mse/rmse/r2、silhouette、混淆矩阵 |
| `ml_engine/explainability.py` | 特征重要性（feature_importances_ / |coef_|）+ 可选 SHAP |
| `experiments/service.py` | **核心编排**：create/run/predict/explain/compare |
| `tools/ml_tools.py` | Agent 侧 7 个 ML 工具，封装到 ExperimentService |
| `api/v1/experiments.py` | ML/实验 REST 端点 |

---

## 1. 功能真实性（是否真的落地）

**结论：训练/预处理/特征工程/权重加载/预测/评估全部为真实 scikit-learn 实现，无硬编码结果、无随机值冒充、无 mock 数据。**

| 能力 | 是否真实 | 证据 |
| --- | --- | --- |
| 数据预处理 | ✅ 真实 | `preprocessing.py:PreprocessingPipeline` 用 sklearn `ColumnTransformer`+`SimpleImputer`+`StandardScaler/MinMaxScaler`+`OneHot/OrdinalEncoder`；`service._execute` 中管道**只在训练集 `fit_transform`、测试集 `transform`**，不存在数据泄漏 |
| 特征工程 | ✅ 真实 | 缺失填补（mean/median/mode/constant）、one-hot/ordinal 编码、standard/min_max 缩放，全部可配置并可从数据画像自动生成（`default_preprocessing_config`） |
| 模型训练 | ✅ 真实 | `base.py:ModelAdapter.fit` → `self.estimator.fit(X_np, y.to_numpy())`，底层是真实 sklearn 模型 |
| 权重加载 | ✅ 真实 | 训练后 `model.to_bytes()`（pickle）写入 Storage 的 `model.pkl`；推理时 `MODEL_REGISTRY.get(exp.model).from_bytes(storage.read(model_key))` 反序列化并 `model.predict`（`service._load_model` / `predict`） |
| 预测接口 | ✅ 真实 | `/ml/predict` 与 `ml.predict` 加载 model.pkl + pipeline.pkl，对新数据跑 `model.predict`，分类模型额外返回 `predict_proba`（`service.predict`） |
| 评估指标 | ✅ 真实 | `evaluation.py` 全部委托 sklearn：`accuracy_score`/`precision_score`/`r2_score`/`silhouette_score` 等 |
| 可复现性 | ✅ 真实 | seed → `random_state` 注入 + 分层切分；`tests/test_ml_predict_e2e.py::test_workflow_ml_train_is_reproducible_with_same_seed` **锁定同 seed 指标一致** |
| 是否有 mock/随机/硬编码 | ❌ 不存在 | 全模块未检索到伪造结果逻辑；唯一 `mock` 是 `agent/llm/mock.py`（MockLLM，仅用于 LLM 单测，与 ML 结果无关） |

**真实性边界（属于"能力偏弱"而非"造假"）：**
- 仅有**单次 train/test 切分（默认 8:2）**，无交叉验证、无超参搜索、无集成/调优。模型均为 sklearn 基线模型，定位是"可跑通的基线"，不是 AutoML。
- `ml.prepare` / `MlPrepareTool` 是**建议**预处理配置，不是自动执行；最终是否生效取决于训练时传入。

---

## 2. 需求匹配度（对照设计目标与真实场景）

> 依据代码注释与 `docs/agent.md` 推断的设计目标：作为"数据→ML"闭环的一环，让非专业用户用自然语言完成"选模型—训练—评估—预测—解释"。以下列出**偏离 / 缺失**项。

| # | 偏离 / 缺失 | 位置 | 影响 |
| --- | --- | --- | --- |
| M1 | **无法导出完整预测结果**——`predict` 只返回预览（`preview`），没有把全量预测落库为新数据集版本 | `service.predict` 返回 `{preview:[...]}`；API `limit` 上限 5000（`experiments.py:55`），工具侧无上限 | 真实服务场景"我要用模型给 100 万行打标"无法满足 |
| M2 | **验证比例不可配置**——`test_size` 硬编码 0.2，请求体 `TrainRequest`/`ExperimentCreate` 均无该字段 | `service._execute:268` `PreprocessingPipeline.train_test_split(X, y, seed=, stratify=)` 未传 test_size | 用户无法控制训练/验证划分 |
| M3 | **目标列自动识别非常朴素**——仅匹配命名 `target/label/y/class`，或整数且唯一值≤20 判分类，否则整数多值直接判回归 | `ml_tools.MlDetectTaskTool` | 名为 `churn` 的标签列会被**静默当作聚类**；整型编码的多分类（0–49）会被误判为回归 |
| M4 | **聚类模型对新数据"不能预测"**——DBSCAN 的 `predict` 直接抛异常；KMeans 虽可预测，但统一返回 400/报错，无引导 | `clustering.py:DBSCANAdapter.predict`；`service.predict`/`ml_tools.MlPredictTool` | 非技术用户点"预测"看到报错会认为系统坏了 |
| M5 | **无交叉验证 / 置信区间** | 整个 `experiments` 模块 | 单次切分指标波动大，难以判断模型稳定性 |
| M6 | **解释能力只到特征重要性**——SHAP 已实现（`explainability.explain_shap`）但 `ml.explain` 工具/API 只走 `feature_importance`，且 SHAP 需要样本 `X` 与 `method` 参数，二者均未在接口暴露 | `ml_tools.MlExplainTool`、`experiments.py:explain` | "模型为什么这么预测"的高级解释用不上 |
| M7 | **无模型版本/再训练/漂移检测** | — | 数据更新后无机制提示重训 |
| M8 | **接口契约基本完整但存在不一致**——错误统一走 `MLEngineException(http_status=400)`/`ValidationException(422)`/`NotFoundException(404)`（`ml_engine/exceptions.py`、`core/exceptions.py`），由 `core/middleware.py` 统一兜底；但工具侧 `ml.predict` 的 `limit` 上限与 API 不一致（见 R7） | — | 契约本身 OK，主要是边界一致性问题 |

**存在项（已实现且匹配）：** 训练/评估/预测/解释四件套、模型目录、实验对比、权重持久化、可复现 seed、防泄漏预处理。

**需补充材料（不要臆测，列出待确认）：**
- 正式的 ML 模块 PRD / 需求文档（目标准确率、延迟 SLA、支持的数据规模、预期用户群）。
- 前端 ML 交互规范（predict/explain 在 UI 上如何呈现、是否期望"导出预测文件"）。
- 数据集目标列的**命名约定**文档（决定 M3 是否真的是问题）。
- 部署/多租户安全模型说明（决定是否必须修 R1）。

---

## 3. 缺陷与风险（逐条，含严重程度）

### 🔴 高

**R1 · 对象级鉴权缺失（IDOR / 越权访问）**
- 位置：`api/v1/experiments.py` 的 `predict`/`explain`/`get_experiment`/`get_run`；`tools/ml_tools.py` 的 `MlPredictTool`/`MlExplainTool`；`models/experiment.py` **无 owner_id / tenant 字段**；`ExperimentService` 没有任何用户上下文。
- 问题：端点以任意 `run_id`/`experiment_id` 直接加载，`predict` 仅校验"该数据集是否可访问"，但**不校验该 run 是否属于当前用户**。攻击者只要知道 id，就能读取他人训练好的模型、预测结果、特征重要性、实验详情。
- 最小修复：① 给 `Experiment` 加 `owner_id`（或复用 `Dataset.owner_id`）；② `ExperimentService` 所有读取方法增加 `owner_id` 过滤 / 校验；③ `predict`/`explain` 校验 `run.experiment.owner_id == 当前用户`。
- 说明：项目已明确"不做鉴权登录"（本科毕设取舍），故**当前单用户本地部署下风险不实发**，但这是上线多租户前的**必修项**，建议在论文"局限与展望"与代码注释中明确标注。

**R2 · 全量预测结果不可得（功能性缺陷，影响核心场景）**
- 位置：`experiments/service.py:predict` 仅返回 `preview`；无"把预测写回数据集版本"的能力。
- 问题：见 M1。对"用模型批量打标"这一最典型 ML 诉求无法满足。
- 最小修复：新增 `predict` 参数 `write_back: bool`，将 `prediction`/`probability_*` 列追加回源数据集（新版本）或生成独立预测数据集；或新增导出端点。

### 🟡 中

**R3 · 目标列含缺失值时不自动处理，报底层 sklearn 错**
- 位置：`service._execute` `y = df[target]` 后直接 `model.fit`，未 `drop_nulls()`。
- 问题：标签列有空值 → sklearn 抛 "Input contains NaN" → 被 `run()` 的 `except` 包成失败，错误信息对业务用户不友好；且整批训练失败，而非仅丢弃不完整标签行。
- 最小修复：训练前 `mask = y.is_not_null()`，对 X、y 同时按 mask 过滤并给出"已剔除 N 行空标签"的提示；或在 `preprocessing` 阶段对目标列也提供策略。

**R4 · 用户自定义部分预处理时，特征 NaN 会泄漏进模型**
- 位置：`service._execute:250` `pp_cfg` 仅当**完全为空**才套用 `default_preprocessing_config`（含缺失填补）；若用户传了 `{"scaling":{...}}` 但漏了 `missing`，含 NaN 的数值列会直接进 `StandardScaler` 报 NaN。
- 最小修复：无论是否用户提供，缺失列都按"有缺失则必填补"兜底；或在 `build_pipeline` 内对仍含 NaN 的列强制插入 imputer 并告警。

**R5 · `test_size` 不可配置**（见 M2）
- 最小修复：`TrainRequest`/`ExperimentCreate` 增加 `test_size: float = 0.2`（`gt=0,lt=1`），透传到 `Experiment.parameters` 或独立字段，`_execute` 调用 `train_test_split(..., test_size=exp.test_size)`。

**R6 · 目标列自动识别误判**（见 M3）
- 最小修复：检测不到约定命名列时，**不要静默降级为聚类**；改为要求显式 `target`，或在 `detect_task` 返回 `needs_target=True` 让上层（Agent/UI）引导用户指定；整型目标列增加"是否为 id 列"的启发式（唯一值接近行数则排除为标签）。

**R7 · Agent 工具 `ml.predict` 的 `limit` 无上限，与 API 不一致**
- 位置：`ml_tools.MlPredictTool:331` `int(params.get("limit") or 20)` 无上限；API 有 `le=5000`（`experiments.py:55`）。
- 问题：Agent 路径可要求任意大预览 → `df.head(limit).to_dicts()` 全量物化，内存放大。
- 最小修复：工具侧也钳制 `limit` 到与 API 一致（如 5000），或抽出共享常量。

**R8 · 推理对全量数据做变换再截断预览，浪费资源**
- 位置：`service.predict` 先 `pipeline.transform(X_raw)` + `model.predict(X)` 跑**全表**，再只返回前 `limit` 行。
- 问题：百万行推理时 CPU/内存被全量占用；`proba.to_dicts()` 也只在 `head(preview)` 物化（这点是好的），但预测本身无此优化。
- 最小修复：仅对 `df.head(limit)`（或分批）做 transform/predict；或在大数据量时改为流式/分批推理。

**R9 · 回归任务常量目标 / 单类未校验**
- 位置：`service._preflight` 仅对分类做 `n_classes >= 2` 校验，回归无 `y` 分布校验。
- 问题：常量目标回归会得到误导性指标（r2=0 / 与 sklearn 警告）；无明确报错。
- 最小修复：`_preflight` 回归分支加 `y.n_unique() < 2 → 提示"目标无变化，无法学习"`。

**R10 · `excluded_columns` 寄生在 `preprocessing` JSON 内（可维护性风险）**
- 位置：`service.create:98` 把 `excluded_columns` 写进 `exp.preprocessing` 字典；`_execute:227` 再解析出来。
- 问题：任何"直接读 preprocessing 配置"的代码（如未来前端回显、审计）都会被 `excluded_columns` 污染；属于为规避迁移而引入的隐式耦合。
- 最小修复：为 `Experiment` 增加独立 `excluded_columns` 列（迁移一次），或在 `preprocessing` 内用显式子键 `{"_meta": {"excluded_columns": [...]}}` 明确隔离。

### 🟢 低

**R11 · 模型/管道序列化无版本与格式防护**
- 位置：`base.py:to_bytes`（pickle）、`service._load_model`/`_load_pipeline`（pickle.loads）。
- 问题：pickle 依赖 sklearn 版本，升级后旧 `model.pkl` 可能反序列化失败；无版本戳记录。
- 最小修复：改用 `joblib` 并随 artifacts 记录 `sklearn.__version__`；加载时版本不匹配给出明确提示而非崩溃。

**R12 · `artifacts` JSON 内联大矩阵，可能撑大 DB 行**
- 位置：`service._execute:291` artifacts 内联 `confusion_matrix`（`O(k²)`）、`per_class`、`residual_stats`、`model_summary`。
- 问题：类别多时单 run 的 JSON 体积可观；多 run 累积。
- 最小修复：将大矩阵外置 Storage，`artifacts` 仅存 key。

**R13 · SHAP 解释不可达**（见 M6）
- 最小修复：`ml.explain` 增加 `method` 与可选 `sample_dataset_id` 参数，API `/ml/explain` 暴露同样参数。

**R14 · DBSCAN 轮廓系数在含噪声标签（-1）时可能告警**
- 位置：`evaluation.evaluate_clustering` 仅用 `k<2` 守卫，`silhouette_score` 对 -1 噪声簇可能 warning/异常。
- 最小修复：计算前剔除 -1 簇或捕获 `ValueError` 置 `silhouette=None` 并附说明。

**R15 · 训练为同步阻塞，单 worker 下长任务拖垮服务**
- 位置：`service.run` 在请求内同步执行（含 fit）；项目约定"必须单 worker"。
- 问题：大模型/大数据训练期间该 worker 无法响应其他请求；结合 R8 更明显。
- 最小修复（中长期）：改为后台任务 + 进度轮询（run.status: pending→running→success/failed 已就绪，只需接异步执行）。

**R16 · 失败 run 的成功路径日志 OK，但无"谁触发"审计**
- 位置：`service.run` 记 run_id/task/model/seed/metrics，但无用户/会话维度（与 R1 同源）。
- 最小修复：接入请求上下文的 user/session（依赖 R1 的 owner 体系）。

---

## 4. 用户视角挑刺（真实使用者 / 非技术用户）

| 场景 | 用户痛点 | 根因 | 对应问题 |
| --- | --- | --- | --- |
| 上传 CSV，标签列叫 `churn` | "我明明是要预测流失，系统却给我做了聚类，毫无意义" | 目标列命名约定识别不到 → 静默降级聚类 | R6 / M3 |
| 目标列有空值 | "训练失败：Input contains NaN"（看不懂） | 空标签行未剔除，底层错直接暴露 | R3 |
| 训练完点"预测" | "只看到前 20 行，我的 10 万行预测在哪？" | 只返回预览，无导出 | R2 / M1 |
| 选了 DBSCAN 点预测 | "DBSCAN 不支持对全新样本预测" | 聚类模型预测语义不清 | R4(文) / M4 |
| 大数据集（百万行）训练 | "界面卡死/转圈很久，像崩了" | 同步阻塞单 worker | R15 |
| 基线逻辑回归拟合非线性数据，准确率很低 | "这 AI 好笨，预测不准" | 无调优/CV，仅基线模型 | M5 |
| 服务重启 / Storage 被清 | "模型产物文件已丢失，请重新训练" | 模型仅存本地 pickle，无版本/备份 | R11 |
| 用新数据集推理，少了一列 | "推理数据缺少训练时的特征列" | 列不匹配报错生硬 | —（已有明确报错，体验尚可，但缺自动对齐/提示） |

---

## 5. 优化方案（按优先级，可落地）

### 短期修复（P0/P1，低风险、改造成本小）
1. **目标列稳健识别 + 显式确认**（R6）：检测不到约定列时不再静默聚类，改为要求/引导指定 target；`MlDetectTaskTool` 增加 `needs_target` 信号。涉及 `ml_tools.MlDetectTaskTool`、`MlTrainTool`。
2. **空标签行自动剔除 + 友好提示**（R3）：`service._execute` 训练前按 `y.is_not_null()` 过滤并提示剔除行数。
3. **缺失填补兜底**（R4）：`build_pipeline` 对仍含 NaN 的列强制插 imputer，避免用户漏配导致 cryptic 失败。
4. **暴露 `test_size`**（R5/M2）：请求体 + `Experiment` + `_execute` 透传。
5. **统一 `limit` 上限 + 预览惰性变换**（R7/R8）：抽取共享上限常量；`predict` 仅对所需行做 transform/predict。
6. **聚类预测引导**（M4）：DBSCAN 在 predict 时返回训练集 `labels_` 并明确说明"新数据预测不支持"，而非裸报错；KMeans 正常预测。

### 中期重构（P2）
7. **全量预测导出**（R2/M1）：`predict(write_back=True)` 将预测列落库为新数据集版本或独立预测集；或新增导出端点。
8. **安全加固**（R1）：`Experiment.owner_id` + Service/API 层 owner 校验；pickle → joblib + sklearn 版本戳（R11）。
9. **大 artifacts 外置**（R12）、**SHAP 接口可达**（R13）、**DBSCAN 轮廓系数守卫**（R14）。
10. **异步训练 + 进度轮询**（R15/R16）：`run.status` 状态机已具备，仅需把执行移出请求线程并暴露进度。

### 中长期（P3）
11. **交叉验证 + 置信区间**（M5）。
12. **模型注册/版本化、再训练与漂移检测**（M7）。
13. **超参搜索 / 自动模型选择**（AutoML 雏形）。

---

## 6. 风险速查表（一页纸）

| ID | 问题 | 严重度 | 存在/缺失 | 最小修复 |
| --- | --- | --- | --- | --- |
| R1 | 实验/run 对象级鉴权缺失（IDOR） | 🔴 高 | 存在（设计上暂不含鉴权） | 加 owner_id + Service/API 校验 |
| R2 | 全量预测不可导出，仅预览 | 🔴 高 | 缺失 | predict 支持 write_back / 导出端点 |
| R3 | 目标列空值不剔除，裸报错 | 🟡 中 | 存在 | 训练前 drop_nulls + 提示 |
| R4 | 部分预处理配置时 NaN 泄漏 | 🟡 中 | 存在 | build_pipeline 强制缺失兜底 |
| R5 | test_size 不可配置 | 🟡 中 | 缺失 | 请求体 + 透传 |
| R6 | 目标列识别误判/静默降级 | 🟡 中 | 存在 | 无法识别时要求显式指定 |
| R7 | 工具侧 predict limit 无上限 | 🟡 中 | 存在（与 API 不一致） | 共享上限常量 5000 |
| R8 | 全量变换后再截断预览 | 🟡 中 | 存在 | 仅变换所需行 |
| R9 | 回归常量目标未校验 | 🟡 中 | 缺失 | _preflight 加 y 分布校验 |
| R10 | excluded_columns 寄生 preprocessing | 🟢 低 | 存在（可维护性） | 独立列或显式子键隔离 |
| R11 | pickle 无版本/格式防护 | 🟢 低 | 存在 | joblib + 版本戳 |
| R12 | artifacts 内联大矩阵 | 🟢 低 | 存在 | 大矩阵外置 Storage |
| R13 | SHAP 解释不可达 | 🟢 低 | 缺失 | 暴露 method + sample 参数 |
| R14 | DBSCAN 轮廓系数噪声标签 | 🟢 低 | 存在（边缘） | 剔除 -1 / 捕获异常 |
| R15 | 同步阻塞训练拖垮单 worker | 🟢 低 | 存在 | 异步执行 + 进度轮询 |
| R16 | 无"谁触发"审计 | 🟢 低 | 缺失 | 接入 user/session（依赖 R1） |

---

## 7. 需补充的材料（避免臆测）

1. **ML 模块 PRD / 需求文档**：目标准确率、延迟 SLA、支持数据规模、目标用户画像。
2. **前端 ML 交互规范**：predict/explain 的预期呈现，是否要求"导出预测文件"。
3. **数据集目标列命名约定**：判断 M3/R6 是否为真问题。
4. **部署与多租户安全模型**：决定是否必须实施 R1（当前项目明确不做鉴权，故 R1 作为上线前必修项记录）。
