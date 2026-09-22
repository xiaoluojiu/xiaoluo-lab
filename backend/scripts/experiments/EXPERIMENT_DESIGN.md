# AI 全链条接入 · 实验设计与复现说明

> 配套脚本：`agent_chain_benchmark.py`
> 结果产物：`results/agent_chain_<时间戳>.json`（原始逐次）+ `.md`（可直接贴论文的汇总表）

## 1. 这套实验要回答的问题

创新点是「AI 的全链条式接入」，评委一定会追问一句：**凭什么说接入 AI 有用？**
只答「它能跑通」是不够的，需要三样东西：

1. **对照组**：没有 AI 规划时，同一批任务能做到什么程度；
2. **量化指标**：不是主观评价，而是能从工具返回值里复算出来的数字；
3. **消融实验**：把 AI 接入带来的各项能力逐个关掉，看哪一项真正起作用。

## 2. 实验设计

### 2.1 任务集（6 个固定任务，覆盖全链条）

| 任务 | 环节 | 期望工具环节（`a\|b` 表示任一命中即算覆盖） |
| --- | --- | --- |
| T1_quality | 数据质量 | `dataset.quality \| dataset.profile` |
| T2_eda | 描述统计与相关性 | `eda.describe`、`eda.correlation` |
| T3_ml | 建模与评估 | `ml.detect_task \| ml.prepare`、`ml.train`、`ml.evaluate` |
| T4_merge | 多表关联 | `data.merge` |
| T5_report | 分析报告 | `report.generate` |
| T6_workflow | 工作流编排 | `workflow.build_and_run \| workflow.create` |

所有任务使用 `data/demo` 下的真实演示数据（users 200×5 / orders 500×7 / events 1000×5），
环境为内存 SQLite + 临时存储目录，与 `learning_cases.py` 相同的隔离方式，可重复执行、不污染开发库。

### 2.2 配置组（1 个对照 + 3 个消融）

| 配置 | 含义 | 实现方式 |
| --- | --- | --- |
| `rule` | **对照组：无 AI 规划** | `llm=None`，走 `AgentPlanner._rule_plan` 规则规划器 |
| `full` | LLM 规划 + 全特性 | 默认配置 |
| `no_retrieval` | 消融：关闭工具检索 | `AGENT_ENABLE_TOOL_RETRIEVAL=False`（候选工具从检索改为全量下发） |
| `no_compression` | 消融：关闭结果压缩 | `AGENT_ENABLE_RESULT_COMPRESSION=False` |
| `no_narration` | 消融：关闭报告 LLM 叙述 | `AGENT_REPORT_NARRATION=False` |

⚠️ 实现细节（曾踩坑）：`report.generate` 内部走 `settings` 里的 Provider（`default_provider()`），
与 planner 的 `llm` **不是同一个开关**。跑对照组 / `--quick` 自检时必须额外把
`AGENT_REPORT_NARRATION` 关掉，否则会在「声称不消耗 token」的场景下偷偷调用真实 LLM。

## 3. 指标口径（写进论文时必须原样保留）

| 指标 | 定义 | 为什么可信 |
| --- | --- | --- |
| **任务完成度 coverage** | 命中的期望工具环节数 / 期望环节总数 | 期望环节在任务定义时人工声明，不依赖 LLM 自评，可逐条复核 |
| **语义正确率** | 通过事实断言的任务数 / 声明了断言的任务数 | 断言直接读工具返回值，不是让模型给自己打分 |
| 工具调用成功率 | `ok / (全部调用 − 授权拦截中间态)` | 授权拦截是中间态不是失败，计进分母会把成功率系统性算低 |
| 总 token | Provider 返回的 `prompt + completion` 真实累计 | 未发生调用即为 0 |
| 估算节省 token | `tools/result.py` 按字符数**本地粗估** | ⚠️ 见下方警告 |

### 3.1 为什么需要「语义正确率」

coverage 只能回答「环节走没走到」，回答不了「做对了没有」。最典型的反例：

> 用户说「用 orders 训练一个**预测 amount** 的模型并评估效果」。
> 规则规划器推断不出目标列 → `ml.detect_task` 判为聚类 → 链路照跑完，**coverage = 100%**，
> 但训练出来的是一个聚类模型，**根本不是用户要的东西**。

所以语义正确率的断言直接读工具返回值：

- `T3_ml`：`ml.detect_task` 识别出的目标列必须确实是 `amount`；
- `T4_merge`：`data.merge` 实际使用的 Join Key 两侧必须都是用户 id。

**这一项才是「AI 有没有真的理解需求」的证据。**

### 3.2 ⚠️「估算节省 token」的口径警告

该值来自 `app/tools/result.py` 的 `for_llm()`，算法是
`_rough_tokens(原始结果) − _rough_tokens(压缩后结果)`，即**按字符数在本地粗估**，
**与是否发生真实 LLM 调用无关**——对照组 `llm_calls=0` 时它依然是正数。

它反映的是「上下文压缩机制少喂了多少字符给模型」，
**不是 Provider 计费口径的节省**。论文中必须标注为本地估算值，
不得与「总 token」直接相减或相除，否则会被质疑虚高。

## 4. 复现命令

```bash
# 在 backend/ 目录下，必须用 venv 解释器
.venv/Scripts/python.exe -m scripts.experiments.agent_chain_benchmark --quick
#   快速自检：只跑 rule 组，零 token 消耗，用于验证环境通不通

.venv/Scripts/python.exe -m scripts.experiments.agent_chain_benchmark --configs rule full
#   完整对照实验（会真实调用 .env 中的 LLM）

.venv/Scripts/python.exe -m scripts.experiments.agent_chain_benchmark \
    --configs full no_retrieval no_compression no_narration
#   消融实验

.venv/Scripts/python.exe -m scripts.experiments.agent_chain_benchmark --repeat 3
#   每组重复 3 次，用于观察计划缓存命中与耗时稳定性
```

## 5. 论文里建议怎么呈现

1. **总体对照表**：`rule` vs `full`，突出「完成度接近、语义正确率拉开差距」——
   说明 AI 的价值不在于「能不能跑」，而在于**是否理解了需求**。
2. **分任务明细表**：逐任务列出调用过的工具链，直观展示「全链条」。
3. **消融对比表**：说明工具检索 / 结果压缩 / 报告叙述各自贡献了什么。
4. **代价维度**：把耗时与 token 一并列出，诚实说明 AI 接入的成本（对照组 0 token、亚秒级，
   全特性组单次约 8k token、几十秒），再论证这个代价换来了语义正确性。

## 6. 本轮实验顺带修掉的功能缺陷

跑这套基准的过程本身就在做健壮性验证，以下是被实验暴露、并已修复的真实缺陷
（详见 `backend/ARCHITECTURE.md` 与各处代码注释）：

| 缺陷 | 表现 | 修复位置 |
| --- | --- | --- |
| `AgentPlanner` 默认 `max_steps=6` | 长链路计划被尾部截断，评估/报告步骤静默丢失 | `agent/runtime/runtime.py` |
| `DATA_TERMS` 漏「关联/拼接/宽表」 | 多表关联请求被当闲聊，只回一句固定话术 | `agent/runtime/runtime.py` |
| 日期/布尔列无法参与训练 | 任何含日期列的数据集训练必失败 | `ml_engine/preprocessing.py` |
| 任务类型与模型不匹配 | `LogisticRegression.fit() missing 1 required positional argument: 'y'` | `tools/ml_tools.py` |
| `data.merge` 的 keys 必填 | 规划阶段拿不到列名，多表关联必然失败 | `tools/data_tools.py`（改为按同名列推断） |
| `{{stepN.field}}` 解析为 None / 字段不存在 | 整步硬失败，重规划耗尽步数 | `agent/runtime/step_resolution.py` |
| 规则规划器不会做关联、不做评估 | 链路缺环节 | `agent/planner/planner.py` |
