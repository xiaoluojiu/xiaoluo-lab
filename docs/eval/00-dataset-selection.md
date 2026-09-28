# 基准数据集选型报告：MMTU vs. data-agent-harbor-test

> 目标：为「小洛实验室」（XiaoLuo Lab）**AI 实验室模块的 AI 回复质量**评测选定基准数据集。
> 评测维度：准确性 / 相关性 / 完整性 / 格式合规性。
> 调研日期：2026-09-27

---

## 0. 被测对象画像（先明确「要测什么」，再谈数据集）

小洛实验室是「数据处理 → 数据探索 → 智能合并 → 机器建模 → Agent 智能分析 → 工作流编排 → 实验报告」一体化平台。
AI 实验室（AI Lab）是一个**对话式数据分析 Agent**：用户消息触发一次 AgentRun，链路为
`决策 → 工具调用 → 校验 → 最终回答`，通过 SSE 事件流实时展示，高风险步骤经用户确认后 resume。

「AI 回复」在系统里只有两个出口（`backend/app/agent/answer.py`）：

| 出口 | 触发条件 | 特征 |
|---|---|---|
| `render_chat()` | 纯对话 / 不涉及具体数据操作的追问 | 无工具结果，靠会话上下文 + 历史 |
| `render()` | 有工具结果（EDA、建模、关系分析…） | 事实摘要 → LLM 润色（1 次调用） |

因此「回复质量」的可观测面是：**给模型一段事实/表格 + 一个用户问题，它产出的中文回答是否准、是否答对所问、是否该说的都说了、是否守格式与红线**。

---

## 1. 候选 A：MMTU（A Massive Multi-Task Table Understanding and Reasoning Benchmark）

- **出处**：NeurIPS'25 论文（arXiv:2506.05587），GitHub `MMTU-Benchmark/MMTU`，数据托管 HuggingFace `MMTU-benchmark/MMTU`
- **规模**：28,136 道题（MMTU 1.0，经质量过滤），源自 52 个源数据集
- **任务覆盖**：**25 类真实表格任务**（已从仓库 `configurations/` 逐一核对）：
  `NL2SQL`、`Table-QA`、`Table-Fact-Verification`、`Error-Detect`、`Data-transform-pbe`、`Entity-Matching`、
  `Table-needle-in-a-haystack`、`Table-Locate-by-Row-Col`、`Schema-Matching`、`Data-transform-reshape`、
  `Data-Imputation`、`List-to-table`、`Formula-prediction-context`、`Transform-by-output-target-schema`、
  `Transform-by-input-output-table`、`semantic-transform`、`semantic-join`、`header-value-matching`、
  `Arithmetic-Relationship`、`Functional-Dependency`、`String-Relationship`、`Cell-entity-annotation`、
  `Column-type-annotation`、`Columns-property-anotation`、`equi-join-detect`
- **评测方式**：每类任务一个确定性 evaluator（`evaluators/`，共 27 个），指标为 `acc` 或 `f1`（`evaluate.py` 的 `summary_metric` 明确规定）。**无 LLM-as-judge**
- **格式约束**：每类任务的 `prompt_template` 里写死输出格式要求。例（NL2SQL 官方模板原文）：
  > "No explanation, return the code only with the following markdown codeblock format : ```sql<SQL CODE>```"
  > "ensure that the generated SQL only return relevant columns being specifically asked in the question"
- **难度参照**：GPT-5 0.696 / o3 0.691 / DeepSeek-R1 0.597 / DeepSeek-V3 0.555 / GPT-4o 0.507
- **表序列化形态**：同一任务有 csv / html / json / markdown × 0-shot / 3-shot 多套配置

## 2. 候选 B：FineEnvs / data-agent-harbor-test（Data Agent — Harbor test）

- **规模**：250 道留出任务；难度 easy 33 / medium 118 / hard 99（含全部最难 L5）
- **来源**：从 jupyter-agent（Kaggle 数据集上的真实数据科学 notebook）抽取问答对，经「强 agent 在沙箱里复现 gold answer」验证，剔除了歧义项
- **任务形态**：给 agent 一个真实数据文件（落在 `/home/user/input/`）+ 一个问题，agent 用代码执行工具探索、计算，最终把**一个裸值**写进 `/workdir/answer.txt`
- **答案类型**：numeric 132 / short-label 87 / yes-no 29 / csv-list 1 / flexible 1
- **评测方式**：`grader.py` 确定性打分 —— exact → numeric tolerance → list/percent 归一化 → symbolic(math-verify)，返回 1.0 / 0.0。**无 LLM judge、无网络**

---

## 3. 逐维对比

| 对比维度 | MMTU | data-agent-harbor-test | 对本项目的影响 |
|---|---|---|---|
| 任务类型多样性 | **25 类**，覆盖理解/推理/变换/标注/匹配/检测 | 单一形态：读文件→算一个数 | 本评测要求「覆盖典型任务类型」，MMTU 天然满足；harbor 只有 1 类 |
| 输出形态多样性 | SQL / Python 代码 / 单值 / 标签 / 集合 / JSON，各类格式要求不同 | 恒为**一个裸值** | 本评测要测「格式合规性」，harbor 的裸值输出**没有格式可测** |
| 能否测「完整性」 | 能：7 类任务用 **f1** 指标（Error-Detect、Schema-Matching、semantic-join、Arithmetic-Relationship、Functional-Dependency、String-Relationship、equi-join-detect），天然考核「该列的都列全了吗」 | 不能：单值答案不存在漏项概念 | harbor 在完整性维度上是**结构性缺失** |
| 能否测「相关性」 | 能：官方模板明令「只返回被问到的列」「不要 SELECT *」，答非所问会被执行结果判死 | 弱：单值对/错不区分是否跑偏 | MMTU 有显式的相关性约束条款 |
| 与项目领域契合 | 表格 = 平台一等公民（数据处理/EDA/合并/建模全链路都建立在表上）；25 类任务中的转换、匹配、补缺、错误检测正是平台已有能力 | 数据科学问答，契合度也高 | 两者都契合，MMTU 更贴合平台的**表操作**内核 |
| 与「AI 回复」形态契合 | 任务是「给表 + 问题 → 产出回答」，与 `render_chat` 的输入形态一致 | 任务是「沙箱执行 → 写文件」，测的是**执行完成度**而非**回复文本质量** | harbor 更偏 agent 端到端 pass@1，偏离「回复质量」主题 |
| 原始语料可获得性 | ❌ 本体走 OneDrive + HuggingFace，本环境网络对 `huggingface.co` / `cdn-lfs` 全部 502，无法下载 28K 题；✅ 但**仓库代码可克隆**，25 类任务的 prompt 模板与 27 个 evaluator 全部可读 | ❌ 同样不可达（HF 域名被拦），且需要沙箱执行环境 | 两者都拿不到原始数据，但 MMTU 的**任务定义与评分逻辑可完整获得**，足以复刻 |
| 复现成本 | 中：本地造小表 + 用官方 evaluator 逻辑判分即可 | 高：需 Docker 沙箱 + 250 个 Kaggle 数据文件 | MMTU 可行，harbor 在本环境不可行 |

---

## 4. 结论：选定 **MMTU**

**选定 MMTU**。决定性理由按权重排序：

1. **维度覆盖完整**——本次评测的四个维度里，harbor-test 只能支撑「准确性」一项；MMTU 的 25 类任务 + 官方格式条款 + f1 指标可以同时支撑准确性、相关性、完整性、格式合规性四维。
2. **格式合规性可测**——MMTU 每类任务的 `prompt_template` 都写死了输出格式（如 NL2SQL 要求「只返回 ```sql 代码块，不要解释」），这正是格式合规性的天然评分依据；harbor 要求输出裸值，等于没有格式维度。
3. **与平台内核同构**——小洛实验室的一切都建立在「表」上，MMTU 的 25 类任务（转换/匹配/补缺/错误检测/依赖发现/模式匹配）与平台已有的数据处理、智能合并、EDA 能力一一对应，测出来的分能反过来指导真实功能。
4. **可落地**——harbor 需要 Docker 沙箱 + 250 份 Kaggle 数据文件，本环境不具备；MMTU 虽然 28K 原题不可下载，但**任务定义（25 类）与评分逻辑（27 个 evaluator）全部可从仓库源码获得**，足以在本地按官方规格复刻出可判定、有真值 gold 的测试集。
5. **难度有区分度**——前沿模型在 MMTU 上也只有 50%~70%，说明它不是一眼过的题，能把回复质量的差异拉开。

### 4.1 数据不可达的处置（必须写清，不能含糊）

MMTU 的 28K 原始题目托管在 OneDrive / HuggingFace，本环境网络对上述域名返回 **502 Bad Gateway**（已实测：`huggingface.co`、`cdn-lfs.huggingface.co`、`datasets-server.huggingface.co` 全部不可达），因此**无法直接载入官方原题**。

处置方式——**按官方规格本地实例化**，而非编造：

- **任务类型**：完整保留 MMTU 的 25 类，一类不删（从仓库 `configurations/` 目录核对）
- **Prompt 约束**：逐类沿用 MMTU 官方 `prompt_template` 中的格式条款（代码必须用 ```sql / ```python 代码块、只返回被问到的列、不要多余解释等）
- **评分逻辑**：从 `evaluators/*.py` 移植判定口径（acc / f1、数值容差、集合匹配、SQL 执行结果比对）
- **题目与 gold**：本地构造小型表格（固定随机种子），gold answer **由程序执行得出**（pandas / SQLite 实跑），不是人工拍脑袋填的期望值 —— 保证 gold 是真值、可复算
- **边界用例**：在 MMTU 25 类之外补充本项目 AI 实验室特有的边界场景（幻觉列名、空表、超长表定位、多轮追问、失败步骤不得包装成计划、无工具不得宣称已执行等）

这套实例化基准命名为 **MMTU-Lab**（MMTU-derived, Lab-instantiated），下文所有轮次均以其为准。

---

## 5. 引用

- MMTU 论文：arXiv:2506.05587 — *MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark*
- MMTU 代码：`https://github.com/MMTU-Benchmark/MMTU`（本次已克隆至本地用于提取任务定义与评分逻辑）
- FineEnvs / data-agent-harbor-test：`https://huggingface.co/datasets/FineEnvs/data-agent-harbor-test`
