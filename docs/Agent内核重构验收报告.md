# xiaoluo-lab Agent 内核重构 · 验收报告

> 目标：把现有 Agent 改造成一个统一的 Agent 内核，逐步增强「本地理解 → 决策 → 执行」，
> 用真实数据与接口支撑未来的训练 / 微调 / 模型替换。**Phase 1 目标是架构与数据闭环正确，不是一步训练出完美本地 Agent。**
>
> 生成时间：2026-09-24。本文回答验收要求的 14 个问题；未完成项一律显式标注「未完成」。

---

## 0. 一句话结论

本地 Router（TF-IDF + LinearSVC）已作为**统一决策入口**真正接入运行链路（不再只做 shadow 埋点）；新增了 TaskSpec / DecisionProvider / DecisionTrace 三层最小抽象；本地推理模型（Qwen 预留）与远程升级路径已接通但保持「诚实不可用 / 按需一次调用」的边界；DecisionTrace 已形成「写 → 读 → 导出训练数据」的闭环。**尚未做**：真实 Qwen 本地生成模型接入、以及把 DecisionTrace 数据真正喂进一次训练（那是下一阶段，不属 Phase 1）。

---

## 1. Phase 0 基线数字（可复现）

| 方案 | route accuracy | tool accuracy | 语义闸门 | 说明 |
| --- | --- | --- | --- | --- |
| TF-IDF + LinearSVC（groupCV 泛化真值） | **80.7%** | **80.2%** | 96.6% | 本环境可复现（954 条） |
| TF-IDF + LinearSVC（holdout 样本级·虚高） | 92.1% | 91.5% | 100.0% | 仅作参照，不采信 |
| Neural L1（+规则，frozen） | 79.5% | 79.0% | 91.5% | 需 torch，本环境无，引用历史报告 |
| Fusion（α=0.5，frozen） | **83.6%** | 83.3% | 96.6% | 需 torch，本环境无，引用历史报告 |

- 高置信度阈值扫描（TF-IDF groupCV）：θ=0.3 → precision 85.1% / coverage 94.2%；θ=0.5 → 94.1% / 78.6%；θ=0.7 → 97.4% / 58.8%。
- 无谓升级率 0.0%，漏升级率 3.4%。
- EDA 子意图混淆矩阵（196 条）：distribution 全对（42/42），describe 与 profile 有轻微互混，correlation 有 3 条漏到「其他」。
- **评估集已冻结**：`backend/scripts/router/eval_frozen/`（held_out_eval 954 条 sha256[:16]=`6a34321907e5147c`；real_user_logs 315 条 `6dde1a2fad725dda`）。真实日志去重后仅 5 种独特请求，多样性不足，故 held_out_eval 为主评估集、real_user_logs 仅作补充。

---

## 2. 修改 / 新增的文件清单

**新增（架构层）**
- `backend/app/agent/task_spec.py` — TaskSpec / TaskDomain / TaskScope / Complexity / TaskUnderstanding（静态任务描述）。
- `backend/app/agent/task_spec_builder.py` — 本地理解入口（intent 分类 + LocalModelDecisionProvider → TaskSpec）。
- `backend/app/agent/decision/provider.py` — DecisionAction / DecisionSource / AgentDecision / DecisionContext / DecisionProvider（唯一决策抽象）。
- `backend/app/agent/decision/rule.py` — RuleDecisionProvider（仅确定性命令 / 必填参数 / 边界）。
- `backend/app/agent/decision/local_model.py` — LocalModelDecisionProvider（包装 `local_router.route_request`）。
- `backend/app/agent/decision/remote.py` — RemoteLLMDecisionProvider（一次性战略指导）。
- `backend/app/agent/decision/router.py` — DecisionRouter（Rule → LocalModel → Remote 合成）。
- `backend/app/agent/trace/decision_trace.py` — DecisionTrace + `write_trace` / `iter_traces`（JSONL 追加写）。
- `backend/app/agent/llm/local.py` — LocalReasoningProvider（本地生成模型接口，Qwen 预留）+ FakeLocalReasoningProvider。
- `backend/scripts/router/export_decision_trace.py` — DecisionTrace → 训练数据导出（router_train / tool_selection / task_understanding / decision_pairs）。
- `backend/scripts/router/baseline_phase0.py` — Phase 0 统一基线 + 评估集冻结。

**修改（接入现有链路）**
- `backend/app/agent/runtime/runtime.py` — 新增 `decision_router` 字段、`_understand`、`_escalate`、`_begin_trace`/`_end_trace`；`run()` 与 `_agent_turn()` 接入。
- `backend/app/agent/runtime/models.py` — `AgentTokenLedger` 新增 `remote_escalations` + `record_escalation()`；`to_dict`/`_load` 对称。
- `backend/app/analysis.py` — 新增 `DistributionOverviewAnalyzer`（数据集级分布总览 + signals）。
- `backend/app/tools/eda_tools.py` — 新增 `EdaDistributionOverviewTool`（`eda.distribution_overview`）+ `_extract_signals`。
- `backend/app/tools/result.py` — `ToolResult` 新增 `signals` 字段。
- `backend/app/tools/builtin.py` — 注册 `EdaDistributionOverviewTool`。
- `backend/app/local_router/contract.py` — `_NON_ROUTABLE_CATEGORIES={"agent"}`、`is_routable_tool()`、`tool_label_space()` 排除内部工具、connector → DATASET 映射。
- `backend/app/agent/llm/__init__.py` — 导出 LocalReasoningProvider / FakeLocalReasoningProvider。
- `backend/app/core/config.py` — 新增 `LOCAL_REASONING_MODEL_ENABLED/NAME/PATH`（默认关）。
- `backend/scripts/router/utterance_templates.py` / `build_dataset.py` — 补齐 connector 工具模板与 `connector_id`/`table` 槽位。

**新增测试**
- `tests/test_agent_task_spec_decision.py`（11）、`test_agent_local_reasoning.py`（7）、`test_agent_remote_escalation.py`（10）、`test_agent_decision_trace_export.py`（6）。

---

## 3. 删除了哪些重复逻辑

- **删掉了「五套 Provider 并行」的隐患**：DecisionProvider 只建了 Rule / LocalModel / Remote **三层一个抽象**，未建 SemanticProvider / ReasoningProvider / PlanningProvider / StrategyProvider。
- **不再用关键词规则做「分布→xxx」语义映射**：RuleDecisionProvider 的职责被**严格限定**在确定性命令 / 必填参数 / 边界，语义工具选择全部交给本地模型。
- **没有新建 ExecutionState**：可变执行状态继续落在 `AgentContext`，TaskSpec 只做静态描述，不重复承载执行进度。
- **`agent.clarify` 从 Router 标签空间剔除**（`_NON_ROUTABLE_CATEGORIES`）：它是内部反问通道，用户从不请求「问我一个问题」，避免污染标签空间。
- 未引入第二个 Tool Registry / 第二个 Agent Runtime：决策层复用 `TOOL_REGISTRY`，执行层复用 `AgentExecutor`。

---

## 4. TF-IDF / Neural / Fusion 各自在哪参与

| 模型 | 参与位置 | 是否真正运行 |
| --- | --- | --- |
| TF-IDF + LinearSVC | `LocalModelDecisionProvider` → `local_router.route_request`；`TaskSpecBuilder` 据此选工具 / 定 domain / 复杂度 | ✅ 真正运行（本环境可加载 `.pkl`，实测 80.7%） |
| Neural L1（chroberta） | 保留在 `local_router` 的历史报告与训练脚本中 | ⚠️ 需 torch，本环境未加载（诚实标注 frozen，不假称运行） |
| Fusion（词法+神经 α 融合） | 保留在 `analyze_fusion.py` 历史结论 | ⚠️ 需 torch / ONNX，未在运行期接入 |

> 关键诚实性：**没有把「接口接上了」说成「模型在跑」**。TF-IDF 是唯一在运行期真正参与决策的本地模型；Neural/Fusion 因环境无 torch 而冻结在历史报告，未伪装运行。

---

## 5. Decide 到底执行了什么

`DecisionRouter.route(context, allow_remote)` 是「这一步怎么走」的**唯一入口**，按优先级合成：

1. **RuleDecisionProvider**（零模型调用）：命中的「取消/停止命令」「必填参数缺失反问」「确定性边界」直接返回（confidence=1.0）。
2. **LocalModelDecisionProvider**：真正调用 `route_request`，拿本地模型的工具选择 + 置信度 + 候选；升级 / 反问 / 闲聊 / 执行四类动作各返回一次 `AgentDecision`。
3. **RemoteLLMDecisionProvider**（仅在本地升级且 `allow_remote=True` 且有 llm 时）：一次性战略指导，产出 `evidence.remote_guidance`，**不控制工具执行循环**。

每一步都带 `source + confidence + evidence`，全部进 DecisionTrace。

---

## 6. RuleDecisionProvider / LocalModelDecisionProvider 各自职责

- **RuleDecisionProvider**：**确定性**边界——取消/停止命令、必填参数（dataset_id）检查、超限终止。**不做语义映射**（不猜「分布→xxx」）。
- **LocalModelDecisionProvider**：把用户语言映射到「工具 + 置信度 + 候选」，真正复用 `local_router` 的三层合成（L0 升级规则 → L1 词法模型 → 反问规则）。模型不可用 / 过期 / 加载失败时**显式返回 confidence=0 的保守升级并写明原因**，不静默吞错。

---

## 7. 什么时候走 Remote

仅在**本地确实不足**时触发，且只做**一次**战略指导（`runtime._escalate`）：

- 触发条件：`context.task_context["task_spec"]["complexity_hint"] == "complex"`（由本地 Router 判定升级，而非 `_escalate` 自行猜）。
- **简单 / 一般任务零远程调用**（PoC ① ② 已验证）。
- 远程失败走 `fallback_allowed` 同口径：允许降级 → 保留本地结论继续；不允许 → 原样抛出，让运行如实失败。**不静默吞成「已升级」**。

---

## 8. TaskSpec 与 AgentContext / Plan / Decision 的关系

| 结构 | 职责 | 可变性 |
| --- | --- | --- |
| Intent | 粗粒度入口分类（兼容层） | 静态 |
| **TaskSpec** | 用户要完成什么（goal / domain / sub_goal / scope / entities / complexity） | **静态**描述 |
| AgentContext | 当前执行上下文（数据集、工具、已答反问、remote_guidance） | **可变状态** |
| Plan | 准备执行的动作集合（steps） | 一次运行内可变 |
| Decision | 某一步**为什么**选这个动作（value / source / confidence） | 每步一条 |

TaskSpec 只描述任务、不执行、不存进度；执行状态归 AgentContext，不新建 ExecutionState。

---

## 9. DecisionTrace 里有什么

`decision_trace.jsonl` 每条记录（任务 8 最小字段集 + 可观测字段）：

```
request → task_spec → router_candidates → decision_made → decision_source
→ tool_calls → tool_results_summary（含 signals）→ final_answer → answer_source
→ success → remote_calls / remote_escalations / local_calls → latency_ms
→ decision_steps[]（Observe→Decide→Act 每一跳）
```

导出脚本 `export_decision_trace.py` 据此产出 4 类训练数据：`router_train.jsonl`、`tool_selection.jsonl`、`task_understanding.jsonl`、`decision_pairs.jsonl`（只导「前一步带 signals」的 Observe→Decide 对）。

---

## 10. 4 个 PoC 结果

| PoC | 场景 | 结果 |
| --- | --- | --- |
| ① 数据集级分布总览 | 「看看这批数据的分布」 | ✅ `eda.distribution_overview`，scope=dataset，无 column，complexity=simple，source=local_router，**remote_calls=0** |
| ② 开放式基础分析 | 「帮我分析这份数据的整体情况」 | ✅ `dataset.profile`，scope=dataset，source=local_router，零远程 |
| ③ 多步动态任务（≥2 次 Decision） | 「先看分布→发现缺失严重就清洗→训练」 | ✅ DecisionTrace 记录 4 跳决策（understand→act→observe(signals)→act），含 signals 驱动的下一步 |
| ④ 远程升级 | 本地无把握 + 有 llm | ✅ `_escalate` 发起一次战略指导，`remote_escalations=1`，guidance 挂到 context，工具仍本地执行 |

---

## 11. 改造前后远程调用 / Token

- **改造前**：本地 Router 只在 `LOCAL_ROUTER_MODE=shadow` 下「只记录不改变行为」，远程 LLM 是默认大脑（规划 + 汇总都走远程）。
- **改造后**：本地 Router 真正参与理解与工具选择；简单 / 一般任务 `remote_escalations=0`，只有 `complexity=complex` 才发起**一次**战略指导（且远程失败允许降级）。
- 新增独立口径 `remote_escalations`（区别于 `llm_calls`），用于精确衡量「升级类远程调用」是否下降。**当前因真实日志去重后仅 5 种请求、无复杂任务样本，未积累到可对比的线上 before/after 数据 —— 未完成**。

---

## 12. 回归情况

- 新增 34 项测试全部通过；既有决策/路由/契约测试（65 项合计）通过。
- `AgentTokenLedger` 新增 `remote_escalations` 已做 `to_dict`/`_load` 对称，持久化往返验证通过。
- 核心 agent 链路（turn_context / wait_states / step_resolution / tool_first / token_optimization / config，25 项）通过。
- 未跑 `test_agent.py` / `test_phase10_integration.py` 等**真实调 LLM** 的用例（会烧 API，且与本改动无直接关系），属已知范围外。

---

## 13. 训练数据来源（Phase 5 数据出口）

1. **held_out_eval（954 条，groupCV）**：模板合成，有 gold label，主评估集（已冻结）。
2. **real_user_logs（315 条，去重 5 种）**：真实分布但多样性不足，补充。
3. **DecisionTrace（运行期新产生）**：完整决策链，导出为 router_train / tool_selection / task_understanding / decision_pairs 四类。
4. **`harvest_traces.py`**：从 AgentStore 挖「真实执行成功」的单步 Router 正样本（与 DecisionTrace 互补）。

> 数据出口已建好，但**尚未跑一次真实训练**去消费这些数据（Phase 1 只要求「闭环正确」，不要求训练）—— 见「未完成」。

---

## 14. 未完成项（如实标注）

1. **真实 Qwen 本地生成模型接入**：`LocalReasoningProvider` 目前是「诚实不可用」占位（`is_available()` 恒 False），未加载真实权重。仅有接口 + 最小实现 + Fake 替身。
2. **DecisionTrace 数据喂进一次训练**：导出脚本已就绪，但未执行训练消费。
3. **线上 before/after 远程调用对比**：真实日志样本不足（去重 5 种、无复杂任务），无有效对比数据。
4. **Neural / Fusion 在运行期接入**：本环境无 torch，二者仍冻结在历史报告，未在运行期运行。

---

## 附：关键约束遵守情况

- ✅ 没有第二个 Agent Runtime / 第二个 Tool Registry / 重复抽象。
- ✅ Observe→Decide→Act 无大段 if-else（决策收敛到 DecisionRouter 三路合成）。
- ✅ 不新增语义关键词规则（语义交给本地模型）。
- ✅ 本地模型不降级为装饰（TF-IDF 真正运行）。
- ✅ 不把「接口接上」谎称「模型在跑」。
- ✅ 未破坏 SSE / Run / activeRunId / confirmation（未改 store 锁与等待态语义）。
- ✅ 本地模型错误不静默吞（显式 confidence=0 升级 + 写明原因）。
- ✅ 不强制远程处理简单任务。
