# AI 实验室 Agent 架构级减法重构 — 需求规格（spec.md）

## Overview
- **Summary**：把当前 AI 实验室后端 Agent 的「Chat/Agent 硬分流 + 三套规划入口 + 两条执行引擎 + 四层 Provider」收敛为**一个统一的 Stateful Agent Loop**（Observe → Decide → Execute/Chat/Ask/Escalate → Update State → Continue/Finish）与**一个统一 TaskState**，形成「确定性工具 / Qwen 低成本决策 / Remote LLM 按需升级」三层资源调度，增量决策、动态复杂度，并把 Token 成本作为一等指标。
- **Purpose**：消除重复路由/规划/fallback/上下文构造/结果总结，让简单任务 Remote 调用为 0、复杂任务仍可完成且 Remote 不接管执行；降低代码量与分支数，同时支撑论文主线「面向数据分析 Agent 的分层决策与动态资源调度机制」。
- **Target Users**：本科毕设演示与论文实验；平台单机部署的数据分析用户。

## Goals
- 一个 Agent Loop、一个 TaskState；Chat 只是 Loop 在任意时刻可采取的动作，不再存在 Chat Flow / Agent Flow 两个系统。
- 决策三层清晰：确定性层（工具/剧本/信号/校验/权限/Pre-flight）→ 本地模型层（local_router：TF-IDF/Qwen，仅低成本下一步决策）→ Remote LLM 层（结构化、限次、仅在不确定/复杂/冲突/失败时介入，控制权立即回到本地 Loop）。
- 增量决策：每步 Observe 真实结果再决定下一步；仅复杂任务允许把 2–4 步的**阶段性动作队列**交给本地执行，禁止「LLM 一次性 1–20 步大规划」成为默认路径。
- Token 一等公民：每次 Run 记录 remote_calls / remote_input_tokens / remote_output_tokens / qwen_calls / tool_calls / task_steps / escalation_count / total_cost 等；紧凑状态、按需工具面、不重复全量上下文、无意义总结调用归零。
- 复杂度是动态状态：随事实、信号、失败次数、置信度、用户追加要求变化，可在任务中途升级或降级。
- 支持任务跨多轮消息持续：分析中追问（读状态作答）、改要求（更新约束/待办）、新任务（重置状态）。
- 架构减法落地：删除职责重叠模块，重构后 `app/agent`（不含 local_router）代码行数与决策分支数下降。

## Non-Goals
- 不引入 LangGraph/OpenAI Agents/ADK 等任何新框架，不新建第二套 Agent 系统。
- 不重写 Tool Registry（36 工具）、Data Engine、ML Engine、Workflow Engine、Permission、Confirmation、Preflight、Validator、SSE、Agent Trace、Session、Error handling——它们作为执行基础设施保留，改为服务统一 Loop。
- 不做用户认证、多 worker 改造、AgentStore 迁移数据库、workflow/report 持久化改造。
- 不强制启用 Qwen（torch 仍为可选依赖，默认行为不依赖 GPU/大权重加载）。
- 不改动 AI 实验室以外的前端页面；LearningWorkspace/LearningCard 两个旁路 SSE 消费者不在本轮（另行处理）。
- 不追求 `intent.py` 关键词表本身的语义升级；它降级为确定性层的**信号**之一，不再决定走 Chat 还是 Agent。

## Background & Context

### 现状真实调用链（2026-09-25 审查取证，重构基线）
入口 `POST /agent/sessions/{id}/messages` → [agent.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/app/api/v1/agent.py) → [runtime.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/app/agent/runtime/runtime.py)（1205 行）：

```text
run()
 ├─ _route()                    intent.classify 关键词 → 硬分 chat / agent   ★删除
 ├─ chat: _direct_chat()
 └─ _agent_turn()
     ├─ ContextBuilder.build（仅元数据）
     ├─ _understand()           TaskSpecBuilder → DecisionRouter(rule→signal→local_model→remote)
     ├─ _escalate()             complexity=complex 时一次远程自由文本指导（实际仅拼进 goal 字符串）
     ├─ 空数据集提示 / Pre-flight 反问
     ├─ 工具检索（零 Token 召回 + 意图域注入 + CORE_TOOLS）
     ├─ _local_direct_plan()    六道闸门                            ┐
     ├─ _task_spec_plan()       单步兜底                            ├─ 三套规划入口 ★合并
     └─ _build_plan()           AgentPlanner：LLM 一次性大规划       │
                                （三级降级）/ rule_plan 关键词链     ┘
     执行二选一：
     ├─ _run_dynamic()          Observe-Decide-Act（signals 驱动）   ┐ 两条执行引擎 ★合并
     └─ _run_plan()             静态计划 for + Replanner 重试/跳过   ┘
     └─ _compose_answer()       只要有 LLM 就再调一次远程总结
```

### 已确认的重复/过时/重叠机制（减法清单依据）
| 现状 | 问题 | 处置 |
|---|---|---|
| `_route` Chat/Agent 硬分流（runtime L455-467、L99-101） | 「你好，请帮我分析数据」被按前缀切流；追问/改要求无法接续任务 | 删除硬分流；CHAT 成为 Loop 动作 |
| `_local_direct_plan` + `_task_spec_plan` + `_build_plan` 三入口 | 同一「下一步做什么」判三次，靠闸门与特判（ml. 不直连等）缝合 | 合并为 Loop 的一次 Decide |
| `_run_plan` 与 `_run_dynamic` 两套等待态/校验/熔断逻辑 | 护栏两处维护（动态路径曾漏熔断） | 合并为单一迭代体 |
| `decision/` 包 5 文件（router/rule/signal/local_model/remote/provider） | 与 TaskSpecBuilder、intent、planner 职责交叠；remote 产自由文本悬空 | 折叠进 Loop 的单一 `_decide`；signal/slot 逻辑下沉确定性层；remote 改为结构化决策 |
| `planner/planner.py`（456 行） | LLM 一次性大规划违背增量原则；rule_plan 的建模/合并/工作流等链是有效确定性资产 | 删 LLM 大规划与 plan cache；确定性链迁移为 playbook 待办队列 |
| `planner/replanner.py` | 失败策略只在静态计划路径生效 | 依赖断裂/重试/跳过/熔断策略并入 Loop 更新阶段 |
| `task_spec.py` + `task_spec_builder.py`（~390 行） | TaskSpec 是 TaskState 的子集 | 并入 TaskState；分句逻辑迁入确定性层 |
| `_compose_answer` 无条件远程总结 | 「多少行」也烧一次远程 | 确定性渲染优先；远程综合仅限必要场景 |
| 复杂度首次理解后固定 | 与「动态复杂度」目标冲突 | TaskState.uncertainty 每轮更新驱动中途升级 |
| 运行间无持续任务状态 | 场景 7/8 无法支持 | TaskState 随会话持久化（agent_store.json 增量字段） |

### 必须保留的基础设施（仅改为被 Loop 调用）
Tool Registry 与 36 个工具、DataEngineService、ExperimentService、WorkflowService、PermissionManager（含一次性授权凭据语义）、run_preflight + clarify 模板与 `agent.clarify`、AgentResultValidator、13 类 SSE 事件与 7 个 RunStatus、AgentStore 原子领取/重启恢复语义、DecisionTrace JSONL 与导出脚本、LLMProvider（熔断/结构化/用量捕获）、ContextBuilder（元数据 + 候选工具召回）、`{{stepN.field}}` 结构化依赖解析（step_resolution）、local_chat（确定性应答/结果渲染/降级文案）、answer_source 诚实标注。

### 外部契约冻结面（HTTP/SSE/存储/前端）
- 14 个 HTTP 端点路径、方法、请求体、`ApiResponse` 壳不变。
- 13 个 event type 名与 7 个 RunStatus 字符串不变；payload **只允许新增可选字段，不允许删改已有字段语义**。
- `agent_store.json` 格式仅允许**新增可选顶层/会话字段**（旧文件缺字段按 None 加载，沿用现有重启恢复语义）。
- 前端硬编码依赖详见 [types/agent.ts](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/frontend/src/types/agent.ts) 与 [agentEvents.ts](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/frontend/src/lib/agentEvents.ts)：route.mode、planning.stage 白名单（未登记 stage 走兜底标签）、completed.mode(chat/agent/notice)、clarification/permission/usage 等字段。

## Functional Requirements

- **FR-1 统一 TaskState**：一个状态对象至少包含 goal、constraints、conversation 引用、dataset context、facts、findings、artifacts、completed_actions、pending_actions、pending_user_questions、last_result、current_phase、confidence/uncertainty、model_usage；可序列化并随会话跨 Run 存活；每个 Run 开始时可「hydrate（接续）/ update（修改）/ reset（新任务）」。
- **FR-2 单一 Agent Loop**：`Observe → Decide Next Action → Execute/Chat/Ask/Escalate → Update State → Continue/Finish` 只有一个迭代实现；等待确认/澄清/取消恢复后进入**同一个**迭代体，不存在第二套执行引擎。
- **FR-3 取消 Chat/Agent 硬分流**：首条与后续消息都先进 Loop；纯问候由确定性/远程 CHAT 动作回答；「问候 + 数据诉求」自然建任务；任务进行中 CHAT/ASK/EXECUTE/ESCALATE 可在同一 Loop 内按状态切换。
- **FR-4 确定性层**：数据读取/统计/EDA/清洗/训练/评估等全部走现有工具；权限、参数校验、Pre-flight、必填槽位、信号→下一步、取消命令、状态/熔断检查零模型调用；建模/合并/工作流/全面分析等成熟多步链以**确定性 playbook 待办队列**表达，仍逐步执行、逐步 Observe、可被新事实与用户改要求调整。
- **FR-5 本地模型层**：local_router（TF-IDF，Qwen 可选）只回答「下一步选哪个工具/是否反问/是否升级/是否闲聊」，调用它必须经过 local_router 唯一入口；确定性可推出下一步时跳过模型；Qwen 缺失/异常/超时不影响运行（诚实降级到词法/确定性层）。
- **FR-6 Remote LLM 层**：仅在①本地低置信度②复杂推理/规划/方案比较③复杂参数推导④开放式知识问题⑤信号/结果冲突⑥可恢复失败重复出现 时介入；远程返回**结构化决策**（next action 或 2–4 步阶段队列 + 理由），写入 pending_actions 后控制权立即回到本地 Loop；远程不直接执行工具、不形成远程运行时；遵循 fallback_allowed 错误分类。
- **FR-7 增量决策**：默认不生成全量计划；每步基于上一步真实 ToolResult（紧凑视图 + signals + facts）决定下一步；远程阶段性队列最多 4 步且每步执行前仍过参数解析/权限/校验。
- **FR-8 Token 成本一等公民**：AgentTokenLedger 在现有字段基础上新增 qwen_calls、tool_calls、task_steps、escalation_count（已有 remote_escalations 复用并对账）、total_cost（无单价时为 0 但字段恒在）等；随 `usage` 事件增量下发；工具结果只以紧凑形态进入状态；发给远程的工具 schema 仅限当前阶段相关候选；远程决策输入不包含完整历史与完整 ToolResult；简单任务不发最终总结远程调用。
- **FR-9 动态复杂度**：TaskState.uncertainty 由初始诉求 + 数据事实 + signals + 连续失败次数 + 本地置信度 + 用户追加每轮重算；可在任务中途触发一次远程升级，也可在事实明确后回到纯本地推进。
- **FR-10 多轮接续**：会话存在未结束 TaskState 时，新消息按「状态内追问（读 facts/findings/last_result 作答，可零工具、至多一次远程）/ 约束修改（更新 goal/constraints/pending_actions 并继续）/ 新任务（归档旧状态、重置）」处理；「不要删这一列，改成填充」类指令必须真实改变后续动作参数。
- **FR-11 等待态与安全不变式保留**：高风险工具 WAITING_CONFIRMATION 且 SSE 保持打开；缺信息 WAITING_CLARIFICATION 且 SSE 立即收流；授权仍是一次性凭据（失败重试需重新授权）；confirm/clarify/deny/cancel 原子领取语义不变；无 shell/eval 通道；ContextBuilder 不读 DataFrame 的禁令不变。
- **FR-12 外部契约兼容**：FR 实现不得破坏冻结面；13 类事件照常发出（route 事件改由首个决策自然得出 mode；planning 的 stage 在现有白名单内选择）；run.summary()/full() JSON 键只增不删。
- **FR-13 DecisionTrace 延续**：每次 Loop 决策（含 source/confidence/action/signals/escalation）写入同一条 trace；新字段只增不删；`export_decision_trace.py` 可正常导出。
- **FR-14 架构减法**：删除 decision/ 包、planner/ 包（models 迁移后）、task_spec*.py、runtime 中三套规划入口与两条执行引擎；不得新增平行 Router/Planner/Agent 体系；净结果是 `app/agent`（不含 local_router）总行数与运行时决策分支数下降。
- **FR-15 可重复评测**：提供离线场景脚本（MockLLM 精确计数），覆盖全部 14 个验收场景，输出每场景 remote_calls/input_tokens/output_tokens/qwen_calls/tool_calls/task_steps/耗时/是否成功；重构前基线与重构后结果用同一脚本产出。
- **FR-16 前端对齐与两处缺陷修复**：前端类型/展示对齐账本新字段；修复①会话运行结束后侧栏列表不回拉（标题/消息数/run_ids 陈旧，切回丢失最后一次 run）②助手回答经 SSE 与 resume 返回值双路径重复上屏（统一唯一追加出口，移除页面层「相邻去重」补丁）。

## Non-Functional Requirements
- **NFR-1 质量**：重构后 `backend/tests` 全部通过；被删内部接口对应的旧测试同步迁移/重写，不以兼容 shim 保留死代码；架构不变式有测试守护。
- **NFR-2 可观测**：每次远程/本地决策在 SSE planning 事件与 DecisionTrace 中可区分来源；账本实时帧不回退。
- **NFR-3 性能**：简单场景（行数/分布/异常值）在无远程调用下平均完成时间不劣于现状；单 Run 事件 2000 硬顶、步数/时长/Token/重规划熔断全部继续生效。
- **NFR-4 安全**：不新增任何代码执行通道；数据集范围校验、权限裁决、参数 schema 预检保持强制。
- **NFR-5 部署**：不新增必需依赖；torch 仍可选；无 Key / 关远程 / Qwen 不可用三种环境均可启动并给出诚实行为。
- **NFR-6 可维护**：runtime.py 去除分号压行；新模块单一职责、带简洁 docstring；README/ARCHITECTURE 的 Agent 章节同步为新架构（含 as-is/to-be 对照）。

## Constraints
- **Technical**：Windows + PowerShell；FastAPI/Pydantic/Polars/sklearn 既有栈；单 worker 进程内状态；不引入新框架/新必需依赖。
- **Business**：毕设项目，减法优先、成熟库优先；每轮改动控制在约 10 处以内，每处给完整代码。
- **Dependencies**：local_router 训练资产（TF-IDF 权重、Qwen LoRA 权重、冻结评测集）只读复用；前端 React18/TS 事件链需同步验证。

## Assumptions
- 14 个验收场景以「绑定一个含数值列/缺失值/潜在异常的示例 CSV 数据集」为标准夹具；场景 9 需要可走通 detect→prepare→train 的数据集。
- MockLLM 可按场景脚本化返回结构化决策并自报 usage，足以精确统计 Remote 指标；真实 API 抽测不在本轮。
- 远程结构化决策复用现有 `structured_output` 通道与熔断，不新增 provider。
- Qwen 默认仍按 `LOCAL_ROUTER_MODE` 配置参与（默认 off 时只有 TF-IDF 生效），「Qwen 不可用」通过现有 unavailable 路径模拟。

## Acceptance Criteria

### AC-1: 14 个验收场景全部通过（离线脚本）
- **Type**: `rule`
- **Given**：标准数据集夹具与离线场景脚本（MockLLM 计数），分别在「远程可用（Mock）」「远程不可用（None）」「Qwen 不可用」配置下
- **When**：依次执行 14 个场景：①你好 ②你好，请帮我分析一下这个数据 ③看看这批数据有多少行 ④分析一下数据分布 ⑤什么是异常值 ⑥检查这批数据的异常值 ⑦分析过程中继续追问（为什么这里缺失这么多）⑧分析过程中修改要求（不要删除这一列，改成填充）⑨多步骤 EDA+预处理+建模 ⑩Qwen 不可用 ⑪Remote LLM 不可用 ⑫本地无法判断时自动升级 ⑬Remote 判断后回本地执行 ⑭同一任务多轮不重复发送完整上下文
- **Then**：每场景实际行为符合场景意图（工具链/回答/等待态/接续正确），脚本退出码 0
- **Pass Condition**：14/14 场景断言通过；场景③④⑥在「远程可用」配置下 remote_calls=0；场景⑩⑪数据类任务仍成功；场景⑫发生升级、场景⑬升级后远程调用次数=1 且后续由本地工具执行；场景⑭断言远程消息载荷不含完整历史 ToolResult 且连续两轮决策输入 Token 不增长超过设定阈值
- **Evidence**：脚本运行输出 JSON 与控制台结果（重构后），以及重构前基线 JSON 对比

### AC-2: 一个 Loop、一个 TaskState、无 Chat/Agent 硬分流
- **Type**: `rule`
- **Given**：重构后代码
- **When**：静态检查 `app/agent`
- **Then**：不存在 `_route`/`_run_plan`/`_run_dynamic`/`_local_direct_plan`/`_task_spec_plan`/`_build_plan`/`AgentPlanner`/`Replanner`/`TaskSpecBuilder`/`decision/` 包的任何残留定义或 import；运行时唯一迭代循环位于 loop 模块；`run()` 不再按 mode 切两条流程；CHAT 是循环内动作枚举之一
- **Pass Condition**：Grep 上述符号在 `app/`（排除 tests）零命中；唯一循环入口有代码与测试为证
- **Evidence**：Grep 输出、loop 模块源码、架构不变式测试

### AC-3: 外部契约零破坏
- **Type**: `rule`
- **Given**：重构后系统
- **When**：运行全部 agent 相关后端测试与前端类型检查/测试；核对 14 端点、13 事件、7 状态、store 加载
- **Then**：HTTP/SSE/存储契约保持；旧格式 agent_store.json 可加载；前端事件消费无未处理类型
- **Pass Condition**：后端 agent 测试全绿、`npm run build`（或 tsc）与前端 agent 测试通过；手工核对 payload 键只增不删
- **Evidence**：pytest 输出、前端构建/测试输出、契约核对清单

### AC-4: Token 账本扩展并实时下发
- **Type**: `rule`
- **Given**：任意一次工具型 Run
- **When**：观察 `usage` 事件与 run.summary().token_usage
- **Then**：至少包含 remote_calls、remote_input_tokens、remote_output_tokens、qwen_calls、tool_calls、task_steps、escalation_count、total_cost（旧字段保留）
- **Pass Condition**：字段齐全且数值随执行单调合理；简单工具任务 remote 三项为 0
- **Evidence**：账本单测、场景脚本指标输出、前端类型与 UI 展示

### AC-5: 重构前后指标对比达标
- **Type**: `rubric`
- **Dimension**：资源效率（同一离线脚本、同一夹具，重构后 vs 重构前）
- **Scale**：1-5
- **Anchors**：1 = 简单任务平均 remote_calls 未下降或复杂任务成功率下降；3 = 简单任务 remote 归零、复杂任务成功率持平、Token 持平；5 = 简单任务 remote 归零且简单任务平均 input_tokens 下降 ≥50%，复杂任务成功率 100%（脚本口径）且远程次数不增加，平均耗时不劣化
- **Pass Threshold**：>= 4
- **Evidence**：基线/重构后两份指标 JSON + 对比表（含 remote 次数、input/output tokens、耗时、工具数、成功率、复杂任务成功率、简单任务 remote 率）

### AC-6: 代码量与分支数净下降
- **Type**: `rubric`
- **Dimension**：架构减法纯度
- **Scale**：1-5
- **Anchors**：1 = 删旧增新、总行数上升；3 = app/agent（不含 local_router）总行数下降 ≥10% 且无平行新体系；5 = 行数下降 ≥20%，runtime 主文件 ≤600 行且无分号压行，决策分支（decide 相关条件分支）可枚举并有单测覆盖
- **Pass Threshold**：>= 3
- **Evidence**：重构前后 `cloc`/行数统计（按文件列清单）、Grep 分支清单

### AC-7: 多轮任务状态真实持续
- **Type**: `rule`
- **Given**：同一会话中已完成若干分析步骤
- **When**：用户发送追问与改要求（场景⑦⑧）
- **Then**：追问能引用 facts/findings/last_result 作答且不重复跑已完成的只读工具；改要求后 pending_actions 中对应工具参数真实变化（如 data.clean 策略为填充而非删除列）
- **Pass Condition**：脚本断言工具调用序列与参数；TaskState 在会话上序列化/反序列化字段对称
- **Evidence**：场景⑦⑧断言、store 持久化对称性测试

### AC-8: 远程升级是「顾问」不是「第二运行时」
- **Type**: `rule`
- **Given**：触发升级的复杂/低置信状态
- **When**：远程返回结构化决策
- **Then**：远程不接触工具执行；结果写入 pending_actions 由本地循环逐步执行；一次升级段内远程调用恰为 1 次（失败按 fallback_allowed 处理）；任务后续仍可再次按状态升级，但每次独立计费与记录
- **Pass Condition**：场景⑫⑬断言 + 决策层单测（MockLLM 调用计数、无工具执行副作用）
- **Evidence**：场景脚本、DecisionTrace 记录

### AC-9: 前端两处缺陷修复
- **Type**: `rule`
- **Given**：AI 实验室页面
- **When**：新建会话并完成一次运行；切换会话再切回；经历 confirm/clarify 恢复
- **Then**：侧栏标题/消息数/run_ids 在运行结束后回拉为最新，切回可恢复最后一次 run；助手回答不重复出现，页面层「相邻同内容去重」补丁删除
- **Pass Condition**：前端测试/手工验证通过
- **Evidence**：前端测试输出、关键代码 diff

## Open Questions
- 无未决项。已确认：评测采用离线场景脚本 + MockLLM 计数；前端做契约对齐并修复两处已知缺陷。其余按 Constraints/Assumptions 执行。
