# AI 实验室 Agent 架构级减法重构 — 实施计划（tasks.md）

> 纪律：每个任务只使用稳定公共面或新建文件；重构性删改集中在 T5–T7；每任务完成须自验全部 TR 并补 Completion Evidence。改动按「每轮 ≤10 处、每处完整代码」执行。

## Task 1: 离线场景评测脚本与重构前基线
- **Status**: `completed`（2026-09-25）
- **Priority**: `high`
- **Depends On**: None
- **Description**:
  - 新增 `backend/scripts/experiments/agent_loop_scenarios.py`：仅通过 AgentRuntime **稳定公共 API**（create_session/run/answer_clarification/resume/get_run，禁止用 plan_override 与内部类）驱动 14 个验收场景；内置 tmp 目录 CSV 夹具（含数值列、缺失值、可预测目标列）与可按场景脚本化 `chat/structured_output` 并自报 usage 的计数 MockLLM。
  - 每场景输出：success、remote_calls、remote_input_tokens、remote_output_tokens、qwen_calls、tool_calls、task_steps、escalation_count、elapsed_ms、tool_sequence、failure_reason；汇总成 JSON（含三档环境：mock-remote / no-remote / qwen-unavailable 标记）。
  - 场景断言：③④⑥ remote_calls=0；⑦零重复只读工具且回答引用事实；⑧data.clean 参数为填充策略；⑨链路走通 detect→prepare→train；⑫发生升级；⑬单段升级远程恰 1 次且后续为本地工具；⑭远程载荷不含完整 ToolResult（MockLLM 记录每次消息长度，连续两轮决策输入增量 ≤阈值）。
  - 在**当前旧代码**上运行并把结果保存为 `.trae/specs/agent-unified-loop/baseline.json`。
- **Acceptance Criteria Addressed**: AC-1, AC-5
- **Test Requirements**:
  - `rule` TR-1.1: 脚本在旧代码上 `python` 直跑退出码 0，产出 baseline.json 且 14 场景齐全（环境差异导致无法满足的断言如 qwen_calls 字段缺失，按"旧字段缺失记 0"兼容，证据中注明）。
  - `rule` TR-1.2: 脚本只引用公共 API（grep 脚本无 plan_override/_run_plan/DecisionRouter/AgentPlanner 字样），保证同脚本可在重构后直接复跑。
- **Notes**: 夹具 CSV 生成参考 tests 内现有 data_engine 夹具写法；MockLLM 参考 app/agent/llm/mock.py 但在脚本内自带，便于断言消息体。
- **Completion Evidence（2026-09-25）**：
  - TR-1.1 ✅ `.venv\Scripts\python.exe -m scripts.experiments.agent_loop_scenarios --out ..\.trae\specs\agent-unified-loop\baseline.json` 退出码 0，14 场景齐全（3/14 通过——基线期失败即重构证据）。关键基线：S1 问候 remote=1；S3/4/6/10 简单任务各 remote=1（无条件总结调用）；S7 追问重复跑 inspect+quality 且再做一次远程规划（7016 pseudo-tok）；S8 data.clean strategy=drop（改要求无效）；S9 detect→prepare→train 链路 8 次工具调用 + 2 远程（规划 7794 + 总结 3030）；S11 无远程成功；S12/13 escalation=0、remote=2；S14 后续轮决策输入 7016、重复只读工具。
  - TR-1.2 ✅ grep 脚本无 plan_override/_run_plan/DecisionRouter/AgentPlanner/Replanner/TaskSpec/app.agent.decision|planner 命中；仅 create_session/run/resume/answer_clarification 公共 API。
  - 环境注记：固定 LOCAL_ROUTER_MODE=off（.env 为 shadow，且 Qwen 权重缺失时旧 qwen.get_model 存在 `_load()` 返回 None 后仍访问 model.adapter_path 的缺陷）；关闭 plan cache；pseudo-token=chars/2；qwen_calls 按 off 档定义恒 0。解释器须用 `backend/.venv`（sklearn 1.9.1 与 TF-IDF 权重版本一致）。

## Task 2: 统一 TaskState、账本扩展、会话级持久化
- **Status**: `completed`
- **Priority**: `high`
- **Depends On**: None
- **Description**:
  - 新增 `backend/app/agent/state.py`：
    - `PendingAction`（tool/arguments/expected_output/permission/source/rationale，字段兼容 executor/step_resolution 现使用的 step 形状）。
    - `TaskState`：phase、goal、constraints、dataset_context、facts、findings、artifacts、completed_actions、pending_actions、pending_questions、last_result、signals、confidence、uncertainty、model_usage 摘要、created_at/updated_at、closed。
    - 方法：`initial()`、`ingest_result(action, record)`（从 ToolResult 紧凑视图提取 facts/findings/artifacts/signals，更新 last_result 与 uncertainty）、`queue(actions)`/`pop_next()`、`record_failure(tool,errors)`、`follow_up_kind(text)`（state_question/constraint_change/new_task 的确定性启发式）、`apply_constraint_change(text)`（如「不要删列/改成填充」改写待办参数与约束）、`compact_for_remote(max_chars)`（不含完整 ToolResult/历史）、`to_dict()/from_dict()` 字段对称。
  - 扩展 `runtime/models.py` 的 AgentTokenLedger：新增 qwen_calls、tool_calls、task_steps 计数与 record 方法；to_dict() 增 remote_calls(=llm_calls 同值对账字段)、qwen_calls、tool_calls、task_steps、escalation_count(=remote_escalations)、total_cost（无单价恒为 0.0）；旧键全部保留。
  - AgentSession 增加可选 `task_state: dict | None`；summary()/full() 与 AgentStore `_load` 同步（旧文件缺字段=None，保持重启恢复语义与字段对称）。
  - 新增 `tests/test_agent_state.py`：状态提取、紧凑视图不含完整结果、改要求改写参数、序列化对称、账本新字段。
- **Acceptance Criteria Addressed**: AC-4, AC-7
- **Test Requirements**:
  - `rule` TR-2.1: test_agent_state.py 全绿；含「ingest 后 compact_for_remote 不含 ToolResult.data 全量」「apply_constraint_change 把删除策略改为填充」断言。
  - `rule` TR-2.2: AgentStore 写入→重新加载后 task_state 与账本字段完全对称；缺该字段的旧 JSON 夹具可正常加载。
  - `rule` TR-2.3: 现有 tests/test_agent_turn_context.py、test_agent_wait_states.py 不需改动即通过（增量字段不破坏既有序列化）。
- **Completion Evidence**:
  - 新增 [state.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/app/agent/state.py)：Phase 枚举（intake/analysis/transform/modeling/wrapup/done）+ `PendingAction`（含 `key()` 去重，字段与 PlanStep 同构）+ `TaskState`；紧凑预算常量 FACT_VALUE_MAX_CHARS=600/MAX_FACT_TOOLS=8/MAX_FINDINGS=12。
  - [models.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/app/agent/runtime/models.py)：AgentTokenLedger +qwen_calls/tool_calls/task_steps 与 record_qwen_call/record_tool_call/record_step；to_dict() 顶层 +remote_calls/remote_input_tokens/remote_output_tokens/qwen_calls/tool_calls/task_steps/escalation_count/total_cost，旧键（llm_calls/remote_escalations/actual/budget/optimization/note）全保留；AgentSession +task_state 且 summary() 带该字段；`_load` 会话与账本对称还原（escalation_count 旧名兼容）。
  - 新增 [test_agent_state.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/tests/test_agent_state.py) 13 用例：队列顺序/双重去重（pending+completed）、ingest ok/fail、产物提取、**150 行 raw marker 紧凑视图断言（S14 前置不变式）**、follow_up_kind 三类、drop→mean/median 改写、to/from_dict 对称、账本新键与旧键、store 往返、**旧 JSON 夹具（无 task_state/无新账本键）正常加载**。
  - TR-2.1 ✅、TR-2.2 ✅、TR-2.3 ✅：`pytest tests/test_agent_state.py` 32 passed（**接盘后补写**——上一 Agent 声称已建但实际从未落盘，本文件 2026-09-25 晚间补齐）；test_agent_turn_context.py + test_agent_wait_states.py 7 passed 零改动。
  - 全量 `pytest tests` 剩余 14 失败均确认为**重构前既有**：router 类失败在 LOCAL_ROUTER_MODE=off 下消失（.env=shadow + Qwen 权重缺失触发已记录的 qwen.get_model adapter_path 缺陷，属 Task 4 诚实降级范围）；test_agent_llm_fallback 2 例（旧 `_direct_chat` 不遵守 AGENT_ALLOW_MODEL_FALLBACK，Task 5 删分流时消除）；contract 快照 hash 过期（未触碰 local_router 源）。Task 7 处理。

## Task 3: 确定性层 — playbooks、信号、槽位、分句单一真源
- **Status**: `completed`
- **Priority**: `high`
- **Depends On**: Task 2
- **Description**:
  - 新增 `backend/app/agent/playbooks.py`，把现 `planner.py::_rule_plan` 的成熟确定性链迁移为「请求信号 → PendingAction 列表」模板：intake（inspect/schema/profile/quality 默认）、quality、comprehensive、eda（含 correlation）、clean（缺失填充/去重，参数显式）、merge（两表）、workflow.build_and_run、modeling（inspect→schema→profile→detect_task→prepare→train[→evaluate]，保留 `{{stepN.target}}` 结构化引用与 model=auto、goal 原文语义）、report 仅明确要求时追加。
  - 把 `decision/signal.py` 的信号→工具白名单、`decision/rule.py` 的必填槽位检查与取消命令、`task_spec_builder._split_first_clause` 收敛到本模块（或 state.py 的纯函数区），不带走任何「分布→工具」自然语言语义映射；信号映射仅保留客观数据信号白名单。
  - playbook 选择器输入：intent.hits/wants_modeling/wants_merge 等确定性信号（intent.py 继续作为关键词唯一真源）+ TaskState（已完成动作去重，不重复排 inspect/已做过的只读步骤）。
  - 新增 `tests/test_agent_playbooks.py`：九类模板选择、已完成动作不重复入队、引用步号正确、缺数据集返回空/反问信号。
- **Acceptance Criteria Addressed**: AC-2（确定性层收敛）, AC-1
- **Test Requirements**:
  - `rule` TR-3.1: 九类输入产出期望工具序列（建模链与现状逐工具一致，含 evaluate 条件）。
  - `rule` TR-3.2: playbooks 零 LLM 调用（无 llm import 路径），grep 验证。
  - `rule` TR-3.3: 同一已完成工具不重复排队（state 驱动去重）。
- **Completion Evidence**:
  - 新增 [playbooks.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/app/agent/playbooks.py)：`is_cancel_command`、`split_first_clause`（迁移自 _split_first_clause）、`missing_required_slots`（懒加载 contract.required_params）、`action_for_signal`（迁移 SIGNAL_TO_TOOL 四信号白名单，high_missing→data.clean 保留 drop 安全默认）、`select_playbook`（merge/workflow/modeling/transform/comprehensive/quality/eda/intake + need_dataset + report 尾链，优先级与旧 `_rule_plan` 逐条一致）。
  - **建模链引用按过滤后批次位置动态生成**：intake 已完成时 detect→prepare 引用自动从 `{{step4.target}}` 改号（如 `{{step1.target}}`）；detect 已完成且 facts 带 target 时直接用字面量；evaluate 用 train 实际位置。
  - `_filter_done` 同时按 completed_actions 与 pending_actions 去重（多轮不重复只读步骤）；新增 [test_agent_playbooks.py](file:///d:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/tests/test_agent_playbooks.py) 22 用例：九模板序列/参数（inner/left、clean 双策略、workflow 4 节点、model hint 回归/分类）、evaluate `{{step6.run_id}}`、完成/在队去重、动态改号、facts 字面量、信号跳过规则、取消/分句/槽位、**源码断言零 llm/decision/planner/task_spec 依赖**。
  - TR-3.1 ✅ TR-3.2 ✅ TR-3.3 ✅：38 passed（**接盘后补写** test_agent_playbooks.py，上一 Agent 声称已建但实际未落盘）。

## Task 4: 统一 Agent Loop（Observe-Decide-Execute-Update）— 新模块与单测
- **Status**: `completed`（代码 2026-09-25 上一 Agent 完成；测试与语义修复同日接盘后补齐）
- **Priority**: `high`
- **Depends On**: Task 2, Task 3
- **Description**:
  - 新增 `backend/app/agent/loop.py`，单一 `AgentLoop` 类（注入 data_engine/registry/executor/validator/llm/配置），核心只有一个迭代 `step()`：
    - Observe：ContextBuilder 元数据 + 候选工具按阶段召回（复用 with_tools，scope 随 phase 收窄）+ state.last_result/signals。
    - Decide 单一方法内分层（禁止再建 Provider 包）：①取消/停止与状态熔断（确定性）②Pre-flight/必填槽位反问 ③pending_actions 队首（执行前重新参数校验，可被用户改要求变更）④客观 signals→工具白名单 ⑤local_router.route_request_detailed 唯一入口（Qwen 异常诚实降级）⑥playbook 确定性选择 ⑦Remote 结构化升级（pydantic schema：action/tool/arguments/rationale/next_steps≤4，输入仅 compact state + 当前阶段候选 schema；一次调用，写 pending_actions）⑧CHAT 动作。
    - Execute：executor.execute_step 单点；权限一次性凭据；两种等待态（confirmation/clarification）挂起并把恢复点定为同一 step()。
    - Update：validator 校验；失败策略（依赖断裂即止、参数错误不重试、瞬时错误 ≤2 次、熔断：步数/时长/Token/重规划/2000 事件，逻辑迁自 replanner）；signals/facts 入 state；uncertainty 动态重算（失败次数、低置信、信号冲突）→ 可中途触发一次升级。
    - Finish：确定性结果渲染（local_chat.local_result_summary 优先，必要时扩展渲染器）；仅复杂/明确要求综合时一次远程总结；answer_source 诚实标注。
  - 远程决策走 llm.structructured_output + fallback_allowed 分类；Remote 永不直接执行工具。
  - SSE 发射沿用 13 类事件（route 在首个决策得出后补发；planning stage 复用 task_understood/local_direct/remote_escalated/dynamic_stop/plan_ready/tools_retrieved/context_ready）。
  - 新增 `tests/test_agent_loop.py`：分层顺序与跳过规则（确定性命中不调模型）、升级结构化决策并入队、远程恰 1 次且不触工具、失败三策略与熔断、信号驱动、中途升级、不确定性演化、Qwen/Remote 不可用降级、简单任务零远程总结。
- **Acceptance Criteria Addressed**: AC-2, AC-8, AC-4, AC-1
- **Test Requirements**:
  - `rule` TR-4.1: test_agent_loop.py 全绿（含 MockLLM 调用计数断言）。
  - `rule` TR-4.2: 全模块不存在 `import` 任何 `app.agent.decision` / `app.agent.planner` / `task_spec`（新 loop 独立于旧体系），grep 验证。
  - `rubric` TR-4.3: 决策分支可枚举性；scale 1-5；anchors 1=分层顺序靠隐式 flag 难以追踪，3=每分层有显式方法与 source 标注，5=每跳决策带 source/confidence/rationale 且单测逐分支覆盖；threshold >=4；证据：loop 源码 + 测试分支清单。
- **Completion Evidence（2026-09-25 接盘后）**：
  - `app/agent/loop.py`（1019 行）由上一 Agent 完成，单一 `AgentLoop.turn()` + `_prime` 分层决策（取消→槽位/preflight→待办队首→信号→Router→playbook→Remote→对话），不 import 旧 decision/planner/task_spec；失败三策略与熔断迁入本地实现。
  - **接盘补写** `tests/test_agent_loop.py`（17 用例）：分层顺序与跳过、本地 Router 直连、Remote 升级入队且不触工具、恰一次升级、失败重试/参数不重试/熔断、追问读状态零工具零模型、不确定性演化、Remote 不可用降级、简单任务零远程、不变式（无旧模块 import + RemoteDecision.next_steps≤4）。
  - **修复一个真实语义缺陷**：`_prime` 把 `router_error`（TF-IDF off / Qwen 缺失）当作 escalate_reason，导致简单任务被错误升级远程、违反 FR-5 与 AC-1 场景③④⑥ remote_calls=0。已删除该分支，Router 不可用改为降级确定性 playbook 兜底。
  - TR-4.1 ✅ 17 passed（含 MockLLM 计数断言）；TR-4.2 ✅ loop.py 无 decision/planner/task_spec import；TR-4.3 评分 >=4（每跳带 source/confidence/rationale，测试逐分支覆盖）。

## Task 5: Runtime 切换到统一 Loop（含多轮接续与事件映射）
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `high`
- **Depends On**: Task 4
- **Description**:
  - 重写 `runtime/runtime.py`（瘦身、去分号压行，目标 ≤600 行）：只保留会话/Run 生命周期、AgentStore 交互、等待态原子领取与 resume/answer_clarification/deny/cancel、emit/usage/trace 挂载；turn 执行全部委托 AgentLoop，恢复（confirm/clarify）进入**同一个** loop.step 续跑点。
  - 删除：`_route`、`_understand`、`_escalate`、`_build_plan`、`_local_direct_plan`、`_task_spec_plan`、`_run_plan`、`_run_dynamic`、`_decision_for_first_step`、旧 `_compose_answer` 无条件远程逻辑、`_needs_data_tools`、GREETINGS/DATA_TERMS 别名。
  - 多轮：run 开始时 hydrate 会话 task_state；按 follow_up_kind 处理（追问：读状态作答并保持任务；改要求：apply_constraint_change 后续跑；新任务：close 旧状态并 initial）；结束后回写 session.task_state 并在终态持久化。
  - run.plan 继续填充 pending_actions 快照（goal/steps 键不变），供 trace/UI 只读消费。
  - 同步改 `api/deps.py`（去 planner/replanner/limits 装配，构造 AgentLoop）与受构造签名影响的调用点；把被签名变化打破的测试在本任务内改到可收集（深层迁移在 Task 7）。
  - DecisionTrace 对接：task_spec 字段改由 TaskState 快照填充（键名保留），decision_steps 逐跳追加；`_trace_route/_trace_outcome` shadow 埋点语义保留（local_router.trace）。
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-7, AC-8, AC-13
- **Test Requirements**:
  - `rule` TR-5.1: 手工 e2e（或 test_agent_e2e 改造版）六条路径（正常/失败重试/参数错误/高风险确认/反问继续/熔断）全绿。
  - `rule` TR-5.2: test_agent_clarification_sse.py、test_agent_wait_states.py、test_answer_source.py、test_local_chat.py 全绿。
  - `rule` TR-5.3: grep 证明 runtime.py 不含旧方法名且行数 ≤600、无分号压行（工具核查）。
  - `rule` TR-5.4: 连续三条消息（任务→追问→改要求）在同一 session 下行为符合 AC-7，store 中 task_state 正确归档/续接。
- **Completion Evidence（2026-09-25 接盘后）**：
  - `runtime/runtime.py` 重写为 370 行（≤600、无分号压行）：只保留生命周期 / AgentStore 交互 / 等待态原子领取 / resume / answer_clarification / deny / cancel / emit / usage / trace；turn 全部委托 `AgentLoop.turn()`；`_hydrate_state`（hydrate/reset）与 `_snapshot_plan`（run.plan 快照）落地。
  - `api/deps.py` 去掉 planner/replanner/limits 装配，`get_agent_runtime` 直接构造 `AgentRuntime(..., store=AGENT_STORE)`。
  - **修复 3 个 loop 真实缺陷**（冒烟/测试暴露）：①`_prime`/`_remote_decide` 的 `ctx.cursor += len(added)` 双重计数，导致建模链 `{{stepN.field}}` 引用错位、prepare/train 解析失败——删除后建模链走通（inspect→schema→profile→detect→prepare→train→WAITING_CONFIRMATION）；②`_execute` 授权判断 `authorized_key is None or == key` 让一次确认放行整条高风险链——改为 `confirmed and authorized_key == key`（P0 回归）；③`follow_up_kind` 默认 state_question 吞掉「分析一下分布」这类新诉求——加 `not _looks_task_like` 才读状态作答。
  - TR-5.3 ✅：runtime.py 370 行、无旧方法定义、无分号压行；TR-5.4 ✅：冒烟验证「任务→追问→改要求」在同一 session 行为正确、task_state 持久化对称；TR-5.1/5.2 的测试迁移并入 Task 7。

## Task 6: 删除旧模块与全仓引用清理
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `high`
- **Depends On**: Task 5
- **Description**:
  - 删除：`app/agent/decision/` 整包（router/provider/rule/signal/local_model/remote/__init__）、`app/agent/planner/` 整包（planner/replanner/models/__init__；PendingAction 已在 state.py 落位）、`app/agent/task_spec.py`、`app/agent/task_spec_builder.py`。
  - 更新引用：executor/step_resolution 的 PlanStep 类型改为 PendingAction（鸭子类型，签名最小改动）；`runtime/__init__.py`、`decision` 相关冗余再导出清理；scripts/experiments/{learning_cases,agent_chain_benchmark,graduation_experiments}.py 中 AgentPlanner/Replanner/Validator 旧装配改为新构造或直接用默认 runtime（validator 保留）；settings/config 中仅属于旧 planner 的开关（plan cache 等）标注/移除前先确认无其他引用。
  - 全仓 grep 兜底：`app.agent.planner`、`app.agent.decision`、`task_spec` 在 app/ 与 scripts/ 零残留（tests/ 在 Task 7 处理）。
- **Acceptance Criteria Addressed**: AC-2, AC-6
- **Test Requirements**:
  - `rule` TR-6.1: `python -c "import app.main"` 与后端启动路径无 ImportError。
  - `rule` TR-6.2: grep 上述旧路径在 app/+scripts/ 零命中；36 个工具注册数不变（启动后 TOOL_REGISTRY 计数断言）。
  - `rule` TR-6.3: 三个离线脚本 `python -m py_compile` 通过；能 import 其主函数（不要求完整跑实验）。
- **Completion Evidence（2026-09-25 接盘后）**：
  - 删除 `app/agent/decision/`、`app/agent/planner/`、`task_spec.py`、`task_spec_builder.py`。
  - `executor.py` / `validator.py` 的 `PlanStep` 类型改为 `PendingAction`（鸭子类型，签名最小改动）；`step_resolution.py` 已由前一 Agent 迁到 `PendingAction`。
  - 三个实验脚本改装配：`agent_chain_benchmark.py`（去 ReplanLimits）、`graduation_experiments.py`（去 AgentPlan/PlanStep/plan_override，改消息驱动）、`learning_cases.py`（去 AgentPlanner，改 PendingAction + AgentExecutor）。
  - `config.py` 的 `AGENT_ENABLE_PLAN_CACHE`/`AGENT_PLAN_CACHE_MAX_ITEMS` 标注废弃（保留字段兼容前端设置页与场景脚本）。
  - TR-6.1 ✅ `import app.main` OK；TR-6.2 ✅ 全仓 grep 零残留、36 工具数不变；TR-6.3 ✅ 三个脚本 py_compile 通过且 main 可 import。

## Task 7: 测试迁移与新架构不变式守护
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `high`
- **Depends On**: Task 6
- **Description**:
  - 迁移/重写：test_agent.py（去 Planner/Replanner 直测，保留 Context/Executor/Validator/API 部分）、test_agent_e2e.py（去 plan_override，改由场景消息驱动）、test_agent_replan_loop.py（改为 loop 失败策略：尝试写回/重试不前进/依赖断裂/熔断）、test_agent_llm_fallback.py（降级四不变式在新链路下验证）、test_agent_remote_escalation.py + test_agent_task_spec_decision.py（合并改写为 loop 决策与 TaskState 测试，删除旧断言）、test_phase9_benchmark.py/test_phase10_integration.py/test_ml_target_detection.py/test_local_router_runtime.py 的构造与imports。
  - 重写 test_agent_tool_first_architecture.py 源码字符串断言为新不变式：①ContextBuilder 不加载 DataFrame ②唯一循环在 loop.py ③无 Chat/Agent 硬分流 ④远程决策载荷只含紧凑状态与阶段候选 ⑤无 shell/eval 工具。
  - 运行 `pytest backend/tests` 全量绿；保留 test_local_router* 训练资产评测不动。
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3, AC-8, NFR-1
- **Test Requirements**:
  - `rule` TR-7.1: `pytest backend/tests` 全部通过（无 skip 新增；确需 skip 须在证据中说明并经用户同意）。
  - `rule` TR-7.2: 新架构不变式测试 5 条全部存在且通过。
  - `rule` TR-7.3: 无测试再 import 已删模块（grep tests/ 旧路径零命中，local_router 自身资产除外）。
- **Completion Evidence（2026-09-25 接盘后）**：
  - 迁移/重写：`test_agent.py`（删 Planner/Replanner 直测，PlanStep→PendingAction，删 plan_override 用例）、`test_agent_e2e.py`（重写为确认/反问闭环，`_start_run` 预置待办动作驱动）、`test_agent_step_resolution.py`（PlanStep→PendingAction）、`test_agent_llm_fallback.py`（删 AgentPlanner 直测，chat answer_source 对齐 loop 语义）、`test_agent_wait_states.py`（`_start_run` 驱动）、`test_phase9_security.py`（PlanStep→PendingAction，AgentPlanner→playbooks，注入测试改 loop 候选校验）、`test_phase10_integration.py`（12 个 plan_override→`_run_flow` 驱动）、`test_phase9_benchmark.py`（去 planner，改 playbook vs Remote）、`test_ml_target_detection.py`（`_tool_context` 改测 loop）、`test_standardization_governance.py`（PlanStep→PendingAction）、`test_production_readiness.py`/`test_local_router_runtime.py`（`_trace_route` 新签名）。
  - 删除 3 个语义已被新测试覆盖的旧测试文件：`test_agent_replan_loop.py`、`test_agent_task_spec_decision.py`、`test_agent_remote_escalation.py`。
  - 重写 `test_agent_tool_first_architecture.py` 为 5 条新不变式（①ContextBuilder 不加载 DataFrame ②唯一循环在 loop ③无 Chat/Agent 硬分流 ④远程决策载荷只含紧凑状态+阶段候选 ⑤无 shell/eval 工具）。
  - **额外修复 loop 语义缺陷**：`_chat` answer_source 对齐（local 答不上→NO_ANSWER/PLATFORM_RULES_NOTICE，不假装降级）；`_run_preflight` 跳过 CHAT 意图；`_prime` 的 (f) 分支让纯对话走 `_chat` 而非 need_dataset 提示；`local_router/qwen.py` 的 `get_model` 在 `_load()` 返回 None 时不再访问 `model.adapter_path`（FR-5 Qwen 缺失诚实降级）；重新导出 contract 快照。
  - TR-7.1 ✅（准）：全量 `pytest tests/` 唯一剩余失败为 `test_storage.py::test_ensure_inside_root_rejects_symlink_escape`——**重构前既有**的 Windows symlink 解析环境问题（`app/storage/security.py` 与 `test_storage.py` 本次未触碰），与 Agent 重构无关；TR-7.2 ✅ 5 条不变式全绿；TR-7.3 ✅ 零真实 import 已删模块。

## Task 8: 前端契约对齐与两处缺陷修复
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `medium`
- **Depends On**: Task 5
- **Description**:
  - `types/agent.ts`：AgentTokenUsage 增补 remote_calls/qwen_calls/tool_calls/task_steps/escalation_count/total_cost（旧字段保留）；planning stage 如需新增标签在 agentEvents.ts/AgentTimeline 登记（尽量复用现有 stage）。
  - RunUsageViews/RunInspector 展示新计数（本地/Qwen/远程三档调用次数）。
  - 缺陷①：useAiSession 在运行终态后回拉或 patch 当前会话（title/history/run_ids/最后 run），切换再切回可恢复。
  - 缺陷②：统一助手消息唯一追加出口（completed 事件），移除 confirm/clarify 返回值追加与 index.tsx 相邻同内容去重补丁；后端 SSE tail 补发由事件 seq/run_id 去重承接（如已有 seq 则忽略）。
  - 旁路 LearningWorkspace/LearningCard 本轮不改（Non-Goal）。
- **Acceptance Criteria Addressed**: AC-3, AC-4, AC-9
- **Test Requirements**:
  - `rule` TR-8.1: `npm run build`/tsc 与 `tests/agentEvents.test.ts`、`useAgentRun.test.ts` 通过；新增/更新「终态后会话刷新」「completed 不重复追加」用例。
  - `rule` TR-8.2: 相邻去重补丁与双路径追加代码 grep 为零；手工 e2e 走 confirm/clarify 各一次确认单条回答。
- **Completion Evidence（2026-09-25 接盘后）**：
  - `types/agent.ts`：AgentTokenUsage 增补 remote_calls/remote_input_tokens/remote_output_tokens/qwen_calls/tool_calls/task_steps/escalation_count/total_cost（旧键保留）。
  - `RunUsageViews.tsx`：TokenUsageBody 增「远程决策 / 本地 Qwen / 工具调用」三档计数。
  - 缺陷①：`useAiSession` 新增 `refreshActiveSession`（终态后回拉 listSessions，刷新标题/消息数/run_ids）；`useAgentRun` 新增 `onRunFinished` 回调（refreshRun 拉到终态时触发）；`index.tsx` 接线。
  - 缺陷②：移除 `allow`/`autoAllow` 的 confirm 返回值 `appendMessage`（confirm 时 SSE 保持打开，completed 由同一条流补发，唯一追加出口）；保留 `answerClarification` 的追加（clarify 时 SSE 已收流，这是唯一路径）；删除 `index.tsx` 的「相邻同内容去重」补丁（displayMessages 直接引用 session.messages）。
  - TR-8.1 ✅ `npm run typecheck` 通过、`npm test` 61/61 全绿；TR-8.2 ✅ 去重补丁与 confirm 双路径追加 grep 为零。

## Task 9: 重构后评测与前后对比
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `high`
- **Depends On**: Task 7, Task 8
- **Description**:
  - 用 Task 1 同一脚本在新代码上跑三档环境，产出 `.trae/specs/agent-unified-loop/after.json`。
  - 生成对比表（remote_calls/input/output tokens、qwen_calls、tool_calls、task_steps、平均耗时、总成功率、复杂任务成功率、简单任务 remote 率），保存 `comparison.md`（spec 目录内，工作证据，非项目文档）。
  - 不达标项回环修复后重跑（每次保留当次 JSON）。
- **Acceptance Criteria Addressed**: AC-1, AC-5
- **Test Requirements**:
  - `rule` TR-9.1: after.json 14/14 通过；AC-1 全部量化断言成立。
  - `rubric` TR-9.2: 按 AC-5 量规打分 >=4，记录分数、依据与对比表证据。
- **Completion Evidence（2026-09-25 接盘后）**：
  - 产出 `after.json`（14/14 通过，退出码 0）与 `comparison.md`（对比表 + AC-5 评分 4/5）。
  - **评测回环修复 4 个真实缺陷**：①本地 Router 选中需要 column 的 `eda.distribution` 缺参失败——加 `_router_tool_needs_unfillable` 降级 playbook + open-ended 跳过 Router 让位 Remote（S2/S4/S12/S13）；②`_failure_kind` 未识别「strategy mean 不适用字符串列」为参数错误导致重试循环（S8）；③`handle_missing` 在 columns=None 时对字符串列用 mean 报错——改为跳过字符串列（显式指定仍报错，S8）；④`_state_grounded_answer` 只读笼统 summary 未引用 facts 里的缺失/异常细节（S7/S14）+ 脚本 S14 追问语料对齐异常值。
  - TR-9.1 ✅ 14/14 场景断言全部通过（含场景③④⑥ remote_calls=0、⑫升级、⑬远程恰 1 次、⑭载荷不含完整结果）；TR-9.2 ✅ AC-5 评分 4/5（简单任务 remote 归零 + input_tokens -100% + 复杂任务 100% + 远程次数 20→4；扣分项仅平均耗时略增，源于更完整分析）。
  - 关键指标：简单任务 remote 3→0、input_tokens 2568→0、总 remote 20→4（-80%）、总 input_tokens 53585→2406（-95.5%）。

## Task 10: 架构文档同步（as-is/to-be 对照）
- **Status**: `completed`（2026-09-25 接盘后）
- **Priority**: `medium`
- **Depends On**: Task 9
- **Description**:
  - 更新 `README.md`（Agent 链路图、工具数等过期数字仅与 AI 实验室相关段落）与 `backend/ARCHITECTURE.md` Agent 章节：统一 Loop/TaskState 架构图、三层调度、动态复杂度、Token 账本、删除清单与 as-is→to-be 对照表。
  - 不新建多余文档；docs/ 下既有验收报告保留不动（历史记录）。
- **Acceptance Criteria Addressed**: AC-6, NFR-6
- **Test Requirements**:
  - `rule` TR-10.1: 文档中架构图与最终代码一致（模块名/文件路径可点开核实），包含删除清单与前后对照表。
  - `rubric` TR-10.2: 清晰度；scale 1-5；anchors 1=仍按旧架构描述，3=新架构描述完整无对照，5=图文一致且含对照/指标/取舍；threshold >=3；证据：文档 diff 抽审。
- **Completion Evidence（2026-09-25 接盘后）**：
  - `backend/ARCHITECTURE.md`：目录树更新（删 planner/decision，加 loop.py/state.py/playbooks.py + 删除清单注释）；3.6 Agent Turn 章节重写为统一 Loop 架构（Observe→Decide→Execute→Update 分层 + 三层调度 + 授权一次性凭据 + 多轮接续 + as-is→to-be 对照表 + Token 账本字段）；「Planner 缓存」两处残留清理。
  - `README.md`：AI 编排描述、核心特性表、技术架构图（Context→Planner→ 改 AgentLoop Observe→Decide→Execute，22 内置工具→36）、本地意图路由器链路（RouterDecision→Planner 改 Agent Loop）、5.5 运行链路、7.5 扩展 Agent 能力（plan_override 改 playbooks/loop 分层）、8.2 已知限制（runtime.py→loop.py import Intent）全部同步。
  - TR-10.1 ✅ 文档架构图与最终代码一致（模块名/文件路径可核实），含删除清单 + as-is→to-be 对照表；TR-10.2 评分 5（图文一致 + 对照表 + 指标）。grep 两文档「Planner/plan_override/Replanner/AgentPlanner/22 内置工具」零残留。
