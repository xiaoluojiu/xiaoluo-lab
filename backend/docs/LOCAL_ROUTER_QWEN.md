# 本地意图路由器：Qwen3-0.6B + LoRA

本文说明「Qwen3-0.6B + LoRA 作为第一层本地意图路由器」是怎么接进现有 Agent 的，
以及它**不负责**什么。

## 一、它在架构里的位置

```
用户请求
  ↓
L0 升级规则          escalation_rules.py   —— 纯规则，零成本，最先跑
  ↓
L1 Qwen 神经路由      qwen.py + qwen_protocol.py  —— 本文新增
  ↓ （未命中则）
L1 词法路由           model.py（TF-IDF + LinearSVC，既有基线）
  ↓
反问规则              router.py::_finalize（缺必填槽位 → 反问，不升级）
  ↓
RouterDecision ──→ Agent Planner → Tool Registry → Permission → Executor
  ↓                                                              ↓
DecisionTrace ←────────────────────────────────── Validator / Replanner
```

**L1 的两种实现是同一层的两个候选，不是两个层。** Qwen 命中就用 Qwen，
不命中自动退回词法；两者产出的都是同一个 `RouterDecision` 对象。

## 二、它不做什么（边界）

| 不做 | 交给谁 |
| --- | --- |
| 工具执行 | `Tool Registry` + `Executor` |
| 复杂推理 / 最终回答 | `Planner` / 远程 LLM |
| 权限判定 | `PermissionManager` |
| 结果校验与重规划 | `Validator` / `Replanner` |
| 具体参数值的事实性保证 | `Pre-flight` + `Tool Registry` 的 schema 校验 |

模型输出的 `content` **只经过 `json.loads`**，解析成 `RouterDecision` 之后
一律走既有的 Planner / Tool Registry / Permission 流程。链路里没有任何
`eval` / `exec` / 直接执行模型文本的路径。

## 三、模型放置与配置

权重**不随仓库分发**（`.gitignore` 已覆盖 `/models/` 与 `backend/models/`）。
默认探测顺序：

1. `LOCAL_ROUTER_MODEL_PATH` 显式指定（相对或绝对路径都可以）
2. `{项目根}/models/qwen3-lora-xiaoluo-router`
3. `{MODEL_ROOT}/qwen3-lora-xiaoluo-router`

基座模型 `LOCAL_ROUTER_BASE_MODEL` 可以是本地目录，也可以是 HuggingFace
repo id（默认回落 `Qwen/Qwen3-0.6B`）。取值像本地路径（以 `.` 开头、含盘符）
时按路径解析，否则按 repo id 下载。

```bash
# 相对路径（推荐，便于开源分发）
LOCAL_ROUTER_MODEL_PATH=models/qwen3-lora-xiaoluo-router
LOCAL_ROUTER_BASE_MODEL=models/Qwen3-0.6B

# 绝对路径（本机开发）
LOCAL_ROUTER_MODEL_PATH=D:/models/qwen3-lora-xiaoluo-router
```

目录里只需要**推理必需文件**：`adapter_config.json`、`adapter_model.safetensors`、
tokenizer 系列文件。`checkpoint-*` 与 `optimizer.pt` 是训练产物，不要拷。

## 四、三档模式

| 档位 | 行为 |
| --- | --- |
| `off` | 完全关闭。不加载模型、不写 trace。默认值。 |
| `shadow` | 跑 Qwen，但**只落 trace**，不参与实际决策。线上行为与 `off` 完全一致。 |
| `active` | Qwen 正式作为第一路由器。 |
| `guard` | 历史别名，等同 `active`（读到会打一条告警，请改用 `active`）。 |

`shadow` 的「不改行为」是在代码里强制的，不是靠调用方自觉：
`router._route_with_qwen(record_only=True)` 时**永远不返回可用决策**。

## 五、降级链

任何一步失败都不会产生「空白回复 / 无限 loading / SSE 永不结束」：

| 失败点 | 处置 |
| --- | --- |
| 依赖没装（无 torch / peft） | 这一层不存在，直接用词法路由，无报错 |
| 权重缺失 / 加载失败 | 记一次 warning，退回词法；60 秒后允许重试一次 |
| 推理超时（默认 800ms） | 放弃本次生成，退回词法 |
| JSON 解析失败 | 退回词法 |
| 工具不在当前注册表 | 退回词法（**不允许模型发明平台没有的能力**） |
| `dataset_id` 幻觉 | 用会话绑定值覆盖；无绑定时校验 id 存在，不存在则丢弃该参数 |
| 词法也不可用 | 保守 `escalate`，交给 Planner / 远程，绝不猜 |

**硬超时是正确性依赖，不是优化项。** SSE worker 要等 `AgentRuntime.run()`
返回才会发终止标记；run() 一旦阻塞在模型推理上，`event: done` 永远不会发出，
前端就一直转圈。这是整条链路上唯一可能无限阻塞的新增点。

## 六、输出协议与契约映射

模型输出一段 JSON：

```json
{"mode": "command", "intent": "dataset", "tool": "connector.list", "params": {}, "confidence": 0.95}
{"mode": "workflow", "intent": "ml", "steps": [{"tool": "ml.train", "arguments": {...}}]}
{"mode": "chat" | "clarification" | "escalate" | "unsupported", ...}
```

提示词口径**必须与训练语料逐字一致**（`SYSTEM_PROMPT` 及上下文拼接格式），
否则已训练的 LoRA 会失效。详见 `qwen_protocol.py` 的模块 docstring。

映射到既有契约（`contract.RouterDecision`）：

| mode | RouterDecision |
| --- | --- |
| `chat` | `intent=CHAT, tool=None` |
| `command` | `intent` 由 tool 反查 + `tool` + `params` |
| `workflow` | 取首步工具（与 runtime 的「首步 + signals 驱动后续」同机制） |
| `clarification` | 有 tool ⇒ 缺槽位反问；无 tool ⇒ `escalate(ambiguous)` |
| `unsupported` | `escalate(out_of_scope)` |
| `escalate` | `escalate(映射后的 reason)` |

**没有第二套 Intent。** `QwenMode` 只是模型的输出词表（与词法的
`chat` / `call::<tool>` 标签同性质）；写进决策的 intent 一律由 `tool` 反查得出，
与词法路径的写法完全一致——工具是硬事实，模型自述的意图只是软判断。

两处刻意的不对称：

1. 训练语料多出一类 `complex_task`，映射为 `multi_step`（原值留在
   `QwenDecision.raw_reason`，离线仍可区分）。
2. `clarification`（611 条样本大多**不带 tool**）无法在契约里表达——
   `decision_to_route()` 在 `tool is None` 时会直接返回 `chat` 丢掉 `missing`。
   与其伪造一个 tool，不如诚实转 `escalate(ambiguous)`：平台本来就有更完备的
   澄清通道（`app/agent/clarify.py` 与 Pre-flight 的 `_await_clarification`）。

## 七、启用步骤

```bash
# 1) 安装可选依赖（约 2GB）
pip install -e "backend[router]"

# 2) 放置权重到 models/qwen3-lora-xiaoluo-router/ 与 models/Qwen3-0.6B/

# 3) 先跑 shadow 观察，确认无误再切 active
echo "LOCAL_ROUTER_MODE=shadow" >> backend/.env

# 4) 分析 trace 后再切 active
python backend/scripts/router/analyze_shadow.py
echo "LOCAL_ROUTER_MODE=active" >> backend/.env
```

不装依赖也能正常跑完整平台：本地 Router 会自动退回词法模型 + L0 规则。

## 八、已知限制

1. **数据集清单会被截断**（`LOCAL_ROUTER_MAX_CONTEXT_DATASETS`，默认 30）。
   训练语料里是 50 个，线上可能成百上千，全量注入会把 prefill 拖慢。
   截断是工程妥协，会造成轻微分布漂移，靠 shadow trace 观察。
2. **参数抽取口径未统一**。现有 `PARAM_EXTRACTION_IMPLEMENTED = False`，
   词法路径不抽参数；Qwen 其实会抽（训练语料里 `params` 有 4318 条）。
   目前 Qwen 抽出的参数只用于填充与校验，**没有翻转那个开关**——
   翻转会同时改变词法路径的行为口径，需要先用真实数据评估。
3. **`workflow` 模式目前只取首步工具**，模型给的后续 `steps` 仅进 trace。
   完整多步编排仍由 Planner / `_run_dynamic` 负责。
4. 训练语料是合成数据（9069 条），真实泛化能力**必须在 shadow 积累真人会话后
   重新验证**。离线数字不能直接当线上指标。
