# 本地 AI Router —— 测试清单

> 对应实现：`app/local_router/{contract,scoring,model,router,trace}.py`、
> `app/api/v1/settings.py::local_router_settings`、`app/agent/runtime/runtime.py::_trace_*`
> 产物训练：`scripts/router/train_runtime_l1.py`　数据分析：`scripts/router/analyze_shadow.py`
>
> **当前接入档位：`shadow`（只记录，不改行为）**。下面第 1 节确认状态，第 3 节是主清单。

---

## 0. 先明确：这一版接了什么、没接什么

| | 状态 |
| --- | --- |
| 本地 Router 模型落地（词法，31 类，2.1 MB，0.7 ms/条） | ✅ 已接入 |
| 三层合成（L0 规则 → L1 模型 → 反问规则）运行时实现 | ✅ 已接入，与离线口径**逐条等价**（844/844） |
| shadow 埋点（记录决策 + 实际执行结果） | ✅ 已接入 |
| 只读状态端点 `GET /api/v1/settings/local_router` | ✅ 已接入 |
| **用本地 Router 取代 / 跳过 LLM planner** | ❌ **未启用**（guard 档预留，需先积累数据） |
| 参数抽取（从话里读出 dataset_id 之外的值） | ❌ 未实现（`PARAM_EXTRACTION_IMPLEMENTED = False`） |

**所以本轮的通过标准不是「Router 准」，而是「什么都没变 + 数据收得到」。**

---

## 1. 前置检查（5 项，任一不过就不要开始测试）

在 `backend/` 目录下执行。

### 1.1 四项体检（一条命令）

```bash
cd D:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend
./.venv/Scripts/python.exe -c "import json;from app.core.config import settings;from app.local_router.model import artifact_path,get_model;m=get_model();print(json.dumps({'mode':settings.local_router_summary(),'artifact':str(artifact_path()),'exists':artifact_path().exists(),'loaded':m is not None,'staleness':(m.staleness() if m else None),'trained_route_acc':(m.meta.get('route_acc_groupcv') if m else None)},ensure_ascii=False,indent=1))"
```

期望输出（`mode` 与你 `.env` 里设的一致）：

```json
{
  "mode": {"mode": "shadow", "active": true, "confidence_threshold": 0.0,
           "model_dir": "...\\backend\\models\\local_router", "trace_max_bytes": 5242880},
  "artifact": "...\\backend\\models\\local_router\\lexical_l1.pkl",
  "exists": true,
  "loaded": true,
  "staleness": null,
  "trained_route_acc": 0.808057
}
```

**必须满足**：`exists=true`、`loaded=true`、`staleness=null`。
若 `staleness` 不是 `null`（说明平台工具清单变了，重跑 1.2）；若 `exists=false`（重跑 1.2）。

### 1.2 重新训练产物（仅在 1.1 不通过时）

```bash
./.venv/Scripts/python.exe scripts/router/train_runtime_l1.py
```

期望末尾：`产物已写入：...lexical_l1.pkl`，且**五项闸门全部通过**。任一闸门不通过时脚本**拒绝写产物**（退出码 1）—— 此时不要继续测试，先把闸门报错贴出来。

| 闸门 | 期望 |
| --- | --- |
| ① 输入口径一致 | 通过（n=844，不一致 0 条） |
| ② 超参同源 | 通过（构造保证） |
| ③ 线上 == 离线 | 通过（n=844，不一致 0 条） |
| ④ 反序列化一致 | 通过（概率和最大误差 ~1e-16） |
| ⑤ 标签空间覆盖 | 通过（缺失类 无） |

### 1.3 自动化测试全绿

```bash
./.venv/Scripts/python.exe -m pytest tests/test_local_router_runtime.py tests/test_local_router_contract_snapshot.py tests/test_local_router_escalation_rules.py tests/test_local_chat.py -q
```

期望 `97 passed`。（后两个文件与 Router 无关：`test_local_chat.py` 钉的是「关闭远程大模型后的内置应答」，
见 3.9；它和 Router 一起跑是因为两者都在「无远程 LLM」这条链上。）
**注意**：若 `test_snapshot_source_hash_is_current` 失败，说明有人改了 `contract.py` 但没重导快照，跑：
`./.venv/Scripts/python.exe scripts/router/export_contract.py`

> 状态端点用例（`test_settings_endpoint_exposes_local_router_state`）按**当前档位**分支断言：
> `off` 档要求不加载产物，`shadow`/`guard` 档只要求「产物存在却加载不了时必须给出原因」。
> 所以把 `.env` 切到 shadow 后它**依然是绿的** —— 这是刻意的，避免用例随 `.env` 红绿。

### 1.4 服务能起、状态端点可读

```bash
# 终端 A：启后端（必须用 venv 的 python，否则缺 polars）
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 终端 B：
curl -s http://127.0.0.1:8000/api/v1/settings/local_router
```

期望 `data.mode` = `shadow`、`data.artifact_loaded` = `true`。
这一条同时就是**第 2 节「验证开关生效」的方法**。

### 1.5 前端能起

普通 `npm run dev`（5173）。若后端报 `no such table`，说明连到了错的库，见项目记忆里的排查顺序。

---

## 2. 开关怎么切

**只有一处**：`backend/.env`

```ini
# off（默认，完全不动）| shadow（只记录）| guard（预留，勿开）
LOCAL_ROUTER_MODE=shadow
# 以下两项按需，通常留空/默认
# LOCAL_ROUTER_CONFIDENCE_THRESHOLD=0.0
# LOCAL_ROUTER_MODEL_DIR=
```

改完**重启后端**（该配置在进程启动时读取）。

> 状态端点是**只读**的，故意不提供 HTTP 切档：切档会改变线上路由行为，
> 不该由一个请求顺手完成。切档请改 `.env`。

---

## 3. 人工测试清单（主清单）

### 3.0 怎么算「通过」

每条用例要核对**两件事**：

| 核对项 | 怎么核对 | 通过标准 |
| --- | --- | --- |
| **A. 行为未变**（最重要） | 与关闭 Router 时的表现对比 | **完全一致**：该闲聊的闲聊、该调工具的调工具、工具调用链与参数不变 |
| **B. 埋点正确** | 看 `backend/models/local_router/shadow.jsonl` 里这条 run 的两行 | `kind:"route"` 有 `router` 段；`kind:"outcome"` 的 `executed_tools` 与实际一致 |
| **C. 判定方向正确**（仅 E 组是硬性） | 看 `router.route` 字段 | E 组必须升级且理由正确；B/C/D 组按你自己的业务判断填「正确工具」列 |

> ⚠️ 不要把 `router.route` 当正确答案看 —— 它就是要被你评估的**待测对象**。
> 你的判断记到「你判的正确工具」列，最后用 `analyze_shadow.py` 汇总。

### 3.1 A 组 —— 回归对照（**必做，先做这一组**）

目的：证明「打开 shadow」没有改变任何行为。

1. `LOCAL_ROUTER_MODE=off`，重启，依次发下面 6 句，记录每次的**工具调用链**与最终答复。
2. 改成 `shadow`，重启，**原样再发一遍**。
3. 逐条比对两轮。

| # | 输入 | 依据 |
| --- | --- | --- |
| A1 | `你好` | 闲聊分支（不走 planner） |
| A2 | `看看这批数据的分布` | 单工具直选 |
| A3 | `做个相关性分析` | 单工具 + 可能反问 |
| A4 | `训练一个分类模型，目标列是 label` | 建模长链路（多步计划） |
| A5 | `先把缺失值清洗掉然后再做一个分布图` | 多步编排 → 规则升级 |
| A6 | `把这个数据集同步到我们的 S3 存储桶` | 应升级（外部系统） |
| A7 | `先清洗缺失值然后再做一个分布图` | **已知漏判**，见 3.8（保留用于观察） |

**通过标准**：`status`、`plan.steps[].tool`、`tool_calls[].tool`、`final_answer` 全部相同。
**唯一允许的差异**：多出 `shadow.jsonl` 文件。

> 想看严格 diff：两轮跑完后，用 `GET /api/v1/agent/runs/{run_id}` 各导出一次 run summary（前端运行详情页可直接看到），保存成两个文件后比对。

### 3.2 B 组 —— 闲聊不该被当成数据任务（误伤方向）

> 下表「当前产物实测」一列是用**本仓库当前产物**离线跑出来的真实输出（附置信度），
> 作用是**检测漂移**：如果你跑完发现与这一列不同，说明产物或代码变了，应停下来查。
> **它不是正确答案** —— 正确答案由你按业务判断填最后一列。

| # | 输入 | 期望方向 | 当前产物实测 | 你判的正确工具 |
| --- | --- | --- | --- | --- |
| B1 | `你好` | `chat` | `chat` (0.901) | — |
| B2 | `你是谁` | `chat` | `chat` (0.904) | — |
| B3 | `谢谢，辛苦了` | `chat` | `chat` (0.902) | — |
| B4 | `这个平台怎么用` | 不该调数据工具 | `chat` (0.846) | — |
| B5 | `今天天气不错` | `chat` | `chat` (0.899) | — |

**不通过信号**：`router.route` 是 `call::xxx` / `ask::xxx`（会把闲聊送进工具流程）。

### 3.3 C 组 —— 单工具直选（**已绑定数据集**时）

先在会话里绑定一个数据集（前端选数据集），再发：

| # | 输入 | 期望方向 | 当前产物实测 | 你判的正确工具 |
| --- | --- | --- | --- | --- |
| C1 | `看看这批数据的分布` | 调工具 | `call::eda.distribution` (0.819) | `eda.distribution` |
| C2 | `看一下数据概况` | 调工具 | `call::dataset.profile` (0.556) | `dataset.profile` |
| C3 | `检查一下数据质量` | 调工具 | `call::dataset.quality` (0.876) | `dataset.quality` |
| C4 | `画一下缺失值的图` | 调工具 | `call::eda.visualize` (**0.404**) | `eda.visualize` |
| C5 | `看看列之间的相关性` | 调工具 | `call::eda.correlation` (0.885) | `eda.correlation` |
| C6 | `把缺失值清洗掉` | 调工具 | `call::data.clean` (0.890) | `data.clean` |
| C7 | `筛选出 age 大于 30 的行` | 调工具 | `call::data.filter` (0.661) | `data.filter` |

**注意**：
- C 组的 `router.missing` 应为 `[]`（因为已绑定数据集）；若 `missing=["dataset_id"]` 而实际**已绑定**，说明 `bound_dataset_id` 没传进埋点，是 bug，记下来。
- **C4 的 0.404 是有价值样本**：工具选对了但置信度低 —— 正是门控阈值要处理的情况。
  这类样本越多，说明「按置信度升级」越有存在必要，请重点记录。

### 3.4 D 组 —— 缺槽位应「反问」而不是「升级」（**未绑定数据集**时）

开一个新会话，**不绑定任何数据集**，再发：

| # | 输入 | 期望方向 | 当前产物实测 | 说明 |
| --- | --- | --- | --- | --- |
| D1 | `看看这批数据的分布` | 反问 | `ask::eda.distribution` (0.833)，`missing=["dataset_id"]` | 正确处置是反问「对哪个数据集」 |
| D2 | `检查一下数据质量` | 反问 | `ask::dataset.quality` (0.893)，`missing=["dataset_id"]` | 同上 |
| D3 | `把这批数据清洗一下` | 反问 | `ask::data.clean` (0.891)，`missing=["dataset_id"]` | 同上 |

**不通过信号**：`router.route` = `escalate::ambiguous`（把可零成本挽回的缺参当成了能力不足）。

### 3.5 E 组 —— 五类升级原因（**硬性期望**）

`escalate` 由 L0 确定性规则判定，**每条必须有确定的理由**。这也是唯一一组可以直接判对错的。
下表「实测」列已用当前规则离线核实过，**必须 5/5 命中**。

| # | 输入 | 期望原因 | 实测 | 触发机制 |
| --- | --- | --- | --- | --- |
| E1 | `把这个数据集同步到我们的 S3 存储桶` | `out_of_scope` | ✅ `escalate::out_of_scope` (1.000) | 外部系统词表（s3 / 存储桶） |
| E2 | `帮我调低学习率再训练一次` | `param_dependency` | ✅ `escalate::param_dependency` (1.000) | 平台未建模的 ML 内部词（学习率） |
| E3 | `不要自动清洗我的数据` | `conflict` | ✅ `escalate::conflict` (1.000) | 转折否定（不要） |
| E4 | `先把缺失值清洗掉然后再做一个分布图` | `multi_step` | ✅ `escalate::multi_step` (1.000) | 3 个连词 + 长度 17 ≥ 16 |
| E5 | `优化一下` | `ambiguous` | ✅ `escalate::ambiguous` (1.000) | 极短(4) + 模糊说法 + 无具体宾语 |

**通过标准**：`router.route` = `escalate::<期望原因>`，且 `router.escalate=true`、`router.intent=null`。
**注意**：E 组必须由**规则**判定，所以 `router.confidence` 恒为 `1.0`。
若某条判成了工具调用，先单独验证规则层：

```bash
./.venv/Scripts/python.exe -c "from app.local_router.escalation_rules import detect_escalation; print(detect_escalation('优化一下'))"
```

输出 `EscalationReason.AMBIGUOUS` 说明规则层正常 —— 那么问题在合成顺序（`router.py`），而不是规则。

### 3.6 F 组 —— 会话上下文（绑定状态必须双向编码）

| # | 步骤 | 期望 |
| --- | --- | --- |
| F1 | 新会话，不绑定 → 发「看看分布」 | trace 里 `bound_dataset_id=null`，`missing=["dataset_id"]` |
| F2 | 同一会话绑定数据集 → 再发「看看分布」 | trace 里 `bound_dataset_id=<id>`，`missing=[]` |
| F3 | 上一句（F2）的 `n_columns` | 记录为 `0` 属**预期**（shadow 不解析 Parquet，见第 6 节） |
| F4 | 多轮的指代：先「看看分布」再「再画一个」 | 两句都应有 trace 记录，run_id 不同、session_id 相同 |

**不通过信号**：`bound_dataset_id` 永远是 `null`（说明没从 session 取到）。

### 3.7 G 组 —— 多步 / 边界（只看埋点是否完整，不判对错）

| # | 输入 | 实测（参考） | 关注点 |
| --- | --- | --- | --- |
| G1 | `先做数据清洗然后训练一个模型最后生成报告` | `escalate::multi_step` | `outcome.executed_tools` 是否记录了**整条链**（多个工具） |
| G2 | `训练一个模型并生成报告` | `call::ml.train` (0.397) ← **漏判**，见 3.8 | `planned_tools` 与 `executed_tools` 可能不同（重规划），这是**信息**不是错误 |
| G3 | `先清洗缺失值然后再做一个分布图` | `call::data.clean` ← **漏判**，见 3.8 | 预期是升级，实际只选了第一个工具 |
| G4 | （故意发一句让工具报错的话，如未绑定却要求训练） | — | run 失败时 `outcome` 仍应写入，`status`/`error` 有值 |
| G5 | 发完消息立刻刷新/关闭页面 | — | `outcome` 仍应写入（`finally` 保证） |

### 3.8 本轮已发现的判定缺陷（**记录，不要当 bug 报**）

这两条是 `escalation_rules.detect_escalation` 的**已知边界**，代码里已有注释标注适用边界；
列在这里是为了让测试结果**可预期**，并作为「积累的数据要重点覆盖哪一类」的线索。

| 缺陷 | 例句 | 实际 | 原因 |
| --- | --- | --- | --- |
| 多步判定长度门限差 1 字 | `先清洗缺失值然后再做一个分布图`（15 字） | 规则返回 `None` → 模型只选了 `data.clean` | `MIN_MULTI_STEP_LEN = 16`，只差 1 个字符 |
| 单个「并」不算强连词 | `训练一个模型并生成报告`（11 字） | 规则返回 `None` → 只选了 `ml.train`，漏掉 `report.generate` | 需 2 个连词且长度 ≥16，单靠「并」不够 |

**为什么这两条很有价值**：它们在 shadow 数据里会表现为
「`planned_tools` / `executed_tools` 有多个，但 `router.route` 只给出一个工具」——
正好是 `analyze_shadow.py` 里的「会出错(工具不符)」或「可省(部分)」样本。
真实测试中遇到同类句子（多步但措辞不长），**请把原文记下来**，这是后续改造规则/补数据的直接依据。

---

## 3.9 H 组 —— 关闭远程大模型后的对话（**与 Router 无关，但同一片测试现场**）

### 背景（这条不是 Router 的问题，是它暴露出来的）

`Router` **只做分类，不生成任何文字**。所以关掉「设置 → AI 服务 → 启用远程 API 大模型」后，
它能把「你好」正确判成 `chat`（shadow 里能看到），但**没有任何东西负责回答** —— 修复前
`_direct_chat` 在 `llm is None` 时只回一句硬编码的「当前尚未配置可用的大模型。」，
而那句文案在关开关时**是事实错误**（凭据原样保留，`PUT /settings/llm/remote` 只切开关）。

现已补 `app/agent/local_chat.py`：**仅覆盖答案确定的意图**（问候 / 自述 / 能力 / 用法 / 致谢 / 告别），
拿不准的句子返回 `None` 并退回**说明性文案**；能力清单**实时读自 `TOOL_REGISTRY`**。
远程大模型**开启时**该分支完全不走，行为一字未变（有专门的 `test_llm_path_is_untouched` 钉住）。

### 用例（前置：后端已重启，`LLM_API_KEY` 保持已配置状态）

| # | 操作 | 期望 | 不通过信号 |
| --- | --- | --- | --- |
| H1 | 设置页**关闭**远程 API 大模型 → 发 `你好` | 出现「我是小洛实验室的 AI 助手」+「**平台自带能力**」字样 + 「看看这批数据的分布」示例 | 出现「当前尚未配置可用的大模型。」（旧文案回归） |
| H2 | 同状态发 `你能做什么` | 列出「**真实注册**的 N 个数据工具」与 6 个能力域（N 应等于当前工具数，当前 **30**） | N 与平台实际工具数不符（说明清单被硬编码了） |
| H3 | 同状态发 `你是谁` | 自述 + 同一份能力清单 | 退化成一句「你好」 |
| H4 | 同状态发 `这个平台怎么用` | 三条路径（直接说需求 / 先绑数据集 / 存成流程） | 落到说明性文案 |
| H5 | 同状态发 `今天天气不错`（拿不准的句子） | **说明性文案**：明说大模型「已停用」、凭据保留「不用重填」、数据类请求不受影响 | 硬聊天气（说明退化成了劣质聊天机器人） |
| H6 | 同状态发 `看看这批数据的分布`（已绑数据集） | **照常走工具流程并出结果** —— 证明关掉的是「生成」，不是「分析」 | 只回文案不执行 |
| H7 | 设置页**重新打开**开关 → 发 `你好` | 正常大模型答复（含自然语气），且**不含**「平台自带能力」那段 | 仍走内置应答（说明开关没接上） |
| H8 | 顺带：`未配置 Key` 的机器上发任意闲聊 | 文案说「**尚未配置**…填 API Key」 | 与 H5 的「已停用」文案混用 |

> H1–H6 与 H5/H8 的**区别就是本次修复的核心**：`llm is None` 有「主动停用」与「从未配置」两种来源，
> 合并成一句会让第一种用户跑去重填 Key。判定逻辑在 `local_chat.no_llm_notice()`。

---

## 4. 数据在哪、怎么读

### 4.1 文件位置

```
backend/models/local_router/
├── lexical_l1.pkl     # 模型产物（已 gitignore）
└── shadow.jsonl       # 埋点数据；超过 5 MB 轮转为 shadow.jsonl.1（只留一代）
```

### 4.2 一行一条，一次运行两行，用 `run_id` 关联

**`kind="route"`**（路由发生时）

| 字段 | 含义 |
| --- | --- |
| `run_id` / `session_id` | 关联键 |
| `utterance` | 用户原话 |
| `bound_dataset_id` / `n_columns` / `recent_tools` | 当时可得的结构化信号 |
| `rules.mode` / `rules.reason` | **既有规则路由**的判定（对照基线） |
| `router.available` | 模型是否可用；`false` 时带 `error` |
| `router.route` | `chat` / `call::<tool>` / `ask::<tool>` / `escalate::<reason>` |
| `router.confidence` | top1 概率（门控阈值的依据） |
| `router.escalate_reason` / `router.missing` | 升级原因 / 缺哪些必填槽位 |

**`kind="outcome"`**（运行结束时）

| 字段 | 含义 |
| --- | --- |
| `status` / `error` / `elapsed` | 运行结果 |
| `planned_tools` | 计划里的工具（planner 或 plan_override） |
| `executed_tools` | **真正执行过**的工具（这是「实际做法」的唯一定义） |

### 4.3 汇总分析

```bash
cd D:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend
./.venv/Scripts/python.exe scripts/router/analyze_shadow.py
```

输出 `models/local_router/shadow_analysis.{md,json}`，核心是把每条运行判成六种后果之一：

| 后果 | 含义 | 对「要不要开 guard」的意义 |
| --- | --- | --- |
| **可省** | Router 预测的工具 = 实际执行链第一步 | 接管可直接省一次 planner 调用（≈1.2 万 token） |
| 可省(部分) | 预测的工具在实际链里但不是第一步 | 接管会改顺序，需人工核对 |
| **会出错(误判闲聊)** | 实际走了工具流程，Router 说是 `chat` | **最危险**，接管会把数据请求当闲聊 |
| 会出错(工具不符) | 预测的工具与实际执行链完全不符 | 接管会跳过 planner 去做别的事 |
| 中性(升级) | Router 说 `escalate` | 与今天行为相同（仍交云端） |
| 待确认(反问) | Router 说 `ask::X` | 接管会反问用户，属行为变更 |

报告同时给出**置信度分布 + 门控模拟**（θ 以下转升级时，能拦掉多少「可省」、多少「会出错」）—— 这是之后选 `LOCAL_ROUTER_CONFIDENCE_THRESHOLD` 的依据。

---

## 5. 通过 / 不通过判据

| 判据 | 阈值 |
| --- | --- |
| A 组回归差异 | **0 处**（任何非「多出 trace 文件」的差异都算不通过） |
| 埋点缺失 | **0 条**（每条 run 都应有 route + outcome 两行；模型不可用时应是 `available=false` 而**不是**没有记录） |
| `router.available=false` 的比例 | **0**（>0 说明产物缺失/过期，回到 1.1） |
| E 组五类升级 | **5/5 原因正确** |
| 后端日志中的 trace 写盘 warning | **0 条** |
| 「会出错(误判闲聊)」 | 前 50 条内应为 **0**；若 >0，把句子原文记下来（这是最有价值的负样本） |

**达到以下条件才考虑切 `guard` 档**（本轮**不做**）：

1. 累计 ≥ 100 条真实配对数据；
2. 「会出错(误判闲聊)」= 0，且「会出错(工具不符)」占比 < 5%；
3. 门控模拟中存在一个 θ，使「可省」保留 ≥ 60% 同时「会出错」全部被拦掉；
4. A 组回归对照 0 差异。

---

## 6. 已知限制（诚实标注，别当成 bug）

1. **shadow 阶段没有 `available_columns`**：要拿到列名须解析 Parquet，代价不该由埋点承担。
   因此 trace 里的 `n_columns` 恒为 `0`。对照的准确率请用 `train_runtime_l1.py` 输出里的
   **「shadow 口径（无列）」** 那一列（当前 **80.8%**），不要用默认列（80.8%，实测两者几乎相同，
   说明列信息对本任务贡献极小 —— 但这是**在本数据集上**的结论）。
2. **升级语料是作者手写的 30 条**，规则与语料同源，存在**循环论证**风险。
   E 组的「硬性期望」只证明**实现与规则一致**，不证明规则在真实分布上够用。
   真实泛化必须靠这一轮收集的数据重新验证 —— 这正是本轮的目的。
3. **`ask::X` 的反问行为尚未在任何地方生效**：shadow 只记录，实际反问仍由 planner 决定。
   「待确认(反问)」这类样本需要你人工判断反问是否得体。
4. **参数抽取未实现**：除 `dataset_id` 外，其余必填参数一律假设下游（planner）补齐。
   所以 trace 里不会出现「缺 column / 缺 model」这类 `missing`。这是**有意为之**，
   详见 `app/local_router/router.py` 顶部第 1 条说明。
5. **并发写盘**：trace 用「线程锁 + 单次 append」，进程级安全；多进程同时写同一文件不保证（当前部署单进程）。
6. **`analyze_shadow.py` 的「可省」是必要条件而非充分条件**：实际跳过 planner 还要求
   参数齐全、权限允许、无高风险确认。别把它直接当成 token 节省量。

---

## 7. 出问题时按这个顺序定位

| 症状 | 先看 |
| --- | --- |
| 开了 shadow 但 `shadow.jsonl` 不出现 | `GET /settings/local_router`：`mode` 是否为 shadow、`artifact_loaded` 是否为 true |
| `router.available=false` | trace 里的 `router.error`；再跑 1.1 看 `staleness` |
| 判定明显离谱（如闲聊→建模型） | 先跑 C 组单句确认可复现，再看 `confidence`；低置信是**预期**行为，门控就是为它准备的 |
| 行为与关闭时不一致 | 1.1 体检 → 1.3 测试 → 检查 `runtime.py` 是否只有那两处 `_trace_*` 调用 |
| 状态端点 500 | 看 `data.probe_error`；通常是模型层 import 失败 |
| 测试报快照过期 | `./.venv/Scripts/python.exe scripts/router/export_contract.py` |
