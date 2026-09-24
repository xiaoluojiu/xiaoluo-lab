# -*- coding: utf-8 -*-
"""生成本轮「AI 实验室稳定性修复 + 容错闭环」交付文档（含全部改动文件的完整代码）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "AI实验室稳定性修复与容错闭环.md"

# (相对路径, 语言, 改动要点)
FILES = [
    ("backend/app/agent/llm/base.py", "python", [
        "新增 `LLMException.fallbackable`：区分「可降级的远程故障」与「不该降级的程序 bug / 权限错误」。",
        "新增 `_BUG_ERRORS`（TypeError / AttributeError / NameError / ImportError / NotImplementedError / AssertionError / RecursionError …）判定清单。",
        "新增 `is_fallbackable_error(exc)`（错误分类 SSOT）与 `fallback_allowed(exc)`（总闸 × 错误分类）。",
    ]),
    ("backend/app/agent/llm/openai_compatible.py", "python", [
        "新增配额类错误识别：`_QUOTA_HINTS`（insufficient_quota / quota / billing / balance / credit / payment / arrears / 欠费 / 余额不足 / 账户余额）+ HTTP 402。",
        "新增 `_error_envelope(body)` 解析 OpenAI 兼容协议 `{\"error\": {...}}`，从 `code` / `type` / `message` 三个字段一起嗅探。",
        "配额/欠费类 429 直接抛不可重试、可降级的 `LLMException`，一次都不重试；真正的瞬时限流才有限重试，并且优先遵守 `Retry-After`，没有则指数退避。",
    ]),
    ("backend/app/agent/planner/models.py", "python", [
        "`AgentPlan` 新增 `planner_fallback: bool` 字段，标记「这份计划是规则规划器降级产出的」，供运行时与前端展示真实来源。",
    ]),
    ("backend/app/agent/planner/planner.py", "python", [
        "`build_plan_resilient()` 原来只捕获 `PlanInvalidError`，远程 LLM 抛 `LLMException` 时直接冒泡 ⇒ 数据分析任务整体失败。",
        "改为捕获 `(PlanInvalidError, LLMException)`，按「候选工具集 → 全量工具集」两轮尝试，最终落到 Rule Planner 并置 `plan.planner_fallback = True`。",
        "新增 `_fallback_allowed(exc)` = 总闸 `AGENT_ALLOW_MODEL_FALLBACK` × 错误分类；程序 bug 与不可降级异常原样抛出，绝不静默吞掉。",
        "一次 run 内远程最多尝试 2 次（两轮 scope），规则规划是终点，不存在循环 fallback。",
    ]),
    ("backend/app/agent/runtime/runtime.py", "python", [
        "P0-2 根因：`_direct_chat()` 捕获异常后直接返回硬编码的「暂时无法完成对话请求」，却把来源标成 `LLM_ERROR_FALLBACK` ⇒ 用户看到报错文案、系统却声称已降级。",
        "改为：可降级则真的调用 `local_reply(..., degraded=True)` 生成回答并标 `LLM_ERROR_FALLBACK`；本地答不上则 `no_llm_notice(reason=...)` 并标 `NO_ANSWER`。",
        "`_compose_answer()` 走同一套判定；远程返回空内容时若总闸关闭则抛 `LLMException` 而不是假装降级。",
        "`run()` 新增 `except LLMException` 分支，保证远程故障落为 `FAILED` 且事件流闭合。",
        "`plan_ready` 事件 payload 透出 `planner_fallback`，前端据此提示。",
    ]),
    ("backend/app/agent/local_chat.py", "python", [
        "新增 `_DEGRADED_NOTE` 与 `_note(*, degraded: bool)`：把「远程大模型已停用」和「远程大模型这一次调用失败」区分开，不再对用户撒谎。",
        "6 个私有应答函数与 `local_reply()` 统一增加 `degraded` 关键字参数（默认 False，兼容既有调用与测试）。",
        "`no_llm_notice(reason)` 在 reason 非空时把真实失败原因写进文案。",
    ]),
    ("backend/app/agent/answer_source.py", "python", [
        "`NO_ANSWER` 的说明补全为「失败 / 被拒绝 / 被取消 / 远程调用失败且内置规则未覆盖」，作为来源语义的一部分而不是兜底垃圾值。",
    ]),
    ("backend/app/core/config.py", "python", [
        "`AGENT_ALLOW_MODEL_FALLBACK` 由 `False` 改为 `True`：它现在是真总闸（planner 与 runtime 都真正读取），平台推荐默认开启本地兜底。",
    ]),
    ("backend/app/api/v1/settings.py", "python", [
        "`AgentSettingsUpdateRequest` 新增 `allow_model_fallback` 字段，去掉「接口里写死 `settings.AGENT_ALLOW_MODEL_FALLBACK = False`」的假配置行为。",
    ]),
    ("backend/tests/test_agent_llm_fallback.py", "python", [
        "新增 22 项回归测试，覆盖第十一节要求的全部后端场景（普通聊天兜底 / no_answer / 数据任务 Rule Planner 接管 / quota 429 只请求一次 / transient 429 有限重试 / Answer Source 不撒谎）。",
    ]),
    ("backend/tests/test_answer_source.py", "python", [
        "原用例断言的是「报错文案 + LLM_ERROR_FALLBACK」这一错误行为，本轮按新的正确语义更新断言。",
    ]),
    ("frontend/src/features/agent/hooks/useAgentRun.ts", "ts", [
        "P0-1 根因一：会话镜像 Effect 依赖数组里放了 `onRunRestored`，而页面传的是内联箭头函数 ⇒ 每次 render 引用都变 ⇒ Effect 每次重跑 ⇒ `stopPolling() / abortStream() / clear() / setRun(null) / setInspectorTab('overview')`。",
        "依赖数组收敛为 `[switchToken]`（只有真正切会话才变），回调经 `onRunRestoredRef` 读取最新值；`lastRunId` 同样改为 ref 读取。",
        "P0-1 根因二：页签被 `refreshRun()` / `applyEffects()` 无条件改写。新增 `tabPinnedRef` + `setInspectorTab`（用户手动选，置 pin）与 `autoInspectorTab`（自动切，pin 住则不动）两条路径。",
        "`startPolling()` 增加代次校验 `seq !== seqRef.current` 后立即停轮询，防止旧会话的轮询结果覆盖新会话状态。",
    ]),
    ("frontend/src/pages/AI/index.tsx", "tsx", [
        "`onRunRestored` 由内联箭头函数改为 `useCallback(() => setPanelOpen(true), [])`，引用恒定——这是让上面 Effect 依赖纪律真正生效的前提。",
    ]),
    ("frontend/src/lib/agentEvents.ts", "ts", [
        "`planning.plan_ready` 且 `planner_fallback === true` 时产出明确 notice：「远程规划不可用，已改用平台内置规则规划（执行结果仍是真实工具跑出来的）。」",
    ]),
    ("frontend/src/api/settings.ts", "ts", [
        "`updateAgentSettings` 新增可选 `allow_model_fallback`，让前端能真正写后端这个总闸。",
    ]),
    ("frontend/src/pages/Settings/AgentModelPanel.tsx", "tsx", [
        "保存时透传 `allow_model_fallback`；策略区新增「模型兜底（远程失败降级）」勾选项。",
    ]),
    ("frontend/tests/useAgentRun.test.ts", "ts", [
        "新增 6 项 React 生命周期回归测试（页签不被 rerender 打回 / 用户手动选择优先 / SSE 收事件不 abort / 只有切会话才清理 / 卸载才中断 / lastRunId 变化不清理）。",
    ]),
    ("frontend/tests/dom.mjs", "js", [
        "新增最小 jsdom 挂载基础设施（唯一新增 devDependency 就是 jsdom），不引入 testing-library / vitest / jest。",
    ]),
    ("frontend/tests/bootstrap.mjs", "js", [
        "预置 `globalThis.__VITE_ENV__`，供 resolve-hook 替换 `import.meta.env`。",
    ]),
    ("frontend/tests/resolve-hook.mjs", "js", [
        "新增 `load()` 钩子：把 `import.meta.env` 替换为 `globalThis.__VITE_ENV__`；`patchSource()` 同时兼容 string 与 Uint8Array 两种 source 形态。",
    ]),
    ("frontend/tests/agentEvents.test.ts", "ts", [
        "新增「远程规划降级到内置规则时必须显式告知用户」用例。",
    ]),
    ("frontend/package.json", "json", [
        "devDependencies 新增 `jsdom@25`（仅为让 React 生命周期测试能在 node:test 里真实挂载组件）。",
    ]),
]

HEAD = """# AI 实验室核心运行问题修复（P0 稳定性 + 容错闭环 + 回归测试）

> 本轮为**稳定性修复与容错闭环修复**，不是新功能开发。范围严格控制在：Effect 生命周期、SSE 不被误杀、本地兜底闭环、Planner 降级、429/quota 识别、来源语义、回归测试。
> 未重写 Agent Runtime / AI 页面 / 状态管理框架 / 网络层，未引入 Redis / Celery / 消息队列 / 第二套执行引擎。

---

## ① 根因

### P0-1　Effect 生命周期错误（AI 不回应 / 运行消失 / SSE 中断 / Inspector 页签锁死概览）

| # | 根因 | 位置 | 表现 |
|---|---|---|---|
| 1 | 会话镜像 Effect 的依赖数组里放了 `onRunRestored`，而 `AI/index.tsx` 传的是**内联箭头函数** `() => setPanelOpen(true)`。每次 render 都产生新引用 ⇒ Effect 每次 render 都重跑一次 | `useAgentRun.ts` 会话镜像 Effect + `pages/AI/index.tsx` | Effect 体里的 `stopPolling() / abortStream() / clear() / setRun(null)` 被反复执行 ⇒ **运行凭空消失、SSE 被自己 abort、请求发出去没有响应** |
| 2 | 同一 Effect 体里还无条件 `setInspectorTab("overview")`；另有 `refreshRun()` 与 `applyEffects()` 两条路径也会改写页签 | `useAgentRun.ts` | 用户切到「活动 / 工具链 / 产物」后，任意一次 rerender（收到 SSE 事件、usage 更新、父组件渲染）就被打回「概览」⇒ **页签锁死** |
| 3 | `startPolling()` 的 `await getRun(...)` 之后没有代次校验 | `useAgentRun.ts` | 快速切会话时，旧会话的轮询结果后到并覆盖新会话状态 ⇒ 状态回退 |

**修复后的纪律**：会话镜像 Effect 的依赖数组**只允许 `switchToken`**（真正切会话/新建/回退时才变）；所有回调与 `lastRunId` 一律通过 ref 读取最新值；「会话切换」与「恢复完成后通知页面」两个职责分离。

### P0-2　普通聊天远程失败没有真的降级（报错文案 + `LLM_ERROR_FALLBACK` 自相矛盾）

`_direct_chat()` 的 `except` 分支捕获远程异常后，**直接返回硬编码的「暂时无法完成对话请求」**，却把 `run.answer_source` 标成 `LLM_ERROR_FALLBACK`。结果是：用户看到的是一句报错，系统记录的却是「已降级为本地回答」——来源在撒谎，兜底闭环根本没走。

**修复后**：远程失败 → 先判 `fallback_allowed(exc)` → 真的调 `local_reply(utterance, degraded=True)` → 有回答标 `LLM_ERROR_FALLBACK`，没有回答标 `NO_ANSWER` 并给 `no_llm_notice(reason=真实原因)`。

### P1　容错与配置层面的四个缺口

| # | 缺口 | 修复 |
|---|---|---|
| 1 | `build_plan_resilient()` 只捕获 `PlanInvalidError`，`LLMException` 直接冒泡 ⇒ 数据任务在远程挂掉时整体失败，永远到不了 Rule Planner | 捕获 `(PlanInvalidError, LLMException)`，按「候选工具集 → 全量工具集」两轮后落 Rule Planner，置 `planner_fallback=True` |
| 2 | `_is_retryable_status()` 只有 `status >= 500 or status == 429` ⇒ 欠费型 429 会傻傻重试到超时 | 解析 `error.code/type/message` 与 HTTP 402：quota / billing / insufficient_quota / balance / credit 直接转可降级的 `LLMException` **一次都不重试**；真瞬时限流才有限重试并遵守 `Retry-After` |
| 3 | `AGENT_ALLOW_MODEL_FALLBACK` 默认 `False`，且 `settings.py` 里被**写死**成 `False`，前端只能显示不能写 ⇒ 假配置 | 默认改为 `True`；接口新增 `allow_model_fallback` 可写；planner / runtime 真正读取它作为总闸 |
| 4 | 缺少回归测试，上述行为没有任何东西兜住 | 后端 +22 项、前端 +7 项，含 React 生命周期与 SSE abort 计数 |

---

## ② 修改文件列表

**后端（11）**
{backend_list}

**前端（11）**
{frontend_list}

---

## ③ 每个改动文件的完整修改后代码

> 以下为**修改后文件的完整内容**，不是 diff。

"""


def fence(lang: str, body: str) -> str:
    return "````" + lang + "\n" + body.rstrip("\n") + "\n````\n"


def main() -> None:
    parts = [HEAD]
    backend_list, frontend_list = [], []
    for rel, _lang, _pts in FILES:
        line = f"- `{rel}`"
        (backend_list if rel.startswith("backend/") else frontend_list).append(line)
    parts[0] = parts[0].replace("{backend_list}", "\n".join(backend_list))
    parts[0] = parts[0].replace("{frontend_list}", "\n".join(frontend_list))

    for idx, (rel, lang, points) in enumerate(FILES, start=1):
        path = ROOT / rel
        body = path.read_text(encoding="utf-8")
        parts.append(f"\n### {idx}. `{rel}`\n\n")
        for p in points:
            parts.append(f"- {p}\n")
        parts.append("\n")
        parts.append(fence(lang, body))

    tail = """
---

## ④ 测试结果

### 后端 pytest

```
$ cd backend && ./.venv/Scripts/python.exe -m pytest -p no:randomly -q --no-header --tb=line -rf
收集用例：1187
EXIT = 0    FAILED = 0    ERROR = 0
```

本次新增 / 受影响的用例：

| 文件 | 用例数 | 结果 |
|---|---|---|
| `tests/test_agent_llm_fallback.py`（新） | 22 | 全绿 |
| `tests/test_answer_source.py` | 全量 | 全绿 |
| `tests/test_local_chat.py` | 全量 | 全绿 |
| `tests/test_config.py` | 全量 | 全绿 |
| `tests/test_agent.py` | 40 | 全绿 |
| `tests/test_agent_e2e.py` | 19 | 全绿 |

过程中真实发生过、并已修掉的问题（不甩锅给环境）：

1. **`_capability_reply() missing 1 required keyword-only argument: 'degraded'`** —— 给 `local_chat.py` 的 6 个私有应答函数与 `_note()` 的 `degraded` 参数补默认值 `False`，保持对既有调用兼容。
2. **`test_chat_with_llm_error_is_marked_as_degraded` 失败** —— 该用例原本断言的正是「报错文案 + `LLM_ERROR_FALLBACK`」这一错误行为，按新语义更新断言。
3. **`useAgentRun.ts(167,15): TS2304 Cannot find name 'seq'`** —— `startPolling()` 里补 `const seq = seqRef.current;`。
4. **前端测试崩溃 `Cannot read properties of undefined (reading 'DEV')`** —— Node 的 `load()` 钩子对 `.ts` 返回的 `source` 是 `Uint8Array`（不是 string），字符串替换没生效；改为同时处理两种形态，并在 bootstrap 里预置 `globalThis.__VITE_ENV__`。
5. **`test_answer_source_never_lies` 的「远程失败+本地答不上」格断言不成立** —— 该格与「本地能答」格复用了同一个问法「你好」（本地规则能覆盖），改为参数化新增 `utterance`，「答不上」两格改用「讲个笑话」。
6. **全量 pytest 出现 5 个连带失败** —— 根因是 `test_fallback_flag_default_is_on` 里用了 `importlib.reload(app.core.config)`：reload 换掉了模块里的 `settings` 单例，后续用例的 monkeypatch 全部打空（表现为「改 A 处、B 处挂」）。改为 `Settings(_env_file=None).AGENT_ALLOW_MODEL_FALLBACK is True`，不 reload。
7. **回归验证**：把 Effect 依赖改回 `[switchToken, lastRunId, onRunRestored]` 后，新前端测试立刻触发 React `Maximum update depth exceeded` 无限更新循环 —— 这是 P0-1 的硬证据，随后恢复为 `[switchToken]`。

### 前端

```
$ cd frontend && npm run typecheck   → EXIT=0（0 error）
$ cd frontend && npm run test        → # tests 61  # pass 61  # fail 0
$ cd frontend && npm run build       → ✓ built in 5.73s   EXIT=0
```

前端测试由 54 → 61（新增 `useAgentRun` 6 项 + `agentEvents` 1 项）。

---

## ⑤ 修改前后行为对比

| 场景 | 修改前 | 修改后 |
|---|---|---|
| **A. 远程正常** | `remote_llm_chat`，但一次 rerender 就可能把运行和 SSE 干掉 | `remote_llm_chat`，运行稳定、SSE 持续、页签保持 |
| **B. 欠费 / quota 429 + 普通聊天** | 无脑重试到超时，最后返回「暂时无法完成对话请求」且标 `LLM_ERROR_FALLBACK`（撒谎） | 一次请求即判定不可重试 → `local_reply(degraded=True)` → `llm_error_fallback`，`{"source":"llm_error_fallback","by_llm":false}` |
| **C. 欠费 + 本地答不上** | 同上，一句报错 | `no_llm_notice(reason=真实原因)` → `no_answer` |
| **D. 欠费 + 数据分析任务** | `LLMException` 冒泡 ⇒ 整条 run 失败，永远到不了 Rule Planner | 两轮远程尝试失败后 Rule Planner 接管 → 真实执行 `dataset.quality` 并返回真实结果，`plan.planner_fallback=True`，前端显式告知「已改用平台内置规则规划」 |
| **E. Inspector 页签** | 切到「活动/工具链」后，任意 rerender 被打回「概览」 | 用户手动选择后 pin 住；自动切换只在未 pin 时生效 |
| **F. SSE 生命周期** | 父组件 rerender 就 `controller.abort()` | 只有「切换会话 / 卸载页面 / 用户显式停止 / 新请求顶替旧请求」四种情况才 abort |
| **G. 切会话后的状态** | 旧会话轮询结果后到，覆盖新会话状态（状态回退） | `startPolling` 带代次校验，代次不符立即停轮询 |
| **H. `AGENT_ALLOW_MODEL_FALLBACK`** | 默认 False 且接口写死 False，形同虚设 | 默认 True，前后端可写，planner/runtime 真正作为总闸读取；`False` 时远程失败直接失败不假装降级 |
| **I. 程序 bug（TypeError 等）** | 与远程故障一样被兜底吞掉，问题被掩盖 | `is_fallbackable_error()` 判定为不可降级，原样抛出 |
"""
    parts.append(tail)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"written: {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
