/** SSE 事件 → 界面增量（lib/agentEvents）单元测试。
 *
 * 为什么值得测：这段映射里埋过三类真实 bug ——
 *   1. 重试时进度不回退，用户看不到「卡住后重来」；
 *   2. 授权事件的字段没做归一，缺字段时弹窗拿不到 tool / step_index；
 *   3. 校验失败事件没有给出「第几步 + 原因」，只在界面上无声跳过。
 * 放在 lib/ 下就是为了能用 Node 原生 runner 直接跑（无 DOM 依赖）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { eventEffects } from "../src/lib/agentEvents.ts";
import type { AgentEvent } from "../src/types/agent.ts";

function ev(type: AgentEvent["type"], payload: Record<string, unknown>): AgentEvent {
  return { seq: 1, run_id: "run-1", type, payload, created_at: 0 };
}

/** progress 可能是定值或 updater，统一取值。 */
function progressOf(fx: ReturnType<typeof eventEffects>, prev: number): number {
  if (fx.progress === undefined) return prev;
  return typeof fx.progress === "function" ? fx.progress(prev) : fx.progress;
}

const label = (tool: string) => (tool === "ml.train" ? "训练模型" : tool);

test("route：对话模式与工具模式给出不同起步进度", () => {
  assert.equal(eventEffects(ev("route", { mode: "chat", reason: "寒暄" })).stage, "对话模式 · 寒暄");
  assert.equal(progressOf(eventEffects(ev("route", { mode: "chat", reason: "寒暄" })), 5), 40);
  assert.equal(progressOf(eventEffects(ev("route", { mode: "agent", reason: "要分析" })), 5), 8);
});

test("planning：plan_ready 推进到 25，未知阶段回落到 8", () => {
  assert.equal(progressOf(eventEffects(ev("planning", { stage: "plan_ready" })), 5), 25);
  assert.equal(eventEffects(ev("planning", { stage: "unknown_stage" })).stage, "unknown_stage");
  assert.equal(progressOf(eventEffects(ev("planning", { stage: "unknown_stage" })), 5), 8);
});

test("tool_call：切到活动页签并推进进度，且不超过 90", () => {
  const fx = eventEffects(ev("tool_call", { tool: "ml.train" }), label);
  assert.equal(fx.tab, "activity");
  assert.equal(fx.stage, "执行工具：训练模型");
  assert.equal(progressOf(fx, 10), 18);
  assert.equal(progressOf(eventEffects(ev("tool_call", { tool: "ml.train" }), label), 88), 90);
});

test("replanning：进度显式回退且有下限 5（历史 bug：曾停在原数字不动）", () => {
  const fx = eventEffects(ev("replanning", { retry: true, notes: "工具超时" }), label);
  assert.equal(fx.stage, "重试中 · 工具超时");
  assert.equal(fx.notice, "工具超时");
  assert.equal(progressOf(fx, 60), 45);
  assert.equal(progressOf(eventEffects(ev("replanning", {})), 10), 5);
});

test("validation：失败给出「第几步 + 原因」，通过时不打扰用户", () => {
  const failed = eventEffects(ev("validation", { valid: false, step_index: 2, errors: ["缺 target", "缺 features"] }));
  assert.equal(failed.notice, "第 3 步校验失败：缺 target；缺 features");
  // step_index 缺失时按第 1 步算，不能出现 NaN
  assert.match(eventEffects(ev("validation", { valid: false })).notice ?? "", /第 1 步/);
  assert.equal(eventEffects(ev("validation", { valid: true })).notice, undefined);
});

test("permission：字段归一，缺字段时给出可用默认值", () => {
  const fx = eventEffects(ev("permission", { tool: "data.write", reason: "会写库", step_index: 3, arguments: { a: 1 } }));
  assert.deepEqual(fx.permission, { tool: "data.write", arguments: { a: 1 }, reason: "会写库", step_index: 3 });
  const minimal = eventEffects(ev("permission", { tool: "data.write" }));
  assert.deepEqual(minimal.permission, { tool: "data.write", arguments: {}, reason: "", step_index: 0 });
  // 没有 tool 名时不产生授权请求：否则会弹出一个无法确认的空弹窗
  assert.equal(eventEffects(ev("permission", {})).permission, undefined);
});

test("completed / failed：把最终答案与失败原因追加为助手消息", () => {
  const done = eventEffects(ev("completed", { final_answer: "分析完成", answer_source: { source: "remote", label: "远程", detail: "d", by_llm: true, model: "m" } }));
  assert.equal(done.append?.role, "assistant");
  assert.equal(done.append?.content, "分析完成");
  assert.equal(done.append?.source?.by_llm, true);
  assert.equal(progressOf(done, 12), 100);
  const failed = eventEffects(ev("failed", { error: "工具不存在" }));
  assert.equal(failed.append?.content, "执行失败：工具不存在");
});

test("usage 事件不产生界面增量：Token 账本由 useAgentUsage 从事件列表派生", () => {
  assert.deepEqual(eventEffects(ev("usage", { token_usage: { llm_calls: 1 } })), {});
});

test("planning：远程规划降级到内置规则时必须显式告知用户", () => {
  // 「数据还是真的，但计划不是大模型定的」——不说出来用户会误以为是模型分析的结果
  const degraded = eventEffects(ev("planning", { stage: "plan_ready", planner_fallback: true }));
  assert.match(String(degraded.notice), /内置规则/);
  // 没有降级标记时不该弹提示，否则每次规划都打扰一次
  assert.equal(eventEffects(ev("planning", { stage: "plan_ready" })).notice, undefined);
  assert.equal(eventEffects(ev("planning", { stage: "plan_ready", planner_fallback: false })).notice, undefined);
});
