/**
 * `useAgentRun` 生命周期回归测试：本轮修的两个 P0 都出在 React Effect 上。
 *
 * 为什么必须真渲染才能测
 * ----------------------
 * 这两个缺陷**不是**纯函数算错，而是「Effect 在不该跑的时候跑了」：
 *
 * 1. `onRunRestored` 是页面传进来的内联箭头函数，每次 render 都是新引用；
 *    它在「会话镜像 Effect」的依赖数组里 ⇒ effect 每次 render 重跑 ⇒
 *    执行 `stopPolling() / abortStream() / clear() / setRun(null) / setInspectorTab("overview")`。
 *    用户看到的就是：AI 不回应、运行消失、SSE 自己断、Inspector 页签锁死在概览。
 * 2. 页签被 `refreshRun()` / `tool_call` 事件无条件改写，用户手动选的页签停不住。
 *
 * 依赖数组的稳定性只能靠「真的 render 一次看 effect 有没有重跑」来证伪，
 * 所以这里用 jsdom + react-dom 真实挂载（基础设施见 tests/dom.mjs），
 * 全程离线：`fetch` 与 `getRun` 都由本地替身接管，不发任何真实请求。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { installDom, mountPoint } from "./dom.mjs";

installDom();

// DOM 就绪之后才加载 react-dom（它在挂载时才需要 document，顺序颠倒会拿不到容器）
const React = (await import("react")).default;
const { createRoot } = await import("react-dom/client");
const { act } = await import("react-dom/test-utils");
const { useAgentRun } = await import("../src/features/agent/hooks/useAgentRun.ts");
type RunApi = ReturnType<typeof useAgentRun>;

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

// ---------------------------------------------------------------------------
// 替身：SSE 流
// ---------------------------------------------------------------------------

interface StreamCtl {
  push(type: string, payload: Record<string, unknown>): void;
  events: number;
  aborts: number;
}

/** 一个可控的 SSE 流：测试自己决定什么时候往下推事件。 */
function installSseFetch(): StreamCtl {
  const encoder = new TextEncoder();
  const ctl: StreamCtl & { controller?: ReadableStreamDefaultController } = {
    push(type, payload) {
      const frame = `event: ${type}\ndata: ${JSON.stringify({ seq: 1, run_id: "r-1", type, payload, created_at: 0 })}\n\n`;
      ctl.controller?.enqueue(encoder.encode(frame));
    },
    events: 0,
    aborts: 0,
  };

  (globalThis as unknown as { fetch: unknown }).fetch = async (_url: string, init: { signal?: AbortSignal } = {}) => {
    init.signal?.addEventListener("abort", () => {
      ctl.aborts += 1;
    });
    const body = new ReadableStream({
      start(controller) {
        ctl.controller = controller;
      },
    });
    return { ok: true, status: 200, body } as unknown as Response;
  };
  return ctl;
}

// ---------------------------------------------------------------------------
// 挂载装置
// ---------------------------------------------------------------------------

interface HarnessProps {
  sessionId: string | null;
  lastRunId: string | null;
  switchToken: number;
  tools: never[];
}

function mountHarness(initial: HarnessProps) {
  const api: { current: RunApi | null } = { current: null };
  let props = initial;
  // 计数：会话镜像 Effect 若在不该跑的时候跑了，这里一定会被加一。
  const counters = { restored: 0, renders: 0, errors: [] as string[] };

  function Harness(p: HarnessProps) {
    counters.renders += 1;
    const run = useAgentRun({
      sessionId: p.sessionId,
      lastRunId: p.lastRunId,
      switchToken: p.switchToken,
      datasetIds: [],
      tools: p.tools,
      ensureSession: async () => "s-new",
      appendMessage: () => undefined,
      onError: (message) => {
        if (message) counters.errors.push(message);
      },
      onNotice: () => undefined,
      // ★ 刻意写成内联箭头函数：真实页面就是这样传的，每次 render 都是新引用。
      // 修复前这一条足以让「会话镜像 Effect」每次 render 重跑。
      onRunRestored: () => {
        counters.restored += 1;
      },
    });
    api.current = run;
    return null;
  }

  const container = mountPoint();
  const root = createRoot(container);

  const render = async (patch: Partial<HarnessProps> = {}) => {
    props = { ...props, ...patch };
    await act(async () => {
      root.render(React.createElement(Harness, props));
      await flush();
    });
  };

  return {
    api,
    counters,
    render,
    state: () => api.current as RunApi,
    unmount: async () => {
      await act(async () => {
        root.unmount();
        await flush();
      });
      container.remove();
    },
  };
}

/** 监听 `AbortController.abort()` —— 「SSE 被谁掐断」的唯一直接证据。 */
function watchAborts() {
  type AbortProto = { abort: (...args: unknown[]) => void };
  const proto = (globalThis as unknown as { AbortController: { prototype: AbortProto } }).AbortController.prototype;
  const original = proto.abort;
  let count = 0;
  proto.abort = function patched(this: unknown, ...args: unknown[]) {
    count += 1;
    return original.apply(this, args);
  };
  return {
    get count() {
      return count;
    },
    restore() {
      proto.abort = original;
    },
  };
}

const BASE: HarnessProps = { sessionId: "s-1", lastRunId: null, switchToken: 0, tools: [] };

// ---------------------------------------------------------------------------
// 一、Inspector 页签：用户选过之后不许被任何自动逻辑打回
// ---------------------------------------------------------------------------

test("Inspector 页签不被 rerender 打回 overview", async () => {
  const aborts = watchAborts();
  const h = await mountHarness(BASE);
  await h.render(); // 首次挂载

  assert.equal(h.state().inspectorTab, "overview");

  // 用户点「Token」看用量
  await act(async () => {
    h.state().setInspectorTab("token");
    await flush();
  });
  assert.equal(h.state().inspectorTab, "token");

  // 连续 rerender（模拟收到 SSE / usage / progress / stage / busy 变化后的重渲染）
  for (let i = 0; i < 3; i += 1) await h.render();

  assert.equal(h.state().inspectorTab, "token", "普通 rerender 不许改写用户选中的页签");
  // 会话镜像 Effect 没重跑 ⇒ 没有清理、没有恢复回调
  assert.equal(h.counters.restored, 0, "lastRunId 为空时不该触发恢复回调");
  assert.equal(h.state().run, null, "effect 重跑会把 run 清空为 null，这里必须保持");
  assert.equal(aborts.count, 0, "没有切会话就不该有任何 abort");

  aborts.restore();
  await h.unmount();
});

test("tool_call 事件尊重用户已选中的页签（未选过时仍自动定位）", async () => {
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  // 未手动选过 ⇒ 自动定位仍然生效
  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("tool_call", { tool: "dataset.quality", status: "ok" });
    await flush();
  });
  assert.equal(h.state().inspectorTab, "activity", "未手动选择时 tool_call 应自动切到活动");

  // 用户切到「预算」
  await act(async () => {
    h.state().setInspectorTab("budget");
    await flush();
  });
  await act(async () => {
    stream.push("tool_call", { tool: "dataset.profile", status: "ok" });
    await flush();
  });
  assert.equal(h.state().inspectorTab, "budget", "用户选过之后不许被 tool_call 弹回活动");

  await h.unmount();
});

// ---------------------------------------------------------------------------
// 二、SSE 不许被前端自己误杀
// ---------------------------------------------------------------------------

test("SSE 收到事件不触发 abort", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  assert.equal(h.state().busy, true, "发送后应处于忙碌态");

  // 连续推事件：每一次都会触发 setState ⇒ rerender。修复前这就是掐断流的时刻。
  for (const [type, payload] of [
    ["route", { mode: "agent", reason: "要分析" }],
    ["planning", { stage: "plan_ready", steps: 2 }],
    ["usage", { token_usage: { actual: { total_tokens: 10 } } }],
    ["tool_call", { tool: "dataset.quality", status: "ok" }],
  ] as const) {
    await act(async () => {
      stream.push(type, payload);
      await flush();
    });
  }

  assert.ok(h.state().events.length >= 4, `事件应被接收，实际 ${h.state().events.length} 条`);
  assert.equal(stream.aborts, 0, "SSE 收到事件不得触发 abort");
  assert.equal(aborts.count, 0, "整个过程中的 rerender 都不许 abort");
  assert.equal(h.state().busy, true, "流还在进行，busy 不该被清掉");
  assert.deepEqual(h.counters.errors, [], "不该有发送失败之类的报错");

  aborts.restore();
  await h.unmount();
});

// ---------------------------------------------------------------------------
// 三、只有切会话 / 卸载 / 新请求顶替才允许清理旧流
// ---------------------------------------------------------------------------

test("切会话才允许清理旧流（并中断上一条 SSE）", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();

  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("route", { mode: "agent", reason: "要分析" });
    await flush();
  });
  assert.equal(h.state().events.length, 1);

  // 切会话：switchToken +1
  await h.render({ switchToken: 1 });

  assert.equal(stream.aborts, 1, "切会话必须中断上一条 SSE（且只中断一次）");
  assert.equal(aborts.count, 1);
  assert.equal(h.state().events.length, 0, "切会话后事件应被清空");
  assert.equal(h.state().run, null);
  assert.equal(h.state().inspectorTab, "overview", "切会话是真正的重置，页签回到概览");

  aborts.restore();
  await h.unmount();
});

test("卸载时才中断 SSE，卸载前不中断", async () => {
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();
  await act(async () => {
    void h.state().send("hi");
    await flush();
  });

  assert.equal(stream.aborts, 0);
  await h.unmount();
  assert.equal(stream.aborts, 1, "卸载必须中断 SSE，避免后端 tail 线程挂到超时上限");
});

test("lastRunId 变化（未切会话）不清理当前运行", async () => {
  const aborts = watchAborts();
  const stream = installSseFetch();
  const h = await mountHarness(BASE);
  await h.render();
  await act(async () => {
    void h.state().send("检查一下数据质量");
    await flush();
  });
  await act(async () => {
    stream.push("route", { mode: "agent", reason: "要分析" });
    await flush();
  });

  // 会话列表刷新把 lastRunId 换了（switchToken 没变）——修复前这也会触发整轮清理
  await h.render({ lastRunId: "r-1" });

  assert.equal(stream.aborts, 0, "lastRunId 变化不等于切会话，不许中断流");
  assert.equal(h.state().events.length, 1, "当前运行的事件必须保留");

  aborts.restore();
  await h.unmount();
});
