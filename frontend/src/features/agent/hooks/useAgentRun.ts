/**
 * Agent 运行的全部状态与流程：发送、SSE、轮询、授权、取消。
 *
 * 页面不再持有这些状态，只负责布局；本 Hook 对外的契约是
 * 「给一组状态 + 几个动作」，不含任何 JSX。
 *
 * 与旧版相比的关键收敛：
 * - `activeRunId` 改为派生值（最后一个事件的 run_id ?? run.id），不再是需要在
 *   切会话时手工清空的 state —— 旧版漏清就会出现「对新会话点了旧 run 的允许」。
 * - 切会话的去重从 `switchSeqRef`（每个 await 后手工比对）改成 effect 的
 *   `cancelled` 闭包，少一类「忘了比对」的隐患。
 * - 进度 / 阶段的计算从 3 处各写一遍收敛为 `progressOfRun` / `stageOfRun`。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelRun,
  confirmRun,
  denyRun,
  getRun,
  sendMessage,
  type AgentToolInfo,
} from "../../../api/agent";
import { shouldAutoConfirm } from "../../../lib/toolPermissions";
import { toolDisplayName } from "../../../store/aiLab";
import type { AgentRun, ChatMessage, InspectorTab, PermissionRequest } from "../../../types/agent";
import { useAgentEvents, type EventEffects } from "./useAgentEvents";
import { useAgentUsage } from "./useAgentUsage";

const STATUS_LABEL: Record<string, string> = {
  pending: "待运行",
  planning: "规划中",
  running: "执行中",
  waiting_confirmation: "等待确认",
  completed: "已完成",
  failed: "失败",
};

/** 轮询失败上限：后端不可达 / run 被删除（404）时不能 2s 一次无限重试。 */
const MAX_POLL_FAILURES = 8;

function stageOfRun(full: AgentRun): string {
  if (full.status === "completed") return "任务完成";
  if (full.status === "failed") return "任务失败";
  return STATUS_LABEL[full.status] ?? full.status;
}

function progressOfRun(full: AgentRun): number {
  if (full.status === "completed" || full.status === "failed") return 100;
  const total = full.plan?.steps?.length || 0;
  const done = full.tool_call_count || 0;
  if (total) return Math.min(95, Math.round((done / total) * 100));
  return Math.min(90, 10 + (full.events?.length ?? 0) * 5);
}

function errText(e: unknown, fallback: string): string {
  return e instanceof Error ? e.message : fallback;
}

export interface UseAgentRunOptions {
  sessionId: string | null;
  /** 当前会话最后一次运行的 id；切会话时用它在面板里回填上一次执行。 */
  lastRunId: string | null;
  /** 会话切换计数（由 useAiSession 提供）：只有显式切换 / 新建 / 删除回退才 +1。 */
  switchToken: number;
  datasetIds: number[];
  tools: AgentToolInfo[];
  /** 没有会话时自动建一个，返回会话 id；失败返回 null。 */
  ensureSession: (title: string) => Promise<string | null>;
  appendMessage: (message: ChatMessage) => void;
  onError: (message: string | null) => void;
  onNotice: (message: string | null) => void;
  /** 切会话时成功回填了上一次运行——页面据此自动展开运行面板。 */
  onRunRestored?: () => void;
}

export function useAgentRun(opts: UseAgentRunOptions) {
  const { sessionId, lastRunId, switchToken, datasetIds, tools, ensureSession, appendMessage, onError, onNotice, onRunRestored } = opts;

  const [run, setRun] = useState<AgentRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("等待任务");
  const [permission, setPermission] = useState<PermissionRequest | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("overview");

  const { events, setEvents, push, clear } = useAgentEvents(tools);
  const { usage, live, history, resetHistory } = useAgentUsage(events, run, busy);

  const pollRef = useRef<number | null>(null);
  /** SSE 流的取消句柄：离开页面 / 切会话时主动中断，避免后端 tail 线程挂到超时上限。 */
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false);
  /** 已自动放行过的授权请求（run_id:step_index:tool），避免同一请求被无限自动确认。 */
  const autoAllowedRef = useRef<Set<string>>(new Set());
  /** 会话代次：任何 await 之后比对，不是最新一代就丢弃结果。 */
  const seqRef = useRef(0);

  // 运行中 SSE 事件已经带 run_id；流结束后才有 run 对象。二者取其一即可，
  // 不再单独维护一个需要在切会话时清空的 activeRunId。
  const activeRunId = events.length ? events[events.length - 1].run_id : (run?.id ?? null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const abortStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (runId: string) => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      let failures = 0;
      // 后台标签页暂停轮询：`document.hidden` 为 true 时跳过本次请求，回到前台自动恢复。
      pollRef.current = window.setInterval(async () => {
        if (document.hidden) return;
        try {
          const full = await getRun(runId);
          failures = 0;
          setRun(full);
          setEvents(full.events ?? []);
          setProgress(progressOfRun(full));
          setStage(stageOfRun(full));
          if (full.pending_confirmation) setPermission(full.pending_confirmation);
          if (full.status !== "pending" && full.status !== "planning" && full.status !== "running") {
            stopPolling();
          }
        } catch (e) {
          failures += 1;
          if (failures >= MAX_POLL_FAILURES) {
            stopPolling();
            setStage("状态获取失败");
            onError(`无法获取运行状态（${errText(e, "网络错误")}），请刷新页面后重试。`);
          }
        }
      }, 2000);
    },
    [setEvents, onError, stopPolling],
  );

  /** 拉取单次运行详情并落到面板（SSE onDone 之后使用）。 */
  const refreshRun = useCallback(
    async (runId: string) => {
      const seq = seqRef.current;
      try {
        const full = await getRun(runId);
        if (seq !== seqRef.current) return;
        setRun(full);
        setEvents(full.events ?? []);
        setProgress(progressOfRun(full));
        setStage(stageOfRun(full));
        if (full.pending_confirmation) setPermission(full.pending_confirmation);
        setInspectorTab(full.tool_calls.length ? "chain" : "activity");
      } catch {
        /* 流已结束，轮询会兜底 */
      }
    },
    [setEvents],
  );

  /** 把一条事件的增量应用到界面。授权请求在这里触发自动放行判定。 */
  const applyEffects = useCallback(
    (fx: EventEffects, runId: string) => {
      if (fx.stage) setStage(fx.stage);
      if (fx.progress !== undefined) setProgress(fx.progress);
      if (fx.notice) onNotice(fx.notice);
      if (fx.tab) setInspectorTab(fx.tab);
      if (fx.append) appendMessage(fx.append);
      if (!fx.permission) return;
      const req = fx.permission;
      const autoKey = `${runId}:${req.step_index}:${req.tool}`;
      // 设置页把该工具设为「自动放行」时直接替用户确认；仍失败则回退为手动弹窗。
      if (shouldAutoConfirm(req.tool, tools) && !autoAllowedRef.current.has(autoKey)) {
        autoAllowedRef.current.add(autoKey);
        void autoAllow(runId, req);
        return;
      }
      setPermission(req);
      setStage("等待确认");
      setProgress(95);
    },
    // autoAllow 在下方定义，此处读取的是最近一次渲染的闭包；它只用 ref 与 setter，行为稳定。
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tools, appendMessage, onNotice],
  );

  /** 设置页已把该工具设为「自动放行」时的免确认路径。失败回退为手动授权弹窗。 */
  const autoAllow = useCallback(
    async (runId: string, req: PermissionRequest) => {
      const seq = seqRef.current;
      try {
        const resumed = await confirmRun(runId);
        if (seq !== seqRef.current) return;
        setRun(resumed);
        setEvents(resumed.events ?? []);
        setPermission(null);
        onNotice(`已按设置自动放行「${toolDisplayName(req.tool, tools)}」。`);
        if (resumed.final_answer) {
          appendMessage({
            role: "assistant",
            content: resumed.final_answer,
            source: resumed.answer_source ?? null,
          });
        }
        if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) {
          setPermission(resumed.pending_confirmation);
        }
        startPolling(runId);
      } catch (e) {
        if (seq !== seqRef.current) return;
        setPermission(req);
        onError(errText(e, "自动放行失败，请手动确认"));
      }
    },
    [tools, appendMessage, onNotice, onError, setEvents, startPolling],
  );

  const send = useCallback(
    async (content: string) => {
      if (!content.trim() || sendingRef.current) return;
      sendingRef.current = true;
      // 先建会话再动运行状态：ensureSession 会写入 sessionId，
      // 若把 setProgress(5) 放在它之前，切会话 effect 有可能在之后把进度清零。
      setBusy(true);
      onError(null);
      let sid = sessionId;
      if (!sid) {
        sid = await ensureSession(content.slice(0, 30) || "数据分析会话");
        if (!sid) {
          setBusy(false);
          sendingRef.current = false;
          return;
        }
      }
      // 发送即开启一条新 SSE 流：先中断上一条（正常情况下 busy 会挡住并发，这里是防御性兜底）。
      abortStream();
      const controller = new AbortController();
      abortRef.current = controller;
      appendMessage({ role: "user", content });
      setProgress(5);
      setStage("理解任务");
      clear();
      onNotice(null);
      setPermission(null);
      let runId: string | null = null;
      try {
        await sendMessage(sid, {
          content,
          stream: true,
          datasetIds,
          signal: controller.signal,
          onEvent: (ev) => {
            runId = ev.run_id;
            applyEffects(push(ev), ev.run_id);
          },
          onDone: () => {
            setBusy(false);
            sendingRef.current = false;
            if (runId) {
              void refreshRun(runId);
              startPolling(runId);
            }
          },
        });
      } catch (e) {
        // abort 导致的异常是「正常中断」（切会话 / 卸载 / 新请求顶替），不应当成发送失败报错。
        const aborted = controller.signal.aborted || (e instanceof DOMException && e.name === "AbortError");
        if (!aborted) onError(errText(e, "发送失败"));
        setBusy(false);
        sendingRef.current = false;
        if (runId && !aborted) startPolling(runId);
      }
    },
    [
      sessionId,
      datasetIds,
      ensureSession,
      appendMessage,
      onError,
      onNotice,
      abortStream,
      clear,
      push,
      applyEffects,
      refreshRun,
      startPolling,
    ],
  );

  /**
   * 确认授权。
   *
   * 历史缺陷：这里原来是 `if (!run || busy) return;` —— SSE 流还没结束时 run 仍为 null、
   * busy 为 true，于是点击「允许」必然静默 return：不发请求、不报错、弹窗也不关，
   * 而「拒绝」用的是事件里已经赋值的 activeRunId，所以看起来只有拒绝能点。
   * 现在与 deny / stop 统一取 `activeRunId`，并且不再用发送锁阻塞授权请求。
   */
  const allow = useCallback(async () => {
    const target = activeRunId;
    if (!target || confirming) return;
    const seq = seqRef.current;
    setConfirming(true);
    onError(null);
    try {
      const resumed = await confirmRun(target);
      if (seq !== seqRef.current) return; // 授权期间已切会话，丢弃结果
      setRun(resumed);
      setEvents(resumed.events ?? []);
      setPermission(null);
      onNotice(null);
      setStage(stageOfRun(resumed));
      if (resumed.final_answer) {
        appendMessage({ role: "assistant", content: resumed.final_answer });
      }
      if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) {
        setPermission(resumed.pending_confirmation);
      }
      startPolling(target);
    } catch (e) {
      // 失败时保留弹窗，让用户可以重试或直接拒绝，而不是把授权请求丢掉。
      if (seq === seqRef.current) onError(errText(e, "确认失败"));
    } finally {
      setConfirming(false);
    }
  }, [activeRunId, confirming, appendMessage, onError, onNotice, setEvents, startPolling]);

  const deny = useCallback(() => {
    // 拒绝必须通知后端终止 run，否则 run 卡在 WAITING_CONFIRMATION，会话被 409 锁死。
    setPermission(null);
    appendMessage({ role: "assistant", content: "已拒绝该高风险操作的授权，对应步骤不会执行。" });
    const target = activeRunId;
    if (!target) return;
    const seq = seqRef.current;
    void denyRun(target)
      .then((denied) => {
        if (seq !== seqRef.current) return;
        setRun(denied);
        setEvents(denied.events ?? []);
      })
      .catch(() => {
        /* 后端不可达时保留本地提示，轮询会兜底同步 */
      });
  }, [activeRunId, appendMessage, setEvents]);

  /**
   * 请求取消当前运行。
   * 后端只在「步骤边界」检查 cancel_requested，长步骤（训练 / 报告生成 / 大模型响应）
   * 期间不会立刻停止，因此补上轮询兜底，让进度条继续反映真实状态。
   */
  const stop = useCallback(async () => {
    const target = activeRunId;
    if (!target) return;
    setStage("正在取消…");
    onError(null);
    try {
      await cancelRun(target);
      setStage("已请求取消，将在当前步骤结束后停止");
      startPolling(target);
    } catch (e) {
      onError(errText(e, "取消失败"));
      setStage("取消失败");
    }
  }, [activeRunId, onError, startPolling]);

  /**
   * 会话镜像：切会话 / 新建 / 删除后回退时，中断上一条流与轮询、清空本轮状态，
   * 再按会话最后一次运行回填面板。
   *
   * 依赖 `switchToken` 而不是 `sessionId`：send() 内部会自动建会话（sessionId 由 null 变新值），
   * 若以 sessionId 为依赖，这条 effect 会在发送途中清空刚收到的事件。
   */
  useEffect(() => {
    let cancelled = false;
    seqRef.current += 1;
    stopPolling();
    abortStream();
    autoAllowedRef.current.clear();
    clear();
    setRun(null);
    setPermission(null);
    setProgress(0);
    setStage("等待任务");
    setInspectorTab("overview");
    resetHistory();
    if (!lastRunId) {
      return () => {
        cancelled = true;
        stopPolling();
        abortStream();
      };
    }
    void (async () => {
      try {
        const full = await getRun(lastRunId);
        if (cancelled) return;
        setRun(full);
        setEvents(full.events ?? []);
        setProgress(progressOfRun(full));
        setStage(stageOfRun(full));
        if (full.pending_confirmation) setPermission(full.pending_confirmation);
        setInspectorTab(full.tool_calls.length ? "chain" : "activity");
        onRunRestored?.();
      } catch {
        if (!cancelled) {
          clear();
          setRun(null);
        }
      }
    })();
    return () => {
      cancelled = true;
      stopPolling();
      abortStream();
    };
  }, [switchToken, lastRunId, clear, resetHistory, setEvents, stopPolling, abortStream, onRunRestored]);

  // 卸载时收尾：中断 SSE，避免后端 tail 线程挂到超时上限。
  useEffect(
    () => () => {
      stopPolling();
      abortStream();
    },
    [stopPolling, abortStream],
  );

  return {
    run,
    events,
    busy,
    progress,
    stage,
    permission,
    confirming,
    inspectorTab,
    setInspectorTab,
    activeRunId,
    usage,
    live,
    history,
    send,
    allow,
    deny,
    stop,
  };
}
