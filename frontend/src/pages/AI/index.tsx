import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./ai-lab.css";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { InfoHint } from "../../components/InfoHint";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { ChatPanel, type ChatMessage } from "../../features/agent/ChatPanel";
import { AgentTimeline } from "../../features/agent/AgentTimeline";
import { ToolCallCard } from "../../features/agent/ToolCallCard";
import { PermissionRequest } from "../../features/agent/PermissionRequest";
import { SmartAnalysisButton } from "../../features/agent/SmartAnalysisButton";
import { CapabilitiesPanel } from "../../features/agent/CapabilitiesPanel";
import { createSession, confirmRun, denyRun, cancelRun, getRun, listSessions, runTraceUrl, sendMessage, archiveSession, deleteSession, getCapabilities } from "../../api/agent";
import { generateReport } from "../../api/reports";
import { useAiLab, toolDisplayName } from "../../store/aiLab";
import { shouldAutoConfirm } from "../../lib/toolPermissions";
import type { AgentCapabilities, AgentEvent, AgentRun, AgentSession, AgentTokenUsage, AnswerSource, PermissionRequest as PermissionRequestType } from "../../types/agent";

const STATUS_LABEL: Record<string, string> = { pending: "待运行", planning: "规划中", running: "执行中", waiting_confirmation: "等待确认", completed: "已完成", failed: "失败" };
const STAGE_LABEL: Record<string, string> = { context_ready: "准备上下文", tools_retrieved: "检索工具", plan_ready: "执行计划", direct_chat: "普通对话" };
type InspectorTab = "overview" | "activity" | "chain" | "token" | "budget";

/** 工具调用状态文案。 */
const TOOL_STATUS_LABEL: Record<string, string> = { pending: "等待中", ok: "成功", failed: "失败", needs_confirmation: "待授权", denied: "已拒绝" };

/** 一次会话内多次运行的用量快照，用于「每次对话的调用次数与调用量变化」。 */
interface RunUsageSnapshot {
  run_id: string;
  at: number;
  status: string;
  llm_calls: number;
  total_tokens: number;
  tool_calls: number;
  elapsed_seconds: number;
}

const QUICK_ACTIONS = [
  { title: "检查数据质量", description: "缺失值、重复值、字段类型与异常概览", prompt: "请检查当前关联的数据集质量，给出缺失值、重复值、字段类型和明显异常的摘要。" },
  { title: "探索数据", description: "生成关键统计，并指出值得进一步分析的变量", prompt: "请对当前关联的数据集做一次探索性分析，给出关键统计、变量关系和最值得继续分析的问题。" },
  { title: "设计机器学习实验", description: "根据数据与目标提出可执行的建模方案", prompt: "请根据当前数据集设计一个机器学习实验方案，说明目标变量、特征、候选模型、评价指标和下一步执行建议。" },
];

function planStepText(step: unknown) { if (typeof step === "string") return step; if (step && typeof step === "object") { const item = step as Record<string, unknown>; return String(item.title ?? item.goal ?? item.action ?? JSON.stringify(step)); } return String(step); }

function relativeTime(ts: number): string {
  const sec = Math.round(Date.now() / 1000 - ts);
  if (sec < 60) return "刚刚";
  if (sec < 3600) return `${Math.floor(sec / 60)} 分钟前`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} 小时前`;
  return `${Math.floor(sec / 86400)} 天前`;
}
function sessionTitle(s: AgentSession): string {
  if (s.title && s.title !== "新会话" && s.title !== "新数据分析会话") return s.title;
  const firstUser = s.history.find((h) => h.role === "user")?.content ?? "";
  if (firstUser) return `${firstUser.slice(0, 30)}${firstUser.length > 30 ? "…" : ""}`;
  return "未命名实验";
}
/** 与会话列表保持同一排序口径：未归档在前，各自按创建时间倒序。 */
function reorderSessions(items: AgentSession[]): AgentSession[] {
  return [...items].sort((a, b) => (a.archived ? 1 : 0) - (b.archived ? 1 : 0) || b.created_at - a.created_at);
}

/** Token 摘要的折叠态一行文案。 */
function tokenBrief(run: AgentRun): string {
  const actual = run.token_usage?.actual; const budget = run.token_usage?.budget;
  if (!actual) return "暂无数据";
  const remaining = budget ? budget.remaining_total_tokens.toLocaleString() : "—";
  return `${actual.total_tokens.toLocaleString()} 已用 · ${remaining} 剩余`;
}

/**
 * 「花了多少 / 省了多少」——**常驻**显示在面板顶部，不藏在 Token 页签里。
 *
 * 历史问题：这两组数字原先只在「Token」页签可见，而运行过程中页签默认停在
 * 「活动」，用户跑一次任务下来完全不知道消耗了多少、平台替他省了多少。
 * 现在放在页签之上，任何页签下都看得到；运行中用 `usage` 事件实时刷新。
 */
function TokenCostStrip({ usage, live }: { usage: AgentTokenUsage | null; live: boolean }) {
  const actual = usage?.actual;
  if (!actual) return null;
  const opt = usage?.optimization;
  const saved = opt?.estimated_saved_tokens ?? 0;
  const remaining = usage?.budget?.remaining_total_tokens;
  return (
    <div className="ai-token-strip">
      <div className="ai-token-strip-row">
        <span className="ai-token-strip-kicker">本次实际消耗</span>
        <strong>{actual.total_tokens.toLocaleString()} Token</strong>
        <span className="muted">
          输入 {actual.input_tokens.toLocaleString()} / 输出 {actual.output_tokens.toLocaleString()}
          {" · "}LLM 调用 {usage?.llm_calls ?? 0} 次
          {remaining != null && ` · 预算剩余 ${remaining.toLocaleString()}`}
        </span>
        {live && <em className="ai-token-live">实时</em>}
      </div>
      <div className="ai-token-strip-row">
        <span className="ai-token-strip-kicker">平台帮你省下</span>
        <strong className="ai-token-saved">≈ {saved.toLocaleString()} Token</strong>
        <span className="muted">
          结果压缩 {(opt?.estimated_result_saved_tokens ?? 0).toLocaleString()}
          {" + "}上下文优化 {(opt?.estimated_context_saved_tokens ?? 0).toLocaleString()}
          {" · "}免去规划调用 {opt?.avoided_planner_calls ?? 0} 次
          {opt?.plan_cache_hits ? ` · Plan Cache 命中 ${opt.plan_cache_hits} 次` : ""}
        </span>
        <InfoHint label="两组数字的口径">
          「实际消耗」来自大模型服务上报的真实 usage；「省下」是平台本地估算值
          （结果压缩 + 上下文优化 + 免掉的规划调用），两者口径不同，不要相加或混用。
        </InfoHint>
      </div>
    </div>
  );
}

/** 回答来源徽标：明确这条回答是远程模型生成的，还是平台内置规则 / 降级兜底的。 */
function AnswerSourceChip({ source }: { source: AnswerSource | null | undefined }) {
  if (!source) return null;
  return (
    <div className={`ai-source-chip${source.by_llm ? " remote" : " local"}`} title={source.detail}>
      <span className="chat-source-dot" aria-hidden="true" />
      <span>
        {source.by_llm ? "由远程大模型回答" : "由平台内置规则回答"}
        {" · "}
        {source.label}
        {source.model ? `（${source.model}）` : ""}
      </span>
      <InfoHint label="为什么能确定来源">{source.detail}</InfoHint>
    </div>
  );
}

/** Token 预算与用量明细。仅保留数据本体，卡片外壳由外层容器提供。 */
function TokenUsageBody({ run }: { run: AgentRun }) {
  const actual = run.token_usage?.actual; const optimization = run.token_usage?.optimization; const budget = run.token_usage?.budget;
  if (!actual || !optimization) return <p className="muted" style={{ margin: 0 }}>本次运行未上报 Token 用量。</p>;  const tokenPct = budget ? Math.min(100, Math.round((actual.total_tokens / Math.max(budget.max_total_tokens, 1)) * 100)) : 0;
  const callPct = budget ? Math.min(100, Math.round((run.token_usage.llm_calls / Math.max(budget.max_llm_calls, 1)) * 100)) : 0;
  return (
    <div className="ai-token-inner">
      <div className="ai-run-summary">
        <div><strong>{actual.total_tokens.toLocaleString()}</strong><span>已用 Token</span></div>
        <div><strong>{budget ? budget.remaining_total_tokens.toLocaleString() : "—"}</strong><span>剩余 Token</span></div>
        <div><strong>{budget ? `${run.token_usage.llm_calls}/${budget.max_llm_calls}` : run.token_usage.llm_calls}</strong><span>LLM 调用</span></div>
      </div>
      <p className="ai-token-note">
        下方带「估算」的条目是本机估算值，其余为 Provider 实际用量，两者不混用。
        <InfoHint label="Token 用量口径说明">
          “实际 Token”来自 Provider usage；“节省”是平台本地估算值，两者不混用。
        </InfoHint>
      </p>
      <div className="ai-token-bars">
        <div className="ai-token-bar"><div className="ai-token-bar-head"><span>输入 / 输出</span><strong>{actual.input_tokens.toLocaleString()} / {actual.output_tokens.toLocaleString()}</strong></div></div>
        <div className="ai-token-bar"><div className="ai-token-bar-head"><span>Token 预算</span><strong>{tokenPct}%</strong></div><div className="ai-progress-track"><div className="ai-progress-fill" style={{ width: `${tokenPct}%`, background: tokenPct > 80 ? "var(--danger)" : undefined }} /></div></div>
        <div className="ai-token-bar"><div className="ai-token-bar-head"><span>调用次数预算</span><strong>{callPct}%</strong></div><div className="ai-progress-track"><div className="ai-progress-fill" style={{ width: `${callPct}%`, background: callPct > 80 ? "var(--danger)" : undefined }} /></div></div>
        <div className="ai-context-row"><span>结果压缩估算节省</span><strong>{optimization.estimated_result_saved_tokens.toLocaleString()} Token</strong></div>
        <div className="ai-context-row"><span>上下文优化估算节省</span><strong>{optimization.estimated_context_saved_tokens.toLocaleString()} Token</strong></div>
        <div className="ai-context-row"><span>Plan Cache 命中</span><strong>{optimization.plan_cache_hits} 次</strong></div>
      </div>
    </div>
  );
}

/** 多次对话的调用次数与调用量变化（含与上一次的差值）。 */
function RunUsageHistory({ history }: { history: RunUsageSnapshot[] }) {
  if (!history.length) return <p className="muted" style={{ margin: 0 }}>当前会话还没有完成过运行。</p>;
  const maxTokens = Math.max(...history.map((h) => h.total_tokens), 1);
  return (
    <table className="data-table ai-history-table">
      <thead>
        <tr><th>#</th><th>时间</th><th>LLM 调用</th><th>Token 总量</th><th>较上次</th><th>工具步骤</th><th>耗时</th></tr>
      </thead>
      <tbody>
        {history.map((item, i) => {
          const prev = i > 0 ? history[i - 1] : null;
          const dTokens = prev ? item.total_tokens - prev.total_tokens : null;
          const dCalls = prev ? item.llm_calls - prev.llm_calls : null;
          return (
            <tr key={item.run_id}>
              <td>{i + 1}</td>
              <td>{new Date(item.at * 1000).toLocaleTimeString("zh-CN", { hour12: false })}</td>
              <td>
                {item.llm_calls}
                {dCalls != null && <em className={`ai-delta${dCalls > 0 ? " up" : dCalls < 0 ? " down" : ""}`}>{dCalls > 0 ? ` +${dCalls}` : ` ${dCalls}`}</em>}
              </td>
              <td>
                <div className="ai-history-bar-wrap">
                  <span>{item.total_tokens.toLocaleString()}</span>
                  <div className="ai-progress-track"><div className="ai-progress-fill" style={{ width: `${Math.round((item.total_tokens / maxTokens) * 100)}%` }} /></div>
                </div>
              </td>
              <td>{dTokens == null ? "—" : <span className={dTokens > 0 ? "ai-delta up" : dTokens < 0 ? "ai-delta down" : "ai-delta"}>{dTokens > 0 ? `+${dTokens.toLocaleString()}` : dTokens.toLocaleString()}</span>}</td>
              <td>{item.tool_calls}</td>
              <td>{item.elapsed_seconds.toFixed(1)}s</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function AI() {
  const { sessionId, setSession, setDatasetIds, loadTools, tools } = useAiLab();
  const navigate = useNavigate();
  const [selectedDatasets, setSelectedDatasets] = useState<number[]>([]); const [sessions, setSessions] = useState<AgentSession[]>([]); const [messages, setMessages] = useState<ChatMessage[]>([]); const [events, setEvents] = useState<AgentEvent[]>([]); const [run, setRun] = useState<AgentRun | null>(null); const [permission, setPermission] = useState<PermissionRequestType | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null); const [inspectorTab, setInspectorTab] = useState<InspectorTab>("overview"); const [generatingReport, setGeneratingReport] = useState(false); const [progress, setProgress] = useState(0); const [stage, setStage] = useState("等待任务"); const [activeRunId, setActiveRunId] = useState<string | null>(null); const [notice, setNotice] = useState<string | null>(null); const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  // 运行面板：空闲默认收起，任务启动 / 切到带运行的会话时自动展开；用户可随时手动收起。
  const [panelOpen, setPanelOpen] = useState(false);
  // 运行面板高度档位：默认 / 放大（面板原高度偏小，长工具链与用量表看不全）。
  const [panelSize, setPanelSize] = useState<"normal" | "large">("normal");
  /**
   * 运行**过程中**的 Token 账本快照（来自新增的 `usage` 事件）。
   * 运行结束后 `run.token_usage` 才是权威值，但过程中轮询还没回来时只能靠它，
   * 否则「跑了半天不知道花了多少」。
   */
  const [liveUsage, setLiveUsage] = useState<AgentTokenUsage | null>(null);
  // 会话列表：宽屏常开；窄屏（<1280）可折叠成一行入口。
  const [listOpen, setListOpen] = useState(true);
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);
  // Agent 能力声明（/agent/capabilities）：步数上限、Token 预算、模型与策略。
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null);
  const [capsLoading, setCapsLoading] = useState(true);
  const [capsError, setCapsError] = useState<string | null>(null);
  // 删除会话的二次确认目标（走统一 ConfirmDialog）。
  const [pendingDelete, setPendingDelete] = useState<AgentSession | null>(null);
  // 批量删除 / 清空：{ ids, all }，all=true 表示「清空全部」。
  const [pendingBulk, setPendingBulk] = useState<{ ids: string[]; all: boolean } | null>(null);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  // 会话多选（批量归档 / 批量删除 / 清空）。
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  // 授权提交中：用于弹窗按钮的 loading 与防连点。
  const [confirming, setConfirming] = useState(false);
  // 当前会话内多次运行的用量快照，用于观察调用次数 / 调用量的变化。
  const [runHistory, setRunHistory] = useState<RunUsageSnapshot[]>([]);
  const pollRef = useRef<number | null>(null); const sendingRef = useRef(false);
  // 已自动放行过的授权请求（run_id:step_index:tool），避免同一请求被无限自动确认。
  const autoAllowedRef = useRef<Set<string>>(new Set());
  /**
   * 会话切换序号：每次切会话 +1。
   * 作用：`switchSessionAsync` 里有 await getRun(...)，若用户在请求返回前又切到别的会话，
   * 慢响应会把旧会话的 run / events / 授权弹窗覆盖到新会话上（pollRef 只能挡轮询，挡不住这个 await）。
   * 每个 await 之后比对序号，不是最新一次就丢弃结果。
   */
  const switchSeqRef = useRef(0);
  /** SSE 流的取消句柄：离开页面 / 切会话时主动中断，避免后端 tail 线程挂到 900s。 */
  const abortRef = useRef<AbortController | null>(null);
  /**
   * 面板自动展开只应发生在「任务开始」这一次事件上，不能压制用户的手动收起。
   * 用户点收起时置 true；运行结束（busy 转 false）时重置。
   */
  const panelUserCollapsedRef = useRef(false);
  const displayMessages = useMemo(() => messages.filter((m, i, a) => i === 0 || m.content !== a[i - 1].content || m.role !== a[i - 1].role), [messages]);

  useEffect(() => {
    void loadTools();
    listSessions().then(async (items) => {
      setSessions(items);
      const saved = items.find((s) => s.id === sessionId) ?? items[0];
      if (saved && !run && !messages.length) await switchSessionAsync(saved);
    }).catch(() => setSessions([]));
  }, []);

  // 拉取 Agent 能力声明，展示在检查器「预算」页签。
  function loadCapabilities() {
    setCapsLoading(true);
    setCapsError(null);
    getCapabilities()
      .then(setCapabilities)
      .catch((e: unknown) => setCapsError(e instanceof Error ? e.message : "读取 Agent 能力失败"))
      .finally(() => setCapsLoading(false));
  }
  useEffect(() => { loadCapabilities(); }, []);
  /**
   * 数据集上下文只允许「有内容」时向外同步。
   *
   * 历史缺陷链：挂载时 `selectedDatasets` 是空数组，而「从 Dataset Workspace 带 ?dataset=7 跳入
   * 自动选中」的副作用藏在 DatasetSelector 内部、要等它自己拉完数据集列表才触发。
   * 于是这个 effect 会先用**空数组**把 store 里的 datasetIds 覆盖掉；
   * 若 send() 发生在 DatasetSelector 完成之前，`createSession` / `sendMessage` 取到的就是空上下文
   * —— 界面上显示「已关联 1 个数据集」，Agent 实际按「未关联数据集」执行。
   * 现在：空数组不写入 store，store 里保留已有值；由 DatasetSelector 真正选中后再同步。
   */
  useEffect(() => { if (selectedDatasets.length) setDatasetIds(selectedDatasets); }, [selectedDatasets, setDatasetIds]);
  /**
   * 面板自动展开：只在「任务开始」这一次转变上触发（busy 由 false → true），
   * 并在出现待授权弹窗时强制展开。
   * 历史缺陷：原来依赖 [busy, permission] 且无条件 setPanelOpen(true)，
   * 导致运行期间用户点「收起运行面板」后，只要 permission 变化就被再次弹开，
   * 与注释里「用户可随时手动收起」的承诺不符。
   * 现在：用户显式收起后，直到本轮运行结束都不再自动展开。
   */
  const prevBusyRef = useRef(false);
  useEffect(() => {
    if (busy && !prevBusyRef.current) panelUserCollapsedRef.current = false; // 新一轮任务开始：重置收起意图
    if (!busy) panelUserCollapsedRef.current = false;                          // 任务结束：复位
    if (permission) { setPanelOpen(true); return; }                            // 授权弹窗必须可见
    if (busy && !panelUserCollapsedRef.current) setPanelOpen(true);
    prevBusyRef.current = busy;
  }, [busy, permission]);
  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    if (abortRef.current) abortRef.current.abort(); // 卸载时中断 SSE，避免后端 tail 线程挂到 900s
  }, []);

  /** 记录一次运行的用量快照（同 run_id 覆盖更新）。 */
  function pushRunHistory(source: AgentRun) {
    const snapshot: RunUsageSnapshot = {
      run_id: source.id,
      at: Date.now() / 1000,
      status: source.status,
      llm_calls: source.token_usage?.llm_calls ?? 0,
      total_tokens: source.token_usage?.actual?.total_tokens ?? 0,
      tool_calls: source.tool_call_count ?? source.tool_calls?.length ?? 0,
      elapsed_seconds: source.elapsed_seconds ?? 0,
    };
    setRunHistory((prev) => {
      const idx = prev.findIndex((item) => item.run_id === snapshot.run_id);
      if (idx === -1) return [...prev, snapshot];
      const next = [...prev]; next[idx] = snapshot; return next;
    });
  }

  async function switchSessionAsync(s: AgentSession) {
    const seq = ++switchSeqRef.current;
    if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; } // U-9：切走会话必须停掉旧 run 的轮询，避免覆盖新会话状态
    // 切会话同样要中断上一条 SSE 流，否则旧流的 onEvent 会继续往新会话里塞事件。
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
    setSession(s.id); setSelectedDatasets(s.dataset_ids ?? []); setMessages(s.history.map((h) => ({ role: h.role, content: h.content }))); setPermission(null); setError(null); setNotice(null); setActiveRunId(null); setMenuSessionId(null); setSelectedIds([]); setRunHistory([]); setLiveUsage(null); autoAllowedRef.current.clear();
    panelUserCollapsedRef.current = false;
    const latestRunId = s.run_ids?.[s.run_ids.length - 1];
    if (!latestRunId) { setEvents([]); setRun(null); setProgress(0); setStage("等待任务"); return; }
    try {
      const full = await getRun(latestRunId);
      if (seq !== switchSeqRef.current) return; // 已被更晚的切换取代，丢弃慢响应
      setRun(full); setEvents(full.events ?? []); setProgress(full.status === "completed" || full.status === "failed" ? 100 : Math.min(95, Math.max(8, (full.events?.length ?? 1) * 8))); setStage(full.status === "completed" ? "任务完成" : STATUS_LABEL[full.status] ?? full.status); if (full.pending_confirmation) setPermission(full.pending_confirmation); setInspectorTab(full.tool_calls.length ? "chain" : "activity"); setPanelOpen(true); pushRunHistory(full);
    } catch {
      if (seq !== switchSeqRef.current) return;
      setEvents([]); setRun(null);
    }
  }

  async function newSession() { setError(null); setMenuSessionId(null); try { const s = await createSession(selectedDatasets, "新数据分析会话"); setSession(s.id); setDatasetIds(selectedDatasets); setSessions((prev) => [s, ...prev.filter((item) => item.id !== s.id)]); setMessages([]); setEvents([]); setRun(null); setPermission(null); setProgress(0); setStage("等待任务"); setInspectorTab("overview"); setSelectedIds([]); } catch (e) { setError(e instanceof Error ? e.message : "创建会话失败"); } }
  function switchSession(s: AgentSession) { void switchSessionAsync(s); }

  /** 归档 / 取消归档。归档只是把会话沉到列表底部，历史与运行记录都保留。 */
  async function toggleArchive(s: AgentSession) {
    const next = !s.archived;
    setSessions((prev) => reorderSessions(prev.map((item) => item.id === s.id ? { ...item, archived: next } : item)));
    setNotice(next ? "会话已归档，已移至列表底部（可再次点击取消归档）。" : "会话已取消归档。");
    try {
      const saved = await archiveSession(s.id, next);
      setSessions((prev) => reorderSessions(prev.map((item) => item.id === s.id ? { ...item, ...saved } : item)));
    } catch (e) {
      setSessions((prev) => reorderSessions(prev.map((item) => item.id === s.id ? { ...item, archived: !next } : item)));
      setError(e instanceof Error ? e.message : "归档失败");
    }
  }

  /** 批量归档：逐个调用，失败项保留在列表中并汇总提示。 */
  async function archiveSelected(ids: string[]) {
    if (!ids.length) return;
    const targets = sessions.filter((s) => ids.includes(s.id) && !s.archived);
    if (!targets.length) { setNotice("选中的会话都已归档。"); return; }
    let ok = 0; const failed: string[] = [];
    for (const s of targets) {
      try { const saved = await archiveSession(s.id, true); ok++; setSessions((prev) => reorderSessions(prev.map((item) => item.id === s.id ? { ...item, ...saved } : item))); }
      catch { failed.push(sessionTitle(s)); }
    }
    setSelectedIds([]);
    setNotice(failed.length ? `已归档 ${ok} 个会话；${failed.length} 个失败：${failed.join("、")}` : `已归档 ${ok} 个会话。`);
  }

  /**
   * 删除会话（单条 / 批量 / 清空共用）。
   * 后端只提供单条删除且活动会话会返回 409，这里逐个删除并汇总失败项，
   * 不因为其中一条失败就中断整个批量操作。
   */
  async function removeSessions(ids: string[]) {
    setPendingDelete(null); setPendingBulk(null); setMenuSessionId(null);
    const targets = sessions.filter((s) => ids.includes(s.id));
    if (!targets.length) return;
    setBulkDeleting(true); setError(null);
    let ok = 0; const failed: string[] = []; const removed: string[] = [];
    for (const s of targets) {
      setDeletingSessionId(s.id);
      try {
        await deleteSession(s.id);
        ok++; removed.push(s.id);
        setSessions((prev) => prev.filter((item) => item.id !== s.id));
      } catch (e) {
        failed.push(`${sessionTitle(s)}（${e instanceof Error ? e.message : "删除失败"}）`);
      } finally { setDeletingSessionId(null); }
    }
    // 注意：这里必须用函数式更新，不能基于函数开头的 `sessions` 闭包快照重算。
    // 历史缺陷：原来写成 `setSessions(sessions.filter(...))`，会把删除循环里刚刚成功移除的
    // 条目又用旧快照加回来（只要批里有任意一条失败就会触发），要等下次刷新才消失。
    let remaining = 0;
    setSessions((prev) => { const next = prev.filter((item) => !removed.includes(item.id)); remaining = next.length; return next; });
    const currentRemoved = sessionId ? removed.includes(sessionId) : false;
    if (currentRemoved) {
      // 删掉的正是当前会话：清空工作区，若还有其它会话则自动切到最近一个。
      if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
      if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
      switchSeqRef.current += 1; // 让在途的 getRun 结果失效，避免回写已删除会话的 run
      setSession(null); setMessages([]); setEvents([]); setRun(null); setPermission(null); setProgress(0); setStage("等待任务"); setActiveRunId(null); setRunHistory([]);
    }
    setSelectedIds((prev) => prev.filter((id) => !removed.includes(id)));
    setBulkDeleting(false);
    if (failed.length) setError(`成功删除 ${ok} 个会话；${failed.length} 个未删除：${failed.join("；")}`);
    else setNotice(ok > 1 ? `已删除 ${ok} 个会话及其运行记录。` : `已删除会话「${sessionTitle(targets[0])}」。`);
    // 自动切换到剩余会话放在状态更新之后，避免与上面的 setSessions 竞态。
    if (currentRemoved && remaining > 0) {
      const rest = sessions.filter((item) => !removed.includes(item.id));
      if (rest.length) void switchSessionAsync(rest[0]);
    }
  }

  function startPolling(runId: string) {
    if (pollRef.current) window.clearInterval(pollRef.current);
    // 轮询失败上限：后端不可达 / run 被删除（404）时不能 2s 一次无限重试。
    // 约 8 次（≈16s）仍拿不到状态就停表并提示，避免静默刷请求。
    let failures = 0;
    // 后台标签页暂停轮询：`document.hidden` 为 true 时跳过本次请求，回到前台自动恢复。
    pollRef.current = window.setInterval(async () => {
      if (document.hidden) return;
      try {
        const full = await getRun(runId);
        failures = 0;
        setRun(full); setEvents(full.events ?? []);
        const total = full.plan?.steps?.length || 0;
        const done = full.tool_call_count || 0;
        setProgress(full.status === "completed" || full.status === "failed" ? 100 : total ? Math.min(95, Math.round((done / total) * 100)) : Math.min(90, 10 + (full.events?.length ?? 0) * 5));
        setStage(full.status === "completed" ? "任务完成" : full.status === "failed" ? "任务失败" : STATUS_LABEL[full.status] ?? full.status);
        if (full.status !== "pending" && full.status !== "planning" && full.status !== "running") {
          pushRunHistory(full);
          if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
        }
      } catch (e) {
        // SSE 断开时轮询仍会尽力恢复状态；连续失败达上限则停止并提示。
        failures += 1;
        if (failures >= 8) {
          if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
          setStage("状态获取失败");
          setError(`无法获取运行状态（${e instanceof Error ? e.message : "网络错误"}），请刷新页面后重试。`);
        }
      }
    }, 2000);
  }

  /** 拉取单次运行详情并落到面板（onDone 后使用）。 */
  async function refreshRun(runId: string) {
    const seq = switchSeqRef.current;
    try {
      const full = await getRun(runId);
      if (seq !== switchSeqRef.current) return; // 切会话后丢弃
      setRun(full); setEvents(full.events ?? []); setProgress(full.status === "completed" || full.status === "failed" ? 100 : 70); setStage(full.status === "completed" ? "任务完成" : STATUS_LABEL[full.status] ?? full.status); if (full.pending_confirmation) setPermission(full.pending_confirmation); setInspectorTab(full.tool_calls.length ? "chain" : "activity"); pushRunHistory(full);
    } catch { /* ignore */ }
  }

  async function send(content: string) {
    if (!content.trim() || sendingRef.current) return;
    sendingRef.current = true;
    // 发送即开启一条新 SSE 流：先中断上一条（正常情况下 busy 会挡住并发，这里是防御性兜底）。
    if (abortRef.current) { abortRef.current.abort(); }
    const controller = new AbortController();
    abortRef.current = controller;
    const sentSeq = switchSeqRef.current;
    setBusy(true); setError(null); setMessages((prev) => [...prev, { role: "user", content }]); let sid = sessionId;
    if (!sid) { try { const s = await createSession(selectedDatasets, content.slice(0, 30) || "数据分析会话"); setSession(s.id); setDatasetIds(selectedDatasets); setSessions((prev) => [s, ...prev.filter((item) => item.id !== s.id)]); sid = s.id; } catch (e) { setError(e instanceof Error ? e.message : "创建会话失败"); setBusy(false); sendingRef.current = false; return; } }
    setProgress(5); setStage("理解任务"); setEvents([]); setNotice(null); setLiveUsage(null); let runId: string | null = null;
    // 显式传 datasetIds：不依赖 store 的时序（见上方 dataset 同步注释），确保这次请求带上真实上下文。
    try { await sendMessage(sid, { content, stream: true, datasetIds: selectedDatasets, signal: controller.signal, onEvent: (ev) => { setEvents((prev) => [...prev, ev]); runId = ev.run_id; setActiveRunId(ev.run_id);
          if (ev.type === "route") { const mode = String(ev.payload?.mode ?? "chat"); const reason = String(ev.payload?.reason ?? ""); setStage(mode === "chat" ? `对话模式 · ${reason}` : `工具模式 · ${reason}`); setProgress((p) => Math.max(p, mode === "chat" ? 40 : 8)); }
          if (ev.type === "planning") { const s = String(ev.payload?.stage ?? "planning"); setStage(STAGE_LABEL[s] ?? s); setProgress((p) => Math.max(p, s === "plan_ready" ? 25 : s === "tools_retrieved" ? 15 : 8)); }
          if (ev.type === "tool_call") { setInspectorTab("activity"); setStage(`执行工具：${toolDisplayName(String(ev.payload?.tool ?? "tool"), tools)}`); setProgress((p) => Math.max(p, Math.min(90, p + 8))); }
          if (ev.type === "tool_result") setStage(`工具完成：${toolDisplayName(String(ev.payload?.tool ?? "tool"), tools)}`);
          if (ev.type === "validation" && ev.payload?.valid === false) { const errs = ((ev.payload?.errors as string[] | undefined) ?? []).join("；"); setNotice(`第 ${(ev.payload?.step_index as number ?? 0) + 1} 步校验失败：${errs || "未知原因"}`); }
          // 账本快照：运行过程中持续更新「花了多少 / 省了多少」。
          if (ev.type === "usage" && ev.payload?.token_usage) setLiveUsage(ev.payload.token_usage as AgentTokenUsage);
          if (ev.type === "replanning") { const notes = String(ev.payload?.notes ?? ""); setStage(ev.payload?.retry ? `重试中 · ${notes}` : `重新规划 · ${notes}`); setProgress((p) => Math.max(5, p - 15)); if (notes) setNotice(notes); } // U-1：重试时进度回退，让用户感知到"卡住后重来"
          if (ev.type === "permission") { const p = ev.payload as { tool?: string; arguments?: Record<string, unknown>; reason?: string; step_index?: number }; if (p?.tool) {
                const req: PermissionRequestType = { tool: p.tool, arguments: p.arguments ?? {}, reason: p.reason ?? "", step_index: p.step_index ?? 0 };
                const autoKey = `${ev.run_id}:${req.step_index}:${req.tool}`;
                // 设置页把该工具设为「自动放行」时直接替用户确认；仍失败则回退为手动弹窗。
                if (shouldAutoConfirm(req.tool, tools) && !autoAllowedRef.current.has(autoKey)) { autoAllowedRef.current.add(autoKey); void autoAllow(ev.run_id, req); return; }
                setPermission(req); setStage("等待确认"); setProgress(95); } }
          // 回答来源随 completed 事件一起下发，直接挂到这条消息上，气泡下方就能看到「谁回答的」。
          if (ev.type === "completed") { setProgress(100); setStage("任务完成"); if (typeof ev.payload?.final_answer === "string") setMessages((prev) => [...prev, { role: "assistant", content: ev.payload.final_answer as string, source: (ev.payload?.answer_source as AnswerSource | undefined) ?? null }]); } if (ev.type === "failed") { setProgress(100); setStage("任务失败"); setMessages((prev) => [...prev, { role: "assistant", content: `执行失败：${String(ev.payload?.error ?? "未知错误")}` }]); }         }, onDone: () => { setBusy(false); sendingRef.current = false; if (runId) { void refreshRun(runId); startPolling(runId); } }, });
    } catch (e) {
      // abort 导致的异常是「正常中断」（切会话 / 卸载 / 新请求顶替），不应当成发送失败报错。
      const aborted = controller.signal.aborted || (e instanceof DOMException && e.name === "AbortError");
      if (!aborted) setError(e instanceof Error ? e.message : "发送失败");
      if (sentSeq === switchSeqRef.current) { setBusy(false); sendingRef.current = false; }
      if (runId && !aborted) startPolling(runId);
    }
  }

  /**
   * 确认授权。
   *
   * 历史缺陷：这里原来是 `if (!run || busy) return;` —— SSE 流还没结束时 run 仍为 null、
   * busy 为 true，于是点击「允许」必然静默 return：不发请求、不报错、弹窗也不关，
   * 而「拒绝」用的是事件里已经赋值的 activeRunId，所以看起来只有拒绝能点。
   * 现在与 deny / stop 统一取 `activeRunId ?? run?.id`，并且不再用发送锁阻塞授权请求。
   */
  async function allow() {
    const target = activeRunId ?? run?.id;
    if (!target || confirming) return;
    const seq = switchSeqRef.current;
    setConfirming(true); setError(null);
    try {
      const resumed = await confirmRun(target);
      if (seq !== switchSeqRef.current) return; // 授权期间已切会话，丢弃结果
      setRun(resumed); setEvents(resumed.events ?? []); setPermission(null); setNotice(null);
      setStage(STATUS_LABEL[resumed.status] ?? resumed.status);
      if (resumed.final_answer) setMessages((prev) => [...prev, { role: "assistant", content: resumed.final_answer }]);
      if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) setPermission(resumed.pending_confirmation);
      pushRunHistory(resumed);
      startPolling(target);
    } catch (e) {
      // 失败时保留弹窗，让用户可以重试或直接拒绝，而不是把授权请求丢掉。
      if (seq === switchSeqRef.current) setError(e instanceof Error ? e.message : "确认失败");
    } finally { setConfirming(false); }
  }

  /** 设置页已把该工具设为「自动放行」时的免确认路径。失败回退为手动授权弹窗。 */
  async function autoAllow(runId: string, req: PermissionRequestType) {
    const seq = switchSeqRef.current;
    try {
      const resumed = await confirmRun(runId);
      if (seq !== switchSeqRef.current) return; // 已切会话 / 已删除，丢弃
      setRun(resumed); setEvents(resumed.events ?? []); setPermission(null);
      setNotice(`已按设置自动放行「${toolDisplayName(req.tool, tools)}」。`);
      if (resumed.final_answer) setMessages((prev) => [...prev, { role: "assistant", content: resumed.final_answer, source: resumed.answer_source ?? null }]);
      if (resumed.status === "waiting_confirmation" && resumed.pending_confirmation) setPermission(resumed.pending_confirmation);
      pushRunHistory(resumed);
      startPolling(runId);
    } catch (e) {
      if (seq !== switchSeqRef.current) return;
      setPermission(req);
      setError(e instanceof Error ? e.message : "自动放行失败，请手动确认");
    }
  }

  async function deny() {
    // S-3：拒绝必须通知后端终止 run，否则 run 卡在 WAITING_CONFIRMATION，会话被 409 锁死。
    setPermission(null);
    setMessages((prev) => [...prev, { role: "assistant", content: "已拒绝该高风险操作的授权，对应步骤不会执行。" }]);
    const target = activeRunId ?? run?.id;
    if (!target) return;
    const seq = switchSeqRef.current;
    try {
      const denied = await denyRun(target);
      if (seq !== switchSeqRef.current) return; // 拒绝期间已切会话，丢弃结果
      setRun(denied); setEvents(denied.events ?? []); pushRunHistory(denied);
    } catch { /* 后端不可达时保留本地提示，轮询会兜底同步 */ }
  }

  /**
   * 请求取消当前运行。
   * 后端只在「步骤边界」检查 cancel_requested，长步骤（训练 / 报告生成 / 大模型响应）
   * 期间不会立刻停止，因此这里补上轮询兜底，让进度条继续反映真实状态，
   * 用户也能从文案确认取消已被受理，而不是反复点同一个按钮。
   */
  async function stop() {
    const target = activeRunId ?? run?.id;
    if (!target) return;
    setStage("正在取消…");
    setError(null);
    try {
      await cancelRun(target);
      setStage("已请求取消，将在当前步骤结束后停止");
      startPolling(target);
    } catch (e) {
      setError(e instanceof Error ? e.message : "取消失败");
      setStage("取消失败");
    }
  }

  /** 生成报告后直接跳到报告详情路由，不再整页刷新跳转列表页。 */
  async function generateDataReport() {
    const datasetId = selectedDatasets[0];
    if (!datasetId) return setError("请先在上方选择一个数据集");
    setGeneratingReport(true); setError(null);
    try {
      const report = await generateReport({ dataset_id: datasetId, title: "AI 数据分析报告", include_quality: true, include_eda: true, include_ml: true, conclusions: run?.final_answer ? [run.final_answer] : undefined });
      const key = typeof report?.metadata?.report_key === "string" ? report.metadata.report_key : "";
      setNotice("报告已生成，正在打开…");
      if (key) navigate(`/reports/${encodeURIComponent(key)}`, { state: { report } });
      else navigate("/reports", { state: { report } });
    } catch (e) { setError(e instanceof Error ? e.message : "报告生成失败"); } finally { setGeneratingReport(false); }
  }

  /**
   * 选中态一律以「当前列表里的 id 集合」为准，并裁掉孤儿 id。
   *
   * 历史缺陷：原来用 `selectedIds.length === allIds.length` 判全选、用 `prev.length === allIds.length`
   * 判是否该取消全选。但 selectedIds 里可能残留已不在列表中的 id（归档 / 删除 / 列表收缩后），
   * 两个长度比较就会错位 —— 勾选框 checked 与「已选 N/M」文案不一致，点「全选」还可能一次性清空。
   * 现在统一用集合包含判断，并在 sessions 变化时把不在列表里的 id 剔除。
   */
  const allIds = useMemo(() => sessions.map((s) => s.id), [sessions]);
  const selectedVisible = useMemo(() => selectedIds.filter((id) => allIds.includes(id)), [selectedIds, allIds]);
  const allSelected = allIds.length > 0 && selectedVisible.length === allIds.length;
  const someSelected = selectedVisible.length > 0 && !allSelected;
  useEffect(() => {
    // sessions 收缩（删除 / 归档过滤）后清掉孤儿选中项，避免计数错位。
    setSelectedIds((prev) => {
      const next = prev.filter((id) => allIds.includes(id));
      return next.length === prev.length ? prev : next;
    });
  }, [allIds]);
  function toggleSelect(id: string) { setSelectedIds((prev) => prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]); }
  function toggleSelectAll() { setSelectedIds(() => (allSelected ? [] : [...allIds])); }
  function invertSelect() { setSelectedIds(() => allIds.filter((id) => !selectedVisible.includes(id))); }

  /** 工具链：按调用顺序展示调用关系（步序 → 工具 → 状态 → 耗时）。 */
  const chainItems = useMemo(() => {
    if (!run?.tool_calls?.length) return [] as { step: number; label: string; status: string; statusText: string; elapsed: string }[];
    return [...run.tool_calls]
      .sort((a, b) => a.step_index - b.step_index)
      .map((call) => ({
        step: call.step_index,
        label: toolDisplayName(call.tool, tools),
        status: call.status,
        statusText: TOOL_STATUS_LABEL[call.status] ?? call.status,
        elapsed: `${(call.elapsed_ms / 1000).toFixed(2)}s`,
      }));
  }, [run, tools]);

  const emptyState = (
    <div className="ai-start-panel">
      <h2>从一个问题开始你的数据实验</h2>
      <p className="muted">{selectedDatasets.length ? "选择一个起点快速开始，或直接在下方输入你的问题。" : "先在上方选择数据集，即可解锁快捷操作；也可以直接输入问题对话。"}</p>
      <div className="ai-action-grid">
        {QUICK_ACTIONS.map((action) => (
          <button key={action.title} className="ai-action-card" type="button" disabled={busy || !selectedDatasets.length} onClick={() => void send(action.prompt)}>
            <strong>{action.title}</strong>
            <span>{action.description}</span>
            <em>开始 →</em>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="ai-lab-page">
      <PageHeader
        title="AI 实验室"
        description="用对话驱动数据分析。"
        actions={<Link className="btn" to="/reports">报告中心 →</Link>}
      />

      {/* 上下文条：数据集选择 + 关联状态 + 授权提示合并为一行，原左栏摘要卡已并入此处。 */}
      <div className="ai-context-bar">
        <div className="ai-context-main">
          <span className="ai-context-title">数据上下文</span>
          <DatasetSelector value={selectedDatasets} onChange={setSelectedDatasets} multi compact showLabel={false} />
          <span className={`ai-chip${selectedDatasets.length ? "" : " warn"}`}>{selectedDatasets.length ? `已关联 ${selectedDatasets.length} 个数据集` : "未选择数据集"}</span>
          {permission && <span className="ai-chip danger">待授权确认</span>}
        </div>
        <div className="ai-context-actions">
          <SmartAnalysisButton disabled={busy || !selectedDatasets.length} busy={busy} onClick={() => void send("请对当前关联的数据集做一次智能分析：先检查数据质量，再给出关键统计与问题摘要。")} />
        </div>
      </div>

      {/* panel-large：面板放大时同步压低对话区，让两者仍在同一屏内可见（整页仍可下滑）。 */}
      <div className={`ai-workspace${panelOpen ? " panel-open" : ""}${panelOpen && panelSize === "large" ? " panel-large" : ""}`}>
        {/* 会话栏：窄屏可折叠；条目单行截断，操作收进「⋯」菜单；支持多选与批量操作。 */}
        <aside className={`ai-sidebar${listOpen ? "" : " sessions-collapsed"}`}>
          <section className="ai-sidebar-section ai-experiment-nav">
            <div className="ai-section-heading">
              <div><div className="ai-section-kicker">会话</div><h3>分析会话</h3></div>
              <div className="ai-heading-actions">
                <button className="btn primary" type="button" onClick={() => void newSession()}>+ 新建</button>
                <button className="btn ai-sidebar-fold" type="button" aria-expanded={listOpen} title={listOpen ? "收起列表" : "展开列表"} onClick={() => setListOpen((v) => !v)}>{listOpen ? "收起" : "展开"}</button>
              </div>
            </div>
            {/* 批量操作条：全选 / 反选 / 批量归档 / 批量删除 / 清空 */}
            <div className="ai-session-toolbar">
              <label className="ai-check-line" title="全选 / 取消全选">
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(el) => { if (el) el.indeterminate = someSelected; }}
                  disabled={!allIds.length}
                  onChange={toggleSelectAll}
                />
                全选
              </label>
              <span className="ai-toolbar-count muted">{selectedVisible.length ? `已选 ${selectedVisible.length}/${allIds.length}` : `${allIds.length} 个会话`}</span>
              <div className="ai-toolbar-actions">
                <button className="btn btn-sm" type="button" disabled={!selectedVisible.length} onClick={invertSelect}>反选</button>
                <button className="btn btn-sm" type="button" disabled={!selectedVisible.length} onClick={() => void archiveSelected(selectedVisible)}>归档</button>
                <button className="btn btn-sm danger" type="button" disabled={!selectedVisible.length || bulkDeleting} onClick={() => setPendingBulk({ ids: [...selectedVisible], all: false })}>删除</button>
                <button className="btn btn-sm danger" type="button" disabled={!allIds.length || bulkDeleting} onClick={() => setPendingBulk({ ids: [...allIds], all: true })} title="删除全部会话及其运行记录（运行中的会话会跳过）">清空</button>
              </div>
            </div>
            <div className="ai-session-list">
              {!sessions.length && <div className="muted">还没有历史会话。</div>}
              {sessions.map((s) => (
                <div key={s.id} className={`ai-session-item ${sessionId === s.id ? "active" : ""} ${s.archived ? "archived" : ""} ${selectedIds.includes(s.id) ? "selected" : ""}`}>
                  <input
                    type="checkbox"
                    className="ai-session-check"
                    checked={selectedIds.includes(s.id)}
                    onChange={() => toggleSelect(s.id)}
                    aria-label={`选择会话 ${sessionTitle(s)}`}
                  />
                  <button type="button" className="ai-session-open" onClick={() => switchSession(s)} title={sessionTitle(s)}>
                    <div className="ai-session-item-title">
                      <span className="ai-session-item-name">{sessionTitle(s)}</span>
                      {s.archived && <em className="ai-session-archived-tag">已归档</em>}
                    </div>
                    <div className="ai-session-item-meta">{relativeTime(s.created_at)} · {s.history.length ? `${s.history.length} 条消息` : "新实验"}{s.run_ids?.length ? ` · ${s.run_ids.length} 次运行` : ""}</div>
                  </button>
                  <button type="button" className="ai-session-more" aria-label="会话操作" aria-expanded={menuSessionId === s.id} onClick={() => setMenuSessionId((id) => (id === s.id ? null : s.id))}>⋯</button>
                  {menuSessionId === s.id && (
                    <div className="ai-session-menu" role="menu">
                      <button type="button" role="menuitem" onClick={() => { setMenuSessionId(null); void toggleArchive(s); }}>{s.archived ? "取消归档" : "归档"}</button>
                      <button type="button" role="menuitem" className="danger" disabled={deletingSessionId === s.id} onClick={() => { setMenuSessionId(null); setPendingDelete(s); }}>{deletingSessionId === s.id ? "删除中…" : "删除会话"}</button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </aside>

        {/* 主列：上对话、下运行面板（运行面板宽度与对话一致，便于观察长文本与工具链）。 */}
        <div className="ai-main-column">
          <section className="card ai-chat-column">
            <div className="ai-chat-header">
              <div>
                <h3>{sessionId ? "当前实验" : "开始一次 AI 分析"}</h3>
                <div className="muted ai-chat-sub">{selectedDatasets.length ? `已关联 ${selectedDatasets.length} 个数据集` : "先选择数据集，也可以直接创建会话。"}</div>
              </div>
              <div className="ai-chat-actions">
                {busy && <button className="btn" type="button" onClick={() => void stop()} title="在当前步骤结束后停止运行">停止</button>}
                <button className={`btn ai-panel-toggle${panelOpen ? " active" : ""}`} type="button" aria-pressed={panelOpen} onClick={() => { panelUserCollapsedRef.current = panelOpen; setPanelOpen((v) => !v); }}>
                  {panelOpen ? "收起运行面板" : "展开运行面板"}{(busy || permission) && <span className="ai-panel-dot" aria-hidden="true" />}
                </button>
                <button className="btn primary" type="button" disabled={!selectedDatasets.length || generatingReport} onClick={() => void generateDataReport()}>{generatingReport ? "报告生成中..." : "生成报告"}</button>
              </div>
            </div>
            {notice && <div className="ai-route-notice" role="status"><span>ℹ️</span><span style={{ flex: 1 }}>{notice}</span><button type="button" className="btn" style={{ padding: "2px 10px" }} onClick={() => setNotice(null)}>知道了</button></div>}
            <ChatPanel messages={displayMessages} busy={busy} onSend={(content) => void send(content)} showEmptyState={false} emptyState={emptyState} />
            {error && <div className="badge failed ai-error">{error}</div>}
          </section>

          {/* 运行面板：位于对话正下方，宽度与主区域一致，支持展开 / 收起。 */}
          {panelOpen && (
            <aside className={`ai-inspector${panelSize === "large" ? " large" : ""}`} aria-label="运行面板">
              <div className="ai-panel-head">
                <div className="ai-panel-head-row">
                  <div><div className="ai-section-kicker">运行</div><h3>本次执行</h3></div>
                  <div className="ai-panel-head-actions">
                    <button
                      className="btn ai-panel-collapse"
                      type="button"
                      aria-pressed={panelSize === "large"}
                      title={panelSize === "large" ? "恢复默认高度" : "放大运行面板（面板太小、内容看不全时用）"}
                      onClick={() => setPanelSize((v) => (v === "large" ? "normal" : "large"))}
                    >
                      {panelSize === "large" ? "还原高度" : "放大面板"}
                    </button>
                    {run && <a className="btn" href={runTraceUrl(run.id, "md")} download={`agent-trace-${run.id}.md`} title="导出完整执行轨迹（Markdown）">轨迹</a>}
                    <button className="btn ai-panel-collapse" type="button" aria-label="收起运行面板" title="收起运行面板" onClick={() => { panelUserCollapsedRef.current = true; setPanelOpen(false); }}>收起</button>
                  </div>
                </div>
                {(busy || run) && (
                  <div className="ai-panel-progress">
                    <div className="ai-panel-progress-text"><span>{stage}</span><strong>{progress}%</strong></div>
                    <div className="ai-progress-track"><div className="ai-progress-fill" style={{ width: `${progress}%` }} /></div>
                  </div>
                )}
              </div>
              <div className="ai-panel-body">
                {!run && !busy && <div className="ai-panel-placeholder muted">发起分析后，这里会显示执行进度、工具链调用顺序与 Token 用量。</div>}
                {run && (
                  <section className="ai-run-section">
                    <AnswerSourceChip source={run.answer_source} />
                    <div className="ai-run-summary">
                      <div><strong>{run.tool_call_count}</strong><span>工具步骤</span></div>
                      <div><strong>{run.elapsed_seconds.toFixed(1)}s</strong><span>耗时</span></div>
                      <div><strong>{run.token_usage?.llm_calls ?? 0}</strong><span>LLM 调用</span></div>
                      <div><strong>{(run.token_usage?.actual?.total_tokens ?? 0).toLocaleString()}</strong><span>已用 Token</span></div>
                      <div><strong>{(run.token_usage?.optimization?.estimated_saved_tokens ?? 0).toLocaleString()}</strong><span>估算省下</span></div>
                      <div><strong>{run.pending_confirmation ? "1" : "0"}</strong><span>待授权</span></div>
                    </div>
                    {/* 花了多少 / 省了多少：常驻，不随页签切换消失 */}
                    <TokenCostStrip usage={liveUsage ?? run.token_usage ?? null} live={busy && liveUsage !== null} />
                  </section>
                )}
                <section className="ai-inspector-main">
                  <div className="ai-inspector-tabs" role="tablist" aria-label="运行详情">
                    <button type="button" className={inspectorTab === "overview" ? "active" : ""} onClick={() => setInspectorTab("overview")}>概览</button>
                    <button type="button" className={inspectorTab === "activity" ? "active" : ""} onClick={() => setInspectorTab("activity")}>活动 {events.length ? `(${events.length})` : ""}</button>
                    <button type="button" className={inspectorTab === "chain" ? "active" : ""} onClick={() => setInspectorTab("chain")}>工具链 {run?.tool_calls.length ? `(${run.tool_calls.length})` : ""}</button>
                    <button type="button" className={inspectorTab === "token" ? "active" : ""} onClick={() => setInspectorTab("token")} title="查看 Token 消耗与多次对话的用量变化">Token</button>
                    <button type="button" className={inspectorTab === "budget" ? "active" : ""} onClick={() => setInspectorTab("budget")} title="查看步数上限、Token 预算与模型信息">预算</button>
                  </div>
                  {inspectorTab === "overview" && (
                    <div className="ai-overview-content">
                      {!run && <div className="muted">还没有运行记录。</div>}
                      {run && <>
                        <div className="ai-overview-row"><span>目标</span><strong>{run.plan?.goal || run.user_request || "当前分析"}</strong></div>
                        {run.plan?.steps?.length ? (
                          <div className="ai-plan-steps">
                            <div className="ai-section-kicker">执行计划</div>
                            {run.plan.steps.map((step, i) => <div className="ai-plan-step" key={i}><span>{i + 1}</span><div>{planStepText(step)}</div></div>)}
                          </div>
                        ) : <div className="muted">AI 尚未返回结构化执行计划。</div>}
                        {run.final_answer && <div className="ai-latest-result"><div className="ai-section-kicker">最新结果</div><p>{run.final_answer}</p></div>}
                      </>}
                    </div>
                  )}
                  {inspectorTab === "activity" && <div className="ai-detail-content">{events.length ? <AgentTimeline events={events} /> : <div className="muted">暂无执行事件。</div>}</div>}
                  {inspectorTab === "chain" && (
                    <div className="ai-detail-content">
                      {chainItems.length ? <>
                        <div className="ai-section-kicker">调用顺序与调用关系</div>
                        <div className="ai-chain">
                          {chainItems.map((item, i) => (
                            <div className={`ai-chain-node status-${item.status}`} key={`${item.step}-${i}`}>
                              <div className="ai-chain-node-head">
                                <span className="ai-chain-index">{item.step + 1}</span>
                                <strong>{item.label}</strong>
                                <span className={`ai-chain-status ${item.status}`}>{item.statusText}</span>
                              </div>
                              <div className="ai-chain-node-meta">耗时 {item.elapsed}</div>
                              {i < chainItems.length - 1 && <span className="ai-chain-arrow" aria-hidden="true">↓</span>}
                            </div>
                          ))}
                        </div>
                        <div className="ai-section-kicker" style={{ marginTop: 16 }}>调用明细</div>
                        {run?.tool_calls.map((call, i) => <ToolCallCard key={`${call.step_index}-${i}`} call={call} />)}
                      </> : <div className="muted">本次运行还没有工具调用。</div>}
                    </div>
                  )}
                  {inspectorTab === "token" && (
                    <div className="ai-detail-content">
                      {run ? <TokenUsageBody run={run} /> : <p className="muted">发起任务后可查看本次运行的 Token 消耗。</p>}
                      <div className="ai-run-history">
                        <div className="ai-section-kicker">每次对话的调用次数与调用量变化</div>
                        <RunUsageHistory history={runHistory} />
                      </div>
                    </div>
                  )}
                  {inspectorTab === "budget" && <div className="ai-detail-content"><CapabilitiesPanel capabilities={capabilities} loading={capsLoading} error={capsError} onRetry={loadCapabilities} /></div>}
                </section>
              </div>
            </aside>
          )}
        </div>
      </div>
      {menuSessionId && <div className="ai-menu-backdrop" onClick={() => setMenuSessionId(null)} aria-hidden="true" />}
      <PermissionRequest request={permission} busy={confirming} onAllow={() => void allow()} onDeny={() => void deny()} />
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除这个会话？"
        message={
          <>
            会话「{pendingDelete ? sessionTitle(pendingDelete) : ""}」下的
            {pendingDelete?.run_ids?.length ?? 0} 条运行记录会一并删除，且无法恢复。
          </>
        }
        confirmText="删除会话"
        danger
        onConfirm={() => { if (pendingDelete) void removeSessions([pendingDelete.id]); }}
        onCancel={() => setPendingDelete(null)}
      />
      <ConfirmDialog
        open={pendingBulk !== null}
        title={pendingBulk?.all ? "清空全部会话？" : `删除选中的 ${pendingBulk?.ids.length ?? 0} 个会话？`}
        message={
          pendingBulk?.all
            ? <>将删除全部 {pendingBulk.ids.length} 个会话及其运行记录，且无法恢复。正在运行中的会话会跳过。</>
            : <>选中的 {pendingBulk?.ids.length ?? 0} 个会话及其运行记录会被删除，且无法恢复。正在运行中的会话会跳过。</>
        }
        confirmText={pendingBulk?.all ? "清空全部" : "批量删除"}
        danger
        onConfirm={() => { if (pendingBulk) void removeSessions(pendingBulk.ids); }}
        onCancel={() => setPendingBulk(null)}
      />
    </div>
  );
}
