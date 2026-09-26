import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./ai-lab.css";
import { PageHeader } from "../../components/PageHeader";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { ChatPanel } from "../../features/agent/ChatPanel";
import { PermissionRequest } from "../../features/agent/PermissionRequest";
import { ClarificationPanel } from "../../features/agent/ClarificationPanel";
import { SmartAnalysisButton } from "../../features/agent/SmartAnalysisButton";
import { RunInspector } from "../../features/agent/RunInspector";
import { SessionSidebar } from "../../features/agent/SessionSidebar";
import { useAiSession, sessionTitle } from "../../features/agent/hooks/useAiSession";
import { useAgentRun } from "../../features/agent/hooks/useAgentRun";
import { generateReport } from "../../api/reports";
import { useAiLab } from "../../store/aiLab";
import type { AgentSession, InspectorTab } from "../../types/agent";

const QUICK_ACTIONS = [
  { title: "检查数据质量", description: "缺失值、重复值、字段类型与异常概览", prompt: "请检查当前关联的数据集质量，给出缺失值、重复值、字段类型和明显异常的摘要。" },
  { title: "探索数据", description: "生成关键统计，并指出值得进一步分析的变量", prompt: "请对当前关联的数据集做一次探索性分析，给出关键统计、变量关系和最值得继续分析的问题。" },
  { title: "设计机器学习实验", description: "根据数据与目标提出可执行的建模方案", prompt: "请根据当前数据集设计一个机器学习实验方案，说明目标变量、特征、候选模型、评价指标和下一步执行建议。" },
];

/**
 * AI 实验室页面：**只负责布局**。
 *
 * 会话域（列表 / 归档 / 删除 / 消息）在 useAiSession，运行域（发送 / SSE / 轮询 /
 * 授权 / 用量）在 useAgentRun；页面只把两者接起来，并持有纯界面状态
 * （面板展开、列表折叠、二次确认弹窗、错误与提示条）。
 */
export default function AI() {
  const tools = useAiLab((s) => s.tools);
  const navigate = useNavigate();

  // 页面级反馈条：会话域与运行域共用同一处展示，故提升到布局层。
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // 运行面板：空闲默认收起，任务启动 / 切到带运行的会话时自动展开；用户可随时手动收起。
  const [panelOpen, setPanelOpen] = useState(false);
  // 运行面板高度档位：默认 / 放大（面板原高度偏小，长工具链与用量表看不全）。
  const [panelSize, setPanelSize] = useState<"normal" | "large">("normal");
  // 会话列表：宽屏常开；窄屏（<1280）可折叠成一行入口。
  const [listOpen, setListOpen] = useState(true);
  const [generatingReport, setGeneratingReport] = useState(false);
  // 删除会话的二次确认目标（走统一 ConfirmDialog）。
  const [pendingDelete, setPendingDelete] = useState<AgentSession | null>(null);
  // 批量删除 / 清空：{ ids, all }，all=true 表示「清空全部」。
  const [pendingBulk, setPendingBulk] = useState<{ ids: string[]; all: boolean } | null>(null);

  const session = useAiSession({ onError: setError, onNotice: setNotice });
  const run = useAgentRun({
    sessionId: session.sessionId,
    lastRunId: session.lastRunId,
    switchToken: session.switchToken,
    datasetIds: session.selectedDatasets,
    tools,
    ensureSession: session.ensureSession,
    appendMessage: session.appendMessage,
    onError: setError,
    onNotice: setNotice,
    // 切到带历史运行的会话时回填成功 ⇒ 展开运行面板（与原行为一致）。
    //
    // ★ 必须 `useCallback(..., [])`：内联箭头函数每次 render 都是新引用，
    // 而它是 `useAgentRun` 里「会话镜像 Effect」的输入。引用不稳定 ⇒
    // 那个 effect 每次 render 重跑 ⇒ 执行 `abortStream() / clear() / setRun(null)`
    // ⇒ 表现为「刚发出去的消息没有回应、运行面板里的运行消失、SSE 被前端自己掐断、
    // Inspector 页签被锁死在概览」。
    onRunRestored: useCallback(() => setPanelOpen(true), []),
    // 运行终态后回拉会话列表，刷新侧栏标题/消息数/run_ids（切回可恢复最后 run）。
    onRunFinished: session.refreshActiveSession,
  });

  // ★ 助手消息唯一追加出口已收敛到 SSE/回放的 completed 事件（见 useAgentRun），
  //   不再需要页面层「相邻同内容去重」补丁——直接展示会话消息。
  const displayMessages = session.messages;

  /**
   * 面板自动展开：只在「任务开始」这一次转变上触发（busy 由 false → true），
   * 并在出现待授权弹窗时强制展开。
   * 历史缺陷：原来依赖 [busy, permission] 且无条件 setPanelOpen(true)，
   * 导致运行期间用户点「收起运行面板」后，只要 permission 变化就被再次弹开，
   * 与注释里「用户可随时手动收起」的承诺不符。
   */
  const panelUserCollapsedRef = useRef(false);
  const prevBusyRef = useRef(false);
  useEffect(() => {
    if (run.busy && !prevBusyRef.current) panelUserCollapsedRef.current = false; // 新一轮任务开始：重置收起意图
    if (!run.busy) panelUserCollapsedRef.current = false;                         // 任务结束：复位
    // 授权弹窗 / 澄清面板都必须可见：它们都是「等待用户」的界面，收起面板等于把问题藏起来。
    if (run.permission || run.clarification) { setPanelOpen(true); return; }
    if (run.busy && !panelUserCollapsedRef.current) setPanelOpen(true);
    prevBusyRef.current = run.busy;
  }, [run.busy, run.permission, run.clarification]);

  /** 生成报告后直接跳到报告详情路由，不再整页刷新跳转列表页。 */
  async function generateDataReport() {
    const datasetId = session.selectedDatasets[0];
    if (!datasetId) return setError("请先在上方选择一个数据集");
    setGeneratingReport(true);
    setError(null);
    try {
      const report = await generateReport({
        dataset_id: datasetId,
        title: "AI 数据分析报告",
        include_quality: true,
        include_eda: true,
        include_ml: true,
        conclusions: run.run?.final_answer ? [run.run.final_answer] : undefined,
      });
      const key = typeof report?.metadata?.report_key === "string" ? report.metadata.report_key : "";
      setNotice("报告已生成，正在打开…");
      if (key) navigate(`/reports/${encodeURIComponent(key)}`, { state: { report } });
      else navigate("/reports", { state: { report } });
    } catch (e) {
      setError(e instanceof Error ? e.message : "报告生成失败");
    } finally {
      setGeneratingReport(false);
    }
  }

  const emptyState = (
    <div className="ai-start-panel">
      <h2>从一个问题开始你的数据实验</h2>
      <p className="muted">{session.selectedDatasets.length ? "选择一个起点快速开始，或直接在下方输入你的问题。" : "先在上方选择数据集，即可解锁快捷操作；也可以直接输入问题对话。"}</p>
      <div className="ai-action-grid">
        {QUICK_ACTIONS.map((action) => (
          <button key={action.title} className="ai-action-card" type="button" disabled={run.busy || !session.selectedDatasets.length} onClick={() => void run.send(action.prompt)}>
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
          <DatasetSelector value={session.selectedDatasets} onChange={session.setSelectedDatasets} multi compact showLabel={false} />
          <span className={`ai-chip${session.selectedDatasets.length ? "" : " warn"}`}>{session.selectedDatasets.length ? `已关联 ${session.selectedDatasets.length} 个数据集` : "未选择数据集"}</span>
          {run.permission && <span className="ai-chip danger">待授权确认</span>}
          {run.clarification && <span className="ai-chip warn">待补充信息</span>}
        </div>
        <div className="ai-context-actions">
          <SmartAnalysisButton disabled={run.busy || !session.selectedDatasets.length} busy={run.busy} onClick={() => void run.send("请对当前关联的数据集做一次智能分析：先检查数据质量，再给出关键统计与问题摘要。")} />
        </div>
      </div>

      {/* panel-large：面板放大时同步压低对话区，让两者仍在同一屏内可见（整页仍可下滑）。 */}
      <div className={`ai-workspace${panelOpen ? " panel-open" : ""}${panelOpen && panelSize === "large" ? " panel-large" : ""}`}>
        <SessionSidebar
          sessions={session.sessions}
          activeId={session.sessionId}
          listOpen={listOpen}
          onToggleList={() => setListOpen((v) => !v)}
          deletingId={session.deletingSessionId}
          bulkBusy={session.bulkDeleting}
          selectedIds={session.selectedVisible}
          allIds={session.allIds}
          allSelected={session.allSelected}
          someSelected={session.someSelected}
          onToggleSelect={session.toggleSelect}
          onToggleSelectAll={session.toggleSelectAll}
          onInvertSelect={session.invertSelect}
          onNew={() => void session.newSession()}
          onOpen={session.activate}
          onToggleArchive={(s) => void session.toggleArchive(s)}
          onArchiveSelected={(ids) => void session.archiveSelected(ids)}
          onRequestDelete={setPendingDelete}
          onRequestBulk={(ids, all) => setPendingBulk({ ids, all })}
        />

        {/* 主列：上对话、下运行面板（运行面板宽度与对话一致，便于观察长文本与工具链）。 */}
        <div className="ai-main-column">
          <section className="card ai-chat-column">
            <div className="ai-chat-header">
              <div>
                <h3>{session.sessionId ? "当前实验" : "开始一次 AI 分析"}</h3>
                <div className="muted ai-chat-sub">{session.selectedDatasets.length ? `已关联 ${session.selectedDatasets.length} 个数据集` : "先选择数据集，也可以直接创建会话。"}</div>
              </div>
              <div className="ai-chat-actions">
                {run.busy && <button className="btn" type="button" onClick={() => void run.stop()} title="在当前步骤结束后停止运行">停止</button>}
                <button className={`btn ai-panel-toggle${panelOpen ? " active" : ""}`} type="button" aria-pressed={panelOpen} onClick={() => { panelUserCollapsedRef.current = panelOpen; setPanelOpen((v) => !v); }}>
                  {panelOpen ? "收起运行面板" : "展开运行面板"}{(run.busy || run.permission) && <span className="ai-panel-dot" aria-hidden="true" />}
                </button>
                <button className="btn primary" type="button" disabled={!session.selectedDatasets.length || generatingReport} onClick={() => void generateDataReport()}>{generatingReport ? "报告生成中..." : "生成报告"}</button>
              </div>
            </div>
            {notice && <div className="ai-route-notice" role="status"><span>ℹ️</span><span style={{ flex: 1 }}>{notice}</span><button type="button" className="btn" style={{ padding: "2px 10px" }} onClick={() => setNotice(null)}>知道了</button></div>}
            <ChatPanel messages={displayMessages} busy={run.busy} onSend={(content) => void run.send(content)} showEmptyState={false} emptyState={emptyState} />
            {error && <div className="badge failed ai-error">{error}</div>}
          </section>

          {/* 运行面板：位于对话正下方，宽度与主区域一致，支持展开 / 收起。 */}
          {panelOpen && (
            <RunInspector
              run={run.run}
              events={run.events}
              busy={run.busy}
              progress={run.progress}
              stage={run.stage}
              tab={run.inspectorTab}
              onTabChange={(tab: InspectorTab) => run.setInspectorTab(tab)}
              usage={run.usage}
              live={run.live}
              history={run.history}
              tools={tools}
              large={panelSize === "large"}
              onToggleSize={() => setPanelSize((v) => (v === "large" ? "normal" : "large"))}
              onCollapse={() => { panelUserCollapsedRef.current = true; setPanelOpen(false); }}
            />
          )}
        </div>
      </div>
      <PermissionRequest request={run.permission} busy={run.confirming} onAllow={() => void run.allow()} onDeny={() => void run.deny()} />
      {/* 待澄清问题：后端停在「等你补充信息」时必须看得见、答得上，
          否则运行会一直挂在等待态，界面只剩一个不动的进度条。 */}
      <ClarificationPanel
        request={run.clarification}
        busy={run.clarifying}
        onSubmit={(answer) => void run.answerClarification(answer)}
        onCancel={() => void run.stop()}
      />
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
        onConfirm={() => { if (pendingDelete) void session.removeSessions([pendingDelete.id]); setPendingDelete(null); }}
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
        onConfirm={() => { if (pendingBulk) void session.removeSessions(pendingBulk.ids); setPendingBulk(null); }}
        onCancel={() => setPendingBulk(null)}
      />
    </div>
  );
}
