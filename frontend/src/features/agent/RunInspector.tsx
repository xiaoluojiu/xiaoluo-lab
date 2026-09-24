/**
 * 运行面板：纯展示。
 *
 * 只接收状态与回调，不发请求、不持有运行逻辑（那些在 useAgentRun）。
 * 唯一的内部 state 是「Agent 能力声明」——它只服务「预算」这一个页签，
 * 提到页面层只会让页面多三个跟布局无关的状态。
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { runTraceUrl, getCapabilities, type AgentToolInfo } from "../../api/agent";
import { toolDisplayName } from "../../store/aiLab";
import { InfoHint } from "../../components/InfoHint";
import type { AgentEvent, AgentRun, AgentTokenUsage, AnswerSource, InspectorTab } from "../../types/agent";
import { AgentTimeline } from "./AgentTimeline";
import { ToolCallCard } from "./ToolCallCard";
import { CapabilitiesPanel } from "./CapabilitiesPanel";
import { RunUsageHistory, TokenUsageBody } from "./RunUsageViews";
import type { RunUsageSnapshot } from "./hooks/useAgentUsage";

const TOOL_STATUS_LABEL: Record<string, string> = {
  pending: "等待中",
  ok: "成功",
  failed: "失败",
  needs_confirmation: "待授权",
  denied: "已拒绝",
};

function planStepText(step: unknown): string {
  if (typeof step === "string") return step;
  if (step && typeof step === "object") {
    const item = step as Record<string, unknown>;
    return String(item.title ?? item.goal ?? item.action ?? JSON.stringify(step));
  }
  return String(step);
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

/**
 * 「花了多少 / 省了多少」——**常驻**显示在面板顶部，不藏在 Token 页签里。
 *
 * 历史问题：这两组数字原先只在「Token」页签可见，而运行过程中页签默认停在
 * 「活动」，用户跑一次任务下来完全不知道消耗了多少、平台替他省了多少。
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

export function RunInspector({
  run,
  events,
  busy,
  progress,
  stage,
  tab,
  onTabChange,
  usage,
  live,
  history,
  tools,
  large,
  onToggleSize,
  onCollapse,
}: {
  run: AgentRun | null;
  events: AgentEvent[];
  busy: boolean;
  progress: number;
  stage: string;
  tab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  usage: AgentTokenUsage | null;
  live: boolean;
  history: RunUsageSnapshot[];
  tools: AgentToolInfo[];
  large: boolean;
  onToggleSize: () => void;
  onCollapse: () => void;
}) {
  const [capabilities, setCapabilities] = useState<Awaited<ReturnType<typeof getCapabilities>> | null>(null);
  const [capsLoading, setCapsLoading] = useState(true);
  const [capsError, setCapsError] = useState<string | null>(null);

  const loadCapabilities = useCallback(() => {
    setCapsLoading(true);
    setCapsError(null);
    getCapabilities()
      .then(setCapabilities)
      .catch((e: unknown) => setCapsError(e instanceof Error ? e.message : "读取 Agent 能力失败"))
      .finally(() => setCapsLoading(false));
  }, []);
  useEffect(() => {
    loadCapabilities();
  }, [loadCapabilities]);

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

  return (
    <aside className={`ai-inspector${large ? " large" : ""}`} aria-label="运行面板">
      <div className="ai-panel-head">
        <div className="ai-panel-head-row">
          <div><div className="ai-section-kicker">运行</div><h3>本次执行</h3></div>
          <div className="ai-panel-head-actions">
            <button
              className="btn ai-panel-collapse"
              type="button"
              aria-pressed={large}
              title={large ? "恢复默认高度" : "放大运行面板（面板太小、内容看不全时用）"}
              onClick={onToggleSize}
            >
              {large ? "还原高度" : "放大面板"}
            </button>
            {run && <a className="btn" href={runTraceUrl(run.id, "md")} download={`agent-trace-${run.id}.md`} title="导出完整执行轨迹（Markdown）">轨迹</a>}
            <button className="btn ai-panel-collapse" type="button" aria-label="收起运行面板" title="收起运行面板" onClick={onCollapse}>收起</button>
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
            <TokenCostStrip usage={usage} live={live} />
          </section>
        )}
        <section className="ai-inspector-main">
          <div className="ai-inspector-tabs" role="tablist" aria-label="运行详情">
            <button type="button" className={tab === "overview" ? "active" : ""} onClick={() => onTabChange("overview")}>概览</button>
            <button type="button" className={tab === "activity" ? "active" : ""} onClick={() => onTabChange("activity")}>活动 {events.length ? `(${events.length})` : ""}</button>
            <button type="button" className={tab === "chain" ? "active" : ""} onClick={() => onTabChange("chain")}>工具链 {run?.tool_calls.length ? `(${run.tool_calls.length})` : ""}</button>
            <button type="button" className={tab === "token" ? "active" : ""} onClick={() => onTabChange("token")} title="查看 Token 消耗与多次对话的用量变化">Token</button>
            <button type="button" className={tab === "budget" ? "active" : ""} onClick={() => onTabChange("budget")} title="查看步数上限、Token 预算与模型信息">预算</button>
          </div>
          {tab === "overview" && (
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
          {tab === "activity" && <div className="ai-detail-content">{events.length ? <AgentTimeline events={events} /> : <div className="muted">暂无执行事件。</div>}</div>}
          {tab === "chain" && (
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
          {tab === "token" && (
            <div className="ai-detail-content">
              {run ? <TokenUsageBody run={run} /> : <p className="muted">发起任务后可查看本次运行的 Token 消耗。</p>}
              <div className="ai-run-history">
                <div className="ai-section-kicker">每次对话的调用次数与调用量变化</div>
                <RunUsageHistory history={history} />
              </div>
            </div>
          )}
          {tab === "budget" && (
            <div className="ai-detail-content">
              <CapabilitiesPanel capabilities={capabilities} loading={capsLoading} error={capsError} onRetry={loadCapabilities} />
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
