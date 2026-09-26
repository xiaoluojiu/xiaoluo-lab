/** Token 用量相关的纯展示块（预算明细 + 多次运行的用量变化）。 */
import { InfoHint } from "../../components/InfoHint";
import type { AgentRun } from "../../types/agent";
import type { RunUsageSnapshot } from "./hooks/useAgentUsage";

/** Token 预算与用量明细。仅保留数据本体，卡片外壳由外层容器提供。 */
export function TokenUsageBody({ run }: { run: AgentRun }) {
  const actual = run.token_usage?.actual;
  const optimization = run.token_usage?.optimization;
  const budget = run.token_usage?.budget;
  if (!actual || !optimization) {
    return <p className="muted" style={{ margin: 0 }}>本次运行未上报 Token 用量。</p>;
  }
  const tokenPct = budget ? Math.min(100, Math.round((actual.total_tokens / Math.max(budget.max_total_tokens, 1)) * 100)) : 0;
  const callPct = budget ? Math.min(100, Math.round((run.token_usage.llm_calls / Math.max(budget.max_llm_calls, 1)) * 100)) : 0;
  // 统一 Loop 的三档调用计数（远程 / 本地 Qwen / 工具），用于直观看到「简单任务零远程」。
  const remoteCalls = run.token_usage.remote_calls ?? run.token_usage.llm_calls ?? 0;
  const qwenCalls = run.token_usage.qwen_calls ?? 0;
  const toolCalls = run.token_usage.tool_calls ?? run.tool_call_count ?? 0;
  return (
    <div className="ai-token-inner">
      <div className="ai-run-summary">
        <div><strong>{actual.total_tokens.toLocaleString()}</strong><span>已用 Token</span></div>
        <div><strong>{budget ? budget.remaining_total_tokens.toLocaleString() : "—"}</strong><span>剩余 Token</span></div>
        <div><strong>{budget ? `${run.token_usage.llm_calls}/${budget.max_llm_calls}` : run.token_usage.llm_calls}</strong><span>LLM 调用</span></div>
        <div><strong>{remoteCalls}</strong><span>远程决策</span></div>
        <div><strong>{qwenCalls}</strong><span>本地 Qwen</span></div>
        <div><strong>{toolCalls}</strong><span>工具调用</span></div>
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
export function RunUsageHistory({ history }: { history: RunUsageSnapshot[] }) {
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
