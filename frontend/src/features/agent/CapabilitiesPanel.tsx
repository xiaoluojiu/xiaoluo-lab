import type { AgentCapabilities } from "../../types/agent";

/**
 * Agent 能力面板：展示后端 /agent/capabilities 暴露的运行上限与策略。
 * 作用：让「我这次能跑几步、还剩多少 Token、用的是哪个模型」在 UI 上可见，
 * 避免用户对着卡住的执行流猜测原因。
 */
export function CapabilitiesPanel({
  capabilities,
  loading,
  error,
  onRetry,
}: {
  capabilities: AgentCapabilities | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  if (loading) {
    return (
      <div className="ai-detail-content">
        <div className="skeleton skeleton-text" style={{ width: "70%" }} />
        <div className="skeleton skeleton-text" style={{ width: "90%" }} />
        <div className="skeleton skeleton-text" style={{ width: "55%" }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="ai-detail-content">
        <div className="error-box">
          <div className="error-title">未能读取 Agent 能力</div>
          <div className="error-detail">{error}</div>
          <div className="error-action"><button className="btn btn-sm" type="button" onClick={onRetry}>重试</button></div>
        </div>
      </div>
    );
  }

  if (!capabilities) return <div className="ai-detail-content"><div className="muted">暂无能力信息。</div></div>;

  /**
   * 逐字段可选化。
   * 历史缺陷：原来直接 `const { agent, llm, tools } = capabilities` 后立刻访问
   * `agent.llm_budget` / `llm.model` / `tools.count`，一旦后端结构变更或中间层裁掉字段，
   * 这里会抛错把整个 AI 页面打成白屏 —— 影响面远超「预算页签坏了」。现在缺字段只降级为 "—"。
   */
  if (!capabilities.agent && !capabilities.llm && !capabilities.tools) {
    return <div className="ai-detail-content"><div className="muted">能力信息结构不完整，无法展示。</div></div>;
  }
  const agent = capabilities.agent;
  const llm = capabilities.llm;
  const tools = capabilities.tools;
  const budget = agent?.llm_budget;
  const policy = agent?.agent_policy;

  return (
    <div className="ai-detail-content ai-capability-panel">
      <div className="ai-capability-group">
        <div className="ai-section-kicker">模型</div>
        <div className="ai-capability-row"><span>提供方</span><strong>{llm?.provider_type ?? "—"}</strong></div>
        <div className="ai-capability-row"><span>模型</span><strong>{llm?.model ?? "—"}</strong></div>
        <div className="ai-capability-row"><span>上下文窗口</span><strong>{formatNumber(llm?.context_window)}</strong></div>
        <div className="ai-capability-row">
          <span>API Key</span>
          <strong className={llm?.api_key_set ? "cap-ok" : "cap-warn"}>{llm?.api_key_set ? "已配置" : "未配置"}</strong>
        </div>
      </div>

      <div className="ai-capability-group">
        <div className="ai-section-kicker">运行上限</div>
        <div className="ai-capability-row"><span>最大步数</span><strong>{agent?.max_steps ?? "—"}</strong></div>
        <div className="ai-capability-row"><span>LLM 调用上限</span><strong>{budget?.max_calls ?? "—"}</strong></div>
        <div className="ai-capability-row"><span>Token 上限</span><strong>{formatNumber(budget?.max_total_tokens)}</strong></div>
        <div className="ai-capability-row"><span>工具总数</span><strong>{tools?.count ?? "—"}</strong></div>
      </div>

      <div className="ai-capability-group">
        <div className="ai-section-kicker">优化策略</div>
        {policySwitches(policy).map((item) => (
          <div className="ai-capability-row" key={item.label}>
            <span>{item.label}</span>
            <strong className={item.on == null ? "cap-off" : item.on ? "cap-ok" : "cap-off"}>{item.on == null ? "未知" : item.on ? "开启" : "关闭"}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function policySwitches(policy: AgentCapabilities["agent"]["agent_policy"] | undefined) {
  return [
    { label: "工具检索", on: policy?.enable_tool_retrieval },
    { label: "结果压缩", on: policy?.enable_result_compression },
    { label: "计划缓存", on: policy?.enable_plan_cache },
    { label: "模型兜底", on: policy?.allow_model_fallback },
  ];
}

/** 数字缺失时不要渲染成 "NaN"，统一降级为 "—"。 */
function formatNumber(value: number | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString("zh-CN");
}
