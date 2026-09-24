import { useCallback, useEffect, useRef, useState } from "react";
import { getAgentSettings, getLlmSettings, updateAgentSettings, type AgentSettings, type LlmModelSettings } from "../../api/settings";
import { listTools, type AgentToolInfo } from "../../api/agent";
import { readBrowserLlm } from "../../lib/browserLlm";
import { applyLlmToBackend, backendModelFingerprint, browserModelFingerprint } from "../../lib/llmSync";
import { numberOrPrevious } from "../../lib/numberInput";
import { InfoHint } from "../../components/InfoHint";
import { ToolPermissionList } from "./ToolPermissionList";

type Props = { onToast?: (text: string) => void };

export default function AgentModelPanel({ onToast }: Props) {
  const [model, setModel] = useState<LlmModelSettings | null>(null);
  const [tools, setTools] = useState<AgentToolInfo[]>([]);
  const [saving, setSaving] = useState(false);
  const [syncingModel, setSyncingModel] = useState(false);
  const [draft, setDraft] = useState<AgentSettings | null>(null);
  const [browserModel, setBrowserModel] = useState(() => readBrowserLlm().model);
  const [loadError, setLoadError] = useState("");

  /**
   * onToast 用 ref 持有，effect 依赖保持空数组。
   *
   * 历史问题：effect 依赖写的是 [onToast]，而调用方一旦传入内联箭头函数，
   * 父组件每次重渲染都会产生新引用 → effect 重跑 → 重复三次网络请求
   * （getLlmSettings + getAgentSettings + listTools）并重复触发自动同步。
   */
  const onToastRef = useRef(onToast);
  useEffect(() => { onToastRef.current = onToast; }, [onToast]);
  const toast = useCallback((text: string) => onToastRef.current?.(text), []);

  /** 首次加载：读后端模型策略 + Agent 策略 + 工具目录，然后尝试一次静默同步。 */
  useEffect(() => {
    let cancelled = false;
    void Promise.all([getLlmSettings(), getAgentSettings(), listTools()])
      .then(([m, a, t]) => {
        if (cancelled) return;
        setModel(m);
        setDraft(a);
        setTools(t);
        setLoadError("");
        void autoSyncModel(m);
      })
      .catch(() => {
        if (!cancelled) setLoadError("无法读取 Agent 模型策略，请检查后端服务是否可用。");
      });
    return () => { cancelled = true; };
    // autoSyncModel 是本组件内的稳定逻辑（只依赖 ref 与 setState），无需进依赖数组
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function patch(partial: Partial<AgentSettings>) {
    setDraft((current) => current ? ({ ...current, ...partial }) : current);
  }

  function patchPolicy(partial: Partial<AgentSettings["agent_policy"]>) {
    setDraft((current) => current ? ({ ...current, agent_policy: { ...current.agent_policy, ...partial } }) : current);
  }

  function patchSection(key: keyof AgentSettings["context_sections"], value: number) {
    setDraft((current) => current ? ({
      ...current,
      context_sections: { ...current.context_sections, [key]: value },
    }) : current);
  }

  /**
   * 把浏览器里的 AI 配置应用到后端。
   *
   * 历史问题：旧实现无条件写入，于是「改了表单没保存 → 切走 → 切回来」会用
   * 浏览器里的旧值反复覆盖后端全局凭据。现在统一走 lib/llmSync，
   * 内部先比较 base_url + model 指纹，已一致就直接返回不写。
   *
   * @param auto true = 进入页面的静默同步（不弹提示、不显示 loading）
   * @param serverState 调用方已取到的后端摘要，避免重复请求
   */
  async function syncModel(auto = false, serverState: LlmModelSettings | null = null) {
    if (!auto) setSyncingModel(true);
    const result = await applyLlmToBackend({ force: !auto, serverState });
    if (!auto) setSyncingModel(false);

    switch (result.status) {
      case "applied":
        setModel(result.model);
        setBrowserModel(readBrowserLlm().model);
        toast("当前 AI 配置已应用到 Agent 后端");
        return true;
      case "already-synced":
        setModel(result.model);
        if (!auto) toast("当前 AI 配置与后端一致，无需重复同步");
        return true;
      case "not-configured":
        if (!auto) toast("请先在上方 AI 服务中填写并保存 Base URL、模型名和 API Key");
        return false;
      case "failed":
        if (!auto) toast(result.message);
        return false;
    }
  }

  /** 进入设置页时静默同步一次，避免用户忘记点同步导致后端仍用旧配置。 */
  async function autoSyncModel(serverState: LlmModelSettings | null) {
    const synced = await syncModel(true, serverState);
    if (synced) toast("已自动将当前 AI 配置应用到 Agent");
  }

  async function save() {
    if (!draft) return;
    setSaving(true);
    try {
      const saved = await updateAgentSettings({
        context_max_chars: draft.context_max_chars,
        context_sections: draft.context_sections,
        history_messages: draft.history_messages,
        dataset_cache_enabled: draft.dataset_cache.enabled,
        dataset_cache_max_items: draft.dataset_cache.max_items,
        max_calls: draft.llm_budget.max_calls,
        max_input_tokens: draft.llm_budget.max_input_tokens,
        max_output_tokens: draft.llm_budget.max_output_tokens,
        max_total_tokens: draft.llm_budget.max_total_tokens,
        enable_tool_retrieval: draft.agent_policy.enable_tool_retrieval,
        tool_retrieval_top_k: draft.agent_policy.tool_retrieval_top_k,
        tool_retrieval_min_score: draft.agent_policy.tool_retrieval_min_score,
        enable_result_compression: draft.agent_policy.enable_result_compression,
        enable_plan_cache: draft.agent_policy.enable_plan_cache,
        plan_cache_max_items: draft.agent_policy.plan_cache_max_items,
        // 真开关：关掉 ⇒ 远程失败直接失败；开启 ⇒ 退回平台内置规则。
        allow_model_fallback: draft.agent_policy.allow_model_fallback,
      });
      setDraft(saved);
      toast("Agent 设置已保存（当前后端进程生效）");
    } catch (error) {
      toast(error instanceof Error ? error.message : "Agent 设置保存失败");
    } finally {
      setSaving(false);
    }
  }

  /** 浏览器配置与后端生效配置是否指向同一组 base_url + model。 */
  const backendFp = backendModelFingerprint(model);
  const browserFp = browserModelFingerprint();
  const missingKey = Boolean(model && !model.api_key_set);
  const synced = Boolean(backendFp && browserFp && backendFp === browserFp && !missingKey);
  const browser = readBrowserLlm();

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            Agent 模型能力
            <InfoHint label="Agent 模型能力说明">
              一次 Agent 任务只使用当前选定的一个模型，不会自动切换或回退；「后端生效模型」是 Agent 实际调用的，
              「浏览器待应用模型」需在「AI 服务」中填写并应用后才生效。
            </InfoHint>
          </h3>
        </div>
        <span className={`badge ${synced ? "success" : "failed"}`}>
          {synced ? "已同步" : "未同步"}
        </span>
      </div>

      {loadError && <p className="settings-note settings-note-warn">{loadError}</p>}

      {model ? <>
        {/* 前后端两个状态并列显示：只显示一个「当前模型」正是旧版误导用户的根源。 */}
        <div className="settings-stat-grid">
          <div>
            <small>后端生效模型</small>
            <strong>{model.model || "—"}</strong>
          </div>
          <div>
            <small>浏览器待应用模型</small>
            <strong>{browser.model || "未填写"}</strong>
          </div>
        </div>

        {(missingKey || !synced) && (
          <p className="settings-note settings-note-warn">
            {missingKey && browser.api_key
              ? "后端尚未收到 API Key，Agent 会退化为规则规划器。点击下方按钮应用配置。"
              : "浏览器配置与后端生效配置不一致，点下方按钮将其应用到 Agent。"}
          </p>
        )}
      </> : !loadError && <p className="settings-empty">读取模型策略...</p>}

      <div className="settings-actions">
        <button className="btn primary" disabled={syncingModel} onClick={() => void syncModel()}>
          {syncingModel ? "同步中..." : "将浏览器配置应用到 Agent"}
        </button>
      </div>
    </section>

    <ToolPermissionList tools={tools} title="Agent 工具目录与授权" />

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            Agent 预算与策略
            <InfoHint label="Agent 预算与策略生效范围">
              这里的开关和数值会立即作用于当前后端进程，重启后仍以后端 .env / 默认配置为准。
            </InfoHint>
          </h3>
        </div>
        <button className="btn primary" disabled={saving || !draft} onClick={() => void save()}>
          {saving ? "保存中..." : "保存 Agent 设置"}
        </button>
      </div>
      {draft ? <>
        <div className="settings-form-grid">
          <NumberField label="最大 LLM 调用次数" min={1} max={50} value={draft.llm_budget.max_calls}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_calls: v } })} />
          <NumberField label="最大上下文字符数" min={1000} max={100000} value={draft.context_max_chars}
            onChange={(v) => patch({ context_max_chars: v })} />
          <NumberField label="输入 Token 上限" min={256} value={draft.llm_budget.max_input_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_input_tokens: v } })} />
          <NumberField label="输出 Token 上限" min={128} value={draft.llm_budget.max_output_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_output_tokens: v } })} />
          <NumberField label="总 Token 上限" min={512} value={draft.llm_budget.max_total_tokens}
            onChange={(v) => patch({ llm_budget: { ...draft.llm_budget, max_total_tokens: v } })} />
          <NumberField label="历史消息数量" min={0} max={50} value={draft.history_messages}
            onChange={(v) => patch({ history_messages: v })} />
        </div>

        <div className="settings-panel-heading" style={{ marginTop: 22 }}>
        <div>
          <h4 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            上下文分区预算
            <InfoHint label="上下文分区预算说明">
              分别控制用户请求、数据集、任务、权限、工具和历史进入 Agent 上下文的最大字符数。
            </InfoHint>
          </h4>
        </div>
        </div>
        <div className="settings-form-grid">
          <NumberField label="用户请求" min={80} value={draft.context_sections.user_request}
            onChange={(v) => patchSection("user_request", v)} />
          <NumberField label="数据集上下文" min={80} value={draft.context_sections.dataset}
            onChange={(v) => patchSection("dataset", v)} />
          <NumberField label="任务上下文" min={80} value={draft.context_sections.task}
            onChange={(v) => patchSection("task", v)} />
          <NumberField label="权限信息" min={80} value={draft.context_sections.permissions}
            onChange={(v) => patchSection("permissions", v)} />
          <NumberField label="工具描述" min={80} value={draft.context_sections.tools}
            onChange={(v) => patchSection("tools", v)} />
          <NumberField label="历史上下文" min={80} value={draft.context_sections.history}
            onChange={(v) => patchSection("history", v)} />
        </div>

        <div className="settings-status-list" style={{ marginTop: "var(--space-4)" }}>
          <label title="开启：远程大模型失败（欠费 / 超时 / 5xx）时退回平台内置规则，数据分析仍能跑出真实结果。关闭：远程失败就直接失败。"><span>模型兜底（远程失败降级）</span>
            <input type="checkbox" checked={draft.agent_policy.allow_model_fallback}
              onChange={(e) => patchPolicy({ allow_model_fallback: e.target.checked })} /></label>
          <label><span>Tool Retrieval</span>
            <input type="checkbox" checked={draft.agent_policy.enable_tool_retrieval}
              onChange={(e) => patchPolicy({ enable_tool_retrieval: e.target.checked })} /></label>
          <label><span>工具结果压缩</span>
            <input type="checkbox" checked={draft.agent_policy.enable_result_compression}
              onChange={(e) => patchPolicy({ enable_result_compression: e.target.checked })} /></label>
          <label><span>Plan Cache</span>
            <input type="checkbox" checked={draft.agent_policy.enable_plan_cache}
              onChange={(e) => patchPolicy({ enable_plan_cache: e.target.checked })} /></label>
          <label><span>数据集上下文缓存</span>
            <input type="checkbox" checked={draft.dataset_cache.enabled}
              onChange={(e) => patch({ dataset_cache: { ...draft.dataset_cache, enabled: e.target.checked } })} /></label>
        </div>
        <div className="settings-form-grid" style={{ marginTop: "var(--space-4)" }}>
          <NumberField label="工具召回数量" min={1} max={30} value={draft.agent_policy.tool_retrieval_top_k}
            onChange={(v) => patchPolicy({ tool_retrieval_top_k: v })} />
          <NumberField label="召回最低相关度" min={0} max={1} step={0.01} value={draft.agent_policy.tool_retrieval_min_score}
            onChange={(v) => patchPolicy({ tool_retrieval_min_score: v })} />
          <NumberField label="Plan Cache 容量" min={1} max={500} value={draft.agent_policy.plan_cache_max_items}
            onChange={(v) => patchPolicy({ plan_cache_max_items: v })} />
          <NumberField label="数据集缓存数量" min={1} max={500} value={draft.dataset_cache.max_items}
            onChange={(v) => patch({ dataset_cache: { ...draft.dataset_cache, max_items: v } })} />
        </div>
      </> : <p className="settings-empty">读取 Agent 策略...</p>}
    </section>
  </>;
}

/**
 * 数字输入框：清空时保持上一个有效值，而不是把空串变成 0。
 *
 * 历史问题：直接 `Number(e.target.value)` 时，用户清空输入框会得到 0，
 * 绕过 HTML 的 min 约束（min 只参与表单校验，不约束 React 受控值），
 * 提交后被后端 Pydantic 的 ge= 约束拦下返回 422——用户看到的是报错而非即时的输入反馈。
 */
function NumberField({
  label, value, onChange, min, max, step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="field">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(numberOrPrevious(e.target.value, value))}
      />
    </label>
  );
}
