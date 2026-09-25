import { useState } from "react";
import { readBrowserLlm, type BrowserLlmConfig } from "../../lib/browserLlm";
import { InfoHint } from "../../components/InfoHint";

/**
 * AI 服务分区：远程 API 大模型总开关 + Base URL / 模型名 / API Key。
 *
 * 与 AgentModelPanel 的分工（修复后的清晰边界）：
 * - 本组件负责「连接什么」——开关远程大模型、编辑并保存到浏览器 + 后端；
 * - AgentModelPanel 负责「怎么用」——预算、策略与工具授权，并展示后端/浏览器两侧的模型状态。
 *
 * 文案口径（用户 2026-09-22 要求）：常驻说明小字一律删掉。状态用徽章表达，
 * 只有「会静默失效 / 不持久化 / 操作前提」这类真正影响判断的信息才折进 ⓘ（components/InfoHint）。
 */
export function AiServiceSection({
  backendKeySet, remoteEnabled, onToggleRemote, onSave, onTest,
}: {
  backendKeySet: boolean;
  /** 「启用远程 API 大模型」总开关；关闭 = Agent 走平台自带模型。 */
  remoteEnabled: boolean;
  onToggleRemote: (enabled: boolean) => void;
  /**
   * 保存表单里的当前值。
   *
   * ★ 必须带 `llm` 入参：表单 state 与 storage 是两份数据，表单编辑不落盘，
   * 若这里不传值，保存函数只能去 storage 读旧值 —— 表现为「测试连接成功、
   * 保存却没生效」。失败时应返回 false，让调用方保留「未应用」标记。
   */
  onSave: (llm: BrowserLlmConfig) => Promise<boolean>;
  onTest: (llm: BrowserLlmConfig) => Promise<string>;
}) {
  const [llm, setLlm] = useState<BrowserLlmConfig>(readBrowserLlm);
  const [keyVisible, setKeyVisible] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testMessage, setTestMessage] = useState("");
  const [dirty, setDirty] = useState(false);

  function patch(partial: Partial<BrowserLlmConfig>) {
    setLlm((current) => ({ ...current, ...partial }));
    setDirty(true);
  }

  const complete = Boolean(llm.base_url.trim() && llm.model.trim() && llm.api_key.trim());
  // 开关关掉时这组配置整体停用，因此不再提示「有未应用的改动」——那时应用了也不生效。
  const pendingApply = remoteEnabled && (dirty || !backendKeySet);
  /**
   * 停用标记：**只影响提示，不再禁用输入与按钮**。
   *
   * 历史缺陷：`locked` 曾经同时给三个输入框和「保存并应用」「测试连接」都加上
   * `disabled`。而开关的语义是「凭据原样保留，重新开启立即生效」（见下方 InfoHint），
   * 也就是说关闭期间配置凭据是**有意义的**操作 —— 一禁用，用户按「关闭外部 LLM 后
   * 测试」的流程走，会发现整块变灰、保存点不动，直接表现为「保存并应用失败」。
   * 现在改为：停用期间照常可编辑可保存，只是明确说明「何时才生效」。
   */
  const locked = !remoteEnabled;

  const badge = locked
    ? { cls: "warning", text: "远程 API 已停用" }
    : backendKeySet
      ? { cls: "success", text: "后端已配置" }
      : { cls: "failed", text: "后端未配置" };

  /**
   * 保存后清「未保存改动」标记。
   *
   * 历史问题：标记原先靠 `useEffect([backendModel, backendKeySet])` 间接清除，
   * 只改 Base URL 时后端返回的 model/api_key_set 都没变，state 不触发 → 标记一直挂着，
   * 页面始终显示「有尚未应用的改动」。现在由保存动作本身按返回结果清除。
   */
  async function runSave() {
    // 把表单当前值交给保存函数（而不是让它去 storage 里读旧值）。
    if (await onSave(llm)) setDirty(false);
  }

  async function runTest() {
    setTesting(true);
    setTestMessage("");
    setTestMessage(await onTest(llm));
    setTesting(false);
  }

  // 不再自带 .settings-stack 外壳：本组件被父级 settings-stack 包裹，
  // 双层 grid 会把 16px 间距叠加成 32px，导致卡片明显错位。
  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            AI 服务
            <InfoHint label="AI 服务配置说明">
              这里的连接配置由 AI Lab、Workflow 等所有 AI 功能共享；保存并应用后 Agent 才会使用它。
            </InfoHint>
          </h3>
        </div>
        <span className={`badge ${badge.cls}`}>{badge.text}</span>
      </div>

      {/* 总开关：整块远程配置的总统领，放在最前面。关掉它，Agent 走平台内置规则。 */}
      <div className="settings-status-list">
        <div style={{ alignItems: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            启用远程 API 大模型
            <InfoHint label="远程 API 大模型开关说明">
              关闭后 Agent 不再调用远程大模型，改由平台内置规则处理：数据请求照常执行，问候／能力／用法类问话由内置应答回复，其余开放式对话没有生成能力（凭据原样保留，重新开启立即生效）。
            </InfoHint>
          </span>
          <label className="settings-tool-switch" style={{ marginTop: 0 }}>
            <span className="switch">
              <input
                type="checkbox"
                checked={remoteEnabled}
                onChange={(e) => onToggleRemote(e.target.checked)}
                aria-label="启用远程 API 大模型"
              />
              <span className="track" />
              <span className="thumb" />
            </span>
            <span className="settings-tool-switch-text">{remoteEnabled ? "已启用" : "已停用"}</span>
          </label>
        </div>
      </div>

      <div className="settings-form-grid" style={{ marginTop: 18 }}>
        <label className="field">
          Base URL（OpenAI Compatible）
          <input type="text" value={llm.base_url} placeholder="https://api.example.com/v1"
            onChange={(e) => patch({ base_url: e.target.value })} />
        </label>
        <label className="field">
          模型名
          <input type="text" value={llm.model} placeholder="例如 gpt-4o-mini / deepseek-chat"
            onChange={(e) => patch({ model: e.target.value })} />
        </label>
      </div>

      {/* htmlFor 是必要的：label 内出现可聚焦的 ⓘ 按钮后，隐式关联会绑到按钮上，
          点「API Key」文字会去聚焦问号而不是输入框。 */}
      <label className="field" style={{ marginTop: 14 }} htmlFor="llm-api-key">
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          API Key
          <InfoHint label="API Key 存储说明">
            只保存在本标签页（关闭标签页即清除）；点「保存并应用」后写入后端进程内存，重启仍以 .env 为准。
          </InfoHint>
        </span>
        <div className="settings-key-wrap">
          <input id="llm-api-key" type={keyVisible ? "text" : "password"} value={llm.api_key} placeholder="输入 API Key"
            onChange={(e) => patch({ api_key: e.target.value })} />
          <button className="btn settings-key-toggle" type="button" onClick={() => setKeyVisible((v) => !v)}>
            {keyVisible ? "隐藏" : "显示"}
          </button>
        </div>
      </label>

      {locked && (
        <p className="settings-note">
          远程模型当前已停用，这组配置保存后<strong>不会立即被 Agent 使用</strong>；凭据会保留，重新开启开关即刻生效。
        </p>
      )}
      {pendingApply && complete && (
        <p className="settings-note settings-note-warn">
          有尚未应用的改动（或后端尚未收到 Key）。点「保存并应用」后 Agent 才会使用这组配置。
        </p>
      )}

      <div className="settings-actions">
        <button className="btn primary" type="button" onClick={() => void runSave()} disabled={!complete}>
          保存并应用
        </button>
        <button className="btn" type="button" disabled={testing} onClick={() => void runTest()}>
          {testing ? "连接测试中..." : "测试连接"}
        </button>
        {dirty && <button className="btn" type="button" onClick={() => { setLlm(readBrowserLlm()); setDirty(false); }}>
          放弃改动
        </button>}
        {testMessage && (
          <span className={`badge ${testMessage.startsWith("连接成功") ? "success" : "failed"}`}>{testMessage}</span>
        )}
      </div>
    </section>
  </>;
}
