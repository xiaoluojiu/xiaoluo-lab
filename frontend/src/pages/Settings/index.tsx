import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getLlmSettings, setRemoteLlm, testLlmConnection } from "../../api/settings";
import AgentModelPanel from "./AgentModelPanel";
import { PageHeader } from "../../components/PageHeader";
import { useToast } from "../../components/ToastProvider";
import { writeBrowserLlm, type BrowserLlmConfig } from "../../lib/browserLlm";
import { applyLlmToBackend } from "../../lib/llmSync";
import { readUi, writeUi, type UiSettings } from "../../lib/uiSettings";
import { AppearanceSection } from "./AppearanceSection";
import { AiServiceSection } from "./AiServiceSection";
import { DataSection } from "./DataSection";
import { SystemSection } from "./SystemSection";
import { NotificationSection } from "./NotificationSection";
import "./settings.css";

export type Section = "appearance" | "ai" | "data" | "system" | "notification";

const sections: { id: Section; title: string; description: string }[] = [
  { id: "appearance", title: "外观与体验", description: "主题、字体和界面密度" },
  { id: "ai", title: "AI 与模型", description: "共享的 AI 服务连接配置" },
  { id: "data", title: "数据与缓存", description: "服务器 Storage 与本地缓存" },
  { id: "notification", title: "通知", description: "通知偏好与邮箱" },
  { id: "system", title: "系统与关于", description: "运行状态与数据存放位置" },
];

export default function Settings() {
  // 分区状态放进 URL：刷新不丢、可分享，也不再手写 history.replaceState。
  const [searchParams, setSearchParams] = useSearchParams();
  const sectionParam = searchParams.get("section");
  const section: Section = sections.some((item) => item.id === sectionParam)
    ? (sectionParam as Section)
    : "appearance";

  const [ui, setUi] = useState<UiSettings>(readUi);
  const [backendModel, setBackendModel] = useState("");
  const [backendKeySet, setBackendKeySet] = useState(false);
  const [backendRemoteEnabled, setBackendRemoteEnabled] = useState(true);
  const toastApi = useToast();

  const selectedSection = useMemo(
    () => sections.find((item) => item.id === section) ?? sections[0],
    [section],
  );

  function changeSection(next: Section) {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("section", next);
    setSearchParams(nextParams, { replace: true });
  }

  useEffect(() => { writeUi(ui); }, [ui]);

  /** 后端生效的模型状态：供「AI 服务」与「系统状态」两处复用，避免各自请求。 */
  const refreshBackendModel = useCallback(async () => {
    try {
      const model = await getLlmSettings();
      setBackendModel(model.model);
      setBackendKeySet(model.api_key_set);
      setBackendRemoteEnabled(model.remote_enabled !== false);
    } catch {
      setBackendModel("");
      setBackendKeySet(false);
      setBackendRemoteEnabled(true);
    }
  }, []);

  useEffect(() => { void refreshBackendModel(); }, [refreshBackendModel]);

  /**
   * 切换「启用远程 API 大模型」总开关。
   *
   * 只切开关、不动凭据：关闭后 Agent 走平台内置规则（规则规划器 + 本地 Router + 内置应答），
   * 重新开启立即生效、不需要重填 Key。
   * 先乐观置位（受控 checkbox 否则会在网络往返期间弹回旧值），失败再以后端为准回滚。
   */
  async function toggleRemoteLlm(enabled: boolean) {
    setBackendRemoteEnabled(enabled);
    try {
      const next = await setRemoteLlm(enabled);
      setBackendRemoteEnabled(next.remote_enabled !== false);
      notify(enabled ? "已启用远程 API 大模型" : "已停用远程 API 大模型，Agent 改用平台内置规则");
    } catch (error) {
      notify(error instanceof Error ? error.message : "切换远程 API 开关失败", "error");
      await refreshBackendModel();
    }
  }

  // 统一的通知出口（全局 ToastProvider），不再维护页面私有的 toast 状态。
  const notify = useCallback((text: string, kind: "success" | "error" = "success") => {
    if (kind === "error") toastApi.error(text);
    else toastApi.success(text);
  }, [toastApi]);

  /** AgentModelPanel 的 onToast 需要稳定引用，否则会触发其内部重复请求（见该组件注释）。 */
  const handleAgentToast = useCallback((text: string) => notify(text), [notify]);

  function updateUi(partial: Partial<UiSettings>) {
    setUi((current) => ({ ...current, ...partial }));
  }

  /**
   * 保存 AI 配置：**同时写浏览器与后端**。
   *
   * 历史问题（严重）：旧实现只调 writeBrowserLlm()，完全不碰后端。而后端 Agent 的
   * Provider 由 settings.LLM_API_KEY 单一决定（deps.py:59，为空则退化为规则规划器）。
   * 于是 UI 显示「当前模型：gpt-4o-mini」，实际一次 LLM 调用都没发出去——
   * 用户以为配好了，去 AI Lab 提问得到的却是规则规划器的乱答。
   *
   * 现在把「保存」的语义补完整：保存到浏览器 + 立即应用到后端。
   *
   * ★ `llm` 必须是**表单里的当前值**（由 AiServiceSection 传入），不能在这里
   * `readBrowserLlm()` 重读 storage —— 那读到的永远是上一次保存的旧值，
   * 用户刚改的东西会被丢弃，而界面因为 force:true 恒返回 applied，还会提示"已保存"。
   * 表单值先进 storage（成为新的唯一事实源），再用同一份值推给后端。
   */
  async function saveLlm(llm: BrowserLlmConfig): Promise<boolean> {
    writeBrowserLlm(llm);
    const result = await applyLlmToBackend({ force: true, config: llm });
    await refreshBackendModel();
    switch (result.status) {
      case "applied":
        notify("AI 配置已保存，并已应用到 Agent 后端");
        return true;
      case "already-synced":
        notify("AI 配置已保存（后端已是同一配置）");
        return true;
      case "not-configured":
        notify("已保存到浏览器；三项未填全，暂未应用到后端", "error");
        return false;
      case "failed":
        notify(`已保存到浏览器，但应用到后端失败：${result.message}`, "error");
        return false;
    }
  }

  /** 测试连接：返回展示给用户的结果文案（不管是成功还是失败都返回字符串）。 */
  async function testConnection(llm: { base_url: string; model: string; api_key: string }): Promise<string> {
    if (!llm.base_url.trim()) { notify("请填写 Base URL", "error"); return "请填写 Base URL"; }
    if (!llm.model.trim()) { notify("请填写模型名", "error"); return "请填写模型名"; }
    if (!llm.api_key.trim()) { notify("请填写 API Key", "error"); return "请填写 API Key"; }
    try {
      const result = await testLlmConnection(llm);
      const message = result.ok ? `连接成功 · ${result.model ?? llm.model}` : result.message;
      notify(result.ok ? `LLM 连接测试成功 · ${result.model ?? llm.model}` : result.message, result.ok ? "success" : "error");
      return message;
    } catch (error) {
      const message = error instanceof Error ? error.message : "连接测试失败";
      notify(message, "error");
      return message;
    }
  }

  return <div className="settings-page">
    <PageHeader title="设置" />
    <div className="settings-layout">
      <aside className="settings-nav card" aria-label="设置分类">
        <div className="settings-nav-title">设置</div>
        <div className="settings-nav-list">
          {sections.map((item) => (
            <button key={item.id} type="button" className="settings-nav-item" data-active={section === item.id} onClick={() => changeSection(item.id)}>
              <span className="settings-nav-copy"><strong>{item.title}</strong><small>{item.description}</small></span>
              {section === item.id && <span className="settings-nav-dot" />}
            </button>
          ))}
        </div>
      </aside>
      <main className="settings-content">
        <div className="settings-content-header">
          {/* 只保留标题：同一句说明在左侧导航项下已经出现过一次，再重复一遍是纯噪声。 */}
          <div><h2>{selectedSection.title}</h2></div>
        </div>
        {/*
          外壳统一由本层提供：所有 section 组件都返回 Fragment，自己不套 .settings-stack。
          双层 stack 的 gap（16px）会叠加成 32px，卡片间距明显不均。
        */}
        <div className="settings-stack">
          {section === "appearance" && <AppearanceSection ui={ui} updateUi={updateUi} />}
          {section === "ai" && <>
            <AiServiceSection
              backendKeySet={backendKeySet}
              remoteEnabled={backendRemoteEnabled}
              onToggleRemote={(enabled) => void toggleRemoteLlm(enabled)}
              onSave={saveLlm}
              onTest={testConnection}
            />
            <AgentModelPanel onToast={handleAgentToast} />
          </>}
          {section === "data" && <DataSection notify={notify} onUiReset={updateUi} />}
          {section === "notification" && <NotificationSection notify={notify} />}
          {section === "system" && <SystemSection backendModel={backendModel} backendKeySet={backendKeySet} remoteEnabled={backendRemoteEnabled} />}
        </div>
      </main>
    </div>
  </div>;
}
