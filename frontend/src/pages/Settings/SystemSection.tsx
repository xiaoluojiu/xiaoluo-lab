import { useEffect, useState } from "react";
import { getStorageSummary } from "../../api/settings";
import { InfoHint } from "../../components/InfoHint";

/**
 * 系统与关于分区：运行状态 + 数据存放位置。
 *
 * 状态口径：旧版「AI 配置」一栏判断的是 `llm.base_url && llm.model`，只看浏览器里填了什么，
 * 不看后端是否真的收到了——因此会出现「显示已配置，但 Agent 实际在用规则规划器」的误导。
 * 现在以后端 api_key_set 为准。
 *
 * 外壳约定：返回 Fragment，`.settings-stack` 由页面层统一提供（避免双层 gap 叠加）。
 */
export function SystemSection({
  backendModel, backendKeySet, remoteEnabled,
}: {
  backendModel: string;
  backendKeySet: boolean;
  /** 「启用远程 API 大模型」总开关；关闭时 Agent 用的是平台内置规则。 */
  remoteEnabled: boolean;
}) {
  const [storageOk, setStorageOk] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStorageSummary()
      .then(() => !cancelled && setStorageOk(true))
      .catch(() => !cancelled && setStorageOk(false));
    return () => { cancelled = true; };
  }, []);

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div><h3>系统状态</h3></div>
      </div>
      <div className="settings-status-list">
        <div>
          <span>Agent 模型</span>
          {/* 三态：远程开关关着时不能报「已连上 xxx」——那会与实际行为了不符（静默失效类误导）。 */}
          <b className={remoteEnabled && backendKeySet ? "status-ok" : "status-warn"}>
            {!remoteEnabled
              ? "远程 API 已停用（用平台内置规则）"
              : backendKeySet ? `已连上 ${backendModel || "—"}` : "未配置（退化为规则规划器）"}
          </b>
        </div>
        <div>
          <span>服务器 Storage</span>
          <b className={storageOk === false ? "status-warn" : undefined}>
            {storageOk === null ? "检查中" : storageOk ? "可用" : "不可用"}
          </b>
        </div>
      </div>
    </section>

    {/* 原来的「系统原则」卡已删；剩余唯一的实质约定（不覆盖输入数据）折进标题旁的 ⓘ。 */}
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            关于小洛实验室
            <InfoHint label="数据处理约定">数据以新版本保存，不覆盖输入数据。</InfoHint>
          </h3>
        </div>
      </div>
      <div className="settings-about">
        <span>外观与工具授权</span><strong>本浏览器（localStorage）</strong>
        <span>AI 连接凭据</span><strong>本标签页（sessionStorage）+ 后端进程</strong>
        <span>Agent 预算与策略</span><strong>后端进程内存（重启回到 .env）</strong>
        <span>服务器数据与实验记录</span><strong>由后端 API 与 Storage 管理</strong>
      </div>
    </section>
  </>;
}
