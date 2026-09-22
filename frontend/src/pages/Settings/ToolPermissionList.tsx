import { useState } from "react";
import type { AgentToolInfo } from "../../api/agent";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { InfoHint } from "../../components/InfoHint";
import {
  effectiveAutoAllow,
  hasExplicitPermission,
  readToolPermissions,
  resetToolPermissions,
  setToolPermission,
  setToolPermissions,
  type ToolPermissionMap,
} from "../../lib/toolPermissions";

/**
 * 工具授权开关列表。
 *
 * 历史问题：设置页里两处工具目录都只是静态展示（名称 + 分类 + 「需授权 / 自动」徽章），
 * 用户无法调整任何一个工具的授权策略。现在每个工具都有独立开关：
 * 开启 = Agent 调用时自动放行；关闭 = 每次调用前弹窗确认。状态写入 localStorage
 * （见 lib/toolPermissions），与外观设置一样是本浏览器作用域。
 *
 * 文案口径（用户 2026-09-22 要求）：本板块内容较长，说明不再平铺在标题下，
 * 统一折进标题旁的 ⓘ，默认不占版面。
 *
 * 「全部开启」的边界（安全考量，勿放宽）：
 * 只作用于**低风险**工具。高风险工具（requires_confirmation=true，如 data.write /
 * ml.train / workflow.generate）必须逐个手动开启——因为「一键解除所有写操作防护」
 * 与「清理浏览器缓存」这种可逆操作的风险完全不对等，后者反倒还弹确认框。
 */
export function ToolPermissionList({
  tools,
  loading,
  onReload,
  title = "可调用工具与授权",
  description = "开启＝调用时自动放行，关闭＝每次调用前弹窗确认；授权状态只保存在本浏览器。",
}: {
  tools: AgentToolInfo[];
  loading?: boolean;
  onReload?: () => void;
  title?: string;
  description?: string;
}) {
  const [perms, setPerms] = useState<ToolPermissionMap>(readToolPermissions);
  const [confirmAllOff, setConfirmAllOff] = useState(false);

  function toggle(name: string, enabled: boolean) {
    setPerms(setToolPermission(name, enabled));
  }

  /** 低风险工具集合：这些才允许被「全部开启」批量放行。 */
  const lowRiskTools = tools.filter((tool) => !tool.requires_confirmation);
  const highRiskCount = tools.length - lowRiskTools.length;

  /** 批量开启：只覆盖低风险工具，高风险工具保持原策略。 */
  function enableLowRisk() {
    setPerms(setToolPermissions(
      Object.fromEntries(lowRiskTools.map((tool) => [tool.name, true])),
    ));
  }

  /** 批量关闭：对所有工具逐个询问，是安全方向的批量操作，无需二次确认。 */
  function disableAll() {
    setPerms(setToolPermissions(Object.fromEntries(tools.map((tool) => [tool.name, false]))));
    setConfirmAllOff(false);
  }

  /**
   * 恢复默认：清空全部显式覆盖，回到「跟随后端风险声明」。
   *
   * 注意：这里**故意不再写一次 localStorage**。`resetToolPermissions()` 的返回值
   * 就是 `{}`，`setPerms({})` 已经完成 state 同步，再写一次是多余的 IO。
   * 这个「少写一次」的写法容易被误认为 bug 而"修好"，故在此锁定语义。
   */
  function restore() {
    setPerms(resetToolPermissions());
  }

  const enabledCount = tools.filter((tool) => effectiveAutoAllow(tool.name, tool.requires_confirmation)).length;

  return (
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            {title}
            {description && <InfoHint label={`${title}说明`}>{description}</InfoHint>}
          </h3>
        </div>
        {onReload && (
          <button className="btn" disabled={loading} onClick={onReload}>
            {loading ? "读取中..." : "刷新"}
          </button>
        )}
      </div>
      <div className="settings-tool-toolbar">
        <span className="badge">自动放行 {enabledCount} / {tools.length} 个工具</span>
        <div className="settings-tool-toolbar-actions">
          <button
            className="btn btn-sm"
            type="button"
            disabled={!lowRiskTools.length}
            onClick={enableLowRisk}
            title={highRiskCount ? `只放行低风险工具；${highRiskCount} 个高风险工具需逐个开启` : undefined}
          >
            放行低风险工具
          </button>
          <button className="btn btn-sm" type="button" disabled={!tools.length} onClick={() => setConfirmAllOff(true)}>全部关闭</button>
          <button className="btn btn-sm" type="button" disabled={!tools.length} onClick={restore} title="清除全部手动设置，恢复为跟随后端风险声明">恢复默认</button>
        </div>
      </div>
      {highRiskCount > 0 && (
        <p className="settings-note settings-note-warn">
          有 {highRiskCount} 个高风险工具（写数据 / 训练 / 发布）默认每次询问，需要逐个手动开启。
        </p>
      )}
      {tools.length === 0 && !loading && <p className="settings-note settings-note-warn">无法读取工具列表</p>}
      {tools.length > 0 && (
        <div className="settings-tool-list">
          {tools.map((tool) => {
            const on = effectiveAutoAllow(tool.name, tool.requires_confirmation);
            return (
              <div key={tool.name} className={`settings-tool-row${on ? " is-on" : " is-off"}`}>
                <div className="settings-tool-main">
                  <span className="settings-tool-name">{tool.name}</span>
                  <span className="badge">{tool.category}</span>
                  <span className={`badge ${tool.requires_confirmation ? "failed" : "success"}`}>
                    {tool.requires_confirmation ? "高风险" : "低风险"}
                  </span>
                  {hasExplicitPermission(tool.name) && <span className="badge">已自定义</span>}
                </div>
                <p className="settings-tool-desc">{tool.description}</p>
                <label className="settings-tool-switch">
                  <span className="switch">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={(e) => toggle(tool.name, e.target.checked)}
                      aria-label={`${tool.name} 自动放行`}
                    />
                    <span className="track" />
                    <span className="thumb" />
                  </span>
                  <span className="settings-tool-switch-text">{on ? "自动放行（不再询问）" : "每次调用前询问"}</span>
                </label>
              </div>
            );
          })}
        </div>
      )}
      <ConfirmDialog
        open={confirmAllOff}
        title="关闭全部自动放行？"
        message={<>所有工具都会回到「每次调用前询问」，包括你手动开启过的低风险工具。高风险工具的自动放行同样会被关闭。</>}
        confirmText="全部关闭"
        onConfirm={disableAll}
        onCancel={() => setConfirmAllOff(false)}
      />
    </section>
  );
}
