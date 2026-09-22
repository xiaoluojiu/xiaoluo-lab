import { useEffect } from "react";
import type { PermissionRequest } from "../types/agent";
import { riskLevelOf } from "../types/agent";

// Prompt 156：权限确认对话框 —— 展示 AI 准备做什么 / 使用什么数据 / 调用什么 Tool / 风险等级。
export function PermissionDialog({
  request,
  onAllow,
  onDeny,
  busy = false,
  note,
}: {
  request: PermissionRequest | null;
  onAllow: () => void;
  onDeny: () => void;
  /** 正在提交授权（避免连点造成重复 confirm）。 */
  busy?: boolean;
  /** 额外提示，例如「已按设置自动放行」。 */
  note?: string;
}) {
  /**
   * Esc = 拒绝。
   * 说明：本弹窗刻意**不**支持点遮罩关闭 —— 授权必须由用户显式选择「允许 / 拒绝」，
   * 避免误点遮罩把一次待确认的高风险调用悬空。Esc 归为「拒绝」是明确表态，语义安全。
   * 提交中（busy）不响应，防止请求在途时语义反转。
   */
  useEffect(() => {
    if (!request) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && !busy) { e.preventDefault(); onDeny(); }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [request, busy, onDeny]);

  if (!request) return null;
  const risk = riskLevelOf(request.tool);
  const riskClass = risk === "高" ? "failed" : risk === "中" ? "warning" : "success";
  return (
    <div className="dialog-mask" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label="AI 请求授权">
        <h3>AI 请求授权</h3>
        <div className="kv-grid">
          <div className="kv-item">
            <div className="k">调用工具</div>
            <div className="v" style={{ fontSize: 14 }}>{request.tool}</div>
          </div>
          <div className="kv-item">
            <div className="k">风险等级</div>
            <div className="v">
              <span className={`badge ${riskClass}`}>{risk}风险</span>
              {/* 风险等级由前端按工具名启发式推断，与后端声明可能不同，这里明确标注口径。 */}
              <span className="muted" style={{ fontSize: 11, marginLeft: 6 }}>前端估算</span>
            </div>
          </div>
        </div>
        <div className="mt">
          <div className="muted">操作参数</div>
          <pre
            style={{
              background: "var(--c-code-bg)",
              padding: 10,
              borderRadius: "var(--radius-sm)",
              fontSize: 12,
              overflow: "auto",
              maxHeight: 180,
            }}
          >
            {/* arguments 缺失时不能直接 JSON.stringify(undefined)：会渲染成空白，让用户无从判断。 */}
            {JSON.stringify(request.arguments ?? {}, null, 2)}
          </pre>
        </div>
        {request.reason && <p className="muted">说明：{request.reason}</p>}
        <p className="muted">
          该操作可能创建新的 Dataset Version 或产生计算开销，请确认后放行。
        </p>
        {note && <p className="muted">提示：{note}</p>}
        <div className="actions">
          <button className="btn danger" type="button" disabled={busy} onClick={onDeny}>
            拒绝
          </button>
          {/* 允许按钮在提交中禁用但仍可点击感知；失败时弹窗保留，用户可重试或拒绝。 */}
          <button className="btn primary" type="button" disabled={busy} onClick={onAllow}>
            {busy ? "授权中…" : "允许"}
          </button>
        </div>
      </div>
    </div>
  );
}
