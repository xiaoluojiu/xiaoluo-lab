import { memo } from "react";
import { useAiLab, toolDisplayName } from "../../store/aiLab";
import type { ToolCall } from "../../types/agent";

const STATUS_LABEL: Record<ToolCall["status"], { text: string; cls: string }> = {
  pending: { text: "待执行", cls: "warning" },
  ok: { text: "成功", cls: "success" },
  failed: { text: "失败", cls: "failed" },
  needs_confirmation: { text: "待确认", cls: "warning" },
  denied: { text: "已拒绝", cls: "failed" },
};

function toLines(value: unknown): string[] {
  if (value == null) return [];
  if (Array.isArray(value)) return value.map((v) => (typeof v === "string" ? v : JSON.stringify(v)));
  if (typeof value === "string") return [value];
  return [JSON.stringify(value)];
}

export const ToolCallCard = memo(function ToolCallCard({ call }: { call: ToolCall }) {
  const tools = useAiLab((s) => s.tools);
  const meta = STATUS_LABEL[call.status] ?? { text: call.status, cls: "warning" };
  const summary = call.result?.summary != null ? String(call.result.summary) : null;
  const displayName = toolDisplayName(call.tool, tools);
  const warnings = toLines(call.result?.warnings);
  const errors = toLines(call.result?.errors);
  const metadata = call.result?.metadata ?? {};
  const hasData = call.result?.data != null;
  const hasMetadata = Object.keys(metadata).length > 0;
  const hasDetail = hasData || warnings.length > 0 || errors.length > 0 || hasMetadata;

  return (
    <details className="ai-tool-call">
      <summary>
        <span className="ai-tool-call-main">
          <strong>{displayName}</strong>
          <span className="ai-tool-call-name">{call.tool}</span>
        </span>
        <span className={`badge ${meta.cls}`}>{meta.text}</span>
      </summary>
      <div className="ai-tool-call-body">
        {summary && <div className="ai-tool-result"><span>结果摘要</span><strong>{summary}</strong></div>}
        {call.error && <div className="ai-tool-error">{call.error}</div>}
        {warnings.length > 0 && (
          <div className="ai-tool-warn">
            <span>警告</span>
            <ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </div>
        )}
        {errors.length > 0 && (
          <div className="ai-tool-error">
            <span>错误明细</span>
            <ul>{errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
          </div>
        )}
        <div className="ai-tool-meta">步骤 {call.step_index + 1}{call.elapsed_ms > 0 ? ` · ${call.elapsed_ms} ms` : ""}{call.attempt > 1 ? ` · 第 ${call.attempt} 次尝试` : ""}</div>
        <details className="ai-tool-advanced">
          <summary>查看参数</summary>
          <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
        </details>
        {hasDetail && (
          <details className="ai-tool-advanced">
            <summary>查看完整结果</summary>
            {hasData && <pre>{JSON.stringify(call.result?.data, null, 2)}</pre>}
            {hasMetadata && <pre>{JSON.stringify(metadata, null, 2)}</pre>}
          </details>
        )}
      </div>
    </details>
  );
});
