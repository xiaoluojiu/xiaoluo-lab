import type { WorkflowNode } from "../../types/workflow";

const STATUS_COLOR: Record<string, string> = {
  pending: "var(--border)",
  running: "var(--c-running)",
  success: "var(--success)",
  failed: "var(--danger)",
  skipped: "var(--c-skipped)",
};

// Prompt 183：单个工作流节点（状态着色，第一版简洁卡片）。
export function WorkflowNode({
  node,
  status = "pending",
  selected,
  onClick,
}: {
  node: WorkflowNode;
  status?: string;
  selected?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      className="wf-node clickable"
      onClick={onClick}
      style={{
        border: `2px solid ${selected ? "var(--primary)" : STATUS_COLOR[status] ?? "var(--border)"}`,
        borderRadius: 8,
        padding: "8px 12px",
        background: "var(--surface)",
        minWidth: 160,
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 13 }}>{node.id}</div>
      <div className="muted" style={{ fontSize: 11 }}>{node.type}</div>
      {status !== "pending" && (
        <div style={{ fontSize: 11, marginTop: 2, color: STATUS_COLOR[status] }}>{status}</div>
      )}
    </div>
  );
}
