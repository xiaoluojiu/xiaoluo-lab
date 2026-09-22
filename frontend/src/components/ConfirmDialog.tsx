import { ReactNode } from "react";

// Prompt 155：确认对话框。
export function ConfirmDialog({
  open,
  title,
  message,
  confirmText = "确认",
  danger,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: ReactNode;
  confirmText?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!open) return null;
  return (
    <div className="dialog-mask" onClick={onCancel}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <div>{message}</div>
        <div className="actions">
          <button className="btn" onClick={onCancel}>
            取消
          </button>
          <button
            className={`btn ${danger ? "danger" : "primary"}`}
            onClick={onConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
