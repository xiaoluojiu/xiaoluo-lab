// Prompt 181：一键智能分析按钮（触发 smart_analysis 指令）。
export function SmartAnalysisButton({
  disabled,
  busy,
  onClick,
}: {
  disabled?: boolean;
  busy?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className="btn primary"
      disabled={disabled || busy}
      onClick={onClick}
      title="对当前关联数据集做一次智能质量检查与分析"
    >
      {busy ? "分析中..." : "一键智能分析"}
    </button>
  );
}
