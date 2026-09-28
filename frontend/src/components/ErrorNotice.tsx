import { Link } from "react-router-dom";
import type { AnalysisError } from "../lib/analysisError";
import { Icon } from "./icons/Icon";

/**
 * 三层错误提示：发生了什么 / 影响 / 怎么解决，并在能给出明确去处时附带行动按钮。
 *
 * 说明：只做「展示」，不重写任何错误边界或错误码 —— 传入的 AnalysisError
 * 由 lib/analysisError.ts 的 formatError 统一产出，保证全站口径一致。
 */
export function ErrorNotice({
  error,
  title = "操作没能完成",
  onAction,
  compact = false,
}: {
  error: AnalysisError;
  /** 覆盖标题（如「这张图暂时生成不了」）。 */
  title?: string;
  /** 点击行动按钮时的回调（如关闭弹窗 / 清理本地状态）。 */
  onAction?: () => void;
  /** 紧凑模式：用于空间受限的侧栏 / 卡片内。 */
  compact?: boolean;
}) {
  return (
    <div
      className={`analysis-error${compact ? " analysis-error-compact" : ""}`}
      role="alert"
    >
      <span className="analysis-error-icon" aria-hidden="true">
        <Icon name="alert" size={18} />
      </span>
      <div className="analysis-error-body">
        <strong className="analysis-error-title">{title}</strong>
        <span className="analysis-error-message">
          <b className="analysis-error-tag">发生了什么</b>
          {error.what}
        </span>
        <span className="analysis-error-hint">
          <b className="analysis-error-tag">影响</b>
          {error.impact}
        </span>
        <span className="analysis-error-hint">
          <b className="analysis-error-tag">怎么解决</b>
          {error.solution}
        </span>
        {error.actionLink && error.actionLabel && (
          <Link
            className="btn btn-primary analysis-error-action"
            to={error.actionLink}
            onClick={onAction}
          >
            {error.actionLabel}
            <Icon name="arrow-right" size={14} />
          </Link>
        )}
      </div>
    </div>
  );
}
