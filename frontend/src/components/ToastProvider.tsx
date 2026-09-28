import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { AnalysisError } from "../lib/analysisError";
import { Icon } from "./icons/Icon";

/**
 * 全局通知中心。
 *
 * 历史问题：项目里没有全局 toast，只有 Settings 页内部一个私有实现，
 * 其余页面靠局部红字或干脆静默；成功/失败反馈口径完全不统一。
 *
 * 约定：
 * - success / info 2.5s 自动消失；
 * - error 常驻，直到用户手动关闭（因为错误信息需要被读完）；
 * - error 支持两种入参：
 *     1) 纯字符串（兼容旧调用）；
 *     2) 结构化 AnalysisError —— 展示「发生了什么 / 影响 / 怎么解决」三层，
 *        并在带 actionLink 时渲染一个可点击的跳转按钮。
 * - 带 action（跳转链接）的提示一律常驻：2.5s 内点不中一个链接，等于没给入口。
 */

export type ToastKind = "success" | "error" | "info";

/** 提示里的可点击跳转（如「已生成新版本 v3 → 查看新版本」）。 */
export interface ToastAction {
  /** react-router 的 to 值。 */
  link: string;
  label: string;
}

export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
  /** 结构化错误（仅 kind === "error" 时可能存在）；存在时渲染三层信息 + 行动按钮。 */
  error?: AnalysisError;
  /** 通用跳转（任意 kind 均可用）。 */
  action?: ToastAction;
}

interface ToastOptions {
  error?: AnalysisError;
  action?: ToastAction;
}

interface ToastApi {
  toast: (message: string, kind?: ToastKind, options?: ToastOptions) => void;
  /** action 传入时提示不自动消失，留给用户点击的时间。 */
  success: (message: string, action?: ToastAction) => void;
  /** 既可传纯文案，也可传 formatError() 产出的标准错误对象。 */
  error: (input: string | AnalysisError) => void;
  /** 中性提示（如「还有 3 项没达成」）。补上是因为 ToastKind 早就有 info，
   *  但没有便捷方法，调用方只能退回 toast(msg, "info") 或误用 success。 */
  info: (message: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const AUTO_DISMISS_MS = 2500;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const seq = useRef(0);
  const timers = useRef(new Map<number, number>());

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer) {
      window.clearTimeout(timer);
      timers.current.delete(id);
    }
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const toast = useCallback(
    (message: string, kind: ToastKind = "success", options?: ToastOptions) => {
      const id = ++seq.current;
      setItems((prev) => [
        ...prev.slice(-3),
        { id, kind, message, error: options?.error, action: options?.action },
      ]);
      // 带跳转的提示不自动消失（见文件头约定）。
      if (kind !== "error" && !options?.action) {
        timers.current.set(id, window.setTimeout(() => dismiss(id), AUTO_DISMISS_MS));
      }
      return id;
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      toast,
      dismiss,
      success: (message: string, action?: ToastAction) => void toast(message, "success", { action }),
      // 结构化错误：三层信息进 message 字段只是为了兼容单行渲染的调用方；
      // 真正的展示走 item.error。
      error: (input: string | AnalysisError) =>
        typeof input === "string"
          ? void toast(input, "error")
          : void toast(input.what, "error", { error: input }),
      info: (message: string, action?: ToastAction) => void toast(message, "info", { action }),
    }),
    [toast, dismiss],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      {items.length > 0 && (
        <div className="toast-stack" role="status" aria-live="polite">
          {items.map((item) => (
            <div key={item.id} className={`toast-item toast-${item.kind}`}>
              {item.error ? (
                <div className="toast-error-body">
                  <strong className="toast-error-title">
                    <Icon name="alert" size={14} />
                    {item.error.what}
                  </strong>
                  <span className="toast-error-line">
                    <b>影响</b>
                    {item.error.impact}
                  </span>
                  <span className="toast-error-line">
                    <b>怎么解决</b>
                    {item.error.solution}
                  </span>
                  {item.error.actionLink && item.error.actionLabel && (
                    <Link
                      className="btn btn-primary toast-error-action"
                      to={item.error.actionLink}
                      onClick={() => dismiss(item.id)}
                    >
                      {item.error.actionLabel}
                      <Icon name="arrow-right" size={13} />
                    </Link>
                  )}
                </div>
              ) : (
                <div className="toast-plain-body">
                  <span>{item.message}</span>
                  {item.action && (
                    <Link
                      className="btn link toast-action-link"
                      to={item.action.link}
                      onClick={() => dismiss(item.id)}
                    >
                      {item.action.label}
                      <Icon name="arrow-right" size={13} />
                    </Link>
                  )}
                </div>
              )}
              <button type="button" className="toast-close" aria-label="关闭提示" onClick={() => dismiss(item.id)}>×</button>
            </div>
          ))}
        </div>
      )}
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (api) return api;
  // 未包裹 Provider 时降级为 no-op，避免调用方全部要做空判断。
  return {
    toast: () => undefined,
    success: () => undefined,
    error: () => undefined,
    info: () => undefined,
    dismiss: () => undefined,
  };
}
