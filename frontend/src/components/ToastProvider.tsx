import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

/**
 * 全局通知中心。
 *
 * 历史问题：项目里没有全局 toast，只有 Settings 页内部一个私有实现，
 * 其余页面靠局部红字或干脆静默；成功/失败反馈口径完全不统一。
 *
 * 约定：
 * - success / info 2.5s 自动消失；
 * - error 常驻，直到用户手动关闭（因为错误信息需要被读完）。
 */

export type ToastKind = "success" | "error" | "info";
export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastApi {
  toast: (message: string, kind?: ToastKind) => void;
  success: (message: string) => void;
  error: (message: string) => void;
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
    (message: string, kind: ToastKind = "success") => {
      const id = ++seq.current;
      setItems((prev) => [...prev.slice(-3), { id, kind, message }]);
      if (kind !== "error") {
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
      success: (message: string) => void toast(message, "success"),
      error: (message: string) => void toast(message, "error"),
      info: (message: string) => void toast(message, "info"),
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
              <span>{item.message}</span>
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
