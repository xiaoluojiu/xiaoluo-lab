import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  markAllNotificationsRead,
  markNotificationRead,
  type AppNotification,
} from "../api/notifications";
import { subscribeNotifications } from "../lib/notificationFeed";

const TYPE_META: Record<AppNotification["type"], { label: string; tone: string }> = {
  training: { label: "训练", tone: "is-training" },
  report: { label: "报告", tone: "is-report" },
  workflow: { label: "工作流", tone: "is-workflow" },
  permission: { label: "授权", tone: "is-permission" },
  system: { label: "系统", tone: "is-system" },
};

function fmtTime(ts: number): string {
  const diff = Date.now() - ts * 1000;
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`;
  return new Date(ts * 1000).toLocaleDateString("zh-CN");
}

/**
 * 顶栏通知铃铛：未读角标 + 下拉通知面板。
 *
 * 数据来自后端进程内通知中心（GET /notifications）。取数策略已由
 * 「固定 15s 轮询」改为 **SSE 推送 + 退避轮询兜底**，实现见
 * `lib/notificationFeed.ts`：后台标签页不发请求，连续失败按
 * 15s/30s/60s/120s 退避而不是原地重试。
 */
export function NotificationBell() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<AppNotification[]>([]);
  const [unread, setUnread] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const feed = subscribeNotifications({
      onSnapshot: (data) => {
        setItems(data.items);
        setUnread(data.unread);
      },
    });
    return () => feed.close();
  }, []);

  // 点击面板外关闭。
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  function openItem(n: AppNotification) {
    if (!n.read) {
      void markNotificationRead(n.id).then((r) => setUnread(r.unread));
      setItems((prev) => prev.map((x) => (x.id === n.id ? { ...x, read: true } : x)));
    }
    if (n.link) {
      setOpen(false);
      navigate(n.link);
    }
  }

  function readAll() {
    void markAllNotificationsRead().then(() => {
      setUnread(0);
      setItems((prev) => prev.map((x) => ({ ...x, read: true })));
    });
  }

  return (
    <div className="notif-wrap" ref={wrapRef}>
      <button
        type="button"
        className={`icon-btn notif-btn${open ? " is-open" : ""}`}
        aria-label="通知"
        title="通知"
        onClick={() => setOpen((v) => !v)}
      >
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.7 21a2 2 0 0 1-3.4 0" />
        </svg>
        {unread > 0 && <span className="notif-badge" aria-hidden="true">{unread > 99 ? "99+" : unread}</span>}
      </button>

      {open && (
        <div className="notif-pop" role="menu" aria-label="通知列表">
          <div className="notif-pop-head">
            <strong>通知</strong>
            {unread > 0 && (
              <button type="button" className="notif-read-all" onClick={readAll}>全部已读</button>
            )}
          </div>
          <div className="notif-pop-list">
            {items.length === 0 ? (
              <div className="notif-empty">暂无通知</div>
            ) : (
              items.map((n) => (
                <button
                  key={n.id}
                  type="button"
                  className={`notif-item${n.read ? "" : " is-unread"}`}
                  onClick={() => openItem(n)}
                >
                  <span className={`notif-item-dot ${TYPE_META[n.type]?.tone ?? ""}`} aria-hidden="true" />
                  <span className="notif-item-main">
                    <span className="notif-item-title">
                      {!n.read && <i className="notif-unread-mark" aria-hidden="true" />}
                      {n.title}
                      <em className="notif-item-type">{TYPE_META[n.type]?.label ?? n.type}</em>
                    </span>
                    <span className="notif-item-body">{n.body}</span>
                    <span className="notif-item-time">{fmtTime(n.created_at)}</span>
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
