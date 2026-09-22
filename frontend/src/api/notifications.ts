/** 通知中心 API。 */
import { client, unwrap } from "./client";

export interface AppNotification {
  id: string;
  type: "training" | "report" | "workflow" | "permission" | "system";
  title: string;
  body: string;
  created_at: number;
  read: boolean;
  link: string | null;
}

export interface NotificationList {
  items: AppNotification[];
  unread: number;
}

export interface NotificationPrefs {
  email: string;
  notify_training: boolean;
  notify_report: boolean;
  notify_workflow: boolean;
  notify_permission: boolean;
  notify_system: boolean;
}

export function listNotifications() {
  return unwrap<NotificationList>(client.get("/notifications"));
}

export function markNotificationRead(notificationId: string) {
  return unwrap<{ unread: number }>(client.post("/notifications/read", { notification_id: notificationId }));
}

export function markAllNotificationsRead() {
  return unwrap<{ unread: number }>(client.post("/notifications/read-all"));
}

export function getNotificationPrefs() {
  return unwrap<NotificationPrefs>(client.get("/notifications/prefs"));
}

export function updateNotificationPrefs(body: Partial<NotificationPrefs>) {
  return unwrap<NotificationPrefs>(client.put("/notifications/prefs", body));
}
