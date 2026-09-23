/** 通知数据源：SSE 优先，轮询兜底。
 *
 * 为什么不再固定 15s 轮询
 * ----------------------
 * 原始实现是「每 15 秒拉一次完整列表」，无论有没有新通知。每个打开的浏览器标签
 * 都在后台周期性序列化/反序列化几十上百条通知，而这些请求里绝大多数返回的是
 * 一模一样的字节。
 *
 * 现在的分层：
 * 1. 优先建立 SSE（`GET /notifications/stream`）。服务端只在**版本号变化**时推
 *    一个几十字节的事件，平时只有每 20 秒一个心跳注释行；
 * 2. 连不上 SSE（退化部署 / 代理不支持长连接）时退回轮询，但带**指数退避**：
 *    连续失败不原地重试，间隔 15s → 30s → 60s → 上限 120s，成功一次立即复位；
 * 3. 标签页不可见时完全不发请求，回到前台立刻补一次。
 *
 * 外部只需提供 `onSnapshot`，两条通道共用同一个回调，UI 不需要知道
 * 现在走的是哪条路。
 */
import { listNotifications, type NotificationList } from "../api/notifications";

/** 退避序列（毫秒）：失败越多等越久，但永远不无限期停摆。 */
const BACKOFF_STEPS = [15_000, 30_000, 60_000, 120_000];

/** SSE 地址：跟随 axios 的 baseURL 规则（同源代理 / VITE_API_BASE_URL）。 */
function sseBaseUrl(): string {
  if (import.meta.env.DEV) return "/api/v1";
  const raw = String(import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");
  if (!raw) return "/api/v1";
  return /\/api\/v1$/i.test(raw) ? raw : `${raw}/api/v1`;
}

export interface NotificationFeedOptions {
  /** 收到新快照（来自 SSE 首帧或轮询结果）时调用。 */
  onSnapshot: (data: NotificationList) => void;
}

export interface NotificationFeed {
  /** 停止所有通道并清理定时器。 */
  close: () => void;
  /** 立即拉一次（用户点开面板 / 回到前台时用）。 */
  refresh: () => void;
}

export function subscribeNotifications(options: NotificationFeedOptions): NotificationFeed {
  const { onSnapshot } = options;

  let closed = false;
  let failures = 0;
  let pollTimer: number | null = null;
  let source: EventSource | null = null;
  let polling = false;

  const clearPoll = () => {
    if (pollTimer !== null) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
  };

  const pull = async () => {
    if (closed) return;
    try {
      onSnapshot(await listNotifications());
      failures = 0; // 成功一次就复位退避，避免一次抖动被拉黑两分钟
    } catch {
      failures += 1;
    }
    // 只有轮询模式下才排下一次 —— 否则 SSE 正常时也会被这条定时器继续
    // 周期拉列表，等于把刚删掉的轮询又加回来了。
    if (polling) scheduleNext();
  };

  const scheduleNext = () => {
    clearPoll();
    if (closed) return;
    const step = BACKOFF_STEPS[Math.min(failures, BACKOFF_STEPS.length - 1)];
    pollTimer = window.setTimeout(() => void pull(), step);
  };

  const startPolling = () => {
    if (polling || closed) return;
    polling = true;
    void pull();
  };

  // ---- 主通道：SSE ----------------------------------------------------
  const openStream = () => {
    if (closed || typeof EventSource === "undefined") {
      startPolling();
      return;
    }
    try {
      source = new EventSource(`${sseBaseUrl()}/notifications/stream`);
    } catch {
      startPolling();
      return;
    }

    const apply = (raw: string) => {
      try {
        onSnapshot(JSON.parse(raw) as NotificationList);
      } catch {
        // 单帧解析失败不能让整条流失能：忽略这一帧即可。
      }
    };

    source.addEventListener("snapshot", (event) => {
      failures = 0;
      apply((event as MessageEvent<string>).data);
    });

    // 只有版本号变了才触发；此处再拉一次列表拿正文。
    source.addEventListener("changed", () => {
      void pull();
    });

    source.onerror = () => {
      // EventSource 自己会重连，但如果它已经关闭（readyState === CLOSED），
      // 说明这条环境根本不支持长连接 —— 关掉它并切到退避轮询，而不是两头都跑。
      if (source && source.readyState === EventSource.CLOSED) {
        source.close();
        source = null;
        failures += 1;
        startPolling();
      }
    };
  };

  // ---- 可见性：后台标签页完全静默 -------------------------------------
  const onVisibility = () => {
    if (document.hidden) {
      clearPoll();
      return;
    }
    clearPoll();
    if (polling) {
      failures = 0; // 回到前台先立刻重试，不必等完完整的退避
      void pull();
    } else {
      void pull();
    }
  };
  document.addEventListener("visibilitychange", onVisibility);

  openStream();

  return {
    close: () => {
      closed = true;
      clearPoll();
      document.removeEventListener("visibilitychange", onVisibility);
      source?.close();
      source = null;
    },
    refresh: () => {
      failures = 0;
      void pull();
    },
  };
}
