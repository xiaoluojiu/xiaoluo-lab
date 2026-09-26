/**
 * 会话域的状态与流程：会话列表、当前会话、数据集上下文、消息、归档 / 删除 / 批量选择。
 *
 * 只处理「会话」这一层；运行态在 useAgentRun。两者通过 `switchToken` 单向耦合：
 * 本 Hook 在用户显式切换 / 新建 / 删除回退时 +1，run 侧据此重建面板。
 *
 * 收敛点：
 * - 选中态一律以「当前列表里的 id 集合」为准，并裁掉孤儿 id。
 *   历史缺陷：原来用 `selectedIds.length === allIds.length` 判全选，但 selectedIds 里
 *   可能残留已不在列表中的 id（归档 / 删除后），两个长度比较错位 —— 勾选框 checked
 *   与「已选 N/M」文案不一致，点「全选」还可能一次性清空。
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { archiveSession, createSession, deleteSession, listSessions } from "../../../api/agent";
import { useAiLab } from "../../../store/aiLab";
import type { AgentSession, ChatMessage } from "../../../types/agent";

export function sessionTitle(s: AgentSession): string {
  if (s.title && s.title !== "新会话" && s.title !== "新数据分析会话") return s.title;
  const firstUser = s.history.find((h) => h.role === "user")?.content ?? "";
  if (firstUser) return `${firstUser.slice(0, 30)}${firstUser.length > 30 ? "…" : ""}`;
  return "未命名实验";
}

export function relativeTime(ts: number): string {
  const sec = Math.round(Date.now() / 1000 - ts);
  if (sec < 60) return "刚刚";
  if (sec < 3600) return `${Math.floor(sec / 60)} 分钟前`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} 小时前`;
  return `${Math.floor(sec / 86400)} 天前`;
}

/** 与会话列表保持同一排序口径：未归档在前，各自按创建时间倒序。 */
function reorderSessions(items: AgentSession[]): AgentSession[] {
  return [...items].sort((a, b) => (a.archived ? 1 : 0) - (b.archived ? 1 : 0) || b.created_at - a.created_at);
}

function errText(e: unknown, fallback: string): string {
  return e instanceof Error ? e.message : fallback;
}

export interface UseAiSessionOptions {
  onError: (message: string | null) => void;
  onNotice: (message: string | null) => void;
}

export function useAiSession({ onError, onNotice }: UseAiSessionOptions) {
  const sessionId = useAiLab((s) => s.sessionId);
  const setSession = useAiLab((s) => s.setSession);
  const setDatasetIds = useAiLab((s) => s.setDatasetIds);
  const loadTools = useAiLab((s) => s.loadTools);

  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [selectedDatasets, setSelectedDatasets] = useState<number[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  /** 会话切换计数。只有显式切换 / 新建 / 删除回退才 +1（send() 自动建会话不算）。 */
  const [switchToken, setSwitchToken] = useState(0);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === sessionId) ?? null,
    [sessions, sessionId],
  );
  const lastRunId = activeSession?.run_ids?.length ? activeSession.run_ids[activeSession.run_ids.length - 1] : null;

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  /**
   * 数据集上下文只允许「有内容」时向外同步。
   *
   * 历史缺陷链：挂载时 `selectedDatasets` 是空数组，而「从 Dataset Workspace 带 ?dataset=7 跳入
   * 自动选中」的副作用藏在 DatasetSelector 内部、要等它自己拉完数据集列表才触发。
   * 于是这个 effect 会先用**空数组**把 store 里的 datasetIds 覆盖掉；
   * 若 send() 发生在 DatasetSelector 完成之前，createSession / sendMessage 取到的就是空上下文
   * —— 界面上显示「已关联 1 个数据集」，Agent 实际按「未关联数据集」执行。
   * 现在：空数组不写入 store，由 DatasetSelector 真正选中后再同步。
   */
  useEffect(() => {
    if (selectedDatasets.length) setDatasetIds(selectedDatasets);
  }, [selectedDatasets, setDatasetIds]);

  /** 切到某个会话：同步上下文与历史，并通知 run 侧重建面板。 */
  const activate = useCallback(
    (s: AgentSession) => {
      setSession(s.id);
      setSelectedDatasets(s.dataset_ids ?? []);
      setMessages(s.history.map((h) => ({ role: h.role, content: h.content })));
      setSelectedIds([]);
      setSwitchToken((v) => v + 1);
    },
    [setSession],
  );

  /** 建会话并切过去。新建按钮与「发第一条消息时自动建会话」共用这一条路径。 */
  const createAndActivate = useCallback(
    async (title: string): Promise<AgentSession | null> => {
      try {
        const s = await createSession(selectedDatasets, title);
        setSession(s.id);
        setDatasetIds(selectedDatasets);
        setSessions((prev) => [s, ...prev.filter((item) => item.id !== s.id)]);
        return s;
      } catch (e) {
        onError(errText(e, "创建会话失败"));
        return null;
      }
    },
    [selectedDatasets, setSession, setDatasetIds, onError],
  );

  /** send() 在没有会话时调用：只建会话，不动运行态（避免打断刚收到的事件）。 */
  const ensureSession = useCallback(
    async (title: string): Promise<string | null> => (await createAndActivate(title))?.id ?? null,
    [createAndActivate],
  );

  /**
   * 运行终态后回拉会话列表：刷新侧栏标题 / 消息数 / run_ids。
   *
   * 历史缺陷：`sessions` 只在挂载时拉一次，运行结束后后端的 title/history/run_ids
   * 已更新，但侧栏仍是旧快照——标题不变、消息数不涨、run_ids 指向旧 run，
   * 切走再切回会「丢失最后一次运行」。这里只在终态后回拉一次（节流在 run 侧保证）。
   */
  const refreshActiveSession = useCallback(async () => {
    try {
      const items = await listSessions();
      setSessions(items);
    } catch {
      /* 回拉失败不打断主流程：下次切换/挂载仍会重新拉取 */
    }
  }, []);

  const newSession = useCallback(async () => {
    onError(null);
    const s = await createAndActivate("新数据分析会话");
    if (!s) return;
    setMessages([]);
    setSelectedIds([]);
    setSwitchToken((v) => v + 1);
  }, [createAndActivate, onError]);

  useEffect(() => {
    void loadTools();
    void listSessions()
      .then((items) => {
        setSessions(items);
        const saved = items.find((s) => s.id === sessionId) ?? items[0];
        if (saved) activate(saved);
      })
      .catch(() => setSessions([]));
    // 只在挂载时执行一次：后续切换由用户显式触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** 归档 / 取消归档。归档只是把会话沉到列表底部，历史与运行记录都保留。 */
  const toggleArchive = useCallback(
    async (s: AgentSession) => {
      const next = !s.archived;
      setSessions((prev) => reorderSessions(prev.map((item) => (item.id === s.id ? { ...item, archived: next } : item))));
      onNotice(next ? "会话已归档，已移至列表底部（可再次点击取消归档）。" : "会话已取消归档。");
      try {
        const saved = await archiveSession(s.id, next);
        setSessions((prev) => reorderSessions(prev.map((item) => (item.id === s.id ? { ...item, ...saved } : item))));
      } catch (e) {
        setSessions((prev) => reorderSessions(prev.map((item) => (item.id === s.id ? { ...item, archived: !next } : item))));
        onError(errText(e, "归档失败"));
      }
    },
    [onError, onNotice],
  );

  /** 批量归档：逐个调用，失败项保留在列表中并汇总提示。 */
  const archiveSelected = useCallback(
    async (ids: string[]) => {
      if (!ids.length) return;
      const targets = sessions.filter((s) => ids.includes(s.id) && !s.archived);
      if (!targets.length) {
        onNotice("选中的会话都已归档。");
        return;
      }
      let ok = 0;
      const failed: string[] = [];
      for (const s of targets) {
        try {
          const saved = await archiveSession(s.id, true);
          ok += 1;
          setSessions((prev) => reorderSessions(prev.map((item) => (item.id === s.id ? { ...item, ...saved } : item))));
        } catch {
          failed.push(sessionTitle(s));
        }
      }
      setSelectedIds([]);
      onNotice(
        failed.length
          ? `已归档 ${ok} 个会话；${failed.length} 个失败：${failed.join("、")}`
          : `已归档 ${ok} 个会话。`,
      );
    },
    [sessions, onNotice],
  );

  /**
   * 删除会话（单条 / 批量 / 清空共用）。
   * 后端只提供单条删除且活动会话会返回 409，这里逐个删除并汇总失败项，
   * 不因为其中一条失败就中断整个批量操作。
   */
  const removeSessions = useCallback(
    async (ids: string[]) => {
      const targets = sessions.filter((s) => ids.includes(s.id));
      if (!targets.length) return;
      setBulkDeleting(true);
      onError(null);
      let ok = 0;
      const failed: string[] = [];
      const removed: string[] = [];
      for (const s of targets) {
        setDeletingSessionId(s.id);
        try {
          await deleteSession(s.id);
          ok += 1;
          removed.push(s.id);
          setSessions((prev) => prev.filter((item) => item.id !== s.id));
        } catch (e) {
          failed.push(`${sessionTitle(s)}（${errText(e, "删除失败")}）`);
        } finally {
          setDeletingSessionId(null);
        }
      }
      // state 更新保持函数式（不丢并发新增），但**剩余列表单独算一份**，
      // 绝不在 updater 里给外部变量赋值：updater 可能被 React 延迟 / 重复执行，
      // 靠它的副作用写 `remaining` 既拿不到稳定值，也让这段逻辑无法单测。
      // 历史缺陷链：这里曾写成基于函数开头 `sessions` 闭包快照的 `setSessions(...)`，
      // 会把删除循环里刚移除的条目又加回来（批里有任意一条失败就触发）。
      setSessions((prev) => prev.filter((item) => !removed.includes(item.id)));
      const remainingSessions = sessions.filter((item) => !removed.includes(item.id));
      const currentRemoved = sessionId ? removed.includes(sessionId) : false;
      if (currentRemoved) {
        setSession(null);
        setMessages([]);
        setSelectedIds([]);
        setSwitchToken((v) => v + 1);
      }
      setSelectedIds((prev) => prev.filter((id) => !removed.includes(id)));
      setBulkDeleting(false);
      if (failed.length) onError(`成功删除 ${ok} 个会话；${failed.length} 个未删除：${failed.join("；")}`);
      else onNotice(ok > 1 ? `已删除 ${ok} 个会话及其运行记录。` : `已删除会话「${sessionTitle(targets[0])}」。`);
      // 自动切换到剩余会话放在状态更新之后，避免与上面的 setSessions 竞态。
      if (currentRemoved && remainingSessions.length > 0) {
        activate(remainingSessions[0]);
      }
    },
    [sessions, sessionId, setSession, activate, onError, onNotice],
  );

  const allIds = useMemo(() => sessions.map((s) => s.id), [sessions]);
  /** 只保留「当前列表里还存在」的选中项，避免计数与勾选框错位。 */
  const selectedVisible = useMemo(() => selectedIds.filter((id) => allIds.includes(id)), [selectedIds, allIds]);
  const allSelected = allIds.length > 0 && selectedVisible.length === allIds.length;
  const someSelected = selectedVisible.length > 0 && !allSelected;

  useEffect(() => {
    setSelectedIds((prev) => {
      const next = prev.filter((id) => allIds.includes(id));
      return next.length === prev.length ? prev : next;
    });
  }, [allIds]);

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  }, []);
  const toggleSelectAll = useCallback(() => {
    setSelectedIds(() => (allSelected ? [] : [...allIds]));
  }, [allSelected, allIds]);
  const invertSelect = useCallback(() => {
    setSelectedIds(() => allIds.filter((id) => !selectedVisible.includes(id)));
  }, [allIds, selectedVisible]);

  return {
    sessions,
    sessionId,
    activeSession,
    lastRunId,
    switchToken,
    selectedDatasets,
    setSelectedDatasets,
    messages,
    appendMessage,
    deletingSessionId,
    bulkDeleting,
    ensureSession,
    refreshActiveSession,
    newSession,
    activate,
    toggleArchive,
    archiveSelected,
    removeSessions,
    selectedIds,
    selectedVisible,
    allIds,
    allSelected,
    someSelected,
    toggleSelect,
    toggleSelectAll,
    invertSelect,
  };
}
