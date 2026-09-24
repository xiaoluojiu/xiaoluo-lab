/**
 * 会话侧栏：纯展示 + 纯 UI 状态（「⋯」菜单的展开项属于 UI，留在组件内，
 * 不再上升到页面层）。所有数据变更都通过回调交给 useAiSession。
 */
import { useState } from "react";
import type { AgentSession } from "../../types/agent";
import { relativeTime, sessionTitle } from "./hooks/useAiSession";

export function SessionSidebar({
  sessions,
  activeId,
  listOpen,
  onToggleList,
  deletingId,
  bulkBusy,
  selectedIds,
  allIds,
  allSelected,
  someSelected,
  onToggleSelect,
  onToggleSelectAll,
  onInvertSelect,
  onNew,
  onOpen,
  onToggleArchive,
  onArchiveSelected,
  onRequestDelete,
  onRequestBulk,
}: {
  sessions: AgentSession[];
  activeId: string | null;
  listOpen: boolean;
  onToggleList: () => void;
  deletingId: string | null;
  bulkBusy: boolean;
  selectedIds: string[];
  allIds: string[];
  allSelected: boolean;
  someSelected: boolean;
  onToggleSelect: (id: string) => void;
  onToggleSelectAll: () => void;
  onInvertSelect: () => void;
  onNew: () => void;
  onOpen: (session: AgentSession) => void;
  onToggleArchive: (session: AgentSession) => void;
  onArchiveSelected: (ids: string[]) => void;
  onRequestDelete: (session: AgentSession) => void;
  onRequestBulk: (ids: string[], all: boolean) => void;
}) {
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);

  return (
    <aside className={`ai-sidebar${listOpen ? "" : " sessions-collapsed"}`}>
      <section className="ai-sidebar-section ai-experiment-nav">
        <div className="ai-section-heading">
          <div><div className="ai-section-kicker">会话</div><h3>分析会话</h3></div>
          <div className="ai-heading-actions">
            <button className="btn primary" type="button" onClick={onNew}>+ 新建</button>
            <button className="btn ai-sidebar-fold" type="button" aria-expanded={listOpen} title={listOpen ? "收起列表" : "展开列表"} onClick={onToggleList}>{listOpen ? "收起" : "展开"}</button>
          </div>
        </div>
        {/* 批量操作条：全选 / 反选 / 批量归档 / 批量删除 / 清空 */}
        <div className="ai-session-toolbar">
          <label className="ai-check-line" title="全选 / 取消全选">
            <input
              type="checkbox"
              checked={allSelected}
              ref={(el) => { if (el) el.indeterminate = someSelected; }}
              disabled={!allIds.length}
              onChange={onToggleSelectAll}
            />
            全选
          </label>
          <span className="ai-toolbar-count muted">{selectedIds.length ? `已选 ${selectedIds.length}/${allIds.length}` : `${allIds.length} 个会话`}</span>
          <div className="ai-toolbar-actions">
            <button className="btn btn-sm" type="button" disabled={!selectedIds.length} onClick={onInvertSelect}>反选</button>
            <button className="btn btn-sm" type="button" disabled={!selectedIds.length} onClick={() => onArchiveSelected([...selectedIds])}>归档</button>
            <button className="btn btn-sm danger" type="button" disabled={!selectedIds.length || bulkBusy} onClick={() => onRequestBulk([...selectedIds], false)}>删除</button>
            <button className="btn btn-sm danger" type="button" disabled={!allIds.length || bulkBusy} onClick={() => onRequestBulk([...allIds], true)} title="删除全部会话及其运行记录（运行中的会话会跳过）">清空</button>
          </div>
        </div>
        <div className="ai-session-list">
          {!sessions.length && <div className="muted">还没有历史会话。</div>}
          {sessions.map((s) => (
            <div key={s.id} className={`ai-session-item ${activeId === s.id ? "active" : ""} ${s.archived ? "archived" : ""} ${selectedIds.includes(s.id) ? "selected" : ""}`}>
              <input
                type="checkbox"
                className="ai-session-check"
                checked={selectedIds.includes(s.id)}
                onChange={() => onToggleSelect(s.id)}
                aria-label={`选择会话 ${sessionTitle(s)}`}
              />
              <button type="button" className="ai-session-open" onClick={() => onOpen(s)} title={sessionTitle(s)}>
                <div className="ai-session-item-title">
                  <span className="ai-session-item-name">{sessionTitle(s)}</span>
                  {s.archived && <em className="ai-session-archived-tag">已归档</em>}
                </div>
                <div className="ai-session-item-meta">{relativeTime(s.created_at)} · {s.history.length ? `${s.history.length} 条消息` : "新实验"}{s.run_ids?.length ? ` · ${s.run_ids.length} 次运行` : ""}</div>
              </button>
              <button type="button" className="ai-session-more" aria-label="会话操作" aria-expanded={menuSessionId === s.id} onClick={() => setMenuSessionId((id) => (id === s.id ? null : s.id))}>⋯</button>
              {menuSessionId === s.id && (
                <div className="ai-session-menu" role="menu">
                  <button type="button" role="menuitem" onClick={() => { setMenuSessionId(null); onToggleArchive(s); }}>{s.archived ? "取消归档" : "归档"}</button>
                  <button type="button" role="menuitem" className="danger" disabled={deletingId === s.id} onClick={() => { setMenuSessionId(null); onRequestDelete(s); }}>{deletingId === s.id ? "删除中…" : "删除会话"}</button>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
      {menuSessionId && <div className="ai-menu-backdrop" onClick={() => setMenuSessionId(null)} aria-hidden="true" />}
    </aside>
  );
}
