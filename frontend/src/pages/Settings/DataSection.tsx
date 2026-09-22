import { useEffect, useMemo, useState } from "react";
import { formatBytes, getStorageSummary, type StorageSummary } from "../../api/settings";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { InfoHint } from "../../components/InfoHint";
import { applyUi, readUi, type UiSettings } from "../../lib/uiSettings";
import { clearLocalState, LOCAL_STATE_ENTRIES, presentStateEntries, type LocalStateId } from "../../lib/localState";

/**
 * 数据与缓存分区：服务器 Storage 概览 + 浏览器本地数据清理。
 *
 * 历史问题（严重）：旧的「清理浏览器缓存」删的是 `xllab.cache.*` 前缀，
 * 而全仓库从来没有写过这个前缀——用户点完确认框实际什么都没删，
 * 是「看起来存在但完全无效」的功能。而真正该被清理的 `xllab.recent.datasets`
 * 与含密钥的 `xllab.llm` 反而不在范围内。
 *
 * 现在改为按清单勾选（lib/localState 是唯一事实源），并在确认框里逐条列出后果。
 *
 * 外壳约定：返回 Fragment，`.settings-stack` 由页面层统一提供（避免双层 gap 叠加）。
 */
export function DataSection({
  notify,
  onUiReset,
}: {
  notify: (text: string, kind?: "success" | "error") => void;
  /** 外观设置被清空后通知父级同步 UI state（否则内存态仍停留在旧主题）。 */
  onUiReset: (next: UiSettings) => void;
}) {
  const [storage, setStorage] = useState<StorageSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [selected, setSelected] = useState<LocalStateId[]>(
    () => LOCAL_STATE_ENTRIES.filter((entry) => entry.defaultChecked).map((entry) => entry.id),
  );

  const present = useMemo(() => presentStateEntries(), []);
  const presentIds = useMemo(() => present.map((entry) => entry.id), [present]);
  /** 只列出「当前确实存在」的项，避免告诉用户会删掉一个本来就没有的东西。 */
  const effectiveSelection = selected.filter((id) => presentIds.includes(id));

  async function refresh() {
    setLoading(true);
    try {
      setStorage(await getStorageSummary());
    } catch {
      setStorage(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void refresh(); }, []);

  function toggleEntry(id: LocalStateId) {
    setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
  }

  function doClear() {
    const removed = clearLocalState(effectiveSelection);
    setConfirmOpen(false);
    // 外观设置被清掉时，需要把 UI 同步回默认值，否则页面还停留在旧主题（内存态仍生效）。
    if (effectiveSelection.includes("ui")) {
      const next = readUi();
      applyUi(next);
      onUiReset(next);
    }
    notify(removed > 0 ? `已清理 ${removed} 项浏览器本地数据` : "没有需要清理的数据");
  }

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>服务器 Storage</h3>
        </div>
        <button className="btn" disabled={loading} onClick={() => void refresh()}>刷新</button>
      </div>
      {loading && <p className="settings-empty">读取 Storage...</p>}
      {!loading && storage && <>
        <div className="settings-stat-grid">
          <div><small>对象数量</small><strong>{storage.total_files}</strong></div>
          <div><small>实际占用</small><strong>{formatBytes(storage.total_bytes)}</strong></div>
        </div>
        <div className="settings-group-list">
          {Object.entries(storage.groups)
            .sort(([, a], [, b]) => b.bytes - a.bytes)
            .map(([group, item]) => (
              <div key={group} className="settings-data-row">
                <span><strong>{group}</strong><small>{item.files} 个对象</small></span>
                <span>{formatBytes(item.bytes)}</span>
              </div>
            ))}
        </div>
      </>}
      {!loading && !storage && <p className="settings-note settings-note-warn">无法读取 Storage 状态</p>}
    </section>

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            浏览器本地数据
            <InfoHint label="浏览器本地数据说明">
              这些数据只存在于当前浏览器，不会同步到服务器；若清除「外观设置」，页面会立即回到默认主题。
            </InfoHint>
          </h3>
        </div>
      </div>

      {present.length === 0
        ? <p className="settings-empty">当前浏览器没有保存任何本地数据。</p>
        : <div className="settings-clear-list">
          {LOCAL_STATE_ENTRIES.map((entry) => {
            const exists = presentIds.includes(entry.id);
            return (
              <label key={entry.id} className={`settings-clear-item${exists ? "" : " is-absent"}`}>
                <input
                  type="checkbox"
                  checked={effectiveSelection.includes(entry.id)}
                  disabled={!exists}
                  onChange={() => toggleEntry(entry.id)}
                />
                <span className="settings-clear-copy">
                  <strong>
                    {entry.label}
                    {entry.sensitive && <span className="badge failed">含凭据</span>}
                    {!exists && <span className="badge">无数据</span>}
                  </strong>
                  <small>{entry.consequence}</small>
                </span>
              </label>
            );
          })}
        </div>}

      <div className="settings-actions">
        <button className="btn" type="button" disabled={!effectiveSelection.length} onClick={() => setConfirmOpen(true)}>
          清理选中的本地数据
        </button>
      </div>
    </section>

    <ConfirmDialog
      open={confirmOpen}
      title="清理选中的浏览器数据？"
      message={<>
        <p style={{ marginTop: 0 }}>以下数据将被永久删除（不影响服务器数据）：</p>
        <ul className="settings-confirm-list">
          {LOCAL_STATE_ENTRIES.filter((entry) => effectiveSelection.includes(entry.id)).map((entry) => (
            <li key={entry.id}><strong>{entry.label}</strong>：{entry.consequence}</li>
          ))}
        </ul>
      </>}
      confirmText="清理"
      danger
      onConfirm={doClear}
      onCancel={() => setConfirmOpen(false)}
    />
  </>;
}
