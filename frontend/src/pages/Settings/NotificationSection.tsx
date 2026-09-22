import { useCallback, useEffect, useState } from "react";
import { getNotificationPrefs, updateNotificationPrefs, type NotificationPrefs } from "../../api/notifications";
import { InfoHint } from "../../components/InfoHint";

/**
 * 开关行：只留标题。原标题 + 一句解释里，解释基本都是把标题换个说法
 * （「训练完成 / 失败」→「模型训练结束后提醒」），因此把信息压回标题本身。
 */
const TOGGLE_ROWS: Array<{ key: keyof NotificationPrefs; title: string }> = [
  { key: "notify_training", title: "训练完成或失败" },
  { key: "notify_report", title: "报告生成完成" },
  { key: "notify_workflow", title: "工作流运行结束（含失败）" },
  { key: "notify_system", title: "系统与配置异常" },
  { key: "notify_permission", title: "Agent 请求高危操作授权" },
];

/**
 * 通知偏好分区：邮箱 + 各类型开关。
 *
 * 偏好只存后端进程内（重启回到默认），与 Agent 策略一致；
 * 邮箱为站外推送的占位字段——当前版本不会真的发送邮件，写进论文「局限与展望」。
 */
export function NotificationSection({
  notify,
}: {
  notify: (text: string, kind?: "success" | "error") => void;
}) {
  const [prefs, setPrefs] = useState<NotificationPrefs | null>(null);
  const [draft, setDraft] = useState<NotificationPrefs | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      const p = await getNotificationPrefs();
      setPrefs(p);
      setDraft(p);
    } catch {
      // 后端不可达时保持空态。
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  function patch(partial: Partial<NotificationPrefs>) {
    setDraft((cur) => (cur ? { ...cur, ...partial } : cur));
  }

  async function save() {
    if (!draft) return;
    setSaving(true);
    try {
      const next = await updateNotificationPrefs(draft);
      setPrefs(next);
      setDraft(next);
      notify("通知偏好已保存");
    } catch (e) {
      notify(e instanceof Error ? e.message : "保存失败", "error");
    } finally {
      setSaving(false);
    }
  }

  const dirty = draft && prefs && JSON.stringify(draft) !== JSON.stringify(prefs);

  return <>
    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>通知偏好</h3>
        </div>
        <button className="btn primary" type="button" disabled={saving || !dirty} onClick={() => void save()}>
          {saving ? "保存中..." : "保存"}
        </button>
      </div>

      {draft ? (
        <>
          <div className="settings-status-list">
            {TOGGLE_ROWS.map((row) => (
              <div key={row.key}>
                <span>{row.title}</span>
                <label className="settings-tool-switch" style={{ marginTop: 2 }}>
                  <span className="switch">
                    <input
                      type="checkbox"
                      checked={Boolean(draft[row.key])}
                      onChange={(e) => patch({ [row.key]: e.target.checked })}
                      aria-label={row.title}
                    />
                    <span className="track" />
                    <span className="thumb" />
                  </span>
                </label>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="settings-empty">读取通知偏好...</p>
      )}
    </section>

    <section className="card settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>
            邮箱通知
            <InfoHint label="邮箱通知说明">
              站外邮件推送尚未接入，保存后仅记录偏好，当前版本不会实际发送邮件。
            </InfoHint>
          </h3>
        </div>
      </div>
      <label className="field" style={{ maxWidth: 360 }}>
        接收邮箱
        <input
          type="email"
          placeholder="you@example.com"
          value={draft?.email ?? ""}
          onChange={(e) => patch({ email: e.target.value })}
        />
      </label>
    </section>
  </>;
}
