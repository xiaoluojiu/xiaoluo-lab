import { useEffect, useState } from "react";
import type { ClarificationRequest } from "../../types/agent";

/**
 * 待澄清问题面板：后端在开工前（Pre-flight）或计划执行中缺少必要信息时的反问。
 *
 * 与授权弹窗（PermissionDialog）的区别要说清楚，两者长得像但语义相反：
 * - 授权问「这个高风险操作**做不做**」→ 允许 / 拒绝；
 * - 澄清问「信息不全，**做哪个**」→ 给一个值，运行从原处继续。
 *
 * ★ 这个组件之前根本不存在，于是后端进入 `waiting_clarification` 之后：
 * 界面没有任何可操作的东西，轮询第一次取到该状态就停止，进度条停在 8% 一动不动
 * —— 用户只能看着它「卡死」。等待用户回答必须**看得见**，否则和死循环没有区别。
 */
export function ClarificationPanel({
  request,
  busy = false,
  onSubmit,
  onCancel,
}: {
  request: ClarificationRequest | null;
  busy?: boolean;
  /** 回传**机读值**（options[].value），不是展示文案。 */
  onSubmit: (answer: string) => void;
  /** 放弃本次运行（后端会终止 run，否则会话被 409 锁死）。 */
  onCancel: () => void;
}) {
  const options = request?.options ?? [];
  // 选中态：有默认值时预选默认值，否则预选第一个选项（没有选项则留空走自由输入）。
  const [value, setValue] = useState("");

  useEffect(() => {
    if (!request) return;
    const preset = request.default ?? (options[0]?.value ?? "");
    setValue(preset);
  }, [request, options]);

  if (!request) return null;
  const trimmed = value.trim();
  const useSelect = options.length > 0;

  function submit() {
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  }

  return (
    <div className="dialog-mask" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label="需要补充信息">
        <h3>需要你补充一点信息</h3>
        <p className="muted" style={{ marginTop: 0 }}>{request.question || "请补充必要信息后继续。"}</p>

        {useSelect ? (
          <div className="mt">
            <div className="muted" style={{ marginBottom: 6 }}>可选项</div>
            <div className="clarify-options">
              {options.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`btn clarify-option${value === opt.value ? " active" : ""}`}
                  onClick={() => setValue(opt.value)}
                  disabled={busy}
                >
                  <strong>{opt.label || opt.value}</strong>
                  {opt.note && <span className="muted">{opt.note}</span>}
                </button>
              ))}
            </div>
            {/* 选项不够用时允许直接填列名：反问给的是「候选」，不是闭集。 */}
            <label className="field mt">
              或直接填写
              <input
                type="text"
                value={value}
                placeholder="例如：DepDelay"
                onChange={(e) => setValue(e.target.value)}
                disabled={busy}
                onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
              />
            </label>
          </div>
        ) : (
          <label className="field mt">
            你的回答
            <input
              type="text"
              value={value}
              placeholder="输入后回车提交"
              onChange={(e) => setValue(e.target.value)}
              disabled={busy}
              onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
              autoFocus
            />
          </label>
        )}

        <div className="dialog-footer">
          <button className="btn primary" type="button" disabled={!trimmed || busy} onClick={submit}>
            {busy ? "提交中…" : "提交并继续"}
          </button>
          <button className="btn" type="button" disabled={busy} onClick={onCancel}>
            放弃本次运行
          </button>
        </div>
      </div>
    </div>
  );
}
