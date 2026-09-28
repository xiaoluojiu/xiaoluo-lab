import { useEffect, useMemo, useState } from "react";
import type { OperationPreview } from "../../api/processing";
import { SampleTable } from "../../components/SampleTable";
import {
  RISK_LABEL,
  assessOperationRisk,
  impactSummary,
  shapeTransition,
} from "../../lib/processingRisk";

/**
 * 数据处理「预览 + 确认」弹窗。
 *
 * 这是所有写操作的统一闸门：用户点「执行」不再直接落新版本，
 * 而是先看到这次操作**会删掉多少行、多少列**以及操作后的数据样例，
 * 确认之后才真正执行。
 *
 * 高风险操作（删列 / 删大量行 / 结果变空表）走第二步强制二次确认 ——
 * 必须勾选知晓后才能点最终的执行按钮，避免「顺手点两下」把表删空。
 *
 * 数据来源是后端已有的 dry-run 接口（POST /processing/datasets/{id}/preview），
 * 没有新增任何后端接口。
 */

/** 样例表最多展示的行数（需求口径：前 10 行）。 */
export const PREVIEW_ROWS = 10;

interface Props {
  open: boolean;
  /** 操作名，用于标题。 */
  title: string;
  /** dry-run 结果；null 表示仍在计算（此时弹窗只展示占位）。 */
  preview: OperationPreview | null;
  /** 执行失败的原因：保留弹窗，用户可直接重试或取消，不必重新预览。 */
  error?: string | null;
  /** 执行请求已发出。 */
  executing?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function OperationPreviewDialog({
  open,
  title,
  preview,
  error,
  executing = false,
  onCancel,
  onConfirm,
}: Props) {
  const [step, setStep] = useState<1 | 2>(1);
  const [acknowledged, setAcknowledged] = useState(false);

  const risk = useMemo(
    () => (preview ? assessOperationRisk(preview.before, preview.after) : null),
    [preview],
  );

  // 每次重新打开 / 换一次预览都回到第一步，且清掉勾选状态：
  // 否则第二次的高风险操作会带着上一次的「已知晓」直接放行。
  useEffect(() => {
    setStep(1);
    setAcknowledged(false);
  }, [open, preview]);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !executing) onCancel();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, executing, onCancel]);

  if (!open) return null;

  const rows = preview?.preview.items?.slice(0, PREVIEW_ROWS) ?? [];
  const columns = preview?.preview.columns ?? [];
  const highRisk = risk?.requiresSecondConfirmation ?? false;

  return (
    <div
      className="dialog-mask"
      onClick={() => {
        if (!executing) onCancel();
      }}
    >
      <div
        className="dialog dialog-lg"
        role="dialog"
        aria-modal="true"
        aria-label={`${title} 影响预览`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <div className="dialog-title">
              {step === 2 ? "高风险操作，请再次确认" : `预览：${title}`}
            </div>
            <div className="muted">
              原版本不会被覆盖 —— 确认后执行，并生成一个新的数据版本。
            </div>
          </div>
          {risk && (
            <span className={`badge ${risk.level === "high" ? "danger" : risk.level === "medium" ? "warning" : "success"}`}>
              {RISK_LABEL[risk.level]}
            </span>
          )}
        </div>

        <div className="dialog-body">
          {error && (
            <div className="alert danger">
              <div>
                <div className="alert-title">执行失败</div>
                <div className="alert-body">{error}</div>
              </div>
            </div>
          )}

          {!preview && <div className="muted">正在计算这次操作的影响…</div>}

          {preview && risk && step === 1 && (
            <>
              <div className="op-preview-impact">
                <div className="op-preview-impact-line">
                  {impactSummary(preview.before, preview.after)}
                </div>
                <div className="muted">
                  {shapeTransition(preview.before, preview.after)}
                  {`（输入版本 v${preview.input_version}）`}
                </div>
                <ul className="op-preview-reasons">
                  {risk.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>

              <div className="op-preview-sample-head">
                <strong>操作后数据样例</strong>
                <span className="muted">
                  {rows.length ? `最多 ${PREVIEW_ROWS} 行` : "结果为空表，没有可展示的行"}
                </span>
              </div>
              {/* 空结果这里沿用的是「一行灰字」的旧样式，故显式传 muted。 */}
              <SampleTable rows={rows} columns={columns} emptyClassName="muted" />
            </>
          )}

          {preview && risk && step === 2 && (
            <>
              <div className="alert danger">
                <div>
                  <div className="alert-title">这次操作会丢弃数据</div>
                  <div className="alert-body">
                    影响不可直接在当前版本上撤销；如需回退，只能把数据重新处理到旧版本。
                  </div>
                </div>
              </div>
              <div className="op-preview-impact">
                <div className="op-preview-impact-line">
                  {impactSummary(preview.before, preview.after)}
                </div>
                <div className="muted">{shapeTransition(preview.before, preview.after)}</div>
                <ul className="op-preview-reasons">
                  {risk.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
              <label className="op-preview-ack">
                <input
                  type="checkbox"
                  checked={acknowledged}
                  onChange={(event) => setAcknowledged(event.target.checked)}
                />
                <span>我已确认上述影响，继续执行</span>
              </label>
            </>
          )}
        </div>

        <div className="dialog-footer">
          {step === 1 ? (
            <>
              <button className="btn" type="button" onClick={onCancel} disabled={executing}>
                取消
              </button>
              <button
                className={`btn ${highRisk ? "danger" : "primary"}`}
                type="button"
                disabled={executing || !preview}
                onClick={() => {
                  if (highRisk) setStep(2);
                  else onConfirm();
                }}
              >
                {executing
                  ? "执行中…"
                  : highRisk
                    ? "继续（此操作有风险）"
                    : "确认执行"}
              </button>
            </>
          ) : (
            <>
              <button
                className="btn"
                type="button"
                onClick={() => setStep(1)}
                disabled={executing}
              >
                返回预览
              </button>
              <button
                className="btn danger"
                type="button"
                disabled={!acknowledged || executing}
                onClick={onConfirm}
              >
                {executing ? "执行中…" : "确认执行"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
