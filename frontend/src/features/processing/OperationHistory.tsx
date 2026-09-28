/**
 * 处理记录（原 `pages/Processing/index.tsx` 末尾的 OperationHistory）。
 *
 * 与 `OperationPanel` 同属数据处理领域，故一并从页面下沉到 `features/processing`。
 * 行为不变：仍按 datasetId 拉 `getProcessingHistory`，并保留「加载中 / 失败可重试 / 空态」三态。
 */
import { useEffect, useState } from "react";
import { getProcessingHistory, type OperationHistoryItem } from "../../api/processing";

/** 处理记录。字段为 operation_type / input_version_id / output_version_id。 */
export function OperationHistory({ datasetId }: { datasetId: number }) {
  const [items, setItems] = useState<OperationHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState<string | null>(null);

  function load() {
    setLoading(true);
    setFailed(null);
    getProcessingHistory(datasetId)
      .then(setItems)
      .catch((error: unknown) => {
        setItems([]);
        setFailed(error instanceof Error ? error.message : "加载处理记录失败");
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, [datasetId]);

  if (loading) return <div className="card" style={{ marginTop: "var(--space-4)" }}><div className="muted">正在加载处理记录…</div></div>;

  return (
    <section className="card" style={{ marginTop: "var(--space-4)" }}>
      <div className="section-title">
        <h3>处理记录</h3>
        <span className="muted">{items.length} 条</span>
      </div>
      {failed && <div className="processing-error">{failed} <button className="btn btn-sm" type="button" onClick={load}>重试</button></div>}
      {!failed && items.length === 0 && <div className="processing-empty">暂无处理记录</div>}
      {items.length > 0 && (
        <div className="processing-history-list">
          {items.map((item) => (
            <div className="processing-history-item" key={item.id}>
              <div>
                <strong>{item.operation_type}</strong>
                <div className="muted">
                  {versionLabel(item.input_version_id)} → {versionLabel(item.output_version_id)} · {item.status}
                </div>
                {item.error && <div className="muted">{item.error}</div>}
              </div>
              <div className="muted">{item.created_at ?? "—"}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function versionLabel(version: number | null) {
  return version === null ? "最新" : `v${version}`;
}
