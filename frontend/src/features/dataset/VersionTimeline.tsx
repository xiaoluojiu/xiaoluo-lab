import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getVersionDiff, getVersionTimeline } from "../../api/datasets";
import type {
  VersionDiffResult,
  VersionTimelineResult,
} from "../../types/dataset";
import { EmptyState, ErrorState, Skeleton } from "../../components/StateBlock";

/**
 * 版本时间线 + 版本差异。
 *
 * 数据全部来自已有的 DatasetVersion（版本）与 Operation（操作来源），
 * 这里只做「读出来并展示」，不参与任何写操作。
 */
export function VersionTimeline({ datasetId }: { datasetId: number }) {
  const [timeline, setTimeline] = useState<VersionTimelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [base, setBase] = useState<number | null>(null);
  const [target, setTarget] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getVersionTimeline(datasetId)
      .then((data) => {
        if (cancelled) return;
        setTimeline(data);
        // 默认对比「最近两个版本」：这是绝大多数时候真正想知道的差异。
        const versions = data.versions.map((v) => v.version);
        setTarget(versions.length ? versions[versions.length - 1] : null);
        setBase(versions.length >= 2 ? versions[versions.length - 2] : versions[0] ?? null);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "版本时间线加载失败");
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const versions = useMemo(() => timeline?.versions ?? [], [timeline]);

  if (error) {
    return (
      <ErrorState
        title="版本时间线没能加载"
        message={error}
        onRetry={() => window.location.reload()}
      />
    );
  }

  if (!timeline) return <Skeleton lines={4} card />;

  if (versions.length === 0) {
    return (
      <EmptyState
        title="这个数据集还没有数据版本"
        description="导入文件或执行一次数据处理后，版本会按时间顺序出现在这里。"
        action={
          <Link className="btn btn-sm" to={`/processing?dataset=${datasetId}`}>
            去处理数据
          </Link>
        }
      />
    );
  }

  return (
    <div style={{ display: "grid", gap: "var(--space-4)" }}>
      <section>
        <div className="flex-between" style={{ marginBottom: "var(--space-2)" }}>
          <h3 style={{ margin: 0 }}>版本时间线</h3>
          <span className="muted">共 {timeline.total} 个版本</span>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>版本</th>
                <th>来源</th>
                <th>规模</th>
                <th>相对上一版本</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {versions.map((item) => (
                <tr key={item.version}>
                  <td>
                    <strong>v{item.version}</strong>
                  </td>
                  <td>
                    {item.origin_label}
                    {item.origin !== "import" ? (
                      <span className="muted" style={{ marginLeft: 6 }}>
                        {item.origin}
                      </span>
                    ) : null}
                  </td>
                  <td>
                    {item.row_count.toLocaleString()} × {item.column_count}
                  </td>
                  <td>
                    {item.delta_rows === null ? (
                      <span className="muted">—</span>
                    ) : (
                      <span>
                        {signed(item.delta_rows)} 行
                        {item.delta_columns ? ` · ${signed(item.delta_columns)} 列` : ""}
                      </span>
                    )}
                  </td>
                  <td className="muted">{(item.created_at ?? "").slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="flex-between" style={{ marginBottom: "var(--space-2)" }}>
          <h3 style={{ margin: 0 }}>版本差异</h3>
          <div style={{ display: "flex", gap: "var(--space-2)", alignItems: "center" }}>
            <label className="muted" htmlFor="diff-base">
              基线
            </label>
            <select
              id="diff-base"
              style={{ width: "auto", minHeight: 32, padding: "4px 8px" }}
              value={base ?? ""}
              onChange={(e) => setBase(Number(e.target.value))}
            >
              {versions.map((v) => (
                <option key={v.version} value={v.version}>
                  v{v.version}
                </option>
              ))}
            </select>
            <span className="muted">→</span>
            <label className="muted" htmlFor="diff-target">
              对比
            </label>
            <select
              id="diff-target"
              style={{ width: "auto", minHeight: 32, padding: "4px 8px" }}
              value={target ?? ""}
              onChange={(e) => setTarget(Number(e.target.value))}
            >
              {versions.map((v) => (
                <option key={v.version} value={v.version}>
                  v{v.version}
                </option>
              ))}
            </select>
          </div>
        </div>

        {versions.length === 1 ? (
          <EmptyState
            title="只有一个版本，还没有可对比的对象"
            description="每次数据处理都会产生新版本而不是覆盖旧版本，之后这里就能看到差异。"
            action={
              <Link className="btn btn-sm" to={`/processing?dataset=${datasetId}`}>
                去处理数据
              </Link>
            }
          />
        ) : (
          <VersionDiff datasetId={datasetId} base={base} target={target} />
        )}
      </section>
    </div>
  );
}

function VersionDiff({
  datasetId,
  base,
  target,
}: {
  datasetId: number;
  base: number | null;
  target: number | null;
}) {
  const [diff, setDiff] = useState<VersionDiffResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (base === null || target === null) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getVersionDiff(datasetId, base, target)
      .then((data) => {
        if (!cancelled) setDiff(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "版本差异计算失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, base, target]);

  if (error) {
    return (
      <ErrorState
        title="差异没有算出来"
        message={error}
        onRetry={() => {
          setError(null);
          setDiff(null);
          if (base !== null && target !== null) {
            void getVersionDiff(datasetId, base, target).then(setDiff).catch(() => null);
          }
        }}
      />
    );
  }

  if (!diff || loading) return <Skeleton lines={3} />;

  const schemaChanges = [
    ...diff.schema_diff.added.map((c) => ({
      column: c.column,
      kind: "新增列",
      detail: c.dtype ?? "",
    })),
    ...diff.schema_diff.removed.map((c) => ({
      column: c.column,
      kind: "删除列",
      detail: c.dtype ?? "",
    })),
    ...diff.schema_diff.type_changed.map((c) => ({
      column: c.column,
      kind: "类型变化",
      detail: `${c.from} → ${c.to}`,
    })),
  ];

  return (
    <div style={{ display: "grid", gap: "var(--space-4)" }}>
      <div className="kv-grid">
        <div className="kv-item">
          <div className="k">行数</div>
          <div className="v">
            {diff.row_count.base.toLocaleString()} → {diff.row_count.target.toLocaleString()}
            {diff.row_count.delta ? `（${signed(diff.row_count.delta)}）` : "（不变）"}
          </div>
        </div>
        <div className="kv-item">
          <div className="k">列数</div>
          <div className="v">
            {diff.column_count.base} → {diff.column_count.target}
            {diff.column_count.delta ? `（${signed(diff.column_count.delta)}）` : "（不变）"}
          </div>
        </div>
        <div className="kv-item">
          <div className="k">结构未变的列</div>
          <div className="v">{diff.schema_diff.unchanged_count}</div>
        </div>
      </div>

      <div>
        <h4 style={{ marginBottom: "var(--space-2)" }}>结构差异</h4>
        {schemaChanges.length ? (
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>列</th>
                  <th>变化</th>
                  <th>类型</th>
                </tr>
              </thead>
              <tbody>
                {schemaChanges.map((change) => (
                  <tr key={`${change.kind}-${change.column}`}>
                    <td>{change.column}</td>
                    <td>{change.kind}</td>
                    <td className="muted">{change.detail || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="muted">两个版本的列结构完全一致。</div>
        )}
      </div>

      <div>
        <h4 style={{ marginBottom: "var(--space-2)" }}>质量变化</h4>
        {diff.quality ? (
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>指标</th>
                  <th>v{diff.base.version}</th>
                  <th>v{diff.target.version}</th>
                  <th>变化</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>问题数</td>
                  <td>{diff.quality.base.issue_count}</td>
                  <td>{diff.quality.target.issue_count}</td>
                  <td>{signed(diff.quality.target.issue_count - diff.quality.base.issue_count)}</td>
                </tr>
                <tr>
                  <td>缺失单元格</td>
                  <td>{diff.quality.base.missing_cells.toLocaleString()}</td>
                  <td>{diff.quality.target.missing_cells.toLocaleString()}</td>
                  <td>
                    {signed(diff.quality.target.missing_cells - diff.quality.base.missing_cells)}
                  </td>
                </tr>
                <tr>
                  <td>重复行</td>
                  <td>{diff.quality.base.duplicate_rows.toLocaleString()}</td>
                  <td>{diff.quality.target.duplicate_rows.toLocaleString()}</td>
                  <td>
                    {signed(diff.quality.target.duplicate_rows - diff.quality.base.duplicate_rows)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <div className="muted">未计算质量变化（可在请求中关闭以加快大表对比）。</div>
        )}
      </div>

      <div>
        <h4 style={{ marginBottom: "var(--space-2)" }}>操作来源</h4>
        {diff.operations.length ? (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {diff.operations.map((op) => (
              <li key={op.id} title={JSON.stringify(op.parameters)}>
                <strong>{op.operation_type}</strong>
                <span className="muted">
                  {" "}
                  · {op.status} · {shortParams(op.parameters)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="muted">
            v{diff.target.version} 不是由已记录的操作产生的（v{diff.base.version} 与 v
            {diff.target.version} 之间可能还有其它版本）。
          </div>
        )}
      </div>
    </div>
  );
}

function signed(value: number): string {
  if (value > 0) return `+${value.toLocaleString()}`;
  return value.toLocaleString();
}

/** 参数太长会撑破布局：截断显示，完整内容放进 title（悬浮可见）。 */
function shortParams(params: Record<string, unknown>): string {
  const raw = JSON.stringify(params ?? {});
  return raw.length > 90 ? `${raw.slice(0, 90)}…` : raw;
}
