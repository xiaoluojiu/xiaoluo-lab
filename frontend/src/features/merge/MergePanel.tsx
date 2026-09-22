/**
 * T0-M1：MergePanel —— 端到端合并流程。
 *
 * 步骤：
 *  1. 选择左/右数据集（首次选择自动固定当前 latest version，整流程不变）
 *  2. 自动加载 schema
 *  3. 获取字段映射建议（可选）
 *  4. 选择 Join Key（支持 composite key）
 *  5. 显示 Key 分析
 *  6. 选择 Join 类型
 *  7. 选择右表需要引入的列（可选，默认全部非 Key 列）
 *  8. Preview（T0-M3：内存预览，不创建版本）
 *  9. Validate（校验冲突/多对多）
 * 10. Execute（T0-M4：使用 preview 返回的固定版本号）
 * 11. 显示新 DatasetVersion + MergeReport
 */
import { useEffect, useMemo, useState } from "react";
import { listDatasets, getSchema } from "../../api/datasets";
import {
  analyzeKeys,
  executeMerge,
  previewMerge,
  suggestMapping,
  validateMerge,
} from "../../api/merge";
import type {
  Dataset,
  SchemaColumn,
} from "../../types/dataset";
import type {
  KeyAnalysisResult,
  MappingCandidate,
  MergeExecuteResult,
  MergePlan,
  MergePreviewResult,
  ValidationResult,
} from "../../types/merge";
import { InfoHint } from "../../components/InfoHint";

const JOIN_TYPES: Array<{ value: MergePlan["join_type"]; label: string; desc: string }> = [
  { value: "inner", label: "Inner", desc: "只保留两侧匹配的行" },
  { value: "left", label: "Left", desc: "保留全部左表" },
  { value: "right", label: "Right", desc: "保留全部右表" },
  { value: "outer", label: "Outer", desc: "两侧并集" },
];

interface DatasetState {
  dataset?: Dataset;
  version: number | null; // 固定版本号
  columns: SchemaColumn[];
  loading: boolean;
  error: string | null;
}

interface SelectedKey {
  left: string;
  right: string;
}

export function MergePanel() {
  // 左/右数据集状态（版本固定）
  const [left, setLeft] = useState<DatasetState>({
    version: null, columns: [], loading: false, error: null,
  });
  const [right, setRight] = useState<DatasetState>({
    version: null, columns: [], loading: false, error: null,
  });
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // 用户选择
  const [selectedKeys, setSelectedKeys] = useState<SelectedKey[]>([
    { left: "", right: "" },
  ]);
  const [joinType, setJoinType] = useState<MergePlan["join_type"]>("inner");
  const [rightColumns, setRightColumns] = useState<string[] | null>(null);
  const [includeRightAll, setIncludeRightAll] = useState(true);

  // 中间结果
  const [mappingCandidates, setMappingCandidates] = useState<MappingCandidate[] | null>(null);
  const [keyAnalysis, setKeyAnalysis] = useState<KeyAnalysisResult | null>(null);
  const [preview, setPreview] = useState<MergePreviewResult | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [executeResult, setExecuteResult] = useState<MergeExecuteResult | null>(null);

  // 错误与忙碌
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  // ---- 数据集列表 ----
  useEffect(() => {
    listDatasets(1, 200)
      .then((r) => setDatasets(r.items.filter((d) => d.latest_version)))
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"));
  }, []);

  // 关键配置变化后清空旧的预览/校验/执行结果，避免展示过期结果
  useEffect(() => {
    setPreview(null);
    setValidation(null);
    setExecuteResult(null);
  }, [selectedKeys, joinType, includeRightAll, rightColumns]);

  // ---- 选中数据集时：固定当前 latest version + 加载 schema ----
  function pickSide(side: "left" | "right", dataset: Dataset) {
    const setter = side === "left" ? setLeft : setRight;
    setter({
      dataset,
      version: dataset.latest_version?.version ?? null,
      columns: [],
      loading: true,
      error: null,
    });
    if (!dataset.latest_version?.version) {
      setter((s) => ({ ...s, loading: false, error: "该数据集无版本" }));
      return;
    }
    getSchema(dataset.id, dataset.latest_version.version)
      .then((s) =>
        setter((prev) => ({ ...prev, columns: s.columns, loading: false }))
      )
      .catch((e) =>
        setter((prev) => ({
          ...prev,
          loading: false,
          error: e instanceof Error ? e.message : "schema 加载失败",
        }))
      );
  }

  // ---- 构造 plan ----
  const plan: MergePlan = useMemo(
    () => ({
      keys: selectedKeys.filter((k) => k.left && k.right),
      mapping: [],
      join_type: joinType,
      right_columns: includeRightAll ? null : rightColumns,
    }),
    [selectedKeys, joinType, includeRightAll, rightColumns]
  );

  // ---- 调用 mapping ----
  async function handleSuggestMapping() {
    if (!left.dataset || !right.dataset) return;
    setBusy(true); setBusyAction("mapping"); setError(null);
    try {
      const result = await suggestMapping({
        left_dataset_id: left.dataset.id,
        right_dataset_id: right.dataset.id,
        left_version: left.version,
        right_version: right.version,
      });
      setMappingCandidates(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "映射建议失败");
    } finally {
      setBusy(false); setBusyAction(null);
    }
  }

  // ---- 调用 keys ----
  async function handleAnalyzeKeys() {
    if (!left.dataset || !right.dataset) return;
    if (!selectedKeys.length || !selectedKeys[0].left || !selectedKeys[0].right) {
      setError("请先选择 Join Key");
      return;
    }
    setBusy(true); setBusyAction("keys"); setError(null);
    try {
      const leftKeys = selectedKeys.map((k) => k.left);
      const rightKeys = selectedKeys.map((k) => k.right);
      const result = await analyzeKeys({
        left_dataset_id: left.dataset.id,
        right_dataset_id: right.dataset.id,
        left_version: left.version,
        right_version: right.version,
        left_keys: leftKeys.length > 1 ? leftKeys : null,
        right_keys: rightKeys.length > 1 ? rightKeys : null,
        left_key: leftKeys.length === 1 ? leftKeys[0] : null,
        right_key: rightKeys.length === 1 ? rightKeys[0] : null,
      });
      setKeyAnalysis(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Key 分析失败");
    } finally {
      setBusy(false); setBusyAction(null);
    }
  }

  // ---- 调用 preview ----
  async function handlePreview() {
    if (!left.dataset || !right.dataset) return;
    if (!plan.keys.length) {
      setError("请先选择至少一个 Join Key");
      return;
    }
    setBusy(true); setBusyAction("preview"); setError(null);
    setPreview(null); setValidation(null); setExecuteResult(null);
    try {
      const result = await previewMerge({
        left_dataset_id: left.dataset.id,
        right_dataset_id: right.dataset.id,
        left_version: left.version,
        right_version: right.version,
        plan,
      });
      setPreview(result);
      // T0-M4：preview 返回的版本号是 execute 必须使用的版本
      // 这里其实和已固定的版本一致，但仍是后端权威返回
    } catch (e) {
      setError(e instanceof Error ? e.message : "预览失败");
    } finally {
      setBusy(false); setBusyAction(null);
    }
  }

  // ---- 调用 validate ----
  async function handleValidate() {
    if (!left.dataset || !right.dataset) return;
    if (!plan.keys.length) {
      setError("请先选择至少一个 Join Key");
      return;
    }
    setBusy(true); setBusyAction("validate"); setError(null);
    try {
      const result = await validateMerge({
        left_dataset_id: left.dataset.id,
        right_dataset_id: right.dataset.id,
        left_version: left.version,
        right_version: right.version,
        plan,
      });
      setValidation(result);
      if (!result.ok) {
        setError(`校验失败：${result.errors.join("; ")}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "校验失败");
    } finally {
      setBusy(false); setBusyAction(null);
    }
  }

  // ---- 调用 execute ----
  async function handleExecute() {
    if (!left.dataset || !right.dataset) return;
    if (!left.version || !right.version) {
      setError("版本号缺失");
      return;
    }
    setBusy(true); setBusyAction("execute"); setError(null);
    setExecuteResult(null);
    try {
      // T0-M4：execute 必须使用 preview/validate 时的固定版本号
      const result = await executeMerge({
        left_dataset_id: left.dataset.id,
        right_dataset_id: right.dataset.id,
        left_version: left.version,
        right_version: right.version,
        plan,
      });
      setExecuteResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "执行失败");
    } finally {
      setBusy(false); setBusyAction(null);
    }
  }

  // ---- UI ----
  const leftColNames = left.columns.map((c) => c.column);
  const rightColNames = right.columns.map((c) => c.column);
  const ready = !!left.dataset && !!right.dataset && !!left.version && !!right.version;

  function updateKey(idx: number, field: "left" | "right", value: string) {
    setSelectedKeys((keys) =>
      keys.map((k, i) => (i === idx ? { ...k, [field]: value } : k))
    );
  }

  return (
    <div className="card" style={{ marginBottom: "var(--space-4)" }}>
      <h3 style={{ marginTop: 0 }}>
        数据集合并
        <InfoHint label="合并流程说明">
          选择数据集 → 字段映射（可选）→ 选 Join Key → Key 分析 → 选 Join 类型 →
          Preview → Validate → Execute → 生成新版本。版本号一旦固定不再变化。
        </InfoHint>
      </h3>

      {/* 1. 选数据集 */}
      <div className="form-row" style={{ gap: "var(--space-4)" }}>
        <div className="col">
          <label>① 左数据集（输出基表）</label>
          <select
            value={left.dataset?.id ?? ""}
            onChange={(e) => {
              const ds = datasets.find((d) => d.id === Number(e.target.value));
              if (ds) pickSide("left", ds);
            }}
          >
            <option value="">请选择</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} (v{d.latest_version?.version ?? 0})
              </option>
            ))}
          </select>
          {left.version && (
            <span className="muted" style={{ marginLeft: "var(--space-2)" }}>
              已固定 v{left.version} · {left.columns.length} 列
            </span>
          )}
        </div>
        <div className="col">
          <label>② 右数据集</label>
          <select
            value={right.dataset?.id ?? ""}
            onChange={(e) => {
              const ds = datasets.find((d) => d.id === Number(e.target.value));
              if (ds) pickSide("right", ds);
            }}
          >
            <option value="">请选择</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} (v{d.latest_version?.version ?? 0})
              </option>
            ))}
          </select>
          {right.version && (
            <span className="muted" style={{ marginLeft: "var(--space-2)" }}>
              已固定 v{right.version} · {right.columns.length} 列
            </span>
          )}
        </div>
      </div>

      {error && (
        <div style={{ color: "var(--danger)", marginTop: "var(--space-3)", whiteSpace: "pre-wrap" }}>
          {error}
        </div>
      )}

      {ready && (
        <>
          {/* 2. 字段映射建议 */}
          <div style={{ marginTop: "var(--space-4)" }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button
                type="button"
                className="btn"
                onClick={() => void handleSuggestMapping()}
                disabled={busy}
              >
                {busyAction === "mapping" ? "分析中..." : "③ 获取字段映射建议"}
              </button>
              {mappingCandidates && (
                <span className="muted">{mappingCandidates.length} 条候选</span>
              )}
            </div>
            {mappingCandidates && (
              <div style={{ marginTop: "var(--space-2)", overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>左表列 (source)</th>
                      <th style={thStyle}>右表列 (target)</th>
                      <th style={thStyle}>置信度</th>
                      <th style={thStyle}>理由</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mappingCandidates.slice(0, 20).map((c, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{c.source_column}</td>
                        <td style={tdStyle}>{c.target_column}</td>
                        <td style={tdStyle}>{(c.confidence * 100).toFixed(0)}%</td>
                        <td style={tdStyle}>{c.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="muted" style={{ marginTop: 6 }}>
                  说明：source_column=左表列，target_column=右表列。仅描述对应关系，不影响执行重命名。
                </p>
              </div>
            )}
          </div>

          {/* 3. Join Key 选择（支持 composite） */}
          <div style={{ marginTop: "var(--space-4)" }}>
            <label style={{ fontWeight: 600 }}>④ Join Key（可多选，支持 composite key）</label>
            {selectedKeys.map((k, idx) => (
              <div
                key={idx}
                className="form-row"
                style={{ marginTop: "var(--space-2)", alignItems: "flex-end" }}
              >
                <div className="col">
                  <label>左表 Key</label>
                  <select
                    value={k.left}
                    onChange={(e) => updateKey(idx, "left", e.target.value)}
                  >
                    <option value="">请选择</option>
                    {leftColNames.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </div>
                <div className="col">
                  <label>右表 Key</label>
                  <select
                    value={k.right}
                    onChange={(e) => updateKey(idx, "right", e.target.value)}
                  >
                    <option value="">请选择</option>
                    {rightColNames.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </div>
                <button
                  type="button"
                  className="btn"
                  onClick={() =>
                    setSelectedKeys((keys) => keys.filter((_, i) => i !== idx))
                  }
                  disabled={selectedKeys.length <= 1}
                >
                  删除
                </button>
              </div>
            ))}
            <button
              type="button"
              className="btn"
              style={{ marginTop: "var(--space-2)" }}
              onClick={() =>
                setSelectedKeys((keys) => [...keys, { left: "", right: "" }])
              }
            >
              + 添加 Key
            </button>

            <div style={{ marginTop: 10 }}>
              <button
                type="button"
                className="btn"
                onClick={() => void handleAnalyzeKeys()}
                disabled={busy}
              >
                {busyAction === "keys" ? "分析中..." : "⑤ Key 分析"}
              </button>
            </div>
            {keyAnalysis && (
              <div style={{ marginTop: "var(--space-2)" }}>
                <span className="badge">基数：{keyAnalysis.cardinality}</span>{" "}
                <span className="muted">
                  左覆盖右：{(keyAnalysis.left_key_coverage_in_right * 100).toFixed(0)}%{" "}
                  右覆盖左：{(keyAnalysis.right_key_coverage_in_left * 100).toFixed(0)}%
                </span>
                <div className="muted" style={{ marginTop: "var(--space-1)" }}>
                  左 unique: {keyAnalysis.left.unique_count} | 右 unique:{" "}
                  {keyAnalysis.right.unique_count} | 重复(左/右):{" "}
                  {keyAnalysis.left.duplicate_count}/{keyAnalysis.right.duplicate_count}
                </div>
              </div>
            )}
          </div>

          {/* 4. Join 类型 */}
          <div style={{ marginTop: "var(--space-4)" }}>
            <label style={{ fontWeight: 600 }}>⑥ Join 类型</label>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                gap: "var(--space-2)",
                marginTop: "var(--space-2)",
              }}
            >
              {JOIN_TYPES.map((j) => (
                <button
                  key={j.value}
                  type="button"
                  className={`btn ${joinType === j.value ? "primary" : ""}`}
                  onClick={() => setJoinType(j.value)}
                  style={{ textAlign: "left", padding: 10 }}
                >
                  <div style={{ fontWeight: 600 }}>{j.label}</div>
                  <div className="muted" style={{ fontSize: 12 }}>{j.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* 5. 右表列选择 */}
          <div style={{ marginTop: "var(--space-4)" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={includeRightAll}
                onChange={(e) => setIncludeRightAll(e.target.checked)}
              />
              ⑦ 引入右表全部非 Key 列（取消后可手动选择）
            </label>
            {!includeRightAll && (
              <div style={{ marginTop: "var(--space-2)" }}>
                <select
                  multiple
                  value={rightColumns ?? []}
                  onChange={(e) =>
                    setRightColumns(
                      Array.from(e.target.selectedOptions).map((o) => o.value)
                    )
                  }
                  style={{ width: "100%", minHeight: 100 }}
                >
                  {rightColNames
                    .filter((n) => !selectedKeys.some((k) => k.right === n))
                    .map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                </select>
                <p className="muted">按住 Ctrl/Cmd 多选。</p>
              </div>
            )}
          </div>

          {/* 6. 操作按钮 */}
          <div style={{ marginTop: 18, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn"
              onClick={() => void handlePreview()}
              disabled={busy || !plan.keys.length}
            >
              {busyAction === "preview" ? "预览中..." : "⑧ Preview"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => void handleValidate()}
              disabled={busy || !plan.keys.length}
            >
              {busyAction === "validate" ? "校验中..." : "⑨ Validate"}
            </button>
            <button
              type="button"
              className="btn primary"
              onClick={() => void handleExecute()}
              disabled={busy || !plan.keys.length || !preview}
            >
              {busyAction === "execute" ? "执行中..." : "⑩ Execute"}
            </button>
          </div>
          {!preview && (
            <p className="muted" style={{ marginTop: 10, marginBottom: 0 }}>
              Execute 必须使用 Preview / Validate 时固定的版本。
            </p>
          )}

          {/* 7. Preview 结果 */}
          {preview && (
            <PreviewCard preview={preview} />
          )}

          {/* 8. Validation 结果 */}
          {validation && (
            <div className="card" style={{ marginTop: "var(--space-3)", background: "var(--surface-muted, transparent)" }}>
              <h4 style={{ marginTop: 0 }}>
                校验结果：{validation.ok ? "通过" : "失败"}
              </h4>
              {validation.errors.length > 0 && (
                <div style={{ color: "var(--danger)" }}>
                  <strong>错误：</strong>
                  <ul>{validation.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
                </div>
              )}
              {validation.warnings.length > 0 && (
                <div className="muted">
                  <strong>警告：</strong>
                  <ul>{validation.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              )}
            </div>
          )}

          {/* 9. Execute 结果 */}
          {executeResult && <ExecuteSuccessCard result={executeResult} />}
        </>
      )}
    </div>
  );
}

/* ============================================================
 * Preview Card
 * ============================================================ */
function PreviewCard({ preview }: { preview: MergePreviewResult }) {
  return (
    <div
      className="card"
      style={{
        marginTop: "var(--space-3)",
        background: "var(--surface-muted, transparent)",
      }}
    >
      <h4 style={{ marginTop: 0 }}>预览结果</h4>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "var(--space-2)",
        }}
      >
        <Stat label="左表" value={`${preview.input_rows_left}行 × ${preview.input_columns_left}列`} />
        <Stat label="右表" value={`${preview.input_rows_right}行 × ${preview.input_columns_right}列`} />
        <Stat label="输出" value={`${preview.output_rows}行 × ${preview.output_columns}列`} />
        <Stat label="匹配行" value={String(preview.matched_rows)} />
        <Stat label="未匹配(左)" value={String(preview.unmatched_rows_left)} />
        <Stat label="未匹配(右)" value={String(preview.unmatched_rows_right)} />
      </div>
      {preview.warnings.length > 0 && (
        <div className="muted" style={{ marginTop: 10 }}>
          <strong>警告：</strong>
          <ul>{preview.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}
      <div style={{ marginTop: "var(--space-3)" }}>
        <h5>数据预览（前 {preview.preview.items.length} 行 / 共 {preview.preview.total}）</h5>
        <PreviewTable preview={preview.preview} />
      </div>
    </div>
  );
}

function PreviewTable({
  preview,
}: {
  preview: { columns: string[]; items: Record<string, unknown>[]; total: number };
}) {
  if (!preview.items.length) {
    return <div className="muted">没有可展示的数据。</div>;
  }
  return (
    <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 600 }}>
        <thead>
          <tr>
            {preview.columns.map((c) => (
              <th key={c} style={thStyle}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.items.map((row, i) => (
            <tr key={i}>
              {preview.columns.map((c) => (
                <td key={c} style={tdStyle} title={String(row[c] ?? "")}>
                  {formatCell(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================
 * Execute Success
 * ============================================================ */
function ExecuteSuccessCard({ result }: { result: MergeExecuteResult }) {
  return (
    <div className="card" style={{ marginTop: "var(--space-3)", borderColor: "var(--success)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <h3 style={{ margin: 0 }}>合并完成</h3>
        <span className="badge success">已生成新版本</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 10,
          marginTop: "var(--space-3)",
        }}
      >
        <Stat label="新版本" value={`v${result.version.version}`} />
        <Stat label="输出行数" value={String(result.version.row_count)} />
        <Stat label="输出列数" value={String(result.version.column_count)} />
        <Stat label="匹配行" value={String(result.report.matched_rows)} />
        <Stat label="未匹配(左)" value={String(result.report.unmatched_rows_left)} />
        <Stat label="未匹配(右)" value={String(result.report.unmatched_rows_right)} />
      </div>
      <p className="muted" style={{ marginBottom: 0, marginTop: 10 }}>
        原输入版本未被覆盖；新版本已记录到 Dataset 的版本链中。
      </p>
    </div>
  );
}

/* ============================================================
 * Small helpers
 * ============================================================ */
const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid var(--border)",
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "7px 10px",
  borderBottom: "1px solid var(--border)",
  maxWidth: 200,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: 10 }}>
      <div className="muted" style={{ fontSize: 12, marginBottom: "var(--space-1)" }}>{label}</div>
      <div style={{ fontWeight: 600 }}>{value}</div>
    </div>
  );
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}
