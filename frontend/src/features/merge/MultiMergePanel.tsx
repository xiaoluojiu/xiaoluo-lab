/**
 * MultiMergePanel —— 多文件合并（N 个数据集纵向整合 / 字段并集）。
 *
 * 与 MergePanel（两表 join）互补，面向「多张同构/半同构表整合到一张宽表」：
 *  1. 勾选多个数据文件（可留空版本 = 用最新版本）
 *  2. 分析字段：列出「全部文件共有」与「仅有部分文件有」的字段
 *  3. 勾选要保留的字段（缺失字段在不存在的文件中补 null）
 *  4. 可选：追加来源标记列，便于追溯每行来自哪个文件
 *  5. 预览（内存预览，不创建版本）
 *  6. 执行：生成新数据集 或 合并至指定已有数据集的新版本
 */
import { useEffect, useMemo, useState } from "react";
import { listDatasets } from "../../api/datasets";
import {
  analyzeMultiMerge,
  executeMultiMerge,
  previewMultiMerge,
  type MultiMergeAnalyzeResult,
  type MultiMergeExecuteResult,
  type MultiMergePreviewResult,
  type MultiMergeRef,
} from "../../api/merge";
import type { Dataset } from "../../types/dataset";

export function MultiMergePanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  const [analysis, setAnalysis] = useState<MultiMergeAnalyzeResult | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [addSource, setAddSource] = useState(false);
  const [sourceColumn, setSourceColumn] = useState("source");

  const [createNew, setCreateNew] = useState(true);
  const [newName, setNewName] = useState("");
  const [targetId, setTargetId] = useState<number | undefined>();

  const [preview, setPreview] = useState<MultiMergePreviewResult | null>(null);
  const [result, setResult] = useState<MultiMergeExecuteResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"analyze" | "preview" | "execute" | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDatasets(1, 100)
      .then((res) => {
        if (!cancelled) setDatasets(res.items ?? []);
      })
      .catch((e: unknown) => {
        if (!cancelled) setListError(e instanceof Error ? e.message : "加载数据集列表失败");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const nameById = useMemo(
    () => new Map(datasets.map((d) => [d.id, d.name])),
    [datasets],
  );

  const refs = useMemo<MultiMergeRef[]>(
    () =>
      selectedIds.map((id) => ({
        dataset_id: id,
        label: nameById.get(id) ?? `dataset_${id}`,
      })),
    [selectedIds, nameById],
  );

  function toggleDataset(id: number) {
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((v) => v !== id) : [...current, id],
    );
    // 选择变化后，之前的字段分析不再有效
    setAnalysis(null);
    setPreview(null);
    setResult(null);
  }

  async function handleAnalyze() {
    setError(null);
    setPreview(null);
    setResult(null);
    if (refs.length < 2) {
      setError("多文件合并至少需要选择 2 个数据集");
      return;
    }
    setBusy("analyze");
    try {
      const data = await analyzeMultiMerge(refs);
      setAnalysis(data);
      setSelectedColumns(data.all_columns);
    } catch (e) {
      setError(e instanceof Error ? e.message : "字段分析失败");
    } finally {
      setBusy(null);
    }
  }

  function toggleColumn(column: string) {
    setSelectedColumns((current) =>
      current.includes(column)
        ? current.filter((c) => c !== column)
        : [...current, column],
    );
  }

  function selectOnlyCommon() {
    if (!analysis) return;
    setSelectedColumns(analysis.common_columns);
  }

  async function handlePreview() {
    setError(null);
    setResult(null);
    if (selectedColumns.length === 0) {
      setError("请至少选择一个要保留的字段");
      return;
    }
    setBusy("preview");
    try {
      setPreview(
        await previewMultiMerge({
          datasets: refs,
          columns: selectedColumns,
          add_source: addSource,
          source_column: sourceColumn || "source",
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "预览失败");
    } finally {
      setBusy(null);
    }
  }

  async function handleExecute() {
    setError(null);
    if (selectedColumns.length === 0) {
      setError("请至少选择一个要保留的字段");
      return;
    }
    if (!createNew && targetId === undefined) {
      setError("合并至已有文件时必须选择目标数据集");
      return;
    }
    setBusy("execute");
    try {
      setResult(
        await executeMultiMerge({
          datasets: refs,
          columns: selectedColumns,
          add_source: addSource,
          source_column: sourceColumn || "source",
          create_new: createNew,
          target_dataset_id: createNew ? null : targetId,
          name: createNew ? newName || null : null,
        }),
      );
      setPreview(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "执行失败");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="processing-operation-panel">
      <div className="processing-config-header">
        <div>
          <h2>多文件合并</h2>
          <span className="muted op-catalog-desc">
            勾选多个数据文件，按字段对齐纵向整合为一张宽表（缺失字段补 null）
          </span>
        </div>
      </div>

      {listError && <div className="processing-error">{listError}</div>}

      <div className="processing-config-grid">
        <div className="processing-config-main">
          <div className="processing-form-fields">
            <label className="field">
              <span>
                数据文件（已选 {selectedIds.length} 个 · 至少 2 个）
              </span>
              <select
                multiple
                size={Math.min(8, Math.max(4, datasets.length))}
                value={selectedIds.map(String)}
                onChange={(e) => {
                  const next = Array.from(e.target.selectedOptions).map((o) => Number(o.value));
                  setSelectedIds(next);
                  setAnalysis(null);
                  setPreview(null);
                  setResult(null);
                }}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}（{d.latest_version?.row_count ?? d.row_count ?? "?"} 行）
                  </option>
                ))}
              </select>
            </label>
            <div className="visualization-download">
              <button
                className="btn primary"
                type="button"
                disabled={busy !== null || selectedIds.length < 2}
                onClick={() => void handleAnalyze()}
              >
                {busy === "analyze" ? "分析中…" : "分析共有 / 独有字段"}
              </button>
            </div>
          </div>

          {analysis && (
            <div className="multi-merge-fields">
              <div className="multi-merge-summary">
                <span>文件数</span>
                <strong>{analysis.file_count}</strong>
                <span>共有字段</span>
                <strong>{analysis.common_columns.length}</strong>
                <span>非共有字段</span>
                <strong>{analysis.unique_columns.length}</strong>
              </div>

              <div className="visualization-download">
                <button className="btn" type="button" onClick={() => setSelectedColumns(analysis.all_columns)}>
                  全选
                </button>
                <button className="btn" type="button" onClick={selectOnlyCommon}>
                  仅保留共有字段（{analysis.common_columns.length}）
                </button>
                <button className="btn" type="button" onClick={() => setSelectedColumns([])}>
                  清空
                </button>
              </div>

              <div className="multi-merge-field-list">
                {analysis.field_presence.map((f) => (
                  <label
                    key={f.column}
                    className={`multi-merge-field${f.in_all ? " is-common" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedColumns.includes(f.column)}
                      onChange={() => toggleColumn(f.column)}
                    />
                    <span className="multi-merge-field-name">{f.column}</span>
                    <span className="multi-merge-field-tag">
                      {f.in_all
                        ? "全部共有"
                        : `${f.in_count}/${analysis.file_count} 个文件`}
                    </span>
                  </label>
                ))}
              </div>

              <div className="processing-form-fields">
                <label className="field">
                  <span>
                    <input
                      type="checkbox"
                      checked={addSource}
                      onChange={(e) => setAddSource(e.target.checked)}
                    />{" "}
                    追加来源标记列
                  </span>
                </label>
                {addSource && (
                  <label className="field">
                    来源列名
                    <input
                      value={sourceColumn}
                      onChange={(e) => setSourceColumn(e.target.value)}
                    />
                  </label>
                )}
              </div>
            </div>
          )}

          {analysis && (
            <div className="processing-form-fields">
              <label className="field">
                <span>
                  <input
                    type="radio"
                    checked={createNew}
                    onChange={() => setCreateNew(true)}
                  />{" "}
                  合并并生成新数据集
                </span>
              </label>
              {createNew && (
                <label className="field">
                  新数据集名称
                  <input
                    placeholder="留空则自动生成"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                  />
                </label>
              )}
              <label className="field">
                <span>
                  <input
                    type="radio"
                    checked={!createNew}
                    onChange={() => setCreateNew(false)}
                  />{" "}
                  合并至已有数据集（写入新版本）
                </span>
              </label>
              {!createNew && (
                <label className="field">
                  目标数据集
                  <select
                    value={targetId ?? ""}
                    onChange={(e) =>
                      setTargetId(e.target.value ? Number(e.target.value) : undefined)
                    }
                  >
                    <option value="">请选择</option>
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          )}
        </div>

        <div className="processing-preview">
          <div className="processing-preview-header">
            <strong>预览</strong>
            {preview && (
              <span className="muted">
                {preview.row_count} 行 × {preview.column_count} 列
              </span>
            )}
          </div>
          {result ? (
            <div className="processing-result-summary">
              <span>{result.is_new ? "新数据集" : "目标数据集"}</span>
              <strong>#{result.dataset_id}</strong>
              <span>新版本</span>
              <strong>v{result.version.version}</strong>
              <span>行数</span>
              <strong>{result.version.row_count}</strong>
              <span>列数</span>
              <strong>{result.version.column_count}</strong>
            </div>
          ) : preview ? (
            <MultiMergePreviewTable preview={preview} />
          ) : (
            <div className="processing-empty">
              选择文件并分析字段后，可预览合并结果。
            </div>
          )}
        </div>
      </div>

      {error && <div className="processing-error">{error}</div>}

      {analysis && (
        <div className="processing-footer-actions">
          <button
            className="btn"
            type="button"
            disabled={busy !== null}
            onClick={() => void handlePreview()}
          >
            {busy === "preview" ? "预览中…" : "预览"}
          </button>
          <button
            className="btn primary"
            type="button"
            disabled={busy !== null}
            onClick={() => void handleExecute()}
          >
            {busy === "execute" ? "执行中…" : createNew ? "执行并生成新数据集" : "执行并写入目标版本"}
          </button>
        </div>
      )}
    </div>
  );
}

function MultiMergePreviewTable({ preview }: { preview: MultiMergePreviewResult }) {
  const rows = preview.preview?.items ?? [];
  const columns = preview.preview?.columns ?? preview.columns ?? [];
  if (!rows.length) return <div className="processing-empty">暂无预览数据</div>;
  return (
    <div className="processing-table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c}>{formatCell(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
