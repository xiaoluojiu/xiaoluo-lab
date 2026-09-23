import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { Link } from "react-router-dom";
import { createDataset, deleteDataset, listDatasets } from "../../api/datasets";
import { FILE_ALLOWED_EXTENSIONS, FILE_MAX_SIZE, formatFileSize, getUploadLimits, uploadFile } from "../../api/files";
import type { Dataset } from "../../types/dataset";
import { DataTable, type DataTableColumn } from "../../components/DataTable";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { PageHeader } from "../../components/PageHeader";
import { Panel, SectionHeader } from "../../components/viz/Blocks";
import { KpiCard } from "../../components/viz/KpiCard";
import { Icon } from "../../components/icons/Icon";

/** 数据资产浏览器：列表负责发现，详情/其他模块负责深入处理。 */
export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [keyword, setKeyword] = useState("");
  const [onlyWithVersion, setOnlyWithVersion] = useState(false);
  const [sortRecent, setSortRecent] = useState(true);
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(0);
  // 上传上限由后端下发，前端不再硬编码（否则调大后端上限前端仍会拦下）。
  const [maxUploadSize, setMaxUploadSize] = useState<number>(FILE_MAX_SIZE);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    listDatasets(1, 200)
      .then((r) => setDatasets(r.items))
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"))
      .finally(() => setLoading(false));
  }, [reloadKey]);

  useEffect(() => {
    let alive = true;
    getUploadLimits()
      .then((limits) => {
        if (alive) setMaxUploadSize(limits.max_size_bytes);
      })
      .catch(() => {
        /* 拉取失败保持兜底值 */
      });
    return () => {
      alive = false;
    };
  }, []);

  const pickFile = (file: File | undefined | null) => {
    if (!file) return;
    const ext = (file.name.match(/\.[^.]+$/)?.[0] ?? "").toLowerCase();
    if (!ext || !FILE_ALLOWED_EXTENSIONS.includes(ext as (typeof FILE_ALLOWED_EXTENSIONS)[number])) {
      setUploadError(`不支持的文件类型 ${ext || "（无扩展名）"}，仅支持 ${FILE_ALLOWED_EXTENSIONS.join(" / ")}`);
      return;
    }
    if (file.size > maxUploadSize) {
      setUploadError(`文件过大：${formatFileSize(file.size)}，上限 ${formatFileSize(maxUploadSize)}`);
      return;
    }
    setUploadError(null);
    setSelectedFile(file);
    setUploadName((prev) => prev || file.name.replace(/\.[^.]+$/, ""));
    setProgress(0);
  };

  const resetSelection = () => {
    setSelectedFile(null);
    setUploadName("");
    setProgress(0);
    setUploadError(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError("请先选择文件");
      return;
    }
    setUploading(true);
    setUploadError(null);
    setProgress(0);
    try {
      const saved = await uploadFile(selectedFile, (pct) => setProgress(pct));
      const name = uploadName.trim() || selectedFile.name.replace(/\.[^.]+$/, "");
      await createDataset(name, "", saved.id);
      resetSelection();
      setUploadOpen(false);
      setReloadKey((k) => k + 1);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  const filtered = useMemo(() => {
    const rows = datasets.filter((d) => {
      if (keyword && !`${d.name}${d.description}`.toLowerCase().includes(keyword.toLowerCase())) return false;
      if (onlyWithVersion && !d.latest_version) return false;
      return true;
    });
    return sortRecent ? [...rows].sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""))) : rows;
  }, [datasets, keyword, onlyWithVersion, sortRecent]);

  const recent = filtered.slice(0, 3);

  // 概览指标：全部从已加载的真实数据派生，不额外发请求。
  // 注意：跨数据集相加行数/列数没有业务意义（不同表不能相加），故不展示「总行数/总字段数」。
  const stats = useMemo(() => {
    const formats = new Set(datasets.map((d) => d.latest_version?.format).filter((f): f is string => !!f));
    return {
      formatCount: formats.size,
      formatList: Array.from(formats),
      unversioned: datasets.filter((d) => !d.latest_version).length,
      versioned: datasets.filter((d) => d.latest_version).length,
    };
  }, [datasets]);

  const columns: DataTableColumn<Dataset>[] = [
    { key: "name", title: "数据资产", sortable: true, render: (d) => (
      <div>
        <Link to={`/datasets/${d.id}`} onClick={(e) => e.stopPropagation()} style={{ fontWeight: 600 }}>{d.name}</Link>
        <div className="muted" style={{ fontSize: 12, marginTop: 3 }}>{d.description || "暂无描述"}</div>
      </div>
    ) },
    {
      key: "latest_version", title: "当前数据", sortable: true,
      render: (d) => d.latest_version ? `v${d.latest_version.version} · ${d.latest_version.row_count} 行 × ${d.latest_version.column_count} 列` : "尚未生成版本",
    },
    { key: "created_at", title: "创建", render: (d) => fmtTime(d.created_at) },
    { key: "actions", title: "", render: (d) => (
      <span style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <Link to={`/datasets/${d.id}`} onClick={(e) => e.stopPropagation()}>打开</Link>
        <a href="#" style={{ color: "var(--danger)" }} onClick={(e) => { e.stopPropagation(); e.preventDefault(); setPendingDelete(d); }}>删除</a>
      </span>
    ) },
  ];

  return (
    <div>
      <PageHeader
        title="数据中心"
        description="上传、管理与预览数据集。"
        actions={<button className="btn btn-primary" type="button" onClick={() => setUploadOpen(true)}>＋ 上传数据</button>}
      />

      <SectionHeader title="资产概览" />
      <div className="grid g4">
        <KpiCard label="数据资产" value={datasets.length} icon="database" tone="primary" loading={loading} />
        <KpiCard
          label="已生成版本"
          value={stats.versioned}
          icon="layers"
          tone="success"
          loading={loading}
          hint={datasets.length > 0 ? `占全部资产 ${Math.round((stats.versioned / datasets.length) * 100)}%` : "上传后自动生成首个版本"}
        />
        <KpiCard
          label="数据格式"
          value={stats.formatCount}
          icon="file"
          tone="info"
          loading={loading}
          hint={stats.formatList.length ? stats.formatList.map((f) => f.toUpperCase()).join(" / ") : "尚无数据格式"}
        />
        <KpiCard
          label="待处理"
          value={stats.unversioned}
          icon="clock"
          tone="warning"
          loading={loading}
          hint={stats.unversioned > 0 ? "尚未生成数据版本" : "全部已生成版本"}
        />
      </div>

      {recent.length > 0 && !keyword && (
        <>
          <SectionHeader title="最近使用" description="最新创建的三个数据集。" />
          <Panel flush>
            <div className="activity-list">
              {recent.map((d) => (
                <Link key={d.id} to={`/datasets/${d.id}`} className="activity-item">
                  <span className="activity-icon">
                    <Icon name="database" size={17} />
                  </span>
                  <span className="activity-text">
                    <strong className="activity-title">{d.name}</strong>
                    <span className="activity-meta">
                      {d.latest_version
                        ? `v${d.latest_version.version} · ${d.latest_version.row_count} 行 × ${d.latest_version.column_count} 列`
                        : "等待首个版本"}
                    </span>
                  </span>
                  <span className="activity-time">{fmtTime(d.created_at)}</span>
                  <Icon className="activity-arrow" name="arrow-right" size={15} />
                </Link>
              ))}
            </div>
          </Panel>
        </>
      )}

      <SectionHeader title="全部数据集" description="支持按名称或描述搜索，表格可点击列头排序。" />
      <Panel>
        <div className="toolbar" style={{ marginBottom: "var(--space-4)" }}>
          <input type="text" placeholder="搜索数据集名称 / 描述..." value={keyword} onChange={(e) => setKeyword(e.target.value)} style={{ width: 280 }} />
          <label className="inline-check">
            <input type="checkbox" checked={onlyWithVersion} onChange={(e) => setOnlyWithVersion(e.target.checked)} />
            仅显示已有版本
          </label>
          <button className="btn" type="button" onClick={() => setSortRecent((v) => !v)}>{sortRecent ? "最近创建" : "原始顺序"}</button>
          <span className="toolbar-spacer toolbar-count">{filtered.length} 个数据集</span>
        </div>
        <DataTable columns={columns} rows={filtered} rowKey={(d) => d.id} loading={loading} error={error} emptyText="暂无数据集，点击右上角上传你的第一个数据集" />
      </Panel>

      {uploadOpen && (
        <div role="dialog" aria-modal="true" aria-label="上传数据集" style={{ position: "fixed", inset: 0, zIndex: 1000, background: "var(--c-scrim)", display: "flex", justifyContent: "flex-end" }} onMouseDown={(e) => { if (e.target === e.currentTarget && !uploading) { resetSelection(); setUploadOpen(false); } }}>
          <div style={{ width: "min(520px, 100%)", height: "100%", background: "var(--surface)", padding: "var(--space-6)", overflow: "auto", boxShadow: "var(--shadow-drawer)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}><div><h2 style={{ margin: 0 }}>上传数据</h2><div className="muted" style={{ marginTop: 5 }}>上传后自动创建第一个数据版本。</div></div><button className="btn" type="button" disabled={uploading} onClick={() => { resetSelection(); setUploadOpen(false); }}>关闭</button></div>
            <div className={`dropzone${dragOver ? " drag-over" : ""}${selectedFile ? " has-file" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={(e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragOver(false); pickFile(e.dataTransfer.files?.[0]); }} onClick={() => !uploading && fileRef.current?.click()} role="button" tabIndex={0}>
              <input ref={fileRef} type="file" accept={FILE_ALLOWED_EXTENSIONS.join(",")} hidden onChange={(e) => pickFile(e.target.files?.[0])} />
              {selectedFile ? <><strong>{selectedFile.name}</strong><div className="muted" style={{ marginTop: 5 }}>{formatFileSize(selectedFile.size)}</div></> : <><div className="dropzone-icon" aria-hidden>↑</div><strong>拖拽文件到这里</strong><div className="muted" style={{ marginTop: 5 }}>或点击选择 · {FILE_ALLOWED_EXTENSIONS.join(" / ").toUpperCase()}</div></>}
            </div>
            <div style={{ marginTop: "var(--space-4)" }}><label className="field">数据集名称<input type="text" placeholder="留空使用文件名" value={uploadName} onChange={(e) => setUploadName(e.target.value)} /></label></div>
            {(uploading || progress > 0) && <div style={{ marginTop: "var(--space-3)" }}><div className="progress-bar"><div style={{ width: `${progress}%` }} /></div><div className="muted" style={{ fontSize: 12, marginTop: "var(--space-1)" }}>{uploading ? `上传中… ${progress}%` : "处理中…"}</div></div>}
            {uploadError && <div className="alert alert-error" style={{ marginTop: "var(--space-3)" }}>{uploadError}</div>}
            <div style={{ display: "flex", gap: "var(--space-2)", marginTop: "var(--space-5)" }}><button className="btn btn-primary" disabled={uploading || !selectedFile} onClick={() => void handleUpload()}>{uploading ? "上传中…" : "上传并创建数据集"}</button><button className="btn" disabled={uploading} onClick={resetSelection}>重新选择</button></div>
          </div>
        </div>
      )}

      <ConfirmDialog open={pendingDelete !== null} title="删除数据集" message={`确定删除「${pendingDelete?.name}」？所有版本快照将一并清理，不可恢复。`} confirmText="删除" danger onCancel={() => setPendingDelete(null)} onConfirm={() => { if (pendingDelete) void deleteDataset(pendingDelete.id).then(() => { setPendingDelete(null); setReloadKey((k) => k + 1); }); }} />
    </div>
  );
}

export function fmtTime(value: string | null | undefined): string { if (!value) return "-"; return value.replace("T", " ").slice(0, 19); }
