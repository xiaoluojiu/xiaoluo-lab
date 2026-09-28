import { useEffect, useState } from "react";
import { PageHeader } from "../../components/PageHeader";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { MergePanel } from "../../features/merge/MergePanel";
import { MultiMergePanel } from "../../features/merge/MultiMergePanel";
import { OperationHistory } from "../../features/processing/OperationHistory";
import { OperationPanel } from "../../features/processing/OperationPanel";
import { TABS, type FormTab, type Tab } from "../../features/processing/operationCatalog";
import { getSchema } from "../../api/datasets";
import type { SchemaColumn } from "../../types/dataset";
import { listOperations, OPERATION_KIND_MAP, type OperationMeta } from "../../api/processing";

/**
 * 数据处理页 —— 只做「路由级布局 + 数据初始化」。
 *
 * 职责边界（本轮归并后）：
 * - 页面：选数据集 / 输入版本、切操作 Tab、拉操作目录与 schema、装配面板；
 * - `features/processing/`：操作面板与其表单、校验、dry-run、执行、处理记录；
 * - `features/merge/`：合并与多文件合并面板。
 *
 * 归并前这里内联了 OperationPanel、9 个操作表单、预览/结果视图、处理记录与
 * TABS/DEFAULT_FORMS/FILTER_OPS 等一整套业务逻辑（516 行），页面既是路由壳又是业务模块。
 * 本次只做搬移与去重，UI 与请求行为均未改动。
 */
export default function Processing() {
  const [tab, setTab] = useState<Tab>("clean");
  const [datasetIds, setDatasetIds] = useState<number[]>([]);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [inputVersion, setInputVersion] = useState<number | undefined>();
  const [loadingOperations, setLoadingOperations] = useState(true);
  const [operationsError, setOperationsError] = useState<string | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [operations, setOperations] = useState<OperationMeta[]>([]);
  const selectedDatasetId = datasetIds[0];

  useEffect(() => { if (!selectedDatasetId) { setColumns([]); return; } getSchema(selectedDatasetId).then((s) => setColumns(s.columns)).catch(() => setColumns([])); }, [selectedDatasetId]);
  useEffect(() => { let cancelled = false; async function loadOperations() { setLoadingOperations(true); setOperationsError(null); try { setOperations(await listOperations()); } catch (error) { if (!cancelled) setOperationsError(error instanceof Error ? error.message : "加载操作目录失败"); } finally { if (!cancelled) setLoadingOperations(false); } } void loadOperations(); return () => { cancelled = true; }; }, []);
  useEffect(() => { setInputVersion(undefined); }, [datasetIds]);

  const tabGroups = Array.from(new Set(TABS.map((item) => item.group)));

  return (
    <div>
      <PageHeader
        breadcrumbs={<>数据中心 / <b>数据处理</b></>}
        title="数据处理"
        description="对数据集执行清洗、转换与聚合，每次执行都会生成一个可回溯的新版本。"
      />
      <div className="processing-workspace">
        <div className="processing-toolbar">
          <div className="processing-toolbar-main"><span>数据集</span><DatasetSelector value={datasetIds} onChange={setDatasetIds} multi={false} requireVersion /></div>
          {selectedDatasetId && <label className="field processing-toolbar-version">输入版本<input type="number" min={1} value={inputVersion ?? ""} placeholder="留空 = 最新版本" onChange={(event) => { const value = event.target.value; if (!value) { setInputVersion(undefined); return; } const parsed = Number(value); if (Number.isInteger(parsed) && parsed > 0) setInputVersion(parsed); }} /></label>}
        </div>
        <div className="processing-tabs">{tabGroups.map((group) => <div className="processing-tabs-group" key={group}><span className="processing-tabs-label">{group}</span><div className="processing-tabs-items">{TABS.filter((item) => item.group === group).map((item) => <button key={item.key} type="button" title={item.description} className={`processing-tab${tab === item.key ? " active" : ""}`} onClick={() => setTab(item.key)}>{item.label}</button>)}</div></div>)}</div>
        <div className="processing-body">
          {loadingOperations && <div className="muted">正在加载操作…</div>}
          {operationsError && <div style={{ color: "var(--danger)" }}>操作目录加载失败：{operationsError}</div>}
          {!loadingOperations && !operationsError && (tab === "merge" ? <MergePanel /> : tab === "multi_merge" ? <MultiMergePanel /> : <OperationPanel key={tab} tab={tab as FormTab} meta={operations.find((item) => item.op_type === OPERATION_KIND_MAP[tab])} datasetId={selectedDatasetId} columns={columns} inputVersion={inputVersion} onCompleted={() => setHistoryRefresh((value) => value + 1)} />)}
        </div>
      </div>
      {selectedDatasetId && <OperationHistory key={historyRefresh} datasetId={selectedDatasetId} />}
    </div>
  );
}
