import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { WorkflowStudio } from "../../features/workflow/WorkflowStudio";
import { WorkflowRunPanel } from "../../features/workflow/WorkflowRunPanel";
import { WorkflowHealthPanel } from "../../features/workflow/WorkflowHealthPanel";
import { metricsFromOutputs } from "../../features/workflow/nodeStatus";
import { generateSuggestedWorkflow, WORKFLOW_TEMPLATES, applyTemplate } from "../../features/workflow/WorkflowCanvas";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { getSchema } from "../../api/datasets";
import { cancelWorkflowRun, cloneWorkflow, createWorkflow, deleteWorkflow, getWorkflow, listWorkflows, runWorkflow, updateWorkflow } from "../../api/workflow";
import type { WorkflowEdge, WorkflowNode, WorkflowRun, WorkflowSummary } from "../../types/workflow";
import type { SchemaColumn } from "../../types/dataset";
import { PageHeader } from "../../components/PageHeader";
import "./workflow.css";

export default function WorkflowPage() {
  const navigate = useNavigate();
  const [list, setList] = useState<WorkflowSummary[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<WorkflowEdge[]>([]);
  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [suggestDatasetId, setSuggestDatasetId] = useState<number[]>([]);
  const [showLibrary, setShowLibrary] = useState(false);
  const [bottomTab, setBottomTab] = useState<"execution" | "health">("execution");
  const [templateId, setTemplateId] = useState<string>("");

  async function refresh() { try { setList(await listWorkflows()); } catch (e) { setError(e instanceof Error ? e.message : "加载工作流失败"); } }
  useEffect(() => { void refresh(); }, []);
  async function loadOne(id: number) { try { const w = await getWorkflow(id); setCurrentId(id); setName(w.name); setNodes(w.nodes); setEdges(w.edges); setRun(null); setSelectedNodeId(null); setShowLibrary(false); } catch (e) { setError(e instanceof Error ? e.message : "加载失败"); } }
  function newWorkflow() { setCurrentId(null); setName(""); setNodes([]); setEdges([]); setRun(null); setSelectedNodeId(null); setColumns([]); setSuggestDatasetId([]); setError(null); setShowLibrary(false); }
  // 编辑动作统一交给沉浸式编辑器，列表页只保留「浏览 / 快速预览」职责。
  function openEditor(id: number | "new") { setShowLibrary(false); navigate(`/workflow/editor/${id}`); }
  async function save() { if (!name.trim()) { setError("请填写工作流名称"); return; } setBusy(true); try { if (currentId != null) await updateWorkflow(currentId, { name: name.trim(), nodes, edges }); else { const w = await createWorkflow({ name: name.trim(), nodes, edges }); setCurrentId(w.id ?? null); } await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "保存失败"); } finally { setBusy(false); } }
  async function doRun() { if (currentId == null) return; setBusy(true); setError(null); try { setRun(await runWorkflow(currentId)); setBottomTab("execution"); } catch (e) { setError(e instanceof Error ? e.message : "运行失败"); } finally { setBusy(false); } }
  async function cancelRun() { if (!run?.run_id) return; try { await cancelWorkflowRun(run.run_id); setRun((current) => current ? { ...current, status: "cancelled" } : current); } catch (e) { setError(e instanceof Error ? e.message : "取消运行失败"); } }
  async function remove(id: number) { if (!confirm(`确认删除工作流 #${id}？`)) return; try { await deleteWorkflow(id); if (currentId === id) newWorkflow(); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "删除失败"); } }
  async function clone(id: number) { try { await cloneWorkflow(id); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "克隆失败"); } }
  async function suggest() {
    const id = suggestDatasetId[0];
    // 选了预置模板：直接按模板填充（数据集可选，有则注入 data.load）。
    if (templateId) {
      const template = WORKFLOW_TEMPLATES.find((item) => item.id === templateId);
      if (!template) { setError("未找到所选模板"); return; }
      const filled = applyTemplate(template, id ?? null);
      setNodes(filled.nodes);
      setEdges(filled.edges);
      setColumns([]);
      if (!name.trim()) setName(template.name);
      setSelectedNodeId(null);
      setRun(null);
      return;
    }
    // 未选模板：沿用智能建议（按 schema 推断 load + clean）。
    if (!id) return;
    try {
      const schema = await getSchema(id);
      setColumns(schema.columns);
      const suggested = generateSuggestedWorkflow(schema.columns.map((column) => column.column));
      setNodes(suggested.nodes.map((node) => node.type === "data.load" ? { ...node, config: { ...node.config, dataset_id: id } } : node));
      setEdges(suggested.edges);
      if (!name.trim()) setName("数据处理工作流");
      setSelectedNodeId(null);
      setRun(null);
    } catch (e) { setError(e instanceof Error ? e.message : "读取数据集 Schema 失败"); }
  }

  const nodeStates = run?.result?.node_states ?? {};
  // 节点卡片 / 连线上的规模标注：与沉浸式编辑器共用同一派生口径。
  const nodeMetrics = metricsFromOutputs(run?.result?.outputs);

  return <div className="workflow-page">
    <PageHeader
      title="Workflow"
      description="可视化编排数据处理、分析、机器学习与 AI 节点，保存并运行端到端流程。"
      actions={<>
        <button className="btn" type="button" onClick={() => setShowLibrary(true)}>工作流 <span className="workflow-count">{list.length}</span></button>
        <button className="btn" type="button" onClick={() => openEditor("new")}>＋ 新建</button>
      </>}
    />

    {showLibrary && <div className="workflow-library-popover"><div className="workflow-library-overlay" onClick={() => setShowLibrary(false)} /><section className="workflow-library-drawer"><div className="workflow-drawer-head"><div><h3>我的工作流</h3><p>选择一个流程继续编辑，或从已有流程克隆。</p></div><button className="btn" type="button" onClick={() => setShowLibrary(false)}>×</button></div>{!list.length ? <div className="workflow-library-empty">还没有工作流。点击右上角“新建”开始。</div> : <div className="workflow-library-list">{list.map((item) => <div className={`workflow-library-row ${item.id === currentId ? "active" : ""}`} key={item.id}><button type="button" className="workflow-library-main" onClick={() => void loadOne(item.id)}><strong>{item.name}</strong><span>#{item.id} · {item.nodes} 节点 · {item.edges} 连接</span></button><div className="workflow-row-actions"><button className="btn primary" type="button" onClick={() => openEditor(item.id)}>沉浸编辑</button><button className="btn" type="button" onClick={() => void loadOne(item.id)}>预览</button><button className="btn" type="button" onClick={() => void clone(item.id)}>克隆</button><button className="btn danger" type="button" onClick={() => void remove(item.id)}>删除</button></div></div>)}</div>}</section></div>}

    <section className="workflow-studio-card card">
      <div className="workflow-studio-header">
        <div className="workflow-title-area"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="未命名工作流" aria-label="工作流名称" /><span className={`badge ${currentId != null ? "success" : ""}`}>{currentId != null ? `#${currentId} 已保存` : "草稿"}</span></div>
        <div className="workflow-studio-actions"><button className="btn" type="button" disabled={busy} onClick={() => void save()}>保存</button><button className="btn primary" type="button" disabled={busy || currentId == null} onClick={() => void doRun()}>▶ 运行</button></div>
      </div>
      <div className="workflow-suggest-strip">
        <div className="workflow-suggest-copy"><strong>快速起步</strong><span>选一个预置模板快速搭出流程，或直接生成智能建议</span></div>
        <select className="workflow-template-select" value={templateId} onChange={(event) => setTemplateId(event.target.value)} aria-label="选择预置模板">
          <option value="">智能建议（按字段推断）</option>
          {WORKFLOW_TEMPLATES.map((tpl) => <option key={tpl.id} value={tpl.id}>{tpl.category} · {tpl.name}</option>)}
        </select>
        <DatasetSelector value={suggestDatasetId} onChange={setSuggestDatasetId} multi={false} compact />
        <button className="btn" type="button" disabled={(!templateId && !suggestDatasetId.length) || busy} onClick={() => void suggest()}>{templateId ? "用模板生成" : "生成流程"}</button>
        {columns.length > 0 && <span className="muted">Schema {columns.length} 字段</span>}
      </div>
      <WorkflowStudio nodes={nodes} edges={edges} nodeStates={nodeStates} selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} onChange={(nextNodes, nextEdges) => { setNodes(nextNodes); setEdges(nextEdges); }} columns={columns} nodeMetrics={nodeMetrics} />
      <div className="workflow-bottom-panel">
        <div className="workflow-bottom-tabs"><button className={bottomTab === "execution" ? "active" : ""} type="button" onClick={() => setBottomTab("execution")}>执行 {run ? `· ${run.status}` : ""}</button><button className={bottomTab === "health" ? "active" : ""} type="button" onClick={() => setBottomTab("health")}>流程检查</button></div>
        <div className="workflow-bottom-content">{bottomTab === "execution" ? <WorkflowRunPanel run={run} onCancel={() => void cancelRun()} onSelectNode={setSelectedNodeId} /> : <WorkflowHealthPanel nodes={nodes} edges={edges} columns={columns} run={run} onSelectNode={setSelectedNodeId} />}</div>
      </div>
    </section>
    {error && <div className="workflow-error" role="alert">{error}<button type="button" onClick={() => setError(null)}>×</button></div>}
  </div>;
}
