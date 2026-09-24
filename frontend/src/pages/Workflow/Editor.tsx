/**
 * 沉浸式工作流编辑器（独立路由 /workflow/editor/:id）。
 *
 * 与 /workflow 列表页的分工：
 * - /workflow 仍然是「列表 + 入口」，保留原有布局与交互，行为不变；
 * - 本页只负责「编辑当前某个工作流」，因此把不属于编辑的元素收起来：
 *   全局左侧导航与顶部 Topbar 由路由层面绕过（本页挂在 MainLayout 之外），
 *   执行记录 / 流程检查降级为按需弹出的右侧抽屉，
 *   模板与智能建议收进空状态，标题面包屑让位给画布。
 *
 * 注意：这里是布局与交互层重构，不新增后端能力，也不改协议；
 * 保存与运行仍走 `api/workflow.ts` 既有接口。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Icon } from "../../components/icons/Icon";
import { DatasetSelector } from "../../features/merge/DatasetSelector";
import { WorkflowHealthPanel } from "../../features/workflow/WorkflowHealthPanel";
import { WorkflowRunPanel } from "../../features/workflow/WorkflowRunPanel";
import { WorkflowStudio, nodeSummary } from "../../features/workflow/WorkflowStudio";
import { applyTemplate, generateSuggestedWorkflow, WORKFLOW_TEMPLATES } from "../../features/workflow/WorkflowCanvas";
import { runPreflight } from "../../features/workflow/configView";
import { metricsFromOutputs } from "../../features/workflow/nodeStatus";
import {
  clearDraft,
  formatSavedAt,
  readDraft,
  useDraftAutosave,
  useEditHistory,
} from "../../features/workflow/editorHistory";
import { getSchema, listDatasets } from "../../api/datasets";
import {
  cancelWorkflowRun,
  createWorkflow,
  getWorkflow,
  listWorkflows,
  runWorkflow,
  updateWorkflow,
} from "../../api/workflow";
import type { SchemaColumn } from "../../types/dataset";
import type { WorkflowEdge, WorkflowNode, WorkflowRun, WorkflowSummary } from "../../types/workflow";
// 沉浸页用自足样式，不复用 workflow.css（那张表里藏了会把三栏压成手机版的媒体查询）。
import "./workflow-editor.css";

/** :id 为 "new" 时表示还未落库的草稿。 */
const NEW_WORKFLOW = "new";

type DrawerTab = "execution" | "health";

export default function WorkflowEditorPage() {
  const { id: routeId = NEW_WORKFLOW } = useParams();
  const navigate = useNavigate();

  const history = useEditHistory({ nodes: [], edges: [] });
  const [name, setName] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<number | null>(null);
  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  /** 每个 data.load 节点选中的数据集 → schema，供下游节点做列过滤。 */
  const [datasetOptions, setDatasetOptions] = useState<{ id: number; name: string }[]>([]);
  const [explorerOpen, setExplorerOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [drawer, setDrawer] = useState<DrawerTab | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [list, setList] = useState<WorkflowSummary[]>([]);
  const [dirty, setDirty] = useState(false);
  const [loaded, setLoaded] = useState(false);
  /* 快速起步（渲染在右侧参数面板，画布为空时）：模板 + 数据集 + 生成按钮 */
  const [quickTemplateId, setQuickTemplateId] = useState("");
  const [quickDatasetId, setQuickDatasetId] = useState<number[]>([]);

  const { nodes, edges, commit, reset } = history;

  /* ---------------------------------------------------------------- */
  /* 加载                                                              */
  /* ---------------------------------------------------------------- */

  const loadWorkflow = useCallback(
    async (target: string) => {
      // 切换目标先作废上一次的已保存指纹，否则新流程会被判成「有未保存修改」。
      setSavedSignature(null);
      promptedRef.current = false;
      if (target === NEW_WORKFLOW) {
        reset({ nodes: [], edges: [] });
        setName("");
        setSavedId(null);
        setRun(null);
        setSelectedNodeId(null);
        setLoaded(true);
        return;
      }
      const numeric = Number(target);
      if (!Number.isFinite(numeric)) {
        setError(`无法识别的流程 id：${target}`);
        setLoaded(true);
        return;
      }
      try {
        const workflow = await getWorkflow(numeric);
        reset({ nodes: workflow.nodes, edges: workflow.edges });
        setName(workflow.name);
        setSavedId(workflow.id ?? numeric);
        setRun(null);
        setSelectedNodeId(null);
        setDirty(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : "加载工作流失败");
      } finally {
        setLoaded(true);
      }
    },
    [reset],
  );

  useEffect(() => {
    setLoaded(false);
    void loadWorkflow(routeId);
  }, [routeId, loadWorkflow]);

  useEffect(() => {
    void listWorkflows()
      .then(setList)
      .catch(() => setList([]));
  }, []);

  /* ---------------------------------------------------------------- */
  /* 数据集选项：data.load 节点的 dataset_id 下拉候选                   */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    let cancelled = false;
    void listDatasets(1, 200)
      .then((result) => {
        if (cancelled) return;
        setDatasetOptions(
          result.items
            .filter((item) => item.id != null)
            .map((item) => ({ id: item.id, name: item.name })),
        );
      })
      // 拉不到列表不影响编辑：页面上 DatasetSelector 与手工填 id 仍可用。
      .catch(() => {
        if (!cancelled) setDatasetOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /* ---------------------------------------------------------------- */
  /* 草稿自动保存：只在有内容时落盘，避免空画布覆盖掉已保存的版本      */
  /* ---------------------------------------------------------------- */

  const draftScope = routeId === NEW_WORKFLOW ? "new" : `wf:${routeId}`;
  const draftPayload = useMemo(() => ({ name, nodes, edges }), [name, nodes, edges]);
  const savedAt = useDraftAutosave(loaded ? draftScope : null, draftPayload, 900);

  /** 首次进入「新建」时，如果本地有未保存草稿，提示恢复。 */
  const [draftPrompt, setDraftPrompt] = useState<{ name: string; nodes: WorkflowNode[]; edges: WorkflowEdge[]; savedAt: number } | null>(null);
  const promptedRef = useRef(false);
  useEffect(() => {
    if (!loaded || routeId !== NEW_WORKFLOW || promptedRef.current) return;
    promptedRef.current = true;
    const cached = readDraft("new");
    if (cached && (cached.nodes.length || cached.name.trim())) setDraftPrompt(cached);
  }, [loaded, routeId]);

  function restoreDraft() {
    if (!draftPrompt) return;
    reset({ nodes: draftPrompt.nodes, edges: draftPrompt.edges });
    setName(draftPrompt.name);
    setDraftPrompt(null);
    setDirty(true);
  }

  function discardDraft() {
    clearDraft("new");
    setDraftPrompt(null);
  }

  /**
   * 「已保存」判定：内容指纹与上一次成功保存时一致。
   *
   * 用 state 而不是 ref：ref 变化不触发渲染，加载完成的那一帧会先把
   * 新流程判成「有未保存修改」，再被 effect 纠正 —— 顶栏会闪一下黄字。
   */
  const [savedSignature, setSavedSignature] = useState<string | null>(null);
  const signature = useMemo(
    () => JSON.stringify({ name: name.trim(), nodes, edges }),
    [name, nodes, edges],
  );
  useEffect(() => {
    if (loaded && savedSignature === null) setSavedSignature(signature);
  }, [loaded, savedSignature, signature]);
  const hasUnsaved = savedId == null || savedSignature !== signature;

  /* ---------------------------------------------------------------- */
  /* 数据集 schema：取第一个 data.load 节点上挂的数据集字段            */
  /* ---------------------------------------------------------------- */

  const loadNodeDatasetId = useMemo(() => {
    const node = nodes.find((item) => item.type === "data.load");
    const config = node?.config as { dataset_id?: unknown; params?: { dataset_id?: unknown } } | undefined;
    const raw = config?.dataset_id ?? config?.params?.dataset_id;
    const numeric = typeof raw === "string" ? Number(raw) : raw;
    return typeof numeric === "number" && Number.isFinite(numeric) ? numeric : null;
  }, [nodes]);

  useEffect(() => {
    if (loadNodeDatasetId == null) {
      setColumns([]);
      return undefined;
    }
    let cancelled = false;
    void getSchema(loadNodeDatasetId)
      .then((schema) => {
        if (cancelled) return;
        setColumns(schema.columns);
      })
      .catch(() => {
        if (!cancelled) setColumns([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loadNodeDatasetId]);

  /* ---------------------------------------------------------------- */
  /* 画布回调                                                          */
  /* ---------------------------------------------------------------- */

  const onCanvasChange = useCallback(
    (nextNodes: WorkflowNode[], nextEdges: WorkflowEdge[], coalesceKey?: string) => {
      commit({ nodes: nextNodes, edges: nextEdges }, coalesceKey);
      setDirty(true);
    },
    [commit],
  );

  function applyTemplateById(templateId: string, datasetId: number | null) {
    const template = WORKFLOW_TEMPLATES.find((item) => item.id === templateId);
    if (!template) {
      setError("未找到所选模板");
      return;
    }
    const filled = applyTemplate(template, datasetId);
    commit({ nodes: filled.nodes, edges: filled.edges });
    if (!name.trim()) setName(template.name);
    setSelectedNodeId(null);
    setRun(null);
    setColumns([]);
    setDirty(true);
  }

  async function applySmartSuggest(datasetId: number) {
    try {
      const schema = await getSchema(datasetId);
      setColumns(schema.columns);
      const suggested = generateSuggestedWorkflow(schema.columns.map((column) => column.column));
      commit({
        nodes: suggested.nodes.map((node) =>
          node.type === "data.load" ? { ...node, config: { ...node.config, dataset_id: datasetId } } : node,
        ),
        edges: suggested.edges,
      });
      if (!name.trim()) setName("数据处理工作流");
      setSelectedNodeId(null);
      setRun(null);
      setDirty(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "读取数据集 Schema 失败");
    }
  }

  /** 快速起步（右侧面板）：选了模板按模板填充，否则按数据集字段智能生成。 */
  async function handleQuickStart() {
    const datasetId = quickDatasetId[0] ?? null;
    if (quickTemplateId) {
      applyTemplateById(quickTemplateId, datasetId);
    } else if (datasetId != null) {
      await applySmartSuggest(datasetId);
    }
    setQuickTemplateId("");
    setQuickDatasetId([]);
  }

  /* ---------------------------------------------------------------- */
  /* 保存 / 运行                                                       */
  /* ---------------------------------------------------------------- */

  async function save(): Promise<number | null> {
    if (!name.trim()) {
      setError("请先给工作流起一个名字。");
      return null;
    }
    setBusy(true);
    try {
      let targetId = savedId;
      if (targetId != null) {
        await updateWorkflow(targetId, { name: name.trim(), nodes, edges });
      } else {
        const created = await createWorkflow({ name: name.trim(), nodes, edges });
        targetId = created.id ?? null;
        if (targetId != null) {
          setSavedId(targetId);
          clearDraft("new");
          // 路由换成已落库的 id，刷新后仍指向同一份流程。
          navigate(`/workflow/editor/${targetId}`, { replace: true });
        }
      }
      setSavedSignature(signature);
      setDirty(false);
      setList(await listWorkflows().catch(() => list));
      return targetId;
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function doRun() {
    const blocker = runPreflight(name, nodes);
    if (blocker) {
      setError(blocker.reason);
      if (blocker.focusNodeId) {
        setSelectedNodeId(blocker.focusNodeId);
        setInspectorOpen(true);
      }
      return;
    }
    let targetId = savedId;
    if (hasUnsaved || targetId == null) targetId = await save();
    if (targetId == null) return;
    setBusy(true);
    setError(null);
    try {
      const nextRun = await runWorkflow(targetId);
      setRun(nextRun);
      setDrawer("execution");
    } catch (e) {
      setError(e instanceof Error ? e.message : "运行失败");
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun() {
    if (!run?.run_id) return;
    try {
      await cancelWorkflowRun(run.run_id);
      setRun((current) => (current ? { ...current, status: "cancelled" } : current));
    } catch (e) {
      setError(e instanceof Error ? e.message : "取消运行失败");
    }
  }

  /* ---------------------------------------------------------------- */
  /* 快捷键：Cmd/Ctrl+S 保存，Cmd/Ctrl+Z 撤销，Shift+Cmd/Ctrl+Z 重做    */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const meta = event.metaKey || event.ctrlKey;
      if (!meta) return;
      if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save();
      } else if (event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) history.redo();
        else history.undo();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  useEffect(() => {
    if (!dirty) return undefined;
    function warn(event: BeforeUnloadEvent) {
      event.preventDefault();
      event.returnValue = "";
    }
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  /* ---------------------------------------------------------------- */
  /* 派生信息                                                          */
  /* ---------------------------------------------------------------- */

  const nodeStates = run?.result?.node_states ?? {};
  // 节点 / 连线上的「多少行 × 多少列」标注：直接从运行产物派生，不落 state，
  // 免得 run 变了还要记得同步刷新（这正是副本 state 最容易漏的地方）。
  const nodeMetrics = useMemo(() => metricsFromOutputs(run?.result?.outputs), [run]);
  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const saveLabel = savedId == null ? "未保存" : hasUnsaved ? "有未保存修改" : "已保存";
  const draftHint = savedAt ? `本地草稿 ${formatSavedAt(savedAt)}` : "";

  return (
    <div className="wf-editor">
      {/* 顶栏：只保留「离开 / 命名 / 保存 / 运行」这条主线 */}
      <header className="wf-editor-topbar">
        <div className="wf-editor-topbar-left">
          <button
            className="wf-editor-back"
            type="button"
            onClick={() => navigate("/workflow")}
            title="返回工作流列表"
          >
            <Icon name="arrow-right" size={16} />
            <span>流程列表</span>
          </button>
          <input
            className="wf-editor-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="未命名工作流"
            aria-label="工作流名称"
          />
          <span className={`wf-editor-save-state ${hasUnsaved ? "dirty" : ""}`}>
            {saveLabel}
            {draftHint && <em>{draftHint}</em>}
          </span>
        </div>

        <div className="wf-editor-topbar-right">
          <div className="wf-editor-history">
            <button
              type="button"
              className="wf-editor-icon-btn"
              onClick={history.undo}
              disabled={!history.canUndo}
              title="撤销（Ctrl/Cmd+Z）"
              aria-label="撤销"
            >
              <Icon name="refresh" size={16} />
            </button>
            <button
              type="button"
              className="wf-editor-icon-btn"
              onClick={history.redo}
              disabled={!history.canRedo}
              title="重做（Ctrl/Cmd+Shift+Z）"
              aria-label="重做"
            >
              <Icon name="refresh" size={16} />
            </button>
          </div>
          <button
            type="button"
            className={`wf-editor-icon-btn ${libraryOpen ? "active" : ""}`}
            onClick={() => setLibraryOpen((open) => !open)}
            title="切换工作流"
            aria-label="切换工作流"
          >
            <Icon name="layers" size={16} />
          </button>
          <button
            type="button"
            className={`wf-editor-icon-btn ${drawer === "execution" ? "active" : ""}`}
            onClick={() => setDrawer((current) => (current === "execution" ? null : "execution"))}
            title="执行记录"
            aria-label="执行记录"
          >
            <Icon name="clock" size={16} />
            {run && <i className={`wf-editor-run-dot ${run.status}`} />}
          </button>
          <button
            type="button"
            className={`wf-editor-icon-btn ${drawer === "health" ? "active" : ""}`}
            onClick={() => setDrawer((current) => (current === "health" ? null : "health"))}
            title="流程检查"
            aria-label="流程检查"
          >
            <Icon name="check" size={16} />
          </button>
          <button className="btn" type="button" disabled={busy} onClick={() => void save()}>
            保存
          </button>
          <button className="btn primary" type="button" disabled={busy} onClick={() => void doRun()}>
            ▶ 运行
          </button>
        </div>
      </header>

      {/* 主体：画布为主，左右两侧面板可收起 */}
      <div className="wf-editor-body">
        <WorkflowStudio
          nodes={nodes}
          edges={edges}
          nodeStates={nodeStates}
          selectedNodeId={selectedNodeId}
          onSelectNode={(id) => {
            setSelectedNodeId(id);
            if (id) setInspectorOpen(true);
          }}
          onChange={onCanvasChange}
          columns={columns}
          nodeMetrics={nodeMetrics}
          datasetOptions={datasetOptions}
          explorerOpen={explorerOpen}
          onCloseExplorer={() => setExplorerOpen(false)}
          inspectorOpen={inspectorOpen}
          onCloseInspector={() => setInspectorOpen(false)}
          surface="editor"
          emptyStateSlot={
            <div className="wf-editor-empty">
              <strong>画布还是空的</strong>
              <span>在右侧面板选一个模板或数据集开始，或双击画布空白处添加节点。</span>
            </div>
          }
          inspectorEmptySlot={
            nodes.length === 0 ? (
              <div className="wf-editor-quickstart">
                <div className="wf-editor-quickstart-head">
                  <strong>从这里开始</strong>
                  <p>选一个预置模板快速搭出流程，或按数据集字段自动生成。</p>
                </div>
                <label className="wf-editor-quickstart-field">
                  <span>预置模板</span>
                  <select
                    value={quickTemplateId}
                    onChange={(event) => setQuickTemplateId(event.target.value)}
                    aria-label="选择预置模板"
                  >
                    <option value="">不使用模板（智能建议）</option>
                    {WORKFLOW_TEMPLATES.map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.category} · {template.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="wf-editor-quickstart-field">
                  <span>数据集</span>
                  <DatasetSelector value={quickDatasetId} onChange={setQuickDatasetId} multi={false} compact />
                </div>
                <button
                  className="btn primary"
                  type="button"
                  disabled={(!quickTemplateId && !quickDatasetId.length) || busy}
                  onClick={() => void handleQuickStart()}
                >
                  {quickTemplateId ? "用模板生成" : "生成流程"}
                </button>
                <p className="wf-editor-quickstart-hint">
                  生成后可继续在画布上添加节点、连线与配置参数。
                </p>
              </div>
            ) : undefined
          }
        />

        {!explorerOpen && (
          <button
            type="button"
            className="wf-editor-rail left"
            onClick={() => setExplorerOpen(true)}
            title="展开节点库"
          >
            <Icon name="layers" size={16} />
            <span>节点库</span>
          </button>
        )}
        {!inspectorOpen && (
          <button
            type="button"
            className="wf-editor-rail right"
            onClick={() => setInspectorOpen(true)}
            title="展开参数面板"
          >
            <Icon name="sliders" size={16} />
            <span>参数</span>
          </button>
        )}

        {selectedNode && !inspectorOpen && (
          <div className="wf-editor-inspector-hint">
            <span>{nodeSummary(selectedNode, columns)}</span>
            <button type="button" className="btn" onClick={() => setInspectorOpen(true)}>
              配置
            </button>
          </div>
        )}
      </div>

      {/* 执行 / 体检抽屉：默认收起，按需出现 */}
      {drawer && (
        <aside className="wf-editor-drawer">
          <div className="wf-editor-drawer-head">
            <div className="wf-editor-drawer-tabs">
              <button
                className={drawer === "execution" ? "active" : ""}
                type="button"
                onClick={() => setDrawer("execution")}
              >
                执行 {run ? `· ${run.status}` : ""}
              </button>
              <button
                className={drawer === "health" ? "active" : ""}
                type="button"
                onClick={() => setDrawer("health")}
              >
                流程检查
              </button>
            </div>
            <button type="button" className="wf-editor-icon-btn" onClick={() => setDrawer(null)} aria-label="关闭">
              <Icon name="trash" size={15} />
            </button>
          </div>
          <div className="wf-editor-drawer-body">
            {drawer === "execution" ? (
              <WorkflowRunPanel run={run} onCancel={() => void cancelRun()} onSelectNode={setSelectedNodeId} />
            ) : (
              <WorkflowHealthPanel
                nodes={nodes}
                edges={edges}
                columns={columns}
                run={run}
                onSelectNode={setSelectedNodeId}
              />
            )}
          </div>
        </aside>
      )}

      {/* 工作流切换：从列表页的常驻抽屉改为按需浮层 */}
      {libraryOpen && (
        <div className="wf-editor-library">
          <div className="wf-editor-library-overlay" onClick={() => setLibraryOpen(false)} />
          <div className="wf-editor-library-panel">
            <div className="wf-editor-library-head">
              <strong>切换工作流</strong>
              <button className="btn" type="button" onClick={() => navigate("/workflow/editor/new")}>
                ＋ 新建
              </button>
            </div>
            {!list.length ? (
              <div className="wf-editor-library-empty">还没有工作流。</div>
            ) : (
              <div className="wf-editor-library-list">
                {list.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`wf-editor-library-row ${String(item.id) === routeId ? "active" : ""}`}
                    onClick={() => {
                      setLibraryOpen(false);
                      navigate(`/workflow/editor/${item.id}`);
                    }}
                  >
                    <strong>{item.name}</strong>
                    <span>
                      #{item.id} · {item.nodes} 节点 · {item.edges} 连接
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {draftPrompt && (
        <div className="wf-editor-prompt" role="dialog">
          <p>
            发现 {formatSavedAt(draftPrompt.savedAt)} 的本地草稿
            {draftPrompt.name ? `「${draftPrompt.name}」` : ""}，要恢复吗？
          </p>
          <div className="wf-editor-prompt-actions">
            <button className="btn" type="button" onClick={discardDraft}>
              丢弃
            </button>
            <button className="btn primary" type="button" onClick={restoreDraft}>
              恢复
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="wf-editor-error" role="alert">
          {error}
          <button type="button" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}
    </div>
  );
}

/** 供外部（如列表页的「沉浸编辑」入口）拼接链接用。 */
export function editorPath(id: number | "new") {
  return `/workflow/editor/${id}`;
}
