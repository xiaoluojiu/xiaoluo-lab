import { useEffect, useMemo, useRef, useState } from "react";

import { Icon } from "../../components/icons/Icon";
import type { SchemaColumn } from "../../types/dataset";
import type { WorkflowEdge, WorkflowNode } from "../../types/workflow";
import { flattenConfig } from "./configView";
import { NodeParamForm } from "./NodeParamForm";
import { nodeDisplay } from "./nodeStatus";
import {
  NODE_LABELS,
  NODE_TYPES,
  categoryOf,
  specOf,
  type NodeSpec,
} from "./nodeSpecs";

type Position = { x: number; y: number };
type DragState = { id: string; sx: number; sy: number; origin: Position };
type PanState = { x: number; y: number; px: number; py: number };

/**
 * 世界坐标系尺寸。原为 3000×1800 固定值，节点一旦拖到边界就「顶住不动」，
 * 用户必须反过来拖动画布才能继续摆放，感受上像画布移动范围受限。
 * 现在：世界足够大（CANVAS_BASE），并且只要已有节点靠近边缘就会自动向外扩张
 * （见 canvasSize / CANVAS_STEP），节点坐标同时允许负值，因此不会提前撞墙。
 */
const CANVAS_BASE = { width: 3000, height: 1800 };
/** 节点逼近边缘时世界向外扩张的步长。 */
const CANVAS_STEP = 1200;
/** 世界坐标允许的绘制起点：负方向预留 CANVAS_MIN 的余量，工作区用 margin 平移对齐节点 0 点。 */
const CANVAS_MIN = -1800;
/** 节点坐标下限：允许拖到负坐标，避免所有节点被挤在原点一角。 */
const NODE_MIN = -1600;
const NODE_W = 250;
const NODE_H = 118;
const DEFAULT_GAP_X = 320;
const DEFAULT_GAP_Y = 170;

function position(node: WorkflowNode, index: number): Position {
  const ui = node.config?.__ui;
  if (ui && typeof ui === "object") {
    const x = Number((ui as Record<string, unknown>).x);
    const y = Number((ui as Record<string, unknown>).y);
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }
  return { x: 100 + (index % 4) * DEFAULT_GAP_X, y: 110 + Math.floor(index / 4) * DEFAULT_GAP_Y };
}

/** 新节点默认落在视口中心；该处已有节点时向右下错位，避免叠在一起。 */
function freeSpot(base: Position, taken: Position[]): Position {
  let p = base;
  for (let i = 0; i < 24 && taken.some((q) => Math.abs(q.x - p.x) < NODE_W && Math.abs(q.y - p.y) < NODE_H); i += 1) {
    p = { x: p.x + 56, y: p.y + 44 };
  }
  return p;
}

/** 节点卡片上的一行摘要：只说「这个节点现在是什么状态」，细节进 tooltip / 检查器。 */
export function nodeSummary(node: WorkflowNode, columns: SchemaColumn[]): string {
  const spec = specOf(node.type);
  const flat = flattenConfig(node.config);
  const missing = (spec?.params ?? []).filter(
    (param) => param.required && flat[param.key] == null,
  );
  if (missing.length) return `还差：${missing.map((param) => param.label).join("、")}`;
  if (node.type === "data.load" && flat.dataset_id != null) return `数据集 #${flat.dataset_id}`;
  if (columns.length) return `${columns.length} 个字段可用`;
  return "已配置";
}

/** 该节点是否缺必填参数（供状态展示与运行前校验共用）。 */
export function nodeNeedsConfig(node: WorkflowNode): boolean {
  const spec = specOf(node.type);
  if (!spec) return false;
  const flat = flattenConfig(node.config);
  return spec.params.some((param) => param.required && flat[param.key] == null);
}

interface Props {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  nodeStates?: Record<string, string>;
  selectedNodeId: string | null;
  onSelectNode: (id: string | null) => void;
  onChange: (nodes: WorkflowNode[], edges: WorkflowEdge[], coalesceKey?: string) => void;
  columns?: SchemaColumn[];
  /** 节点执行产出的行/列摘要（来自运行结果），用于分层信息展示。 */
  nodeMetrics?: Record<string, { rows?: number; columns?: number }>;
  /** 数据集下拉选项（供 data.load 的 dataset_id 使用）。 */
  datasetOptions?: Array<{ id: number; name: string }>;
  /** 显示 / 隐藏节点库（沉浸模式下由外层顶栏控制）。 */
  explorerOpen?: boolean;
  onCloseExplorer?: () => void;
  /** 显示 / 隐藏右侧参数面板（沉浸模式下由外层顶栏控制）。 */
  inspectorOpen?: boolean;
  onCloseInspector?: () => void;
  /** 数据集下拉选项（供 data.load 的 dataset_id 使用）。 */
  onDatasetOptionsChange?: (options: Array<{ id: number; name: string }>) => void;
  /** 画布空状态里渲染的引导内容（模板卡片等）。 */
  emptyStateSlot?: React.ReactNode;
  /**
   * 参数面板「未选中任何节点」时的插槽。
   * 沉浸式编辑器用它承载快速起步（模板 / 数据集）——表单类控件放侧栏，
   * 画布空状态只留一句提示，不再把下拉框铺在画布中央。
   */
  inspectorEmptySlot?: React.ReactNode;
  /**
   * 外观命名空间。
   * - 不传 / "classic"：列表页内嵌，走 `workflow.css`（含响应式断点）；
   * - "editor"：沉浸式编辑器，根容器额外带 `wf-editor-workspace` class，
   *   由 `workflow-editor.css` 用独立命名空间重定义视觉，**无视**列表页的
   *   媒体查询（那些断点会把三栏压成两栏/单栏、固定 600px 高）。
   */
  surface?: "classic" | "editor";
}

export function WorkflowStudio({
  nodes,
  edges,
  nodeStates = {},
  selectedNodeId,
  onSelectNode,
  onChange,
  columns = [],
  nodeMetrics = {},
  datasetOptions = [],
  explorerOpen,
  onCloseExplorer,
  inspectorOpen,
  onCloseInspector,
  emptyStateSlot,
  inspectorEmptySlot,
  surface = "classic",
}: Props) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [canvas, setCanvas] = useState(CANVAS_BASE);
  const [search, setSearch] = useState("");
  const [group, setGroup] = useState("全部");
  const [connectFrom, setConnectFrom] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [panDrag, setPanDrag] = useState<PanState | null>(null);
  // 内部抽屉状态仅用于「非受控」场景；沉浸模式下由外层顶栏受控。
  const [explorerHalfOpen, setExplorerHalfOpen] = useState(false);
  const drawerOpen = explorerOpen ?? explorerHalfOpen;

  const canvasSize = useMemo(() => {
    const ps = nodes.map((node, index) => position(node, index));
    const maxX = ps.length ? Math.max(...ps.map((p) => p.x)) : 0;
    const maxY = ps.length ? Math.max(...ps.map((p) => p.y)) : 0;
    const minX = ps.length ? Math.min(...ps.map((p) => p.x)) : 0;
    const minY = ps.length ? Math.min(...ps.map((p) => p.y)) : 0;
    const needW = Math.max(CANVAS_BASE.width, maxX + NODE_W + CANVAS_STEP, -minX + CANVAS_STEP);
    const needH = Math.max(CANVAS_BASE.height, maxY + NODE_H + CANVAS_STEP, -minY + CANVAS_STEP);
    if (needW === canvas.width && needH === canvas.height) return canvas;
    return { width: needW, height: needH };
  }, [nodes, canvas]);
  useEffect(() => { if (canvasSize !== canvas) setCanvas(canvasSize); }, [canvasSize, canvas]);

  const selected = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedSpec = selected ? specOf(selected.type) : undefined;
  // 受控时由外层决定参数面板去留；非受控（经典页面）恒展开，行为与改造前一致。
  const inspectorVisible = inspectorOpen ?? true;
  const filtered = useMemo(
    () =>
      NODE_TYPES.filter((type) => {
        const spec = specOf(type);
        const text = `${type} ${spec?.label ?? ""} ${spec?.description ?? ""}`.toLowerCase();
        return (group === "全部" || spec?.category === group) && text.includes(search.trim().toLowerCase());
      }),
    [group, search],
  );

  const changeNode = (id: string, next: Position) => {
    onChange(
      nodes.map((node) => (node.id === id ? { ...node, config: { ...node.config, __ui: next } } : node)),
      edges,
      // 合并连续拖拽：否则每帧压一条历史，撤销一次只退回几像素。
      `move:${id}`,
    );
  };
  const removeNode = (id: string) => {
    onChange(nodes.filter((node) => node.id !== id), edges.filter((edge) => edge.source !== id && edge.target !== id));
    onSelectNode(null);
    setConnectFrom(null);
  };
  const addNode = (type: string, at?: Position) => {
    const base = type.replace(/\./g, "_");
    let id = base;
    let n = 2;
    while (nodes.some((node) => node.id === id)) id = `${base}_${n++}`;
    const rect = viewportRef.current?.getBoundingClientRect();
    const center = rect
      ? { x: (rect.width / 2 - pan.x) / zoom - NODE_W / 2, y: (rect.height / 2 - pan.y) / zoom - NODE_H / 2 }
      : { x: 120, y: 120 };
    const spot = freeSpot(center, nodes.map((node, index) => position(node, index)));
    const p = at ?? { x: Math.max(NODE_MIN, spot.x), y: Math.max(NODE_MIN, spot.y) };
    onChange([...nodes, { id, type, config: { __ui: p } }], edges);
    onSelectNode(id);
    onCloseExplorer?.();
  };
  const connect = (id: string) => {
    if (!connectFrom) {
      setConnectFrom(id);
      return;
    }
    if (connectFrom !== id && !edges.some((edge) => edge.source === connectFrom && edge.target === id)) {
      onChange(nodes, [...edges, { source: connectFrom, target: id }]);
    }
    setConnectFrom(null);
  };
  const centerOf = (id: string) => {
    const node = nodes.find((item) => item.id === id);
    const p = node ? position(node, nodes.indexOf(node)) : { x: 0, y: 0 };
    return { x: p.x + NODE_W, y: p.y + 59 };
  };
  const fit = () => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    if (!nodes.length) {
      setZoom(1);
      setPan({ x: rect.width / 2, y: rect.height / 2 });
      return;
    }
    const ps = nodes.map((node, index) => position(node, index));
    const minX = Math.min(...ps.map((p) => p.x));
    const minY = Math.min(...ps.map((p) => p.y));
    const maxX = Math.max(...ps.map((p) => p.x + NODE_W));
    const maxY = Math.max(...ps.map((p) => p.y + NODE_H));
    const contentW = Math.max(NODE_W, maxX - minX);
    const contentH = Math.max(NODE_H, maxY - minY);
    const next = Math.max(0.55, Math.min(1.2, Math.min((rect.width - 64) / contentW, (rect.height - 64) / contentH)));
    setZoom(next);
    setPan({ x: (rect.width - contentW * next) / 2 - minX * next, y: (rect.height - contentH * next) / 2 - minY * next });
  };
  const arrange = () => {
    if (!nodes.length) return;
    const indegree = new Map(nodes.map((node) => [node.id, 0]));
    edges.forEach((edge) => indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1));
    const level = new Map<string, number>();
    const queue = nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
    nodes.forEach((node) => { if (!queue.includes(node.id)) level.set(node.id, 0); });
    queue.forEach((id) => level.set(id, 0));
    for (let i = 0; i < queue.length; i += 1) {
      const id = queue[i];
      edges.filter((edge) => edge.source === id).forEach((edge) => {
        level.set(edge.target, Math.max(level.get(edge.target) ?? 0, (level.get(id) ?? 0) + 1));
        const left = (indegree.get(edge.target) ?? 1) - 1;
        indegree.set(edge.target, left);
        if (left === 0) queue.push(edge.target);
      });
    }
    const rows = new Map<number, number>();
    onChange(
      nodes.map((node) => {
        const l = level.get(node.id) ?? 0;
        const row = rows.get(l) ?? 0;
        rows.set(l, row + 1);
        return { ...node, config: { ...node.config, __ui: { x: 80 + l * DEFAULT_GAP_X, y: 90 + row * DEFAULT_GAP_Y } } };
      }),
      edges,
    );
    requestAnimationFrame(fit);
  };
  useEffect(() => { requestAnimationFrame(fit); }, []);
  useEffect(() => {
    if (nodes.length === 0) return;
    const timer = window.setTimeout(fit, 0);
    return () => window.clearTimeout(timer);
  }, [currentSignature(nodes)]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.isContentEditable) return;
      if ((event.key === "Delete" || event.key === "Backspace") && selectedNodeId) {
        event.preventDefault();
        removeNode(selectedNodeId);
      }
      if (event.key === "Escape") setConnectFrom(null);
      if (event.key.toLowerCase() === "f") fit();
      if (event.code === "Space") document.body.dataset.workflowSpace = "true";
    };
    const up = () => {
      delete document.body.dataset.workflowSpace;
      setPanDrag(null);
      setDrag(null);
    };
    window.addEventListener("keydown", key);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("keyup", up);
    };
  });

  const updateSelectedConfig = (config: Record<string, unknown>) => {
    if (!selected) return;
    const ui = selected.config?.__ui;
    onChange(
      nodes.map((node) =>
        node.id === selected.id ? { ...node, config: { ...config, ...(ui ? { __ui: ui } : {}) } } : node,
      ),
      edges,
    );
  };
  const configWithoutUi = selected
    ? Object.fromEntries(Object.entries(selected.config ?? {}).filter(([key]) => key !== "__ui"))
    : {};

  /**
   * 世界层用负 margin（CANVAS_MIN）预留负坐标空间，缩放后节点会额外偏移 CANVAS_MIN*(zoom-1)。
   * 这里在 translate 里抵消掉，使「节点坐标 → 屏幕」严格等于 screen = 坐标*zoom + pan，
   * 与 fit / 滚轮缩放 / 拖拽 / 新增节点的计算口径一致（否则 zoom≠1 时新增节点会落到视口外）。
   */
  const worldOffset = CANVAS_MIN * (zoom - 1);

  return (
    <div className={`workflow-workspace${surface === "editor" ? " wf-editor-workspace" : ""}${selectedNodeId ? " node-selected" : ""}${drawerOpen ? " explorer-open" : ""}${inspectorVisible ? "" : " inspector-closed"}`}>
      <aside className="workflow-explorer">
        {/*
          顺序刻意是「当前流程 → 节点库」而不是反过来：
          大纲放最上面（等同 Figma 的图层面板）才能在长列表滚动时始终可见，
          否则它会被节点库挤到折叠线以下，等于没有。
        */}
        <div className="workflow-panel-title">
          <strong>当前流程</strong>
          <span>{nodes.length}</span>
        </div>
        <div className="workflow-current-outline">
          {nodes.length ? (
            nodes.map((node) => {
              // 大纲里同时给出运行状态，节点多了以后不必回画布逐个找哪个没配好。
              const display = nodeDisplay(nodeStates[node.id], nodeNeedsConfig(node));
              return (
                <button
                  key={node.id}
                  type="button"
                  className={node.id === selectedNodeId ? "active" : ""}
                  title={`${NODE_LABELS[node.type] ?? node.type} · ${display.label}`}
                  onClick={() => onSelectNode(node.id)}
                >
                  <span className={`workflow-state-dot ${display.tone}`} />
                  <span>{NODE_LABELS[node.type] ?? node.type}</span>
                  <small>{node.id}</small>
                </button>
              );
            })
          ) : (
            <p>还没有节点</p>
          )}
        </div>

        <div className="workflow-panel-title">
          <strong>节点库</strong>
          <span>{NODE_TYPES.length}</span>
          {onCloseExplorer && (
            <button type="button" className="workflow-panel-collapse" onClick={onCloseExplorer} title="收起节点库" aria-label="收起节点库">‹</button>
          )}
        </div>
        <input
          className="workflow-search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="搜索节点（支持中文名与类型）"
          aria-label="搜索节点"
        />
        <div className="workflow-tabs">
          {["全部", "数据", "机器学习", "AI"].map((item) => (
            <button key={item} type="button" className={group === item ? "active" : ""} onClick={() => setGroup(item)}>
              {item}
            </button>
          ))}
        </div>
        <div className="workflow-palette-list">
          {filtered.map((type) => {
            const spec = specOf(type);
            return (
              <button
                key={type}
                className="workflow-palette-node"
                type="button"
                title={spec?.description}
                onClick={() => addNode(type)}
              >
                <span className="workflow-node-icon" data-category={categoryOf(type)}>
                  <Icon name={spec?.icon ?? "flow"} size={14} />
                </span>
                <span>
                  <strong>{spec?.label ?? type}</strong>
                  <small>{spec?.description ?? type}</small>
                </span>
                <b>＋</b>
              </button>
            );
          })}
          {!filtered.length && <p className="workflow-palette-empty">没有匹配「{search}」的节点。</p>}
        </div>
      </aside>

      <section className="workflow-canvas-area">
        <div className="workflow-canvas-toolbar">
          <div>
            <strong>画布</strong>
            <span>{nodes.length} 节点 · {edges.length} 连接</span>
            {connectFrom && <em>从 {connectFrom} 开始连接</em>}
          </div>
          <div className="workflow-toolbar-actions">
            <button className="btn" type="button" onClick={() => setZoom((z) => Math.min(1.8, z + 0.1))} aria-label="放大">＋</button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="btn" type="button" onClick={() => setZoom((z) => Math.max(0.45, z - 0.1))} aria-label="缩小">−</button>
            <button className="btn" type="button" onClick={fit}>适应</button>
            <button className="btn" type="button" onClick={arrange}>自动排列</button>
            {connectFrom && <button className="btn" type="button" onClick={() => setConnectFrom(null)}>取消连线</button>}
          </div>
        </div>
        <div
          ref={viewportRef}
          className="workflow-canvas"
          onPointerDown={(event) => {
            if (event.button !== 0 || (event.target as HTMLElement).closest(".workflow-node-card,.workflow-edge")) return;
            setPanDrag({ x: pan.x, y: pan.y, px: event.clientX, py: event.clientY });
          }}
          onPointerMove={(event) => {
            if (panDrag) setPan({ x: panDrag.x + event.clientX - panDrag.px, y: panDrag.y + event.clientY - panDrag.py });
          }}
          onPointerUp={() => setPanDrag(null)}
          onPointerLeave={() => setPanDrag(null)}
          onDoubleClick={(event) => {
            if ((event.target as HTMLElement).closest(".workflow-node-card,.workflow-edge")) return;
            const rect = viewportRef.current?.getBoundingClientRect();
            if (!rect) return;
            addNode("noop", {
              x: Math.max(NODE_MIN, (event.clientX - rect.left - pan.x) / zoom - NODE_W / 2),
              y: Math.max(NODE_MIN, (event.clientY - rect.top - pan.y) / zoom - NODE_H / 2),
            });
          }}
          onWheel={(event) => {
            event.preventDefault();
            const factor = event.deltaY < 0 ? 1.06 : 0.94;
            const rect = viewportRef.current?.getBoundingClientRect();
            if (!rect) return;
            const mx = event.clientX - rect.left;
            const my = event.clientY - rect.top;
            const next = Math.max(0.45, Math.min(1.8, zoom * factor));
            setPan({ x: mx - ((mx - pan.x) * next) / zoom, y: my - ((my - pan.y) * next) / zoom });
            setZoom(next);
          }}
        >
          <div
            className="workflow-world"
            style={{
              width: canvasSize.width + CANVAS_MIN * -1,
              height: canvasSize.height + CANVAS_MIN * -1,
              marginLeft: CANVAS_MIN,
              marginTop: CANVAS_MIN,
              transform: `translate(${pan.x + worldOffset}px, ${pan.y + worldOffset}px) scale(${zoom})`,
            }}
          >
            <svg
              className="workflow-edges"
              width={canvasSize.width + CANVAS_MIN * -1}
              height={canvasSize.height + CANVAS_MIN * -1}
              style={{ marginLeft: -CANVAS_MIN, marginTop: -CANVAS_MIN }}
            >
              {edges.map((edge, index) => {
                const a = centerOf(edge.source);
                const b = centerOf(edge.target);
                const mx = a.x + Math.max(60, (b.x - a.x) / 2);
                // 连线标注：上游节点的产出行数。审查 #10 指出连线无数据量提示。
                const from = nodeMetrics[edge.source];
                return (
                  <g
                    key={`${edge.source}-${edge.target}-${index}`}
                    className="workflow-edge"
                    onDoubleClick={() => onChange(nodes, edges.filter((item) => item !== edge))}
                  >
                    <title>
                      {`${edge.source} → ${edge.target}`}
                      {from?.rows != null ? ` · ${from.rows.toLocaleString()} 行` : ""}
                    </title>
                    <path d={`M ${a.x} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`} />
                    <circle cx={b.x} cy={b.y} r="4" />
                    {from?.rows != null && (
                      <text className="workflow-edge-label" x={mx} y={(a.y + b.y) / 2 - 8} textAnchor="middle">
                        {from.rows.toLocaleString()} 行
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
            {nodes.map((node, index) => {
              const p = position(node, index);
              const display = nodeDisplay(nodeStates[node.id], nodeNeedsConfig(node));
              const spec = specOf(node.type);
              const metric = nodeMetrics[node.id];
              return (
                <article
                  key={node.id}
                  className={`workflow-node-card ${node.id === selectedNodeId ? "selected" : ""} ${display.tone}`}
                  style={{ left: p.x - CANVAS_MIN, top: p.y - CANVAS_MIN }}
                  onPointerDown={(event) => {
                    if ((event.target as HTMLElement).closest("button")) return;
                    event.stopPropagation();
                    onSelectNode(node.id);
                    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
                    setDrag({ id: node.id, sx: event.clientX, sy: event.clientY, origin: p });
                  }}
                  onPointerMove={(event) => {
                    if (drag?.id === node.id) {
                      changeNode(node.id, {
                        x: Math.max(NODE_MIN, drag.origin.x + (event.clientX - drag.sx) / zoom),
                        y: Math.max(NODE_MIN, drag.origin.y + (event.clientY - drag.sy) / zoom),
                      });
                    }
                  }}
                  onPointerUp={() => setDrag(null)}
                >
                  {/* 分层信息：卡片只留标题 + 一行摘要 + 状态；细节放 tooltip。 */}
                  <header>
                    <span className="workflow-node-icon" data-category={categoryOf(node.type)}>
                      <Icon name={spec?.icon ?? "flow"} size={14} />
                    </span>
                    <div>
                      <strong>{spec?.label ?? node.type}</strong>
                      <small>{node.id}</small>
                    </div>
                    <button type="button" onClick={() => removeNode(node.id)} aria-label={`删除节点 ${node.id}`}>×</button>
                  </header>
                  <div className="workflow-node-summary">{nodeSummary(node, columns)}</div>
                  <footer>
                    <span className={`workflow-state-dot ${display.tone}`} />
                    {display.label}
                    {metric?.rows != null && (
                      <span className="workflow-node-metric">
                        {metric.rows.toLocaleString()} 行
                        {metric.columns != null ? ` × ${metric.columns} 列` : ""}
                      </span>
                    )}
                  </footer>
                  <button
                    className="workflow-port input"
                    type="button"
                    aria-label={`连接输入到 ${node.id}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (connectFrom) connect(node.id);
                    }}
                  />
                  <button
                    className={`workflow-port output${connectFrom === node.id ? " active" : ""}`}
                    type="button"
                    aria-label={`从 ${node.id} 开始连接`}
                    onClick={(event) => {
                      event.stopPropagation();
                      connect(node.id);
                    }}
                  />
                </article>
              );
            })}
            {!nodes.length && (
              <div className="workflow-empty-canvas" style={{ left: -CANVAS_MIN, top: -CANVAS_MIN }}>
                {emptyStateSlot ?? (
                  <>
                    <strong>从模板或一个数据节点开始</strong>
                    <span>在左侧节点库添加节点，或双击画布空白处。</span>
                  </>
                )}
                <small>滚轮缩放 · 拖动画布 · 双击空白添加 · F 适应</small>
              </div>
            )}
          </div>
          <div className="workflow-canvas-status">
            <span>{nodes.length} 节点</span>
            <span>{edges.length} 连接</span>
            <button type="button" onClick={fit}>定位全部</button>
          </div>
        </div>
      </section>

      <aside className="workflow-inspector-panel">
        {selected && selectedSpec ? (
          <>
            <div className="workflow-inspector-heading">
              <div>
                <span className="workflow-eyebrow">{selectedSpec.category}</span>
                <h3>{selectedSpec.label}</h3>
                <small>{selected.id}</small>
              </div>
              <div className="workflow-inspector-actions">
                {onCloseInspector && (
                  <button className="btn workflow-inspector-collapse" type="button" onClick={onCloseInspector} title="收起参数面板">›</button>
                )}
                <button className="btn workflow-inspector-close" type="button" onClick={() => onSelectNode(null)}>关闭</button>
                <button className="btn danger" type="button" onClick={() => removeNode(selected.id)}>删除</button>
              </div>
            </div>

            <p className="workflow-inspector-desc">{selectedSpec.description}</p>

            <section>
              <label>运行状态</label>
              {(() => {
                const display = nodeDisplay(nodeStates[selected.id], nodeNeedsConfig(selected));
                return (
                  <div className={`workflow-inspector-status ${display.tone}`}>
                    <span className={`workflow-state-dot ${display.tone}`} />
                    {display.label}
                  </div>
                );
              })()}
            </section>

            <section>
              <label>参数 <small>{selectedSpec.params.length} 项</small></label>
              <NodeParamForm
                specs={selectedSpec.params}
                config={configWithoutUi}
                columns={columns}
                onChange={updateSelectedConfig}
              />
            </section>

            {(datasetOptions.length > 0 && selected.type === "data.load") && (
              <section>
                <label>数据集</label>
                <select
                  className="wf-param-input"
                  value={String(flattenConfig(selected.config).dataset_id ?? "")}
                  onChange={(event) => {
                    const raw = event.target.value;
                    updateSelectedConfig({
                      ...flattenConfig(selected.config),
                      dataset_id: raw === "" ? undefined : Number(raw),
                    });
                  }}
                >
                  <option value="">（未选择）</option>
                  {datasetOptions.map((option) => (
                    <option key={option.id} value={option.id}>#{option.id} {option.name}</option>
                  ))}
                </select>
              </section>
            )}

            <section>
              <label>数据上下文</label>
              {columns.length ? (
                <>
                  <p>{columns.length} 个字段可用。</p>
                  <div className="workflow-schema-list">
                    {columns.slice(0, 24).map((column) => (
                      <span key={column.column} title={column.dtype}>{column.column}</span>
                    ))}
                  </div>
                </>
              ) : (
                <p>还没有上游 schema。选中「加载数据」节点并选择数据集后，这里会列出可用字段。</p>
              )}
            </section>
          </>
        ) : selected ? (
          <div className="workflow-inspector-empty">
            <strong>未知节点类型</strong>
            <p>{selected.type} 不在当前节点库中。</p>
          </div>
        ) : (
          inspectorEmptySlot ?? (
            <div className="workflow-inspector-empty">
              <strong>选择节点</strong>
              <div>
                <kbd>Delete</kbd> 删除　<kbd>Esc</kbd> 取消连线　<kbd>F</kbd> 适应
              </div>
            </div>
          )
        )}
      </aside>
    </div>
  );
}

function currentSignature(nodes: WorkflowNode[]) {
  return nodes.map((node) => `${node.id}:${node.type}`).join("|");
}

export type { NodeSpec };
