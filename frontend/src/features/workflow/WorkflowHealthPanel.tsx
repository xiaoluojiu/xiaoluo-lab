import type { WorkflowEdge, WorkflowNode, WorkflowRun } from "../../types/workflow";
import type { SchemaColumn } from "../../types/dataset";
import { missingConfigWarnings } from "./configView";
import "./workflow-health.css";

const DATA_NODE_TYPES = new Set([
  "noop",
  "data.load",
  "data.clean",
  "data.duplicate",
  "data.cast",
  "data.string",
  "data.filter",
  "data.transform",
  "data.aggregate",
  "data.pivot",
  "data.melt",
]);

function validate(nodes: WorkflowNode[], edges: WorkflowEdge[], columns: SchemaColumn[]) {
  const errors: string[] = [];
  const warnings: string[] = [];
  const ids = new Set(nodes.map((node) => node.id));
  if (!nodes.length) warnings.push("工作流还没有节点。");
  const duplicateIds = nodes.filter((node, index) => nodes.findIndex((item) => item.id === node.id) !== index);
  if (duplicateIds.length) errors.push(`存在重复节点 ID：${[...new Set(duplicateIds.map((node) => node.id))].join("、")}`);
  const edgeKeys = new Set<string>();
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) errors.push(`连接引用了不存在的节点：${edge.source} → ${edge.target}`);
    if (edge.source === edge.target) errors.push(`节点 ${edge.source} 存在自连接。`);
    const key = `${edge.source}->${edge.target}`;
    if (edgeKeys.has(key)) warnings.push(`存在重复连接：${edge.source} → ${edge.target}`);
    edgeKeys.add(key);
  }
  const incoming = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1));
  if (nodes.length > 1 && !nodes.some((node) => (incoming.get(node.id) ?? 0) === 0)) errors.push("没有找到起始节点，流程可能存在循环依赖。");
  warnings.push(...missingConfigWarnings(nodes));
  if (nodes.length && !edges.length && nodes.length > 1) warnings.push("有多个节点但尚未建立连接；运行时它们不会形成数据处理链。");
  if (columns.length && nodes.some((node) => node.type === "data.load")) warnings.push(`当前数据上下文有 ${columns.length} 个字段，建议在筛选、转换等节点中明确使用字段。`);
  return { errors, warnings };
}

function outputSummary(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.row_count === "number" || typeof record.column_count === "number") {
    return `${record.row_count ?? "?"} 行 × ${record.column_count ?? "?"} 列`;
  }
  return null;
}

export function WorkflowHealthPanel({
  nodes, edges, columns = [], run, onSelectNode,
}: {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  columns?: SchemaColumn[];
  run?: WorkflowRun | null;
  onSelectNode?: (id: string) => void;
}) {
  const { errors, warnings } = validate(nodes, edges, columns);
  const outputs = run?.result?.outputs ?? {};
  const executed = nodes.filter((node) => outputs[node.id] != null);
  const dataOutputs = executed
    .map((node) => ({ node, summary: outputSummary(outputs[node.id]) }))
    .filter((item) => item.summary);
  const unsupported = nodes.filter((node) => !DATA_NODE_TYPES.has(node.type));
  const score = errors.length ? 0 : warnings.length ? 70 : nodes.length ? 100 : 0;

  return <section className="workflow-health-panel">
    <div className="workflow-health-header">
      <div><span className="learning-kicker">WORKFLOW HEALTH</span><h3>流程体检与数据流</h3><p>先做确定性的结构检查，再用真实执行结果观察数据如何经过每个节点。</p></div>
      <div className={`workflow-health-score ${score === 100 ? "good" : score === 70 ? "warn" : "bad"}`}><strong>{score}</strong><span>结构状态</span></div>
    </div>

    <div className="workflow-health-grid">
      <div className="workflow-health-card">
        <div className="workflow-health-card-title">结构检查</div>
        {errors.length === 0 && warnings.length === 0 && <div className="workflow-health-ok">✓ 当前没有发现结构问题</div>}
        {errors.map((item) => <button className="workflow-health-item error" key={item} type="button">× {item}</button>)}
        {warnings.map((item) => <button className="workflow-health-item warning" key={item} type="button">! {item}</button>)}
      </div>

      <div className="workflow-health-card">
        <div className="workflow-health-card-title">数据流</div>
        {!run && <div className="workflow-health-empty">运行一次工作流后，这里会显示各节点实际产生的数据规模。</div>}
        {run && !dataOutputs.length && <div className="workflow-health-empty">本次运行还没有可展示的数据输出。</div>}
        {dataOutputs.map(({ node, summary }) => <button key={node.id} className="workflow-lineage-item" type="button" onClick={() => onSelectNode?.(node.id)}><span>{node.type === "data.load" ? "数据集" : "处理"}</span><strong>{node.id}</strong><em>{summary}</em></button>)}
      </div>
    </div>

    <div className="workflow-health-footer">
      <span>节点 {nodes.length}</span><span>连接 {edges.length}</span><span>执行输出 {executed.length}</span><span>数据字段 {columns.length}</span>
      {unsupported.length > 0 && <span className="muted">尚未接入执行器：{unsupported.length} 个节点</span>}
    </div>
  </section>;
}
