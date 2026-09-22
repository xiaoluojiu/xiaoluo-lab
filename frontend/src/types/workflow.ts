export interface WorkflowNode {
  id: string;
  type: string;
  config: Record<string, unknown>;
}

export interface WorkflowEdge {
  source: string;
  target: string;
}

export interface Workflow {
  id: number | null;
  name: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata: Record<string, unknown>;
}

export interface WorkflowSummary {
  id: number;
  name: string;
  nodes: number;
  edges: number;
}

export interface NodeLog {
  node_id: string;
  status: string;
  started_at: number | null;
  finished_at: number | null;
  duration_ms: number | null;
  error: string | null;
}

export interface WorkflowRunResult {
  status: string;
  node_states: Record<string, string>;
  outputs: Record<string, unknown>;
  logs: NodeLog[];
}

export interface WorkflowRun {
  run_id: string;
  workflow_id: number;
  status: string;
  result: WorkflowRunResult | null;
}
