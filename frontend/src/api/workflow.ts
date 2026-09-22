/** Workflow API（Prompt 201 对应前端）。 */
import { client, unwrap } from "./client";
import type {
  Workflow,
  WorkflowRun,
  WorkflowSummary,
} from "../types/workflow";

export function createWorkflow(body: {
  name: string;
  nodes: Workflow["nodes"];
  edges: Workflow["edges"];
  metadata?: Record<string, unknown>;
}) {
  return unwrap<Workflow>(client.post("/workflows", body));
}

export function listWorkflows() {
  return unwrap<WorkflowSummary[]>(client.get("/workflows"));
}

export function getWorkflow(id: number) {
  return unwrap<Workflow>(client.get(`/workflows/${id}`));
}

export function updateWorkflow(
  id: number,
  body: { name: string; nodes: Workflow["nodes"]; edges: Workflow["edges"] },
) {
  return unwrap<Workflow>(client.put(`/workflows/${id}`, body));
}

export function deleteWorkflow(id: number) {
  return unwrap<{ deleted: boolean }>(client.delete(`/workflows/${id}`));
}

export function runWorkflow(id: number, context: Record<string, unknown> = {}) {
  return unwrap<WorkflowRun>(client.post(`/workflows/${id}/run`, { context }));
}

export function getWorkflowRun(runId: string) {
  return unwrap<WorkflowRun>(client.get(`/workflows/runs/${runId}`));
}

export function cancelWorkflowRun(runId: string) {
  return unwrap<{ cancelled: boolean }>(
    client.post(`/workflows/runs/${runId}/cancel`),
  );
}

export function cloneWorkflow(id: number) {
  return unwrap<Workflow>(client.post(`/workflows/${id}/clone`));
}
