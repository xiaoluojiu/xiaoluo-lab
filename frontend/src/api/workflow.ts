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

/**
 * 工作流运行的预计耗时文案。
 *
 * `POST /workflows/{id}/run` 是同步接口（节点全部跑完才返回），前端拿不到中间进度。
 * 这时能给用户的确定信息只有「在跑」与「大概多久」，所以由这里集中提供，
 * 页面不必各自编一句口径不同的文案。
 */
export const WORKFLOW_RUN_ESTIMATE = "流程运行中，预计需要 30 秒";

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
