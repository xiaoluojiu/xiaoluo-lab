/**
 * Experiments API（Prompt 200-202 对应前端）。
 *
 * 为什么从 `api/ml.ts` 拆出来：实验与 ML 共用同一批后端资源，但**调用方不同** ——
 * 实验中心页只碰实验、机器学习页只碰训练与推理。拆开后，「同步接口的阶段说明」
 * （`EXPERIMENT_RUN_*`）才有地方安放，不必挤在流式训练接口旁边。
 *
 * `api/ml.ts` 保留同名 re-export，既有调用方（`pages/ML`）无需改动。
 */
import { client, unwrap } from "./client";
import type { Pagination } from "../types/common";
import type { CompareResult, Experiment, ExperimentRun, MlHpoResult } from "../types/ml";

export function listExperiments(page = 1, pageSize = 20, withMetrics = false) {
  return unwrap<Pagination<Experiment>>(
    client.get("/experiments", {
      params: { page, page_size: pageSize, ...(withMetrics ? { with_metrics: true } : {}) },
    }),
  );
}

export function getExperiment(id: number) {
  return unwrap<Experiment>(client.get(`/experiments/${id}`));
}

export function getExperimentRuns(id: number) {
  return unwrap<ExperimentRun[]>(client.get(`/experiments/${id}/runs`));
}

/**
 * 实验叙述（`GET /experiments/{id}/narrative`）。
 *
 * 后端把「想验证什么 / 结果说明什么 / 下一步做什么 / 为什么失败」压成固定四字段，
 * 让实验记录不只是状态与数字。四个字段都可能为空（例如还没有任何运行），
 * 调用方需自行判断是否渲染。
 */
export interface ExperimentNarrative {
  hypothesis?: string;
  conclusion?: string;
  next_action?: string;
  failure_reason?: string;
}

export function getExperimentNarrative(id: number) {
  return unwrap<ExperimentNarrative>(client.get(`/experiments/${id}/narrative`));
}

export function runExperiment(id: number) {
  return unwrap<ExperimentRun>(client.post(`/experiments/${id}/run`));
}

/**
 * 实验重跑的粗粒度阶段骨架。
 *
 * ⚠️ `POST /experiments/{id}/run` 是**同步接口**（后端跑完才返回），没有 SSE、
 * 也没有轮询用的进度端点，所以这四个阶段**不会**被逐个点亮 —— 它们的作用是让用户
 * 看清「这次重跑会经历哪些环节」，进度本身走不确定进度（流动条 + 预计耗时）。
 * 想变成真实阶段推进需要后端提供进度通道，前端不伪造。
 */
export const EXPERIMENT_RUN_STAGES = [
  "加载实验配置",
  "准备特征与目标",
  "训练与评估模型",
  "保存运行结果",
];

/** 实验重跑的预计耗时文案（同样因为是同步接口，只能给量级、给不了剩余秒数）。 */
export const EXPERIMENT_RUN_ESTIMATE = "实验运行中，预计需要 30 秒";

export function compareRuns(runIds: number[]) {
  return unwrap<CompareResult>(client.post("/experiments/compare", { run_ids: runIds }));
}

/** 实验级对比：每个实验取最近一次成功运行，比较逻辑仍走后端 ExperimentComparator。 */
export function compareExperiments(experimentIds: number[]) {
  return unwrap<CompareResult>(
    client.post("/experiments/compare", { experiment_ids: experimentIds }),
  );
}

/**
 * 自动超参搜索（`POST /experiments/{id}/optimize`）。
 *
 * 后端跑 RandomizedSearchCV over 5 折 CV（预处理折内拟合），返回最佳参数与
 * Top10；**不改动实验配置** —— 采纳与否由界面上的「应用最佳参数」决定。
 * 一次搜索的训练量是普通训练的 n_iter × folds 倍，静默改配置再重训既昂贵
 * 也无法解释这些参数是哪来的。
 */
export function optimizeExperiment(id: number, nIter = 20) {
  return unwrap<MlHpoResult>(client.post(`/experiments/${id}/optimize`, { n_iter: nIter }));
}

export function deleteExperiment(id: number) {
  return unwrap<{ deleted: boolean; experiment_id: number }>(
    client.delete(`/experiments/${id}`),
  );
}

export function deleteAllExperiments() {
  return unwrap<{ deleted: boolean; count: number }>(client.delete("/experiments"));
}

/** 批量删除实验（前端「删除选中」）。返回实际删除条数，不存在的 id 会被跳过。 */
export function deleteExperiments(experimentIds: number[]) {
  return unwrap<{ deleted: boolean; count: number; requested: number }>(
    client.post("/experiments/batch-delete", { experiment_ids: experimentIds }),
  );
}

/** 删除单次运行：实验配置保留，只清掉这一次的运行记录与模型产物。 */
export function deleteRun(runId: number) {
  return unwrap<{ deleted: boolean; run_id: number }>(
    client.delete(`/experiments/runs/${runId}`),
  );
}

/** 批量删除运行记录（含各自的模型产物），返回实际删除条数。 */
export function deleteRuns(runIds: number[]) {
  return unwrap<{ deleted: boolean; count: number; requested: number }>(
    client.post("/experiments/runs/batch-delete", { run_ids: runIds }),
  );
}
