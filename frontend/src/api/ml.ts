/** ML / Experiments API（Prompt 200-202 对应前端）。 */
import { client, unwrap } from "./client";
import type { Pagination } from "../types/common";
import type {
  CompareResult,
  Experiment,
  ExperimentRun,
  ExplainResult,
  MlCatalog,
  MlModel,
  PredictionResult,
  TrainProgress,
  TrainResult,
} from "../types/ml";

export function listModels() {
  return unwrap<MlModel[]>(client.get("/ml/models"));
}

/**
 * 教学目录：流程环节 / 预处理参数 / 训练参数 / 各模型参数 / 指标解读。
 * 与后端 `app/ml_engine/metadata.py` 同源，供参数说明与流程说明面板消费。
 */
export function getMlCatalog() {
  return unwrap<MlCatalog>(client.get("/ml/catalog"));
}

export interface TrainBody {
  dataset_id: number;
  version?: number | null;
  task: string;
  model: string;
  target_column?: string | null;
  parameters?: Record<string, unknown>;
  preprocessing?: Record<string, unknown>;
  excluded_columns?: string[];
  seed?: number | null;
  /** 测试集比例，缺省 0.2（后端默认）。 */
  test_size?: number | null;
  description?: string;
}

export function trainModel(body: TrainBody) {
  return unwrap<TrainResult>(client.post("/ml/train", body));
}

/**
 * 流式训练：`POST /ml/train/stream` 以 SSE 边训练边回推进度。
 *
 * 大表训练可能 30s+，同步接口会撞上 axios 默认 15s 超时；这里改用原生
 * fetch + ReadableStream 读取，无超时上限，且能实时拿到每个阶段的事件。
 * onProgress 在每次阶段完成时回调一次；返回的 Promise 在收到 result 事件后 resolve。
 */
export function trainModelStream(
  body: TrainBody,
  onProgress?: (p: TrainProgress) => void,
): Promise<TrainResult> {
  const url = client.getUri({ url: "/ml/train/stream" });
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((resp) => {
    if (!resp.ok || !resp.body) {
      return resp
        .text()
        .catch(() => "")
        .then((t) => {
          throw new Error(t.slice(0, 200) || `训练请求失败（HTTP ${resp.status}）`);
        });
    }
    return readTrainStream(resp.body, onProgress);
  });
}

/** 逐块解析 SSE：progress → 回调；result → 解析结果；error → 记录失败；done → 结束。 */
async function readTrainStream(
  stream: ReadableStream<Uint8Array>,
  onProgress?: (p: TrainProgress) => void,
): Promise<TrainResult> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: TrainResult | null = null;
  let failure: Error | null = null;

  try {
    let receivedDone = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const { event, data } = parseSseEvent(raw);
        if (event === "done") {
          // done 是最后一个事件：主动停止读取，不依赖后端同步生成器的收尾，
          // 避免收尾时连接重置被浏览器报成 network error（与 agent SSE 同策略）。
          receivedDone = true;
          break;
        }
        if (event === "progress" && onProgress) {
          try {
            onProgress(JSON.parse(data) as TrainProgress);
          } catch {
            /* 忽略解析失败的单条事件 */
          }
        } else if (event === "result") {
          try {
            result = JSON.parse(data) as TrainResult;
          } catch {
            /* 忽略 */
          }
        } else if (event === "error") {
          try {
            failure = new Error((JSON.parse(data) as { error?: string }).error ?? "训练失败");
          } catch {
            failure = new Error("训练失败");
          }
        }
      }
      if (receivedDone) break;
    }
  } catch (e) {
    // 后端 SSE 流被中断（网络层重置连接）时会走到这里，而不是 done。
    failure = new Error(`训练连接中断（${e instanceof Error ? e.message : "网络错误"}），请重试`);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* 已释放 */
    }
  }

  if (failure) throw failure;
  if (!result) throw new Error("训练流意外结束，未收到结果");
  return result;
}

/** 解析单个 SSE 事件块（`event: xxx\ndata: {...}`）。 */
function parseSseEvent(raw: string): { event: string; data: string } {
  let event = "message";
  const data: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  return { event, data: data.join("\n") };
}

/** 用一次成功运行的模型对新数据推理（走训练时的预处理管道）。 */
export function predictWithRun(body: {
  run_id: number;
  dataset_id?: number | null;
  version?: number | null;
  limit?: number;
  /** 决策阈值（仅二分类）：训练后参数，改它不需要重新训练。 */
  threshold?: number | null;
}) {
  return unwrap<PredictionResult>(client.post("/ml/predict", body));
}

/** 模型解释：特征重要性（优先复用训练时落库的结果）。 */
export function explainRun(runId: number) {
  return unwrap<ExplainResult>(client.get(`/ml/explain/${runId}`));
}

export function listExperiments(page = 1, pageSize = 20) {
  return unwrap<Pagination<Experiment>>(
    client.get("/experiments", { params: { page, page_size: pageSize } }),
  );
}

export function getExperiment(id: number) {
  return unwrap<Experiment>(client.get(`/experiments/${id}`));
}

export function getExperimentRuns(id: number) {
  return unwrap<ExperimentRun[]>(client.get(`/experiments/${id}/runs`));
}

export function runExperiment(id: number) {
  return unwrap<ExperimentRun>(client.post(`/experiments/${id}/run`));
}

export function compareRuns(runIds: number[]) {
  return unwrap<CompareResult>(client.post("/experiments/compare", { run_ids: runIds }));
}

export function deleteExperiment(id: number) {
  return unwrap<{ deleted: boolean; experiment_id: number }>(
    client.delete(`/experiments/${id}`),
  );
}

export function deleteAllExperiments() {
  return unwrap<{ deleted: boolean; count: number }>(client.delete("/experiments"));
}
