/** 学习中心 API。
 *
 * 与后端 `backend/app/api/v1/learning.py` 一一对应。
 * 判定在后端完成（AST 静态分析 + 真实数据探针），前端只负责展示与交互，
 * 因此这里没有「本地猜一个结果」的逻辑——所有结论都来自服务端。
 */
import { client, unwrap } from "./client";

export type TrackId = "ml" | "llm";
export type Level = "基础" | "进阶" | "挑战";
export type ProgressStatus = "not_started" | "in_progress" | "completed";

/** 单条检查结果。 */
export interface LearningCheck {
  requirement_index: number;
  kind: "ast" | "data";
  label: string;
  passed: boolean;
  detail: string;
  /** false 表示「未经校验」（如没选数据集时的数据检查），前端应显示为「未验证」而非「已通过」。 */
  verified: boolean;
}

/** 一次检查的完整结果。 */
export interface LearningReview {
  passed: boolean;
  score: number;
  checks: LearningCheck[];
  next_steps: string[];
  /** 语法错误等一票否决原因，有值时其他检查结果无意义。 */
  blocked: string | null;
  /** 数据集读取失败时的提示（结构检查仍已完成）。 */
  dataset_warning?: string;
  progress?: LearningProgressDetail;
}

export interface LearningProgressSummary {
  status: ProgressStatus;
  attempts: number;
  score: number;
  last_passed: boolean | null;
  completed_at: string | null;
  saved_at: string | null;
}

export interface LearningAttempt {
  score: number;
  passed: boolean;
  created_at: string | null;
  summary: string;
}

export interface LearningProgressDetail extends LearningProgressSummary {
  /** 已保存的代码；没有记录时后端返回 starter。 */
  code: string;
  dataset_id: number | null;
  /** 要求文本 -> 是否达成。 */
  checklist: Record<string, boolean>;
  last_result: Partial<LearningReview>;
  history: LearningAttempt[];
}

export interface LearningConcept {
  term: string;
  body: string;
}

export interface LearningReference {
  label: string;
  url: string;
}

export interface LearningExperiment {
  key: string;
  title: string;
  track: TrackId;
  level: Level;
  summary: string;
  why: string;
  goal: string;
  requirements: string[];
  starter: string;
  needs: string;
  minutes: number;
  prerequisite: string | null;
  concepts: LearningConcept[];
  references: LearningReference[];
  check_count: number;
  progress: LearningProgressSummary;
}

export interface LearningExperimentDetail extends Omit<LearningExperiment, "progress"> {
  progress: LearningProgressDetail;
  prerequisite_info?: {
    key: string;
    title: string;
    completed: boolean;
  };
}

export interface LearningTrack {
  id: TrackId;
  label: string;
  description: string;
  accent: string;
}

export interface LearningTrackStat {
  id: TrackId;
  label: string;
  total: number;
  completed: number;
  in_progress: number;
}

export interface LearningRecentItem {
  key: string;
  title: string;
  track: TrackId;
  status: ProgressStatus;
  score: number;
  attempts: number;
  updated_at: string | null;
}

export interface LearningOverview {
  total: number;
  started: number;
  completed: number;
  completion_rate: number;
  tracks: LearningTrackStat[];
  by_level: Record<string, number>;
  by_key: Record<string, { status: ProgressStatus; score: number; attempts: number }>;
  recommended: string | null;
  recent: LearningRecentItem[];
}

/** 数据体检面板的数据。 */
export interface LearningContextColumn {
  name: string;
  dtype: string;
  is_numeric: boolean;
  null_count: number;
  null_rate: number;
  unique_count: number;
}

export interface LearningContext {
  dataset_id: number;
  dataset_name: string;
  version: number;
  row_count: number;
  column_count: number;
  columns: LearningContextColumn[];
  numeric_columns: string[];
  label_candidates: string[];
  target_candidates: string[];
  missing_cells: number;
}

export interface CheckPayload {
  code: string;
  dataset_id?: number | null;
  version?: number | null;
  learner_id?: string;
  persist?: boolean;
}

/** 自建学习卡片。 */
export type CustomCardKind = "practice" | "checklist";

export interface CustomCardItem {
  text: string;
  done: boolean;
}

export interface CustomCardProgress {
  done: number;
  total: number;
  rate: number;
}

export interface CustomCard {
  id: number;
  learner_id: string;
  title: string;
  kind: CustomCardKind;
  goal: string;
  code: string;
  items: CustomCardItem[];
  progress: CustomCardProgress;
  created_at: string | null;
  updated_at: string | null;
}

export interface CustomCardCreate {
  title: string;
  kind: CustomCardKind;
  goal?: string;
  code?: string;
  items?: CustomCardItem[];
  learner_id?: string;
}

export interface CustomCardUpdate {
  title?: string;
  goal?: string;
  code?: string;
  items?: CustomCardItem[];
  learner_id?: string;
}

// tracks 是静态目录（两条主线），进程内缓存一次，避免每次进入学习中心都重新请求。
let _tracksCache: LearningTrack[] | null = null;

export function listTracks() {
  if (_tracksCache) return Promise.resolve(_tracksCache);
  return unwrap<LearningTrack[]>(client.get("/learning/tracks")).then((tracks) => {
    _tracksCache = tracks;
    return tracks;
  });
}

export function listExperiments(learnerId = "anonymous", track?: TrackId) {
  return unwrap<LearningExperiment[]>(
    client.get("/learning/experiments", {
      params: { learner_id: learnerId, track: track ?? undefined },
    }),
  );
}

export function getExperiment(key: string, learnerId = "anonymous") {
  return unwrap<LearningExperimentDetail>(
    client.get(`/learning/experiments/${key}`, {
      params: { learner_id: learnerId },
    }),
  );
}

export function checkSubmission(key: string, payload: CheckPayload) {
  return unwrap<LearningReview>(
    client.post(`/learning/experiments/${key}/check`, {
      code: payload.code,
      dataset_id: payload.dataset_id ?? null,
      version: payload.version ?? null,
      learner_id: payload.learner_id ?? "anonymous",
      persist: payload.persist ?? true,
    }),
  );
}

export function saveDraft(
  key: string,
  code: string,
  datasetId?: number | null,
  learnerId = "anonymous",
) {
  return unwrap<{ saved: boolean; saved_at: string | null; progress: LearningProgressSummary }>(
    client.post(`/learning/experiments/${key}/draft`, {
      code,
      dataset_id: datasetId ?? null,
      learner_id: learnerId,
    }),
  );
}

export function resetProgress(key: string, learnerId = "anonymous") {
  return unwrap<{ reset: boolean; key: string }>(
    client.delete(`/learning/experiments/${key}/progress`, {
      params: { learner_id: learnerId },
    }),
  );
}

export function getOverview(learnerId = "anonymous") {
  return unwrap<LearningOverview>(
    client.get("/learning/overview", { params: { learner_id: learnerId } }),
  );
}

export function getLearningContext(datasetId: number, version?: number) {
  return unwrap<LearningContext>(
    client.get(`/learning/datasets/${datasetId}/learning-context`, {
      params: { version: version ?? undefined },
    }),
  );
}

// ---------------------------------------------------------------------------
// 自建学习卡片（practice / checklist）
// ---------------------------------------------------------------------------

export function listCustomCards(learnerId = "anonymous") {
  return unwrap<CustomCard[]>(
    client.get("/learning/custom-cards", { params: { learner_id: learnerId } }),
  );
}

export function getCustomCard(cardId: number, learnerId = "anonymous") {
  return unwrap<CustomCard>(
    client.get(`/learning/custom-cards/${cardId}`, {
      params: { learner_id: learnerId },
    }),
  );
}

export function createCustomCard(payload: CustomCardCreate) {
  return unwrap<CustomCard>(
    client.post("/learning/custom-cards", {
      title: payload.title,
      kind: payload.kind,
      goal: payload.goal ?? "",
      code: payload.code ?? "",
      items: payload.items ?? [],
      learner_id: payload.learner_id ?? "anonymous",
    }),
  );
}

export function updateCustomCard(cardId: number, payload: CustomCardUpdate) {
  return unwrap<CustomCard>(
    client.patch(`/learning/custom-cards/${cardId}`, {
      title: payload.title,
      goal: payload.goal,
      code: payload.code,
      items: payload.items,
      learner_id: payload.learner_id ?? "anonymous",
    }),
  );
}

export function deleteCustomCard(cardId: number, learnerId = "anonymous") {
  return unwrap<{ deleted: boolean; card_id: number }>(
    client.delete(`/learning/custom-cards/${cardId}`, {
      params: { learner_id: learnerId },
    }),
  );
}
