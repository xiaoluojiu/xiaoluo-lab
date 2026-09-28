/**
 * `<TaskProgress>` 的**纯模型层**：阶段归一、当前阶段定位、完成度推算。
 *
 * 为什么把这段逻辑从组件里拆出来（与 `lib/agentEvents.ts` 同一套理由）：
 * 真正的边界条件都在这里，而不是在 JSX 里 ——
 *   1. 阶段引用有「下标 / id / 名称」三种写法，调用方各写各的；
 *   2. 顺序任务的中间事件偶尔丢一条，进度**不允许因此回退**；
 *   3. 不确定进度（同步接口）下不能乱猜当前阶段，否则点亮的格子是假信息。
 * 埋进 JSX 就只能靠肉眼看界面回归；放在 lib/ 下可被 Node 原生 runner 直接导入测试
 * （零 React 依赖，也不需要 JSX 转译）。
 */

/** 阶段描述：`id` 用于与进度事件对齐，`name` 用于展示。 */
export interface TaskStage {
  id: string;
  name: string;
  detail?: string;
}

/** 允许只写中文阶段名（如 `['数据加载','特征工程']`），此时 id 与 name 同名。 */
export type TaskStageInput = string | TaskStage;

/** 阶段定位：下标 / 阶段 id / 阶段名三种写法都接受。 */
export type TaskStageRef = string | number;

export type TaskStatus = "running" | "success" | "error";

/** 把 `['数据加载', { id, name }]` 归一成 `TaskStage[]`。 */
export function normalizeStages(stages: TaskStageInput[] = []): TaskStage[] {
  return stages.map((stage, index) => {
    if (typeof stage === "string") return { id: stage, name: stage };
    return {
      id: stage.id || String(index),
      name: stage.name || stage.id,
      detail: stage.detail,
    };
  });
}

/** 把阶段引用解析成下标；解析不出来返回 -1（调用方据此走兜底）。 */
export function resolveStageIndex(stages: TaskStage[], ref: TaskStageRef | null | undefined): number {
  if (ref == null) return -1;
  if (typeof ref === "number") {
    return Number.isInteger(ref) && ref >= 0 && ref < stages.length ? ref : -1;
  }
  return stages.findIndex((stage) => stage.id === ref || stage.name === ref);
}

/** 收敛到 0-100 的整数百分比。 */
export function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

export interface TaskProgressModel {
  stages: TaskStage[];
  /** 已完成阶段下标（升序）。 */
  doneIndexes: number[];
  /** 当前阶段下标；-1 表示没有可高亮的阶段。 */
  activeIndex: number;
  /** 0-100。 */
  percent: number;
  /** true 表示百分比是按「已完成阶段数」推算的，而不是调用方给的真实进度。 */
  estimated: boolean;
}

export interface TaskProgressInput {
  stages?: TaskStageInput[];
  currentStage?: TaskStageRef | null;
  doneStages?: TaskStageRef[];
  progress?: number | null;
  status?: TaskStatus;
  /** 不确定进度：不自动推断当前阶段，百分比也不谎报。 */
  indeterminate?: boolean;
}

export function resolveTaskProgress(input: TaskProgressInput = {}): TaskProgressModel {
  const {
    currentStage = null,
    doneStages = [],
    progress = null,
    status = "running",
    indeterminate = false,
  } = input;
  const stages = normalizeStages(input.stages);
  const total = stages.length;

  const done = new Set<number>();
  for (const ref of doneStages) {
    const index = resolveStageIndex(stages, ref);
    if (index >= 0) done.add(index);
  }

  let activeIndex = resolveStageIndex(stages, currentStage);
  // 只有在「确实知道自己在哪」时才兜底推断当前阶段；不确定进度下不猜。
  if (activeIndex < 0 && total > 0 && status === "running" && !indeterminate) {
    activeIndex = stages.findIndex((_, index) => !done.has(index));
  }
  // 当前阶段之前的一律视为已完成：阶段是顺序执行的，中间丢一条事件
  // 也不该让进度条**倒退**（历史 bug：进度条倒着走）。
  if (activeIndex > 0) {
    for (let index = 0; index < activeIndex; index += 1) done.add(index);
  }
  // 成功即全部阶段完成。放在这里而不是让每个调用方自己铺一遍 doneStages ——
  // 漏铺一格就会出现「全部跑完了，最后一步还是灰的」。
  if (status === "success") {
    for (let index = 0; index < total; index += 1) done.add(index);
    activeIndex = -1;
  }

  const estimatedPercent = total > 0 ? clampPercent((done.size / total) * 100) : 0;
  const percent =
    status === "success" ? 100 : progress != null ? clampPercent(progress) : estimatedPercent;

  return {
    stages,
    doneIndexes: [...done].sort((a, b) => a - b),
    activeIndex,
    percent,
    estimated: status !== "success" && progress == null,
  };
}
