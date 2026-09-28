/**
 * 通用分阶段任务进度（`<TaskProgress>`）。
 *
 * 解决的问题：长耗时任务（ML 训练、Agent 执行、Workflow 运行）此前各自实现了一套
 * 「加载中」——有的只有一行灰字，有的拿定时器凑假百分比。用户分不清「在跑」和
 * 「卡死」，也看不到自己排在流程的哪一步。
 *
 * 三条设计原则：
 *
 * 1. **不伪造进度**。有 SSE / 轮询拿得到真实阶段就按阶段推进（`stages` + `currentStage`）；
 *    拿不到（同步接口）就显式进入不确定进度（`indeterminate`），只给流动条 + 预计耗时，
 *    不谎报百分比。
 * 2. **阶段名只允许有一个来源**。组件不内置任何业务阶段名，`stages` 一律由调用方
 *    从既有事实源注入（ML 取 `/ml/catalog`、Agent 取事件映射、Workflow 取节点顺序），
 *    避免界面说的阶段名和实际流程对不上。
 * 3. **终态要可操作**。成功不是显示一句「完成」，而是给「查看结果」；
 *    失败不是只甩一行红字，而是给失败原因 + 「重试」。
 *
 * 阶段定位与完成度推算在 `lib/taskProgress.ts`（纯函数、可单测），本文件只管渲染。
 */
import type { ReactNode } from "react";
import { resolveTaskProgress } from "../lib/taskProgress";
import type { TaskStageInput, TaskStageRef, TaskStatus } from "../lib/taskProgress";

export type { TaskStage, TaskStageInput, TaskStageRef, TaskStatus } from "../lib/taskProgress";

export interface TaskProgressProps {
  /** 阶段骨架，按执行顺序。字符串会被当作阶段名。 */
  stages?: TaskStageInput[];
  /** 当前阶段。`null` / 不传表示尚未定位。 */
  currentStage?: TaskStageRef | null;
  /** 已完成阶段。缺省时「当前阶段之前的都算完成」。 */
  doneStages?: TaskStageRef[];
  /** 显式进度 0-100；不传则按已完成阶段数推算。 */
  progress?: number | null;
  /** 任务状态，默认 `running`。 */
  status?: TaskStatus;
  /**
   * 不确定进度（无 SSE 的同步接口）。为 true 时不会自动高亮任何阶段 ——
   * 我们并不知道跑到哪一步了，随便点亮一格就是假信息。
   */
  indeterminate?: boolean;
  /** 进行中的标题，如「模型训练中」。 */
  title?: string;
  /** 预计耗时文案，如「预计需要 30 秒」。仅在没有真实百分比时展示。 */
  estimate?: string;
  /**
   * 进行中的一行补充说明（如当前具体步骤「执行工具：训练模型」）。
   *
   * 存在的理由：`stages` 是**粗粒度**归并（4 步），会丢掉「正在调哪个工具」这类
   * 用户真正关心的细节；直接丢掉信息不如多给一行。
   */
  note?: string;
  /** 已耗时（秒）。 */
  elapsed?: number;
  /** 各阶段完成后的真实摘要，key = 阶段 id 或名称（透明化：看得见每一步做了什么）。 */
  stageDetails?: Record<string, string>;
  /** 失败原因。`status="error"` 时展示在组件内，而不是散落在页面上别处。 */
  error?: string | null;
  /** 成功文案，默认「任务已完成」。 */
  successText?: string;
  /** 成功后「查看结果」的回调；不传则不渲染该按钮。 */
  onViewResult?: () => void;
  /** 失败后「重试」的回调；不传则不渲染该按钮。 */
  onRetry?: () => void;
  viewResultText?: string;
  retryText?: string;
  className?: string;
}

export function TaskProgress({
  stages = [],
  currentStage = null,
  doneStages = [],
  progress = null,
  status = "running",
  indeterminate = false,
  title = "任务进行中",
  estimate,
  note,
  elapsed,
  stageDetails,
  error,
  successText = "任务已完成",
  onViewResult,
  onRetry,
  viewResultText = "查看结果",
  retryText = "重试",
  className,
}: TaskProgressProps) {
  const model = resolveTaskProgress({ stages, currentStage, doneStages, progress, status, indeterminate });
  const { stages: list, doneIndexes, activeIndex, percent } = model;
  const done = new Set(doneIndexes);

  const heading = status === "success" ? successText : status === "error" ? "任务失败" : title;
  const isRunning = status === "running";
  const showTrack = isRunning || list.length > 0;
  const showEstimated = isRunning && Boolean(estimate) && (indeterminate || model.estimated);

  let footer: ReactNode = null;
  if (status === "success") {
    // 标题已承担「成功了」这句话，这里只放动作，避免同一句文案上下各出现一次。
    footer = onViewResult ? (
      <div className="task-progress-foot">
        <button type="button" className="btn primary" onClick={onViewResult}>
          {viewResultText}
        </button>
      </div>
    ) : null;
  } else if (status === "error") {
    footer = (
      <div className="task-progress-foot">
        <span className="task-progress-error" style={{ flex: 1 }}>
          {error || "任务未能完成，请重试。"}
        </span>
        {onRetry ? (
          <button type="button" className="btn" onClick={onRetry}>
            {retryText}
          </button>
        ) : null}
      </div>
    );
  } else if (showEstimated) {
    // 没有真实进度时（同步接口，或流式任务刚发出、还没收到第一条事件），
    // 「还要多久」是用户唯一能拿到的确定信息，必须显式给出。
    // 一旦拿到了真实百分比就不再显示 —— 免得「已 87%」和「预计 30 秒」互相打架。
    footer = <div className="task-progress-foot muted">{estimate}</div>;
  }

  return (
    <div
      className={`task-progress is-${status}${indeterminate && isRunning ? " is-indeterminate" : ""}${
        className ? ` ${className}` : ""
      }`}
      role="status"
      aria-live="polite"
    >
      <div className="task-progress-head">
        <span className="task-progress-title">
          {isRunning && indeterminate ? <span className="spinner" aria-hidden="true" /> : null}
          {status === "success" ? <span aria-hidden="true">✓</span> : null}
          {status === "error" ? <span aria-hidden="true">✕</span> : null}
          <span>{heading}</span>
        </span>
        <span className="muted task-progress-meta">
          {isRunning && !indeterminate ? `${percent}%` : null}
          {typeof elapsed === "number" && elapsed > 0 ? `已耗时 ${elapsed.toFixed(1)}s` : null}
        </span>
      </div>

      {isRunning && note ? <div className="task-progress-note muted">{note}</div> : null}

      {showTrack ? (
        indeterminate && isRunning ? (
          <div className="task-progress-track is-running" role="progressbar" aria-valuetext="进行中">
            <div className="task-progress-fill" />
          </div>
        ) : (
          <div
            className="task-progress-track"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
          >
            <div className="task-progress-fill" style={{ width: `${percent}%` }} />
          </div>
        )
      ) : null}

      {list.length > 0 ? (
        <ol className="task-progress-steps">
          {list.map((stage, index) => {
            const isDone = done.has(index);
            const isCurrent = !isDone && index === activeIndex;
            const detail = stage.detail ?? stageDetails?.[stage.id] ?? stageDetails?.[stage.name];
            return (
              <li key={`${stage.id}-${index}`} className={isDone ? "is-done" : isCurrent ? "is-current" : ""}>
                <span className="task-progress-step-name">{stage.name}</span>
                {isDone && detail ? <span className="task-progress-step-detail">✓ {detail}</span> : null}
                {isCurrent ? <span className="task-progress-step-detail">…进行中</span> : null}
              </li>
            );
          })}
        </ol>
      ) : null}

      {footer}
    </div>
  );
}
