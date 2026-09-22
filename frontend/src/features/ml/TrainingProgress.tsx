import type { TrainProgress } from "../../types/ml";

export interface TrainStage {
  id: string;
  name: string;
}

interface Props {
  /** 训练请求进行中 */
  active: boolean;
  /** 已耗时（秒），由调用方在请求期间累计 */
  elapsed?: number;
  /** 完整阶段骨架（按训练顺序；聚类任务应由调用方过滤掉 split）。 */
  stages: TrainStage[];
  /** 已完成的进度事件（按发生顺序）。 */
  completed: TrainProgress[];
}

/**
 * 实时训练进度条。
 *
 * 数据源是 `POST /ml/train/stream` 的 progress 事件（后端每个阶段完成时推一条，
 * 带真实 done/total/detail），不是前端定时器伪造。已完成的阶段会回显该步真实
 * 摘要（行数、切分比例、指标等），让进度条同时承担「透明化」：看得见每一步做了什么。
 */
export function TrainingProgress({ active, elapsed = 0, stages = [], completed = [] }: Props) {
  if (!active) return null;

  const latest = completed[completed.length - 1];
  const done = latest?.done ?? 0;
  const total = latest?.total ?? stages.length;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  // 已完成 done 个 → 下一个（下标 = done）即「进行中」
  const currentIndex = done;

  return (
    <div className="ml-training-progress mt">
      <div className="ml-tp-head">
        <strong>训练进行中</strong>
        <span className="muted">已耗时 {elapsed.toFixed(1)}s · {pct}%</span>
      </div>
      <div className="ml-progress-track">
        <div className="ml-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <ol className="ml-tp-steps">
        {stages.map((s, i) => {
          const ev = completed.find((c) => c.stage === s.id);
          const isDone = Boolean(ev);
          const isCurrent = !isDone && i === currentIndex;
          return (
            <li key={s.id} className={isDone ? "is-done" : isCurrent ? "is-current" : ""}>
              <span className="ml-tp-name">{s.name}</span>
              {isDone && ev && <span className="muted">✓ {ev.detail}</span>}
              {isCurrent && <span className="muted">…进行中</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
