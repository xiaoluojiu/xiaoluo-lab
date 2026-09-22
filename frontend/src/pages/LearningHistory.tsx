import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { PageHeader } from "../components/PageHeader";
import { EmptyState, ErrorState, Skeleton } from "../components/StateBlock";
import {
  getOverview,
  listExperiments,
  type LearningExperiment,
  type LearningOverview,
  type ProgressStatus,
} from "../api/learning";
import "./learning.css";

const STATUS_LABEL: Record<ProgressStatus, string> = {
  not_started: "未开始",
  in_progress: "进行中",
  completed: "已完成",
};

type Filter = "all" | "completed" | "in_progress" | "not_started";

type PageState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; experiments: LearningExperiment[]; overview: LearningOverview };

export default function LearningHistory() {
  const [state, setState] = useState<PageState>({ kind: "loading" });
  const [filter, setFilter] = useState<Filter>("all");

  const load = useCallback(() => {
    setState({ kind: "loading" });
    Promise.all([listExperiments(), getOverview()])
      .then(([experiments, overview]) =>
        setState({ kind: "ready", experiments, overview }),
      )
      .catch((e: unknown) =>
        setState({
          kind: "error",
          message: e instanceof Error ? e.message : "学习记录加载失败",
        }),
      );
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(() => {
    if (state.kind !== "ready") return [];
    if (filter === "all") return state.experiments;
    return state.experiments.filter((e) => e.progress.status === filter);
  }, [state, filter]);

  const counts = useMemo(() => {
    if (state.kind !== "ready") return { all: 0, completed: 0, in_progress: 0, not_started: 0 };
    const acc = { all: state.experiments.length, completed: 0, in_progress: 0, not_started: 0 };
    for (const e of state.experiments) acc[e.progress.status] += 1;
    return acc;
  }, [state]);

  if (state.kind === "loading") {
    return (
      <div className="learning-page">
        <PageHeader title="学习记录" breadcrumbs={<Link to="/learning">学习中心</Link>} />
        <div className="card learning-panel">
          <Skeleton lines={4} card />
        </div>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="learning-page">
        <PageHeader title="学习记录" breadcrumbs={<Link to="/learning">学习中心</Link>} />
        <ErrorState title="学习记录加载失败" message={state.message} onRetry={load} />
      </div>
    );
  }

  const { overview } = state;

  return (
    <div className="learning-page">
      <PageHeader
        title="学习记录"
        breadcrumbs={<Link to="/learning">学习中心</Link>}
        actions={
          <Link className="btn" to="/learning">
            返回学习中心
          </Link>
        }
      />

      <section className="learning-overview">
        <div className="card learning-stat">
          <span className="learning-stat-value">
            {overview.completed}
            <em>/{overview.total}</em>
          </span>
          <span className="learning-stat-label">已完成</span>
          <div className="learning-progress-bar" aria-hidden="true">
            <span style={{ width: `${overview.completion_rate}%` }} />
          </div>
        </div>
        <div className="card learning-stat">
          <span className="learning-stat-value">{overview.started}</span>
          <span className="learning-stat-label">已开始</span>
          <span className="learning-stat-sub">
            还有 {overview.total - overview.started} 个没碰过
          </span>
        </div>
        {overview.tracks.map((t) => (
          <div className="card learning-stat" key={t.id}>
            <span className="learning-stat-value">
              {t.completed}
              <em>/{t.total}</em>
            </span>
            <span className="learning-stat-label">{t.label}</span>
          </div>
        ))}
      </section>

      {overview.recent.length > 0 ? (
        <section className="card learning-panel">
          <div className="panel-title">
            <span>最近活动</span>
            <span className="learning-muted">按更新时间排序</span>
          </div>
          <ul className="recent-list">
            {overview.recent.map((r) => (
              <li key={r.key}>
                <Link to={`/learning/workspace?experiment=${r.key}`}>{r.title}</Link>
                <span className={`learning-status ${r.status}`}>
                  {STATUS_LABEL[r.status]}
                </span>
                <span className="learning-muted">
                  {r.attempts > 0 ? `提交 ${r.attempts} 次 · ${r.score} 分` : "尚未提交"}
                </span>
                <span className="learning-muted">
                  {r.updated_at
                    ? new Date(r.updated_at).toLocaleString("zh-CN")
                    : "—"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="learning-toolbar">
        <div className="learning-filter-tabs" role="tablist" aria-label="按状态筛选">
          {(
            [
              ["all", "全部"],
              ["completed", "已完成"],
              ["in_progress", "进行中"],
              ["not_started", "未开始"],
            ] as [Filter, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={filter === id}
              className={`learning-filter-tab${filter === id ? " active" : ""}`}
              onClick={() => setFilter(id)}
            >
              {label} <span>{counts[id]}</span>
            </button>
          ))}
        </div>
      </section>

      {visible.length === 0 ? (
        <EmptyState
          title="这个状态下没有实验"
          description={
            filter === "in_progress"
              ? "还没有正在进行的实验，挑一个开始吧。"
              : filter === "completed"
                ? "还没有完成的实验。完成一次客观检查的全部要求就会记在这里。"
                : "全部实验都已经开始了，去学习中心继续推进。"
          }
          action={
            <Link className="btn btn-sm primary" to="/learning">
              去学习中心
            </Link>
          }
        />
      ) : (
        <div className="history-list">
          {visible.map((exp) => (
            <article className={`card history-item ${exp.progress.status}`} key={exp.key}>
              <div className="history-main">
                <div className="history-head">
                  <span className={`learning-track-tag track-${exp.track}`}>
                    {exp.track === "ml" ? "机器学习" : "大模型"}
                  </span>
                  <span className="learning-level">{exp.level}</span>
                  <span className={`learning-status ${exp.progress.status}`}>
                    {STATUS_LABEL[exp.progress.status]}
                  </span>
                </div>
                <h3>{exp.title}</h3>
                <div className="history-meta">
                  <span>
                    {exp.progress.attempts > 0
                      ? `提交 ${exp.progress.attempts} 次`
                      : "尚未提交"}
                  </span>
                  {exp.progress.status === "completed" ? (
                    <span className="score-ok">{exp.progress.score} 分</span>
                  ) : exp.progress.attempts > 0 ? (
                    <span>当前 {exp.progress.score} 分</span>
                  ) : null}
                  {exp.progress.completed_at ? (
                    <span>
                      完成于 {new Date(exp.progress.completed_at).toLocaleString("zh-CN")}
                    </span>
                  ) : exp.progress.saved_at ? (
                    <span>
                      最后保存 {new Date(exp.progress.saved_at).toLocaleString("zh-CN")}
                    </span>
                  ) : null}
                </div>
              </div>
              <Link
                className="btn btn-sm"
                to={`/learning/workspace?experiment=${exp.key}`}
              >
                {exp.progress.status === "not_started" ? "开始" : "查看"}
              </Link>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
