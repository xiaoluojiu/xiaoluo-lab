import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { PageHeader } from "../components/PageHeader";
import { EmptyState, ErrorState, Skeleton } from "../components/StateBlock";
import { useToast } from "../components/ToastProvider";
import {
  createCustomCard,
  deleteCustomCard,
  getOverview,
  listCustomCards,
  listExperiments,
  listTracks,
  type CustomCard,
  type CustomCardItem,
  type CustomCardKind,
  type LearningExperiment,
  type LearningOverview,
  type LearningTrack,
  type ProgressStatus,
  type TrackId,
} from "../api/learning";
import "./learning.css";

const STATUS_LABEL: Record<ProgressStatus, string> = {
  not_started: "未开始",
  in_progress: "进行中",
  completed: "已完成",
};

const LEVEL_CLASS: Record<string, string> = {
  基础: "level-basic",
  进阶: "level-advanced",
  挑战: "level-challenge",
};

const KIND_LABEL: Record<CustomCardKind, string> = {
  practice: "练习卡",
  checklist: "清单卡",
};

type ViewFilter = TrackId | "all" | "custom";

type PageState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | {
      kind: "ready";
      tracks: LearningTrack[];
      experiments: LearningExperiment[];
      overview: LearningOverview;
      customCards: CustomCard[];
    };

export default function Learning() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const toast = useToast();
  const [state, setState] = useState<PageState>({ kind: "loading" });
  const [trackFilter, setTrackFilter] = useState<ViewFilter>("all");
  const [keyword, setKeyword] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);

  // 从 /learning?tab=custom 进入时直接落到「我的卡片」视图
  useEffect(() => {
    const tab = params.get("tab");
    if (tab === "custom") setTrackFilter("custom");
  }, [params]);

  const load = useCallback(() => {
    setState({ kind: "loading" });
    Promise.all([listTracks(), listExperiments(), getOverview(), listCustomCards()])
      .then(([tracks, experiments, overview, customCards]) => {
        setState({ kind: "ready", tracks, experiments, overview, customCards });
      })
      .catch((e: unknown) => {
        setState({
          kind: "error",
          message: e instanceof Error ? e.message : "学习中心加载失败",
        });
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(() => {
    if (state.kind !== "ready") return [];
    const kw = keyword.trim().toLowerCase();
    return state.experiments.filter((exp) => {
      if (trackFilter !== "all" && exp.track !== trackFilter) return false;
      if (!kw) return true;
      return (
        exp.title.toLowerCase().includes(kw) ||
        exp.summary.toLowerCase().includes(kw) ||
        exp.goal.toLowerCase().includes(kw) ||
        exp.requirements.some((r) => r.toLowerCase().includes(kw))
      );
    });
  }, [state, trackFilter, keyword]);

  // 自定义卡片的搜索：按标题 / 目标 / 清单要点过滤，让搜索框同样覆盖「我的卡片」。
  const visibleCustomCards = useMemo(() => {
    if (state.kind !== "ready") return [];
    const kw = keyword.trim().toLowerCase();
    if (!kw) return state.customCards;
    return state.customCards.filter((card) => {
      if (card.title.toLowerCase().includes(kw)) return true;
      if (card.goal.toLowerCase().includes(kw)) return true;
      return card.items.some((it) => it.text.toLowerCase().includes(kw));
    });
  }, [state, keyword]);

  async function handleDeleteCard(card: CustomCard) {
    try {
      await deleteCustomCard(card.id);
      toast.success(`已删除「${card.title}」`);
      load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "删除失败");
    }
  }

  if (state.kind === "loading") {
    return (
      <div className="learning-page">
        <PageHeader title="学习中心" />
        <div className="card learning-panel">
          <Skeleton lines={5} card />
        </div>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="learning-page">
        <PageHeader title="学习中心" />
        <ErrorState
          title="学习中心加载失败"
          message={state.message}
          onRetry={load}
        />
      </div>
    );
  }

  const { tracks, overview, customCards } = state;
  const nextSpec = overview.recommended
    ? state.experiments.find((e) => e.key === overview.recommended)
    : null;

  // 我的卡片分组（「我的卡片」视图与「全部」视图共用）。
  const customSection = (
    <section className="learning-track">
      <div className="learning-track-head">
        <h2>我的卡片</h2>
        <p>练习卡写代码交给 AI 点评，清单卡记录要点手动勾选。</p>
      </div>
      <div className="learning-grid">
        {visibleCustomCards.map((card) => (
          <CustomCardTile
            key={card.id}
            card={card}
            onDelete={() => handleDeleteCard(card)}
          />
        ))}
        <button type="button" className="learning-card-new" onClick={() => setDialogOpen(true)}>
          <span className="learning-card-new-plus" aria-hidden="true">
            +
          </span>
          <span className="learning-card-new-title">新建学习卡片</span>
          <span className="learning-card-new-sub">练习卡或清单卡，随手记一个目标</span>
        </button>
      </div>
    </section>
  );

  return (
    <div className="learning-page">
      <PageHeader
        title="学习中心"
        actions={
          <>
            <button
              className="btn"
              type="button"
              onClick={() => setDialogOpen(true)}
            >
              新建学习卡片
            </button>
            <Link className="btn" to="/learning/history">
              学习记录
            </Link>
          </>
        }
      />

      {/* ---- 总览 ---- */}
      <section className="learning-overview">
        <div className="card learning-stat">
          <span className="learning-stat-value">
            {overview.completed}
            <em>/{overview.total}</em>
          </span>
          <span className="learning-stat-label">已完成实验</span>
          <div className="learning-progress-bar" aria-hidden="true">
            <span style={{ width: `${overview.completion_rate}%` }} />
          </div>
        </div>
        {overview.tracks.map((t) => (
          <div className="card learning-stat" key={t.id}>
            <span className="learning-stat-value">
              {t.completed}
              <em>/{t.total}</em>
            </span>
            <span className="learning-stat-label">{t.label}</span>
            <span className="learning-stat-sub">
              {t.in_progress > 0 ? `${t.in_progress} 个进行中` : "暂无进行中"}
            </span>
          </div>
        ))}
      </section>

      {/* ---- 推荐下一步（仅在实验视图显示） ---- */}
      {trackFilter !== "custom" &&
        (nextSpec ? (
          <section className="card learning-next">
            <div className="learning-next-main">
              <span className="learning-kicker">建议下一步</span>
              <h2>{nextSpec.title}</h2>
            </div>
            <button
              className="btn primary"
              type="button"
              onClick={() => navigate(`/learning/workspace?experiment=${nextSpec.key}`)}
            >
              开始实验
            </button>
          </section>
        ) : (
          <section className="card learning-next">
            <div className="learning-next-main">
              <span className="learning-kicker">全部完成</span>
              <h2>两条主线都已通关</h2>
              <p>可以回到任一实验，用不同的数据集或不同的算法重做一遍，观察结论如何变化。</p>
            </div>
          </section>
        ))}

      {/* ---- 筛选 ---- */}
      <section className="learning-toolbar">
        <div className="learning-filter-tabs" role="tablist" aria-label="按主线筛选">
          <button
            type="button"
            role="tab"
            aria-selected={trackFilter === "all"}
            className={`learning-filter-tab${trackFilter === "all" ? " active" : ""}`}
            onClick={() => setTrackFilter("all")}
          >
            全部 <span>{state.experiments.length + customCards.length}</span>
          </button>
          {tracks.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={trackFilter === t.id}
              className={`learning-filter-tab${trackFilter === t.id ? " active" : ""}`}
              onClick={() => setTrackFilter(t.id)}
            >
              {t.label}{" "}
              <span>
                {state.experiments.filter((e) => e.track === t.id).length}
              </span>
            </button>
          ))}
          <button
            type="button"
            role="tab"
            aria-selected={trackFilter === "custom"}
            className={`learning-filter-tab${trackFilter === "custom" ? " active" : ""}`}
            onClick={() => setTrackFilter("custom")}
          >
            我的卡片 <span>{customCards.length}</span>
          </button>
        </div>
        <input
          className="learning-search"
          type="search"
          value={keyword}
          placeholder="搜索实验或学习卡片…"
          aria-label="搜索实验或卡片"
          onChange={(e) => setKeyword(e.target.value)}
        />
      </section>

      {/* ---- 实验 / 卡片列表 ---- */}
      {trackFilter === "custom" ? (
        keyword && visibleCustomCards.length === 0 ? (
          <EmptyState
            title="没有匹配的学习卡片"
            description={`没有标题、目标或要点包含「${keyword}」的卡片，换个关键词试试。`}
            action={
              <button className="btn btn-sm" type="button" onClick={() => setKeyword("")}>
                清空搜索
              </button>
            }
          />
        ) : (
          customSection
        )
      ) : (
        <>
          {tracks
            .filter((t) => trackFilter === "all" || t.id === trackFilter)
            .map((track) => {
              const items = visible.filter((e) => e.track === track.id);
              if (items.length === 0) return null;
              return (
                <section className="learning-track" key={track.id}>
                  <div className="learning-track-head">
                    <h2>{track.label}</h2>
                  </div>
                  <div className="learning-grid">
                    {items.map((exp) => (
                      <ExperimentCard key={exp.key} experiment={exp} />
                    ))}
                  </div>
                </section>
              );
            })}
          {trackFilter === "all" && visibleCustomCards.length > 0 && customSection}
          {visible.length === 0 &&
            !(trackFilter === "all" && visibleCustomCards.length > 0) && (
              <EmptyState
                title={keyword ? "没有匹配的内容" : "暂无实验"}
                description={
                  keyword
                    ? `没有匹配「${keyword}」的实验或卡片，换个关键词试试。`
                    : "当前主线暂无实验。"
                }
                action={
                  keyword ? (
                    <button className="btn btn-sm" type="button" onClick={() => setKeyword("")}>
                      清空搜索
                    </button>
                  ) : null
                }
              />
            )}
        </>
      )}

      <NewCardDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onCreated={(card) => {
          setDialogOpen(false);
          load();
          navigate(`/learning/card/${card.id}`);
        }}
      />
    </div>
  );
}

function ExperimentCard({ experiment }: { experiment: LearningExperiment }) {
  const { progress } = experiment;
  const locked = Boolean(
    experiment.prerequisite &&
      progress.status === "not_started",
  );

  return (
    <article className={`card learning-card ${progress.status}`}>
      <div className="learning-card-head">
        <span className={`learning-level ${LEVEL_CLASS[experiment.level] ?? ""}`}>
          {experiment.level}
        </span>
        <span className={`learning-status ${progress.status}`}>
          {STATUS_LABEL[progress.status]}
          {progress.status === "completed" && progress.score > 0
            ? ` ${progress.score} 分`
            : ""}
        </span>
      </div>

      <h3>{experiment.title}</h3>
      <p className="learning-card-summary">{experiment.summary}</p>

      <ul className="learning-card-meta">
        <li>
          <span>产出</span>
          {experiment.requirements.length} 项要求
        </li>
        <li>
          <span>数据</span>
          {experiment.needs}
        </li>
        <li>
          <span>投入</span>
          约 {experiment.minutes} 分钟
        </li>
        {progress.attempts > 0 ? (
          <li>
            <span>尝试</span>
            已提交 {progress.attempts} 次
          </li>
        ) : null}
      </ul>

      <div className="learning-card-foot">
        {locked ? (
          <span className="learning-locked-hint">
            建议先完成前一个实验，也可以直接开始
          </span>
        ) : null}
        <Link
          className="btn btn-sm primary"
          to={`/learning/workspace?experiment=${experiment.key}`}
        >
          {progress.status === "completed"
            ? "再做一次"
            : progress.status === "in_progress"
              ? "继续实验"
              : "开始实验"}
        </Link>
      </div>
    </article>
  );
}

/** 自建学习卡片瓦片。 */
function CustomCardTile({
  card,
  onDelete,
}: {
  card: CustomCard;
  onDelete: () => void;
}) {
  return (
    <article className={`card learning-card learning-card-custom ${card.kind}`}>
      <div className="learning-card-head">
        <span className={`learning-level ${card.kind === "practice" ? "level-basic" : "level-advanced"}`}>
          {KIND_LABEL[card.kind]}
        </span>
        {card.kind === "checklist" ? (
          <span className="learning-status in_progress">
            {card.progress.done}/{card.progress.total} 完成
          </span>
        ) : (
          <span className="learning-muted">
            {card.updated_at
              ? `更新于 ${new Date(card.updated_at).toLocaleDateString("zh-CN")}`
              : ""}
          </span>
        )}
      </div>

      <h3>{card.title}</h3>
      <p className="learning-card-summary">
        {card.kind === "practice"
          ? card.goal || "（未写目标描述）"
          : card.items.length
            ? `${card.items.length} 个要点`
            : "（还没有要点）"}
      </p>

      {card.kind === "checklist" && card.items.length > 0 ? (
        <div className="learning-progress-bar" aria-hidden="true">
          <span style={{ width: `${card.progress.rate}%` }} />
        </div>
      ) : null}

      <div className="learning-card-foot">
        <button
          className="btn btn-sm learning-card-del"
          type="button"
          onClick={onDelete}
        >
          删除
        </button>
        <Link className="btn btn-sm primary" to={`/learning/card/${card.id}`}>
          {card.kind === "practice" ? "打开练习" : "打开清单"}
        </Link>
      </div>
    </article>
  );
}

/** 新建学习卡片对话框。 */
function NewCardDialog({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (card: CustomCard) => void;
}) {
  const toast = useToast();
  const [kind, setKind] = useState<CustomCardKind>("practice");
  const [title, setTitle] = useState("");
  const [goal, setGoal] = useState("");
  const [code, setCode] = useState("");
  const [items, setItems] = useState<CustomCardItem[]>([]);
  const [itemDraft, setItemDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (open) {
      setKind("practice");
      setTitle("");
      setGoal("");
      setCode("");
      setItems([]);
      setItemDraft("");
    }
  }, [open]);

  if (!open) return null;

  function addItem() {
    const t = itemDraft.trim();
    if (!t) return;
    setItems((prev) => [...prev, { text: t, done: false }]);
    setItemDraft("");
  }

  async function handleSubmit() {
    if (!title.trim()) {
      toast.error("给卡片起个标题");
      return;
    }
    setSubmitting(true);
    try {
      const card = await createCustomCard({
        title: title.trim(),
        kind,
        goal: kind === "practice" ? goal.trim() : "",
        code: kind === "practice" ? code : "",
        items: kind === "checklist" ? items : [],
      });
      toast.success("学习卡片已创建");
      onCreated(card);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "创建失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="dialog-mask" onClick={onClose}>
      <div className="dialog learning-new-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>新建学习卡片</h3>
        <p className="learning-new-dialog-intro">
          记下一个想练的题目或想跟进的清单，不进主线、不做判分。
        </p>

        <label className="learning-field">
          卡片类型
          <div className="learning-kind-switch" role="radiogroup" aria-label="卡片类型">
            {(["practice", "checklist"] as CustomCardKind[]).map((k) => (
              <button
                key={k}
                type="button"
                role="radio"
                aria-checked={kind === k}
                className={`learning-kind-option${kind === k ? " active" : ""}`}
                onClick={() => setKind(k)}
              >
                {KIND_LABEL[k]}
                <span>{k === "practice" ? "写代码 · AI 点评" : "要点 · 手动勾选"}</span>
              </button>
            ))}
          </div>
        </label>

        <label className="learning-field">
          标题
          <input
            type="text"
            value={title}
            placeholder="例如：手写一个二分查找"
            onChange={(e) => setTitle(e.target.value)}
            autoFocus
          />
        </label>

        <label className="learning-field">
          {kind === "practice" ? "目标描述（给 AI 点评做依据）" : "目标描述（可选）"}
          <textarea
            rows={2}
            value={goal}
            placeholder="想通过这个练习掌握什么？写清楚，AI 才能针对性地点评。"
            onChange={(e) => setGoal(e.target.value)}
          />
        </label>

        {kind === "practice" ? (
          <label className="learning-field">
            起始代码（可选，可留空）
            <textarea
              className="learning-new-code"
              rows={5}
              value={code}
              placeholder={"# 可以先写个开头，进入后还能继续改\n# 例如：\ndef binary_search(arr, target):\n    ..."}
              spellCheck={false}
              onChange={(e) => setCode(e.target.value)}
            />
          </label>
        ) : (
          <div className="learning-field">
            要点清单
            <div className="learning-items-editor">
              {items.length > 0 ? (
                <ul className="learning-items-list">
                  {items.map((it, i) => (
                    <li key={i}>
                      <span>{it.text}</span>
                      <button
                        type="button"
                        aria-label={`删除第 ${i + 1} 项`}
                        onClick={() =>
                          setItems((prev) => prev.filter((_, idx) => idx !== i))
                        }
                      >
                        ✕
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="learning-muted">还没有要点，先加一条吧。</p>
              )}
              <div className="learning-items-add">
                <input
                  type="text"
                  value={itemDraft}
                  placeholder="例如：看懂注意力机制"
                  onChange={(e) => setItemDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addItem();
                    }
                  }}
                />
                <button className="btn btn-sm" type="button" onClick={addItem}>
                  添加
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="actions">
          <button className="btn" type="button" onClick={onClose}>
            取消
          </button>
          <button
            className="btn primary"
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitting ? "创建中…" : "创建卡片"}
          </button>
        </div>
      </div>
    </div>
  );
}
