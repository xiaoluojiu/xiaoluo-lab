import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { createSession, sendMessage } from "../api/agent";
import { listDatasets } from "../api/datasets";
import { useToast } from "../components/ToastProvider";
import { ConfirmDialog } from "../components/ConfirmDialog";
import {
  checkSubmission,
  getExperiment,
  getLearningContext,
  resetProgress,
  saveDraft,
  type LearningCheck,
  type LearningContext,
  type LearningExperimentDetail,
  type LearningReview,
} from "../api/learning";
import type { Dataset } from "../types/dataset";
import "./learning.css";

type HelpMode = "check" | "hint" | "explain" | "data";

const HELP_MODES: { id: HelpMode; label: string; prompt: string }[] = [
  {
    id: "check",
    label: "检查我的代码",
    prompt: "先检查完成情况和最重要的一个问题，不要直接给完整答案。",
  },
  {
    id: "hint",
    label: "给我一个提示",
    prompt: "只给一个能推动学生继续思考的提示，不要泄露完整实现。",
  },
  {
    id: "explain",
    label: "解释这个问题",
    prompt: "用初学者能理解的方式解释最关键的问题，并用当前实验上下文举例。",
  },
  {
    id: "data",
    label: "结合数据分析",
    prompt: "重点结合当前真实数据集的字段、规模和可见样本分析代码是否合理；不要编造统计结果。",
  },
];

/** 页面级加载状态：区分「加载中 / 不存在 / 出错」，不再把三者混成空数组。 */
type PageState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; experiment: LearningExperimentDetail };

export default function LearningWorkspace() {
  const [params, setParams] = useSearchParams();
  const experimentKey = params.get("experiment") ?? "";
  const toast = useToast();

  const [page, setPage] = useState<PageState>({ kind: "loading" });
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [context, setContext] = useState<LearningContext | null>(null);
  const [contextError, setContextError] = useState<string | null>(null);

  const [code, setCode] = useState("");
  const [review, setReview] = useState<LearningReview | null>(null);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  // 光标位置（行/列），给高编辑器一个可读的定位反馈
  const [caret, setCaret] = useState({ line: 1, col: 1 });

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [lastMode, setLastMode] = useState<HelpMode | null>(null);

  const [confirmReset, setConfirmReset] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // ---------------------------------------------------------------
  // 加载实验（含草稿）。非法 key 不再静默回退到第一个实验，
  // 而是明确告知并给出回列表的路。
  // ---------------------------------------------------------------
  const loadExperiment = useCallback(
    (key: string) => {
      if (!key) {
        setPage({
          kind: "error",
          message: "地址里没有指定实验。请从学习中心选择一个实验进入。",
        });
        return;
      }
      setPage({ kind: "loading" });
      getExperiment(key)
        .then((experiment) => {
          setPage({ kind: "ready", experiment });
          setCode(experiment.progress.code || experiment.starter);
          setDirty(false);
          const savedDataset = experiment.progress.dataset_id;
          setDatasetId(savedDataset ?? "");
          setReview(
            experiment.progress.last_result &&
              typeof experiment.progress.last_result.score === "number"
              ? (experiment.progress.last_result as LearningReview)
              : null,
          );
        })
        .catch((e: unknown) => {
          setPage({
            kind: "error",
            message:
              e instanceof Error
                ? e.message
                : `找不到实验「${key}」，它可能已被移除。`,
          });
        });
    },
    [],
  );

  useEffect(() => {
    loadExperiment(experimentKey);
  }, [experimentKey, loadExperiment]);

  // 切换实验时清空对话与残留状态
  useEffect(() => {
    setSessionId(null);
    setFeedback("");
    setLastMode(null);
    return () => {
      abortRef.current?.abort();
    };
  }, [experimentKey]);

  // 数据集列表（失败要显式暴露，不能吞成空数组）
  useEffect(() => {
    listDatasets(1, 100)
      .then((r) => setDatasets(r.items))
      .catch((e: unknown) => {
        toast.error(
          e instanceof Error ? `数据集列表加载失败：${e.message}` : "数据集列表加载失败",
        );
      });
  }, [toast]);

  // 数据体检
  useEffect(() => {
    if (!datasetId) {
      setContext(null);
      setContextError(null);
      return;
    }
    let alive = true;
    setContext(null);
    setContextError(null);
    getLearningContext(Number(datasetId))
      .then((c) => {
        if (alive) setContext(c);
      })
      .catch((e: unknown) => {
        if (alive) {
          setContextError(
            e instanceof Error ? e.message : "数据集信息读取失败",
          );
        }
      });
    return () => {
      alive = false;
    };
  }, [datasetId]);

  const experiment = page.kind === "ready" ? page.experiment : null;

  const achievedCount = useMemo(() => {
    if (!review) return 0;
    const total = experiment?.requirements.length ?? 0;
    const ok = new Set<number>();
    for (const c of review.checks) {
      if (c.passed && c.verified) ok.add(c.requirement_index);
    }
    // 一条要求下所有检查都过才算达成
    for (const c of review.checks) {
      if (!c.passed) ok.delete(c.requirement_index);
    }
    return Math.min(ok.size, total);
  }, [review, experiment]);

  // ---------------------------------------------------------------
  // 保存草稿（真实落库）
  // ---------------------------------------------------------------
  async function handleSaveDraft() {
    if (!experiment || saving) return;
    setSaving(true);
    try {
      const r = await saveDraft(experiment.key, code, datasetId || null);
      setDirty(false);
      toast.success(
        r.saved_at
          ? `草稿已保存（${new Date(r.saved_at).toLocaleTimeString("zh-CN")}）`
          : "草稿已保存",
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "草稿保存失败");
    } finally {
      setSaving(false);
    }
  }

  // ---------------------------------------------------------------
  // 客观检查（不执行用户代码，后端 AST + 数据探针）
  // ---------------------------------------------------------------
  async function handleCheck() {
    if (!experiment || checking) return;
    setChecking(true);
    try {
      const r = await checkSubmission(experiment.key, {
        code,
        dataset_id: datasetId || null,
      });
      setReview(r);
      setDirty(false);
      if (r.blocked) {
        toast.error(r.blocked);
      } else if (r.passed) {
        toast.success("全部要求已达成，这个实验通关了。");
      } else {
        toast.info(
          `还有 ${r.next_steps.length} 项没达成，看下面的检查明细。`,
        );
      }
      if (r.dataset_warning) toast.info(r.dataset_warning);
      // 进度可能变化（完成 / 尝试次数），刷新一次头部数据
      const fresh = await getExperiment(experiment.key);
      setPage({ kind: "ready", experiment: fresh });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "检查失败");
    } finally {
      setChecking(false);
    }
  }

  // ---------------------------------------------------------------
  // AI 辅导：显式传 datasetIds，不再漏到全局会话存储
  // ---------------------------------------------------------------
  async function askAI(mode: HelpMode) {
    if (!experiment || streaming) return;
    setStreaming(true);
    setLastMode(mode);
    setFeedback("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let sid = sessionId;
      if (!sid) {
        const session = await createSession(
          datasetId ? [Number(datasetId)] : [],
          `学习实验：${experiment.title}`,
        );
        sid = session.id;
        setSessionId(sid);
      }

      const failing = review?.checks.filter((c) => !c.passed) ?? [];
      const contextLines = [
        `实验：${experiment.title}`,
        `教学目标：${experiment.goal}`,
        `实验要求：\n${experiment.requirements.map((r, i) => `${i + 1}. ${r}`).join("\n")}`,
        datasetId && context
          ? `数据集：${context.dataset_name}（${context.row_count} 行 × ${context.column_count} 列）\n字段：${context.columns.map((c) => `${c.name}(${c.dtype})`).join("、")}`
          : "数据集：未选择",
        failing.length
          ? `客观检查未通过项：\n${failing.map((c) => `- ${c.label}：${c.detail}`).join("\n")}`
          : "客观检查：尚未运行或已全部通过",
        `学生代码：\n\`\`\`python\n${code}\n\`\`\``,
      ];

      const selected = HELP_MODES.find((m) => m.id === mode) ?? HELP_MODES[0];
      let text = "";
      await sendMessage(sid, {
        content: `你是机器学习与大模型教学助手。请严格基于下面的实验上下文辅导，不要编造运行结果或数据统计。\n\n辅导模式：${selected.label}\n要求：${selected.prompt}\n\n${contextLines.join("\n\n")}\n\n通用原则：不要为了让学生完成任务而直接替写完整答案；优先指出证据、解释原因、提出下一步思考。只有当学生明确需要概念解释时才展开知识讲解。`,
        stream: true,
        // 显式传入 datasetIds：不传会回落到 AI 实验室的全局 store，
        // 把别处选的数据集带进来（历史 P0 缺陷）。
        datasetIds: datasetId ? [Number(datasetId)] : [],
        // 外部取消句柄：组件卸载 / 切实验 / 点取消时中断流，
        // 否则后端 tail 线程会一直挂到 AGENT_SSE_CONFIRM_WAIT_SECONDS。
        signal: controller.signal,
        onEvent: (event) => {
          const payload = event.payload as {
            delta?: string;
            content?: string;
            text?: string;
            final_answer?: string;
            run?: { output?: string; final_answer?: string };
          };
          text +=
            payload.delta ??
            payload.content ??
            payload.text ??
            payload.run?.output ??
            payload.run?.final_answer ??
            payload.final_answer ??
            "";
          if (text) setFeedback(text);
        },
      });
      if (!text) setFeedback("AI 已完成分析，但返回内容为空，请稍后重试。");
    } catch (e) {
      if (controller.signal.aborted) {
        setFeedback("已取消本次请求。");
      } else {
        setFeedback(
          e instanceof Error ? `AI 辅导失败：${e.message}` : "AI 辅导失败",
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  async function handleReset() {
    if (!experiment) return;
    try {
      await resetProgress(experiment.key);
      setConfirmReset(false);
      toast.success("进度已重置，代码回到初始骨架。");
      loadExperiment(experiment.key);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "重置失败");
    }
  }

  // 把 textarea 的字符偏移换算成「第几行第几列」。
  // 直接读 el.value 而不是闭包里的 code —— onChange 里 code 还没提交，用闭包会差一拍。
  function syncCaret(el: HTMLTextAreaElement) {
    const upto = el.value.slice(0, el.selectionStart);
    const lines = upto.split("\n");
    setCaret({ line: lines.length, col: lines[lines.length - 1].length + 1 });
  }

  function handleEditorKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Tab") {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const next = `${code.slice(0, start)}    ${code.slice(end)}`;
      setCode(next);
      setDirty(true);
      requestAnimationFrame(() => {
        target.selectionStart = start + 4;
        target.selectionEnd = start + 4;
      });
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void handleCheck();
    }
  }

  // ---------------------------------------------------------------
  // 渲染
  // ---------------------------------------------------------------
  if (page.kind === "loading") {
    return (
      <div className="learning-page learning-workspace-page">
        <div className="card learning-panel">
          <div className="loading-block">正在读取实验…</div>
        </div>
      </div>
    );
  }

  if (page.kind === "error") {
    return (
      <div className="learning-page learning-workspace-page">
        <div className="card learning-panel">
          <div className="learning-error-block">
            <strong>无法打开这个实验</strong>
            <p>{page.message}</p>
            <div className="learning-error-actions">
              <Link className="btn primary" to="/learning">
                返回学习中心
              </Link>
              {experimentKey ? (
                <button
                  className="btn"
                  type="button"
                  onClick={() => loadExperiment(experimentKey)}
                >
                  重试
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    );
  }

  const exp = page.experiment;
  const progress = exp.progress;

  return (
    <div className="learning-page learning-workspace-page">
      <div className="learning-crumbs">
        <Link to="/learning">学习中心</Link>
        <span>/</span>
        <Link to={`/learning?track=${exp.track}`}>
          {exp.track === "ml" ? "机器学习" : "大模型"}
        </Link>
        <span>/</span>
        <span className="current">{exp.title}</span>
      </div>

      <header className="learning-ws-header">
        <div className="learning-ws-title">
          <div className="learning-ws-tags">
            <span className={`learning-track-tag track-${exp.track}`}>
              {exp.track === "ml" ? "机器学习" : "大模型"}
            </span>
            <span className="learning-level">{exp.level}</span>
            <span className={`learning-status ${progress.status}`}>
              {progress.status === "completed"
                ? `已完成 · ${progress.score} 分`
                : progress.status === "in_progress"
                  ? `进行中 · 已提交 ${progress.attempts} 次`
                  : "未开始"}
            </span>
          </div>
          <h1>{exp.title}</h1>
        </div>
        <div className="learning-ws-actions">
          <button
            className="btn btn-sm"
            type="button"
            onClick={() => setConfirmReset(true)}
            disabled={progress.status === "not_started" && !dirty}
          >
            重置进度
          </button>
          <Link className="btn btn-sm" to="/learning/history">
            学习记录
          </Link>
        </div>
      </header>

      {exp.prerequisite_info && !exp.prerequisite_info.completed ? (
        <div className="learning-notice">
          这个实验建议先完成
          <Link to={`/learning/workspace?experiment=${exp.prerequisite_info.key}`}>
            《{exp.prerequisite_info.title}》
          </Link>
          .你可以直接开始。
        </div>
      ) : null}

      <div className="learning-ws-grid">
        {/* ---- 左：要求 + 编辑器 ---- */}
        <main className="learning-ws-main">
          <section className="card learning-panel">
            <div className="panel-title">
              <span>实验要求</span>
              <span className="learning-muted">
                达成 {achievedCount}/{exp.requirements.length}
              </span>
            </div>
            <p className="learning-goal">{exp.goal}</p>
            <ol className="requirement-list">
              {exp.requirements.map((req, i) => {
                const related = review?.checks.filter(
                  (c) => c.requirement_index === i,
                );
                const done =
                  related && related.length > 0 && related.every((c) => c.passed);
                const pending =
                  related && related.some((c) => !c.passed);
                return (
                  <li
                    key={req}
                    className={done ? "done" : pending ? "pending" : ""}
                  >
                    <span className="req-mark" aria-hidden="true">
                      {done ? "✓" : pending ? "!" : String(i + 1)}
                    </span>
                    <div>
                      <span className="req-text">{req}</span>
                      {related?.map((c, ci) => (
                        <span
                          key={ci}
                          className={`req-detail ${c.passed ? "pass" : "fail"}`}
                        >
                          {c.kind === "data" ? "数据 " : ""}
                          {c.verified ? "" : "未验证 · "}
                          {c.detail}
                        </span>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>

          <section className="card learning-panel code-panel">
            <div className="panel-title">
              <span>实验代码</span>
              <span className="learning-muted">
                {code.split("\n").length} 行 · Tab 缩进
                {dirty ? " · 未保存" : ""}
              </span>
            </div>
            <textarea
              className="learning-code-editor"
              value={code}
              spellCheck={false}
              aria-label="实验代码编辑器"
              onChange={(e) => {
                setCode(e.target.value);
                setDirty(true);
                syncCaret(e.target);
              }}
              onKeyUp={(e) => syncCaret(e.currentTarget)}
              onClick={(e) => syncCaret(e.currentTarget)}
              onSelect={(e) => syncCaret(e.currentTarget)}
              onKeyDown={handleEditorKeyDown}
            />
            <div className="code-actions">
              <button
                className="btn btn-sm"
                type="button"
                onClick={handleSaveDraft}
                disabled={saving}
              >
                {saving ? "保存中…" : "保存草稿"}
              </button>
              <button
                className="btn btn-sm primary"
                type="button"
                onClick={handleCheck}
                disabled={checking}
              >
                {checking ? "检查中…" : "检查代码"}
              </button>
              <span className="learning-muted code-hint">
                第 {caret.line} 行 · 第 {caret.col} 列 · Ctrl / ⌘ + Enter 快速检查
              </span>
            </div>
          </section>

          {review ? <ReviewPanel review={review} /> : null}
        </main>

        {/* ---- 右：数据 + AI ---- */}
        <aside className="learning-ws-side">
          <section className="card learning-panel">
            <div className="panel-title">
              <span>实验数据</span>
              <span className="learning-muted">{exp.needs}</span>
            </div>
            <label className="learning-field">
              选择数据集
              <select
                value={datasetId}
                onChange={(e) =>
                  setDatasetId(e.target.value ? Number(e.target.value) : "")
                }
              >
                <option value="">不使用数据集</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            {context ? (
              <ContextPanel context={context} />
            ) : contextError ? (
              <div className="learning-inline-error">
                数据信息读取失败：{contextError}
              </div>
            ) : datasetId ? (
              <div className="learning-muted">读取数据集信息…</div>
            ) : (
              <p className="learning-muted">
                未选择数据集时，数据侧校验项会标为「未验证」，代码结构检查仍会执行。
              </p>
            )}
          </section>

          <section className="card learning-panel ai-panel">
            <div className="panel-title">
              <span>AI 辅导</span>
              <span className="ai-status">
                {streaming ? "分析中…" : "已准备"}
              </span>
            </div>
            <p className="ai-intro">
              AI 会看到实验要求、你的代码、所选数据集与未通过项；它不会替你写完整答案。
            </p>
            <div className="ai-mode-list">
              {HELP_MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  className={`ai-mode${lastMode === m.id ? " active" : ""}`}
                  onClick={() => void askAI(m.id)}
                  disabled={streaming}
                >
                  {m.label}
                </button>
              ))}
            </div>
            {streaming ? (
              <button
                className="btn btn-sm"
                type="button"
                onClick={() => abortRef.current?.abort()}
              >
                取消
              </button>
            ) : null}
            {feedback ? (
              <div className="ai-feedback" aria-live="polite">
                {feedback}
              </div>
            ) : (
              <div className="ai-feedback empty">
                选择一种辅导方式开始。
              </div>
            )}
          </section>

          {exp.concepts.length > 0 ? (
            <section className="card learning-panel">
              <div className="panel-title">
                <span>关键概念</span>
              </div>
              <div className="concept-list">
                {exp.concepts.map((c) => (
                  <details key={c.term} className="concept-item">
                    <summary>{c.term}</summary>
                    <p>{c.body}</p>
                  </details>
                ))}
              </div>
            </section>
          ) : null}

          {exp.references.length > 0 ? (
            <section className="card learning-panel">
              <div className="panel-title">
                <span>延伸阅读</span>
              </div>
              <div className="reference-list">
                {exp.references.map((r) => (
                  <a
                    key={r.url}
                    href={r.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {r.label} ↗
                  </a>
                ))}
              </div>
            </section>
          ) : null}

          {progress.history.length > 0 ? (
            <section className="card learning-panel">
              <div className="panel-title">
                <span>提交曲线</span>
                <span className="learning-muted">
                  最近 {progress.history.length} 次
                </span>
              </div>
              <AttemptHistory
                history={progress.history}
                total={exp.requirements.length}
              />
            </section>
          ) : null}
        </aside>
      </div>

      <ConfirmDialog
        open={confirmReset}
        title="重置这个实验的进度？"
        message="代码将回到初始骨架，完成状态与历史提交记录都会被清除。这个操作无法撤销。"
        confirmText="重置"
        danger
        onConfirm={handleReset}
        onCancel={() => setConfirmReset(false)}
      />
    </div>
  );
}

/** 客观检查结果面板。 */
function ReviewPanel({ review }: { review: LearningReview }) {
  if (review.blocked) {
    return (
      <section className="card learning-panel review-panel blocked">
        <div className="panel-title">
          <span>检查结果</span>
          <span className="learning-muted">无法判定</span>
        </div>
        <div className="learning-error-block">
          <strong>代码有语法错误，检查无法进行</strong>
          <p>{review.blocked}</p>
          <p className="learning-muted">
            语法错误会让所有检查项都失去意义，请先修好再检查。
          </p>
        </div>
      </section>
    );
  }

  const passed = review.checks.filter((c) => c.passed).length;
  const unverified = review.checks.filter((c) => !c.verified).length;

  return (
    <section className="card learning-panel review-panel">
      <div className="panel-title">
        <span>检查结果</span>
        <span className={`learning-score ${review.passed ? "pass" : ""}`}>
          {review.score} 分 · {passed}/{review.checks.length} 项通过
          {unverified > 0 ? ` · ${unverified} 项未验证` : ""}
        </span>
      </div>

      <ul className="check-list">
        {review.checks.map((c, i) => (
          <CheckRow key={i} check={c} />
        ))}
      </ul>

      {review.next_steps.length > 0 ? (
        <div className="next-steps">
          <strong>还差这些</strong>
          <ul>
            {review.next_steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="next-steps done">
          <strong>全部要求已达成</strong>
          <p>可以换一个数据集重做，观察结论是否还成立。</p>
        </div>
      )}
    </section>
  );
}

function CheckRow({ check }: { check: LearningCheck }) {
  const state = !check.passed ? "fail" : check.verified ? "pass" : "unverified";
  const icon = state === "pass" ? "✓" : state === "fail" ? "✕" : "—";
  return (
    <li className={`check-row ${state}`}>
      <span className="check-icon" aria-hidden="true">
        {icon}
      </span>
      <div className="check-body">
        <span className="check-label">
          {check.label}
          {check.kind === "data" ? (
            <em className="check-kind">数据校验</em>
          ) : null}
        </span>
        <span className="check-detail">{check.detail}</span>
      </div>
    </li>
  );
}

/** 数据集体检面板。 */
function ContextPanel({ context }: { context: LearningContext }) {
  const [showAll, setShowAll] = useState(false);
  const cols = showAll ? context.columns : context.columns.slice(0, 8);

  return (
    <div className="context-panel">
      <div className="context-summary">
        <strong>
          {context.row_count} 行 × {context.column_count} 列
        </strong>
        <span>版本 v{context.version}</span>
        {context.missing_cells > 0 ? (
          <span className="warn">缺失 {context.missing_cells} 个单元格</span>
        ) : (
          <span className="ok">无缺失值</span>
        )}
      </div>

      {context.label_candidates.length > 0 ? (
        <div className="context-candidate">
          <span className="candidate-tag">可作分类标签</span>
          {context.label_candidates.join("、")}
        </div>
      ) : null}
      {context.target_candidates.length > 0 ? (
        <div className="context-candidate">
          <span className="candidate-tag">可作回归目标</span>
          {context.target_candidates.join("、")}
        </div>
      ) : null}

      <div className="context-columns">
        {cols.map((c) => (
          <div className="context-column" key={c.name}>
            <span className="col-name">{c.name}</span>
            <span className="col-dtype">{c.dtype}</span>
            <span className="col-meta">
              {c.unique_count} 个取值
              {c.null_count > 0 ? ` · 缺 ${c.null_count}` : ""}
            </span>
          </div>
        ))}
      </div>
      {context.columns.length > 8 ? (
        <button
          className="btn btn-sm"
          type="button"
          onClick={() => setShowAll((v) => !v)}
        >
          {showAll ? "收起字段" : `展开全部 ${context.columns.length} 个字段`}
        </button>
      ) : null}
    </div>
  );
}

/** 提交曲线：用 CSS 画条形，不引入图表库。 */
function AttemptHistory({
  history,
  total,
}: {
  history: { score: number; passed: boolean; created_at: string | null }[];
  total: number;
}) {
  const ordered = [...history].reverse();
  return (
    <div className="attempt-history">
      <div className="attempt-bars">
        {ordered.map((a, i) => (
          <div
            key={i}
            className={`attempt-bar${a.passed ? " pass" : ""}`}
            style={{ height: `${Math.max(a.score, 6)}%` }}
            title={`${a.score} 分${a.passed ? " · 通过" : ""}${
              a.created_at
                ? ` · ${new Date(a.created_at).toLocaleString("zh-CN")}`
                : ""
            }`}
          />
        ))}
      </div>
      <div className="attempt-legend">
        <span>最早</span>
        <span>
          满分 {total} 项要求
        </span>
        <span>最近</span>
      </div>
    </div>
  );
}
