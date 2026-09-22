import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { createSession, sendMessage } from "../api/agent";
import { useToast } from "../components/ToastProvider";
import { ConfirmDialog } from "../components/ConfirmDialog";
import {
  deleteCustomCard,
  getCustomCard,
  updateCustomCard,
  type CustomCard,
  type CustomCardItem,
} from "../api/learning";
import "./learning.css";

type PageState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; card: CustomCard };

/** 练习卡的 AI 点评方式。 */
const REVIEW_MODES = [
  { id: "review", label: "点评我的代码", prompt: "指出写得好的地方和存在的问题，按重要性排序，不要直接重写整段代码。" },
  { id: "hint", label: "给我一个提示", prompt: "只给一个能推动我继续思考的提示，不要泄露完整实现。" },
  { id: "explain", label: "解释这个概念", prompt: "结合我的目标描述，用初学者能理解的方式讲解涉及的核心概念。" },
] as const;

export default function LearningCard() {
  const { cardId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();

  const [page, setPage] = useState<PageState>({ kind: "loading" });

  // 练习卡状态
  const [code, setCode] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // 清单卡状态
  const [items, setItems] = useState<CustomCardItem[]>([]);
  const [itemDraft, setItemDraft] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);

  const id = Number(cardId);

  const load = useCallback(() => {
    if (!id) {
      setPage({ kind: "error", message: "地址里没有指定卡片。" });
      return;
    }
    setPage({ kind: "loading" });
    getCustomCard(id)
      .then((card) => {
        setPage({ kind: "ready", card });
        setCode(card.code || "");
        setItems(card.items || []);
        setDirty(false);
      })
      .catch((e: unknown) => {
        setPage({
          kind: "error",
          message: e instanceof Error ? e.message : "卡片不存在或已被删除。",
        });
      });
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // 卸载时中断进行中的 AI 流
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const card = page.kind === "ready" ? page.card : null;

  async function handleSaveCode() {
    if (!card || saving) return;
    setSaving(true);
    try {
      await updateCustomCard(card.id, { code });
      setDirty(false);
      toast.success("代码已保存");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function askReview(mode: (typeof REVIEW_MODES)[number]) {
    if (!card || streaming) return;
    setStreaming(true);
    setFeedback("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let sid = sessionId;
      if (!sid) {
        const session = await createSession([], `学习卡片：${card.title}`);
        sid = session.id;
        setSessionId(sid);
      }

      const context = [
        `练习目标：${card.goal || "（未填写）"}`,
        `我的代码：\n\`\`\`python\n${code || "# 空"}\n\`\`\``,
      ].join("\n\n");

      let text = "";
      await sendMessage(sid, {
        content: `你是编程学习助手，正在点评一张自建练习卡。请严格基于下面的目标与代码给出反馈，不要编造运行结果。\n\n点评方式：${mode.label}\n要求：${mode.prompt}\n\n${context}\n\n通用原则：以鼓励为主，指出证据、解释原因、提出下一步；只有明确要求解释概念时才展开知识讲解。`,
        stream: true,
        datasetIds: [],
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
      if (!text) setFeedback("AI 已完成点评，但返回内容为空，请稍后重试。");
    } catch (e) {
      if (controller.signal.aborted) {
        setFeedback("已取消本次请求。");
      } else {
        setFeedback(
          e instanceof Error ? `AI 点评失败：${e.message}` : "AI 点评失败",
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  // ---- 清单卡：本地更新 + 落库 ----
  async function persistItems(next: CustomCardItem[]) {
    setItems(next);
    if (!card) return;
    try {
      await updateCustomCard(card.id, { items: next });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "保存清单失败");
    }
  }

  function toggleItem(idx: number) {
    const next = items.map((it, i) => (i === idx ? { ...it, done: !it.done } : it));
    void persistItems(next);
  }

  function addItem() {
    const t = itemDraft.trim();
    if (!t) return;
    void persistItems([...items, { text: t, done: false }]);
    setItemDraft("");
  }

  function removeItem(idx: number) {
    void persistItems(items.filter((_, i) => i !== idx));
  }

  async function handleDelete() {
    if (!card) return;
    try {
      await deleteCustomCard(card.id);
      setConfirmDelete(false);
      toast.success("卡片已删除");
      navigate("/learning?tab=custom");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "删除失败");
    }
  }

  if (page.kind === "loading") {
    return (
      <div className="learning-page learning-workspace-page">
        <div className="card learning-panel">
          <div className="loading-block">正在读取卡片…</div>
        </div>
      </div>
    );
  }

  if (page.kind === "error" || !card) {
    return (
      <div className="learning-page learning-workspace-page">
        <div className="card learning-panel">
          <div className="learning-error-block">
            <strong>无法打开这张卡片</strong>
            <p>{page.kind === "error" ? page.message : "卡片不存在。"}</p>
            <div className="learning-error-actions">
              <Link className="btn primary" to="/learning">
                返回学习中心
              </Link>
              {id ? (
                <button className="btn" type="button" onClick={load}>
                  重试
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    );
  }

  const isPractice = card.kind === "practice";
  const doneCount = items.filter((i) => i.done).length;

  return (
    <div className="learning-page learning-workspace-page learning-card-page">
      <div className="learning-crumbs">
        <Link to="/learning">学习中心</Link>
        <span>/</span>
        <Link to="/learning?tab=custom">我的卡片</Link>
        <span>/</span>
        <span className="current">{card.title}</span>
      </div>

      <header className="learning-ws-header">
        <div className="learning-ws-title">
          <div className="learning-ws-tags">
            <span className={`learning-level ${isPractice ? "level-basic" : "level-advanced"}`}>
              {isPractice ? "练习卡" : "清单卡"}
            </span>
            {isPractice ? (
              <span className="learning-muted">
                {dirty ? "有未保存的改动" : "代码已保存"}
              </span>
            ) : (
              <span className="learning-status in_progress">
                完成 {doneCount}/{items.length}
              </span>
            )}
          </div>
          <h1>{card.title}</h1>
          {card.goal ? <p>{card.goal}</p> : null}
        </div>
        <div className="learning-ws-actions">
          <button
            className="btn btn-sm"
            type="button"
            onClick={() => setConfirmDelete(true)}
          >
            删除卡片
          </button>
          <Link className="btn btn-sm" to="/learning">
            返回学习中心
          </Link>
        </div>
      </header>

      {isPractice ? (
        <div className="learning-ws-grid">
          {/* 左：代码编辑器 */}
          <main className="learning-ws-main">
            <section className="card learning-panel code-panel">
              <div className="panel-title">
                <span>练习代码</span>
                <span className="learning-muted">
                  {code.split("\n").length} 行 · Tab 缩进
                  {dirty ? " · 未保存" : ""}
                </span>
              </div>
              <textarea
                className="learning-code-editor"
                value={code}
                spellCheck={false}
                aria-label="练习代码编辑器"
                placeholder="# 在这里写你的练习代码\n# 写完后可以点「AI 点评」得到反馈"
                onChange={(e) => {
                  setCode(e.target.value);
                  setDirty(true);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Tab") {
                    e.preventDefault();
                    const t = e.currentTarget;
                    const s = t.selectionStart;
                    const en = t.selectionEnd;
                    setCode(`${code.slice(0, s)}    ${code.slice(en)}`);
                    setDirty(true);
                    requestAnimationFrame(() => {
                      t.selectionStart = s + 4;
                      t.selectionEnd = s + 4;
                    });
                  }
                }}
              />
              <div className="code-actions">
                <button
                  className="btn btn-sm"
                  type="button"
                  onClick={handleSaveCode}
                  disabled={saving}
                >
                  {saving ? "保存中…" : "保存代码"}
                </button>
                <span className="learning-muted code-hint">
                  练习卡不做客观判分，用 AI 点评替代
                </span>
              </div>
            </section>
          </main>

          {/* 右：AI 点评 */}
          <aside className="learning-ws-side">
            <section className="card learning-panel ai-panel">
              <div className="panel-title">
                <span>AI 点评</span>
                <span className="ai-status">
                  {streaming ? "分析中…" : "已准备"}
                </span>
              </div>
              <p className="ai-intro">
                AI 会看到你的目标描述和代码，给出针对性反馈；它不会替你重写完整答案。
              </p>
              <div className="ai-mode-list">
                {REVIEW_MODES.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    className="ai-mode"
                    onClick={() => void askReview(m)}
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
                <div className="ai-feedback empty">选择一种点评方式开始。</div>
              )}
            </section>
          </aside>
        </div>
      ) : (
        <div className="learning-ws-grid">
          {/* 清单卡：单列居中 */}
          <main className="learning-ws-main learning-card-checklist">
            <section className="card learning-panel">
              <div className="panel-title">
                <span>要点清单</span>
                <span className="learning-muted">
                  {items.length > 0
                    ? `完成 ${doneCount}/${items.length}`
                    : "还没有要点"}
                </span>
              </div>

              {items.length > 0 ? (
                <ul className="checklist-list">
                  {items.map((it, i) => (
                    <li key={i} className={it.done ? "done" : ""}>
                      <label className="checklist-row">
                        <input
                          type="checkbox"
                          checked={it.done}
                          onChange={() => toggleItem(i)}
                        />
                        <span className="checklist-text">{it.text}</span>
                      </label>
                      <button
                        className="checklist-del"
                        type="button"
                        aria-label={`删除第 ${i + 1} 项`}
                        onClick={() => removeItem(i)}
                      >
                        ✕
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="learning-muted">
                  还没有要点。在下面输入一条，按回车或点「添加」。
                </p>
              )}

              <div className="checklist-add">
                <input
                  type="text"
                  value={itemDraft}
                  placeholder="例如：看完 Attention Is All You Need"
                  onChange={(e) => setItemDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addItem();
                    }
                  }}
                />
                <button className="btn btn-sm primary" type="button" onClick={addItem}>
                  添加
                </button>
              </div>

              {items.length > 0 ? (
                <div className="learning-progress-bar" aria-hidden="true">
                  <span
                    style={{
                      width: `${items.length ? Math.round((doneCount / items.length) * 100) : 0}%`,
                    }}
                  />
                </div>
              ) : null}
            </section>
          </main>
        </div>
      )}

      <ConfirmDialog
        open={confirmDelete}
        title="删除这张卡片？"
        message="卡片内容和进度都会被清除，这个操作无法撤销。"
        confirmText="删除"
        danger
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
}
