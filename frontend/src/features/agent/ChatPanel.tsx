import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ChatChart, ChatMessage } from "../../types/agent";

export type { ChatMessage };

/**
 * 对话里的图表附件。
 *
 * ★ 历史缺陷：后端 eda.visualize 只返回数据，答案层把它翻译成一段文字
 *   （「最低箱是 [8.1,15.2)，计数 284」），这里又只渲染纯文本 ——
 *   用户要的图从头到尾没出现过，这就是「有图看不到」。
 *   现在后端随 completed 事件下发内联 SVG，这里必须真的画出来。
 *
 * 用 <img src="data:image/svg+xml"> 而不是 dangerouslySetInnerHTML：
 * SVG 里可能含用户上传文件带进来的列名，走 <img> 上下文不会执行其中的脚本。
 */
function ChartAttachments({ charts }: { charts: ChatChart[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const usable = charts.filter((c) => c?.svg);
  if (!usable.length) return null;
  return (
    <div className="chat-charts">
      {usable.map((c, i) => {
        const key = `${c.title || c.kind || "chart"}-${i}`;
        const src = `data:image/svg+xml;utf8,${encodeURIComponent(c.svg ?? "")}`;
        const expanded = open === key;
        return (
          <figure key={key} className={`chat-chart${expanded ? " expanded" : ""}`}>
            <figcaption className="chat-chart-title">{c.title || "图表"}</figcaption>
            <img
              className="chat-chart-img"
              src={src}
              alt={c.title || "分析图表"}
              onClick={() => setOpen(expanded ? null : key)}
            />
          </figure>
        );
      })}
    </div>
  );
}

export function ChatPanel({
  messages,
  busy,
  disabled,
  placeholder = "向 AI 描述你的分析需求...",
  onSend,
  showEmptyState = true,
  emptyState,
}: {
  messages: ChatMessage[];
  busy: boolean;
  disabled?: boolean;
  placeholder?: string;
  onSend: (content: string) => void;
  showEmptyState?: boolean;
  /** 无消息时在消息区居中渲染的内容（如快捷问题卡），输入框仍可用。 */
  emptyState?: ReactNode;
}) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages.length, busy]);

  function submit() {
    const content = draft.trim();
    if (!content || busy || disabled) return;
    setDraft("");
    onSend(content);
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {emptyState}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role === "user" ? "user" : "assistant"}`}>
            <div className="chat-role">{m.role === "user" ? "你" : "AI 助手"}</div>
            <div className="chat-bubble">{m.content}</div>
            {m.role !== "user" && m.charts?.length ? <ChartAttachments charts={m.charts} /> : null}
            {m.role !== "user" && m.source && (
              <div
                className={`chat-source-tag${m.source.by_llm ? " remote" : " local"}`}
                title={m.source.detail}
              >
                <span className="chat-source-dot" aria-hidden="true" />
                <span className="chat-source-label">
                  {m.source.by_llm ? "由远程大模型回答" : "由平台内置规则回答"}
                </span>
                <span className="chat-source-name">{m.source.label}{m.source.model ? ` · ${m.source.model}` : ""}</span>
              </div>
            )}
          </div>
        ))}
        {!messages.length && !emptyState && showEmptyState && (
          <div className="muted" style={{ textAlign: "center", padding: "var(--space-8)" }}>
            还没有消息，试着输入一个分析需求吧
          </div>
        )}
        {busy && (
          <div className="chat-msg assistant chat-live-msg" aria-live="polite">
            <div className="chat-role">AI 助手 · 正在工作</div>
            <div className="chat-bubble chat-thinking-bubble">
              <span className="chat-thinking-dot" />
              <span>正在理解任务、准备工具并执行当前步骤…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-composer">
        <div className="chat-input-shell">
          <textarea
            value={draft}
            disabled={disabled || busy}
            rows={2}
            placeholder={busy ? "AI 正在执行当前 Turn…" : placeholder}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {/* 发送按钮内嵌输入区右下角：不再单独占一行，输入区保持全宽。 */}
          <div className="chat-input-foot">
            <span className="chat-composer-hint">Enter 发送 · Shift + Enter 换行</span>
            <button
              className="btn primary chat-send-button"
              disabled={disabled || busy || !draft.trim()}
              onClick={submit}
            >
              {busy ? "执行中…" : "发送"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
