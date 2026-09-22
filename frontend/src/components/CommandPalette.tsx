import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useUiShell } from "../store/ui";
// 命令面板与主导航共用 config/nav.ts，避免路由新增后两处列表不同步。
import { FLAT_NAV, type NavItem } from "../config/nav";

interface RecentDataset {
  id: number;
  name: string;
}

function readRecentDatasets(): RecentDataset[] {
  try {
    const raw = localStorage.getItem("xllab.recent.datasets");
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? (arr.slice(0, 4) as RecentDataset[]) : [];
  } catch {
    return [];
  }
}

/**
 * 命令面板：Ctrl/Cmd + K 唤起，支持实时过滤、键盘上下选择、回车跳转。
 * 同时展示「最近数据集」快捷入口（读自 localStorage，无则隐藏该组）。
 */
export function CommandPalette() {
  const open = useUiShell((s) => s.commandOpen);
  const close = useUiShell((s) => s.closeCommand);
  const toggle = useUiShell((s) => s.toggleCommand);
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const recent = useMemo(readRecentDatasets, []);

  // 全局快捷键：Ctrl/Cmd + K 切换面板
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      const id = window.setTimeout(() => inputRef.current?.focus(), 0);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [open]);

  const results = useMemo<NavItem[]>(() => {
    const q = query.trim().toLowerCase();
    const recentHits: NavItem[] = recent
      .filter((r) => !q || r.name.toLowerCase().includes(q))
      .map((r) => ({ path: `/datasets/${r.id}`, label: r.name, icon: "table", hint: "最近数据集" }));
    const navHits = FLAT_NAV.filter(
      (c) => !q || c.label.toLowerCase().includes(q) || (c.hint ?? "").toLowerCase().includes(q),
    );
    return [...recentHits, ...navHits];
  }, [query, recent]);

  if (!open) return null;

  function go(path: string) {
    close();
    navigate(path);
  }

  function handleKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const r = results[active];
      if (r) go(r.path);
    }
  }

  return (
    <div className="overlay cmd-overlay" onClick={close} role="presentation">
      <div
        className="palette"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="命令面板"
      >
        <div className="palette-input-row">
          <svg
            className="palette-search-icon"
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" />
          </svg>
          <input
            ref={inputRef}
            className="palette-input"
            placeholder="搜索页面或数据集，回车跳转…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            onKeyDown={handleKey}
            aria-label="命令面板搜索框"
          />
          <kbd className="palette-kbd">Esc</kbd>
        </div>
        <ul className="palette-list">
          {results.length === 0 && <li className="palette-empty">无匹配结果</li>}
          {results.map((r, i) => (
            <li key={`${r.path}__${r.label}__${i}`}>
              <button
                type="button"
                className={`palette-item${i === active ? " active" : ""}`}
                onMouseEnter={() => setActive(i)}
                onClick={() => go(r.path)}
              >
                <span className="palette-item-label">{r.label}</span>
                {r.hint ? <span className="palette-item-hint">{r.hint}</span> : null}
                <span className="palette-item-path">{r.path}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
