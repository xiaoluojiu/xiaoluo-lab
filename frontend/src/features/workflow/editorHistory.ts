/**
 * 编辑历史（撤销 / 重做）+ 草稿自动保存。
 *
 * 背景：
 * - 原 Workflow 编辑器没有撤销/重做，误删节点或拖错位置只能手动改回；
 * - 也没有草稿落盘，刷新即丢（`store/ui.ts` 只持久化了侧栏折叠状态）。
 *
 * 设计取舍：
 * - 历史栈只存「节点 + 边」的快照，不存 UI 态（选中节点、缩放、平移）。
 *   撤销一次拖拽不应连视口一起跳回去，那比不撤更烦人。
 * - 合并连续同类编辑：拖拽过程中每帧都会 onChange，不能每帧压栈，
 *   否则撤销一次只退回 1px。用 `coalesceKey`（如 `move:<nodeId>`）在
 *   时间窗内合并为一条历史。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { WorkflowEdge, WorkflowNode } from "../../types/workflow";

export interface EditorSnapshot {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

const HISTORY_LIMIT = 80;
/** 同一 coalesceKey 在该毫秒窗口内视为一次连续编辑。 */
const COALESCE_MS = 600;

/**
 * 深拷贝快照。
 *
 * 必须拷贝：节点/边对象在后续编辑中会被替换，若只存引用，
 * 历史栈里所有条目会指向同一份最新数据，撤销等于没撤。
 */
function clone(snapshot: EditorSnapshot): EditorSnapshot {
  return {
    nodes: snapshot.nodes.map((node) => ({
      ...node,
      config: JSON.parse(JSON.stringify(node.config ?? {})) as Record<string, unknown>,
    })),
    edges: snapshot.edges.map((edge) => ({ ...edge })),
  };
}

export function useEditHistory(initial: EditorSnapshot) {
  const [snapshot, setSnapshot] = useState<EditorSnapshot>(() => clone(initial));
  const past = useRef<EditorSnapshot[]>([]);
  const future = useRef<EditorSnapshot[]>([]);
  const lastKey = useRef<string | null>(null);
  const lastAt = useRef(0);
  // 版本号驱动撤销/重做的可用态刷新（ref 变化不触发渲染）。
  const [version, setVersion] = useState(0);

  /**
   * 记录一次编辑。
   *
   * @param next     编辑后的节点与边
   * @param coalesceKey 连续编辑的合并键（拖拽传 `move:节点`，其余传 undefined）
   */
  const commit = useCallback((next: EditorSnapshot, coalesceKey?: string) => {
    const now = Date.now();
    const shouldCoalesce =
      coalesceKey != null && coalesceKey === lastKey.current && now - lastAt.current < COALESCE_MS;
    lastKey.current = coalesceKey ?? null;
    lastAt.current = now;

    setSnapshot((current) => {
      // 内容没变就不进历史（例如 fit/arrange 后的重复 setState）。
      if (JSON.stringify(current) === JSON.stringify(next)) return current;
      if (!shouldCoalesce) {
        past.current = [...past.current, clone(current)].slice(-HISTORY_LIMIT);
      }
      // 任何新编辑都清空「重做」分支 —— 这是编辑器的通行语义。
      future.current = [];
      return clone(next);
    });
    setVersion((v) => v + 1);
  }, []);

  /** 整体替换（打开另一个工作流 / 应用模板），清空历史。 */
  const reset = useCallback((next: EditorSnapshot) => {
    past.current = [];
    future.current = [];
    lastKey.current = null;
    setSnapshot(clone(next));
    setVersion((v) => v + 1);
  }, []);

  const undo = useCallback(() => {
    setSnapshot((current) => {
      const previous = past.current[past.current.length - 1];
      if (!previous) return current;
      past.current = past.current.slice(0, -1);
      future.current = [clone(current), ...future.current].slice(0, HISTORY_LIMIT);
      lastKey.current = null;
      return clone(previous);
    });
    setVersion((v) => v + 1);
  }, []);

  const redo = useCallback(() => {
    setSnapshot((current) => {
      const [next, ...rest] = future.current;
      if (!next) return current;
      future.current = rest;
      past.current = [...past.current, clone(current)].slice(-HISTORY_LIMIT);
      lastKey.current = null;
      return clone(next);
    });
    setVersion((v) => v + 1);
  }, []);

  return {
    nodes: snapshot.nodes,
    edges: snapshot.edges,
    commit,
    reset,
    undo,
    redo,
    canUndo: past.current.length > 0,
    canRedo: future.current.length > 0,
    version,
  };
}

/* ------------------------------------------------------------------ */
/* 草稿自动保存                                                        */
/* ------------------------------------------------------------------ */

const DRAFT_PREFIX = "xiaoluo.workflow.draft.";

export interface DraftPayload extends EditorSnapshot {
  name: string;
  savedAt: number;
}

function draftKey(scope: string) {
  return `${DRAFT_PREFIX}${scope}`;
}

export function readDraft(scope: string): DraftPayload | null {
  try {
    const raw = localStorage.getItem(draftKey(scope));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<DraftPayload>;
    if (!parsed || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) return null;
    return {
      name: typeof parsed.name === "string" ? parsed.name : "",
      nodes: parsed.nodes as WorkflowNode[],
      edges: parsed.edges as WorkflowEdge[],
      savedAt: typeof parsed.savedAt === "number" ? parsed.savedAt : 0,
    };
  } catch {
    return null;
  }
}

export function clearDraft(scope: string) {
  try {
    localStorage.removeItem(draftKey(scope));
  } catch {
    /* localStorage 不可用时静默降级 */
  }
}

function writeDraft(scope: string, payload: DraftPayload) {
  try {
    localStorage.setItem(draftKey(scope), JSON.stringify(payload));
  } catch {
    /* 容量超限或隐私模式：草稿功能降级为不可用，不影响编辑 */
  }
}

/** 把 HH:MM 格式化抽出来，方便单测（不依赖 Date 的本地化输出差异）。 */
export function formatSavedAt(timestamp: number): string {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

/**
 * 草稿自动保存：内容变化后延迟写入，避免每次拖拽都写 localStorage。
 *
 * @returns 最近一次成功保存的时间戳（0 表示尚未保存）
 */
export function useDraftAutosave(
  scope: string | null,
  payload: Omit<DraftPayload, "savedAt">,
  delayMs = 800,
) {
  const [savedAt, setSavedAt] = useState(0);
  const serialized = JSON.stringify({ name: payload.name, nodes: payload.nodes, edges: payload.edges });

  useEffect(() => {
    if (!scope) return undefined;
    const timer = setTimeout(() => {
      const stamp = Date.now();
      writeDraft(scope, { ...payload, savedAt: stamp });
      setSavedAt(stamp);
    }, delayMs);
    return () => clearTimeout(timer);
    // payload 逐字段参与序列化，serialized 变化即内容变化。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, serialized, delayMs]);

  return savedAt;
}
