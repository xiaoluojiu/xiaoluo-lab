"""Agent 会话与运行的持久化。

设计取舍
--------
单文件 JSON，不做数据库表：Agent 运行记录是**会话态**，生命周期短、
读写模式是「整存整取」，用 ORM 反而要把嵌套的事件/工具调用拆成多张表。

三条防线（旧架构在这里都踩过坑）
--------------------------------
1. **事件上限**：每个 run 最多保留 ``MAX_EVENTS_PER_RUN`` 条事件。
   旧架构出现过单 run 20 万条 ``replanning`` 事件、store 文件几十 MB
   （一次读取就把内存打满）。新架构没有重规划循环，事件数天然很小，
   这个上限是**防御性**的 —— 防止将来任何新增的循环把磁盘写满。
2. **原子写**：先写 ``.tmp`` 再 ``os.replace``。进程在写一半被杀掉时，
   磁盘上留下的要么是旧文件要么是完整新文件，不会是半截 JSON。
3. **损坏自愈**：JSON 解析失败时把坏文件改名为 ``.corrupt.<时间戳>`` 后
   以空存储启动，而不是让整个后端起不来。数据宁可丢一次，不可一直不可用。

并发
----
进程内用一把 ``threading.RLock`` 保护。这是单实例本地平台的取舍：
多实例部署时改用数据库表（接口不变）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.agent.limits import MAX_LLM_CALLS, max_total_tokens
from app.agent.models import (
    AgentEvent,
    AgentRun,
    AgentSession,
    ClarificationRequest,
    EventType,
    PermissionRequest,
    RunStatus,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)

#: 单 run 事件上限。超出后丢弃**最早**的事件（保留最近的，因为最新状态最有用）。
MAX_EVENTS_PER_RUN = 500
#: 单会话保留的运行数上限（超出淘汰最旧的已完成运行）。
MAX_RUNS_PER_SESSION = 50
#: 会话历史消息上限。
MAX_HISTORY_PER_SESSION = 200


class AgentStore:
    """会话 / 运行的存储。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else self._default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.sessions: dict[str, AgentSession] = {}
        self.runs: dict[str, AgentRun] = {}
        #: run_id -> 事件监听者。SSE 流靠它实时拿到新事件，
        #: 而不是靠轮询 store（轮询会让「确认后继续」这类恢复路径迟滞）。
        self._listeners: dict[str, list[Callable[[AgentEvent], None]]] = {}
        self._load()

    # ---- 事件订阅 ----------------------------------------------------
    def add_listener(self, run_id: str, callback: Callable[[AgentEvent], None]) -> None:
        with self._lock:
            self._listeners.setdefault(run_id, []).append(callback)

    def remove_listener(self, run_id: str, callback: Callable[[AgentEvent], None]) -> None:
        with self._lock:
            callbacks = self._listeners.get(run_id)
            if not callbacks:
                return
            try:
                callbacks.remove(callback)
            except ValueError:
                pass
            if not callbacks:
                self._listeners.pop(run_id, None)

    def _notify(self, event: AgentEvent) -> None:
        """通知订阅者。回调抛异常不得影响其他订阅者，更不能中断落盘。"""
        for callback in list(self._listeners.get(event.run_id, ())):
            try:
                callback(event)
            except Exception:  # noqa: BLE001
                logger.exception("Agent 事件监听者回调异常（run=%s）", event.run_id)

    @staticmethod
    def _default_path() -> Path:
        from app.core.config import settings

        return settings.data_root_path / "agent_store.json"

    # ---- 持久化 ------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except (OSError, ValueError) as exc:
            # 拿损坏文件去覆盖备份可能把上一份好的备份也毁掉，因此备份名带时间戳。
            backup = self.path.with_suffix(f".corrupt.{int(time.time())}.json")
            try:
                self.path.replace(backup)
            except OSError:
                logger.exception("Agent store 损坏且无法备份：%s", self.path)
            logger.warning("Agent store 解析失败，已备份到 %s 并以空存储启动：%s", backup, exc)
            return

        for item in data.get("sessions") or []:
            try:
                session = _session_from_dict(item)
                self.sessions[session.id] = session
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("跳过损坏的会话记录：%s", exc)
        for item in data.get("runs") or []:
            try:
                run = _run_from_dict(item)
                self.runs[run.id] = run
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("跳过损坏的运行记录：%s", exc)

    def save(self) -> None:
        """整存。调用频率低（每次状态变更一次），不做增量。"""
        with self._lock:
            payload = {
                "version": 2,
                "sessions": [s.to_dict() for s in self.sessions.values()],
                "runs": [r.to_dict() for r in self.runs.values()],
            }
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                logger.exception("Agent store 写入失败：%s", self.path)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    # ---- 会话 --------------------------------------------------------
    def create_session(
        self, *, user_id: str = "anonymous", title: str = "新会话", dataset_ids: list[int] | None = None
    ) -> AgentSession:
        with self._lock:
            session = AgentSession(
                id=f"s-{uuid.uuid4().hex[:12]}",
                user_id=user_id or "anonymous",
                title=title or "新会话",
                dataset_ids=list(dataset_ids or []),
            )
            self.sessions[session.id] = session
            self.save()
            return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self.sessions.get(session_id)

    def list_sessions(self, *, user_id: str = "anonymous", include_archived: bool = False) -> list[AgentSession]:
        items = [s for s in self.sessions.values() if s.user_id == user_id]
        if not include_archived:
            items = [s for s in items if not s.archived]
        return sorted(items, key=lambda s: s.created_at, reverse=True)

    def update_session(self, session: AgentSession) -> None:
        with self._lock:
            self.sessions[session.id] = session
            self.save()

    def delete_session(self, session_id: str) -> tuple[bool, str]:
        """删除会话。

        只对**真正在推进**的运行（pending / planning / running）返回拒绝：那时引擎
        线程还活着，强删会把 run 变成没人引用的孤儿。

        **挂起态**（等待确认 / 等待补充信息）**不算**「未结束」—— 引擎线程在挂起点
        就退出了，没人会再推进它，拦下来只会让会话永远删不掉。是否在删除前把挂起
        运行优雅收成终态（并发 completed、关 SSE 通道）由 API 层负责，见
        ``api/v1/agent.py: delete_session``。
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return False, "会话不存在"
            for run_id in session.run_ids:
                run = self.runs.get(run_id)
                if run is not None and run.status.in_flight:
                    return False, "该会话存在未结束的运行，请先停止或等待其结束"
            for run_id in session.run_ids:
                self.runs.pop(run_id, None)
            self.sessions.pop(session_id, None)
            self.save()
            return True, ""

    def append_history(
        self,
        session_id: str,
        role: str,
        content: str,
        charts: list[dict] | None = None,
    ) -> None:
        """追加一条会话历史。

        ``charts`` 是这一轮真正画出来的图（SVG）。会话历史是前端重建对话的
        **唯一数据源**，图不写进来，刷新/切换页面后图就消失了 —— 那和没画一样。
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            entry: dict[str, Any] = {"role": role, "content": content}
            if charts:
                entry["charts"] = charts
            session.history.append(entry)
            if len(session.history) > MAX_HISTORY_PER_SESSION:
                session.history = session.history[-MAX_HISTORY_PER_SESSION:]
            self.save()

    # ---- 运行 --------------------------------------------------------
    def create_run(self, session_id: str, user_request: str, *, dataset_ids: list[int] | None = None) -> AgentRun:
        with self._lock:
            run = AgentRun(
                id=f"r-{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                user_request=user_request,
                status=RunStatus.PENDING,
                # ★ 预算必须在建 run 时就写进去。用默认 TokenUsage() 的话
                # max_llm_calls / max_total_tokens 恒为 0，前端 Token 面板会显示
                # 「剩余 0 次 / 剩余 0 token」，而 /capabilities 明明写着上限 4 ——
                # 两处自相矛盾，用户只会以为「额度已经用光了」。
                token_usage=TokenUsage(
                    max_llm_calls=MAX_LLM_CALLS,
                    max_total_tokens=max_total_tokens(),
                ),
            )
            self.runs[run.id] = run
            session = self.sessions.get(session_id)
            if session is not None:
                session.run_ids.append(run.id)
                # 只保留最近的若干次运行，避免 store 无限增长
                if len(session.run_ids) > MAX_RUNS_PER_SESSION:
                    for stale_id in session.run_ids[:-MAX_RUNS_PER_SESSION]:
                        stale = self.runs.get(stale_id)
                        if stale is not None and stale.status.terminal:
                            self.runs.pop(stale_id, None)
                    session.run_ids = session.run_ids[-MAX_RUNS_PER_SESSION:]
            self.save()
            return run

    def get_run(self, run_id: str) -> AgentRun | None:
        return self.runs.get(run_id)

    def update_run(self, run: AgentRun) -> None:
        with self._lock:
            # 事件上限：只保留最近的，避免 store 被单个 run 撑爆
            if len(run.events) > MAX_EVENTS_PER_RUN:
                run.events = run.events[-MAX_EVENTS_PER_RUN:]
            self.runs[run.id] = run
            self.save()

    def add_event(self, run: AgentRun, event_type: EventType, payload: dict[str, Any] | None = None) -> AgentEvent:
        """追加一条事件并落盘。

        ``run_id`` 必须写进每条事件：前端用最后一条事件的 run_id 作为
        confirm / deny / cancel / clarify 的目标，缺了它授权按钮就点不动。
        """
        with self._lock:
            seq = (run.events[-1].seq + 1) if run.events else 1
            event = AgentEvent(seq=seq, run_id=run.id, type=event_type, payload=payload or {})
            run.events.append(event)
            if len(run.events) > MAX_EVENTS_PER_RUN:
                run.events = run.events[-MAX_EVENTS_PER_RUN:]
            self.runs[run.id] = run
            self.save()
            self._notify(event)
            return event


# ---- 反序列化 --------------------------------------------------------
# 单独写成函数而不是方法：反序列化不需要 store 实例状态，
# 也方便测试直接构造对象而不落盘。


def _session_from_dict(data: dict[str, Any]) -> AgentSession:
    return AgentSession(
        id=str(data["id"]),
        user_id=str(data.get("user_id") or "anonymous"),
        title=str(data.get("title") or "新会话"),
        dataset_ids=[int(i) for i in (data.get("dataset_ids") or []) if str(i).lstrip("-").isdigit()],
        history=[{"role": str(m.get("role") or ""), "content": str(m.get("content") or "")} for m in (data.get("history") or [])],
        run_ids=[str(i) for i in (data.get("run_ids") or [])],
        created_at=float(data.get("created_at") or time.time()),
        archived=bool(data.get("archived")),
        # 键在 JSON 里是字符串，转回 int 才能与 dataset_id 对上
        baseline_versions={
            int(k): int(v)
            for k, v in (data.get("baseline_versions") or {}).items()
            if str(k).lstrip("-").isdigit() and str(v).lstrip("-").isdigit()
        },
    )


def _run_from_dict(data: dict[str, Any]) -> AgentRun:
    return AgentRun(
        id=str(data["id"]),
        session_id=str(data.get("session_id") or ""),
        user_request=str(data.get("user_request") or ""),
        status=RunStatus(data.get("status") or RunStatus.PENDING.value),
        final_answer=str(data.get("final_answer") or ""),
        error=str(data.get("error") or ""),
        plan=data.get("plan"),
        tool_calls=[ToolCall.from_dict(c) for c in (data.get("tool_calls") or [])],
        pending_confirmation=PermissionRequest.from_dict(data.get("pending_confirmation")),
        pending_clarification=ClarificationRequest.from_dict(data.get("pending_clarification")),
        cancel_requested=bool(data.get("cancel_requested")),
        token_usage=TokenUsage.from_dict(data.get("token_usage")),
        events=[AgentEvent.from_dict(e) for e in (data.get("events") or [])],
        created_at=float(data.get("created_at") or time.time()),
        started_at=float(data.get("started_at") or time.time()),
        finished_at=data.get("finished_at"),
        resume_step=int(data.get("resume_step") or 0),
        authorized_key=data.get("authorized_key"),
        clarification_answers=data.get("clarification_answers") or {},
    )
