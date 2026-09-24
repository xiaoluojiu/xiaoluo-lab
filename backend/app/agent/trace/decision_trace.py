"""DecisionTrace —— 一次 Agent 运行的**完整决策链**结构化记录。

这是「未来训练的数据闭环」的**数据出口**（任务 8/9）。不建 MLOps / 数据湖 /
向量库 / 自动训练平台，只建立**干净、可训练、可复现**的结构化记录。

记录什么（任务 9）
------------------
    用户请求
    → 本地 Router 候选
    → 本地 confidence
    → 最终 Decision
    → 实际 Tool
    → Tool result（signals）
    → 下一次 Decision
    → 最终结果

据此可生成（任务 9）：
    Router training data / Tool selection data / Task understanding data /
    Parameter resolution data / Agent decision data / SFT / LoRA data

关键约束
--------
* **不是** ``logger.info(...)`` 的替代品 —— 它是**结构化、可反序列化、可 join** 的记录；
* 追加写 JSONL，与既有 ``local_router/trace.py`` 的 shadow 埋点同目录不同文件
  （shadow 是「本地 Router 若接管会怎样」的对照，这里是「Agent 实际怎么决策」的全链）；
* 任何异常都**吞掉只记一次**（埋点失败不得打断用户请求）；
* 字段对齐任务 8 的最小字段集。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DECISION_TRACE_FILE = "decision_trace.jsonl"

# 轮转上限（与 shadow 一致，只留一代）。
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

_LOCK = threading.Lock()
_WARNED: set[str] = set()


@dataclass
class DecisionTrace:
    """任务 8 定义的最小字段集（外加若干可观测字段）。

    字段与任务 8 的 ``DecisionTrace`` 一一对应，不擅自增删语义，
    只补充 ``started_at`` / ``finished_at`` 等时间戳便于 join。
    """

    trace_id: str
    request: str

    task_spec: dict[str, Any] = field(default_factory=dict)

    router_candidates: list[dict[str, Any]] = field(default_factory=list)
    # candidate / score / source 等

    decision_made: dict[str, Any] = field(default_factory=dict)
    decision_source: str = ""
    # rule / local_model / remote

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results_summary: list[dict[str, Any]] = field(default_factory=list)

    final_answer: str = ""
    answer_source: str = ""

    success: bool | None = None

    remote_calls: int = 0
    remote_escalations: int = 0
    local_calls: int = 0

    latency_ms: int = 0

    #: 决策步骤序列（Observe→Decide→Act 的每一跳），供「至少两次 Decision」验收。
    decision_steps: list[dict[str, Any]] = field(default_factory=list)

    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request": self.request,
            "task_spec": self.task_spec,
            "router_candidates": self.router_candidates,
            "decision_made": self.decision_made,
            "decision_source": self.decision_source,
            "tool_calls": self.tool_calls,
            "tool_results_summary": self.tool_results_summary,
            "final_answer": self.final_answer,
            "answer_source": self.answer_source,
            "success": self.success,
            "remote_calls": self.remote_calls,
            "remote_escalations": self.remote_escalations,
            "local_calls": self.local_calls,
            "latency_ms": self.latency_ms,
            "decision_steps": self.decision_steps,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def record_decision(self, step: dict[str, Any]) -> None:
        """追加一跳「Decide」记录（Observe→Decide→Act 的一环）。"""
        self.decision_steps.append(step)


def trace_path() -> Path:
    """DecisionTrace 与 shadow trace 同目录（`{MODEL_ROOT}/local_router/`）。"""
    from app.local_router.model import artifact_path

    return artifact_path().parent / DECISION_TRACE_FILE


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message)


def write_trace(trace: DecisionTrace) -> None:
    """追加写一条 DecisionTrace。任何异常吞掉只记一次。"""
    try:
        path = trace_path()
        if trace.finished_at is None:
            trace.finished_at = time.time()
        line = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — 埋点失败不得影响业务
        _warn_once("decision_trace", f"DecisionTrace 写入失败（不影响运行）：{exc}")


def iter_traces(path: Path | None = None):
    """逐行读 DecisionTrace（供离线分析 / 训练数据生成）。坏行跳过。"""
    target = path or trace_path()
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item
