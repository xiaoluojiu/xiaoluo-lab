"""把 DecisionTrace（`decision_trace.jsonl`）导出为可训练的样本（Phase 5 数据闭环出口）。

与 ``harvest_traces.py`` 的分工
------------------------------
``harvest_traces.py`` 从 AgentStore 挖「真实执行结果」作为 Router 的**单步**训练正样本
（标签来自 tool_result 成功与否）。本脚本消费的是 ``DecisionTrace`` —— 一次运行的
**完整决策链**（请求 → Router 候选 → 置信度 → 决策 → 工具 → signals → 下一步决策 →
最终结果），据此导出**多类**训练数据：

1. `router_train.jsonl`       Router 选工具样本（utterance → 工具 + 置信度 + 候选）
2. `tool_selection.jsonl`     工具选择样本（任务理解 → 选哪个工具）
3. `task_understanding.jsonl` 任务理解样本（utterance → TaskSpec）
4. `decision_pairs.jsonl`     决策对（上一步 signals → 下一步决策），供 SFT / 强化

关键约束
--------
* **只导出结构化记录，不训练**（训练是独立脚本，避免把闭环写成「导出即训练」）；
* 标签来自**真实运行**（决策链里 recorded 的 decision_source / 实际执行的 tool），
  不靠外部 LLM 打分；
* 任何异常吞掉只记一次（导出失败不得影响运行）；
* 坏行 / 缺字段的行跳过并计入诊断，不静默丢弃。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.trace.decision_trace import iter_traces, trace_path  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "dataset"


def _router_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Router 选工具样本：utterance → 工具 + 置信度 + 候选。"""
    request = str(trace.get("request") or "").strip()
    candidates = trace.get("router_candidates") or []
    if not request:
        return None
    # 决策链里实际执行的第一个工具作为 gold（来自真实运行）。
    tool_calls = trace.get("tool_calls") or []
    gold_tool = None
    if tool_calls:
        gold_tool = str(tool_calls[0].get("tool") or "") or None
    decision = trace.get("decision_made") or {}
    return {
        "id": f"rt-{trace.get('trace_id', '?')}",
        "utterance": request,
        "gold_tool": gold_tool,
        "decision_tool": decision.get("tool"),
        "source": trace.get("decision_source"),
        "candidates": candidates,
        "confidence": (candidates[0].get("score") if candidates else None),
        "remote_escalations": int(trace.get("remote_escalations") or 0),
    }


def _tool_selection_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    """工具选择样本：任务理解（TaskSpec）→ 选哪个工具。"""
    task_spec = trace.get("task_spec") or {}
    decision = trace.get("decision_made") or {}
    tool = decision.get("tool")
    if not tool:
        return None
    return {
        "id": f"ts-{trace.get('trace_id', '?')}",
        "task_spec": task_spec,
        "tool": tool,
        "source": trace.get("decision_source"),
    }


def _task_understanding_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    """任务理解样本：utterance → TaskSpec。"""
    request = str(trace.get("request") or "").strip()
    task_spec = trace.get("task_spec") or {}
    if not request or not task_spec:
        return None
    return {
        "id": f"tu-{trace.get('trace_id', '?')}",
        "utterance": request,
        "task_spec": task_spec,
        "source": trace.get("decision_source"),
    }


def _decision_pair_samples(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """决策对：上一步 ToolResult signals → 下一步决策（供 SFT / 强化）。"""
    steps = trace.get("decision_steps") or []
    pairs: list[dict[str, Any]] = []
    for i in range(1, len(steps)):
        prev = steps[i - 1]
        cur = steps[i]
        # 只有「前一步带 signals（Observe）」的才是有效的 Observe→Decide 样本。
        prev_signals = prev.get("signals")
        if not prev_signals:
            continue
        pairs.append({
            "id": f"dp-{trace.get('trace_id', '?')}-{i}",
            "trace_id": trace.get("trace_id"),
            "prev_step": prev,
            "next_decision": cur,
        })
    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = trace_path()
    if not source.exists():
        print(json.dumps({"ok": False, "reason": f"未找到 DecisionTrace：{source}"}, ensure_ascii=False))
        return

    router_samples: list[dict[str, Any]] = []
    tool_selection: list[dict[str, Any]] = []
    task_understanding: list[dict[str, Any]] = []
    decision_pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    traces: list[dict[str, Any]] = []

    for item in iter_traces(source):
        traces.append(item)

    for trace in traces:
        s = _router_sample(trace)
        if s is None:
            skipped["router_no_request"] += 1
        else:
            router_samples.append(s)
        s2 = _tool_selection_sample(trace)
        if s2 is None:
            skipped["tool_selection_no_tool"] += 1
        else:
            tool_selection.append(s2)
        s3 = _task_understanding_sample(trace)
        if s3 is None:
            skipped["task_understanding_empty"] += 1
        else:
            task_understanding.append(s3)
        decision_pairs.extend(_decision_pair_samples(trace))

    def _write(name: str, rows: list[dict[str, Any]]) -> None:
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write("router_train.jsonl", router_samples)
    _write("tool_selection.jsonl", tool_selection)
    _write("task_understanding.jsonl", task_understanding)
    _write("decision_pairs.jsonl", decision_pairs)

    report = {
        "source_file": str(source),
        "traces": len(traces),
        "exports": {
            "router_train": len(router_samples),
            "tool_selection": len(tool_selection),
            "task_understanding": len(task_understanding),
            "decision_pairs": len(decision_pairs),
        },
        "skipped": dict(skipped),
        "note": (
            "导出即结构化记录，不在此训练；标签来自真实运行（decision_source / 实际 tool）。"
            "decision_pairs 只有「前一步带 signals」的 Observe→Decide 样本才被导出。"
        ),
    }
    (OUT_DIR / "decision_trace_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
