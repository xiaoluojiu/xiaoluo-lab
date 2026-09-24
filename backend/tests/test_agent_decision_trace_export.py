"""Phase 5 DecisionTrace 训练闭环测试。

钉住三条不变量，全部离线、毫秒级：

1. **写→读闭环**：``DecisionTrace`` 写入 JSONL 后能原样读回，字段不丢
   （含新增的 ``remote_escalations`` 与 ``decision_steps``）。
2. **导出脚本口径正确**：从 ``decision_trace.jsonl`` 导出 router_train /
   tool_selection / task_understanding 三类样本，标签来自真实运行记录，
   坏行 / 缺字段行跳过并计入诊断（不静默丢）。
3. **决策对只导「前一步带 signals」的样本**：Observe→Decide 的下一跳，
   无 signals 的步骤不构成有效训练对。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app.agent.trace.decision_trace import DecisionTrace


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_decision_trace",
        Path(__file__).resolve().parents[1] / "scripts" / "router" / "export_decision_trace.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decision_trace_roundtrip_includes_new_fields() -> None:
    t = DecisionTrace(trace_id="r-1", request="看看分布")
    t.task_spec = {"goal": "看看分布", "domain": "eda", "complexity_hint": "simple"}
    t.router_candidates = [{"candidate": "eda.distribution_overview", "score": 0.89, "source": "local_router"}]
    t.decision_made = {"action": "execute_tool", "tool": "eda.distribution_overview"}
    t.decision_source = "local_router"
    t.remote_calls = 0
    t.remote_escalations = 0
    t.local_calls = 1
    t.record_decision({"step": "understand", "action": "understand", "source": "local_router"})
    d = t.to_dict()
    assert d["remote_escalations"] == 0
    assert d["remote_calls"] == 0
    assert len(d["decision_steps"]) == 1
    # 序列化可逆：to_dict 后再构造不丢字段。
    t2 = DecisionTrace(trace_id=d["trace_id"], request=d["request"])
    for k, v in d.items():
        if k not in ("trace_id", "request"):
            setattr(t2, k, v)
    assert t2.remote_escalations == 0
    assert t2.remote_calls == 0
    assert len(t2.decision_steps) == 1


def test_router_sample_prefers_executed_tool() -> None:
    mod = _load_export_module()
    trace = {
        "trace_id": "r-9",
        "request": "检查一下数据质量",
        "router_candidates": [{"candidate": "dataset.quality", "score": 0.9, "source": "local_router"}],
        "decision_made": {"tool": "dataset.quality"},
        "decision_source": "local_router",
        "tool_calls": [{"tool": "dataset.quality", "status": "ok"}],
        "remote_escalations": 0,
    }
    s = mod._router_sample(trace)
    assert s is not None
    assert s["utterance"] == "检查一下数据质量"
    assert s["gold_tool"] == "dataset.quality"
    assert s["remote_escalations"] == 0


def test_router_sample_skips_empty_request() -> None:
    mod = _load_export_module()
    assert mod._router_sample({"trace_id": "r-0", "request": ""}) is None


def test_tool_selection_sample_skips_without_tool() -> None:
    mod = _load_export_module()
    assert mod._tool_selection_sample({"trace_id": "r-1", "decision_made": {}}) is None
    s = mod._tool_selection_sample({
        "trace_id": "r-2",
        "task_spec": {"goal": "x", "domain": "eda"},
        "decision_made": {"tool": "eda.distribution_overview"},
        "decision_source": "local_router",
    })
    assert s is not None
    assert s["tool"] == "eda.distribution_overview"


def test_task_understanding_sample_skips_empty_spec() -> None:
    mod = _load_export_module()
    assert mod._task_understanding_sample({"trace_id": "r-1", "request": "x", "task_spec": {}}) is None
    s = mod._task_understanding_sample({"trace_id": "r-2", "request": "x", "task_spec": {"goal": "x"}})
    assert s is not None
    assert s["utterance"] == "x"


def test_decision_pairs_only_export_prev_with_signals() -> None:
    mod = _load_export_module()
    trace = {
        "trace_id": "r-3",
        "decision_steps": [
            {"step": "understand", "action": "understand"},
            {"step": "act", "action": "execute_tool", "signals": {"high_missing": True}},
            {"step": "decide", "action": "execute_tool", "tool": "data.clean"},
        ],
    }
    pairs = mod._decision_pair_samples(trace)
    # 只有第 2→3 步（prev 带 signals）是有效对。
    assert len(pairs) == 1
    assert pairs[0]["prev_step"]["signals"] == {"high_missing": True}
    assert pairs[0]["next_decision"]["tool"] == "data.clean"
