"""Task 2 — 统一 TaskState / 账本 / 会话级持久化 的单元测试。

覆盖：状态构造与队列去重、结果紧凑吸收（facts/findings/artifacts/signals）、
紧凑远程视图不含完整 ToolResult、多轮接续（follow_up_kind / apply_constraint_change）、
序列化对称、账本新字段、AgentStore 往返与旧 JSON 夹具兼容。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.runtime.models import AgentSession, AgentStore, AgentTokenLedger
from app.agent.state import (
    MAX_COMPLETED,
    PendingAction,
    Phase,
    TaskState,
)
from app.tools.result import ToolResult


def _record(tool: str, status: str = "ok", data=None, summary: str = "", signals=None, error: str = ""):
    """构造一个与 ToolCallRecord 形状兼容的最小 mock（ingest_result 只读 result/status/error）。"""
    result = ToolResult.ok(data=data, summary=summary, signals=signals) if status == "ok" else ToolResult.fail(errors=[error] if error else ["failed"])
    return SimpleNamespace(tool=tool, status=status, result=result, error=error)


# --------------------------------------------------------------------------- 构造与队列


class TestConstruction:
    def test_initial_sets_goal_and_defaults(self):
        s = TaskState.initial("分析这批数据")
        assert s.goal == "分析这批数据"
        assert s.phase == Phase.INTAKE
        assert s.constraints == []
        assert s.pending_actions == []

    def test_initial_uncertainty_high_when_no_goal(self):
        assert TaskState.initial("").uncertainty == pytest.approx(0.8)
        assert TaskState.initial("x").uncertainty == pytest.approx(0.3)

    def test_initial_keeps_dataset_context(self):
        s = TaskState.initial("g", dataset_context={"42": {"name": "d", "rows": 10}})
        assert s.dataset_context == {"42": {"name": "d", "rows": 10}}


class TestQueue:
    def test_queue_adds_and_orders(self):
        s = TaskState.initial("g")
        a1 = PendingAction(tool="data.inspect", arguments={"x": 1})
        a2 = PendingAction(tool="data.quality", arguments={"x": 1})
        added = s.queue([a1, a2])
        assert len(added) == 2
        assert [a.tool for a in s.pending_actions] == ["data.inspect", "data.quality"]

    def test_queue_dedupes_against_pending(self):
        s = TaskState.initial("g")
        a = PendingAction(tool="data.inspect", arguments={"x": 1})
        s.queue(a)
        again = s.queue(PendingAction(tool="data.inspect", arguments={"x": 1}))
        assert again == []
        assert len(s.pending_actions) == 1

    def test_queue_dedupes_against_completed(self):
        s = TaskState.initial("g")
        s.completed_actions.append({"tool": "data.inspect", "args_key": '{"x": 1}'})
        added = s.queue(PendingAction(tool="data.inspect", arguments={"x": 1}))
        assert added == []

    def test_pop_next_fifo(self):
        s = TaskState.initial("g")
        a1 = PendingAction(tool="t1")
        a2 = PendingAction(tool="t2")
        s.queue([a1, a2])
        assert s.pop_next().tool == "t1"
        assert s.pop_next().tool == "t2"
        assert s.pop_next() is None
        assert not s.has_pending()


# --------------------------------------------------------------------------- 结果吸收


class TestIngestResult:
    def test_ok_records_fact_finding_and_completed(self):
        s = TaskState.initial("g")
        a = PendingAction(tool="data.inspect", arguments={"x": 1})
        s.ingest_result(a, _record("data.inspect", data={"rows": 100, "cols": 5}, summary="100 行 5 列"))
        assert s.facts["data.inspect"] == {"rows": 100, "cols": 5}
        assert s.findings[0].startswith("[data.inspect]")
        assert s.completed_actions[-1]["tool"] == "data.inspect"

    def test_ok_lowers_uncertainty(self):
        s = TaskState.initial("g")
        before = s.uncertainty
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data={"rows": 1}, summary="s"))
        assert s.uncertainty < before

    def test_signals_collected(self):
        s = TaskState.initial("g")
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data={}, signals=["high_missing"]))
        assert "high_missing" in s.signals
        assert "high_missing" in s.last_result["signals"]

    def test_artifact_collected(self):
        s = TaskState.initial("g")
        s.ingest_result(PendingAction(tool="ml.train"), _record("ml.train", data={"run_id": "abc"}, summary="ok"))
        assert {"tool": "ml.train", "field": "run_id", "value": "abc"} in s.artifacts

    def test_failure_records_and_raises_uncertainty(self):
        s = TaskState.initial("g")
        before = s.uncertainty
        s.ingest_result(PendingAction(tool="data.clean"), _record("data.clean", status="failed", error="boom"))
        assert s.failures["data.clean"] == 1
        assert s.uncertainty > before
        assert "data.clean" not in s.facts

    def test_fact_replaced_not_appended(self):
        s = TaskState.initial("g")
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data={"rows": 1}, summary="a"))
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data={"rows": 2}, summary="b"))
        assert s.facts["data.inspect"] == {"rows": 2}

    def test_completed_bounded(self):
        s = TaskState.initial("g")
        for i in range(MAX_COMPLETED + 5):
            s.ingest_result(PendingAction(tool=f"t{i}"), _record(f"t{i}", data={}, summary="s"))
        assert len(s.completed_actions) == MAX_COMPLETED


# --------------------------------------------------------------------------- 紧凑远程视图


class TestCompactForRemote:
    def test_compact_excludes_full_tool_result(self):
        s = TaskState.initial("g")
        # 150 行原始数据不应进入远程载荷（S14 前置不变式）。
        big = {"rows": [{"col" + str(i): i} for i in range(150)]}
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data=big, summary="150 rows"))
        payload = s.compact_for_remote()
        # facts 以截断字符串形态出现，绝不包含原始 dict/完整行对象。
        facts_blob = payload["facts"].get("data.inspect", "")
        assert isinstance(facts_blob, str)
        assert len(facts_blob) <= 600  # 受 per_tool 预算限制
        # 载荷不包含 "ToolResult"/完整历史字段
        assert "tool_calls" not in payload
        assert "history" not in payload

    def test_compact_has_required_keys(self):
        s = TaskState.initial("g", dataset_context={"1": {"name": "d", "rows": 5, "columns": ["a"]}})
        p = s.compact_for_remote()
        for key in ("phase", "goal", "constraints", "datasets", "findings", "artifacts", "signals", "facts", "completed_tools", "next_actions", "uncertainty"):
            assert key in p


# --------------------------------------------------------------------------- 多轮接续


class TestFollowUp:
    def test_new_task_markers(self):
        s = TaskState.initial("g")
        assert s.follow_up_kind("重新开始") == "new_task"
        assert s.follow_up_kind("换个任务") == "new_task"

    def test_constraint_change(self):
        s = TaskState.initial("g")
        assert s.follow_up_kind("不要删这一列，改成填充") == "constraint_change"
        assert s.follow_up_kind("把这个换成中位数填充") == "constraint_change"

    def test_state_question_default(self):
        s = TaskState.initial("g")
        assert s.follow_up_kind("为什么这里缺失这么多") == "state_question"


class TestApplyConstraintChange:
    def test_rewrites_clean_strategy_drop_to_mean(self):
        s = TaskState.initial("g")
        s.queue(PendingAction(tool="data.clean", arguments={"missing": {"strategy": "drop"}}))
        changed = s.apply_constraint_change("不要删这一列，改成填充")
        assert changed is True
        assert s.pending_actions[0].arguments["missing"]["strategy"] == "mean"

    def test_rewrites_to_median(self):
        s = TaskState.initial("g")
        s.queue(PendingAction(tool="data.clean", arguments={"missing": {"strategy": "drop"}}))
        s.apply_constraint_change("不要删列，用中位数填充")
        assert s.pending_actions[0].arguments["missing"]["strategy"] == "median"

    def test_no_clean_action_no_change(self):
        s = TaskState.initial("g")
        changed = s.apply_constraint_change("不要删这一列，改成填充")
        assert changed is False

    def test_appends_constraint(self):
        s = TaskState.initial("g")
        s.apply_constraint_change("不要删这一列")
        assert any("不要删" in c for c in s.constraints)


# --------------------------------------------------------------------------- 序列化对称


class TestSerialization:
    def test_roundtrip_symmetry(self):
        s = TaskState.initial("g", dataset_context={"1": {"name": "d"}})
        s.queue(PendingAction(tool="data.clean", arguments={"missing": {"strategy": "drop"}}))
        s.ingest_result(PendingAction(tool="data.inspect"), _record("data.inspect", data={"rows": 3}, summary="3 rows"))
        s.signals.append("high_missing")
        s.failures["x"] = 2
        s.uncertainty = 0.7
        restored = TaskState.from_dict(s.to_dict())
        assert restored is not None
        assert restored.to_dict() == s.to_dict()
        assert restored.pending_actions[0].tool == "data.clean"
        assert restored.uncertainty == pytest.approx(0.7)

    def test_from_dict_none(self):
        assert TaskState.from_dict(None) is None


# --------------------------------------------------------------------------- 账本


class TestLedger:
    def test_new_keys_present(self):
        l = AgentTokenLedger()
        d = l.to_dict()
        for key in ("remote_calls", "remote_input_tokens", "remote_output_tokens", "qwen_calls", "tool_calls", "task_steps", "escalation_count", "total_cost"):
            assert key in d, key

    def test_old_keys_preserved(self):
        l = AgentTokenLedger()
        d = l.to_dict()
        for key in ("llm_calls", "remote_escalations", "actual", "budget", "optimization"):
            assert key in d

    def test_remote_calls_mirrors_llm_calls(self):
        l = AgentTokenLedger()
        l.record_usage({"input_tokens": 10, "output_tokens": 5})
        d = l.to_dict()
        assert d["remote_calls"] == d["llm_calls"] == 1
        assert d["remote_input_tokens"] == 10
        assert d["remote_output_tokens"] == 5

    def test_counters_increment(self):
        l = AgentTokenLedger()
        l.record_qwen_call(2)
        l.record_tool_call(3)
        l.record_step(4)
        l.record_escalation()
        d = l.to_dict()
        assert d["qwen_calls"] == 2
        assert d["tool_calls"] == 3
        assert d["task_steps"] == 4
        assert d["escalation_count"] == 1

    def test_total_cost_zero_by_default(self):
        assert AgentTokenLedger().to_dict()["total_cost"] == 0.0


# --------------------------------------------------------------------------- Store 往返与旧夹具


class TestStoreRoundtrip:
    def test_session_task_state_persists(self, tmp_path):
        store = AgentStore(path=tmp_path / "store.json")
        sess = AgentSession(id="s1", user_id="u1", title="t")
        sess.task_state = TaskState.initial("g", dataset_context={"1": {"name": "d"}}).to_dict()
        store.add_session(sess)
        loaded = store.get_session("s1")
        assert loaded.task_state is not None
        assert loaded.task_state["goal"] == "g"

    def test_old_json_without_new_fields_loads(self, tmp_path):
        # 旧格式：无 task_state、无账本新键。
        old = {
            "session_seq": 1,
            "run_seq": 1,
            "sessions": [{"id": "s1", "user_id": "u1", "title": "t", "dataset_ids": [], "history": [], "run_ids": ["r1"], "created_at": 0.0, "archived": False}],
            "runs": [{"id": "r1", "session_id": "s1", "user_id": "u1", "user_request": "hi", "status": "completed", "final_answer": "hi", "error": "", "token_usage": {"llm_calls": 1, "actual": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "optimization": {}}}],
        }
        path = tmp_path / "old.json"
        path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        store = AgentStore(path=path)
        sess = store.get_session("s1")
        assert sess.task_state is None
        run = store.get_run("r1")
        assert run.token_ledger.llm_calls == 1
        assert run.token_ledger.qwen_calls == 0
        assert run.token_ledger.task_steps == 0
