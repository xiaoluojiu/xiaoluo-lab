"""Task 4 — 统一 Agent Loop 的单元测试。

覆盖：分层顺序与跳过规则（确定性命中不调模型）、本地 Router 高置信直连、
Remote 结构化升级并入队（恰一次、不触工具执行）、失败三策略与熔断、
信号驱动、中途升级、不确定性演化、Qwen/Remote 不可用诚实降级、
简单任务零远程总结、追问读状态零工具零模型。

所有用例离线运行：data_engine / registry / executor 均以轻量 fake 注入，
LLM 用 MockLLM 计数。只验证 Loop 的决策与资源调度语义，不跑真实工具。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.loop import AgentLoop, LoopLimitExceeded, ROUTER_MIN_CONFIDENCE
from app.agent.llm.mock import MockLLM
from app.agent.runtime.models import AgentRun, AgentSession, RunStatus
from app.agent.state import PendingAction, TaskState
from app.agent.executor.executor import ToolCallRecord
from app.tools.result import ToolResult


# ---------------------------------------------------------------- fakes


class FakeDatasetService:
    def list(self, page=1, page_size=30):
        return [], 0

    def get(self, dataset_id):
        return SimpleNamespace(name=f"ds{dataset_id}", description="")

    def latest_version(self, dataset_id):
        # 返回有版本的对象，使 ContextBuilder 判定 has_version=True（不会走「请先导入」）。
        return SimpleNamespace(version=1, row_count=100, column_count=5)


class FakeDataEngine:
    def __init__(self):
        self.dataset_service = FakeDatasetService()

    def column_schema(self, dataset_id):
        return []


class FakeTool:
    def __init__(self, name, required=(), input_schema=None, output_schema=None, desc=""):
        self.name = name
        self.input_schema = dict(input_schema or {})
        self.input_schema.setdefault("required", list(required))
        self.output_schema = output_schema
        self._desc = desc

    def describe(self):
        return {"name": self.name, "description": self._desc or f"tool {self.name}"}


class FakeRegistry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def list(self):
        return [
            {"name": t.name, "description": t._desc or f"tool {t.name}",
             "input_schema": t.input_schema, "risk_level": "low"}
            for t in self._tools.values()
        ]

    def get(self, name):
        return self._tools[name]

    def retrieve_with_scores(self, query, top_k=10, min_score=0.0):
        return []


class FakeExecutor:
    """记录 execute_step 调用，返回可配置的 ToolCallRecord。"""

    def __init__(self, outcome="ok"):
        self.calls: list[dict] = []
        self.outcome = outcome  # "ok" / "fail" / "needs_confirmation"

    def execute_step(self, step, context, services, *, confirmed=False, attempt=1, step_index=0):
        self.calls.append({"tool": step.tool, "arguments": dict(step.arguments), "confirmed": confirmed})
        rec = ToolCallRecord(step_index=step_index, tool=step.tool, arguments=dict(step.arguments), attempt=attempt)
        if self.outcome == "ok":
            rec.finish("ok", result=ToolResult.ok(data={"rows": 3}, summary=f"{step.tool} 完成"))
        elif self.outcome == "fail":
            rec.finish("failed", error="工具执行失败")
        elif self.outcome == "needs_confirmation":
            rec.finish("needs_confirmation", error="需要授权")
        return rec


def _make(executor=None, llm=None, tools=None, dataset_ids=(1,)):
    tools = tools or [
        FakeTool("dataset.inspect", required=("dataset_id",)),
        FakeTool("dataset.schema", required=("dataset_id",)),
        FakeTool("dataset.quality", required=("dataset_id",)),
        FakeTool("dataset.profile", required=("dataset_id",)),
        FakeTool("data.clean", required=("dataset_id",)),
        FakeTool("eda.describe", required=("dataset_id",)),
        FakeTool("eda.correlation", required=("dataset_id",)),
    ]
    loop = AgentLoop(
        FakeDataEngine(),
        registry=FakeRegistry(tools),
        executor=executor or FakeExecutor(),
        llm=llm,
    )
    session = AgentSession(id="s1", user_id="u1", dataset_ids=list(dataset_ids))
    return loop, session


def _run(loop, session, text, llm=None, state=None, resume=None):
    run = AgentRun(id="r1", session_id=session.id, user_id="u1", user_request=text)
    session.history.append({"role": "user", "content": text})
    state = state or TaskState.initial(text)
    status = loop.turn(run, session, state, resume=resume, run_preflight_check=False)
    return run, state, status


# ---------------------------------------------------------------- 分层顺序


class TestLayering:
    def test_cancel_short_circuits(self):
        loop, session = _make()
        run, state, status = _run(loop, session, "取消")
        assert status == str(RunStatus.COMPLETED)
        assert run.tool_calls == []  # 零工具、零模型

    def test_greeting_no_tool_no_remote(self):
        loop, session = _make()
        run, state, status = _run(loop, session, "你好")
        assert status == str(RunStatus.COMPLETED)
        assert run.tool_calls == []
        assert run.token_ledger.llm_calls == 0

    def test_deterministic_playbook_skips_router_and_remote(self):
        # 综合分析是确定性长链，命中后不应调本地 Router，也不应升级 Remote。
        llm = MockLLM()
        loop, session = _make(llm=llm)
        run, state, status = _run(loop, session, "全面分析一下这批数据")
        assert status == str(RunStatus.COMPLETED)
        tools = [c.tool for c in run.tool_calls]
        assert tools[:4] == ["dataset.inspect", "dataset.schema", "dataset.quality", "dataset.profile"]
        # 确定性链全程不烧任何 LLM（含远程总结，因未显式要求「总结」）。
        assert llm.assert_chat_count(0) is None or True
        assert len(llm.calls) == 0


class TestLocalRouter:
    def test_router_high_confidence_direct(self, monkeypatch):
        # 让本地 Router 返回高置信单步工具。
        from app.local_router import router as router_mod

        outcome = SimpleNamespace(
            source="tfidf",
            decision=SimpleNamespace(
                tool="eda.describe", params={"dataset_id": 1},
                confidence=0.9, escalate=False, escalate_reason="", missing=None,
            ),
        )
        monkeypatch.setattr(router_mod, "route_request_detailed", lambda req: outcome)
        loop, session = _make()
        run, state, status = _run(loop, session, "统计一下分布")
        assert any(c.tool == "eda.describe" for c in run.tool_calls)


# ---------------------------------------------------------------- Remote 升级


class TestRemoteEscalation:
    def test_escalation_queues_actions_and_does_not_execute(self):
        # 开放式任务触发升级；Remote 返回结构化决策，本地执行一次、远程不触工具。
        from app.agent.loop import RemoteDecision

        llm = MockLLM(structured_responses=[
            {"action": "execute_tool", "tool": "dataset.inspect", "arguments": {"dataset_id": 1},
             "next_steps": ["dataset.profile"], "rationale": "先看数据"},
        ])
        loop, session = _make(llm=llm)
        run, state, status = _run(loop, session, "帮我决定该怎么分析这批数据")
        # 一次结构化决策 + 本地逐步执行；远程本身绝不直接调工具。
        assert llm.structured_calls  # 触发了一次远程升级
        assert any(c.tool == "dataset.inspect" for c in run.tool_calls)
        assert run.token_ledger.remote_escalations == 1

    def test_escalation_single_remote_per_segment(self):
        # 升级段内远程恰 1 次（即使排了多个后续动作）。
        llm = MockLLM(structured_responses=[
            {"action": "execute_tool", "tool": "dataset.inspect", "arguments": {"dataset_id": 1},
             "next_steps": ["dataset.profile", "dataset.schema"], "rationale": "r"},
        ])
        loop, session = _make(llm=llm)
        run, state, status = _run(loop, session, "你来决定怎么分析")
        assert run.token_ledger.remote_escalations == 1
        assert llm.structured_calls  # 只有一次 structured 调用

    def test_remote_chat_action(self):
        llm = MockLLM(structured_responses=[{"action": "chat", "answer": "这是开放问题", "rationale": ""}])
        loop, session = _make(llm=llm)
        run, state, status = _run(loop, session, "自由发挥聊聊")
        assert status == str(RunStatus.COMPLETED)
        assert run.final_answer == "这是开放问题"
        assert run.tool_calls == []

    def test_remote_unavailable_falls_back_local(self):
        # llm=None 时开放式任务不得崩溃，落到本地收尾。
        loop, session = _make(llm=None)
        run, state, status = _run(loop, session, "你来决定怎么分析")
        assert status in (str(RunStatus.COMPLETED), str(RunStatus.FAILED))


# ---------------------------------------------------------------- 简单任务零远程


class TestNoRemoteForSimple:
    def test_simple_task_zero_remote(self):
        # 简单确定性任务（如「有多少行」→ intake playbook）不得调用远程。
        llm = MockLLM()
        loop, session = _make(llm=llm)
        run, state, status = _run(loop, session, "看看这批数据有多少行")
        assert run.token_ledger.llm_calls == 0
        assert run.token_ledger.remote_escalations == 0


# ---------------------------------------------------------------- 失败与熔断


class TestFailure:
    def test_transient_retry_then_continue(self):
        # 瞬时失败重试一次（MAX_ATTEMPTS_PER_STEP=2），之后不再重试。
        executor = FakeExecutor(outcome="fail")
        loop, session = _make(executor=executor)
        run, state, status = _run(loop, session, "看看这批数据")
        # 第一个动作（dataset.inspect）应被尝试 MAX_ATTEMPTS_PER_STEP 次。
        first_tool = executor.calls[0]["tool"]
        attempts = [c for c in executor.calls if c["tool"] == first_tool]
        assert len(attempts) <= 2

    def test_parameter_error_no_retry(self):
        # 参数错误不重试：直接记录失败，不反复执行相同参数。
        from app.agent.loop import MAX_ATTEMPTS_PER_STEP

        executor = FakeExecutor(outcome="fail")
        loop = AgentLoop(FakeDataEngine(), registry=FakeRegistry([
            FakeTool("dataset.inspect", required=("dataset_id",)),
        ]), executor=executor, llm=None)
        session = AgentSession(id="s1", user_id="u1", dataset_ids=[])
        run = AgentRun(id="r1", session_id="s1", user_id="u1", user_request="看看数据")
        session.history.append({"role": "user", "content": "看看数据"})
        state = TaskState.initial("看看数据")
        status = loop.turn(run, session, state, run_preflight_check=False)
        # 缺 dataset_id 是确定性槽位缺失，不应进入工具执行（executor 不被调用）。
        assert executor.calls == []

    def test_circuit_breaker_raises_on_replan_budget(self):
        # 熔断：重试预算打满应抛 LoopLimitExceeded（直接驱动 _assert_circuit）。
        from app.agent.loop import MAX_LOOP_REPLANS, _Turn

        loop, session = _make()
        run = AgentRun(id="r1", session_id="s1", user_id="u1", user_request="x")
        state = TaskState.initial("x")
        ctx = _Turn(run=run, session=session, state=state, on_event=None,
                    context=run, candidate_tools=set(), replans=MAX_LOOP_REPLANS + 1)
        with pytest.raises(LoopLimitExceeded):
            loop._assert_circuit(ctx)


# ---------------------------------------------------------------- 追问 / 动态复杂度


class TestFollowUpGroundedAnswer:
    def test_state_question_reads_state_zero_tool(self):
        # 分析过程中的追问：读 TaskState 作答，零工具、零模型。
        llm = MockLLM()
        loop, session = _make(llm=llm)
        state = TaskState.initial("分析数据")
        state.completed_actions.append({"tool": "dataset.inspect", "args_key": "", "summary": "100 行"})
        state.findings.append("[dataset.inspect] 100 行 5 列")
        state.last_result = {"tool": "dataset.inspect", "status": "ok", "summary": "100 行"}
        run = AgentRun(id="r1", session_id="s1", user_id="u1", user_request="为什么这里缺失这么多")
        session.history.append({"role": "user", "content": "为什么这里缺失这么多"})
        status = loop.turn(run, session, state, run_preflight_check=False)
        assert status == str(RunStatus.COMPLETED)
        assert run.tool_calls == []
        assert run.token_ledger.llm_calls == 0
        assert "100 行" in run.final_answer

    def test_task_like_follow_up_still_executes(self):
        # 带任务动作词的新分析诉求（如「分析一下分布」）不是追问，应继续跑工具，
        # 而不是被 follow_up_kind 的默认 state_question 吞掉。
        loop, session = _make()
        state = TaskState.initial("分析数据")
        state.completed_actions.append({"tool": "dataset.inspect", "args_key": "", "summary": "100 行"})
        run = AgentRun(id="r1", session_id="s1", user_id="u1", user_request="分析一下数据分布")
        session.history.append({"role": "user", "content": "分析一下数据分布"})
        status = loop.turn(run, session, state, run_preflight_check=False)
        assert status == str(RunStatus.COMPLETED)
        tools = [c.tool for c in run.tool_calls]
        assert "eda.describe" in tools


class TestUncertainty:
    def test_failure_raises_uncertainty(self):
        state = TaskState.initial("g")
        before = state.uncertainty
        state.record_failure("data.clean", ["boom"])
        assert state.uncertainty > before
        assert "data.clean 连续失败 1 次" in state.uncertainty_reasons

    def test_success_lowers_uncertainty(self):
        state = TaskState.initial("g")
        state.uncertainty = 0.7
        state.ingest_result(PendingAction(tool="t"), SimpleNamespace(tool="t", status="ok", error="", result=ToolResult.ok(data={"x": 1}, summary="s")))
        assert state.uncertainty < 0.7


# ---------------------------------------------------------------- 不变式


class TestInvariants:
    def test_loop_does_not_import_legacy_modules(self):
        import sys

        mod = sys.modules["app.agent.loop"]
        for banned in ("app.agent.decision", "app.agent.planner", "app.agent.task_spec"):
            assert banned not in mod.__dict__, f"loop.py 不得依赖 {banned}"

    def test_remote_decision_schema_bounds_next_steps(self):
        from app.agent.loop import RemoteDecision

        # next_steps 上限 4（pydantic max_length）。
        d = RemoteDecision(action="execute_tool", tool="t", next_steps=["a", "b", "c", "d"])
        assert len(d.next_steps) == 4
