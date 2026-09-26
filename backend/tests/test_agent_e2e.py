"""Agent 端到端链路测试：确认 / 反问闭环（真实 Executor + 自定义工具）。

重构后（统一 Agent Loop）：决策（playbook / local_router / remote）与执行分离。
失败重试、参数错误、熔断等「执行层语义」已由 ``test_agent_loop.py``（FakeExecutor）
覆盖；本文件只保留**必须经真实 Executor + PermissionManager 才能验证**的闭环：

1. 高风险工具 → WAITING_CONFIRMATION → resume（原子领取 + 授权一次性凭据）
2. 反问 → WAITING_CLARIFICATION → answer_clarification（从原步骤继续，不重跑前置步骤）

驱动方式：不再用 plan_override（已随旧 Planner 删除），改用「预置待办动作 + loop.turn」。
"""
from __future__ import annotations

import time

import polars as pl
import pytest
from app.agent.clarify import (
    ANSWERS_KEY,
    Clarification,
    ClarificationOption,
    ClarificationRequired,
)
from app.agent.permission.models import ROLE_PERMISSIONS, Permission
from app.agent.permission.rules import RiskLevel  # noqa: F401
from app.agent.runtime.models import AgentRun, RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.agent.state import PendingAction, TaskState
from app.core.exceptions import NotFoundException, ValidationException
from app.data_engine.service import DataEngineService
from app.services.dataset_service import DatasetService
from app.tools.base import Tool
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


# ---------------------------------------------------------------------------
# 测试工具：行为可控，便于构造高风险 / 反问分支
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    name = "test.echo"
    description = "回显输入文本"
    category = "test"
    input_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, params, context, services) -> ToolResult:
        self.calls.append(dict(params))
        return ToolResult.ok({"echo": params.get("text")}, summary=f"回显：{params.get('text')}")


class RiskyTool(Tool):
    """高风险工具：必须经过 PermissionManager → Confirmation → Execute。"""

    name = "test.risky"
    description = "高风险的测试工具（写操作）"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    permission = Permission.TRAIN_MODEL
    risk_level = RiskLevel.HIGH

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        return ToolResult.ok({"done": True}, summary="高风险操作已执行")


class SecondRiskyTool(RiskyTool):
    """第二个高风险工具：验证「一次确认不会顺带放行第二个高风险调用」。"""

    name = "test.risky2"
    description = "第二个高风险的测试工具（写操作）"


class ClarifyTool(Tool):
    """计划执行中发起结构化反问；回答一次后不再问。"""

    name = "test.clarify"
    description = "需要用户补充信息的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        extra = getattr(context, "extra", {}) or {}
        answers = extra.get(ANSWERS_KEY) or {}
        if answers.get("test.target"):
            return ToolResult.ok({"target": answers["test.target"]}, summary="已按回答继续")
        raise ClarificationRequired(
            Clarification(
                code="test.target",
                question="请问要分析哪一列？",
                options=[ClarificationOption(value="x1", label="x1"), ClarificationOption(value="x2", label="x2")],
                default="x1",
            )
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine(db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("e2e", "端到端测试数据")
    ds.create_version(dataset.id, pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    return {"ds": ds, "engine": DataEngineService(ds), "dataset_id": dataset.id}


def _runtime(engine, *, tools) -> AgentRuntime:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRuntime(engine["engine"], llm=None, registry=registry)


def _start_run(runtime: AgentRuntime, session, actions, request: str = "执行操作") -> AgentRun:
    """用预置待办动作启动一轮运行（等价于旧 plan_override 固定计划）。"""
    state = TaskState.initial(request)
    state.queue([PendingAction(tool=t, arguments=dict(args or {}), source="test") for t, args in actions])
    run = AgentRun(
        id=runtime.store.next_run_id(), session_id=session.id,
        user_id=session.user_id, user_request=request,
    )
    runtime.store.add_run(run)
    if run.id not in session.run_ids:
        session.run_ids.append(run.id)
    session.history.append({"role": "user", "content": request})
    run.started_at = time.time()
    runtime.loop.turn(run, session, state, run_preflight_check=False)
    session.task_state = state.to_dict()
    return run


# ---------------------------------------------------------------------------
# 场景：高风险工具 → 待确认 → 确认 → 继续
# ---------------------------------------------------------------------------


class TestPermissionConfirmation:
    def test_waiting_confirmation_then_resume(self, engine):
        risky = RiskyTool()
        runtime = _runtime(engine, tools=[risky])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = _start_run(runtime, session, [("test.risky", {})])

        assert run.status == RunStatus.WAITING_CONFIRMATION
        assert run.pending_confirmation is not None
        assert risky.calls == 0

        resumed = runtime.resume(session, run.id)

        assert resumed.status == RunStatus.COMPLETED, resumed.error
        assert risky.calls == 1
        assert resumed.pending_confirmation is None

    def test_each_high_risk_step_requires_its_own_confirmation(self, engine):
        """★ P0 回归：一次确认最多授权一次高风险调用。"""
        first = RiskyTool()
        second = SecondRiskyTool()
        runtime = _runtime(engine, tools=[first, second])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = _start_run(runtime, session, [("test.risky", {}), ("test.risky2", {})])
        assert run.status == RunStatus.WAITING_CONFIRMATION
        assert first.calls == 0
        assert second.calls == 0

        resumed = runtime.resume(session, run.id)
        assert resumed.status == RunStatus.WAITING_CONFIRMATION, resumed.error
        assert first.calls == 1
        assert second.calls == 0  # 关键断言：不能因为上一步确认过就被静默放行

        resumed2 = runtime.resume(session, run.id)
        assert resumed2.status == RunStatus.COMPLETED, resumed2.error
        assert second.calls == 1
        assert resumed2.pending_confirmation is None

    def test_low_risk_steps_are_not_blocked_after_confirmation(self, engine):
        risky = RiskyTool()
        echo = EchoTool()
        runtime = _runtime(engine, tools=[risky, echo])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = _start_run(runtime, session, [("test.risky", {}), ("test.echo", {"text": "hi"})])
        assert run.status == RunStatus.WAITING_CONFIRMATION

        resumed = runtime.resume(session, run.id)
        assert resumed.status == RunStatus.COMPLETED, resumed.error
        assert risky.calls == 1
        assert [c.get("text") for c in echo.calls] == ["hi"]

    def test_confirmation_cannot_be_bypassed_by_direct_tool_call(self):
        """权限边界：即使直接走 Registry，未经 confirmed 也必须被拦下。"""
        from app.tools.base import ToolConfirmationRequired

        risky = RiskyTool()
        registry = ToolRegistry()
        registry.register(risky)
        context = ToolExecutionContext(
            user_id="u1", session_id="s1", dataset_ids=set(),
            permissions=set(ROLE_PERMISSIONS["analyst"]),
        )
        with pytest.raises(ToolConfirmationRequired):
            registry.execute("test.risky", {}, context, None, confirmed=False)
        assert risky.calls == 0

    def test_deny_terminates_run(self, engine):
        runtime = _runtime(engine, tools=[RiskyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = _start_run(runtime, session, [("test.risky", {})])

        denied = runtime.deny(run.id)

        assert denied.status == RunStatus.FAILED
        assert denied.pending_confirmation is None
        assert "拒绝" in denied.error


# ---------------------------------------------------------------------------
# 场景：反问 → 待澄清 → 回答 → 从原步骤继续
# ---------------------------------------------------------------------------


class TestClarification:
    def test_clarify_then_answer_resumes_current_step(self, engine):
        echo = EchoTool()
        clarify = ClarifyTool()
        runtime = _runtime(engine, tools=[echo, clarify])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = _start_run(runtime, session, [("test.echo", {"text": "first"}), ("test.clarify", {})])

        assert run.status == RunStatus.WAITING_CLARIFICATION
        assert run.pending_clarification is not None
        assert run.pending_confirmation is None
        assert run.pending_clarification["code"] == "test.target"
        assert echo.calls == [{"text": "first"}]

        resumed = runtime.answer_clarification(run.id, "x2")

        assert resumed.status == RunStatus.COMPLETED, resumed.error
        assert echo.calls == [{"text": "first"}]  # 前面的步骤不重复执行
        assert clarify.calls == 2  # 反问一次 + 回答后重跑该步一次
        assert run.clarification_answers["test.target"] == "x2"
        assert resumed.pending_clarification is None

    def test_duplicate_answer_rejected(self, engine):
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = _start_run(runtime, session, [("test.clarify", {})])
        assert run.status == RunStatus.WAITING_CLARIFICATION

        runtime.answer_clarification(run.id, "x1")

        with pytest.raises(ValidationException):
            runtime.answer_clarification(run.id, "x1")

    def test_empty_answer_rejected(self, engine):
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = _start_run(runtime, session, [("test.clarify", {})])

        with pytest.raises(ValidationException):
            runtime.answer_clarification(run.id, "   ")

        assert run.status == RunStatus.WAITING_CLARIFICATION
        assert run.pending_clarification is not None

    def test_answer_unknown_run(self, engine):
        runtime = _runtime(engine, tools=[ClarifyTool()])
        with pytest.raises(NotFoundException):
            runtime.answer_clarification("r-does-not-exist", "x1")

    def test_cancel_while_waiting_clarification(self, engine):
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = _start_run(runtime, session, [("test.clarify", {})])
        assert run.status == RunStatus.WAITING_CLARIFICATION

        canceled = runtime.cancel(run.id)

        assert canceled.status == RunStatus.FAILED
        assert canceled.pending_clarification is None
        assert "澄清" in canceled.error

    def test_deny_while_waiting_confirmation_keeps_message(self, engine):
        runtime = _runtime(engine, tools=[RiskyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = _start_run(runtime, session, [("test.risky", {})])

        denied = runtime.deny(run.id)
        assert "拒绝授权" in denied.error
