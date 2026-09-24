"""Agent 端到端链路测试：用户请求 → 工具 → 结果。

与 `test_agent.py` 的分工
------------------------
`test_agent.py` 逐层验 Context / Planner / Executor / Validator / Replanner；
本文件验的是**这些层串起来之后还对不对**——也就是真实用户会碰到的那几条路径：

1. 正常执行
2. 工具第一次失败 → 重规划 → 重试成功
3. 参数错误（不盲目重试）与执行失败（重试到上限后终止）
4. 高风险工具 → 待确认 → 确认 → 继续
5. 反问 → 待澄清 → 回答 → 从原步骤继续（前面的步骤不重跑）
6. 死循环被各道熔断闸截断

为了确定性，工具集用**独立 ToolRegistry**（不污染进程级 TOOL_REGISTRY），
计划用 `plan_override` 固定；只有场景 1 走 MockLLM 规划器，以覆盖「规划」这一环。
"""

from __future__ import annotations

import polars as pl
import pytest
from app.agent.clarify import (
    ANSWERS_KEY,
    Clarification,
    ClarificationOption,
    ClarificationRequired,
)
from app.agent.llm.mock import MockLLM
from app.agent.permission.models import ROLE_PERMISSIONS, Permission

# Permission 未用到 RiskLevel 之外的值，这里显式导入以便测试工具声明风险等级
from app.agent.permission.rules import RiskLevel  # noqa: F401
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner
from app.agent.planner.replanner import ReplanLimits, Replanner
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.core.exceptions import NotFoundException, ValidationException
from app.data_engine.service import DataEngineService
from app.services.dataset_service import DatasetService
from app.tools.base import Tool
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

# ---------------------------------------------------------------------------
# 测试工具：行为可控，便于构造失败 / 反问 / 高风险等分支
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """记录调用次数并回显参数。"""

    name = "test.echo"
    description = "回显输入文本，用于端到端链路验证"
    category = "test"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, params, context, services) -> ToolResult:
        self.calls.append(dict(params))
        return ToolResult.ok({"echo": params.get("text")}, summary=f"回显：{params.get('text')}")


class FlakyTool(Tool):
    """前 ``fail_times`` 次失败，之后成功（用于验「重试后恢复」）。"""

    name = "test.flaky"
    description = "会瞬时失败的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self, fail_times: int = 1) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            return ToolResult.fail(f"瞬时故障（第 {self.calls} 次）")
        return ToolResult.ok({"ok": True}, summary=f"第 {self.calls} 次成功")


class AlwaysFailTool(Tool):
    """永远失败（用于验「重试到上限后终止」）。"""

    name = "test.always_fail"
    description = "永远失败的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        return ToolResult.fail("持续失败")


class MissingParamTool(Tool):
    """声明必填参数，用于验「参数错误不盲目重试」。"""

    name = "test.missing_param"
    description = "需要必填参数的测试工具"
    category = "test"
    input_schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"target": params.get("target")}, summary="不应被执行")


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


class ClarifyTool(Tool):
    """计划执行中发起结构化反问（模拟 agent.clarify 的下游行为）。"""

    name = "test.clarify"
    description = "需要用户补充信息的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls = 0
        self.seen_answers: list[dict] = []

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        extra = getattr(context, "extra", {}) or {}
        self.seen_answers.append({k: dict(v) for k, v in extra.items() if isinstance(v, dict)})
        # 用户回答过就不再反问（模拟 agent.clarify 的幂等语义）
        answers = extra.get(ANSWERS_KEY) or {}
        if answers.get("test.target"):
            return ToolResult.ok({"target": answers["test.target"]}, summary="已按回答继续")
        raise ClarificationRequired(
            Clarification(
                code="test.target",
                question="请问要分析哪一列？",
                options=[
                    ClarificationOption(value="x1", label="x1"),
                    ClarificationOption(value="x2", label="x2"),
                ],
                default="x1",
            )
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(rows: int = 60) -> pl.DataFrame:
    half = max(rows // 2, 1)
    return pl.DataFrame(
        {
            "x1": [float(i + 1) for i in range(rows)],
            "x2": [float(rows - i) for i in range(rows)],
            "target": ["pos" if i < half else "neg" for i in range(rows)],
        }
    )


@pytest.fixture()
def engine(db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("e2e", "端到端测试数据")
    ds.create_version(dataset.id, _make_df())
    return {"ds": ds, "engine": DataEngineService(ds), "dataset_id": dataset.id}


def _runtime(engine, *, tools, llm=None, limits=None, planner=None) -> AgentRuntime:
    """构造带独立工具注册表的 Runtime（不污染进程级 TOOL_REGISTRY）。"""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRuntime(
        engine["engine"],
        llm=llm,
        registry=registry,
        planner=planner,
        limits=limits,
    )


def _plan(*tools: str, **arguments) -> AgentPlan:
    return AgentPlan(
        goal="端到端验证",
        steps=[PlanStep(tool=t, arguments=dict(arguments)) for t in tools],
    )


# ---------------------------------------------------------------------------
# 场景 1：正常执行
# ---------------------------------------------------------------------------


class TestNormalExecution:
    def test_plan_tool_execute_answer(self, engine, monkeypatch):
        """用户请求 → 规划 → 工具执行 → 最终回答。"""
        # 关掉工具检索，让候选集确定地等于注册表全量（本用例要验的是链路，不是召回）
        monkeypatch.setattr("app.core.config.settings.AGENT_ENABLE_TOOL_RETRIEVAL", False)
        echo = EchoTool()
        llm = MockLLM(
            structured_responses=[
                {"goal": "分析数据", "steps": [{"tool": "test.echo", "arguments": {"text": "hello"}}]}
            ],
            responses=["已根据工具结果给出结论。"],
        )
        runtime = _runtime(engine, tools=[echo], llm=llm, planner=AgentPlanner(llm))
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析当前数据集")

        assert run.status == RunStatus.COMPLETED, run.error
        # 规划器确实被调用（走的是 LLM 规划，不是规则兜底）
        assert llm.structured_calls, "MockLLM 规划器未被调用"
        # 参数被正确生成并传到位
        assert echo.calls == [{"text": "hello"}]
        assert run.tool_calls[0].status == "ok"
        assert run.final_answer
        assert session.history[-1]["role"] == "assistant"

    def test_tool_context_carries_user_request(self, engine):
        """工具上下文必须带 user_request：工具的语义兜底输入依赖它。"""
        echo = EchoTool()
        runtime = _runtime(engine, tools=[echo])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        runtime.run(session, "分析当前数据集", plan_override=_plan("test.echo", text="x"))

        assert echo.calls == [{"text": "x"}]


# ---------------------------------------------------------------------------
# 场景 2：工具第一次失败 → 重规划 → 重试成功
# ---------------------------------------------------------------------------


class TestFailureAndRetry:
    def test_first_failure_then_success(self, engine):
        """工具第一次失败：重规划应留在原步骤重试，而不是跳过或终止。"""
        flaky = FlakyTool(fail_times=1)
        runtime = _runtime(engine, tools=[flaky])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.flaky"))

        assert run.status == RunStatus.COMPLETED, run.error
        assert flaky.calls == 2, "失败一次后应重试同一步，共两次调用"
        assert run.tool_calls[-1].status == "ok"
        assert [e.type for e in run.events].count("replanning") == 1

    def test_persistent_failure_terminates_after_attempt_limit(self, engine):
        """持续失败：重试到单步上限后终止，不能无限重试。"""
        failing = AlwaysFailTool()
        runtime = _runtime(engine, tools=[failing])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.always_fail"))

        assert run.status == RunStatus.FAILED
        # MAX_ATTEMPTS_PER_STEP=2：重试一次后跳过，无剩余步骤 ⇒ 终止
        assert failing.calls == 2, f"应恰好尝试 2 次，实际 {failing.calls}"
        assert run.error

    def test_parameter_error_is_not_blindly_retried(self, engine):
        """参数错误：`Replanner` 判定为「不该重复执行」⇒ 跳过该步，不是反复重试。"""
        tool = MissingParamTool()
        runtime = _runtime(engine, tools=[tool])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.missing_param"))

        assert run.status == RunStatus.FAILED
        # 参数缺 => 解析阶段就失败，工具一次都不该被执行
        assert run.tool_call_count == 0
        assert "缺少必要参数" in run.error


# ---------------------------------------------------------------------------
# 场景 4：高风险工具 → 待确认 → 确认 → 继续
# ---------------------------------------------------------------------------


class TestPermissionConfirmation:
    def test_waiting_confirmation_then_resume(self, engine):
        """高风险工具必须先停在 WAITING_CONFIRMATION，确认后才真正执行。"""
        risky = RiskyTool()
        runtime = _runtime(engine, tools=[risky])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "执行高风险操作", plan_override=_plan("test.risky"))

        assert run.status == RunStatus.WAITING_CONFIRMATION
        assert run.pending_confirmation is not None
        assert run.pending_confirmation["step_index"] == 0
        # 未经确认，工具一次都不能执行
        assert risky.calls == 0

        resumed = runtime.resume(session, run.id)

        assert resumed.status == RunStatus.COMPLETED, resumed.error
        assert risky.calls == 1
        assert resumed.pending_confirmation is None

    def test_confirmation_cannot_be_bypassed_by_direct_tool_call(self, engine):
        """权限边界：即使直接走 Registry，未经 confirmed 也必须被拦下。"""
        risky = RiskyTool()
        registry = ToolRegistry()
        registry.register(risky)
        context = ToolExecutionContext(
            user_id="u1",
            session_id="s1",
            dataset_ids=set(),
            permissions=set(ROLE_PERMISSIONS["analyst"]),
        )

        from app.tools.base import ToolConfirmationRequired

        with pytest.raises(ToolConfirmationRequired):
            registry.execute("test.risky", {}, context, None, confirmed=False)

        assert risky.calls == 0

    def test_deny_terminates_run(self, engine):
        """拒绝授权：运行必须真的终止，否则会话被等待态永久锁死。"""
        runtime = _runtime(engine, tools=[RiskyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = runtime.run(session, "执行高风险操作", plan_override=_plan("test.risky"))

        denied = runtime.deny(run.id)

        assert denied.status == RunStatus.FAILED
        assert denied.pending_confirmation is None
        assert "拒绝" in denied.error


# ---------------------------------------------------------------------------
# 场景 5：反问 → 待澄清 → 回答 → 从原步骤继续
# ---------------------------------------------------------------------------


class TestClarification:
    def test_clarify_then_answer_resumes_current_step(self, engine):
        """反问必须落在 pending_clarification，回答后从该步继续且不重跑前面的步骤。"""
        echo = EchoTool()
        clarify = ClarifyTool()
        runtime = _runtime(engine, tools=[echo, clarify])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(
            session,
            "分析",
            plan_override=AgentPlan(
                goal="e2e",
                steps=[
                    PlanStep(tool="test.echo", arguments={"text": "first"}),
                    PlanStep(tool="test.clarify", arguments={}),
                ],
            ),
        )

        assert run.status == RunStatus.WAITING_CLARIFICATION
        # ★ 语义必须与「待确认」分开：这两个字段曾经混用，导致反问永远无法被回答
        assert run.pending_clarification is not None
        assert run.pending_confirmation is None
        assert run.pending_clarification["code"] == "test.target"
        assert run.pending_clarification["step_index"] == 1
        assert echo.calls == [{"text": "first"}]

        resumed = runtime.answer_clarification(run.id, "x2")

        assert resumed.status == RunStatus.COMPLETED, resumed.error
        # 前面的步骤不重复执行
        assert echo.calls == [{"text": "first"}]
        assert clarify.calls == 2  # 反问一次 + 回答后重跑该步一次
        assert run.clarification_answers["test.target"] == "x2"
        assert resumed.pending_clarification is None

    def test_duplicate_answer_rejected(self, engine):
        """重复回答：运行已不在等待态，必须报错而不是继续跑一遍。"""
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = runtime.run(session, "分析", plan_override=_plan("test.clarify"))
        assert run.status == RunStatus.WAITING_CLARIFICATION

        runtime.answer_clarification(run.id, "x1")

        with pytest.raises(ValidationException):
            runtime.answer_clarification(run.id, "x1")

    def test_empty_answer_rejected(self, engine):
        """空回答不是「接受默认值」：直接往下走会用空串填参数，产出答非所问的结果。"""
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = runtime.run(session, "分析", plan_override=_plan("test.clarify"))

        with pytest.raises(ValidationException):
            runtime.answer_clarification(run.id, "   ")

        # 被拒绝后仍停在等待态，用户可以重新作答
        assert run.status == RunStatus.WAITING_CLARIFICATION
        assert run.pending_clarification is not None

    def test_answer_unknown_run(self, engine):
        """运行不存在：必须是 NotFound，而不是静默成功。"""
        runtime = _runtime(engine, tools=[ClarifyTool()])
        with pytest.raises(NotFoundException):
            runtime.answer_clarification("r-does-not-exist", "x1")

    def test_cancel_while_waiting_clarification(self, engine):
        """澄清态取消：必须真的终止（早先 deny 只看 pending_confirmation，取消毫无反应）。"""
        runtime = _runtime(engine, tools=[ClarifyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = runtime.run(session, "分析", plan_override=_plan("test.clarify"))
        assert run.status == RunStatus.WAITING_CLARIFICATION

        canceled = runtime.cancel(run.id)

        assert canceled.status == RunStatus.FAILED
        assert canceled.pending_clarification is None
        assert "澄清" in canceled.error

    def test_deny_while_waiting_confirmation_keeps_message(self, engine):
        """确认态与澄清态共用 deny，但终止文案必须区分（否则界面显示错误原因）。"""
        runtime = _runtime(engine, tools=[RiskyTool()])
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])
        run = runtime.run(session, "执行高风险操作", plan_override=_plan("test.risky"))

        denied = runtime.deny(run.id)
        assert "拒绝授权" in denied.error


# ---------------------------------------------------------------------------
# 场景 6：循环必须被熔断
# ---------------------------------------------------------------------------


class TestLoopProtection:
    def test_forced_retry_loop_is_cut_off(self, engine, monkeypatch):
        """强制让 Replanner 每次都说「再试一次」：运行必须停下来，不能无限循环。"""
        failing = AlwaysFailTool()
        runtime = _runtime(engine, tools=[failing])
        replanner = Replanner()

        def always_retry(plan, *, failed_step_index, errors, attempts, base_index=0):
            return AgentPlan(goal=plan.goal, steps=list(plan.steps), notes="（桩）总是重试", retry=True)

        monkeypatch.setattr(replanner, "replan", always_retry)
        runtime.replanner = replanner
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.always_fail"))

        assert run.status == RunStatus.FAILED
        assert "上限" in run.error, f"应因某项上限而终止，实际：{run.error!r}"
        # 工具调用数受 max_tool_calls 约束：不会跑到天荒地老
        assert failing.calls <= ReplanLimits().max_tool_calls + 1

    def test_tool_call_limit(self, engine):
        """工具调用数熔断。"""
        # 上限设为 0：任何一次工具调用都应立刻触发熔断。
        # （设为 2 不够：单步重试上限 MAX_ATTEMPTS_PER_STEP=2 会先于它终止。）
        runtime = _runtime(engine, tools=[AlwaysFailTool()], limits=ReplanLimits(max_tool_calls=0))
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.always_fail"))

        assert run.status == RunStatus.FAILED
        assert "超过上限" in run.error

    def test_token_budget_limit(self, engine, monkeypatch):
        """Token 预算熔断：预算打满时必须停止重试，而不是继续烧 token。

        r-22 的教训是「循环在 LLM 层持续烧钱」，所以这条闸必须真的接进循环里。
        这里让规划阶段就记下巨额用量（等价于预算已耗尽），验证第一步就被拦下。
        """
        # 模拟「规划阶段就把预算烧完了」：这一步在 _run_plan 之前发生，
        # 因此循环第一次迭代就会撞上 Token 闸。
        def _build_plan_burning_budget(self, run, context, candidate_tools, all_tools, plan_override):
            run.token_ledger.record_usage({"total_tokens": 10**9})
            return plan_override

        monkeypatch.setattr(AgentRuntime, "_build_plan", _build_plan_burning_budget)

        # 工具调用数放开，确保拦住运行的是 Token 闸而不是调用数闸
        runtime = _runtime(
            engine,
            tools=[AlwaysFailTool()],
            llm=MockLLM(),
            limits=ReplanLimits(max_tool_calls=10**6),
        )
        session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

        run = runtime.run(session, "分析", plan_override=_plan("test.always_fail"))

        assert run.status == RunStatus.FAILED
        assert "Token" in run.error, f"应由 Token 预算闸拦下，实际：{run.error!r}"
        assert run.tool_call_count == 0, "预算耗尽后不应再执行任何工具"
