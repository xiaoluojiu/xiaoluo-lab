"""钉住 Agent 重规划的**死循环防线**（2026-09-22 r-22 P0）。

背景（有真实数据为证，不是臆测）
--------------------------------
用户发「选用适合的机器学习模型训练并把结果给我」→ 计划第 5 步 `ml.train` 用了
`model="auto"`（未注册）→ 失败 → 后续 `ml.evaluate` / `ml.explain` 依赖
`{{step5.run_id}}` 拿不到 → 参数解析失败 → 重规划。此后**永不结束**：
r-3 留下 **199987 条 `replanning` 事件**（51MB），r-22 当场复现 773 条且仍在增长，
前端只能看见「正在工作」，后台每 700ms 轮询一次 `GET /agent/runs/{id}`。

三处叠加缺陷（缺一不可，所以修一处不够）
--------------------------------------
1. 参数解析失败分支**不写回 `attempts`** ⇒ `MAX_ATTEMPTS_PER_STEP` 永远看到 attempt=1。
2. 同一分支**无条件 `offset=abs_idx+1`**，即使 replanner 说的是「重试同一步」。
   步号一直 +1、计划原封不动 ⇒ 同一个失败步骤以**新下标**反复重试，1 的计数随之归零。
3. `assert_limits`（调用数/超时/token）**只在执行工具前调用** ⇒ 解析失败的迭代完全不过熔断。
4. 另：`_step_depends_on` 用的是**相对**下标，而 `{{stepN}}` 写的是**绝对**步号；
   `resume()` 会把计划切成 `steps[step_index:]` ⇒ 授权确认后依赖判定必然失配，
   「上游失败、下游等它」没被识别成依赖链断裂，反被当成瞬时错误。

本文件全部毫秒级：不启服务、不连数据库、不训练模型、不调 LLM。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.executor.executor import ToolCallRecord
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.replanner import AgentLimitExceeded, ReplanLimits, Replanner
from app.agent.runtime.models import AgentRun, AgentSession, RunStatus
from app.agent.runtime.runtime import MAX_EVENTS_PER_RUN, AgentRuntime
from app.core.exceptions import ValidationException
from app.tools.base import ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.ml_tools import MlTrainTool
from app.tools.result import ToolResult


# ---------------------------------------------------------------------------
# 一、依赖判定必须用**绝对**步号
# ---------------------------------------------------------------------------


def _plan_with_dependency() -> AgentPlan:
    """模拟 `resume()` 之后的切片：本切片从原计划的第 4 步（0 基）开始。

    后两步引用的是**绝对**步号 5（`{{step5.run_id}}`），即本切片第 0 步的输出。
    """
    return AgentPlan(
        goal="建模并评估",
        steps=[
            PlanStep(tool="ml.train", arguments={"model": "auto"}, expected_output="run_id"),
            PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step5.run_id}}"}, expected_output="指标"),
            PlanStep(tool="ml.explain", arguments={"run_id": "{{step5.run_id}}"}, expected_output="解释"),
        ],
    )


def test_dependency_detected_when_plan_is_sliced_by_resume():
    """授权确认后计划被切片 ⇒ 只有传 `base_index` 才能认出「下游等的是失败那一步」。"""
    replan = Replanner().replan(
        _plan_with_dependency(), failed_step_index=0, errors=["模型 'auto' 未注册"], attempts=2, base_index=4
    )
    assert replan.steps == [], "依赖链断裂应立即停止，而不是把下游步骤拿去重试"
    assert "依赖" in replan.notes


def test_dependency_missed_without_base_index_documents_the_old_bug():
    """同一份计划不传 base_index ⇒ 判定失配（这正是旧行为）。留着它，防止有人改回相对下标。"""
    replan = Replanner().replan(
        _plan_with_dependency(), failed_step_index=0, errors=["模型 'auto' 未注册"], attempts=1
    )
    assert replan.steps, "不传 base_index 时依赖判定失配 —— 这条用例把这个事实钉住"
    assert getattr(replan, "retry", False) is True


# ---------------------------------------------------------------------------
# 二、重规划次数总闸
# ---------------------------------------------------------------------------


def test_replan_budget_raises_at_limit():
    limits = ReplanLimits()
    replanner = Replanner(limits)
    replanner.assert_replan_budget(limits.max_replans - 1)  # 不抛
    with pytest.raises(AgentLimitExceeded):
        replanner.assert_replan_budget(limits.max_replans)


# ---------------------------------------------------------------------------
# 三、`_run_plan` 的行为（用桩对象驱动，不构造整个 AgentRuntime）
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str, required: list[str]) -> None:
        self.name = name
        self.input_schema = {
            "type": "object",
            "properties": {k: {"type": "string"} for k in required},
            "required": required,
        }
        self.output_schema = None


class _FakeRegistry:
    def __init__(self, spec: dict[str, _FakeTool]) -> None:
        self._spec = spec

    def get(self, name: str) -> _FakeTool:
        return self._spec[name]


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_step(self, step, tool_ctx, services, *, confirmed, attempt, step_index) -> ToolCallRecord:
        self.calls.append(step.tool)
        # status/result 是构造后赋值的字段（见 models.AgentStore._load 的复原方式），
        # 不在 __init__ 里 —— 这里照同一口径写。
        record = ToolCallRecord(step_index=step_index, tool=step.tool, arguments=dict(step.arguments), attempt=attempt)
        record.status = "ok"
        record.result = ToolResult.ok({"ok": True}, summary="成功")
        return record


class _FakeValidator:
    def validate(self, step, result, schema):
        return SimpleNamespace(valid=True, errors=[], warnings=[])


class _RuntimeStub(SimpleNamespace):
    """只补 `_run_plan` 真正用到的几个方法，避免构造完整 AgentRuntime（需数据库与引擎）。"""

    def _resolve_step_arguments(self, step, run, session, context):
        return AgentRuntime._resolve_step_arguments(self, step, run, session, context)

    def _emit(self, run, event_type, payload, on_event):
        AgentRuntime._emit(self, run, event_type, payload, on_event)

    def _emit_usage(self, run, on_event):
        AgentRuntime._emit_usage(self, run, on_event)

    def _fail(self, run, message, on_event):
        AgentRuntime._fail(self, run, message, on_event)

    def _complete(self, run, session, on_event):
        AgentRuntime._complete(self, run, session, on_event)

    # 签名必须与 AgentRuntime._tool_context 保持一致（现多了一个关键字参数
    # user_request，用于把用户诉求注入工具上下文，见 target_inference）。
    def _tool_context(self, session, role, *, user_request=""):
        return self.ctx

    def _services(self):
        return self.services

    def _tool_output_schema(self, tool_name):
        return None

    def _result_for_llm(self, run, record, *, compact=False):
        return AgentRuntime._result_for_llm(self, run, record, compact=compact)

    def _compose_answer(self, run):
        return AgentRuntime._compose_answer(self, run)


def _stub(*, registry: _FakeRegistry, replanner: Replanner | None = None) -> _RuntimeStub:
    return _RuntimeStub(
        llm=None,
        registry=registry,
        replanner=replanner or Replanner(),
        executor=_FakeExecutor(),
        validator=_FakeValidator(),
        services=ToolServices(dataset_service=None, data_engine_service=None, experiment_service=None, db=None),
        ctx=SimpleNamespace(dataset_ids=lambda: [1]),
    )


def _new_run() -> AgentRun:
    run = AgentRun(id="r-t", session_id="s-t", user_id="u-t", user_request="选用适合的机器学习模型训练并把结果给我")
    run.started_at = run.created_at
    return run


def test_run_plan_stops_on_broken_dependency_chain():
    """r-22 的真实形状：`ml.train` 失败后，下游拿不到 `{{step5.run_id}}`。

    期望：**一步就停**，并且错误里说清是依赖链断裂 —— 而不是重试到天荒地老。
    """
    registry = _FakeRegistry({
        "ml.train": _FakeTool("ml.train", []),
        "ml.evaluate": _FakeTool("ml.evaluate", ["run_id"]),
        "ml.explain": _FakeTool("ml.explain", ["run_id"]),
    })
    stub = _stub(registry=registry)
    run, session = _new_run(), AgentSession(id="s-t", user_id="u-t")

    # 模拟 resume()：计划已被切成 steps[4:]，offset=4
    plan = AgentPlan(goal="建模并评估", steps=[
        PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step5.run_id}}"}),
        PlanStep(tool="ml.explain", arguments={"run_id": "{{step5.run_id}}"}),
    ])
    AgentRuntime._run_plan(stub, run, session, SimpleNamespace(), "analyst", plan,
                           offset=4, attempts={4: 1}, confirmed=True, on_event=None)

    assert run.status == RunStatus.FAILED
    assert "依赖第 5 步输出" in run.error
    assert [e.type for e in run.events].count("replanning") == 1, "依赖链断裂应一次判死"


def test_run_plan_cannot_loop_forever_even_if_replanner_always_retries(monkeypatch):
    """**结构性防线**：就算 replanner 每次都说「再试一次」，`_run_plan` 也必须停下来。

    这条不依赖任何语义判断，纯粹靠重规划计数 —— 只要将来有人又把记账写漏，
    它仍然会把运行掐断（而不是像 r-3 那样攒出 199987 条事件）。
    """
    registry = _FakeRegistry({"ml.evaluate": _FakeTool("ml.evaluate", ["run_id"])})
    replanner = Replanner()

    def _always_retry(plan, *, failed_step_index, errors, attempts, base_index=0):
        return AgentPlan(goal=plan.goal, steps=list(plan.steps), notes="（桩）总是重试", retry=True)

    monkeypatch.setattr(replanner, "replan", _always_retry)
    stub = _stub(registry=registry, replanner=replanner)
    run, session = _new_run(), AgentSession(id="s-t", user_id="u-t")

    plan = AgentPlan(goal="x", steps=[PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step9.run_id}}"})])
    AgentRuntime._run_plan(stub, run, session, SimpleNamespace(), "analyst", plan,
                           offset=0, attempts={}, confirmed=True, on_event=None)

    n_replan = [e.type for e in run.events].count("replanning")
    assert run.status == RunStatus.FAILED
    assert "上限" in run.error, f"应因重规划次数上限而终止，实际：{run.error!r}"
    assert n_replan == ReplanLimits().max_replans, f"重规划次数应恰好被截断在上限，实际 {n_replan}"


def test_emit_has_a_hard_cap_against_runaway_loops():
    """最后一道护栏：不管循环从哪来，事件数到顶就必须炸，不能再写进 store。

    r-3 一次死循环写了 199987 条事件（51MB）；终态事件必须放行，否则失败原因发不出去。
    """
    stub = _stub(registry=_FakeRegistry({}))
    run = _new_run()
    for _ in range(MAX_EVENTS_PER_RUN):
        AgentRuntime._emit(stub, run, "replanning", {"n": 1}, None)
    assert len(run.events) == MAX_EVENTS_PER_RUN
    with pytest.raises(AgentLimitExceeded):
        AgentRuntime._emit(stub, run, "replanning", {"n": 2}, None)
    AgentRuntime._emit(stub, run, "failed", {"error": "x"}, None)  # 终态放行
    assert run.events[-1].type == "failed"


def test_resolve_failure_is_counted_as_an_attempt():
    """尝试数必须写回：同一步解析失败两次后应**跳过**，而不是无限重试。"""
    registry = _FakeRegistry({
        "ml.evaluate": _FakeTool("ml.evaluate", ["run_id"]),
        "report.generate": _FakeTool("report.generate", ["title"]),
    })
    stub = _stub(registry=registry)
    run, session = _new_run(), AgentSession(id="s-t", user_id="u-t")

    # 第 0 步永远解析失败（依赖第 9 步，且错误不含「参数」类标记 ⇒ 会走重试分支）；
    # 第 1 步是合法步骤，应当在两次尝试后被跳过并执行到。
    plan = AgentPlan(goal="x", steps=[
        PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step9.run_id}}"}),
        PlanStep(tool="report.generate", arguments={"title": "报告"}),
    ])
    AgentRuntime._run_plan(stub, run, session, SimpleNamespace(), "analyst", plan,
                           offset=0, attempts={}, confirmed=True, on_event=None)

    n_replan = [e.type for e in run.events].count("replanning")
    assert n_replan <= 2, f"同一失败步骤最多重试一次，实际重规划 {n_replan} 次"
    assert run.status == RunStatus.COMPLETED, f"跳过失败步骤后应继续跑完剩余步骤：{run.error}"
    assert stub.executor.calls == ["report.generate"]


# ---------------------------------------------------------------------------
# 四、触发点：`ml.train` 必须接受 model="auto"
# ---------------------------------------------------------------------------


class _FakeExperimentService:
    def __init__(self) -> None:
        self.created: dict | None = None

    def create(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id=1, model=kwargs.get("model"))

    def run(self, experiment_id):
        return SimpleNamespace(id=2, status="success", metrics={"r2": 0.9}, error=None)


def test_ml_train_accepts_auto_model(monkeypatch):
    """「选用适合的机器学习模型」落到计划里就是 model="auto"。

    不接受它 ⇒ 每次这类请求的第一步必然失败（'模型 auto 未注册'），
    而失败又会触发重规划（在修复前即死循环）。
    """
    monkeypatch.setattr(
        "app.tools.ml_tools.MlDetectTaskTool.execute",
        lambda self, params, context, services: ToolResult.ok({"task": "classification", "target": "y"}),
    )
    service = _FakeExperimentService()
    tool = MlTrainTool()
    result = tool.execute(
        {"dataset_id": 5, "model": "auto", "target": "y"},
        ToolExecutionContext(user_id="u", session_id="s", dataset_ids={5}, permissions=set()),
        ToolServices(
            dataset_service=SimpleNamespace(get_version_row=lambda dataset_id, version: SimpleNamespace(id=11, version=1)),
            data_engine_service=None,
            experiment_service=service,
            db=None,
        ),
    )
    assert result.success, result.errors
    assert service.created is not None and service.created["model"] == "logistic_regression"
    assert result.data.get("model_adjusted", {}).get("from") == "auto"


def test_ml_train_auto_model_reports_unknown_task(monkeypatch):
    """任务类型拿不到默认模型时，明确失败而不是带着 'auto' 往下走。"""
    monkeypatch.setattr(
        "app.tools.ml_tools.MlDetectTaskTool.execute",
        lambda self, params, context, services: ToolResult.ok({"task": "timeseries", "target": "y"}),
    )
    tool = MlTrainTool()
    result = tool.execute(
        {"dataset_id": 5, "model": "auto", "target": "y"},
        ToolExecutionContext(user_id="u", session_id="s", dataset_ids={5}, permissions=set()),
        ToolServices(
            dataset_service=SimpleNamespace(get_version_row=lambda dataset_id, version: SimpleNamespace(id=11, version=1)),
            data_engine_service=None,
            experiment_service=_FakeExperimentService(),
            db=None,
        ),
    )
    assert not result.success
    assert "auto" in " ".join(result.errors) or "默认模型" in " ".join(result.errors)
