"""高风险工具的确认闭环。

这里钉的是两个真实发生过的缺陷，它们的**表现完全不同、成因却在同一处**：

1. ``ToolRegistry.execute`` 先判 ``decision.denied`` 再判 ``needs_confirmation``。
   而 NEEDS_CONFIRMATION 的 ``allowed`` 也是 False，于是高风险工具被当成
   「权限不足」抛出 ``ToolPermissionError`` —— 用户点完「允许」之后，
   仍以「需要你确认后才会执行」失败一次。
2. ``Tool.describe`` 原样下发类属性 ``requires_confirmation``，而几乎没有工具
   显式声明过它。``ml.train`` 明明写着 ``risk_level = "high"``，对外却自称
   「低风险、可自动放行」—— 前端据此默认自动放行，用户**连确认弹窗都见不到**。

两者叠加的最终观感就是：高风险操作既不给确认机会，报的错又永远是同一句。
"""

from __future__ import annotations

import pytest

from app.agent.engine import AgentEngine
from app.agent.models import EventType, RunStatus, ToolCallStatus
from app.agent.store import AgentStore
from app.agent.permission import PermissionManager, Permission, risk_needs_confirmation
from app.tools.base import Tool, ToolConfirmationRequired, ToolPermissionError
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="tester",
        session_id="s-1",
        dataset_ids={1},
        permissions=set(Permission),
    )


# ---------------------------------------------------------------- 裁决三态


def test_confirmation_is_not_denied():
    """NEEDS_CONFIRMATION 的 allowed 也是 False，但它绝不是「权限不足」。

    把它算进 denied 的后果是：确认之后仍以同一句话失败。
    """
    manager = PermissionManager()
    tool = type("T", (), {"name": "ml.train", "permission": "train_model", "risk_level": "high"})()
    decision = manager.check("u", tool, _context(), {})

    assert decision.needs_confirmation is True
    assert decision.allowed is False
    assert decision.denied is False  # ← 缺陷 1 就出在这一行上
    assert decision.as_label() == "REQUIRE_CONFIRMATION"


def test_missing_permission_is_denied_even_for_high_risk():
    """权限缺失优先于确认：没有权限时让用户确认也没有意义。"""
    manager = PermissionManager()
    tool = type("T", (), {"name": "ml.train", "permission": "train_model", "risk_level": "high"})()
    ctx = _context()
    ctx.permissions = {Permission.READ_DATA}

    decision = manager.check("u", tool, ctx, {})
    assert decision.denied is True
    assert decision.needs_confirmation is False


def test_risk_needs_confirmation_helper_matches_check():
    """自描述用的判定必须与裁决器一致，否则前端看到的工具和后端执行的是两个工具。"""
    assert risk_needs_confirmation("high") is True
    assert risk_needs_confirmation("critical") is True
    assert risk_needs_confirmation("medium") is False
    assert risk_needs_confirmation("low") is False


# ---------------------------------------------------------------- 注册表闸门


class _HighRiskTool(Tool):
    name = "fake.high_risk"
    description = "高风险测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}}
    output_schema = {"type": "object"}
    permission = "modify_data"
    risk_level = "high"
    calls: int = 0

    def execute(self, params, context, services) -> ToolResult:
        type(self).calls += 1
        return ToolResult.ok({"done": True}, "已执行")


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    _HighRiskTool.calls = 0
    reg.register(_HighRiskTool())
    return reg


def test_unconfirmed_high_risk_raises_confirmation_not_permission_error(registry: ToolRegistry):
    """缺陷 1 的正面回归：必须是「要确认」，不能是「权限不足」。"""
    with pytest.raises(ToolConfirmationRequired):
        registry.execute("fake.high_risk", {}, _context())


def test_confirmed_high_risk_actually_executes(registry: ToolRegistry):
    """用户点过「允许」之后必须真的跑起来，而不是再报一次「需要确认」。"""
    result = registry.execute("fake.high_risk", {}, _context(), confirmed=True)

    assert result.success is True
    assert _HighRiskTool.calls == 1


def test_missing_permission_raises_permission_error(registry: ToolRegistry):
    ctx = _context()
    ctx.permissions = {Permission.READ_DATA}
    with pytest.raises(ToolPermissionError):
        registry.execute("fake.high_risk", {}, ctx, confirmed=True)


def test_high_risk_self_description_reports_confirmation(registry: ToolRegistry):
    """缺陷 2 的正面回归：自描述必须如实说「我是高风险」。"""
    meta = registry.get("fake.high_risk").describe()
    assert meta["risk_level"] == "high"
    assert meta["requires_confirmation"] is True


# ---------------------------------------------------------------- 真实注册表


def test_every_high_risk_registered_tool_declares_confirmation():
    """工具目录对外不能出现「risk_level=high 却自称无需确认」的工具。

    前端设置页按 requires_confirmation 区分高风险/低风险，并决定默认是否自动
    放行；这里漏一个，用户就少一次确认机会。
    """
    import app.tools.builtin  # noqa: F401  触发注册
    from app.tools.registry import TOOL_REGISTRY

    wrong: list[str] = []
    for meta in TOOL_REGISTRY.list():
        tool = TOOL_REGISTRY.get(meta["name"])
        if risk_needs_confirmation(tool.risk_level) and meta["requires_confirmation"] is not True:
            wrong.append(f"{meta['name']}(risk={meta['risk_level']})")
    assert not wrong, f"这些高风险工具对外自称无需确认：{wrong}"


def test_tools_endpoint_advertises_confirmation(client):
    """前端只读这一个接口来区分高风险/低风险，这里错了前端就跟着错。"""
    resp = client.get("/api/v1/agent/tools")
    assert resp.status_code == 200, resp.text

    tools = resp.json()["data"]
    train = next(t for t in tools if t["name"] == "ml.train")
    assert train["risk_level"] == "high"
    assert train["requires_confirmation"] is True


def test_ml_train_is_advertised_as_high_risk():
    """用户报的就是 ml.train，单独钉住它。"""
    import app.tools.builtin  # noqa: F401
    from app.tools.registry import TOOL_REGISTRY

    meta = next(m for m in TOOL_REGISTRY.list() if m["name"] == "ml.train")
    assert meta["risk_level"] == "high"
    assert meta["requires_confirmation"] is True


# ---------------------------------------------------------------- 端到端闭环


class _FakeDetect(Tool):
    name = "ml.detect_task"
    description = "假任务识别"
    category = "ml"
    input_schema = {"type": "object", "properties": {"dataset_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"target": "y", "task": "classification"}, "识别为分类任务")


class _FakeTrain(Tool):
    name = "ml.train"
    description = "假训练"
    category = "ml"
    input_schema = {"type": "object", "properties": {"dataset_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "train_model"
    risk_level = "high"

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"run_id": 1}, "训练完成")


class _FakeEvaluate(Tool):
    name = "ml.evaluate"
    description = "假评估"
    category = "ml"
    input_schema = {"type": "object", "properties": {"run_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "analyze_data"

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"accuracy": 0.9}, "准确率 0.90")


@pytest.fixture()
def ml_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (_FakeDetect(), _FakeTrain(), _FakeEvaluate()):
        reg.register(tool)
    return reg


def test_ml_train_suspends_then_completes_after_confirmation(
    store: AgentStore, ml_registry: ToolRegistry, bound_session
):
    """用户报的那句话：设计机器学习实验方案 → 挂起等确认 → 确认 → 真跑完。"""
    run = store.create_run(
        bound_session.id,
        "请根据当前数据集设计一个机器学习实验方案，说明目标变量、特征、候选模型、评价指标和下一步执行建议。",
    )
    AgentEngine(store, tools=ml_registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.pending_confirmation.tool == "ml.train"
    assert run.pending_confirmation.step_index == 1
    assert run.error == ""
    assert EventType.PERMISSION.value in [e.type.value for e in run.events]
    # 确认之前一步都不许跑
    assert [c.tool for c in run.tool_calls] == ["ml.detect_task"]

    # ---- 用户点「允许」----
    run.authorized_key = f"{run.pending_confirmation.step_index}:{run.pending_confirmation.tool}"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    AgentEngine(store, tools=ml_registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    assert "需要你确认" not in (run.error or "")
    assert [c.tool for c in run.tool_calls] == ["ml.detect_task", "ml.train", "ml.evaluate"]
    assert all(c.status is ToolCallStatus.OK for c in run.tool_calls)
    assert run.final_answer


def test_denied_confirmation_never_reports_permission_reason(
    store: AgentStore, ml_registry: ToolRegistry, bound_session
):
    """纵深防御：就算注册表仍要求确认，也回到挂起，不许变成「执行失败」。"""
    run = store.create_run(bound_session.id, "训练一个预测模型")
    AgentEngine(store, tools=ml_registry, llm=None).run(run, bound_session)
    assert run.status is RunStatus.WAITING_CONFIRMATION

    # 给一个对不上的凭据：引擎必须重新挂起，而不是把确认理由当失败抛给用户
    run.authorized_key = "9:other.tool"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    AgentEngine(store, tools=ml_registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.error == ""
    assert [c.tool for c in run.tool_calls] == ["ml.detect_task"]


# ---------------------------------------------------------------- 首步即确认


class _FakeClean(Tool):
    name = "data.clean"
    description = "假清洗"
    category = "data"
    input_schema = {"type": "object", "properties": {"dataset_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "modify_data"
    risk_level = "high"

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"rows": 10}, "已清洗")


@pytest.fixture()
def clean_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_FakeClean())
    return reg


def test_resume_from_step_zero_does_not_replay_route_and_plan(
    store: AgentStore, clean_registry: ToolRegistry, bound_session
):
    """首步就要确认时，恢复后不能再发一遍 route / planning。

    ``resume_step`` 在首步挂起时等于 0，而引擎原先用 ``resume_step == 0``
    判断「是不是新运行」—— 于是确认之后时间线上多出两条一模一样的计划，
    ``started_at`` 也被重置，耗时统计丢掉了用户思考的那几分钟。
    """
    run = store.create_run(bound_session.id, "清洗一下数据")
    AgentEngine(store, tools=clean_registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.pending_confirmation.step_index == 0
    first_started_at = run.started_at

    run.authorized_key = f"0:{run.pending_confirmation.tool}"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    AgentEngine(store, tools=clean_registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    counts = {t: [e.type.value for e in run.events].count(t)
              for t in (EventType.ROUTE.value, EventType.PLANNING.value)}
    assert counts == {EventType.ROUTE.value: 1, EventType.PLANNING.value: 1}, counts
    # 计时不能被恢复重置：否则「等用户确认」的那段凭空消失
    assert run.started_at == first_started_at


# ---------------------------------------------------------------- 挂起态取消


def _suspend(run, *, tool: str = "data.clean", step: int = 0, clarify: bool = False) -> None:
    """把一条 run 摆成挂起态（等待确认 / 等待补充信息）。"""
    from app.agent.models import ClarificationRequest, PermissionRequest, RunStatus

    if clarify:
        run.status = RunStatus.WAITING_CLARIFICATION
        run.pending_clarification = ClarificationRequest(question="分析哪一列？", code="slot.column", tool=tool)
    else:
        run.status = RunStatus.WAITING_CONFIRMATION
        run.pending_confirmation = PermissionRequest(
            tool=tool, arguments={"dataset_id": 1}, step_index=step, reason="会改写数据"
        )
    run.resume_step = step


def test_cancel_while_waiting_confirmation_closes_the_run(client, api_store: AgentStore):
    """挂起态点「停止」必须当场收成终态。

    引擎线程在挂起时已经返回，没人会再去读 ``cancel_requested`` —— 只打标记
    的结果是界面永远停在「等待确认」：既不能继续，也不能结束，只能刷新页面。
    """
    sid = api_store.create_session(user_id="tester", title="取消用例").id
    run = api_store.create_run(sid, "清洗一下数据")
    _suspend(run)
    api_store.update_run(run)

    resp = client.post(f"/api/v1/agent/runs/{run.id}/cancel")
    assert resp.status_code == 200, resp.text

    data = resp.json()["data"]
    assert data["status"] == "completed"
    assert data["pending_confirmation"] is None
    assert "已停止" in (data["final_answer"] or "")
    assert "data.clean" in (data["final_answer"] or "")
    # 最终话术必须进事件流：等待期间 SSE 还开着，那是它进聊天区的唯一出口
    completed = [e for e in data["events"] if e["type"] == "completed"]
    assert completed and completed[-1]["payload"]["final_answer"] == data["final_answer"]
    # 取消不是失败：不该挂 error，也不该留下「等待确认」的尾巴
    assert data["error"] == ""
    assert api_store.get_run(run.id).status.value == "completed"


def test_cancel_while_waiting_clarification_closes_the_run(client, api_store: AgentStore):
    """等待补充信息时点「停止」同样要收口，否则澄清面板会一直堵着输入框。"""
    sid = api_store.create_session(user_id="tester", title="取消用例2").id
    run = api_store.create_run(sid, "分析数据的分布")
    _suspend(run, clarify=True)
    api_store.update_run(run)

    resp = client.post(f"/api/v1/agent/runs/{run.id}/cancel")
    assert resp.status_code == 200, resp.text

    data = resp.json()["data"]
    assert data["status"] == "completed"
    assert data["pending_clarification"] is None
    assert "已停止" in (data["final_answer"] or "")


def test_cancel_of_running_run_only_sets_flag(client, api_store: AgentStore):
    """运行中的取消仍然只打标记：强杀正在跑的工具会留下半截状态。"""
    sid = api_store.create_session(user_id="tester", title="取消用例3").id
    run = api_store.create_run(sid, "训练一个模型")
    run.status = RunStatus.RUNNING
    api_store.update_run(run)

    resp = client.post(f"/api/v1/agent/runs/{run.id}/cancel")
    assert resp.status_code == 200, resp.text

    data = resp.json()["data"]
    assert data["cancel_requested"] is True
    assert data["status"] == "running"


def test_cancel_is_idempotent_for_finished_run(client, api_store: AgentStore):
    """已结束的运行再点停止，不能把 final_answer 改掉。"""
    sid = api_store.create_session(user_id="tester", title="取消用例4").id
    run = api_store.create_run(sid, "看一下数据")
    run.status = RunStatus.COMPLETED
    run.final_answer = "原始答案"
    api_store.update_run(run)

    resp = client.post(f"/api/v1/agent/runs/{run.id}/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["final_answer"] == "原始答案"
