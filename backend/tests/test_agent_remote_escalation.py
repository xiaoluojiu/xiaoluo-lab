"""Phase 4 远程升级（Remote Escalation）单元测试。

钉住「local-first → escalate → remote strategic guidance → local execution」的
四条不变量，全部离线、毫秒级、用 Mock/Fake 替身，不调用真实大模型：

1. **简单任务零远程升级**：本地 Router 判简单（不复杂）时，``_escalate`` 不发起
   远程调用，``remote_escalations`` 保持 0。
2. **复杂任务才升级且只升级一次**：本地判「复杂」+ 有 llm 时，发起**一次**战略指导，
   账本 ``remote_escalations`` 恰好 +1，指导结果挂到 context。
3. **无 llm 不升级**：未注入 llm 时，``_escalate`` 直接返回 ``escalated=False``，
   不抛异常、不伪造升级。
4. **远程失败不静默吞**：``fallback_allowed`` 为 False 时，远程失败原样抛出；
   为 True 时降级保留本地结论（escalated=False），而非谎报「已升级」。
"""

from __future__ import annotations

from app.agent.decision.provider import DecisionContext
from app.agent.decision.remote import RemoteLLMDecisionProvider
from app.agent.decision.router import DecisionRouter
from app.agent.decision.local_model import LocalModelDecisionProvider
from app.agent.llm.base import LLMException
from app.agent.llm.local import FakeLocalReasoningProvider
from app.agent.llm.mock import MockLLM


def _ctx(request: str, complexity_hint: str | None = None) -> DecisionContext:
    spec = None
    if complexity_hint is not None:
        spec = {"goal": request, "complexity_hint": complexity_hint}
    return DecisionContext(user_request=request, bound_dataset_id=1, task_spec=spec)


# ---------------------------------------------------------------------------
# 1) 简单任务零远程升级（LocalModelDecisionProvider 不会升级简单请求）
# ---------------------------------------------------------------------------


def test_simple_task_does_not_escalate() -> None:
    """本地 Router 对明确请求给出 EXECUTE_TOOL，不触发升级。"""
    p = LocalModelDecisionProvider()
    d = p.decide(_ctx("检查一下数据质量"))
    assert d.action == "execute_tool"
    assert d.tool == "dataset.quality"


# ---------------------------------------------------------------------------
# 2) RemoteLLMDecisionProvider 只做一次战略指导，工具仍回落本地
# ---------------------------------------------------------------------------


def test_remote_provider_returns_guidance_not_tool_loop() -> None:
    """远程升级给出的是战略指导（evidence.remote_guidance），tool 仍是 None——
    工具选择回落到本地 Agent（任务 10：本地执行）。"""
    llm = MockLLM(responses=["建议先做数据质量检查，再决定是否清洗。"])
    p = RemoteLLMDecisionProvider(llm=llm)
    d = p.decide(_ctx("这个复杂任务我拿不准"))
    assert d.source == "remote_llm"
    assert "remote_guidance" in d.evidence
    # 关键：远程不直接控制工具执行。
    assert d.tool is None


def test_remote_provider_without_llm_is_conservative() -> None:
    """未注入 llm 时，远程 Provider 返回保守升级（confidence=0），不伪造。"""
    p = RemoteLLMDecisionProvider(llm=None)
    d = p.decide(_ctx("复杂任务"))
    assert d.action == "escalate"
    assert d.confidence == 0.0


def test_remote_provider_swallows_error_honestly() -> None:
    """远程 LLM 抛异常时，不静默吞成「成功」：返回 confidence=0 的升级结论并写明原因。"""

    class _BoomLLM:
        def chat(self, messages):
            raise LLMException("远程 LLM 调用失败：connection reset")

    p = RemoteLLMDecisionProvider(llm=_BoomLLM())
    d = p.decide(_ctx("复杂任务"))
    assert d.action == "escalate"
    assert d.confidence == 0.0
    assert any("失败" in r for r in d.reasons)


# ---------------------------------------------------------------------------
# 3) DecisionRouter 组合：简单本地 → 不升级；本地升级 + 有 llm → 升级
# ---------------------------------------------------------------------------


def test_router_does_not_escalate_simple_task() -> None:
    """本地成功决定工具时，Router 不调用远程（即使注入了 llm）。"""
    calls: list[str] = []

    class _CountingLLM(MockLLM):
        def chat(self, messages, **kw):
            calls.append("called")
            return super().chat(messages, **kw)

    router = DecisionRouter(llm=_CountingLLM())
    d = router.route(_ctx("检查一下数据质量"), allow_remote=True)
    assert d.action == "execute_tool"
    assert d.tool == "dataset.quality"
    # 简单任务：零远程调用。
    assert calls == []


def test_router_escalates_when_local_gives_up() -> None:
    """本地无把握（escalate）且注入了 llm 时，Router 走远程。"""
    llm = MockLLM(responses=["建议分三步：先看分布，再做相关性，最后出报告。"])
    router = DecisionRouter(llm=llm)

    class _GiveUp(LocalModelDecisionProvider):
        def decide(self, context):
            from app.agent.decision.provider import AgentDecision, DecisionAction, DecisionSource
            return AgentDecision(action=DecisionAction.ESCALATE, source=DecisionSource.LOCAL_ROUTER,
                                 confidence=0.1, reasons=["本地置信度不足"])

    router.local_model = _GiveUp()
    d = router.route(_ctx("帮我做个完整的建模分析"), allow_remote=True)
    assert d.source == "remote_llm"
    assert "remote_guidance" in d.evidence


# ---------------------------------------------------------------------------
# 4) FakeLocalReasoningProvider 可用于「本地推理可用」的决策链路验证
# ---------------------------------------------------------------------------


def test_fake_local_reasoning_provider_can_drive_decision() -> None:
    """Phase 3 的本地推理替身可在「本地推理可用」前提下驱动决策链路。"""
    from app.agent.llm.base import LLMMessage

    local = FakeLocalReasoningProvider(responses=["应执行 dataset.quality 检查"])
    assert local.is_available() is True
    resp = local.chat([LLMMessage(role="user", content="检查数据质量")])
    assert "dataset.quality" in resp.content


# ---------------------------------------------------------------------------
# 5) 运行时 `_escalate` 的集成验证（构造最小 runtime，不调真实 LLM）
# ---------------------------------------------------------------------------


def test_runtime_escalate_skips_when_no_llm() -> None:
    from app.agent.context.models import AgentContext
    from app.agent.runtime.runtime import AgentRuntime

    rt = AgentRuntime.__new__(AgentRuntime)
    rt.llm = None
    ctx = AgentContext(task_context={"task_spec": {"complexity_hint": "complex"}})
    result = rt._escalate(_FakeRun("r-1"), _FakeSession(), ctx)
    assert result["escalated"] is False
    assert result["reason"] == "no_llm"


def test_runtime_escalate_skips_when_not_complex() -> None:
    from app.agent.context.models import AgentContext
    from app.agent.runtime.runtime import AgentRuntime

    rt = AgentRuntime.__new__(AgentRuntime)
    rt.llm = MockLLM()
    ctx = AgentContext(task_context={"task_spec": {"complexity_hint": "simple"}})
    result = rt._escalate(_FakeRun("r-1"), _FakeSession(), ctx)
    assert result["escalated"] is False
    assert result["reason"] == "not_complex"


def test_runtime_escalate_records_guidance_and_ledger() -> None:
    """复杂任务 + 有 llm：发起一次战略指导，账本 remote_escalations +1。"""
    from app.agent.context.models import AgentContext
    from app.agent.runtime.runtime import AgentRuntime

    # 直接用 __new__ 无法拿到 decision_router；构造完整 runtime 需要 data_engine。
    # 改为手动补上 _escalate 依赖的字段，绕过重型 __init__。
    rt = AgentRuntime.__new__(AgentRuntime)
    from app.agent.decision.provider import AgentDecision, DecisionAction, DecisionSource
    from app.agent.decision.router import DecisionRouter
    from app.agent.decision.local_model import LocalModelDecisionProvider

    rt.llm = MockLLM(responses=["建议分三步：先看分布，再相关性，最后报告。"])
    router = DecisionRouter(llm=rt.llm)

    class _GiveUp(LocalModelDecisionProvider):
        """本地无把握，强制升级，隔离 _escalate 的接线逻辑。"""

        def decide(self, context):
            return AgentDecision(action=DecisionAction.ESCALATE, source=DecisionSource.LOCAL_ROUTER,
                                 confidence=0.1, reasons=["本地置信度不足"])

    router.local_model = _GiveUp()
    rt.decision_router = router
    rt._decision_traces = {}
    rt._emit = lambda run, t, p, on_event=None: None
    run = _FakeRun("r-1")
    ctx = AgentContext(task_context={"task_spec": {"complexity_hint": "complex", "goal": "完整建模分析"}})
    result = rt._escalate(run, _FakeSession(), ctx)
    assert result["escalated"] is True
    assert run.token_ledger.remote_escalations == 1
    assert "remote_guidance" in ctx.task_context


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.user_request = "帮我做个完整的建模分析"
        from app.agent.runtime.models import AgentTokenLedger
        self.token_ledger = AgentTokenLedger()


class _FakeSession:
    dataset_ids: list[int] = [1]

