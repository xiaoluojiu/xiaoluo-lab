"""Agent 等待态（确认 / 澄清）的**原子性**与持久化的**对称性**。

两条都是「并发 / 重启」才暴露的问题，靠单线程 happy path 永远测不出来：

1. 原子性：两个 confirm 请求同时到达时，只能有一个领到授权、只执行一次高风险工具。
   前端的 `confirming` 标记不算数 —— 它只在单个标签页内有效，挡不住第二个标签、
   网络重放或客户端重试。后端必须在锁内完成「检查 + 清除 + 改状态」。

2. 对称性：``persist()`` 写出去的字段，``_load()`` 必须能原样还原。
   漏一个字段的表现是「重启后界面显示与重启前不一致」，而不是报错，
   因此只能靠 persist → reload → 断言相等来钉住。

本文件不启服务、不打 LLM：并发用线程 + barrier 制造真实的竞态窗口。
"""

from __future__ import annotations

import threading

import polars as pl
import pytest
from app.agent.answer_source import PLATFORM_RULES_SUMMARY
from app.agent.runtime.models import AgentRun, AgentSession, AgentStore, RunStatus
from app.core.exceptions import ValidationException
from app.data_engine.service import DataEngineService
from app.services.dataset_service import DatasetService
from app.tools.base import Tool
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

# Permission 只用于构造工具对象；风险等级来自 RiskLevel
from app.agent.permission.models import Permission
from app.agent.permission.rules import RiskLevel
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.runtime.runtime import AgentRuntime


class RiskyTool(Tool):
    """高风险工具：记录每次真实执行。"""

    name = "wait.risky"
    description = "高风险的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}, "required": []}
    permission = Permission.TRAIN_MODEL
    risk_level = RiskLevel.HIGH

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, params, context, services) -> ToolResult:
        self.calls += 1
        return ToolResult.ok({"done": True}, summary="高风险操作已执行")


class ClarifyTool(Tool):
    """计划执行中发起结构化反问，回答一次后不再问。"""

    name = "wait.clarify"
    description = "需要用户补充信息的测试工具"
    category = "test"
    input_schema = {"type": "object", "properties": {}, "required": []}
    permission = Permission.ANALYZE_DATA
    risk_level = RiskLevel.LOW

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def execute(self, params, context, services) -> ToolResult:
        with self._lock:
            self.calls += 1
        extra = getattr(context, "extra", {}) or {}
        answers = {}
        for value in extra.values():
            if isinstance(value, dict):
                answers.update(value)
        picked = answers.get("wait.target")
        if picked:
            return ToolResult.ok({"target": picked}, summary="已按回答继续")
        from app.agent.clarify import Clarification, ClarificationOption, ClarificationRequired

        raise ClarificationRequired(
            Clarification(
                code="wait.target",
                question="请问要处理哪一列？",
                options=[ClarificationOption(value="a", label="a")],
                default="a",
            )
        )


@pytest.fixture()
def engine(db, storage, tmp_path):
    ds = DatasetService(db, storage)
    dataset = ds.create("wait-states", "等待态测试数据")
    ds.create_version(
        dataset.id,
        pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    )
    return {"ds": ds, "engine": DataEngineService(ds), "dataset_id": dataset.id}


def _runtime(engine, tools, *, store: AgentStore | None = None) -> AgentRuntime:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRuntime(engine["engine"], llm=None, registry=registry, store=store or AgentStore())


def _plan(*tools: str) -> AgentPlan:
    return AgentPlan(goal="等待态验证", steps=[PlanStep(tool=t, arguments={}) for t in tools])


# ---------------------------------------------------------------------------
# 一、原子性：并发 confirm 只放行一次
# ---------------------------------------------------------------------------
def test_concurrent_confirm_only_executes_once(engine):
    """两个 confirm 同时到达 ⇒ 只有一个拿到授权，高风险工具只执行一次。

    `resume()` 内部会跑完计划，这里让两个线程都卡在「领取授权」之前，
    用 barrier 同时放行，制造真实的竞态窗口（而不是靠 sleep 赌时序）。
    """
    risky = RiskyTool()
    runtime = _runtime(engine, [risky])
    session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

    run = runtime.run(session, "执行高风险操作", plan_override=_plan("wait.risky"))
    assert run.status == RunStatus.WAITING_CONFIRMATION

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def _worker() -> None:
        barrier.wait()  # 两个线程同时发起 resume
        try:
            resumed = runtime.resume(session, run.id)
            outcomes.append(f"ok:{resumed.status}")
        except ValidationException as exc:
            outcomes.append(f"rejected:{exc.message}")

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    ok = [o for o in outcomes if o.startswith("ok")]
    rejected = [o for o in outcomes if o.startswith("rejected")]
    assert len(ok) == 1, f"只能有一个请求领到授权，实际：{outcomes}"
    assert len(rejected) == 1, f"另一个必须被明确拒绝，实际：{outcomes}"
    assert risky.calls == 1, f"高风险工具只能执行一次，实际 {risky.calls} 次"


def test_concurrent_clarify_answer_only_accepted_once(engine):
    """两个 answer 同时到达 ⇒ 只有一个被受理，另一个收到明确拒绝。"""
    clarify = ClarifyTool()
    runtime = _runtime(engine, [clarify])
    session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

    run = runtime.run(session, "处理一下数据", plan_override=_plan("wait.clarify"))
    assert run.status == RunStatus.WAITING_CLARIFICATION

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def _worker(value: str) -> None:
        barrier.wait()
        try:
            answered = runtime.answer_clarification(run.id, value)
            outcomes.append(f"ok:{answered.status}")
        except ValidationException as exc:
            outcomes.append(f"rejected:{exc.message}")

    threads = [threading.Thread(target=_worker, args=("a",)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len([o for o in outcomes if o.startswith("ok")]) == 1, outcomes
    assert len([o for o in outcomes if o.startswith("rejected")]) == 1, outcomes
    # 回答只被记一次，不会因为两个请求都写而出现「后写的覆盖先写的」
    assert run.clarification_answers.get("wait.target") == "a"


# ---------------------------------------------------------------------------
# 二、持久化对称性：persist → reload → 相等
# ---------------------------------------------------------------------------
def test_store_persist_reload_is_symmetric(engine, tmp_path):
    """落盘再读回：answer_source / clarification_answers / cancel_requested 必须原样回来。

    这三条此前都漏了 —— 表现不是报错，而是「重启后一条已完成的运行被显示成
    未产出回答」「用户已回答的反问又问一遍」，只能靠断言相等来钉住。
    """
    risky = RiskyTool()
    path = tmp_path / "agent_store.json"
    store = AgentStore(path)
    runtime = _runtime(engine, [risky], store=store)
    session = runtime.create_session(dataset_ids=[engine["dataset_id"]])

    run = runtime.run(session, "执行高风险操作", plan_override=_plan("wait.risky"))
    assert run.status == RunStatus.WAITING_CONFIRMATION
    run.clarification_answers["q1"] = "a"
    run.cancel_requested = True
    run.answer_source = PLATFORM_RULES_SUMMARY
    store.persist(force=True)

    reloaded_store = AgentStore(path)
    loaded = reloaded_store.get_run(run.id)

    assert loaded.status == RunStatus.FAILED, "等待态重启后统一标记为中断"
    assert loaded.answer_source == PLATFORM_RULES_SUMMARY
    assert loaded.clarification_answers == {"q1": "a"}
    assert loaded.cancel_requested is True
    assert loaded.user_request == run.user_request
    assert [t.tool for t in loaded.tool_calls] == [t.tool for t in run.tool_calls]


# ---------------------------------------------------------------------------
# 三、空数据集是正常状态，不是异常
# ---------------------------------------------------------------------------
def test_dataset_without_version_gives_notice_not_error(db, storage):
    """建了数据集但还没导入数据 ⇒ 明确提示「请先导入数据」，而不是 500 / failed。

    旧行为：`ContextBuilder` 用 `get_version_row(ds, None)`，无版本时抛
    NotFoundException，最终被 runtime 兜成「Agent 运行异常：dataset has no versions」。
    """
    ds = DatasetService(db, storage)
    dataset = ds.create("empty-dataset", "还没导入数据")
    runtime = AgentRuntime(DataEngineService(ds), llm=None)
    session = runtime.create_session(dataset_ids=[dataset.id])

    run = runtime.run(session, "帮我看看这份数据有什么问题")

    assert run.status == RunStatus.COMPLETED, f"空数据集不该把运行打成失败：{run.error}"
    assert "还没有数据版本" in run.final_answer
    assert "导入" in run.final_answer, "必须告诉用户下一步怎么办"
    assert run.answer_source != "", "来源标记要如实填写，不能留空让前端猜"
