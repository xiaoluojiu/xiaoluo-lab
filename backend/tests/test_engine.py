"""Agent 引擎：有界执行、可挂起、可恢复、无重规划。

旧架构的病理是「失败 → 重规划 → 再失败」的循环，单 run 能产生 20 万条
replanning 事件。新架构没有重规划：缺参数就问，高风险就等确认，其余一路到底。
这里锁住的就是这条行为边界。
"""

from __future__ import annotations

import time

import pytest

from app.agent.engine import MAX_STEPS, AgentEngine
from app.agent.models import EventType, RunStatus, ToolCallStatus
from app.agent.store import AgentStore


def _types(run) -> list[str]:
    return [e.type.value for e in run.events]


# ---------------------------------------------------------------- 对话路径


def test_chat_request_completes_without_tools(engine: AgentEngine, store: AgentStore, session):
    run = store.create_run(session.id, "你好")
    engine.run(run, session)

    assert run.status is RunStatus.COMPLETED
    assert run.tool_calls == []
    assert run.token_usage.llm_calls == 0
    assert _types(run) == ["route", "chat", "usage", "completed"]


# ---------------------------------------------------------------- 工具路径


def test_tool_request_executes_real_tool(engine: AgentEngine, store: AgentStore, session):
    run = store.create_run(session.id, "有哪些数据集")
    engine.run(run, session)

    assert run.status is RunStatus.COMPLETED
    assert [c.tool for c in run.tool_calls] == ["dataset.list"]
    assert run.tool_calls[0].status is ToolCallStatus.OK
    assert run.final_answer
    assert run.token_usage.llm_calls == 0  # 零 Token 是硬指标
    assert "tool_call" in _types(run) and "tool_result" in _types(run)


def test_analysis_tool_is_not_denied_by_permission(engine: AgentEngine, store: AgentStore, bound_session):
    """回归：Permission 枚举与工具层取值不一致时，这里会变成「权限不足」而根本不跑工具。"""
    run = store.create_run(bound_session.id, "看看这个数据集的质量")
    engine.run(run, bound_session)

    assert [c.tool for c in run.tool_calls] == ["dataset.quality"]
    assert "权限" not in (run.error or "")


def test_unbound_session_asks_for_dataset_id(engine: AgentEngine, store: AgentStore, session):
    """没绑定数据集时必须反问，不能把 None 传进工具（那会是一次莫名的 KeyError）。"""
    run = store.create_run(session.id, "看看这个数据集的质量")
    engine.run(run, session)

    assert run.status is RunStatus.WAITING_CLARIFICATION
    assert run.pending_clarification.code == "slot.dataset_id"
    assert run.tool_calls == []


def test_completed_run_has_elapsed_and_source(engine: AgentEngine, store: AgentStore, session):
    run = store.create_run(session.id, "有哪些数据集")
    engine.run(run, session)

    payload = run.to_dict()
    # 前端对这两个字段没有兜底，缺了就是空白
    assert isinstance(payload["elapsed_seconds"], (int, float))
    assert payload["tool_call_count"] == 1
    # 零 Token 路径必须如实标成规则来源，不能冒充大模型
    assert payload["answer_source"]["by_llm"] is False


def test_terminal_run_is_noop(engine: AgentEngine, store: AgentStore, session):
    run = store.create_run(session.id, "有哪些数据集")
    engine.run(run, session)
    events_before = len(run.events)

    engine.run(run, session)
    assert len(run.events) == events_before


# ---------------------------------------------------------------- 缺失参数 → 澄清


def test_missing_required_slot_suspends_for_clarification(engine: AgentEngine, store: AgentStore, session):
    """「合并数据集」缺右表 id，且没有 fallback：必须挂起问用户，不能瞎编一个。"""
    run = store.create_run(session.id, "合并数据集")
    engine.run(run, session)

    assert run.status is RunStatus.WAITING_CLARIFICATION
    assert run.pending_clarification is not None
    assert run.pending_clarification.code == "slot.right_dataset_id"
    assert run.pending_clarification.question
    assert "clarification" in _types(run)
    assert run.tool_calls == []


def test_clarification_answer_resumes_execution(engine: AgentEngine, store: AgentStore, bound_session):
    run = store.create_run(bound_session.id, "筛选数据")
    engine.run(run, bound_session)
    assert run.status is RunStatus.WAITING_CLARIFICATION

    run.clarification_answers = {"conditions": [{"column": "x", "op": "gt", "value": 1}]}
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    engine.run(run, bound_session)

    # 列不存在，工具会失败；但链路必须走到终态而不是再次挂起
    assert run.status.terminal
    assert run.tool_calls


def test_clarification_answer_accepts_prefixed_key(engine: AgentEngine, store: AgentStore, bound_session):
    """键写成 slot.conditions 也要认 —— 认错的代价是同一个问题被反复问下去。"""
    run = store.create_run(bound_session.id, "筛选数据")
    engine.run(run, bound_session)

    run.clarification_answers = {"slot.conditions": [{"column": "x", "op": "gt", "value": 1}]}
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    engine.run(run, bound_session)

    assert run.status.terminal
    assert run.tool_calls


def test_missing_slot_with_fallback_does_not_ask(engine: AgentEngine, store: AgentStore, bound_session):
    """分布分析缺列名时不反问，改用粗粒度的概览工具 —— 问用户「哪一列」体验更差。"""
    run = store.create_run(bound_session.id, "看看数据分布")
    engine.run(run, bound_session)

    assert run.pending_clarification is None
    assert [c.tool for c in run.tool_calls] == ["eda.distribution_overview"]


def test_fallback_also_checks_dataset_binding(engine: AgentEngine, store: AgentStore, session):
    """退化到兜底工具后仍缺 dataset_id 时，必须继续反问而不是直接撞 KeyError。"""
    run = store.create_run(session.id, "看看数据分布")
    engine.run(run, session)

    assert run.status is RunStatus.WAITING_CLARIFICATION
    assert run.pending_clarification.code == "slot.dataset_id"


# ---------------------------------------------------------------- 高风险 → 确认


def test_high_risk_tool_waits_for_confirmation(engine: AgentEngine, store: AgentStore, bound_session):
    run = store.create_run(bound_session.id, "清洗一下数据")
    engine.run(run, bound_session)

    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.pending_confirmation is not None
    assert run.pending_confirmation.step_index == 0
    assert "permission" in _types(run)
    assert run.tool_calls == []


def test_permission_gate_runs_before_execution(engine: AgentEngine, store: AgentStore, bound_session):
    """高风险工具在确认前绝不能被执行，哪怕只执行一步。"""
    run = store.create_run(bound_session.id, "清洗一下数据")
    engine.run(run, bound_session)
    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.tool_calls == []


def test_confirmation_resumes_and_completes(engine: AgentEngine, store: AgentStore, bound_session):
    run = store.create_run(bound_session.id, "清洗一下数据")
    engine.run(run, bound_session)
    assert run.status is RunStatus.WAITING_CONFIRMATION

    # 一次性凭据：只对「当前步骤 + 当前工具」生效
    run.authorized_key = f"0:{run.pending_confirmation.tool}"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    engine.run(run, bound_session)

    assert run.status.terminal
    assert run.tool_calls
    assert run.authorized_key is None  # 用完即焚
    # 回归：确认之后不能再报「需要你确认后才会执行」——
    # 那意味着授权被当成权限不足吞掉了（ToolRegistry 曾先判 denied 再判确认）。
    assert "需要你确认" not in (run.error or "")


def test_stale_authorized_key_does_not_release_other_steps(engine: AgentEngine, store: AgentStore, bound_session):
    """一次确认不能放行整条高风险链。"""
    run = store.create_run(bound_session.id, "清洗一下数据")
    engine.run(run, bound_session)

    run.authorized_key = "7:some.other.tool"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    engine.run(run, bound_session)

    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.tool_calls == []


# ---------------------------------------------------------------- 取消 / 超时


def test_cancel_request_stops_the_run(engine: AgentEngine, store: AgentStore, session):
    run = store.create_run(session.id, "有哪些数据集")
    run.cancel_requested = True
    engine.run(run, session)

    assert run.status is RunStatus.COMPLETED
    assert "已停止" in run.final_answer


def test_unexpected_exception_still_terminates(engine: AgentEngine, store: AgentStore, session):
    """任何未预期异常都必须落到终态，否则前端会永远轮询一个不结束的运行。"""
    run = store.create_run(session.id, "有哪些数据集")
    engine.tools = None  # 制造一个必炸的内部状态
    engine.run(run, session)

    assert run.status is RunStatus.FAILED
    assert run.error
    # 失败也要有话可说：只有一句技术报错时，用户既不知道哪步挂了也不知下一步怎么问
    assert run.final_answer


def test_failed_run_explains_and_suggests(engine: AgentEngine, store: AgentStore, bound_session):
    """技术报错不能直接甩给用户。

    实测 ``data.aggregate`` 因槽位没给出 func 而失败，用户看到的只有
    ``unsupported aggregation func: None``。这里要求：失败也要说人话 + 给下一步。
    """
    from app.tools.base import Tool
    from app.tools.registry import ToolRegistry
    from app.tools.result import ToolResult

    class _Boom(Tool):
        name = "data.aggregate"
        description = "假聚合"
        category = "data"
        input_schema = {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "group_by": {"type": "array"},
                "aggregations": {"type": "array"},
            },
        }
        output_schema = {"type": "object"}
        permission = "modify_data"

        def execute(self, params, context, services) -> ToolResult:
            return ToolResult.fail(["unsupported aggregation func: None"])

    registry = ToolRegistry()
    registry.register(_Boom())
    run = store.create_run(bound_session.id, "按 purpose 分组统计 credit_amount 的均值和笔数")
    # 槽位直接注入：本用例要测的是「工具失败之后说什么」，不是抽槽位
    run.clarification_answers = {"group_by": ["purpose"], "aggregations": [{"credit_amount": "mean"}]}
    AgentEngine(store, tools=registry, llm=None).run(run, bound_session)

    assert run.status is RunStatus.FAILED
    answer = run.final_answer
    assert answer and answer != run.error, "失败答案不能只是把报错抄一遍"
    assert "data.aggregate" in answer
    assert "平均值" in answer, "聚合失败要给得出「换个说法」的建议"

    failed = [e for e in run.events if e.type is EventType.FAILED]
    assert failed and failed[-1].payload.get("final_answer") == answer


# ---------------------------------------------------------------- 有界性


def test_step_bound_is_respected(engine: AgentEngine, store: AgentStore, bound_session):
    run = store.create_run(bound_session.id, "全面分析一下这份数据")
    engine.run(run, bound_session)
    assert len(run.tool_calls) <= MAX_STEPS


def test_run_is_fast(engine: AgentEngine, store: AgentStore, session):
    """零 Token 路径不该有任何等待：真实工具调用除外，这里只跑列表工具。"""
    run = store.create_run(session.id, "有哪些数据集")
    started = time.time()
    engine.run(run, session)
    assert time.time() - started < 10


def test_engine_closes_own_db_session(engine: AgentEngine, store: AgentStore, session):
    """引擎自建的 db 会话必须自己关掉，否则 SSE 后台线程会泄漏连接。"""
    run = store.create_run(session.id, "有哪些数据集")
    engine.run(run, session)
    assert engine._services_cache is None
    assert engine._owns_db is False
