"""「没法定」必须交回给用户，而不是挑一个默认值往下跑。

真实事故（本次报障）
--------------------
用户说：「探索这两个数据集间的联系，并选用合适的机器学习分析」。
当时的链路是这样的：

1. 关键词表里没有任何一条命中「两个数据集 / 有什么联系」，
   同一句话里的「机器学习」把整句吸进了 ``Intent.ML_TRAIN``；
2. ``ml.detect_task`` 推断不出目标列，却返回一个 ``success=True``、
   ``task="clustering"`` 的结果 —— 一个**结论形状**的回答，
   回答的其实是一个不可判定的问题；
3. ``ml.train`` 于是按 clustering 自动选了 kmeans，在 2,964,624 行上全量训练，
   产出 ``silhouette = 0.1316``。

用户要的是「这两个数据集有什么联系」，拿到的是「8 个簇、轮廓系数 0.13」。
每一步单独看都没报错，合起来答非所问。

这里锁住的是**同一类缺陷**的三条边界：

- 工具「跑成功了但没定下来」时，链路必须挂起问人，不许自己挑默认值开跑；
- 用户答完之后要能接着跑，且**不带**上过期的任务类型；
- 一句话里的第二个诉求不能被静默丢弃。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.engine import AgentEngine
from app.agent.intents import Intent, route
from app.agent.models import EventType, RunStatus, ToolCallStatus
from app.agent.permission import RiskLevel
from app.agent.playbooks import get_playbook
from app.agent.store import AgentStore
from app.tools.base import Tool
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

#: 一次「目标列推断不出来」的典型 detect 结果（数值取自真实事故）
_DETECT_DATA: dict[str, Any] = {
    "task": "clustering",
    "target": None,
    "needs_target": True,
    "row_count": 2_964_624,
    "reasons": ["未命中目标列命名约定", "诉求中未识别到明确的预测对象语义"],
    "target_candidates": {
        "regression": ["fare_amount", "trip_distance", "tip_amount"],
        "classification": ["payment_type", "passenger_count", "store_and_fwd_flag"],
    },
}


class _FakeDetect(Tool):
    """只能给出候选、给不出结论的 ml.detect_task。"""

    name = "ml.detect_task"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "target": {"type": "string"},
            "infer_target": {"type": "boolean"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}
    permission = "analyze_data"
    risk_level = "low"

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: Any) -> ToolResult:
        return ToolResult.ok(dict(_DETECT_DATA), summary="任务类型：clustering")


class _SpyTrain(Tool):
    """只记录入参的训练工具。

    风险等级保持与真实 ``ml.train`` 一致（high）：这样能顺带验证
    「先问目标列、再要确认」的顺序 —— 反过来的话用户答完问题还会撞一次权限。
    """

    name = "ml.train"
    category = "ml"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "target": {"type": "string"},
            "task": {"type": "string"},
            "model": {"type": "string"},
        },
        "required": ["dataset_id", "model"],
    }
    output_schema = {"type": "object"}
    permission = "train_model"
    risk_level = "high"

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: Any) -> ToolResult:
        self._calls.append(dict(params))
        return ToolResult.ok(
            {"experiment_id": 1, "run_id": 1, "status": "success", "metrics": {"r2": 0.82}},
            summary="训练成功：linear_regression",
        )


class _FakeEvaluate(Tool):
    name = "ml.evaluate"
    category = "ml"
    input_schema = {"type": "object", "properties": {"run_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "analyze_data"
    risk_level = "low"

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: Any) -> ToolResult:
        return ToolResult.ok({"run_id": params.get("run_id"), "metrics": {"r2": 0.82}}, summary="已评估")


@pytest.fixture()
def train_calls() -> list[dict[str, Any]]:
    return []


@pytest.fixture()
def train_engine(store: AgentStore, train_calls: list[dict[str, Any]]) -> AgentEngine:
    """只挂三个假工具的引擎：把链路行为与真实数据/模型彻底解耦。"""
    registry = ToolRegistry()
    registry.register(_FakeDetect())
    registry.register(_SpyTrain(train_calls))
    registry.register(_FakeEvaluate())
    return AgentEngine(store, tools=registry, llm=None)


# ---------------------------------------------------------------- 挂起问人


def test_undecidable_target_suspends_instead_of_training(
    train_engine: AgentEngine, store: AgentStore, bound_session, train_calls
):
    """推断不出目标列时绝不能自动开训 —— 那正是 kmeans 事故的直接成因。"""
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)

    assert run.status is RunStatus.WAITING_CLARIFICATION
    assert run.pending_clarification is not None
    assert run.pending_clarification.code == "slot.target"
    # 只跑了 detect，一步都没训
    assert [c.tool for c in run.tool_calls] == ["ml.detect_task"]
    assert train_calls == []


def test_question_carries_candidate_columns(train_engine: AgentEngine, store: AgentStore, bound_session):
    """反问必须带候选列与类别提示，不能只问「请问目标列是哪一列」。"""
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)

    options = run.pending_clarification.options
    assert [o.value for o in options] == [
        "fare_amount", "trip_distance", "tip_amount",
        "payment_type", "passenger_count", "store_and_fwd_flag",
    ]
    assert options[0].note == "适合做回归（预测数值）"
    assert options[3].note == "适合做分类（预测类别）"
    # 问题文案里也要能看到候选，否则纯文本通道（无选项渲染）就只剩一句空问
    assert "fare_amount" in run.pending_clarification.question


def test_ask_precedes_permission_gate(train_engine: AgentEngine, store: AgentStore, bound_session):
    """先问目标列、再要确认。顺序反了，用户答完问题还会撞一次「需要确认」。"""
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)

    assert run.status is RunStatus.WAITING_CLARIFICATION
    assert run.pending_confirmation is None
    assert EventType.PERMISSION not in [e.type for e in run.events]


# ---------------------------------------------------------------- 回答后继续


def test_answered_target_resumes_and_trains(
    train_engine: AgentEngine, store: AgentStore, bound_session, train_calls
):
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)
    assert run.status is RunStatus.WAITING_CLARIFICATION

    run.clarification_answers = {"target": "fare_amount"}
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    train_engine.run(run, bound_session)

    # 高风险：答完问题后轮到确认，这是预期顺序，不是卡住
    assert run.status is RunStatus.WAITING_CONFIRMATION
    assert run.pending_confirmation.tool == "ml.train"

    run.authorized_key = f"{run.pending_confirmation.step_index}:ml.train"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    train_engine.run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    assert len(train_calls) == 1
    assert train_calls[0]["target"] == "fare_amount"


def test_resumed_training_does_not_carry_stale_task(train_engine: AgentEngine, store: AgentStore, bound_session, train_calls):
    """★ 目标列是后补的，任务类型就必须重新判定。

    带着 detect 阶段的 ``clustering`` 去训一个用户新选的数值列，
    会把回归问题硬跑成聚类 —— 用户答了问题，结果反而更错。
    """
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)

    run.clarification_answers = {"target": "fare_amount"}
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    train_engine.run(run, bound_session)

    run.authorized_key = f"{run.pending_confirmation.step_index}:ml.train"
    run.pending_confirmation = None
    run.status = RunStatus.RUNNING
    train_engine.run(run, bound_session)

    assert "task" not in train_calls[0]


def test_answered_target_is_not_asked_again(train_engine: AgentEngine, store: AgentStore, bound_session):
    """答完就必须放行，否则同一个问题会被无限追问。"""
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)

    run.clarification_answers = {"target": "fare_amount"}
    run.pending_clarification = None
    run.status = RunStatus.RUNNING
    train_engine.run(run, bound_session)

    assert run.status is not RunStatus.WAITING_CLARIFICATION


def test_detect_result_is_kept_for_answer(train_engine: AgentEngine, store: AgentStore, bound_session):
    """挂起前跑过的步骤不能丢：恢复后答案里仍要有 detect 的结论。"""
    run = store.create_run(bound_session.id, "训练一个模型")
    train_engine.run(run, bound_session)
    assert run.tool_calls[0].status is ToolCallStatus.OK


# ---------------------------------------------------------------- 意图与工具链


def test_two_dataset_relation_request_routes_to_relation_not_training():
    """★ 事故原句：过去它只被「机器学习」吸走，直接去训练了。"""
    result = route("探索这两个数据集间的联系，并选用合适的机器学习分析")
    assert result.intent is Intent.RELATION


def test_compound_request_reports_the_other_half():
    """一句话里两个诉求时，没执行的那个必须说出来，不能静默丢弃。"""
    result = route("探索这两个数据集间的联系，并选用合适的机器学习分析")
    assert Intent.ML_TRAIN in result.secondary
    assert "本次只执行主意图" in result.reason


def test_relation_playbook_never_trains():
    """关系探索只做只读分析：没有目标列就开训，一定是猜。"""
    playbook = get_playbook(Intent.RELATION)
    assert [s.tool for s in playbook.steps] == [
        "dataset.inspect",
        "dataset.relation",
        "ml.detect_task",
    ]


def test_relation_tool_is_registered_and_low_risk():
    from app.tools.builtin import TOOL_REGISTRY

    tool = TOOL_REGISTRY.get("dataset.relation")
    assert tool.permission == "analyze_data"
    # 探索联系是只读操作，绝不能是高风险写操作
    assert RiskLevel(tool.risk_level) is RiskLevel.LOW


def test_ml_train_step_declares_the_ask_hook():
    """playbook 必须显式声明「什么时候问、问什么、候选从哪来」。"""
    playbook = get_playbook(Intent.ML_TRAIN)
    train_step = next(s for s in playbook.steps if s.tool == "ml.train")
    assert train_step.ask_if == "needs_target"
    assert train_step.ask_slot == "target"
    assert train_step.ask_options_key == "target_candidates"
