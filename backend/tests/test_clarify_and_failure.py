"""澄清答案与失败步骤：两次实机事故留下的护栏。

事故一（第十三轮）：等系统问「筛选条件是什么？」时，用户回答
「credit_amount 大于 5000 且 age 大于 60」，这句自然语言被**原样**写进
``conditions``，data.filter 拿到字符串直接抛异常 —— 用户被反问一次、
补了条件，最后还是失败。

事故二（同轮）：工作流创建失败（ml.train 缺 target_column），但失败的步骤
**没有进入结果集**，答案里看不到这次失败，于是被包装成
「清洗与训练工作流可这样搭：1… 2…」—— 用户以为流程已经搭好了。
"""

from __future__ import annotations

import json

import pytest

from app.agent.answer import _facts_for_llm, _failed
from app.agent.engine import _STRUCTURED_PARAMS, AgentEngine
from app.agent.llm import LLMResponse
from app.agent.models import AgentRun
from app.agent.playbooks import PlaybookStep
from app.agent.store import AgentStore
from app.tools.base import Tool
from app.tools.result import ToolResult


class _Usage:
    input_tokens = 0
    output_tokens = 0


class _RecordingProvider:
    """记录喂给模型的文本，并按指定 JSON 作答。"""

    model = "fake"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.texts: list[str] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.texts.append("\n".join(str(m.content) for m in messages))
        return LLMResponse(content=json.dumps(self.payload, ensure_ascii=False), usage=_Usage())


class _FilterTool(Tool):
    name = "data.filter"
    description = "筛选"
    category = "data"
    permission = "write_data"
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            "version": {"type": "integer"},
            "conditions": {"type": "array"},
        },
        "required": ["dataset_id", "conditions"],
    }
    output_schema = {"type": "object"}

    def execute(self, params, context, services) -> ToolResult:
        return ToolResult.ok({"rows": 1}, summary="ok")


class _FakeDatasetService:
    latest = 3

    def get_version_row(self, dataset_id, version):  # noqa: ANN001
        return type("Row", (), {"version": self.latest})()

    def scan_version(self, dataset_id, version=None, *, columns=None):  # noqa: ANN001
        return type("Frame", (), {"collect_schema": lambda self: type("S", (), {"names": lambda self: ["age", "credit_amount"]})()})()


def _engine(store: AgentStore, provider: _RecordingProvider) -> AgentEngine:
    return AgentEngine(store, llm=provider, dataset_service=_FakeDatasetService())


# ---------------------------------------------------------------- 事故一


def test_clarification_answer_is_parsed_not_pasted(store: AgentStore):
    """自然语言答案不能直接当参数值，必须交给抽取器解析。"""
    conditions = [
        {"column": "credit_amount", "op": "gt", "value": 5000},
        {"column": "age", "op": "gt", "value": 60},
    ]
    provider = _RecordingProvider({"conditions": conditions})
    engine = _engine(store, provider)
    session = store.create_session(user_id="t", title="澄清", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="帮我筛选一下")
    run.clarification_answers = {"slot.conditions": "credit_amount 大于 5000 且 age 大于 60"}

    # 不加 tuned_slots：即便抽取器没覆盖，参数里也不能出现那句原始文本。
    # 上一版就是靠 tuned 覆盖蒙混过关的，实机照样失败。
    step = PlaybookStep(tool="data.filter", title="筛选数据", required_slots=("conditions",))
    params = engine._build_params(run, session, step, _FilterTool(), {})

    assert params["conditions"] == conditions
    assert not isinstance(params["conditions"], str)
    assert "大于" not in json.dumps(params["conditions"], ensure_ascii=False)


def test_clarification_answer_never_reaches_a_required_slot_raw(store: AgentStore):
    """required_slots 那条注入路径同样要挡住（第一版只挡了一处，实机仍失败）。"""
    engine = _engine(store, _RecordingProvider({}))
    session = store.create_session(user_id="t", title="澄清", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="帮我筛选一下")
    run.clarification_answers = {"conditions": "age 大于 30"}

    step = PlaybookStep(tool="data.filter", title="筛选数据", required_slots=("conditions",))
    params = engine._build_params(run, session, step, _FilterTool(), {})

    assert not isinstance(params.get("conditions"), str)


def test_clarification_answer_reaches_the_extractor(store: AgentStore):
    """答案要拼进抽取文本，否则模型无从知道用户补了什么。"""
    provider = _RecordingProvider({"conditions": [{"column": "age", "op": "gt", "value": 30}]})
    engine = _engine(store, provider)
    session = store.create_session(user_id="t", title="澄清", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="帮我筛选一下")
    run.clarification_answers = {"slot.conditions": "age 大于 30"}

    step = PlaybookStep(tool="data.filter", title="筛选数据", required_slots=("conditions",))
    engine._build_params(run, session, step, _FilterTool(), {})

    assert any("age 大于 30" in t for t in provider.texts), provider.texts


def test_structured_params_are_declared_once():
    """结构化参数清单是解析与直传的分界线，改一处就够。"""
    assert "conditions" in _STRUCTURED_PARAMS
    assert "aggregations" in _STRUCTURED_PARAMS
    assert "nodes" in _STRUCTURED_PARAMS


def test_scalar_clarification_answer_is_still_used_directly(store: AgentStore):
    """普通标量槽位（列名之类）原样用就行，不必多花一次模型调用。"""
    engine = _engine(store, _RecordingProvider({}))
    session = store.create_session(user_id="t", title="澄清", dataset_ids=[1])
    run = AgentRun(id="r-1", session_id=session.id, user_request="看看某列的分布")
    run.clarification_answers = {"slot.column": "credit_amount"}

    step = PlaybookStep(tool="fake.read", title="分布", required_slots=("column",))
    params = engine._build_params(run, session, step, _FilterTool(), {})

    assert params["column"] == "credit_amount"


# ---------------------------------------------------------------- 事故二


def test_failed_steps_show_up_in_facts():
    """失败的步骤必须进事实摘要，答案才可能如实说出来。"""
    steps = [
        (
            PlaybookStep("workflow.build_and_run", "创建并执行工作流"),
            type("R", (), {"success": False, "summary": "", "data": {}, "errors": ["缺少 target_column"]})(),
        ),
    ]
    facts = _facts_for_llm(steps)
    assert "没有成功" in facts
    assert "target_column" in facts


def test_failed_helper_ignores_successes():
    ok = type("R", (), {"success": True, "summary": "s", "data": {}, "errors": []})()
    bad = type("R", (), {"success": False, "summary": "", "data": {}, "errors": ["x"]})()
    steps = [(PlaybookStep("a", "A"), ok), (PlaybookStep("b", "B"), bad)]
    assert [s.tool for s, _ in _failed(steps)] == ["b"]


@pytest.mark.parametrize(
    "text",
    ["算了", "不用清洗了", "不要了", "取消", "先别做了", "别执行", "跳过"],
)
def test_abandonment_is_recognized(text: str):
    from app.agent.intents import is_abandonment

    assert is_abandonment(text)


@pytest.mark.parametrize(
    "text",
    ["筛选出 age 大于 50 的数据", "看看 age 的分布", "训练一个模型", "生成一份报告"],
)
def test_real_requests_are_not_abandonment(text: str):
    from app.agent.intents import is_abandonment

    assert not is_abandonment(text)
