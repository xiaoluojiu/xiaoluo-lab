"""工具细节参数能不能被「微调」。

★ 事故：用户说「用随机森林，200 棵树 / 分成 5 簇」，链路照样跑 model="auto"，
超参一个都没生效。现象看起来是「Agent 只会调工具、不会按诉求调参数」，
根因有三层，这里逐层锁住：

1. ``model`` / ``params`` 根本不是槽位 —— 抽出来了也没地方放；
2. 即使抽出来，也会被 playbook 的 ``defaults={"model": "auto"}`` 盖掉；
3. LLM 抽取时只拿到参数名（``["model"]``），看不到 enum，只能填 auto。

对应三层修复：``tuned_slots``（可微调槽位，覆盖 defaults 但绝不反问）、
规则抽取模型族名与超参、把工具 schema 的 enum / 描述喂给 LLM。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.engine import AgentEngine
from app.agent.intents import Intent
from app.agent.models import RunStatus, ToolCallStatus
from app.agent.playbooks import PLAYBOOKS, PlaybookStep
from app.agent.slots import _slot_help, extract_slots, llm_extract, rule_extract
from app.agent.store import AgentStore
from app.tools.base import Tool
from app.tools.context import ToolExecutionContext
from app.tools.ml_tools import _MODEL_FAMILIES, _trainable_model_enum, MlTrainTool
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


def _train_step() -> PlaybookStep:
    """ml.train 这一步在真实 playbook 里的定义（含 tuned_slots）。"""
    for step in PLAYBOOKS[Intent.ML_TRAIN].steps:
        if step.tool == "ml.train":
            return step
    raise AssertionError("playbook 里应该有 ml.train")


# ---------------------------------------------------------------- 槽位存在性


def test_ml_train_declares_tunable_slots():
    """★ 根因 1：model / params 必须是可抽取的槽位，否则用户点了名也没处填。"""
    assert set(_train_step().tuned_slots) >= {"model", "params"}


def test_tunable_slots_are_never_asked():
    """可微调槽位不属于必填槽位：没点名就用默认值，绝不反问「请问用什么模型」。"""
    assert "model" not in _train_step().required_slots
    assert "params" not in _train_step().required_slots


# ---------------------------------------------------------------- 规则抽取


def test_random_forest_is_recognised():
    found = rule_extract("用随机森林训练一个模型", _train_step())
    assert found["model"] == "random_forest"


def test_english_model_name_is_recognised():
    assert rule_extract("use random forest", _train_step())["model"] == "random_forest"


def test_cluster_count_is_extracted_into_params():
    """★ 「分成 5 簇」必须变成 n_clusters=5，而不是被丢掉。"""
    assert rule_extract("把数据分成 5 簇做聚类", _train_step())["params"] == {"n_clusters": 5}


def test_n_clusters_written_as_keyword_is_extracted():
    assert rule_extract("kmeans，n_clusters=8", _train_step())["params"] == {"n_clusters": 8}


def test_tree_count_is_extracted_into_params():
    assert rule_extract("用 200 棵树", _train_step())["params"] == {"n_estimators": 200}


def test_model_and_params_together():
    found = rule_extract("用随机森林训练，200 棵树，最大深度 6", _train_step())
    assert found["model"] == "random_forest"
    assert found["params"]["n_estimators"] == 200
    assert found["params"]["max_depth"] == 6


def test_no_model_mentioned_yields_no_tuning():
    """没点名就一个都不填 —— 交给默认值，不猜。"""
    assert rule_extract("训练一个预测模型", _train_step()) == {}


def test_tuning_costs_no_tokens():
    """规则读得出来的参数，绝不能再去问模型。"""
    class _BoomProvider:
        def chat(self, *args, **kwargs):
            raise AssertionError("规则已命中，不该再调 LLM")

    found = extract_slots(
        "用随机森林训练，分成 5 簇", _train_step(), provider=_BoomProvider()
    )
    assert found["model"] == "random_forest"
    assert found["params"] == {"n_clusters": 5}


# ---------------------------------------------------------------- 工具 Schema 可见性


def test_ml_train_schema_lists_model_choices():
    """★ 根因 3：schema 必须把可选模型列全，否则抽取器只能一律填 auto。"""
    enum = MlTrainTool.input_schema["properties"]["model"]["enum"]
    assert "auto" in enum
    assert "random_forest_classifier" in enum
    assert "random_forest" in enum


def test_ml_train_schema_describes_params_keys():
    desc = MlTrainTool.input_schema["properties"]["params"]["description"]
    assert "n_clusters" in desc
    assert "n_estimators" in desc


def test_slot_help_puts_enum_into_prompt():
    """喂给模型的参数说明必须带上可选值，光给键名等于让模型盲填。"""
    schema = {
        "properties": {
            "model": {
                "type": "string",
                "description": "模型名",
                "enum": ["auto", "random_forest", "kmeans"],
            }
        }
    }
    text = _slot_help(schema, ["model"])
    assert "random_forest" in text
    assert "只能取" in text


class _CapturingProvider:
    """记录模型到底看到了什么 —— 用来证明 enum 真的进了 prompt。"""

    def __init__(self, payload: str):
        self.payload = payload
        self.seen = ""

    def chat(self, messages, *args, **kwargs):
        self.seen = "\n".join(getattr(m, "content", "") for m in messages)

        class _Resp:
            content = self.payload
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

        return _Resp()


def test_llm_sees_the_enum_not_just_the_key_name():
    schema = {
        "properties": {
            "model": {"type": "string", "enum": ["auto", "random_forest"], "description": "模型名"}
        }
    }
    provider = _CapturingProvider('{"model": "random_forest"}')
    result = llm_extract(
        "用随机森林",
        PlaybookStep(tool="ml.train", title="训练", tuned_slots=("model",)),
        tool_schema=schema,
        provider=provider,
    )
    assert result == {"model": "random_forest"}
    assert "random_forest" in provider.seen
    assert "只能取" in provider.seen


def test_llm_can_fill_tunable_slots():
    """规则读不出时（例如「换个性能更好的模型」），模型补位也要落到 model 上。"""
    provider = _CapturingProvider('{"model": "random_forest"}')
    step = PlaybookStep(tool="ml.train", title="训练", tuned_slots=("model",))
    assert llm_extract("换个更强的模型", step, provider=provider) == {"model": "random_forest"}


# ---------------------------------------------------------------- 模型族解析


def test_family_resolves_per_task():
    assert _MODEL_FAMILIES["random_forest"]["classification"] == "random_forest_classifier"
    assert _MODEL_FAMILIES["random_forest"]["regression"] == "random_forest_regressor"


def test_family_enum_covers_real_models():
    """族名枚举里必须含真实的注册模型名，不能只停留在族名。"""
    enum = _trainable_model_enum()
    for name in ("logistic_regression", "kmeans", "dbscan", "random_forest_classifier"):
        assert name in enum


def test_family_enum_has_no_duplicates():
    enum = _trainable_model_enum()
    assert len(enum) == len(set(enum))


# ---------------------------------------------------------------- 端到端


class _OkDetect(Tool):
    """给出确定结论的 detect：目标列与任务类型都定下来了。"""

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

    def execute(self, params, context, services):
        return ToolResult.ok(
            {"task": "classification", "target": "survived", "reasons": ["用户指定了目标列"]},
            summary="任务类型：classification",
        )


class _RecordingTrain(Tool):
    """记录真实入参的 ml.train：用它验证「用户点名的参数有没有到底层」。"""

    name = "ml.train"
    category = "ml"
    input_schema = dict(MlTrainTool.input_schema)
    output_schema = {"type": "object"}
    permission = "train_model"
    risk_level = "low"  # 本用例只关心参数传递，不走高风险确认

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    def execute(self, params, context, services):
        self._calls.append(dict(params))
        return ToolResult.ok(
            {"experiment_id": 1, "run_id": 1, "status": "success", "metrics": {"f1": 0.8}},
            summary="训练成功",
        )


class _FakeEvaluate(Tool):
    """链路最后一步。不注册的话整条链会以「工具未注册」失败，看不出参数问题。"""

    name = "ml.evaluate"
    category = "ml"
    input_schema = {"type": "object", "properties": {"run_id": {"type": "integer"}}}
    output_schema = {"type": "object"}
    permission = "analyze_data"
    risk_level = "low"

    def execute(self, params, context, services):
        return ToolResult.ok({"run_id": params.get("run_id"), "metrics": {"f1": 0.8}}, summary="已评估")


@pytest.fixture()
def train_calls() -> list[dict[str, Any]]:
    return []


@pytest.fixture()
def tuning_engine(store: AgentStore, train_calls: list[dict[str, Any]]) -> AgentEngine:
    registry = ToolRegistry()
    registry.register(_OkDetect())
    registry.register(_RecordingTrain(train_calls))
    registry.register(_FakeEvaluate())
    return AgentEngine(store, tools=registry, llm=None)


def test_tuned_model_reaches_the_tool_not_the_default(
    tuning_engine: AgentEngine, store: AgentStore, bound_session, train_calls
):
    """★ 端到端：「用随机森林」必须让工具收到 random_forest，而不是默认的 auto。

    只测规则抽取是不够的 —— 抽取出来却被 ``defaults`` 盖掉，正是这次事故的形状。
    """
    run = store.create_run(bound_session.id, "用随机森林预测 survived")
    tuning_engine.run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    assert train_calls, "训练工具应该被调用"
    assert train_calls[0]["model"] == "random_forest"


def test_tuned_hyperparams_reach_the_tool(
    tuning_engine: AgentEngine, store: AgentStore, bound_session, train_calls
):
    """★ 「200 棵树」必须变成 params，而不是被无声丢弃。"""
    run = store.create_run(bound_session.id, "预测 survived，用 200 棵树")
    tuning_engine.run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    assert train_calls[0]["params"] == {"n_estimators": 200}


def test_default_still_applies_when_user_says_nothing(
    tuning_engine: AgentEngine, store: AgentStore, bound_session, train_calls
):
    """没点名就走默认值 —— 微调槽位不能反过来把默认行为搞坏。"""
    run = store.create_run(bound_session.id, "预测 survived")
    tuning_engine.run(run, bound_session)

    assert run.status is RunStatus.COMPLETED
    assert train_calls[0]["model"] == "auto"
