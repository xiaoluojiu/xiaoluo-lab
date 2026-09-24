"""TaskSpec / DecisionProvider / DecisionTrace 的单元测试（Phase 1 架构接通）。"""

from __future__ import annotations

import pytest

from app.agent.decision import (
    DecisionAction,
    DecisionContext,
    DecisionRouter,
    DecisionSource,
    RuleDecisionProvider,
    LocalModelDecisionProvider,
)
from app.agent.task_spec import TaskDomain, TaskScope, TaskSpec
from app.agent.task_spec_builder import TaskSpecBuilder, _split_first_clause
from app.agent.trace.decision_trace import DecisionTrace


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_minimal_fields(self):
        spec = TaskSpec(goal="看看这批数据的分布", domain=TaskDomain.EDA,
                        sub_goal="distribution", scope=TaskScope.DATASET)
        d = spec.to_dict()
        assert d["goal"] == "看看这批数据的分布"
        assert d["domain"] == "eda"
        assert d["sub_goal"] == "distribution"
        assert d["scope"] == "dataset"
        # TaskSpec 不执行工具：entities 里的 tool 只是描述，不是执行能力。
        assert "tool" not in d or d["tool"] is None

    def test_resolved(self):
        assert TaskSpec(goal="x").resolved
        assert not TaskSpec(goal="").resolved


# ---------------------------------------------------------------------------
# DecisionProvider
# ---------------------------------------------------------------------------


class TestRuleDecisionProvider:
    def test_cancel_command(self):
        p = RuleDecisionProvider()
        d = p.decide(DecisionContext(user_request="取消"))
        assert d.action == DecisionAction.STOP
        assert d.source == DecisionSource.RULE
        assert d.confidence == 1.0

    def test_no_semantic_mapping(self):
        """规则不承担「分布→工具」这类语义映射：未绑定数据集时它只做槽位检查，
        不会硬猜工具。"""
        p = RuleDecisionProvider()
        d = p.decide(DecisionContext(user_request="看看这批数据的分布",
                                     bound_dataset_id=None))
        # 规则不会给出语义工具选择（那是本地模型的职责）。
        assert d.action != DecisionAction.EXECUTE_TOOL or d.tool is None


class TestLocalModelDecisionProvider:
    def test_quality_request(self):
        p = LocalModelDecisionProvider()
        d = p.decide(DecisionContext(user_request="检查一下数据质量", bound_dataset_id=1))
        assert d.action == DecisionAction.EXECUTE_TOOL
        assert d.tool == "dataset.quality"
        assert d.source in (DecisionSource.LOCAL_ROUTER, DecisionSource.LOCAL_MODEL)

    def test_distribution_request(self):
        p = LocalModelDecisionProvider()
        d = p.decide(DecisionContext(user_request="看看这批数据的分布", bound_dataset_id=1))
        assert d.action == DecisionAction.EXECUTE_TOOL
        # 数据集级分布总览（无需 column），不再被误判为单列 distribution。
        assert d.tool == "eda.distribution_overview"


class TestDecisionRouter:
    def test_rule_priority(self):
        router = DecisionRouter(llm=None)
        d = router.route(DecisionContext(user_request="取消"), allow_remote=False)
        assert d.action == DecisionAction.STOP
        assert d.source == DecisionSource.RULE

    def test_local_model_used(self):
        """本地模型真正参与（不是摆设）。"""
        router = DecisionRouter(llm=None)
        d = router.route(DecisionContext(user_request="检查一下数据质量", bound_dataset_id=1),
                         allow_remote=False)
        assert d.action == DecisionAction.EXECUTE_TOOL
        assert d.tool == "dataset.quality"
        assert d.source == DecisionSource.LOCAL_ROUTER


# ---------------------------------------------------------------------------
# TaskSpecBuilder
# ---------------------------------------------------------------------------


class TestTaskSpecBuilder:
    def test_distribution_domain(self):
        u = TaskSpecBuilder().understand("看看这批数据的分布", bound_dataset_id=1)
        assert u.spec.domain == TaskDomain.EDA
        # 数据集级分布总览（sub_goal=distribution_overview，无需猜列）。
        assert u.spec.sub_goal == "distribution_overview"
        assert u.spec.entities.get("tool") == "eda.distribution_overview"

    def test_ml_domain(self):
        u = TaskSpecBuilder().understand("帮我训练一个分类模型", bound_dataset_id=1)
        assert u.spec.domain == TaskDomain.ML
        assert u.spec.entities.get("tool") == "ml.train"

    def test_multi_step_decomposed_to_first_tool(self):
        """「先X再Y」的多步编排应拆解为「首步工具 + medium」，本地循环推进，
        而不是升级远程（complex）。首步工具从第一个分句识别。"""
        u = TaskSpecBuilder().understand("先看看数据分布，再根据结果分析异常情况", bound_dataset_id=1)
        assert u.spec.entities.get("tool") == "eda.distribution_overview"
        # medium 表示「本地多步循环」，不是 complex（远程升级）。
        assert u.spec.complexity_hint.value == "medium"
        # source 仍是本地 Router（不是远程强塞）。
        assert u.spec.source == "local_router"


class TestSplitFirstClause:
    def test_strip_leading_先_and_trailing_comma(self):
        assert _split_first_clause("先看看数据分布，再根据结果分析异常情况") == "看看数据分布"

    def test_再_marker(self):
        assert _split_first_clause("先清洗缺失值然后再做分布图") == "清洗缺失值"

    def test_然后_marker(self):
        assert _split_first_clause("看看分布，然后分析相关性") == "看看分布"

    def test_no_marker_returns_original(self):
        assert _split_first_clause("看看这批数据的分布") == "看看这批数据的分布"

    def test_empty(self):
        assert _split_first_clause("") == ""


# ---------------------------------------------------------------------------
# DecisionTrace
# ---------------------------------------------------------------------------


class TestDecisionTrace:
    def test_roundtrip_fields(self):
        t = DecisionTrace(trace_id="r-1", request="看看分布")
        t.task_spec = {"goal": "看看分布", "domain": "eda"}
        t.router_candidates = [{"candidate": "eda.distribution", "score": 0.8, "source": "local_router"}]
        t.decision_made = {"action": "execute_tool", "tool": "eda.distribution"}
        t.decision_source = "local_router"
        t.record_decision({"step": 1, "action": "execute_tool", "source": "local_router"})
        t.record_decision({"step": 2, "action": "ask_user", "source": "rule"})
        t.success = True
        t.remote_calls = 0
        t.local_calls = 2

        d = t.to_dict()
        assert d["trace_id"] == "r-1"
        assert len(d["decision_steps"]) == 2
        assert d["remote_calls"] == 0
        assert d["local_calls"] == 2
