"""六层改造的标准化回归测试。

分层对应 Prompt 的六层改造清单：

* TestStandardKernel      —— 统一契约 + 统一注册表（标准化内核）
* TestIntent              —— 意图分类唯一真源（第一层前置）
* TestPreflight           —— 第一层：Pre-flight Check
* TestClarify             —— 第一层：反问协议
* TestSemantics           —— 第二层：字段语义标准
* TestOutlierStrategy     —— 第二层：异常值方法适配
* TestFeatureOps          —— 第三层：特征工程操作注册
* TestMetricAdvisor       —— 第四层：指标推荐
* TestExperimentNarrative —— 第五层：实验结构化解读
* TestReportLayering      —— 第六层：报告分层与决策化

设计原则（与被测代码一致）：**断言 code 不断言中文文案**，
文案可以自由改，code 是契约。
"""

from __future__ import annotations

import pytest
import polars as pl

# 工具注册表是进程级全局：单独跑本文件时不会被别的文件触发注册，
# 取 TOOL_REGISTRY 前必须显式注册一次（与生产 deps 的行为一致）。
from app.tools.builtin import register_builtin_tools

register_builtin_tools()

from app.agent.clarify import (
    Clarification,
    ClarificationOption,
    ClarificationRequired,
    render,
)
from app.agent.intent import classify, hits, model_hint, wants_merge
from app.agent.preflight import PreflightInput, PreflightOutcome, run_preflight
from app.core.contracts import Decision, Finding, Severity, finding, merge_findings
from app.core.registry import Registry, RegistryConflictError, RegistryNotFoundError
from app.experiments.structured import build_narrative, classify_failure, conclude
from app.ml_engine.feature import (
    FEATURE_OP_REGISTRY,
    apply_plans,
    recommend_features,
)
from app.ml_engine.metric_advisor import (
    BusinessMetric,
    business_metric_report,
    explain_metric,
    recommend_metrics,
    stratified_regression_report,
)
from app.quality.outlier_strategy import (
    METHOD_SPECS,
    assess_column,
    select_outlier_method,
    should_suggest_cleaning,
)
from app.quality.semantics import (
    BusinessRule,
    ColumnRole,
    DistributionShape,
    ValueDomain,
    column_tokens,
    infer_semantics,
)
from app.reports.layering import (
    PRIORITIES,
    build_layers,
    classify_layer,
    dedupe,
    render_markdown,
)

# --------------------------------------------------------------------------- #


class _Run:
    """测试替身：模拟 ExperimentRun（只提供 comparator / narrative 用到的字段）。"""

    def __init__(self, run_id, experiment_id, status, metrics=None, parameters=None, error=""):
        self.id = run_id
        self.experiment_id = experiment_id
        self.status = status
        self.metrics = dict(metrics or {})
        self.parameters = dict(parameters or {})
        self.error = error
        self.runtime = 1.0
        self.experiment = None


class _Exp:
    def __init__(self, exp_id=1, task_type="regression", target_column="DepDelay"):
        self.id = exp_id
        self.task_type = task_type
        self.target_column = target_column


# =========================================================================== #
# 标准化内核
# =========================================================================== #


class TestStandardKernel:
    def test_decision_resolved_and_clarification_flags(self):
        d = Decision.of("iqr", source="requested", confidence=0.9)
        assert d.resolved and not d.needs_clarification
        assert d.severity == Severity.INFO

    def test_blocking_finding_forces_clarification(self):
        d = Decision.of(
            None, source="preflight", confidence=0.0,
            findings=[finding("preflight.target_unknown", Severity.BLOCK, "缺目标列")],
        )
        assert not d.resolved
        assert d.needs_clarification
        assert d.blocking[0].code == "preflight.target_unknown"

    def test_severity_rank_ordering(self):
        assert Severity.BLOCK.rank > Severity.CLARIFY.rank > Severity.WARN.rank > Severity.INFO.rank

    def test_merge_findings_sorts_by_severity(self):
        merged = merge_findings(
            [finding("a", Severity.INFO, "x"), finding("b", Severity.BLOCK, "y")]
        )
        assert [f.code for f in merged] == ["b", "a"]

    def test_decision_to_dict_is_serializable(self):
        payload = Decision.of(1, source="x", confidence=0.5, evidence={"k": 1}).to_dict()
        assert payload["resolved"] is True and payload["evidence"] == {"k": 1}

    def test_registry_conflict_and_missing_use_hooks(self):
        reg = Registry[int](label="数字")
        reg.register("a", 1)
        with pytest.raises(RegistryConflictError):
            reg.register("a", 2)
        with pytest.raises(RegistryNotFoundError):
            reg.get("nope")
        # replace=True 用于测试隔离与热更新
        assert reg.register("a", 3, replace=True) == 3

    def test_registry_keys_are_sorted_and_describe_all(self):
        reg = Registry[dict](label="项")
        reg.register("b", {"name": "b"})
        reg.register("a", {"name": "a"})
        assert reg.keys() == ["a", "b"]
        assert [d["name"] for d in reg.describe_all()] == ["a", "b"]

    def test_registry_subclass_can_override_errors(self):
        class _R(Registry[int]):
            def _missing_error(self, key):  # noqa: D102
                return RegistryNotFoundError(f"缺 {key}")

        r = _R(label="定制")
        with pytest.raises(RegistryNotFoundError) as exc:
            r.get("x")
        assert "缺 x" in str(exc.value)

    def test_tool_and_model_registry_share_protocol(self):
        from app.ml_engine.registry import MODEL_REGISTRY
        from app.tools.registry import TOOL_REGISTRY

        for reg in (TOOL_REGISTRY, MODEL_REGISTRY):
            assert hasattr(reg, "keys") and hasattr(reg, "try_get")
            assert reg.keys() == sorted(reg.keys())
        assert len(TOOL_REGISTRY) > 0 and len(MODEL_REGISTRY) > 0

    def test_tool_registry_keeps_legacy_names_api(self):
        from app.tools.registry import TOOL_REGISTRY

        assert "report.generate" in TOOL_REGISTRY.names()


# =========================================================================== #
# 第一层前置：意图分类唯一真源
# =========================================================================== #


class TestIntent:
    def test_greeting_is_chat(self):
        assert classify("你好").value == "chat"
        assert classify("你好").source == "greeting"

    def test_modeling_request_is_ml(self):
        d = classify("训练一个模型预测延误")
        assert d.value == "ml"
        assert d.confidence >= 0.6

    def test_merge_words_are_recognized(self):
        # 历史事故：只加了「合并」没加「关联/拼接/宽表」→ 路由放行了但候选集没注入
        for text in ("把两张表关联一下", "拼接成宽表", "join 这两个数据集", "merge 一下"):
            assert wants_merge(text), text

    def test_fallback_with_bound_dataset(self):
        d = classify("关联一下", has_datasets=True)
        assert d.value != "chat"

    def test_hits_returns_all_matched_domains(self):
        # 「建模并生成报告」两个域都要命中，否则候选工具集会漏
        got = hits("建模并生成报告")
        assert "ml" in got and "report" in got

    def test_model_hint_defaults_to_auto(self):
        # 硬编监督模型是事故来源：说不清任务类型时一律 auto
        assert model_hint("选个合适的模型") == "auto"
        assert model_hint("做回归") == "linear_regression"
        assert model_hint("做聚类") == "kmeans"


# =========================================================================== #
# 第一层：Pre-flight
# =========================================================================== #


def _cols():
    return [
        {"name": "Month", "dtype": "Float64"},
        {"name": "CRSDepTime", "dtype": "Float64"},
        {"name": "DepDelay", "dtype": "Float64"},
        {"name": "Distance", "dtype": "Float64"},
        {"name": "UniqueCarrier", "dtype": "String"},
    ]


class TestPreflight:
    def test_target_resolved_from_dataset_name_semantics(self):
        result = run_preflight(PreflightInput(
            user_request="训练一个模型",
            intent=classify("训练一个模型"),
            dataset_ids=[9],
            dataset_meta={9: {"name": "航空公司出发延误预测", "rows": 10_000_000}},
            columns=_cols(),
        ))
        assert result.outcome == PreflightOutcome.PROCEED
        assert result.resolved["target"] == "DepDelay"
        assert result.resolved["target_source"] == "goal_match"

    def test_unknown_target_blocks_with_candidates(self):
        result = run_preflight(PreflightInput(
            user_request="训练一个模型",
            intent=classify("训练一个模型"),
            dataset_ids=[9],
            dataset_meta={9: {"name": "data", "rows": 1000}},
            columns=_cols(),
        ))
        assert result.outcome == PreflightOutcome.BLOCK
        assert result.findings[0].code == "preflight.target_unknown"
        question = result.first_question()
        assert question is not None and question.code == "preflight.target_unknown"
        # 日历列不能排在候选第一位（默认选项错了等于默认值错了）
        assert question.options[0].value not in ("Month", "CRSDepTime")
        assert question.default == question.options[0].value

    def test_task_type_conflict_is_blocked(self):
        result = run_preflight(PreflightInput(
            user_request="做分类",
            intent=classify("做分类"),
            dataset_ids=[9],
            dataset_meta={9: {"name": "data", "rows": 1000}},
            columns=_cols(),
            target="DepDelay",
            task_type="classification",
        ))
        assert result.outcome == PreflightOutcome.BLOCK
        assert any(f.code == "preflight.task_type_conflict" for f in result.findings)

    def test_non_modeling_request_is_not_asked_about_target(self):
        result = run_preflight(PreflightInput(
            user_request="看下数据质量",
            intent=classify("看下数据质量"),
            dataset_ids=[9],
            dataset_meta={9: {"name": "data", "rows": 1000}},
            columns=_cols(),
        ))
        assert result.outcome == PreflightOutcome.PROCEED
        assert not any(f.code.startswith("preflight.target") for f in result.findings)

    def test_answered_question_is_not_asked_again(self):
        result = run_preflight(PreflightInput(
            user_request="训练一个模型",
            intent=classify("训练一个模型"),
            dataset_ids=[9],
            dataset_meta={9: {"name": "data", "rows": 1000}},
            columns=_cols(),
            answers={"preflight.target_unknown": "DepDelay"},
        ))
        assert result.outcome == PreflightOutcome.PROCEED
        assert result.resolved["answered"]["preflight.target_unknown"] == "DepDelay"

    def test_missing_dataset_blocks(self):
        result = run_preflight(PreflightInput(
            user_request="训练一个模型",
            intent=classify("训练一个模型"),
            dataset_ids=[],
        ))
        assert result.outcome == PreflightOutcome.BLOCK
        assert result.findings[0].code == "preflight.dataset_required"

    def test_no_cascading_noise_when_dataset_missing(self):
        # 没有数据集 ⇒ 必然没有目标列；不该再追问目标列
        result = run_preflight(PreflightInput(
            user_request="训练一个模型",
            intent=classify("训练一个模型"),
            dataset_ids=[],
        ))
        assert [f.code for f in result.findings] == ["preflight.dataset_required"]

    def test_result_is_serializable(self):
        payload = run_preflight(PreflightInput(
            user_request="训练一个模型", intent=classify("训练一个模型"), dataset_ids=[],
        )).to_dict()
        assert payload["outcome"] == "block"
        assert payload["clarifications"][0]["code"] == "preflight.dataset_required"


# =========================================================================== #
# 第一层：反问协议
# =========================================================================== #


class TestClarify:
    def test_render_uses_template_by_code(self):
        f = finding("preflight.target_unknown", Severity.BLOCK, "缺目标列",
                    evidence={"regression_candidates": ["DepDelay", "Distance"]})
        c = render(f)
        assert c.code == "preflight.target_unknown"
        assert c.severity == Severity.BLOCK
        assert [o.value for o in c.options] == ["DepDelay", "Distance"]
        assert c.default == "DepDelay"

    def test_render_falls_back_for_unknown_code(self):
        c = render(finding("some.new.code", Severity.WARN, "一句话"))
        assert c.code == "some.new.code" and c.options == []

    def test_clarify_tool_registered(self):
        from app.tools.registry import TOOL_REGISTRY

        tool = TOOL_REGISTRY.get("agent.clarify")
        assert tool.input_schema["required"] == ["code", "question"]

    def test_clarify_tool_raises_when_unanswered(self):
        from app.tools.context import ToolExecutionContext
        from app.tools.result import ToolResult

        tool = FEATURE_OP_REGISTRY and None  # 占位，避免误用
        from app.tools.registry import TOOL_REGISTRY

        tool = TOOL_REGISTRY.get("agent.clarify")
        ctx = ToolExecutionContext(user_id="u", session_id="s", dataset_ids=set(),
                                   permissions=set(), extra={})
        with pytest.raises(ClarificationRequired) as exc:
            tool.execute({"code": "c1", "question": "选哪一列?",
                          "options": [{"value": "DepDelay"}]}, ctx, None)
        assert exc.value.clarification.options[0].value == "DepDelay"

    def test_clarify_tool_returns_answer_when_already_answered(self):
        from app.agent.clarify import ANSWERS_KEY
        from app.agent.permission.models import ROLE_PERMISSIONS
        from app.tools.context import ToolExecutionContext
        from app.tools.registry import TOOL_REGISTRY
        from app.tools.result import ToolResult

        tool = TOOL_REGISTRY.get("agent.clarify")
        ctx = ToolExecutionContext(user_id="u", session_id="s", dataset_ids=set(),
                                   permissions=set(ROLE_PERMISSIONS["analyst"]),
                                   extra={ANSWERS_KEY: {"c1": "DepDelay"}})
        result = tool.execute({"code": "c1", "question": "选哪一列?"}, ctx, None)
        assert isinstance(result, ToolResult) and result.data["answer"] == "DepDelay"

    def test_executor_maps_clarification_to_needs_clarification(self):
        from app.agent.executor.executor import AgentExecutor
        from app.agent.permission.models import ROLE_PERMISSIONS
        from app.agent.state import PendingAction
        from app.tools.context import ToolExecutionContext

        record = AgentExecutor().execute_step(
            PendingAction(tool="agent.clarify", arguments={"code": "c1", "question": "q"}),
            ToolExecutionContext(user_id="u", session_id="s", dataset_ids=set(),
                                 permissions=set(ROLE_PERMISSIONS["analyst"]), extra={}),
            None,
        )
        assert record.status == "needs_clarification"
        assert record.clarification.code == "c1"

    def test_run_status_has_waiting_clarification(self):
        from app.agent.runtime.models import RunStatus

        assert RunStatus.WAITING_CLARIFICATION == "waiting_clarification"

    def test_clarification_option_defaults_label_to_value(self):
        opt = ClarificationOption(value="DepDelay")
        assert opt.to_dict()["label"] == "DepDelay"

    def test_clarification_is_serializable(self):
        c = Clarification(code="c", question="q", options=[ClarificationOption(value="v")])
        assert c.to_dict()["options"][0]["value"] == "v"


# =========================================================================== #
# 第二层：字段语义
# =========================================================================== #


class TestSemantics:
    @staticmethod
    def _frame(n=3000):
        return pl.DataFrame({
            "Month": [float(i % 12 + 1) for i in range(n)],
            "CRSDepTime": [float((i // 60 % 24) * 100 + i % 60) for i in range(n)],
            "DepDelay": [0.0 if i % 3 else float(i % 300) for i in range(n)],
            "Distance": [float(100 + (i * 37) % 2500) for i in range(n)],
            "UniqueCarrier": [f"C{i % 30}" for i in range(n)],
            "Origin": [f"A{i % 369}" for i in range(n)],
        })

    def test_column_tokens(self):
        assert column_tokens("CRSDepTime") == ["crs", "dep", "time"]
        assert column_tokens("DayofMonth") == ["dayof", "month"]

    def test_hhmm_detected_and_calendar_not(self):
        sem = infer_semantics(self._frame())
        assert sem["CRSDepTime"].pseudo_time_format == "HHMM"
        assert sem["CRSDepTime"].role == ColumnRole.TEMPORAL
        # 日历列（Month=1..12）不能被误判成 HHMM
        assert sem["Month"].pseudo_time_format is None
        assert sem["Month"].role == ColumnRole.CATEGORICAL

    def test_non_negative_domain(self):
        sem = infer_semantics(self._frame())
        assert sem["Distance"].domain == ValueDomain.NON_NEGATIVE
        assert sem["Distance"].is_non_negative

    def test_zero_inflated_target(self):
        sem = infer_semantics(self._frame(), target="DepDelay")
        assert sem["DepDelay"].is_target
        assert sem["DepDelay"].shape == DistributionShape.ZERO_INFLATED

    def test_high_cardinality_flag(self):
        sem = infer_semantics(self._frame())
        assert sem["Origin"].is_high_cardinality
        assert not sem["UniqueCarrier"].is_high_cardinality

    def test_business_rule_registered_and_wins(self):
        rules = [BusinessRule(column="Distance", lower=0, strict_lower=True, note="距离必须为正")]
        sem = infer_semantics(self._frame(), business_rules=rules)
        assert sem["Distance"].stats["business_rule"]["lower"] == 0
        assert "business_rule" in sem["Distance"].tags

    def test_semantics_is_serializable(self):
        payload = infer_semantics(self._frame())["Distance"].to_dict()
        assert payload["role_label"] and payload["shape_label"]


# =========================================================================== #
# 第二层：异常值方法适配
# =========================================================================== #


class TestOutlierStrategy:
    @staticmethod
    def _frame(n=2000):
        # 注意：Distance 的取值数必须明显少于行数，否则会被判成标识列
        # （「每行唯一」是主键/哈希的判据），测的就不是非负字段了。
        return pl.DataFrame({
            "Distance": [float(100 + (i * 37) % 900) for i in range(n)],
            "DepDelay": [0.0 if i % 3 else float(i % 300) for i in range(n)],
            "Profit": [float((i % 200) - 100) for i in range(n)],
            "Carrier": [f"C{i % 15}" for i in range(n)],
        })

    def test_non_negative_column_rejects_iqr_with_reason(self):
        sem = infer_semantics(self._frame())
        d = select_outlier_method(sem["Distance"], requested="iqr")
        # 非负字段上 IQR 下界会为负 ⇒ 必须改用别的口径，且要说出原因
        assert d.value != "iqr"
        assert any(f.code == "quality.method_mismatch" for f in d.findings)
        assert "非负" in d.findings[0].message

    def test_signed_column_keeps_iqr(self):
        sem = infer_semantics(self._frame())
        # 有正负的连续列：IQR 成立（形态未知/正态时）
        d = select_outlier_method(sem["Profit"], requested="iqr")
        assert d.value == "iqr" or d.source == "auto_adapted"

    def test_categorical_uses_low_frequency(self):
        sem = infer_semantics(self._frame())
        d = select_outlier_method(sem["Carrier"], requested="iqr")
        assert d.value == "low_frequency"

    def test_target_column_is_protected(self):
        sem = infer_semantics(self._frame(), target="DepDelay")
        d = select_outlier_method(sem["DepDelay"], requested="iqr")
        assert d.value == "none" and d.source == "target_protected"
        assert not should_suggest_cleaning(sem["DepDelay"])

    def test_business_rule_beats_statistics(self):
        sem = infer_semantics(self._frame(), business_rules=[
            BusinessRule(column="Distance", upper=3000, note="航线距离上限"),
        ])
        d = select_outlier_method(sem["Distance"], requested="iqr")
        assert d.value == "business_rule" and d.source == "business_rule"

    def test_assess_bounds_never_negative_for_non_negative_column(self):
        df = self._frame()
        sem = infer_semantics(df)
        d = assess_column(sem["Distance"], df["Distance"], requested="iqr")
        assert d.value is None or d.value.method == "none" or (
            d.value.lower is None or d.value.lower >= 0
        )

    def test_method_table_is_declared_not_branchy(self):
        # 方法表是声明式的：新增方法只加一条记录，调度代码不变
        codes = {s.code for s in METHOD_SPECS}
        assert {"business_rule", "low_frequency", "quantile", "mad", "iqr", "zscore", "none"} <= codes
        assert all(s.guard is not None for s in METHOD_SPECS)

    def test_unknown_method_falls_back_with_warning(self):
        sem = infer_semantics(self._frame())
        d = select_outlier_method(sem["Distance"], requested="no_such_method")
        assert any(f.code == "quality.unknown_method" for f in d.findings)


# =========================================================================== #
# 第三层：特征工程
# =========================================================================== #


class TestFeatureOps:
    @staticmethod
    def _frame(n=2400):
        return pl.DataFrame({
            "CRSDepTime": [float((i // 60 % 24) * 100 + i % 60) for i in range(n)],
            "DepDelay": [0.0 if i % 3 else float(i % 300) for i in range(n)],
            "Origin": [f"A{i % 369}" for i in range(n)],
        })

    def test_registry_has_expected_ops(self):
        for code in ("time.parse_hhmm", "time.cyclic", "encode.frequency",
                     "encode.target", "aggregate.group_stat", "collinearity.flag"):
            assert FEATURE_OP_REGISTRY.try_get(code) is not None, code

    def test_ops_declare_leakiness(self):
        assert FEATURE_OP_REGISTRY.get("encode.target").leaky is True
        assert FEATURE_OP_REGISTRY.get("aggregate.group_stat").leaky is True
        assert FEATURE_OP_REGISTRY.get("encode.frequency").leaky is False

    def test_hhmm_parse_produces_hour_and_minutes(self):
        df = self._frame()
        sem = infer_semantics(df)
        op = FEATURE_OP_REGISTRY.get("time.parse_hhmm")
        assert op.check(sem["CRSDepTime"]) is None
        out, receipts = apply_plans(df, [op.build(df, sem["CRSDepTime"], {})])
        assert "CRSDepTime_hour" in out.columns and "CRSDepTime_minutes" in out.columns
        assert receipts[0]["columns_after"] > receipts[0]["columns_before"]
        # HHMM → 当日分钟数：2359 应解析成 1439
        assert out["CRSDepTime_minutes"].max() <= 1439

    def test_frequency_encoding_adds_single_column(self):
        df = self._frame()
        sem = infer_semantics(df)
        op = FEATURE_OP_REGISTRY.get("encode.frequency")
        assert op.check(sem["Origin"]) is None
        out, _ = apply_plans(df, [op.build(df, sem["Origin"], {})])
        assert "Origin_freq" in out.columns
        # 高基数列频次编码后列数只 +1（one-hot 会 +369）
        assert out.width == df.width + 1

    def test_target_encoding_refuses_without_folds(self):
        df = self._frame()
        sem = infer_semantics(df)
        op = FEATURE_OP_REGISTRY.get("encode.target")
        plan = op.build(df, sem["Origin"], {"target": "DepDelay"})
        assert plan.new_columns == []
        assert any("out-of-fold" in w for w in plan.warnings)

    def test_target_encoding_with_folds_produces_column(self):
        df = self._frame()
        sem = infer_semantics(df)
        folds = pl.Series("__fold", [i % 3 for i in range(df.height)])
        op = FEATURE_OP_REGISTRY.get("encode.target")
        plan = op.build(df, sem["Origin"], {"target": "DepDelay", "folds": folds})
        assert plan.new_columns == ["Origin_te"]
        out, _ = apply_plans(df, [plan])
        assert "Origin_te" in out.columns

    def test_group_stat_refuses_without_folds(self):
        df = self._frame()
        sem = infer_semantics(df)
        op = FEATURE_OP_REGISTRY.get("aggregate.group_stat")
        plan = op.build(df, sem["Origin"], {"group_by": "Origin", "target": "DepDelay"})
        assert plan.new_columns == [] and plan.warnings

    def test_advisor_flags_hhmm_and_high_cardinality(self):
        df = self._frame()
        sem = infer_semantics(df, target="DepDelay")
        d = recommend_features(sem, target="DepDelay")
        ops = {(a.column, a.op) for a in d.value}
        assert ("CRSDepTime", "time.parse_hhmm") in ops
        assert ("Origin", "encode.frequency") in ops
        assert any(a.priority == "必须" for a in d.value)

    def test_advisor_flags_collinearity(self):
        df = self._frame()
        sem = infer_semantics(df, target="DepDelay")
        d = recommend_features(
            sem, target="DepDelay",
            corr_pairs=[{"a": "CRSDepTime", "b": "DepDelay", "r": 0.78}],
        )
        assert any(a.op == "collinearity.flag" for a in d.value)

    def test_advisor_does_not_touch_target_column(self):
        df = self._frame()
        sem = infer_semantics(df, target="DepDelay")
        d = recommend_features(sem, target="DepDelay")
        assert all(a.column != "DepDelay" for a in d.value)

    def test_priority_vocabulary_shared(self):
        assert PRIORITIES == ("必须做", "建议做", "可选做")


# =========================================================================== #
# 第四层：指标顾问
# =========================================================================== #


class TestMetricAdvisor:
    @staticmethod
    def _sem():
        n = 2000
        df = pl.DataFrame({"DepDelay": [0.0 if i % 3 else float(i % 300) for i in range(n)]})
        return infer_semantics(df, target="DepDelay")["DepDelay"]

    def test_long_tail_target_gets_mae_and_quantile(self):
        d = recommend_metrics("regression", self._sem())
        codes = {a.code for a in d.value}
        assert "mae" in codes and "quantile_loss" in codes

    def test_rmse_warned_for_long_tail(self):
        d = recommend_metrics("regression", self._sem())
        assert any(f.code == "metric.rmse_misleading" for f in d.findings)

    def test_business_threshold_enables_hit_rate(self):
        d = recommend_metrics("regression", self._sem(), business_threshold=15.0)
        assert any(a.code == "hit_rate" for a in d.value)

    def test_explain_metric_guards_against_r2_misread(self):
        text = explain_metric("r2", target_sem=self._sem())
        assert "不等于" in text or "不等于模型不可用" in text

    def test_stratified_report_splits_by_quantile(self):
        rows = stratified_regression_report(
            [0.0, 1.0, 50.0, 200.0, 3.0, 0.0], [0.5, 1.2, 30.0, 120.0, 2.0, 1.0], segments=3,
        )
        assert len(rows) >= 2
        # 大值区间的误差必须能被单独看到（总量指标会掩盖它）
        assert rows[-1]["mae"] > rows[0]["mae"]

    def test_business_metric_report(self):
        out = business_metric_report(
            [0.0, 20.0, 30.0, 1.0], [1.0, 25.0, 10.0, 2.0],
            [BusinessMetric(code="delay15", label="延误>15分钟", threshold=15.0)],
        )
        assert out[0]["code"] == "delay15"
        assert out[0]["recall"] == 0.5 and out[0]["precision"] == 1.0

    def test_classification_imbalance_gets_auc(self):
        d = recommend_metrics("classification", None, class_counts={"否": 950, "是": 50})
        codes = {a.code for a in d.value}
        assert "auc" in codes and "recall" in codes


# =========================================================================== #
# 第五层：实验结构化解读
# =========================================================================== #


class TestExperimentNarrative:
    def test_failure_classification_known_codes(self):
        cases = {
            "target_column 不能出现在 excluded_columns 中": "target_in_excluded",
            "LogisticRegression.__init__() got an unexpected keyword argument 'n_clusters'": "model_param_mismatch",
            "Unable to allocate 57.1 GiB for an array": "memory_overflow",
            "未能按命名约定确定目标列": "missing_target",
            "测试集样本不足以覆盖 70 个类别": "stratify_on_regression",
        }
        for message, code in cases.items():
            assert classify_failure(message).value.code == code, message

    def test_unknown_failure_is_honest(self):
        d = classify_failure("某种全新的报错")
        assert d.value.code == "unknown" and d.value.matched is False
        assert d.confidence < 0.5

    def test_narrative_has_all_four_fields(self):
        exp = _Exp()
        n = build_narrative(exp, [_Run(1, 1, "success", {"r2": 0.02})])
        for field_name in ("hypothesis", "conclusion", "next_action", "failure_reason"):
            assert hasattr(n, field_name)
        assert n.hypothesis and n.next_action
        assert n.failure_reason == ""

    def test_failed_experiment_gets_actionable_fix(self):
        exp = _Exp()
        run = _Run(2, 1, "failed", error="target_column 不能出现在 excluded_columns 中")
        n = build_narrative(exp, [run])
        assert n.failure_reason.startswith("[target_in_excluded]")
        assert "excluded_columns" in n.next_action

    def test_conclude_identifies_driver_change(self):
        from app.experiments.comparator import ExperimentComparator

        runs = [
            _Run(1, 1, "success", {"r2": 0.02}, {"model": "linear_regression", "fe": False}),
            _Run(2, 1, "success", {"r2": 0.18}, {"model": "linear_regression", "fe": True}),
        ]
        compare = ExperimentComparator().compare(runs)
        conclusion = conclude(compare, primary_metric="r2")
        assert conclusion["best_run_id"] == 2
        assert conclusion["driver_changes"] == {"fe": {"before": False, "after": True}}
        assert conclusion["delta_vs_baseline"].startswith("+")

    def test_conclude_single_run_cannot_attribute(self):
        from app.experiments.comparator import ExperimentComparator

        compare = ExperimentComparator().compare([_Run(1, 1, "success", {"r2": 0.02})])
        assert conclude(compare)["driver_changes"] == {}

    def test_narrative_is_serializable(self):
        payload = build_narrative(_Exp(), [_Run(1, 1, "success", {"r2": 0.1})]).to_dict()
        assert set(payload) >= {"hypothesis", "conclusion", "next_action", "failure_reason"}


# =========================================================================== #
# 第六层：报告分层
# =========================================================================== #


class TestReportLayering:
    def test_layer_classification(self):
        assert classify_layer("建议清洗后再建模") == "action"
        assert classify_layer("RMSE 可能被极端值主导") == "judgment"
        assert classify_layer("共 1000 行 × 10 列") == "fact"

    def test_dedupe_removes_contained_duplicates(self):
        items = [
            "数据存在 12 个质量问题",
            "数据存在 12 个质量问题，其中 3 个需在建模前处理",
            "完全不同的另一条结论",
        ]
        kept = dedupe(items)
        assert len(kept) == 2
        assert "需在建模前处理" in kept[0]

    def test_dedupe_keeps_order(self):
        assert dedupe(["甲", "乙", "甲"]) == ["甲", "乙"]

    def test_build_layers_splits_three_layers(self):
        layers = build_layers([
            "数据集共 1000 行 × 10 列。",
            "R² 低不代表模型不可用，需结合 MAE 判断。",
            "必须做：先清洗阻塞性问题再建模。",
            "建议做：做时间字段解析后重跑。",
        ])
        assert len(layers.facts) == 1
        assert len(layers.judgments) == 1
        assert len(layers.actions) == 2

    def test_actions_sorted_by_priority(self):
        layers = build_layers(["建议做：调优参数", "必须做：补目标列", "可选做：换配色"])
        assert [a.priority for a in layers.actions] == ["必须做", "建议做", "可选做"]

    def test_next_action_falls_back_to_top_action(self):
        layers = build_layers(["建议做：调优参数", "必须做：补目标列"])
        assert layers.next_action == "必须做：补目标列"

    def test_explicit_next_action_wins(self):
        layers = build_layers(["必须做：补目标列"], next_action="先跑基线再谈调优")
        assert layers.next_action == "先跑基线再谈调优"

    def test_render_markdown_has_three_sections_and_next(self):
        md = render_markdown(build_layers(
            ["数据集共 10 列。", "可能被极端值主导。", "必须做：先清洗。"],
            next_action="先跑基线",
        ))
        assert "事实" in md and "判断" in md and "行动" in md and "下一步" in md

    def test_layers_are_serializable(self):
        payload = build_layers(["必须做：先清洗"]).to_dict()
        assert payload["actions"][0]["priority"] == "必须做"
