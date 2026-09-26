"""Task 3 — 确定性层 playbooks / 信号 / 槽位 / 分句 的单元测试。

覆盖：九类 playbook 选择与工具序列、已完成动作不重复入队、建模链引用步号、
客观信号→工具白名单、取消命令、必填槽位、多步分句、缺失值策略，
以及「确定性层零 LLM 依赖」的源码不变式。
"""
from __future__ import annotations

from app.agent import playbooks as pb
from app.agent.playbooks import (
    action_for_signal,
    is_cancel_command,
    missing_fill_strategy,
    missing_required_slots,
    select_playbook,
    split_first_clause,
)
from app.agent.state import PendingAction, TaskState


def _tools(sel) -> list[str]:
    return [a.tool for a in sel.actions]


def _sel(text, state=None, dataset_ids=None, base_index=0):
    return select_playbook(text, state, dataset_ids or [1], base_index=base_index)


# --------------------------------------------------------------------------- 分句 / 取消 / 槽位


class TestSplitFirstClause:
    def test_split_on_marker(self):
        assert split_first_clause("先看一下数据再分析分布") == "看一下数据"

    def test_no_marker_returns_original(self):
        assert split_first_clause("看看有多少行") == "看看有多少行"

    def test_leading_xian_stripped(self):
        assert split_first_clause("先统计分布") == "统计分布"

    def test_empty(self):
        assert split_first_clause("") == ""


class TestCancel:
    def test_cancel_commands(self):
        assert is_cancel_command("取消")
        assert is_cancel_command("停止")
        assert is_cancel_command("退出")

    def test_non_cancel(self):
        assert not is_cancel_command("分析数据")


class TestMissingSlots:
    def test_no_dataset(self):
        # dataset.inspect 需要 dataset_id，无数据集时应报缺 dataset_id。
        assert "dataset_id" in missing_required_slots("dataset.inspect", [])

    def test_has_dataset(self):
        assert missing_required_slots("dataset.inspect", [1]) == []

    def test_empty_tool(self):
        assert missing_required_slots("", []) == []


# --------------------------------------------------------------------------- 信号 → 工具


class TestActionForSignal:
    def test_high_missing_maps_to_clean(self):
        a = action_for_signal(["high_missing"], dataset_id=1)
        assert a is not None and a.tool == "data.clean"
        assert a.arguments["missing"]["strategy"] == "drop"  # 安全默认

    def test_outlier_maps_to_eda(self):
        a = action_for_signal(["outliers_detected"], dataset_id=1)
        assert a is not None and a.tool == "eda.outlier"

    def test_done_tool_skipped(self):
        a = action_for_signal(["high_missing"], dataset_id=1, done_tools={"data.clean"})
        assert a is None

    def test_no_dataset_skipped(self):
        assert action_for_signal(["high_missing"], dataset_id=None) is None

    def test_missing_strategy_override(self):
        a = action_for_signal(["high_missing"], dataset_id=1, missing_strategy="median")
        assert a.arguments["missing"]["strategy"] == "median"


class TestMissingFillStrategy:
    def test_no_fill_intent(self):
        assert missing_fill_strategy("看看数据", None) is None

    def test_median(self):
        assert missing_fill_strategy("用中位数填充", None) == "median"

    def test_from_constraints(self):
        s = TaskState.initial("g", constraints=["不要删列，改成均值填充"])
        assert missing_fill_strategy("继续清洗", s) == "mean"


# --------------------------------------------------------------------------- 九类 playbook


class TestPlaybooks:
    def test_need_dataset(self):
        sel = select_playbook("分析数据", None, [])
        assert sel.name == "need_dataset"
        assert sel.actions == []

    def test_merge(self):
        sel = _sel("把这两张表合并", dataset_ids=[1, 2])
        assert sel.name == "merge"
        assert _tools(sel) == ["dataset.inspect", "data.merge"]
        assert sel.actions[1].arguments["join_type"] == "inner"

    def test_merge_left(self):
        sel = _sel("左连接合并", dataset_ids=[1, 2])
        assert sel.actions[1].arguments["join_type"] == "left"

    def test_workflow(self):
        sel = _sel("用工作流编排")
        assert sel.name == "workflow"
        assert "workflow.build_and_run" in _tools(sel)

    def test_modeling_chain(self):
        sel = _sel("训练一个模型")
        assert sel.name == "modeling"
        assert _tools(sel) == ["dataset.inspect", "dataset.schema", "dataset.profile", "ml.detect_task", "ml.prepare", "ml.train"]

    def test_modeling_with_evaluate(self):
        sel = _sel("训练模型并评估准确率")
        assert "ml.evaluate" in _tools(sel)

    def test_model_hint(self):
        sel = _sel("用回归训练模型")
        train = next(a for a in sel.actions if a.tool == "ml.train")
        assert train.arguments["model"] == "linear_regression"

    def test_transform_clean(self):
        sel = _sel("清洗缺失值")
        assert sel.name == "transform"
        assert _tools(sel) == ["dataset.inspect", "data.clean"]
        assert sel.actions[1].arguments["missing"]["strategy"] == "drop"

    def test_transform_dedupe(self):
        sel = _sel("去重")
        clean = next(a for a in sel.actions if a.tool == "data.clean")
        assert clean.arguments["deduplicate"]["keep"] == "first"

    def test_comprehensive(self):
        sel = _sel("全面分析一下")
        assert sel.name == "comprehensive"
        assert _tools(sel)[:4] == ["dataset.inspect", "dataset.schema", "dataset.quality", "dataset.profile"]

    def test_quality(self):
        sel = _sel("检查数据质量")
        assert sel.name == "quality"
        assert _tools(sel) == ["dataset.inspect", "dataset.quality"]

    def test_eda(self):
        sel = _sel("分析数据分布")
        assert sel.name == "eda"
        assert _tools(sel) == ["dataset.inspect", "eda.describe"]

    def test_eda_correlation(self):
        sel = _sel("分析相关性")
        assert "eda.correlation" in _tools(sel)

    def test_intake_default(self):
        sel = _sel("看看这个数据")
        assert sel.name == "intake"
        assert _tools(sel) == ["dataset.inspect", "dataset.profile"]

    def test_report_tail(self):
        sel = _sel("分析一下并生成报告")
        assert _tools(sel)[-1] == "report.generate"


# --------------------------------------------------------------------------- 去重与引用步号


class TestFilterDone:
    def test_completed_tool_not_requeued(self):
        s = TaskState.initial("g")
        s.completed_actions.append({"tool": "dataset.inspect", "args_key": '{"dataset_id": 1}'})
        sel = _sel("看看这个数据", state=s)
        assert "dataset.inspect" not in _tools(sel)

    def test_pending_tool_not_requeued(self):
        s = TaskState.initial("g")
        s.pending_actions.append(PendingAction(tool="dataset.profile", arguments={"dataset_id": 1}))
        sel = _sel("看看这个数据", state=s)
        assert "dataset.profile" not in _tools(sel)

    def test_modeling_reference_step_number(self):
        # 无已完成步骤时，prepare 引用 detect 的位置（第 4 个动作，1 基 = step4）。
        sel = _sel("训练模型", base_index=0)
        prepare = next(a for a in sel.actions if a.tool == "ml.prepare")
        assert prepare.arguments["target"] == "{{step4.target}}"

    def test_modeling_reference_with_base_index(self):
        # 多轮续跑：base_index 偏移引用步号。
        sel = _sel("训练模型", base_index=5)
        prepare = next(a for a in sel.actions if a.tool == "ml.prepare")
        assert prepare.arguments["target"] == "{{step9.target}}"

    def test_modeling_uses_fact_target(self):
        s = TaskState.initial("g")
        s.facts["ml.detect_task"] = {"target": "price"}
        # detect 已完成，inspect/schema/profile 也已完成 → 只剩 prepare/train。
        s.completed_actions = [
            {"tool": "dataset.inspect", "args_key": ""},
            {"tool": "dataset.schema", "args_key": ""},
            {"tool": "dataset.profile", "args_key": ""},
            {"tool": "ml.detect_task", "args_key": ""},
        ]
        sel = _sel("训练模型", state=s)
        prepare = next(a for a in sel.actions if a.tool == "ml.prepare")
        assert prepare.arguments["target"] == "price"


# --------------------------------------------------------------------------- 源码不变式：零 LLM


class TestNoLLMDependency:
    def test_module_has_no_llm_import(self):
        # 只检查真实 import 语句（docstring 里的历史说明不算依赖）。
        import sys

        banned = ("app.agent.decision", "app.agent.planner", "app.agent.task_spec", "app.agent.llm")
        mod = sys.modules["app.agent.playbooks"]
        for name in banned:
            assert name not in mod.__dict__, f"playbooks.py 不得依赖 {name}"
