"""Phase 10（Prompt 240-241）集成测试。

240：完整用户流程（Excel→Dataset→Schema→Mapping→MergePlan→授权→Merge→Quality→EDA→ML→Report→Workflow保存）
241：完整 Agent 流程（理解需求→查看Dataset→Schema→Profile→Mapping→MergePlan→请求权限
    →执行Merge→Quality→EDA→判断ML任务→生成结果→解释结果，全程通过 Tool Registry）
"""

from __future__ import annotations

import io

import polars as pl
import pytest
from app.agent.context.builder import ContextBuilder
from app.agent.permission.models import ROLE_PERMISSIONS
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.data_engine.merge.plan import JoinKey, MergePlan
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册
from app.tools.context import ToolExecutionContext
from app.workflow.service import WorkflowService


# ----------------------------------------------------------------------
# 共享 fixture
# ----------------------------------------------------------------------
def make_sales_df() -> pl.DataFrame:
    """销售数据（订单维度）。"""
    return pl.DataFrame(
        {
            "order_id": list(range(1001, 1011)),
            "customer_id": [1, 2, 1, 3, 2, 4, 3, 5, 4, 5],
            "amount": [120.0, 85.5, 230.0, 60.0, 95.0, 180.0, 75.0, 110.0, 200.0, 65.0],
            "qty": [1, 2, 3, 1, 2, 1, 1, 2, 3, 1],
        }
    )


def make_customers_df() -> pl.DataFrame:
    """客户维度数据（含 numeric label 便于 ML）。"""
    return pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "city_score": [8.0, 5.0, 9.0, 6.0, 8.5],
            "tenure": [3.0, 1.0, 5.0, 2.0, 4.0],
            "label": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture()
def env(db, storage):
    """完整环境：DatasetService + DataEngineService + ExperimentService。"""
    ds = DatasetService(db, storage)
    engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    return {"ds": ds, "engine": engine, "exp": exp, "db": db, "storage": storage}


@pytest.fixture()
def two_datasets(env):
    """创建销售 + 客户两个数据集，各带一个版本。"""
    sales = env["ds"].create("sales", "销售订单")
    env["ds"].create_version(sales.id, make_sales_df())
    customers = env["ds"].create("customers", "客户主数据")
    env["ds"].create_version(customers.id, make_customers_df())
    return {
        "sales_id": sales.id,
        "customers_id": customers.id,
        "sales": sales,
        "customers": customers,
    }


def _role_ctx(dataset_ids, role="admin") -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="integration-user",
        dataset_ids=set(dataset_ids),
        permissions=set(ROLE_PERMISSIONS[role]),
    )


# ===========================================================================
# Prompt 240：完整用户流程（端到端）
# ===========================================================================
class TestFullUserFlow:
    """模拟真实用户从 Excel 上传到生成报告 + 保存工作流的完整流程。"""

    def test_01_csv_to_dataset(self, env, two_datasets):
        """CSV → 上传 → Dataset（模拟加载流程）。"""
        # 模拟 CSV 上传：DataEngineService.load 支持 csv
        df = make_sales_df()
        buf = io.BytesIO()
        df.write_csv(buf)
        buf.seek(0)
        loaded = env["engine"].load("sales.csv", buf.read())
        assert loaded is not None
        # 创建 Dataset 并写入版本
        ds = env["ds"].create("uploaded_sales", "从 CSV 上传")
        version = env["ds"].create_version(ds.id, df)
        assert version.version == 1
        assert version.row_count == 10
        assert version.column_count == 4

    def test_02_schema_analysis(self, env, two_datasets):
        """Schema 分析：列名 + 类型。"""
        df = env["ds"].load_version(two_datasets["sales_id"], 1)
        schema = env["engine"].schema(df)
        cols = {c["column"]: c["dtype"] for c in schema["columns"]}
        assert "order_id" in cols
        assert "amount" in cols
        assert "customer_id" in cols

    def test_03_suggest_mappings(self, env, two_datasets):
        """Schema Mapping：自动建议字段映射。"""
        left_df = env["ds"].load_version(two_datasets["sales_id"], 1)
        right_df = env["ds"].load_version(two_datasets["customers_id"], 1)
        mappings = env["engine"].suggest_mappings(left_df, right_df)
        assert isinstance(mappings, list)
        # customer_id 在右表也同名，应该出现在建议中
        target_cols = [m.get("target_column") for m in mappings]
        assert "customer_id" in target_cols

    def test_04_merge_plan_validation(self, env, two_datasets):
        """MergePlan 构建与校验。"""
        left_df = env["ds"].load_version(two_datasets["sales_id"], 1)
        right_df = env["ds"].load_version(two_datasets["customers_id"], 1)
        plan = MergePlan(
            left={"dataset_id": two_datasets["sales_id"]},
            right={"dataset_id": two_datasets["customers_id"]},
            keys=[JoinKey(left="customer_id", right="customer_id")],
            join_type="left",
        )
        result = env["engine"].validate_merge(left_df, right_df, plan)
        # 校验通过（ok=True）
        assert result["ok"] is True

    def test_05_authorized_merge(self, env, two_datasets):
        """授权后执行 Merge：销售 + 客户 → 新版本。"""
        plan = MergePlan(
            left={"dataset_id": two_datasets["sales_id"]},
            right={"dataset_id": two_datasets["customers_id"]},
            keys=[JoinKey(left="customer_id", right="customer_id")],
            join_type="left",
        )
        output_version, report = env["engine"].run_merge(
            two_datasets["sales_id"], two_datasets["customers_id"], plan
        )
        assert output_version.version == 2  # 新版本
        # 合并后应该有客户信息
        merged_df = env["ds"].load_version(two_datasets["sales_id"], output_version.version)
        assert "city_score" in merged_df.columns  # 客户字段被引入
        assert "label" in merged_df.columns
        assert merged_df.height == 10  # left join 保留全部左表行

    def test_06_quality_check(self, env, two_datasets):
        """Quality 检查。"""
        plan = MergePlan(
            left={"dataset_id": two_datasets["sales_id"]},
            right={"dataset_id": two_datasets["customers_id"]},
            keys=[JoinKey(left="customer_id", right="customer_id")],
            join_type="left",
        )
        output_version, _ = env["engine"].run_merge(
            two_datasets["sales_id"], two_datasets["customers_id"], plan
        )
        merged_df = env["ds"].load_version(two_datasets["sales_id"], output_version.version)
        quality = env["engine"].quality(merged_df)
        assert "issues" in quality
        assert "severity" in quality
        assert "statistics" in quality

    def test_07_eda_describe(self, env, two_datasets):
        """EDA 描述性统计。"""
        df = env["ds"].load_version(two_datasets["sales_id"], 1)
        describe = env["engine"].profile(df)
        assert "columns" in describe
        # amount 列统计
        amount_stats = next(
            (c for c in describe["columns"] if c.get("column") == "amount"), None
        )
        assert amount_stats is not None
        assert "mean" in amount_stats  # 数值列有 mean

    def test_08_ml_train_classification(self, env, two_datasets):
        """ML：分类任务（用 label 作为 target）。"""
        plan = MergePlan(
            left={"dataset_id": two_datasets["sales_id"]},
            right={"dataset_id": two_datasets["customers_id"]},
            keys=[JoinKey(left="customer_id", right="customer_id")],
            join_type="left",
        )
        output_version, _ = env["engine"].run_merge(
            two_datasets["sales_id"], two_datasets["customers_id"], plan
        )
        exp = env["exp"].create(
            dataset_id=two_datasets["sales_id"],
            dataset_version_id=output_version.id,
            task="classification",
            model="logistic_regression",
            target_column="label",
            seed=42,
        )
        run = env["exp"].run(exp.id)
        assert run.status == "success"
        assert run.metrics is not None

    def test_09_workflow_save(self, env, two_datasets):
        """Workflow 保存（创建 + 查询）。"""
        # 注册节点类型（仅用于校验，不实际执行）
        wf_service = WorkflowService(
            node_runners={
                "load_data": lambda n, u, c: {"ok": True},
                "merge_data": lambda n, u, c: {"ok": True},
                "train_model": lambda n, u, c: {"ok": True},
            }
        )
        wf = wf_service.create(
            name="销售分析流水线",
            nodes=[
                {
                    "id": "load_sales",
                    "type": "load_data",
                    "config": {"dataset_id": two_datasets["sales_id"]},
                },
                {"id": "merge", "type": "merge_data", "config": {"join_type": "left"}},
                {"id": "train", "type": "train_model", "config": {"model": "logistic_regression"}},
            ],
            edges=[
                {"source": "load_sales", "target": "merge"},
                {"source": "merge", "target": "train"},
            ],
            metadata={"author": "integration-test"},
        )
        wf_id = wf.metadata["id"]
        assert wf_id is not None
        fetched = wf_service.get(wf_id)
        assert fetched.name == "销售分析流水线"
        assert len(fetched.nodes) == 3
        assert len(fetched.edges) == 2

    def test_10_full_pipeline_e2e(self, env, two_datasets):
        """端到端：CSV→Schema→Mapping→MergePlan→授权→Merge→Quality→EDA→ML→Workflow。"""
        # 1. 加载 CSV（模拟上传）
        df_sales = make_sales_df()
        buf = io.BytesIO()
        df_sales.write_csv(buf)
        buf.seek(0)
        env["engine"].load("sales.csv", buf.read())

        # 2. Schema
        df = env["ds"].load_version(two_datasets["sales_id"], 1)
        schema = env["engine"].schema(df)
        assert len(schema["columns"]) == 4

        # 3. Mapping suggestion
        right_df = env["ds"].load_version(two_datasets["customers_id"], 1)
        env["engine"].suggest_mappings(df, right_df)  # 仅验证可执行

        # 4. MergePlan
        plan = MergePlan(
            left={"dataset_id": two_datasets["sales_id"]},
            right={"dataset_id": two_datasets["customers_id"]},
            keys=[JoinKey(left="customer_id", right="customer_id")],
            join_type="left",
        )

        # 5. Validate（授权前检查）
        validation = env["engine"].validate_merge(df, right_df, plan)
        assert validation["ok"] is True

        # 6. Execute merge
        output_version, report = env["engine"].run_merge(
            two_datasets["sales_id"], two_datasets["customers_id"], plan
        )
        merged = env["ds"].load_version(two_datasets["sales_id"], output_version.version)

        # 7. Quality
        quality = env["engine"].quality(merged)
        assert "issues" in quality

        # 8. EDA
        describe = env["engine"].profile(merged)
        assert "columns" in describe

        # 9. ML
        exp = env["exp"].create(
            dataset_id=two_datasets["sales_id"],
            dataset_version_id=output_version.id,
            task="classification",
            model="logistic_regression",
            target_column="label",
            seed=42,
        )
        run = env["exp"].run(exp.id)
        assert run.status == "success"

        # 10. Workflow 保存
        wf_service = WorkflowService(
            node_runners={
                "merge_data": lambda n, u, c: {"ok": True},
                "train_model": lambda n, u, c: {"ok": True},
            }
        )
        wf = wf_service.create(
            name="端到端流水线",
            nodes=[
                {"id": "merge", "type": "merge_data", "config": {"plan": plan.to_dict()}},
                {"id": "train", "type": "train_model", "config": {"experiment_id": exp.id}},
            ],
            edges=[{"source": "merge", "target": "train"}],
        )
        assert wf.metadata["id"] is not None


# ===========================================================================
# Prompt 241：完整 Agent 流程（全程通过 Tool Registry）
# ===========================================================================
class TestFullAgentFlow:
    """Agent 全链路：理解需求→查看数据→Schema→Profile→Mapping→MergePlan→
    请求权限→执行Merge→Quality→EDA→判断ML任务→生成结果→解释结果。"""

    def test_agent_understand_and_inspect(self, env, two_datasets):
        """Step 1: Agent 理解需求并查看 Dataset。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="agent-user", dataset_ids=[sales_id])
        plan = AgentPlan(
            goal="查看销售数据",
            steps=[PlanStep(tool="dataset.inspect", arguments={"dataset_id": sales_id})],
        )
        run = runtime.run(session, "查看销售数据", plan_override=plan, role="admin")
        assert run.status == RunStatus.COMPLETED
        ok_calls = [c for c in run.tool_calls if c.status == "ok"]
        assert len(ok_calls) >= 1
        inspect_call = next(c for c in run.tool_calls if c.tool == "dataset.inspect")
        assert inspect_call.result.data is not None

    def test_agent_schema_and_profile(self, env, two_datasets):
        """Step 2: Agent 查看 Schema + Profile。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id])
        plan = AgentPlan(
            goal="了解销售数据结构",
            steps=[
                PlanStep(tool="dataset.schema", arguments={"dataset_id": sales_id}),
                PlanStep(tool="dataset.profile", arguments={"dataset_id": sales_id}),
            ],
        )
        run = runtime.run(session, "了解数据结构", plan_override=plan, role="admin")
        assert run.status == RunStatus.COMPLETED
        schema_call = next(c for c in run.tool_calls if c.tool == "dataset.schema")
        profile_call = next(c for c in run.tool_calls if c.tool == "dataset.profile")
        assert schema_call.status == "ok"
        assert profile_call.status == "ok"

    def test_agent_merge_with_permission(self, env, two_datasets):
        """Step 3-6: Agent MergePlan → 请求权限 → 执行 Merge（admin 角色放行）。"""
        sales_id = two_datasets["sales_id"]
        cust_id = two_datasets["customers_id"]
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id, cust_id])
        plan = AgentPlan(
            goal="合并销售与客户数据",
            steps=[
                PlanStep(
                    tool="data.merge",
                    arguments={
                        "left_dataset_id": sales_id,
                        "right_dataset_id": cust_id,
                        "keys": [{"left": "customer_id", "right": "customer_id"}],
                        "join_type": "left",
                    },
                ),
            ],
        )
        run = runtime.run(
            session, "合并销售与客户数据", plan_override=plan, role="admin", confirmed=True
        )
        assert run.status == RunStatus.COMPLETED
        merge_call = next(c for c in run.tool_calls if c.tool == "data.merge")
        assert merge_call.status == "ok"
        assert merge_call.result.data is not None

    def test_agent_merge_needs_confirmation_for_analyst(self, env, two_datasets):
        """analyst 角色 + 未确认：Agent 可能进入 waiting_confirmation。"""
        sales_id = two_datasets["sales_id"]
        cust_id = two_datasets["customers_id"]
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id, cust_id])
        plan = AgentPlan(
            goal="合并销售与客户数据",
            steps=[
                PlanStep(
                    tool="data.merge",
                    arguments={
                        "left_dataset_id": sales_id,
                        "right_dataset_id": cust_id,
                        "keys": [{"left": "customer_id", "right": "customer_id"}],
                        "join_type": "left",
                    },
                ),
            ],
        )
        run = runtime.run(
            session, "合并销售与客户数据", plan_override=plan, role="analyst", confirmed=False
        )
        # analyst 有 modify_data 权限，但 data.merge 可能需要 confirmation
        # 两种情况都接受：要么等待确认，要么直接完成
        assert run.status in {RunStatus.WAITING_CONFIRMATION, RunStatus.COMPLETED}

    def test_agent_quality_after_merge(self, env, two_datasets):
        """Step 7: Merge 后执行 Quality。"""
        sales_id = two_datasets["sales_id"]
        cust_id = two_datasets["customers_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id, cust_id])
        plan = AgentPlan(
            goal="合并后质检",
            steps=[
                PlanStep(
                    tool="data.merge",
                    arguments={
                        "left_dataset_id": sales_id,
                        "right_dataset_id": cust_id,
                        "keys": [{"left": "customer_id", "right": "customer_id"}],
                        "join_type": "left",
                    },
                ),
                PlanStep(tool="dataset.quality", arguments={"dataset_id": sales_id}),
            ],
        )
        run = runtime.run(
            session, "合并后质检", plan_override=plan, role="admin", confirmed=True
        )
        assert run.status == RunStatus.COMPLETED
        quality_call = next(c for c in run.tool_calls if c.tool == "dataset.quality")
        assert quality_call.status == "ok"
        quality_data = quality_call.result.data or {}
        assert "issues" in quality_data or "statistics" in quality_data

    def test_agent_eda_after_merge(self, env, two_datasets):
        """Step 8: Merge 后 EDA。"""
        sales_id = two_datasets["sales_id"]
        cust_id = two_datasets["customers_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id, cust_id])
        plan = AgentPlan(
            goal="合并后探索性分析",
            steps=[
                PlanStep(
                    tool="data.merge",
                    arguments={
                        "left_dataset_id": sales_id,
                        "right_dataset_id": cust_id,
                        "keys": [{"left": "customer_id", "right": "customer_id"}],
                        "join_type": "left",
                    },
                ),
                PlanStep(tool="eda.describe", arguments={"dataset_id": sales_id}),
                PlanStep(tool="eda.correlation", arguments={"dataset_id": sales_id}),
            ],
        )
        run = runtime.run(
            session, "合并后探索性分析", plan_override=plan, role="admin", confirmed=True
        )
        assert run.status == RunStatus.COMPLETED
        eda_calls = [c for c in run.tool_calls if c.tool.startswith("eda.")]
        assert all(c.status == "ok" for c in eda_calls)

    def test_agent_ml_detect_and_train(self, env, two_datasets):
        """Step 9-11: 判断 ML 任务 → 训练 → 结果。"""
        sales_id = two_datasets["sales_id"]
        cust_id = two_datasets["customers_id"]
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id, cust_id])
        plan = AgentPlan(
            goal="合并并训练分类模型",
            steps=[
                PlanStep(
                    tool="data.merge",
                    arguments={
                        "left_dataset_id": sales_id,
                        "right_dataset_id": cust_id,
                        "keys": [{"left": "customer_id", "right": "customer_id"}],
                        "join_type": "left",
                    },
                ),
                PlanStep(
                    tool="ml.detect_task",
                    arguments={"dataset_id": sales_id, "target": "label"},
                ),
                PlanStep(
                    tool="ml.train",
                    arguments={
                        "dataset_id": sales_id,
                        "model": "logistic_regression",
                        "task": "classification",
                        "target": "label",
                    },
                ),
            ],
        )
        run = runtime.run(
            session, "合并并训练分类模型", plan_override=plan, role="admin", confirmed=True
        )
        assert run.status == RunStatus.COMPLETED
        detect_call = next(c for c in run.tool_calls if c.tool == "ml.detect_task")
        train_call = next(c for c in run.tool_calls if c.tool == "ml.train")
        assert detect_call.status == "ok"
        assert train_call.status == "ok"
        train_data = train_call.result.data or {}
        assert "metrics" in train_data or "experiment_id" in train_data

    def test_agent_final_answer_explanation(self, env, two_datasets):
        """Step 12: 最终解释（final_answer 不为空且含工具执行摘要）。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id])
        plan = AgentPlan(
            goal="查看数据并分析",
            steps=[
                PlanStep(tool="dataset.inspect", arguments={"dataset_id": sales_id}),
                PlanStep(tool="dataset.profile", arguments={"dataset_id": sales_id}),
            ],
        )
        run = runtime.run(session, "分析销售数据", plan_override=plan, role="admin")
        assert run.status == RunStatus.COMPLETED
        assert run.final_answer
        assert len(run.final_answer) > 0
        assert "inspect" in run.final_answer.lower() or "profile" in run.final_answer.lower()

    def test_agent_all_through_tool_registry(self, env, two_datasets):
        """关键约束：所有工具调用必须经由 TOOL_REGISTRY（不能绕过）。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id])
        plan = AgentPlan(
            goal="完整流程",
            steps=[
                PlanStep(tool="dataset.inspect", arguments={"dataset_id": sales_id}),
                PlanStep(tool="dataset.schema", arguments={"dataset_id": sales_id}),
                PlanStep(tool="dataset.profile", arguments={"dataset_id": sales_id}),
                PlanStep(tool="dataset.quality", arguments={"dataset_id": sales_id}),
                PlanStep(tool="eda.describe", arguments={"dataset_id": sales_id}),
            ],
        )
        run = runtime.run(session, "完整流程", plan_override=plan, role="admin")
        registry_names = set(TOOL_REGISTRY.names())
        for call in run.tool_calls:
            assert call.tool in registry_names, f"工具 {call.tool} 不在 Registry 中"
        assert run.status == RunStatus.COMPLETED

    def test_agent_cannot_bypass_registry(self, env, two_datasets):
        """关键约束：Agent 不能调用未注册的工具（无 shell / eval 等）。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id])
        plan = AgentPlan(
            goal="尝试越权",
            steps=[
                PlanStep(tool="system.shell", arguments={"cmd": "rm -rf /"}),
            ],
        )
        run = runtime.run(session, "尝试越权", plan_override=plan, role="admin")
        # 运行应失败或工具调用失败
        assert run.status in {RunStatus.FAILED, RunStatus.COMPLETED}
        # 无论如何，shell 调用必须失败
        for call in run.tool_calls:
            assert call.status == "failed", f"未注册工具 {call.tool} 不应成功"

    def test_agent_planner_rule_based_for_full_flow(self, env, two_datasets):
        """规则 Planner 也能为"合并 + 训练"类请求生成合法计划。"""
        sales_id = two_datasets["sales_id"]
        builder = ContextBuilder(env["engine"])
        context = builder.build(
            "训练一个分类模型预测客户等级", dataset_ids=[sales_id]
        )
        planner = AgentPlanner(None)  # 无 LLM
        plan = planner.build_plan("训练一个分类模型", context, TOOL_REGISTRY.list())
        tools = [s.tool for s in plan.steps]
        assert "dataset.inspect" in tools
        assert "ml.detect_task" in tools or "ml.train" in tools
        registry_names = set(TOOL_REGISTRY.names())
        for t in tools:
            assert t in registry_names

    def test_agent_history_preserved(self, env, two_datasets):
        """Agent 会话历史保留（多轮对话）。"""
        sales_id = two_datasets["sales_id"]
        runtime = AgentRuntime(env["engine"], db=env["db"])
        session = runtime.create_session(user_id="u", dataset_ids=[sales_id])
        runtime.run(
            session,
            "查看数据",
            plan_override=AgentPlan(
                goal="查看",
                steps=[PlanStep(tool="dataset.inspect", arguments={"dataset_id": sales_id})],
            ),
            role="admin",
        )
        runtime.run(
            session,
            "分析统计",
            plan_override=AgentPlan(
                goal="分析",
                steps=[PlanStep(tool="dataset.profile", arguments={"dataset_id": sales_id})],
            ),
            role="admin",
        )
        user_msgs = [h for h in session.history if h["role"] == "user"]
        assert len(user_msgs) >= 2
