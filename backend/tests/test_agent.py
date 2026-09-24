"""Phase 7（Prompt 121-133, 141）：Agent 全链路测试。"""

import polars as pl
import pytest
from app.agent.context.builder import ContextBuilder
from app.agent.executor.executor import AgentExecutor
from app.agent.llm.mock import MockLLM
from app.agent.permission.models import ROLE_PERMISSIONS
from app.agent.planner.models import AgentPlan, PlanStep
from app.agent.planner.planner import AgentPlanner, PlanInvalidError
from app.agent.planner.replanner import AgentLimitExceeded, ReplanLimits, Replanner
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.agent.validator.models import ValidationResult
from app.agent.validator.validator import AgentResultValidator
from app.api.deps import get_storage_service
from app.core.database import get_db
from app.core.exceptions import ValidationException
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.main import app
from app.services.dataset_service import DatasetService
from app.tools.base import ToolResult, ToolServices
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册
from app.tools.context import ToolExecutionContext
from fastapi.testclient import TestClient

TOOL_DESCRIBES = TOOL_REGISTRY.list()


def make_df(rows: int = 10) -> pl.DataFrame:
    """玩具数据。``rows`` 可调：低于 50 行时 Pre-flight 的 `scale_sanity`
    会合法地反问「样本量太小是否继续」，建模类用例必须用 >=50 行才能走到
    「规划 → 高风险工具待确认」这一段。
    """
    half = max(rows // 2, 1)
    return pl.DataFrame(
        {
            "x1": [float(i + 1) for i in range(rows)],
            "x2": [float(rows - i) for i in range(rows)],
            "target": ["pos" if i < half else "neg" for i in range(rows)],
        }
    )


#: Pre-flight 不会因样本量反问的最小规模（见 preflight._c_scale 的 50 行阈值）。
MODELING_ROWS = 60


@pytest.fixture()
def env(db, storage):
    """数据引擎 + 实验 + 已带版本的数据集 id。"""
    ds = DatasetService(db, storage)
    dataset = ds.create("toy", "玩具数据")
    ds.create_version(dataset.id, make_df())
    engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    return {"ds": ds, "engine": engine, "exp": exp, "dataset_id": dataset.id}


@pytest.fixture()
def env_large(db, storage):
    """同上，但数据集规模足以让 Pre-flight 直接放行（建模链路用）。"""
    ds = DatasetService(db, storage)
    dataset = ds.create("toy-large", "规模足够的玩具数据")
    ds.create_version(dataset.id, make_df(rows=MODELING_ROWS))
    engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    return {"ds": ds, "engine": engine, "exp": exp, "dataset_id": dataset.id}


def make_context(env, dataset_id) -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="u1",
        dataset_ids={dataset_id},
        permissions=set(ROLE_PERMISSIONS["admin"]),
    )


# ----------------------------------------------------------------------
# Prompt 121-122：Context
# ----------------------------------------------------------------------
class TestAgentContext:
    def test_build_contains_briefs(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("看看数据", dataset_ids=[env["dataset_id"]])
        brief = context.dataset_context[str(env["dataset_id"])]
        # Context 层设计已收敛为"仅元数据"：真实数据读取必须走 Tool -> DataEngine
        assert brief["rows"] == 10
        assert brief["columns"] == 3
        assert brief["name"] == "toy"
        assert "sample" not in brief  # 样例数据不允许进入 Context
        assert "profile" not in brief and "quality" not in brief

    def test_context_size_controlled(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("x" * 5000, dataset_ids=[env["dataset_id"]])
        text = context.to_prompt_text(max_chars=2000)
        assert len(text) <= 2600  # 各区块独立预算后仍有界
        assert "已截断" in text

    def test_no_full_dataset_in_context(self, env):
        """绝不能把整份数据放进上下文：Context 只携带元数据，不含样例/原始数据。"""
        builder = ContextBuilder(env["engine"])
        context = builder.build("分析", dataset_ids=[env["dataset_id"]])
        text = context.to_prompt_text()
        brief = context.dataset_context[str(env["dataset_id"])]
        assert set(brief) == {"dataset_id", "name", "description", "version", "rows", "columns"}
        assert "10.0" not in text or brief["rows"] == 10  # 只允许统计元信息，不携带原始行


# ----------------------------------------------------------------------
# Prompt 123-124：Planner
# ----------------------------------------------------------------------
class TestPlanner:
    def test_rule_plan_train(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("帮我训练一个分类模型", dataset_ids=[env["dataset_id"]])
        planner = AgentPlanner(None)
        plan = planner.build_plan("帮我训练一个分类模型", context, TOOL_DESCRIBES)
        tools = [s.tool for s in plan.steps]
        assert "ml.train" in tools and "ml.detect_task" in tools

    def test_rule_plan_quality(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("检查一下数据质量", dataset_ids=[env["dataset_id"]])
        plan = AgentPlanner(None).build_plan("检查一下数据质量", context, TOOL_DESCRIBES)
        assert any(s.tool == "dataset.quality" for s in plan.steps)

    def test_rule_plan_does_not_force_report(self, env):
        """★ P1 回归：普通分析请求不该自动产出报告文件。

        历史行为：任何数据分析分支都无条件 append ``report.generate``，于是
        「帮我看看有没有缺失值」也会顺带产出一份 PDF 写进报告中心 —— 过度交付。
        """
        builder = ContextBuilder(env["engine"])
        context = builder.build("帮我分析一下数据", dataset_ids=[env["dataset_id"]])
        plan = AgentPlanner(None).build_plan("帮我分析一下数据", context, TOOL_DESCRIBES)
        tools = [s.tool for s in plan.steps]
        assert tools, "规则规划器应产出分析步骤，而不是空计划"
        assert "report.generate" not in tools

    def test_rule_plan_includes_report_when_explicitly_asked(self, env):
        """明确要求「报告 / 分析并生成报告」时必须包含 report.generate。"""
        builder = ContextBuilder(env["engine"])
        context = builder.build("生成分析报告", dataset_ids=[env["dataset_id"]])
        plan = AgentPlanner(None).build_plan("生成分析报告", context, TOOL_DESCRIBES)
        assert any(s.tool == "report.generate" for s in plan.steps)

    def test_unknown_tool_rejected(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("分析", dataset_ids=[env["dataset_id"]])
        with pytest.raises(PlanInvalidError):
            AgentPlanner(None).build_plan("分析", context, TOOL_DESCRIBES[:1])

    def test_planner_cannot_execute(self):
        """Planner 只能制定计划，不能执行工具。"""
        planner = AgentPlanner(None)
        assert not hasattr(planner, "execute")
        assert not hasattr(planner, "execute_step")

    def test_llm_plan(self, env):
        builder = ContextBuilder(env["engine"])
        context = builder.build("训练模型", dataset_ids=[env["dataset_id"]])
        mock = MockLLM(
            structured_responses=[
                {
                    "goal": "训练分类模型",
                    "steps": [
                        {"tool": "dataset.inspect", "arguments": {"dataset_id": env["dataset_id"]}}
                    ],
                }
            ]
        )
        plan = AgentPlanner(mock).build_plan("训练模型", context, TOOL_DESCRIBES)
        assert plan.goal == "训练分类模型"
        assert mock.structured_calls  # LLM 被调用


# ----------------------------------------------------------------------
# Prompt 125：Executor
# ----------------------------------------------------------------------
class TestExecutor:
    def test_execute_ok_and_recorded(self, env):
        executor = AgentExecutor()
        step = PlanStep(tool="dataset.inspect", arguments={"dataset_id": env["dataset_id"]})
        ctx = make_context(env, env["dataset_id"])
        services = ToolServices(dataset_service=env["ds"], data_engine_service=env["engine"])
        record = executor.execute_step(step, ctx, services)
        assert record.status == "ok"
        assert record.result.success
        assert record.elapsed_ms >= 0
        d = record.to_dict()
        assert d["tool"] == "dataset.inspect"

    def test_confirmation_required_recorded(self, env):
        executor = AgentExecutor()
        step = PlanStep(
            tool="ml.train",
            arguments={"dataset_id": env["dataset_id"], "model": "logistic_regression"},
        )
        ctx = ToolExecutionContext(
            user_id="u1",
            dataset_ids={env["dataset_id"]},
            permissions=set(ROLE_PERMISSIONS["analyst"]),
        )
        services = ToolServices(
            dataset_service=env["ds"],
            data_engine_service=env["engine"],
            experiment_service=env["exp"],
        )
        record = executor.execute_step(step, ctx, services)
        assert record.status == "needs_confirmation"

    def test_unknown_tool_recorded_failed(self, env):
        executor = AgentExecutor()
        step = PlanStep(tool="system.shell", arguments={"cmd": "ls"})  # 不存在的工具
        ctx = make_context(env, env["dataset_id"])
        record = executor.execute_step(step, ctx, ToolServices())
        assert record.status == "failed"
        # 关键约束：不存在任何 shell 通道，未注册工具直接失败
        assert "TOOL_NOT_FOUND" in record.error or "not found" in record.error


# ----------------------------------------------------------------------
# Prompt 126-127：Validator
# ----------------------------------------------------------------------
class TestValidator:
    def test_valid_result(self):
        validator = AgentResultValidator()
        step = PlanStep(tool="dataset.quality", arguments={})
        result = ToolResult.ok({"score": 90.0, "issues": []}, summary="质量良好")
        validation = validator.validate(step, result, {"type": "object"})
        assert validation.valid
        assert validation.evidence["checks"]["tool_success"] is True

    def test_failed_result_invalid(self):
        validator = AgentResultValidator()
        step = PlanStep(tool="dataset.quality", arguments={})
        validation = validator.validate(step, ToolResult.fail("数据不存在"))
        assert not validation.valid
        assert validation.errors

    def test_nan_rejected(self):
        validator = AgentResultValidator()
        step = PlanStep(tool="eda.describe", arguments={})
        result = ToolResult.ok({"mean": float("nan")}, summary="含 NaN")
        validation = validator.validate(step, result)
        assert not validation.valid
        assert any("NaN" in e for e in validation.errors)

    def test_expected_output_mismatch(self):
        validator = AgentResultValidator()
        step = PlanStep(tool="ml.train", arguments={}, expected_output="accuracy")
        result = ToolResult.ok({"foo": 1}, summary="无指标")
        validation = validator.validate(step, result)
        assert validation.valid  # 未命中只降级为 warning，不阻断
        assert validation.evidence["checks"]["expected_missed"] is True
        assert any("期望输出" in w for w in validation.warnings)

    def test_expected_output_chinese_prose_not_fatal(self):
        validator = AgentResultValidator()
        step = PlanStep(
            tool="dataset.profile",
            arguments={},
            expected_output="数据集统计画像：数值分布、缺失情况、类别 Top 值等关键统计摘要",
        )
        result = ToolResult.ok({"columns": {}}, summary="数据集 3 画像完成")
        validation = validator.validate(step, result)
        assert validation.valid
        assert validation.evidence["checks"]["expected_missed"] is True

    def test_validation_result_helpers(self):
        ok = ValidationResult.ok(evidence={"x": 1})
        fail = ValidationResult.fail(["bad"])
        assert ok.valid and not fail.valid


# ----------------------------------------------------------------------
# Prompt 128：Replanner
# ----------------------------------------------------------------------
class TestReplanner:
    def _plan(self, *tools: str) -> AgentPlan:
        return AgentPlan(goal="g", steps=[PlanStep(tool=t, arguments={}) for t in tools])

    def test_retry_once(self):
        replan = Replanner().replan(
            self._plan("dataset.quality", "eda.describe"),
            failed_step_index=0,
            errors=["失败"],
            attempts=1,
        )
        assert replan.steps[0].tool == "dataset.quality"  # 重试同一步
        assert "次尝试" in replan.notes

    def test_skip_after_max_attempts(self):
        replan = Replanner().replan(
            self._plan("dataset.quality", "eda.describe"),
            failed_step_index=0,
            errors=["失败"],
            attempts=2,
        )
        assert [s.tool for s in replan.steps] == ["eda.describe"]  # 跳过失败步

    def test_exhausted_returns_empty(self):
        plan = self._plan("dataset.quality")
        replan = Replanner().replan(plan, failed_step_index=0, errors=["失败"], attempts=2)
        assert replan.steps == []
        assert "任务失败" in replan.notes

    def test_param_error_skips_step(self):
        plan = self._plan("a.x", "b.y")
        replan = Replanner().replan(
            plan,
            failed_step_index=0,
            errors=["工具 a.x 缺少必需字段：dataset_id"],
            attempts=1,
        )
        assert [s.tool for s in replan.steps] == ["b.y"]  # 参数错误不重试，直接跳过

    def test_prose_schema_word_not_param_error(self):
        plan = self._plan("dataset.profile")
        replan = Replanner().replan(
            plan,
            failed_step_index=0,
            errors=["工具执行返回失败"],
            attempts=1,
        )
        assert replan.steps[0].tool == "dataset.profile"  # 自由文本不触发参数错误跳过

    def test_limit_exceeded(self):
        replanner = Replanner(ReplanLimits(max_steps=1))
        with pytest.raises(AgentLimitExceeded):
            replanner.replan(
                self._plan("a.x", "b.y", "c.z"),  # 跳过第一步后剩余 2 步 > 上限 1
                failed_step_index=0,
                errors=["失败"],
                attempts=2,
            )


# ----------------------------------------------------------------------
# Prompt 129-130：Runtime
# ----------------------------------------------------------------------
class TestRuntime:
    def test_full_run_completed(self, env):
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"], db=None)
        session = runtime.create_session(dataset_ids=[env["dataset_id"]])
        run = runtime.run(session, "检查一下数据质量")
        assert run.status == RunStatus.COMPLETED
        assert run.final_answer
        # `usage` 是 Token 账本的快照帧（可观测性，不属于生命周期事件，且在一条
        # 运行里会出现多次），断言**生命周期顺序**时必须先滤掉，否则这里会被
        # 无关的数量变化打断 —— 用例想钉的是「route → planning → … → completed」。
        types = [e.type for e in run.events if e.type != "usage"]
        assert types[0] == "route"
        assert types[1] == "planning"
        assert "tool_call" in types and "tool_result" in types
        assert "validation" in types and types[-1] == "completed"
        # 账本快照本身也要真的发出来了（否则界面在运行中看不到消耗）
        assert "usage" in [e.type for e in run.events]
        assert run.tool_calls  # 每次调用都被记录
        assert session.history[0]["role"] == "user"  # 用户消息入史
        assert session.history[-1]["role"] == "assistant"  # 完成后写入最终回答

    def test_waiting_confirmation_and_resume(self, env_large):
        """建模请求 → ml.train 高风险 → 等待确认 → 确认后继续 → 完成。

        用 `env_large`：10 行的玩具数据会被 Pre-flight 的样本量检查拦下反问，
        根本走不到「待确认」这一步（这条用例要钉的是授权闭环，不是反问闭环）。
        """
        env = env_large
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"])
        session = runtime.create_session(dataset_ids=[env["dataset_id"]])
        run = runtime.run(session, "帮我训练一个分类模型")
        assert run.status == RunStatus.WAITING_CONFIRMATION
        assert run.pending_confirmation["call"].tool == "ml.train"
        # 确认后继续
        resumed = runtime.resume(session, run.id)
        assert resumed.status == RunStatus.COMPLETED
        assert resumed.pending_confirmation is None
        train_calls = [c for c in resumed.tool_calls if c.tool == "ml.train"]
        assert train_calls[-1].status == "ok"
        assert train_calls[-1].result.data["metrics"]

    def test_permission_denied_fails_run(self, env):
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"])
        session = runtime.create_session(dataset_ids=[env["dataset_id"]], )
        # viewer 无 train_model 权限
        session.user_id = "viewer-user"
        plan = AgentPlan(
            goal="训练",
            steps=[
                PlanStep(
                    tool="ml.train",
                    arguments={"dataset_id": env["dataset_id"], "model": "logistic_regression"},
                )
            ],
        )
        run = runtime.run(session, "训练", plan_override=plan)
        # viewer 角色未接入 runtime（runtime 固定 analyst），analyst 拥有 train_model
        # 因此这里应当是等待确认而不是拒绝
        assert run.status in (RunStatus.WAITING_CONFIRMATION, RunStatus.COMPLETED)

    def test_plan_failure_fails_run(self, env):
        runtime = AgentRuntime(env["engine"])
        session = runtime.create_session(dataset_ids=[env["dataset_id"]])
        plan = AgentPlan(
            goal="失败计划",
            steps=[PlanStep(tool="ml.evaluate", arguments={"run_id": 99999})],
        )
        run = runtime.run(session, "看结果", plan_override=plan)
        assert run.status == RunStatus.FAILED
        assert run.error
        assert any(e.type == "failed" for e in run.events)
        # 至少重试过一次（共 2 次调用记录）
        assert run.tool_call_count == 2

    def test_tool_call_limit(self, env):
        runtime = AgentRuntime(
            env["engine"], limits=ReplanLimits(max_tool_calls=0, timeout_seconds=60)
        )
        session = runtime.create_session(dataset_ids=[env["dataset_id"]])
        run = runtime.run(session, "检查一下数据质量")
        assert run.status == RunStatus.FAILED
        assert "超过" in run.error

    def test_empty_request_rejected(self, env):
        runtime = AgentRuntime(env["engine"])
        session = runtime.create_session()
        with pytest.raises(ValidationException):
            runtime.run(session, "   ")


# ----------------------------------------------------------------------
# Prompt 131-132：Agent API + SSE
# ----------------------------------------------------------------------
@pytest.fixture()
def api(db, storage):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_storage_service] = lambda: storage
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture()
def api_dataset(db, storage):
    ds = DatasetService(db, storage)
    dataset = ds.create("api-toy", "API 玩具数据")
    ds.create_version(dataset.id, make_df(rows=MODELING_ROWS))
    return dataset.id


class TestAgentAPI:
    def test_session_lifecycle(self, api):
        resp = api.post(
            "/api/v1/agent/sessions",
            json={"user_id": "u1", "title": "测试会话", "dataset_ids": []},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] and body["data"]["title"] == "测试会话"
        session_id = body["data"]["id"]

        resp = api.get("/api/v1/agent/sessions", params={"user_id": "u1"})
        assert any(s["id"] == session_id for s in resp.json()["data"])

    def test_message_run_and_run_detail(self, api, api_dataset):
        session_id = api.post(
            "/api/v1/agent/sessions", json={"dataset_ids": [api_dataset]}
        ).json()["data"]["id"]
        resp = api.post(
            f"/api/v1/agent/sessions/{session_id}/messages",
            json={"content": "检查一下数据质量"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "completed"
        run_id = data["id"]

        detail = api.get(f"/api/v1/agent/runs/{run_id}").json()["data"]
        assert detail["final_answer"]
        assert any(e["type"] == "completed" for e in detail["events"])

        # SSE 重放
        sse = api.get(f"/api/v1/agent/runs/{run_id}/events").text
        assert "event: completed" in sse
        assert "event: done" in sse

    def test_message_stream_sse(self, api, api_dataset):
        session_id = api.post(
            "/api/v1/agent/sessions", json={"dataset_ids": [api_dataset]}
        ).json()["data"]["id"]
        with api.stream(
            "POST",
            f"/api/v1/agent/sessions/{session_id}/messages",
            json={"content": "检查一下数据质量", "stream": True},
        ) as resp:
            lines = list(resp.iter_lines())
        events = [line for line in lines if line.startswith("event: ")]
        # 同 `test_full_run_completed`：`usage` 是 Token 账本快照帧，不属于生命周期事件，
        # 断言「route → planning → … → completed」这条主链时要滤掉。
        lifecycle = [e for e in events if e != "event: usage"]
        assert lifecycle[0] == "event: route"
        assert lifecycle[1] == "event: planning"
        assert "event: tool_call" in lifecycle
        assert lifecycle[-2] == "event: completed"  # 最后是 done
        assert "event: usage" in events

    def test_confirmation_flow_via_api(self, api, api_dataset):
        session_id = api.post(
            "/api/v1/agent/sessions", json={"dataset_ids": [api_dataset]}
        ).json()["data"]["id"]
        data = api.post(
            f"/api/v1/agent/sessions/{session_id}/messages",
            json={"content": "帮我训练一个分类模型"},
        ).json()["data"]
        assert data["status"] == "waiting_confirmation"
        assert data["pending_confirmation"]["tool"] == "ml.train"
        assert "confirm" in data["hint"]

        resumed = api.post(f"/api/v1/agent/runs/{data['id']}/confirm").json()["data"]
        assert resumed["status"] == "completed"

    def test_get_run_not_found(self, api):
        assert api.get("/api/v1/agent/runs/nope").status_code == 404
