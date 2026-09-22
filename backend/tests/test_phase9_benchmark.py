"""Phase 9（Prompt 229-230）性能基准测试。

229: Data Engine Benchmark —— load/preview/filter/aggregate/profile 记录 runtime。
230: Agent Benchmark —— 简单 / 多Tool / 失败重规划 统计
     成功率 / ToolCalls / Steps / Runtime / HumanIntervention。

标记：
- 默认（small）：CI 必跑，< 5 秒
- @pytest.mark.slow：100MB+ 规模，需要 `pytest -m slow` 显式启用
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import polars as pl
import pytest
from app.agent.llm.mock import MockLLM
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401

# slow 标记已在 pyproject.toml 注册；
# 默认 CI 跑 small（10K），大文件基准需 `pytest -m slow` 显式启用。


def _make_df(rows: int, cols: int = 5) -> pl.DataFrame:
    """生成确定性的 DataFrame（rows × cols），便于基准对比。"""
    rng = pl.Series("id", list(range(rows)))
    data = {"id": rng}
    for c in range(cols - 1):
        data[f"col_{c}"] = pl.Series(
            f"col_{c}", [(i * (c + 1)) % 1000 for i in range(rows)], dtype=pl.Float64
        )
    return pl.DataFrame(data)


def _timed(fn):
    """简单计时装饰：返回 (result, seconds)。"""
    start = time.perf_counter()
    result = fn()
    return result, round(time.perf_counter() - start, 4)


# ===========================================================================
# Prompt 229：Data Engine Benchmark
# ===========================================================================
class TestDataEngineBenchmark:
    """记录 load/preview/filter/aggregate/profile 的 runtime。

    CI 跑 small（10K 行），@pytest.mark.slow 启用 large（100K+）。
    """

    @pytest.fixture()
    def engine_env(self, db, storage):
        ds = DatasetService(db, storage)
        dataset = ds.create("bench", "基准测试数据")
        engine = DataEngineService(ds)
        return {"ds": ds, "engine": engine, "dataset_id": dataset.id}

    def _bench_load_and_profile(self, engine_env, rows: int) -> dict[str, Any]:
        ds: DatasetService = engine_env["ds"]
        engine: DataEngineService = engine_env["engine"]
        dataset_id: int = engine_env["dataset_id"]

        df = _make_df(rows)
        _, t_create = _timed(lambda: ds.create_version(dataset_id, df))

        version_row = ds.get_version_row(dataset_id)
        _, t_load = _timed(lambda: ds.load_version(dataset_id, version_row.version))
        loaded = ds.load_version(dataset_id, version_row.version)

        _, t_schema = _timed(lambda: engine.schema(loaded))
        _, t_profile = _timed(lambda: engine.profile(loaded))
        _, t_quality = _timed(lambda: engine.quality(loaded))
        _, t_preview = _timed(lambda: engine.preview(loaded, page_size=10))

        _, t_filter = _timed(lambda: engine.apply_operation(
            loaded, "filter", {"conditions": [
                {"column": "col_0", "op": "gt", "value": 500}
            ]}
        ))

        _, t_aggregate = _timed(lambda: engine.apply_operation(
            loaded, "aggregate",
            {"group_by": ["id"], "aggregations": [
                {"column": "col_0", "func": "mean"},
                {"column": "col_1", "func": "sum"},
            ]},
        ))

        return {
            "rows": rows,
            "create_s": t_create,
            "load_s": t_load,
            "schema_s": t_schema,
            "profile_s": t_profile,
            "quality_s": t_quality,
            "preview_s": t_preview,
            "filter_s": t_filter,
            "aggregate_s": t_aggregate,
        }

    def test_bench_small_10k(self, engine_env, capsys):
        """10K 行基准（CI 必跑）。"""
        result = self._bench_load_and_profile(engine_env, 10_000)
        with capsys.disabled():
            print(f"\n[BENCH small 10K] {result}")
        # 断言：每一步都在合理时间范围内
        assert result["load_s"] < 5.0
        assert result["profile_s"] < 5.0
        assert result["filter_s"] < 5.0
        assert result["aggregate_s"] < 5.0

    @pytest.mark.slow
    def test_bench_medium_100k(self, engine_env, capsys):
        """100K 行基准（需要 -m slow）。"""
        result = self._bench_load_and_profile(engine_env, 100_000)
        with capsys.disabled():
            print(f"\n[BENCH medium 100K] {result}")
        assert result["load_s"] < 30.0
        assert result["profile_s"] < 30.0

    @pytest.mark.slow
    def test_bench_large_500k(self, engine_env, capsys):
        """500K 行基准（需要 -m slow）。"""
        result = self._bench_load_and_profile(engine_env, 500_000)
        with capsys.disabled():
            print(f"\n[BENCH large 500K] {result}")
        assert result["load_s"] < 120.0


# ===========================================================================
# Prompt 230：Agent Benchmark
# ===========================================================================
class TestAgentBenchmark:
    """统计 Agent 在三种场景下的：
    - success_rate
    - tool_calls（平均）
    - steps（平均）
    - runtime（平均）
    - human_intervention（需要确认的次数）
    """

    @pytest.fixture()
    def env(self, db, storage):
        ds = DatasetService(db, storage)
        dataset = ds.create("agent-bench", "Agent 基准数据")
        ds.create_version(dataset.id, pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        }))
        engine = DataEngineService(ds)
        exp = ExperimentService(db, ds)
        return {"ds": ds, "engine": engine, "exp": exp, "dataset_id": dataset.id}

    # ---- 场景 1：简单只读分析（inspect + profile + quality）----
    def test_bench_simple_analysis(self, env, capsys):
        runtime = AgentRuntime(env["engine"])
        runtimes = []
        successes = 0
        total_tool_calls = 0
        total_steps = 0
        interventions = 0

        for _ in range(3):
            session = runtime.create_session(dataset_ids=[env["dataset_id"]])
            start = time.perf_counter()
            run = runtime.run(session, "检查一下数据质量")
            elapsed = time.perf_counter() - start
            runtimes.append(elapsed)
            total_tool_calls += run.tool_call_count
            total_steps += len(run.plan.get("steps", []) if run.plan else [])
            if run.pending_confirmation is not None:
                interventions += 1
            if run.status == RunStatus.COMPLETED:
                successes += 1

        n = 3
        result = {
            "scenario": "simple_analysis",
            "runs": n,
            "success_rate": round(successes / n, 2),
            "avg_tool_calls": round(total_tool_calls / n, 2),
            "avg_steps": round(total_steps / n, 2),
            "avg_runtime_s": round(statistics.mean(runtimes), 4),
            "human_interventions": interventions,
        }
        with capsys.disabled():
            print(f"\n[BENCH agent simple] {result}")
        assert result["success_rate"] == 1.0
        assert result["avg_tool_calls"] > 0
        assert result["human_interventions"] == 0

    # ---- 场景 2：多工具 + 高风险（训练 → 需要确认）----
    def test_bench_multi_tool_with_confirmation(self, env, capsys):
        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"])
        runtimes = []
        successes = 0
        total_tool_calls = 0
        interventions = 0

        for _ in range(3):
            session = runtime.create_session(dataset_ids=[env["dataset_id"]])
            start = time.perf_counter()
            run = runtime.run(session, "帮我训练一个分类模型", confirmed=False)
            elapsed = time.perf_counter() - start
            runtimes.append(elapsed)
            total_tool_calls += run.tool_call_count
            if run.pending_confirmation is not None:
                interventions += 1
                # 确认后继续
                run = runtime.resume(session, run.id)
            if run.status == RunStatus.COMPLETED:
                successes += 1

        n = 3
        result = {
            "scenario": "multi_tool_with_confirmation",
            "runs": n,
            "success_rate": round(successes / n, 2),
            "avg_tool_calls": round(total_tool_calls / n, 2),
            "avg_runtime_s": round(statistics.mean(runtimes), 4),
            "human_interventions": interventions,
        }
        with capsys.disabled():
            print(f"\n[BENCH agent multi-tool] {result}")
        assert result["success_rate"] == 1.0
        assert result["human_interventions"] == n  # 训练都需要确认

    # ---- 场景 3：失败重规划（计划首步失败 → 重试或跳过）----
    def test_bench_failure_replan(self, env, capsys):
        from app.agent.planner.models import AgentPlan, PlanStep

        runtime = AgentRuntime(env["engine"], experiment_service=env["exp"])
        # 计划首步引用不存在的 run_id，必然失败 → Replanner 跳过
        plan = AgentPlan(
            goal="测试失败重规划",
            steps=[
                PlanStep(tool="ml.evaluate", arguments={"run_id": 99999}),
                PlanStep(tool="dataset.inspect", arguments={"dataset_id": env["dataset_id"]}),
            ],
        )

        runtimes = []
        successes = 0
        total_tool_calls = 0
        total_replans = 0

        for _ in range(3):
            session = runtime.create_session(dataset_ids=[env["dataset_id"]])
            start = time.perf_counter()
            run = runtime.run(session, "查看结果", plan_override=plan)
            elapsed = time.perf_counter() - start
            runtimes.append(elapsed)
            total_tool_calls += run.tool_call_count
            total_replans += sum(1 for e in run.events if e.type == "replanning")
            if run.status == RunStatus.COMPLETED:
                successes += 1

        n = 3
        result = {
            "scenario": "failure_replan",
            "runs": n,
            "success_rate": round(successes / n, 2),
            "avg_tool_calls": round(total_tool_calls / n, 2),
            "avg_replans": round(total_replans / n, 2),
            "avg_runtime_s": round(statistics.mean(runtimes), 4),
        }
        with capsys.disabled():
            print(f"\n[BENCH agent replan] {result}")
        # 失败重规划后，第二步 inspect 仍可完成（或最终失败但被记录）
        assert result["avg_replans"] > 0
        assert result["avg_tool_calls"] >= 2  # 至少两次调用（失败 + 后续）

    # ---- 场景 4：LLM 规划对比规则规划 ----
    def test_bench_llm_vs_rule_planner(self, env, capsys):
        from app.agent.planner.planner import AgentPlanner

        # 规划同一个小任务，对比两种 Planner 的步数
        context_builder = env["engine"]
        from app.agent.context.builder import ContextBuilder
        context = ContextBuilder(context_builder).build(
            "分析", dataset_ids=[env["dataset_id"]]
        )
        tools = TOOL_REGISTRY.list()

        rule_planner = AgentPlanner(None)
        rule_plan = rule_planner.build_plan("分析", context, tools)

        mock_llm = MockLLM(structured_responses=[{
            "goal": "inspect dataset",
            "steps": [{
                "tool": "dataset.inspect",
                "arguments": {"dataset_id": env["dataset_id"]},
                "expected_output": "dataset metadata",
                "permission": "read_data",
            }],
        }])
        llm_planner = AgentPlanner(mock_llm)
        llm_plan = llm_planner.build_plan("分析", context, tools)

        result = {
            "rule_plan_steps": len(rule_plan.steps),
            "llm_plan_steps": len(llm_plan.steps),
            "rule_tools": [s.tool for s in rule_plan.steps],
            "llm_tools": [s.tool for s in llm_plan.steps],
        }
        with capsys.disabled():
            print(f"\n[BENCH planner compare] {result}")
        assert result["rule_plan_steps"] > 0
        assert result["llm_plan_steps"] > 0
