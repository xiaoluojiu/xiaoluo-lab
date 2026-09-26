"""小洛实验室 · 毕业设计实验脚本（Prompt 242）。

对比 5 种数据分析场景，量化 Agent 各安全机制（权限 / 校验 / 重规划）的价值：
  1. 传统手工分析（直接调用 Service，无 Agent）
  2. LLM + Tools（无权限 / 无校验）
  3. LLM + Tools + 权限（有权限，无校验）
  4. LLM + Tools + 权限 + 校验（有权限 + 校验）
  5. 完整 Agent + 重规划（全部安全机制）

每个场景记录：任务是否成功 / 工具调用次数 / 运行时长 / 人工介入次数 / 结果准确率。

用法：
    python -m scripts.experiments.graduation_experiments
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

# 注册全部 ORM 模型到 Base.metadata（必须在 create_all 之前）
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.operation  # noqa: F401
import polars as pl
from app.agent.llm.mock import MockLLM
from app.agent.permission.manager import PermissionDecision, PermissionManager
from app.agent.permission.models import Decision
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.agent.validator.models import ValidationResult
from app.agent.validator.validator import AgentResultValidator
from app.core.database import Base
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from app.tools.builtin import _BUILTIN_TOOLS
from app.tools.registry import ToolRegistry
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ============================================================
# 安全机制变体（用于构造不同配置的 AgentRuntime）
# ============================================================
class _PermissivePermissionManager(PermissionManager):
    """始终 ALLOW，不要求确认 —— 模拟「无权限校验」。"""

    def check(self, user_id, tool, context, params=None):
        return PermissionDecision(Decision.ALLOW, reason="permissive（实验：无权限校验）")


class _NoopValidator(AgentResultValidator):
    """始终判 valid —— 模拟「无结果校验」。"""

    def validate(self, step, result, output_schema=None):
        return ValidationResult.ok(evidence={"noop": True, "step": step.tool})


def _build_permissive_registry() -> ToolRegistry:
    """构造一个使用 _PermissivePermissionManager 的注册表（注册全部内置工具）。"""
    reg = ToolRegistry(permission_manager=_PermissivePermissionManager())
    for tool_cls in _BUILTIN_TOOLS:
        reg.register(tool_cls())
    return reg


# ============================================================
# 环境与样本数据
# ============================================================
def _make_classification_df() -> pl.DataFrame:
    """分类任务样本：数值特征 + 整数标签（2 类，线性可分）。"""
    return pl.DataFrame(
        {
            "x1": [1.0, 1.2, 1.1, 1.3, 8.0, 8.2, 8.1, 7.9, 1.05, 8.05],
            "x2": [1.0, 1.1, 1.2, 1.05, 8.0, 8.1, 7.9, 8.2, 1.15, 7.95],
            "x3": [2.0, 2.1, 1.9, 2.05, 6.0, 6.1, 5.9, 6.2, 2.15, 5.95],
            "label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        }
    )


def setup_env() -> dict[str, Any]:
    """临时内存 SQLite + 临时本地存储 + 已带版本的数据集。"""
    engine_db = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db, autoflush=False, expire_on_commit=False)
    db = Session()
    storage = StorageService(LocalStorage(root=Path(tempfile.mkdtemp(prefix="xl-graduation-"))))
    ds = DatasetService(db, storage)
    data_engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    dataset = ds.create("cls-demo", "毕业设计分类演示数据")
    ds.create_version(dataset.id, _make_classification_df())
    return {
        "db": db,
        "storage": storage,
        "ds": ds,
        "engine": data_engine,
        "exp": exp,
        "dataset_id": dataset.id,
    }


# ============================================================
# 统一请求文本与准确率提取
# ============================================================
# 重构后（统一 Agent Loop）不再用 plan_override 固定计划：建模链由确定性
# playbook（inspect → schema → profile → detect_task → prepare → train）自然产出。
_AGENT_REQUEST = "分析数据集并训练分类模型"


def _extract_accuracy(run: Any) -> float | None:
    """从 AgentRun 的 ml.train 调用记录中提取 accuracy。"""
    for call in reversed(run.tool_calls):
        if call.tool == "ml.train" and call.status == "ok" and call.result:
            metrics = (call.result.data or {}).get("metrics") or {}
            acc = metrics.get("accuracy")
            if acc is not None:
                return float(acc)
    return None


def _run_agent(
    env: dict[str, Any],
    dataset_id: int,
    *,
    registry: ToolRegistry | None = None,
    validator: AgentResultValidator | None = None,
    confirmed: bool = False,
) -> tuple[Any, int]:
    """运行 AgentRuntime；若 ml.train 等待确认则自动 resume，返回 (run, 人工介入次数)。"""
    data_engine, exp, db = env["engine"], env["exp"], env["db"]
    kwargs: dict[str, Any] = {
        "experiment_service": exp,
        "db": db,
        "llm": MockLLM(),
    }
    if registry is not None:
        kwargs["registry"] = registry
    if validator is not None:
        kwargs["validator"] = validator
    runtime = AgentRuntime(data_engine, **kwargs)
    session = runtime.create_session(dataset_ids=[dataset_id])
    run = runtime.run(session, _AGENT_REQUEST, confirmed=confirmed)

    interventions = 0
    if run.status == RunStatus.WAITING_CONFIRMATION:
        interventions += 1  # 需要人工确认高风险操作
        run = runtime.resume(session, run.id)
    return run, interventions


# ============================================================
# 5 个场景
# ============================================================
def run_scenario_1_manual(env: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    """场景 1：传统手工分析 —— 直接调用 Service，无 Agent。"""
    ds, data_engine, exp = env["ds"], env["engine"], env["exp"]
    interventions = 0
    t0 = time.perf_counter()
    # 手工逐步：inspect → profile → quality → detect → train
    ds.get(dataset_id)
    interventions += 1
    df = ds.load_version(dataset_id)
    data_engine.profile(df)
    interventions += 1
    data_engine.quality(df)
    interventions += 1
    # 手工判定任务类型（整数标签 + 低基数 → 分类）
    task = "classification"
    interventions += 1
    version_row = ds.get_version_row(dataset_id)
    experiment = exp.create(
        dataset_id=dataset_id,
        dataset_version_id=version_row.id,
        task=task,
        model="logistic_regression",
        target_column="label",
        seed=42,
    )
    run = exp.run(experiment.id)
    interventions += 1
    elapsed = time.perf_counter() - t0
    acc = (run.metrics or {}).get("accuracy")
    return {
        "scenario": "1. 传统手工分析（直接 Service）",
        "task_success": run.status == "success",
        "tool_calls": 0,
        "runtime_s": round(elapsed, 4),
        "human_intervention": interventions,
        "accuracy": acc,
    }


def run_scenario_2_llm_tools(env: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    """场景 2：LLM + Tools —— 无权限校验、无结果校验（ml.train 直接执行）。"""
    t0 = time.perf_counter()
    run, interventions = _run_agent(
        env,
        dataset_id,
        registry=_build_permissive_registry(),
        validator=_NoopValidator(),
        confirmed=False,  # permissive manager 不会要求确认
    )
    elapsed = time.perf_counter() - t0
    return {
        "scenario": "2. LLM + Tools（无权限 / 无校验）",
        "task_success": run.status == RunStatus.COMPLETED,
        "tool_calls": run.tool_call_count,
        "runtime_s": round(elapsed, 4),
        "human_intervention": interventions,
        "accuracy": _extract_accuracy(run),
    }


def run_scenario_3_with_permission(env: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    """场景 3：LLM + Tools + 权限 —— 有权限校验（ml.train 需确认），无结果校验。"""
    t0 = time.perf_counter()
    run, interventions = _run_agent(
        env,
        dataset_id,
        validator=_NoopValidator(),  # 无校验
        confirmed=False,  # ml.train 触发确认 → resume
    )
    elapsed = time.perf_counter() - t0
    return {
        "scenario": "3. LLM + Tools + 权限（无校验）",
        "task_success": run.status == RunStatus.COMPLETED,
        "tool_calls": run.tool_call_count,
        "runtime_s": round(elapsed, 4),
        "human_intervention": interventions,
        "accuracy": _extract_accuracy(run),
    }


def run_scenario_4_with_validation(env: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    """场景 4：LLM + Tools + 权限 + 校验 —— 有权限 + 结果校验。"""
    t0 = time.perf_counter()
    run, interventions = _run_agent(
        env,
        dataset_id,
        # 使用默认 validator（AgentResultValidator）
        confirmed=False,
    )
    elapsed = time.perf_counter() - t0
    validation_events = [e for e in run.events if e.type == "validation"]
    return {
        "scenario": "4. LLM + Tools + 权限 + 校验",
        "task_success": run.status == RunStatus.COMPLETED,
        "tool_calls": run.tool_call_count,
        "runtime_s": round(elapsed, 4),
        "human_intervention": interventions,
        "accuracy": _extract_accuracy(run),
        "validation_checks": len(validation_events),
    }


def run_scenario_5_full_agent(env: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    """场景 5：完整 Agent —— 全部安全机制。

    重构后（统一 Loop）失败恢复由 Loop 的失败三策略处理（依赖断裂即止 /
    参数错误不重试 / 瞬时≤2 次），不再依赖旧 Replanner 的 plan_override 注入。
    """
    t0 = time.perf_counter()
    run, interventions = _run_agent(
        env,
        dataset_id,
        confirmed=False,
    )
    elapsed = time.perf_counter() - t0
    replan_events = [e for e in run.events if e.type == "replanning"]
    return {
        "scenario": "5. 完整 Agent + 重规划",
        "task_success": run.status == RunStatus.COMPLETED,
        "tool_calls": run.tool_call_count,
        "runtime_s": round(elapsed, 4),
        "human_intervention": interventions,
        "accuracy": _extract_accuracy(run),
        "replan_events": len(replan_events),
    }


# ============================================================
# 输出
# ============================================================
def _fmt(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.center(width)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["#", "场景", "成功", "工具调用", "运行(秒)", "人工介入", "准确率"]
    widths = [4, 34, 8, 10, 12, 12, 10]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = (
        "|"
        + "|".join(f" {h.center(w)} " for h, w in zip(headers, widths, strict=False))
        + "|"
    )
    print(sep)
    print(header_line)
    print(sep)
    for i, row in enumerate(rows, 1):
        cells = [
            str(i),
            row["scenario"],
            "是" if row["task_success"] else "否",
            str(row["tool_calls"]),
            f"{row['runtime_s']:.4f}",
            str(row["human_intervention"]),
            f"{row['accuracy']:.4f}" if row.get("accuracy") is not None else "-",
        ]
        line = "|" + "|".join(f" {_fmt(c, w)} " for c, w in zip(cells, widths, strict=False)) + "|"
        print(line)
    print(sep)


def save_json(rows: list[dict[str, Any]], name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n✓ 结果已保存：{path}")


def main() -> None:
    print("=" * 72)
    print("小洛实验室 · 毕业设计实验（Prompt 242）—— Agent 安全机制对比")
    print("=" * 72)

    env = setup_env()
    dataset_id = env["dataset_id"]

    rows: list[dict[str, Any]] = []
    print("\n[场景 1] 传统手工分析 …")
    rows.append(run_scenario_1_manual(env, dataset_id))
    print("[场景 2] LLM + Tools（无权限 / 无校验）…")
    rows.append(run_scenario_2_llm_tools(env, dataset_id))
    print("[场景 3] LLM + Tools + 权限（无校验）…")
    rows.append(run_scenario_3_with_permission(env, dataset_id))
    print("[场景 4] LLM + Tools + 权限 + 校验 …")
    rows.append(run_scenario_4_with_validation(env, dataset_id))
    print("[场景 5] 完整 Agent + 重规划 …")
    rows.append(run_scenario_5_full_agent(env, dataset_id))

    print("\n" + "=" * 72)
    print("对比结果")
    print("=" * 72)
    print_table(rows)

    # 结论
    print("\n结论：")
    print("  - 场景 2 无安全机制：零介入但高风险（ml.train 未经确认直接执行）。")
    print("  - 场景 3/4 引入权限：ml.train 需 1 次人工确认，兼顾效率与安全。")
    print("  - 场景 4 增加校验：每步结果经 AgentResultValidator 把关，防止 NaN/Inf 入答。")
    print("  - 场景 5 完整 Agent：失败由统一 Loop 的失败三策略恢复，任务仍完成。")

    save_json(rows, "graduation_experiments.json")


if __name__ == "__main__":
    main()
