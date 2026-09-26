"""AI 全链条基准实验：对照实验 + 消融实验。

用途
----
为毕业论文的「实验与结果分析」提供**可复现、可量化**的数据，回答一个问题：
AI 全链条式接入（规划 → 工具检索 → 授权 → 执行 → 校验 → 重规划 → 报告叙述）
到底带来了什么可度量的提升？

设计
----
- **对照组**：`rule`（llm=None，规则规划器，即「无 AI 规划」） vs `full`（LLM 规划 + 全特性）
- **消融组**：在 `full` 基础上逐项关闭一项能力
  - `no_retrieval`   关工具检索（AGENT_ENABLE_TOOL_RETRIEVAL=False）
  - `no_compression` 关结果压缩（AGENT_ENABLE_RESULT_COMPRESSION=False）
  - `no_narration`   关报告 LLM 叙述（AGENT_REPORT_NARRATION=False）
- **任务集**：6 个覆盖全链条的固定任务（质量 → EDA → 建模 → 合并 → 报告 → 工作流），
  每个任务声明「期望工具集合」，用于计算客观的**任务完成度 coverage**。

指标
----
每次运行采集：终态、耗时、计划步数、工具调用数与成功率、重规划次数、授权拦截次数、
LLM 调用次数、输入/输出/总 token、估算节省 token、计划缓存命中、最终答案长度、
任务完成度、报告章节/图表/结论数。

用法
----
    # 快速自检（不消耗 token，只跑规则规划器）
    python -m scripts.experiments.agent_chain_benchmark --quick

    # 完整对照（会真实调用 .env 中的 LLM，消耗 token）
    python -m scripts.experiments.agent_chain_benchmark --configs rule full

    # 消融
    python -m scripts.experiments.agent_chain_benchmark \
        --configs full no_retrieval no_compression no_narration

产物
----
    scripts/experiments/results/agent_chain_<timestamp>.json   原始逐次数据
    scripts/experiments/results/agent_chain_<timestamp>.md     可直接贴论文的汇总表
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 注册全部 ORM 模型到 Base.metadata
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.operation  # noqa: F401
import polars as pl
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.agent.runtime.agent_runtime import AgentRuntime
from app.agent.runtime.models import AgentStore, RunStatus
from app.core.config import settings
from app.core.database import Base
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"

SEP = "-" * 78
SECTION = "=" * 78

# 需要授权的高风险工具触发 WAITING_CONFIRMATION 时，基准实验自动放行（模拟用户点确认）
MAX_AUTO_CONFIRMS = 3


# ============================================================
# 任务集：覆盖 AI 全链条的 6 个固定任务
# ============================================================
# expect 中的 "a|b" 表示二者之一命中即算该环节被覆盖
TASKS: list[dict[str, Any]] = [
    {
        "id": "T1_quality",
        "name": "数据质量检查",
        "datasets": ["orders"],
        "request": "检查 orders 数据集的数据质量，指出缺失值、重复行和异常值的情况",
        "expect": ["dataset.quality|dataset.profile"],
    },
    {
        "id": "T2_eda",
        "name": "描述统计与相关性",
        "datasets": ["orders"],
        "request": "对 orders 做描述统计与相关性分析，说明哪些字段关系密切",
        "expect": ["eda.describe", "eda.correlation"],
    },
    {
        "id": "T3_ml",
        "name": "建模与评估",
        "datasets": ["orders"],
        "request": "用 orders 训练一个预测 amount 的模型并评估效果",
        "expect": ["ml.detect_task|ml.prepare", "ml.train", "ml.evaluate"],
    },
    {
        "id": "T4_merge",
        "name": "多表关联",
        "datasets": ["users", "orders"],
        "request": "把 users 和 orders 按用户关联起来，生成一张宽表",
        "expect": ["data.merge"],
    },
    {
        "id": "T5_report",
        "name": "分析报告生成",
        "datasets": ["orders"],
        "request": "基于 orders 生成一份完整的数据分析报告",
        "expect": ["report.generate"],
    },
    {
        "id": "T6_workflow",
        "name": "工作流编排执行",
        "datasets": ["orders"],
        "request": "编排一个工作流：读取数据、做质量检查、做统计分析、生成报告，然后执行它",
        "expect": ["workflow.build_and_run|workflow.create"],
    },
]


# ============================================================
# 配置组（对照 + 消融）
# ============================================================
CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "rule": {"__no_llm__": True},
    "no_retrieval": {"AGENT_ENABLE_TOOL_RETRIEVAL": False},
    "no_compression": {"AGENT_ENABLE_RESULT_COMPRESSION": False},
    "no_narration": {"AGENT_REPORT_NARRATION": False},
}

# ---------------------------------------------------------------------------
# 语义正确性检查（比「有没有调工具」更硬的指标）
# ---------------------------------------------------------------------------
# coverage 只回答「环节有没有走到」，无法回答「做对了没有」。最典型的反例：
# 用户说「预测 amount」，规则规划器推断不出目标列 -> 判为聚类 -> 链路照样跑完、coverage 100%，
# 但结果根本不是用户要的东西。下面这些检查只对**有明确事实断言**的任务生效，
# 判定依据是工具真实返回值，不接受 LLM 自评。
SEMANTIC_CHECKS: dict[str, Any] = {
    "T3_ml": lambda f: str(f.get("ml_target") or "").lower() == "amount",
    "T4_merge": lambda f: any(
        "user" in str(k.get("left", "")).lower() and "user" in str(k.get("right", "")).lower()
        for k in (f.get("merge_keys") or [])
    ),
}

CONFIG_LABEL = {
    "full": "LLM 规划 + 全特性",
    "rule": "规则规划（无 AI 对照组）",
    "no_retrieval": "消融：关闭工具检索",
    "no_compression": "消融：关闭结果压缩",
    "no_narration": "消融：关闭报告 LLM 叙述",
}


# ============================================================
# 环境
# ============================================================
def setup_env(tmp_root: Path) -> dict[str, Any]:
    """内存 SQLite + 临时存储，与 learning_cases.py 同一套隔离方式。"""
    engine_db = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db, autoflush=False, expire_on_commit=False)
    db = Session()
    storage = StorageService(LocalStorage(root=tmp_root / "storage"))
    ds = DatasetService(db, storage)
    data_engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    return {"db": db, "storage": storage, "ds": ds, "engine": data_engine, "exp": exp, "tmp": tmp_root}


def load_demo_datasets(env: dict[str, Any]) -> dict[str, int]:
    """载入 data/demo 下的真实演示数据（含缺失值，便于质量类任务有真实结论）。"""
    ids: dict[str, int] = {}
    for name in ("users", "orders", "events"):
        path = DEMO_DIR / f"{name}.csv"
        if not path.exists():
            print(f"  ! 跳过 {name}：{path} 不存在")
            continue
        df = pl.read_csv(path, infer_schema_length=200, try_parse_dates=True)
        dataset = env["ds"].create(name, f"基准实验用数据集 {name}")
        env["ds"].create_version(dataset.id, df)
        ids[name] = dataset.id
        print(f"  · 载入 {name}：{df.height} 行 × {df.width} 列 → dataset_id={dataset.id}")
    return ids


def build_llm(use_llm: bool) -> Any | None:
    if not use_llm:
        return None
    if not settings.LLM_API_KEY:
        print("  ! 未配置 LLM_API_KEY，回退为规则规划器")
        return None
    return OpenAICompatibleProvider(settings.LLM_BASE_URL, settings.LLM_MODEL, settings.LLM_API_KEY)


# ============================================================
# 单次运行与指标采集
# ============================================================
def run_once(env: dict[str, Any], task: dict[str, Any], use_llm: bool, store: AgentStore) -> dict[str, Any]:
    dataset_ids = [env["dataset_ids"][name] for name in task["datasets"] if name in env["dataset_ids"]]
    runtime = AgentRuntime(
        env["engine"],
        experiment_service=env["exp"],
        db=env["db"],
        llm=build_llm(use_llm),
        store=store,
    )
    session = runtime.create_session(title=f"benchmark-{task['id']}", dataset_ids=dataset_ids)

    started = time.perf_counter()
    try:
        run = runtime.run(session, task["request"])
    except Exception as exc:  # noqa: BLE001 - 单次失败不应中断整个基准
        return {
            "task": task["id"], "status": "error", "error": str(exc)[:300],
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    confirms = 0
    while run.status == RunStatus.WAITING_CONFIRMATION and confirms < MAX_AUTO_CONFIRMS:
        try:
            run = runtime.resume(session, run.id)
        except Exception:  # noqa: BLE001
            break
        confirms += 1
    elapsed = round(time.perf_counter() - started, 2)

    tools = [c.tool for c in run.tool_calls]
    ok = sum(1 for c in run.tool_calls if c.status == "ok")
    # 口径说明：授权拦截（needs_confirmation）与未开始（pending）是中间态，不是执行失败。
    # 高危工具被拦截后由 resume 重新执行同一步，会在 tool_calls 里留下两条记录
    # （一条 needs_confirmation + 一条 ok），把它们算进失败会系统性低估成功率。
    blocked = sum(1 for c in run.tool_calls if c.status in {"needs_confirmation", "pending"})
    settled = len(run.tool_calls) - blocked
    failed = sum(1 for c in run.tool_calls if c.status not in {"ok", "pending", "needs_confirmation"})

    # 任务完成度：期望工具组（"a|b" 任一命中即算该组覆盖）
    groups = [str(g).split("|") for g in task["expect"]]
    hit = sum(1 for group in groups if any(t in tools for t in group))
    coverage = round(hit / max(len(groups), 1), 3)

    ledger = run.token_ledger

    # 事实采集：语义正确性判定的依据只能是工具真实返回值
    facts: dict[str, Any] = {"ml_target": None, "ml_task": None, "merge_keys": [], "model_adjusted": False}
    for call in run.tool_calls:
        data = getattr(getattr(call, "result", None), "data", None)
        if not isinstance(data, dict):
            continue
        if call.tool == "ml.detect_task" and data.get("target") is not None:
            facts["ml_target"] = data.get("target")
            facts["ml_task"] = data.get("task")
        if call.tool == "ml.train" and data.get("model_adjusted"):
            facts["model_adjusted"] = True
        plan = data.get("plan") if call.tool == "data.merge" else None
        if isinstance(plan, dict) and plan.get("keys"):
            facts["merge_keys"] = plan["keys"]
    check = SEMANTIC_CHECKS.get(task["id"])
    semantic = None if check is None else bool(check(facts))

    report_calls = [c for c in run.tool_calls if c.tool == "report.generate" and c.result is not None]
    report_charts = 0
    report_narrated = False
    report_sections = 0
    for call in report_calls:
        meta = getattr(call.result, "metadata", None) or {}
        report_charts = max(report_charts, int(meta.get("chart_count") or 0))
        compact = getattr(call.result, "compact_data", None) or {}
        if isinstance(compact, dict):
            report_sections = max(report_sections, len(compact.get("sections") or []))
            report_narrated = report_narrated or bool(compact.get("narrated"))

    return {
        "task": task["id"],
        "task_name": task["name"],
        "status": str(run.status),
        "error": run.error or "",
        "elapsed_s": elapsed,
        "plan_steps": len((run.plan or {}).get("steps", [])),
        "tool_calls": len(run.tool_calls),
        "tool_ok": ok,
        "tool_failed": failed,
        "tool_blocked": blocked,
        "tool_success_rate": round(ok / settled, 3) if settled else 0.0,
        "replans": sum(1 for e in run.events if e.type == "replanning"),
        "permission_prompts": sum(1 for e in run.events if e.type == "permission"),
        "auto_confirms": confirms,
        "llm_calls": ledger.llm_calls,
        "input_tokens": ledger.actual_input_tokens,
        "output_tokens": ledger.actual_output_tokens,
        "total_tokens": ledger.actual_total_tokens,
        "est_saved_tokens": ledger.estimated_context_saved_tokens + ledger.estimated_result_saved_tokens,
        "plan_cache_hits": ledger.cache_hits,
        "answer_chars": len(run.final_answer or ""),
        "coverage": coverage,
        "semantic_ok": semantic,
        "facts": facts,
        "tools_used": tools,
        "report_charts": report_charts,
        "report_sections": report_sections,
        "report_narrated": report_narrated,
    }


# ============================================================
# 汇总与输出
# ============================================================
def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(r.get(key) or 0) for r in rows]
    return round(statistics.mean(values), 2) if values else 0.0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    out = []
    for config, items in grouped.items():
        done = [r for r in items if r.get("status") not in {"error"}]
        out.append({
            "config": config,
            "label": CONFIG_LABEL.get(config, config),
            "runs": len(items),
            "completed": sum(1 for r in items if r.get("status") == "runstatus.completed" or r.get("status") == "completed"),
            "coverage": round(statistics.mean([r.get("coverage", 0) for r in done]), 3) if done else 0,
            # 语义正确率：只在声明了事实断言的任务上统计，分母是这些任务的条数
            "semantic_accuracy": (
                round(
                    sum(1 for r in done if r.get("semantic_ok"))
                    / sum(1 for r in done if r.get("semantic_ok") is not None),
                    3,
                )
                if any(r.get("semantic_ok") is not None for r in done)
                else None
            ),
            "tool_success_rate": round(statistics.mean([r.get("tool_success_rate", 0) for r in done]), 3) if done else 0,
            "avg_tool_calls": _mean(done, "tool_calls"),
            "avg_plan_steps": _mean(done, "plan_steps"),
            "avg_replans": _mean(done, "replans"),
            "avg_elapsed_s": _mean(done, "elapsed_s"),
            "avg_llm_calls": _mean(done, "llm_calls"),
            "avg_total_tokens": _mean(done, "total_tokens"),
            "avg_saved_tokens": _mean(done, "est_saved_tokens"),
            "avg_answer_chars": _mean(done, "answer_chars"),
            "report_charts": max([r.get("report_charts", 0) for r in done], default=0),
            "narrated_reports": sum(1 for r in done if r.get("report_narrated")),
        })
    return out


def render_markdown(summary: list[dict[str, Any]], rows: list[dict[str, Any]], stamp: str) -> str:
    section = 0

    def heading(title: str) -> str:
        nonlocal section
        section += 1
        return f"## {section}. {title}"

    lines = [
        f"# AI 全链条接入 · 基准实验结果（{stamp}）",
        "",
        heading("总体对照"),
        "",
        "| 配置 | 说明 | 任务完成度 | **语义正确率** | 工具调用成功率 | 平均工具调用 | 平均耗时(s) | LLM 调用 | 总 token（真实） | 估算节省 token＊ | 平均答案字数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        sem = f"{s['semantic_accuracy']:.0%}" if s.get("semantic_accuracy") is not None else "—"
        lines.append(
            f"| `{s['config']}` | {s['label']} | {s['coverage']:.0%} | {sem} | {s['tool_success_rate']:.0%} | "
            f"{s['avg_tool_calls']} | {s['avg_elapsed_s']} | {s['avg_llm_calls']} | {s['avg_total_tokens']} | "
            f"{s['avg_saved_tokens']} | {s['avg_answer_chars']} |"
        )

    lines += ["", heading("分任务明细"), "", "| 配置 | 任务 | 终态 | 完成度 | 语义正确 | 工具调用 | 成功 | 失败 | 授权拦截 | 重规划 | 耗时(s) | token | 使用工具 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        tools = "、".join((r.get("tools_used") or [])[:8]) or "-"
        sem = "—" if r.get("semantic_ok") is None else ("✓" if r.get("semantic_ok") else "✗")
        lines.append(
            f"| `{r['config']}` | {r['task']} {r.get('task_name','')} | {r.get('status','')} | {r.get('coverage',0):.0%} | {sem} | "
            f"{r.get('tool_calls',0)} | {r.get('tool_ok',0)} | {r.get('tool_failed',0)} | {r.get('tool_blocked',0)} | "
            f"{r.get('replans',0)} | {r.get('elapsed_s',0)} | {r.get('total_tokens',0)} | {tools} |"
        )

    base = next((s for s in summary if s["config"] == "full"), None)
    if base:
        lines += ["", heading("消融对比（相对 `full`）"), "", "| 配置 | 关闭的能力 | 完成度 Δ | 耗时 Δ(s) | token Δ | 说明 |", "| --- | --- | --- | --- | --- | --- |"]
        for s in summary:
            if s["config"] == "full":
                continue
            lines.append(
                f"| `{s['config']}` | {s['label']} | {s['coverage'] - base['coverage']:+.0%} | "
                f"{s['avg_elapsed_s'] - base['avg_elapsed_s']:+.1f} | "
                f"{s['avg_total_tokens'] - base['avg_total_tokens']:+.0f} | "
                f"{'对照组（无 AI 规划）' if s['config'] == 'rule' else '消融项'} |"
            )
    lines += [
        "",
        heading("指标口径（写进论文时必须保留）"),
        "",
        "- **任务完成度 coverage** = 命中的期望工具环节数 / 该任务的期望环节总数。"
        "期望环节由任务定义时人工声明（`TASKS[*].expect`，`a|b` 表示任一命中即算覆盖），"
        "不依赖 LLM 自评，可逐条复核。",
        "- **语义正确率** = 通过事实断言的任务数 / 声明了断言的任务数（当前为 T3、T4）。"
        "断言直接读工具返回值：T3 要求 `ml.detect_task` 识别出的目标列确实是 `amount`；"
        "T4 要求 `data.merge` 实际使用的 Join Key 两侧都是用户 id。"
        "**这一项才是「AI 有没有真的理解需求」的证据**——只靠 coverage 看不出对照组把"
        "「预测 amount」做成了无监督聚类（链路照跑完，结果不是用户要的）。",
        "- **工具调用成功率** = `ok / (全部调用 − 授权拦截中间态)`。"
        "高危工具被拦截时会留下一条 `needs_confirmation` 记录，用户确认后同一步重新执行并留下一条 `ok` 记录；"
        "把中间态计入失败会系统性低估成功率。",
        "- **总 token** = LLM Provider 返回的 `prompt_tokens + completion_tokens` 真实累计（未发生调用即为 0）。",
        "- **⚠️ 估算节省 token**：来自 `app/tools/result.py` 的 `for_llm()`，是按**字符数本地粗估**"
        "（`_rough_tokens(原始结果) − _rough_tokens(压缩后结果)`），**与是否发生真实 LLM 调用无关**——"
        "对照组 `llm_calls=0` 时该值依然为正。它反映的是「上下文压缩机制少喂了多少字符给模型」，"
        "**不是 Provider 计费口径的节省**，论文中必须标注为本地估算值，不得与总 token 直接相减或相除。",
        "",
        f"> 本次实验环境：模型 `{settings.LLM_MODEL}`，数据集 `data/demo`（users / orders / events）。",
    ]
    return "\n".join(lines)


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="AI 全链条基准实验（对照 + 消融）")
    parser.add_argument("--configs", nargs="*", default=None, help=f"配置组，可选：{list(CONFIGS)}")
    parser.add_argument("--tasks", nargs="*", default=None, help=f"任务 id，可选：{[t['id'] for t in TASKS]}")
    parser.add_argument("--repeat", type=int, default=1, help="每个任务重复次数（>1 可观察计划缓存效果）")
    parser.add_argument("--quick", action="store_true", help="只跑规则规划器，不消耗 LLM token")
    args = parser.parse_args()

    configs = args.configs or (["rule"] if args.quick else ["rule", "full"])
    tasks = [t for t in TASKS if (args.tasks is None or t["id"] in args.tasks)]

    print(SECTION)
    print("小洛实验室 · AI 全链条基准实验")
    print(SECTION)
    print(f"配置组：{configs}")
    print(f"任务数：{len(tasks)}，每组重复 {args.repeat} 次")
    print(SEP)

    tmp_root = Path(tempfile.mkdtemp(prefix="xl-benchmark-"))
    env = setup_env(tmp_root)
    print("载入演示数据集：")
    env["dataset_ids"] = load_demo_datasets(env)
    print(SEP)

    rows: list[dict[str, Any]] = []
    for config in configs:
        if config not in CONFIGS:
            print(f"! 未知配置组 {config}，跳过")
            continue
        overrides = dict(CONFIGS[config])
        use_llm = not overrides.pop("__no_llm__", False)
        # 无 AI 对照组与 --quick 自检一律关闭报告 LLM 叙述：
        # report.generate 内部走的是 settings 里的 Provider（与 planner 的 llm 不是同一个开关），
        # 不关掉会在「不消耗 token」的场景下偷偷调用真实 LLM。
        if args.quick or not use_llm:
            overrides.setdefault("AGENT_REPORT_NARRATION", False)
        saved = {k: getattr(settings, k) for k in overrides}
        for k, v in overrides.items():
            setattr(settings, k, v)
        try:
            store = AgentStore(path=tmp_root / f"agent_store_{config}.json")
            for rep in range(args.repeat):
                for task in tasks:
                    print(f"▶ [{config}] {task['id']} {task['name']}" + (f" (#{rep+1})" if args.repeat > 1 else ""))
                    row = run_once(env, task, use_llm, store)
                    row["config"] = config
                    row["repeat"] = rep + 1
                    rows.append(row)
                    print(f"    status={row.get('status')} 完成度={row.get('coverage',0):.0%} "
                          f"工具={row.get('tool_calls',0)} 耗时={row.get('elapsed_s',0)}s "
                          f"token={row.get('total_tokens',0)}")
        finally:
            for k, v in saved.items():
                setattr(settings, k, v)

    print(SECTION)
    print("汇总")
    print(SECTION)
    summary = aggregate(rows)
    for s in summary:
        sem = f"  语义正确率 {s['semantic_accuracy']:.0%}" if s.get("semantic_accuracy") is not None else ""
        print(f"  {s['config']:<16} 完成度 {s['coverage']:.0%}{sem}  工具成功率 {s['tool_success_rate']:.0%}  "
              f"平均工具 {s['avg_tool_calls']}  耗时 {s['avg_elapsed_s']}s  token {s['avg_total_tokens']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"agent_chain_{stamp}.json"
    md_path = RESULTS_DIR / f"agent_chain_{stamp}.md"
    json_path.write_text(json.dumps({"summary": summary, "runs": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(summary, rows, stamp), encoding="utf-8")
    print(f"\n✓ 原始数据：{json_path}")
    print(f"✓ 汇总表格：{md_path}")


if __name__ == "__main__":
    main()
