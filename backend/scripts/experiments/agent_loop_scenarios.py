"""Agent 统一 Loop 14 场景离线评测脚本（重构前后同脚本复跑）。

设计约束
--------
* 只通过 AgentRuntime 的**稳定公共 API** 驱动：create_session / run / resume /
  answer_clarification / get_run / get_session。不引用任何内部规划/决策类。
* 远程 LLM 使用脚本内自带的 CountingLLM（可按场景脚本化 chat / structured_output），
  自报确定性 pseudo-token（按字符数线性折算，重构前后口径一致，只用于相对比较）。
* 三档环境由场景 env 字段标记：mock（Mock 远程 + 仅词法本地路由）、
  no_remote（不注入远程）、qwen_unavailable（Qwen 层缺席，词法兜底）。
* 断言只记录 pass/fail，不影响脚本退出码（基线期允许断言失败，失败本身即证据）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe -m scripts.experiments.agent_loop_scenarios \\
        --out ..\\.trae\\specs\\agent-unified-loop\\baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# 必须先导入全部 ORM 模型再 create_all（与 tests/conftest.py 同口径）。
import app.models.connector  # noqa: F401
import app.models.dataset  # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.experiment  # noqa: F401
import app.models.experiment_run  # noqa: F401
import app.models.file  # noqa: F401
import app.models.learning  # noqa: F401
import app.models.operation  # noqa: F401

from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, LLMResponse
from app.agent.runtime.models import AgentStore, RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.core.config import settings
from app.data_engine.cache import get_version_frame_cache
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.storage.local import LocalStorage
from app.storage.service import StorageService

# ---------------------------------------------------------------------------
# 全局环境：Qwen 权重层缺席（LOCAL_ROUTER_MODE=off），本地决策只走 TF-IDF。
# 与线上默认口径一致；在 import runtime 前设置，避免 .env 的 shadow 档干扰。
# ---------------------------------------------------------------------------
settings.LOCAL_ROUTER_MODE = "off"
settings.AGENT_ENABLE_PLAN_CACHE = False  # 场景间不共享计划缓存，保证计数可复现

#: 每行唯一的原始数据标记：紧凑视图不应把整列原始值塞进远程载荷。
RAW_MARKER = "raw_note_%03d"

#: 只读工具集合（用于断言「追问不重复跑已完成的只读步骤」）。
READONLY_TOOLS = {
    "dataset.inspect", "dataset.schema", "dataset.profile", "dataset.quality",
    "dataset.preview", "eda.describe", "eda.correlation", "eda.distribution",
    "eda.distribution_overview", "eda.outlier", "eda.visualize",
}


# ---------------------------------------------------------------------------
# 计数 Mock LLM
# ---------------------------------------------------------------------------
class CountingLLM(LLMProvider):
    """按场景脚本化的远程 LLM 替身，精确记录每次调用的载荷与 pseudo-token。"""

    name = "counting-mock"

    def __init__(
        self,
        *,
        chat_answers: dict[str, str] | None = None,
        structured_plans: list[dict[str, Any]] | None = None,
        structured_decisions: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._chat_answers = chat_answers or {}
        self._plans = list(structured_plans or [])
        self._decisions = list(structured_decisions or [])
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _pseudo_tokens(text: str) -> int:
        # 中文语料下字符数与 token 近似线性；固定折算率保证前后口径一致。
        return max(1, math.ceil(len(text or "") / 2))

    def _log(self, kind: str, messages: list[LLMMessage], schema: str | None, output: str) -> dict[str, Any]:
        entry = {
            "kind": kind,
            "schema": schema,
            "input_chars": sum(len(m.content or "") for m in messages),
            "output_chars": len(output or ""),
            "messages": [{"role": m.role, "content": m.content or ""} for m in messages],
            "output": output,
        }
        entry["input_tokens"] = self._pseudo_tokens(
            "\n".join(m.content or "" for m in messages)
        )
        entry["output_tokens"] = self._pseudo_tokens(output)
        self.calls.append(entry)
        return entry

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs) -> LLMResponse:
        self.begin_call()
        blob = "\n".join(m.content or "" for m in messages)
        if "战略" in blob:
            kind, default = "guidance", "建议先读取数据概况与统计画像，基于真实结果再决定后续分析。"
        elif "数据分析助手" in blob:
            kind, default = "summary", "【mock 总结】基于真实工具结果，分析已完成，关键指标以工具输出为准。"
        else:
            kind, default = "chat", "【mock 对话】你好，我是小洛实验室助手，可以帮你做数据分析。"
        content = self._chat_answers.get(kind, default)
        entry = self._log("chat", messages, None, content)
        entry["purpose"] = kind
        response = LLMResponse(content=content, model="counting-mock")
        response.usage = {
            "input_tokens": entry["input_tokens"],
            "output_tokens": entry["output_tokens"],
        }
        self.record_usage(response.usage)
        return response

    def structured_output(self, messages, schema, *, max_attempts=3, **kwargs) -> dict[str, Any]:
        self.begin_call()
        fields = set(getattr(schema, "model_fields", {}) or {})
        schema_name = getattr(schema, "__name__", "unknown")
        if {"goal", "steps"} <= fields:
            if not self._plans:
                # 未脚本化的计划请求：记为意外远程规划（基线期用于发现路径分叉）。
                result = {"goal": "mock 空计划", "steps": []}
                shape = "plan_unscripted"
            else:
                result = self._plans.pop(0)
                shape = "plan"
        elif "action" in fields:
            shape = "decision"
            result = self._decisions.pop(0) if self._decisions else {
                "action": "finish", "rationale": "mock 默认收尾",
            }
        else:
            shape = "generic"
            result = schema.model_validate({}).model_dump() if hasattr(schema, "model_validate") else {}
        output = json.dumps(result, ensure_ascii=False, default=str)
        entry = self._log("structured", messages, schema_name, output)
        entry["shape"] = shape
        self.record_usage({
            "input_tokens": entry["input_tokens"],
            "output_tokens": entry["output_tokens"],
        })
        return result


# ---------------------------------------------------------------------------
# 夹具：tmp CSV + 内存库 + 数据引擎
# ---------------------------------------------------------------------------
def make_dataframe(rows: int = 80) -> pl.DataFrame:
    """构造含数值列、缺失值、可预测二分类目标与原始行标记的夹具。"""
    x1 = [float(i) for i in range(rows)]
    x2 = [float((i * 7) % 13 + (i % 5)) for i in range(rows)]
    age = [None if i % 17 == 0 else float(20 + (i % 45)) for i in range(rows)]
    # 刻意制造几个极端值用于异常值检测。
    score = [float(50 + (i % 20)) for i in range(rows)]
    score[5], score[42] = 999.0, -999.0
    label = ["pos" if (x2[i] + i) % 3 != 0 else "neg" for i in range(rows)]
    notes = [RAW_MARKER % i for i in range(rows)]
    return pl.DataFrame({
        "x1": x1, "x2": x2, "age": age, "score": score,
        "label": label, "note": notes,
    })


def build_runtime(tmp_dir: Path, *, use_remote: bool, plans=None, decisions=None, chat_answers=None):
    """每个场景一套全新运行时（内存库 + tmp 存储 + 独立 store），互不串状态。"""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import app.models.connector  # noqa: F401
    from app.core.database import Base

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()

    storage = StorageService(LocalStorage(root=tmp_dir / "storage_root"))
    csv_path = tmp_dir / "fixture.csv"
    make_dataframe().write_csv(csv_path)
    ds = DatasetService(db, storage)
    dataset = ds.create("scenario-fixture", "统一 Loop 场景夹具")
    ds.create_version(dataset.id, pl.read_csv(csv_path))

    data_engine = DataEngineService(ds)
    exp = ExperimentService(db, ds)
    store = AgentStore(path=tmp_dir / "agent_store.json")

    llm = None
    if use_remote:
        llm = CountingLLM(
            chat_answers=chat_answers,
            structured_plans=plans,
            structured_decisions=decisions,
        )
    runtime = AgentRuntime(
        data_engine, experiment_service=exp, db=db, llm=llm, store=store
    )
    session = runtime.create_session(dataset_ids=[dataset.id])
    return runtime, session, llm, dataset.id


# ---------------------------------------------------------------------------
# 运行一个 Turn（自动处理确认 / 澄清等待态）
# ---------------------------------------------------------------------------
TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED}


def run_turn(runtime: AgentRuntime, session, text: str, *, clarify_answer: str = "知道了") -> dict[str, Any]:
    started = time.perf_counter()
    run = runtime.run(session, text)
    continuations = 0
    while run.status not in TERMINAL and continuations < 6:
        continuations += 1
        if run.status == RunStatus.WAITING_CONFIRMATION:
            run = runtime.resume(session, run.id)
        elif run.status == RunStatus.WAITING_CLARIFICATION:
            run = runtime.answer_clarification(run.id, clarify_answer)
        else:
            break
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "run_id": run.id,
        "request": text,
        "status": str(run.status),
        "elapsed_ms": elapsed_ms,
        "final_answer": run.final_answer or "",
        "error": run.error or "",
        "tool_calls": [c.to_dict() for c in run.tool_calls],
        "ledger": run.token_ledger.to_dict(),
        "continuations": continuations,
    }


def ledger_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    """两轮账本快照之差（to_dict 的嵌套结构）。"""
    return {
        "remote_calls": int(b.get("llm_calls", 0)) - int(a.get("llm_calls", 0)),
        "remote_input_tokens": int(b.get("actual", {}).get("input_tokens", 0))
        - int(a.get("actual", {}).get("input_tokens", 0)),
        "remote_output_tokens": int(b.get("actual", {}).get("output_tokens", 0))
        - int(a.get("actual", {}).get("output_tokens", 0)),
        "escalation_count": int(b.get("remote_escalations", 0))
        - int(a.get("remote_escalations", 0)),
    }


# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------
# 旧 Planner 的 AgentPlan 结构化脚本（Mock 远程扮演「合格远程规划器」）。
PLAN_PROFILE = [
    {"tool": "dataset.inspect", "arguments": {}, "expected_output": "数据概况"},
    {"tool": "dataset.profile", "arguments": {}, "expected_output": "统计画像"},
]
PLAN_MODELING = [
    {"tool": "dataset.inspect", "arguments": {}, "expected_output": "数据概况"},
    {"tool": "dataset.schema", "arguments": {}, "expected_output": "列结构"},
    {"tool": "dataset.profile", "arguments": {}, "expected_output": "统计画像"},
    {"tool": "dataset.quality", "arguments": {}, "expected_output": "数据质量"},
    {"tool": "ml.detect_task", "arguments": {"infer_target": True,
        "goal": "全面分析并训练一个分类模型预测label"}, "expected_output": "推断目标列"},
    {"tool": "ml.prepare", "arguments": {"target": "{{step5.target}}"}, "expected_output": "特征准备"},
    {"tool": "ml.train", "arguments": {"model": "logistic_regression",
        "target": "{{step5.target}}",
        "goal": "全面分析并训练一个分类模型预测label"}, "expected_output": "训练分类模型"},
]
PLAN_QUALITY = [
    {"tool": "dataset.inspect", "arguments": {}, "expected_output": "数据概况"},
    {"tool": "dataset.quality", "arguments": {}, "expected_output": "数据质量"},
]
# 新 Loop 的结构化远程决策（重构后复跑时消费；旧代码不识别此 shape）。
# 字段与 loop.RemoteDecision 对齐：action ∈ {execute_tool, ask_user, chat, stop}，
# tool/arguments 是「当前动作」，next_steps 是后续工具名（≤4）。
DECISION_INSPECT_PROFILE = {
    "action": "execute_tool",
    "tool": "dataset.inspect",
    "arguments": {},
    "next_steps": ["dataset.profile"],
    "rationale": "开放式诉求，先做概况与画像两个阶段动作",
}
DECISION_STRATEGY = {
    "action": "execute_tool",
    "tool": "dataset.inspect",
    "arguments": {},
    "next_steps": ["dataset.profile"],
    "rationale": "本地低置信度，远程给出阶段性动作：概况→画像",
}

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": 1, "name": "普通问候", "env": "mock",
        "turns": ["你好"],
        "plans": [], "decisions": [],
    },
    {
        "id": 2, "name": "问候中启动分析任务", "env": "mock",
        "turns": ["你好，请帮我分析一下这个数据。"],
        "plans": [{"goal": "分析数据集", "steps": PLAN_PROFILE}],
        "decisions": [DECISION_INSPECT_PROFILE],
    },
    {
        "id": 3, "name": "查询行数（简单任务）", "env": "mock",
        "turns": ["看看这批数据有多少行"],
        "plans": [], "decisions": [],
    },
    {
        "id": 4, "name": "分析数据分布（简单任务）", "env": "mock",
        "turns": ["分析一下数据分布"],
        "plans": [], "decisions": [],
    },
    {
        "id": 5, "name": "知识型问题：什么是异常值", "env": "mock",
        "turns": ["什么是异常值？"],
        "plans": [], "decisions": [],
    },
    {
        "id": 6, "name": "检查异常值（简单任务）", "env": "mock",
        "turns": ["检查这批数据的异常值"],
        "plans": [], "decisions": [],
    },
    {
        "id": 7, "name": "分析过程中继续追问", "env": "mock",
        "turns": ["检查一下数据质量", "为什么这里缺失这么多？"],
        "plans": [{"goal": "解释缺失情况", "steps": PLAN_QUALITY}],
        "decisions": [],
    },
    {
        "id": 8, "name": "分析过程中修改要求（删除改填充）", "env": "mock",
        "turns": ["检查一下数据质量", "缺失值不要删除，改成均值填充"],
        "plans": [], "decisions": [],
    },
    {
        "id": 9, "name": "多步骤 EDA+预处理+建模", "env": "mock",
        "turns": ["帮我全面分析这个数据集并训练一个分类模型预测label"],
        "plans": [{"goal": "全面分析并训练分类模型", "steps": PLAN_MODELING}],
        "decisions": [],
    },
    {
        "id": 10, "name": "Qwen 不可用（词法兜底）", "env": "qwen_unavailable",
        "turns": ["检查这批数据的异常值"],
        "plans": [], "decisions": [],
    },
    {
        "id": 11, "name": "Remote LLM 不可用", "env": "no_remote",
        "turns": ["看看这批数据有多少行"],
        "plans": [], "decisions": [],
    },
    {
        "id": 12, "name": "本地无法判断时自动升级", "env": "mock",
        "turns": ["这个分析我总觉得哪里不对，你看着办，多比较几种思路，给我最合适的方案"],
        "plans": [{"goal": "低置信开放式分析", "steps": PLAN_PROFILE}],
        "decisions": [DECISION_STRATEGY],
    },
    {
        "id": 13, "name": "远程判断后回到本地执行", "env": "mock",
        "turns": ["这个分析我总觉得哪里不对，你看着办，多比较几种思路，给我最合适的方案"],
        "plans": [{"goal": "低置信开放式分析", "steps": PLAN_PROFILE}],
        "decisions": [DECISION_STRATEGY],
    },
    {
        "id": 14, "name": "多轮不重复发送完整上下文", "env": "mock",
        "turns": ["检查这批数据的异常值", "为什么这里异常这么多？"],
        "turn_clarify": {"为什么这里异常这么多？": "存在极端值可能是数据采集或同步环节的异常"},
        "plans": [{"goal": "解释缺失情况", "steps": PLAN_QUALITY}],
        "decisions": [],
    },
]


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------
def _tools(turn: dict[str, Any], status: str | None = None) -> list[str]:
    out = []
    for c in turn["tool_calls"]:
        if status is None or c.get("status") == status:
            out.append(str(c.get("tool")))
    return out


def _agg(turns: list[dict[str, Any]]) -> dict[str, int]:
    if not turns:
        return {"remote_calls": 0, "remote_input_tokens": 0,
                "remote_output_tokens": 0, "escalation_count": 0}
    last = turns[-1]["ledger"]
    return {
        "remote_calls": int(last.get("llm_calls", 0)),
        "remote_input_tokens": int(last.get("actual", {}).get("input_tokens", 0)),
        "remote_output_tokens": int(last.get("actual", {}).get("output_tokens", 0)),
        "escalation_count": int(last.get("remote_escalations", 0)),
    }


def evaluate(scenario: dict[str, Any], turns: list[dict[str, Any]], llm: CountingLLM | None) -> list[dict[str, Any]]:
    """每个断言产出 {name, passed, detail}；基线期失败只记录。"""
    results: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    agg = _agg(turns)
    all_tools = [t for turn in turns for t in _tools(turn)]
    ok_tools = [t for turn in turns for t in _tools(turn, status="ok")]

    add("运行完成", all(t["status"] == str(RunStatus.COMPLETED) for t in turns),
        "|".join(f"{t['status']}:{t['error'][:60]}" for t in turns))
    add("产出非空回答", all(bool(t["final_answer"].strip()) for t in turns))

    sid = scenario["id"]
    if sid in (3, 4, 6, 10):
        add("简单任务零远程调用", agg["remote_calls"] == 0,
            f"remote_calls={agg['remote_calls']}")
    if sid in (3, 11):
        add("执行了 dataset.inspect", "dataset.inspect" in ok_tools, str(ok_tools))
    if sid == 4:
        add("执行了分布类工具", any(t.startswith("eda.") for t in ok_tools), str(ok_tools))
    if sid in (6, 10):
        add("执行了 eda.outlier", "eda.outlier" in ok_tools, str(ok_tools))
    if sid == 2:
        add("建任务并执行本地工具", len(ok_tools) >= 1, str(ok_tools))
        add("任务零远程调用", agg["remote_calls"] == 0,
            f"remote_calls={agg['remote_calls']}")
    if sid == 1:
        add("问候零远程调用", agg["remote_calls"] == 0,
            f"remote_calls={agg['remote_calls']}")
        add("问候不调工具", len(all_tools) == 0, str(all_tools))
    if sid == 11:
        add("无远程时数据任务成功", len(ok_tools) >= 1 and agg["remote_calls"] == 0,
            f"tools={ok_tools}")
    if sid == 9:
        order = {t: i for i, t in enumerate(ok_tools)}
        chain_ok = all(k in order for k in ("ml.detect_task", "ml.prepare", "ml.train")) and \
            order.get("ml.detect_task", -1) < order.get("ml.prepare", -1) < order.get("ml.train", -1)
        add("detect→prepare→train 链路", chain_ok, str(ok_tools))
        add("训练成功", any(c.get("tool") == "ml.train" and c.get("status") == "ok"
                            for t in turns for c in t["tool_calls"]))
    if sid in (7, 14) and len(turns) >= 2:
        t2_tools = _tools(turns[1])
        repeated = [t for t in t2_tools if t in READONLY_TOOLS]
        add("追问不重复只读工具", len(repeated) == 0, f"turn2_tools={t2_tools}")
        add("追问零工具零远程（读状态作答）",
            len(t2_tools) == 0 and ledger_diff(turns[0]["ledger"], turns[1]["ledger"])["remote_calls"] == 0,
            f"tools={t2_tools}, diff={ledger_diff(turns[0]['ledger'], turns[1]['ledger'])}")
        # 追问引用的事实词随场景而异：S7 第一轮是质量检查（含缺失），S14 第一轮是异常值。
        fact_word = "缺失" if sid == 7 else "异常"
        add("追问回答引用事实", fact_word in turns[1]["final_answer"],
            turns[1]["final_answer"][:80])
    if sid == 8 and len(turns) >= 2:
        cleans = [c for t in turns for c in t["tool_calls"] if c.get("tool") == "data.clean"]
        fill_ok = False
        detail = "未调用 data.clean"
        for c in cleans:
            strat = ((c.get("arguments") or {}).get("missing") or {}).get("strategy")
            detail = f"strategy={strat}"
            if strat in {"mean", "median", "mode", "fill", "constant"}:
                fill_ok = True
        add("清洗策略为填充而非删除", fill_ok, detail)
    if sid == 12:
        add("发生远程升级", agg["escalation_count"] >= 1,
            f"escalation_count={agg['escalation_count']}")
    if sid == 13:
        add("升级后本地执行工具", len(ok_tools) >= 2, str(ok_tools))
        add("升级后远程仅调用1次", agg["remote_calls"] == 1,
            f"remote_calls={agg['remote_calls']}, escalations={agg['escalation_count']}")
    if sid == 14 and llm is not None:
        # ① 远程载荷不得携带整列原始行标记（允许极少量样例，不允许整列）。
        marker_hits = 0
        for call in llm.calls:
            blob = "\n".join(m["content"] for m in call["messages"])
            marker_hits += sum(blob.count(RAW_MARKER % i) for i in range(80))
        add("远程载荷不含完整原始结果", marker_hits < 10, f"marker_hits={marker_hits}")
        # ② 连续两轮的结构化决策输入不得膨胀（紧凑状态 + 按需工具面）。
        structured_calls = [c for c in llm.calls if c["kind"] == "structured"]
        later_calls = [c for c in structured_calls if c.get("turn") and c["turn"] >= 2]
        if later_calls:
            later_max = max(c["input_tokens"] for c in later_calls)
            first_turn = [c for c in structured_calls if c.get("turn") == 1]
            growth = later_max - (first_turn[0]["input_tokens"] if first_turn else 0)
            add("后续决策输入不膨胀", growth <= 400 and later_max <= 1600,
                f"later_max={later_max}, growth={growth}")
        else:
            add("后续决策输入不膨胀", True, "后续轮次无结构化决策（零膨胀）")
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_one(scenario: dict[str, Any]) -> dict[str, Any]:
    use_remote = scenario["env"] != "no_remote"
    with tempfile.TemporaryDirectory(prefix="agent_loop_") as tmp:
        get_version_frame_cache().clear()
        runtime, session, llm, _dataset_id = build_runtime(
            Path(tmp),
            use_remote=use_remote,
            plans=[dict(p) for p in scenario.get("plans", [])],
            decisions=[dict(d) for d in scenario.get("decisions", [])],
        )
        turns: list[dict[str, Any]] = []
        call_ranges: list[tuple[int, int]] = []
        for text in scenario["turns"]:
            start = len(llm.calls) if llm is not None else 0
            clarify = scenario.get("turn_clarify", {}).get(text, "知道了")
            turn = run_turn(runtime, session, text, clarify_answer=clarify)
            end = len(llm.calls) if llm is not None else 0
            call_ranges.append((start, end))
            turns.append(turn)
        if llm is not None:
            for i, (s, e) in enumerate(call_ranges):
                for idx in range(s, e):
                    llm.calls[idx]["turn"] = i + 1

    agg = _agg(turns)
    assertions = evaluate(scenario, turns, llm)
    ok_tools = [t for turn in turns for t in _tools(turn, status="ok")]
    # qwen_calls：本脚本固定 off 档（Qwen 层缺席），按定义恒为 0；
    # task_steps / tool_calls 以真实工具调用记录为准。
    metrics = {
        "remote_calls": agg["remote_calls"],
        "remote_input_tokens": agg["remote_input_tokens"],
        "remote_output_tokens": agg["remote_output_tokens"],
        "qwen_calls": 0,
        "tool_calls": sum(len(t["tool_calls"]) for t in turns),
        "tool_calls_ok": len(ok_tools),
        "task_steps": sum(len(t["tool_calls"]) for t in turns),
        "escalation_count": agg["escalation_count"],
        "elapsed_ms": round(sum(t["elapsed_ms"] for t in turns), 1),
    }
    remote_log = None
    if llm is not None:
        remote_log = [
            {
                "turn": next((i + 1 for i, (s, e) in enumerate(call_ranges) if s <= idx < e), None),
                "kind": c["kind"], "purpose": c.get("purpose", ""),
                "shape": c.get("shape", ""), "schema": c["schema"],
                "input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"],
            }
            for idx, c in enumerate(llm.calls)
        ]
    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "env": scenario["env"],
        "turns": [
            {k: v for k, v in t.items() if k in
             ("run_id", "request", "status", "elapsed_ms", "final_answer",
              "error", "continuations", "ledger")}
            | {"tool_sequence": _tools(t), "tool_sequence_ok": _tools(t, status="ok")}
            for t in turns
        ],
        "metrics": metrics,
        "tool_sequence": [t for turn in turns for t in _tools(turn)],
        "assertions": assertions,
        "passed": all(a["passed"] for a in assertions),
        "remote_calls_detail": remote_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 统一 Loop 14 场景离线评测")
    parser.add_argument("--out", default="", help="指标 JSON 输出路径")
    args = parser.parse_args()

    started = time.time()
    results = []
    for scenario in SCENARIOS:
        t0 = time.time()
        try:
            result = run_one(scenario)
        except Exception as exc:  # noqa: BLE001 — 单场景异常不影响其余场景取证
            import traceback

            result = {
                "id": scenario["id"], "name": scenario["name"], "env": scenario["env"],
                "turns": [], "metrics": {}, "tool_sequence": [],
                "assertions": [{"name": "harness", "passed": False,
                                "detail": f"{type(exc).__name__}: {exc}"}],
                "passed": False, "traceback": traceback.format_exc()[-2000:],
                "remote_calls_detail": None,
            }
        result["wall_ms"] = round((time.time() - t0) * 1000, 1)
        results.append(result)
        flag = "PASS" if result["passed"] else "FAIL"
        m = result.get("metrics", {})
        print(
            f"[{flag}] S{result['id']:>2} {result['name']} "
            f"remote={m.get('remote_calls', '-')} in_tok={m.get('remote_input_tokens', '-')} "
            f"tools={m.get('tool_calls', '-')} esc={m.get('escalation_count', '-')} "
            f"({result['wall_ms']}ms)",
            file=sys.stderr,
        )

    passed = sum(1 for r in results if r["passed"])
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - started, 1),
        "environment": {
            "local_router_mode": settings.LOCAL_ROUTER_MODE,
            "remote": "CountingLLM(pseudo-token=chars/2)",
            "plan_cache": settings.AGENT_ENABLE_PLAN_CACHE,
            "fixture_rows": 80,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "simple_remote_rate": (
                sum(1 for r in results if r["id"] in (3, 4, 6)
                    and r.get("metrics", {}).get("remote_calls", 0) > 0) / 3
            ),
        },
        "scenarios": results,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n指标已写入 {out_path}", file=sys.stderr)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n{passed}/{len(results)} 场景断言全部通过（基线期允许失败，失败即重构证据）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
