"""从 AgentStore 的真实执行轨迹里挖取 Router 样本。

与 `build_dataset.py` 的根本区别：这里的标签来自**真实执行结果**，
不是人工模板，也不是让外部 LLM 打分。一份计划跑没跑通，系统自己知道 ——
这是这个平台相比通用数据集最不可替代的资产。

标签可信度分级（保守策略，宁缺毋滥）
----------------------------------
- `ok`        → **正样本**。执行成功同时验证了「工具选对」与「参数可用」。
- `failed`    → 只进诊断报告，**不进训练集**。失败无法区分是「工具选错」
                 还是「参数写错」，当负标签用会污染数据。
- `denied` / `needs_confirmation` → 授权与确认场景样本（平台能力边界的一部分）。
- **多步 run 只进诊断**：多步的正确标注需要逐步上下文，当前无此信息，
  强行把第一步当 gold 会把「模型想做的」当成「做成的」。
- 计划中出现但未执行的步骤 → 一律不计入。

产出（写入 `scripts/router/dataset/`）
------------------------------------
- `traces.jsonl`       真实轨迹样本
- `trace_report.json`  轨迹画像 / 可挖取量 / 失败与异常诊断
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.local_router import contract as C  # noqa: E402

STORE_PATH = BACKEND_ROOT / "data" / "agent_store.json"
OUT_DIR = Path(__file__).resolve().parent / "dataset"

# 计划里未解析的步骤引用，如 {{step1.dataset_id}}
STEP_REF_RE = re.compile(r"\{\{.*?\}\}")


def load_store() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def clean_params(tool: str, raw: Any) -> tuple[dict[str, Any], list[str]]:
    """只保留工具 schema 认识的参数，并记录异常。"""
    declared = C.known_params(tool)
    params: dict[str, Any] = {}
    issues: list[str] = []
    if not isinstance(raw, dict):
        return params, ["arguments 不是对象"]
    for key, value in raw.items():
        if key not in declared:
            issues.append(f"schema 未声明参数 {key}")
            continue
        if isinstance(value, str) and STEP_REF_RE.search(value):
            issues.append(f"未解析的步骤引用 {key}")
        params[key] = value
    return params, issues


def trace_id(run_id: str, step_index: int) -> str:
    digest = hashlib.sha1(f"{run_id}|{step_index}".encode("utf-8")).hexdigest()[:10]
    return f"trace-{digest}"


def build_sample(
    *,
    run_id: str,
    call: dict[str, Any],
    utterance: str,
    bound_dataset_id: int | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """把一条成功的工具调用变成样本；返回 (sample, issues)。"""
    tool = str(call.get("tool") or "")
    issues: list[str] = []
    if tool not in set(C.tool_label_space()):
        return None, [f"工具 {tool!r} 不在当前注册表"]

    intent = C.intent_of_tool(tool)
    if intent is None:
        return None, [f"无法派生 {tool} 的 intent"]

    params, param_issues = clean_params(tool, call.get("arguments"))
    issues.extend(param_issues)

    target = {
        "intent": intent.value,
        "tool": tool,
        "params": params,
        "missing": [],
        "escalate": False,
        "escalate_reason": None,
        "confidence": 1.0,
    }
    decision = C.RouterDecision(**target)
    outcome = C.validate_decision(decision)
    if outcome.problems:
        return None, outcome.problems
    if outcome.missing_required:
        # 真实调用缺必填参数却执行成功 —— 说明平台有默认值兜底，
        # 这不是「缺槽位」，但也不该当完整 gold，记入诊断。
        return None, [f"成功执行却缺必填槽位 {outcome.missing_required}"]

    step_index = int(call.get("step_index") or 0)
    sid = trace_id(run_id, step_index)
    return (
        {
            "id": sid,
            "category": "trace_routing",
            "split": "train",
            "request": {
                "utterance": utterance,
                "bound_dataset_id": bound_dataset_id,
                "available_columns": [],
                "recent_tools": [],
            },
            "target": target,
            "meta": {
                "source": "agent_store",
                "run_id": run_id,
                "step_index": step_index,
                "verified_by": "tool_result",
                "status": call.get("status"),
                "attempt": call.get("attempt"),
                "elapsed_ms": call.get("elapsed_ms"),
            },
        },
        issues,
    )


def dataset_id_of(params: dict[str, Any]) -> int | None:
    for key in ("dataset_id", "left_dataset_id"):
        value = params.get(key)
        if isinstance(value, int):
            return value
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = load_store()
    runs = store.get("runs") or []
    sessions = store.get("sessions") or []

    samples: list[dict[str, Any]] = []
    issues_all: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    multi_step: list[dict[str, Any]] = []
    authorization: list[dict[str, Any]] = []
    run_profile: list[dict[str, Any]] = []
    event_types: Counter[str] = Counter()
    real_utterances: list[str] = []
    tools_seen: Counter[str] = Counter()

    for run in runs:
        run_id = str(run.get("id") or "")
        utterance = str(run.get("user_request") or "").strip()
        status = str(run.get("status") or "")
        calls = run.get("tool_calls") or []
        events = run.get("events") or []

        for event in events:
            event_types[str(event.get("type"))] += 1
        if utterance:
            real_utterances.append(utterance)

        run_profile.append(
            {
                "run_id": run_id,
                "status": status,
                "utterance": utterance,
                "tool_call_count": len(calls),
                "event_count": len(events),
                "error": str(run.get("error") or "")[:200],
                "elapsed_seconds": run.get("elapsed_seconds"),
            }
        )

        if len(calls) > 1:
            multi_step.append(
                {
                    "run_id": run_id,
                    "utterance": utterance,
                    "tools": [str(c.get("tool")) for c in calls],
                    "status": status,
                }
            )
            continue

        for call in calls:
            tool = str(call.get("tool") or "")
            tools_seen[tool] += 1
            call_status = str(call.get("status") or "")

            if call_status in ("denied", "needs_confirmation"):
                authorization.append(
                    {
                        "run_id": run_id,
                        "utterance": utterance,
                        "tool": tool,
                        "status": call_status,
                        "reason": str(call.get("error") or "")[:200],
                    }
                )
                continue

            if call_status != "ok":
                failures.append(
                    {
                        "run_id": run_id,
                        "utterance": utterance,
                        "tool": tool,
                        "status": call_status,
                        "error": str(call.get("error") or "")[:300],
                    }
                )
                continue

            params, _ = clean_params(tool, call.get("arguments"))
            sample, issues = build_sample(
                run_id=run_id,
                call=call,
                utterance=utterance,
                bound_dataset_id=dataset_id_of(params),
            )
            if issues:
                issues_all.append({"run_id": run_id, "tool": tool, "issues": issues})
            if sample is not None:
                samples.append(sample)

    # 去重：同一句 utterance + 同一工具只留一条
    seen: set[tuple[str, str | None]] = set()
    unique: list[dict[str, Any]] = []
    for item in samples:
        key = (item["request"]["utterance"], item["target"]["tool"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    samples = unique

    path = OUT_DIR / "traces.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "source_file": str(STORE_PATH),
        "source_bytes": STORE_PATH.stat().st_size if STORE_PATH.exists() else 0,
        "sessions": len(sessions),
        "runs": len(runs),
        "runs_profile": run_profile,
        "event_type_distribution": dict(event_types.most_common()),
        "tool_call_distribution": dict(tools_seen.most_common()),
        "harvestable": {
            "samples_written": len(samples),
            "note": "只有 status=ok 的调用才计入；失败与多步只进诊断",
        },
        "diagnostics": {
            "failures": failures,
            "multi_step_runs": multi_step,
            "authorization_cases": authorization,
            "issues": issues_all,
        },
        "real_utterances": real_utterances,
        "verdict": (
            "真实轨迹数量不足以独立训练 Router，但足以"
            "① 校验合成数据是否覆盖真实表达，② 建立可持续的增量采集管线。"
        ),
    }
    (OUT_DIR / "trace_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(json.dumps(
        {
            "sessions": len(sessions),
            "runs": len(runs),
            "samples_written": len(samples),
            "event_types_top": dict(event_types.most_common(5)),
            "tool_calls": dict(tools_seen.most_common()),
            "failures": len(failures),
            "multi_step": len(multi_step),
            "authorization": len(authorization),
            "issues": len(issues_all),
        },
        ensure_ascii=False,
        indent=1,
    ))


if __name__ == "__main__":
    main()
