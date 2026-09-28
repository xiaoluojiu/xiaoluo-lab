"""MMTU-Lab 评测执行器：真实调用被测链路 + 真实调用 DeepSeek 评分。

用法
----
    backend/.venv/Scripts/python.exe backend/scripts/eval/run_eval.py --round 1
    backend/.venv/Scripts/python.exe backend/scripts/eval/run_eval.py --round 2 --only M01,M05

被测对象就是线上真实出口，没有任何替身：
- entry=chat   → app.agent.answer.render_chat
- entry=render → app.agent.answer.render

评分：准确性/格式由 graders.py 确定性判定，相关性/完整性由 judge.py 评审。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 让 `import app.*` 与 `import graders/cases/judge` 都能解析
_BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.answer import render, render_chat  # noqa: E402
from app.agent.llm import OpenAICompatibleProvider  # noqa: E402
from app.agent.playbooks import PlaybookStep  # noqa: E402
from app.tools.result import ToolResult  # noqa: E402

import graders  # noqa: E402
from cases import build_cases  # noqa: E402
from judge import judge as llm_judge  # noqa: E402

ROOT = _BACKEND.parent
DOCS = ROOT / "docs" / "eval"
RESULTS = DOCS / "results"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_KEY = "sk-dcf4d589ed684052ad1cc63fe8c560c8"

PASS_LINE = 90  # 单用例达标线（总分 100）


def _provider() -> OpenAICompatibleProvider:
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    api_key = os.environ.get("LLM_API_KEY") or DEFAULT_KEY
    return OpenAICompatibleProvider(base_url, model, api_key, timeout=90.0, max_retries=2)


def _run_chat(case: dict, provider) -> str:
    text, _src = render_chat(
        case["question"],
        provider=provider,
        history=case.get("history") or [],
        context=case.get("context", ""),
        timeout=90.0,
        # 引擎侧会把当前数据集的真实列名传进来；这里如实传，才能测到对话路径的列名护栏
        columns=case.get("columns"),
    )
    return text


def _run_render(case: dict, provider) -> str:
    steps = []
    for tool, title, ok, data, summary, errors in case["steps"]:
        steps.append((
            PlaybookStep(tool=tool, title=title),
            ToolResult(success=ok, data=data, summary=summary, errors=errors or []),
        ))
    text, _src = render(
        case["question"],
        steps,
        provider=provider,
        columns=case.get("columns"),
        timeout=90.0,
    )
    return text


def evaluate(case: dict, provider) -> dict:
    started = time.time()
    try:
        if case["entry"] == "render":
            reply = _run_render(case, provider)
        else:
            reply = _run_chat(case, provider)
    except Exception as exc:  # noqa: BLE001
        reply = f"[调用异常] {type(exc).__name__}: {exc}"

    acc_ratio, acc_note = graders.score_accuracy(reply, case["grade"])
    fmt_ratio, fmt_detail = graders.score_format(reply, case.get("fmt_checks") or [])
    redlines = graders.hit_redlines(reply, case.get("redlines") or [])
    rel, comp, reason = llm_judge(
        provider, question=case["question"], gold=case["grade"].get("gold"), reply=reply
    )

    total = round(acc_ratio * 25 + fmt_ratio * 25 + rel + comp, 1)
    passed = total >= PASS_LINE and not redlines
    if redlines:
        total = 0.0

    return {
        "id": case["id"],
        "mmtu_task": case["mmtu_task"],
        "tier": case["tier"],
        "title": case["title"],
        "entry": case["entry"],
        "accuracy": round(acc_ratio * 25, 1),
        "relevance": rel,
        "completeness": comp,
        "format": round(fmt_ratio * 25, 1),
        "total": total,
        "passed": passed,
        "redlines": redlines,
        "acc_note": acc_note,
        "fmt_detail": [{"check": n, "ok": o} for n, o in fmt_detail],
        "judge_reason": reason,
        "elapsed_s": round(time.time() - started, 1),
        "reply": reply,
        "question": case["question"],
        "gold": case["grade"].get("gold"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--only", default="", help="只跑指定用例，逗号分隔")
    ap.add_argument("--tag", default="", help="本轮标记，如 baseline / after-fix")
    args = ap.parse_args()

    provider = _provider()
    cases = build_cases()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in wanted]

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n=== MMTU-Lab 第 {args.round} 轮 | 用例 {len(cases)} 条 | model={provider.model} ===\n")
    for i, case in enumerate(cases, 1):
        row = evaluate(case, provider)
        rows.append(row)
        flag = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{i:>2}/{len(cases)}] {row['id']:<5} {flag} "
            f"总{row['total']:>5} (准{row['accuracy']:>5} 相{row['relevance']:>3} "
            f"完{row['completeness']:>3} 格{row['format']:>5}) {row['mmtu_task'][:20]}"
        )
        if row["redlines"]:
            print(f"        红线：{'、'.join(row['redlines'])}")

    passed = [r for r in rows if r["passed"]]
    avg = round(sum(r["total"] for r in rows) / len(rows), 2) if rows else 0.0
    dim = lambda k: round(sum(r[k] for r in rows) / len(rows), 2) if rows else 0.0  # noqa: E731

    summary = {
        "round": args.round,
        "tag": args.tag,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": provider.model,
        "case_count": len(rows),
        "passed_count": len(passed),
        "failed_count": len(rows) - len(passed),
        "avg_total": avg,
        "avg_accuracy": dim("accuracy"),
        "avg_relevance": dim("relevance"),
        "avg_completeness": dim("completeness"),
        "avg_format": dim("format"),
        "pass_line": PASS_LINE,
        "all_passed": len(passed) == len(rows),
        "rows": rows,
    }

    out = RESULTS / f"round_{args.round}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"第 {args.round} 轮汇总  用例 {len(rows)}   达标 {len(passed)}   未达标 {len(rows)-len(passed)}")
    print(f"平均分 {avg}    准确性 {dim('accuracy')}  相关性 {dim('relevance')}  "
          f"完整性 {dim('completeness')}  格式 {dim('format')}")
    print(f"结果文件：{out}")
    print("=" * 72 + "\n")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
