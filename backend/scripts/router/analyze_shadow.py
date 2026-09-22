"""读 shadow trace，回答一个问题：**如果当时让本地 Router 接管，会怎样？**

判断口径（决定要不要接管，而不是决定模型准不准）
------------------------------------------------
今天所有请求都会走 LLM planner（约 1.2 万 token）。本地 Router 的唯一价值是
**在某些请求上跳过 planner**。所以「准不准」要翻译成四种后果：

| 后果 | 条件 | 含义 |
| --- | --- | --- |
| **可省** | Router 预测了工具，且它就是**实际执行链的第一步** | 接管可省一次 planner 调用 |
| **可省(部分)** | Router 预测的工具在**实际执行链里，但不是第一步** | 接管会改变顺序，需人工核对 |
| **会出错** | 实际走了工具流程，但 Router 说是 `chat` | 接管会把数据请求当闲聊 —— **最危险** |
| **会出错(工具)** | Router 预测了工具 X，实际执行链里没有 X | 接管会跳过 planner 去做别的事 |
| 中性 | Router 说 `escalate` | 与今天行为相同（仍然交云端） |
| 待确认 | Router 说 `ask::X` | 接管会**反问用户**（行为变更，需人工判是否恰当） |

⚠️ 这套口径的输入是 shadow trace，而 shadow 阶段**没有 `available_columns`**。
`train_runtime_l1.py` 的消融列（`shadow 口径`）就是这一列对应的预期值 ——
两者不可混用。另外 trace 里 `rules.mode` 是既有规则路由的判定，它是**对照基线**：
没有它就无法区分「本地 Router 判错」与「规则本来就判错」。

用法
    backend/.venv/Scripts/python.exe backend/scripts/router/analyze_shadow.py
    ... analyze_shadow.py --path <shadow.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.local_router import trace as T  # noqa: E402

CONFIDENCE_BANDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _kind(route: str) -> str:
    return str(route).split("::", 1)[0]


def _tool(route: str) -> str | None:
    kind, _, rest = str(route).partition("::")
    return rest or None if kind in ("call", "ask") else None


def join(records: list[dict]) -> list[dict]:
    """按 run_id 把 route 与 outcome 配对。"""
    routes: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}
    for record in records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        if record.get("kind") == "route":
            routes[run_id] = record
        elif record.get("kind") == "outcome":
            outcomes[run_id] = record
    return [{**routes[k], "outcome": outcomes.get(k)} for k in routes]


def verdict(pair: dict) -> str:
    router = pair.get("router") or {}
    if not router.get("available"):
        return "模型不可用"
    route = router.get("route") or ""
    kind = _kind(route)
    if kind == "escalate":
        return "中性(升级)"
    if kind == "ask":
        return "待确认(反问)"

    outcome = pair.get("outcome")
    if outcome is None:
        return "无结果(未配对)"
    executed = list(outcome.get("executed_tools") or [])
    rules_mode = ((pair.get("rules") or {}).get("mode") or "").lower()

    if kind == "chat":
        return "会出错(误判闲聊)" if (executed or rules_mode == "agent") else "一致(闲聊)"

    predicted = _tool(route)
    if executed and predicted == executed[0]:
        return "可省"
    if predicted in executed:
        return "可省(部分)"
    return "会出错(工具不符)"


def summarize(pairs: list[dict]) -> dict:
    verdicts = Counter(verdict(p) for p in pairs)
    confidences = [float((p.get("router") or {}).get("confidence") or 0.0)
                   for p in pairs if (p.get("router") or {}).get("available")]
    rules_modes = Counter(((p.get("rules") or {}).get("mode") or "?") for p in pairs)

    by_rules: dict[str, Counter] = defaultdict(Counter)
    for pair in pairs:
        by_rules[(((pair.get("rules") or {}).get("mode") or "?"))][verdict(pair)] += 1

    # 门控模拟：把置信度低于 θ 的「可省」降级为「中性(升级)」，同时看能拦掉多少「会出错」
    gate = []
    for theta in CONFIDENCE_BANDS:
        low = [p for p in pairs
               if (p.get("router") or {}).get("available")
               and float((p.get("router") or {}).get("confidence") or 0.0) < theta]
        low_verdicts = Counter(verdict(p) for p in low)
        gate.append({
            "theta": theta,
            "gated": len(low),
            "gated_ratio": round(len(low) / max(len(pairs), 1), 4),
            "gated_savable": low_verdicts.get("可省", 0) + low_verdicts.get("可省(部分)", 0),
            "gated_wrong": low_verdicts.get("会出错(误判闲聊)", 0) + low_verdicts.get("会出错(工具不符)", 0),
        })

    n = max(len(pairs), 1)
    return {
        "n_runs": len(pairs),
        "rule_mode_distribution": dict(rules_modes),
        "verdicts": dict(verdicts),
        "verdict_ratio": {k: round(v / n, 4) for k, v in verdicts.items()},
        "verdict_by_rule_mode": {k: dict(v) for k, v in by_rules.items()},
        "confidence": {
            "n": len(confidences),
            "mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "bands": {f"<{b}": sum(1 for c in confidences if c < b) for b in CONFIDENCE_BANDS},
        },
        "gate_simulation": gate,
        "savable": verdicts.get("可省", 0),
        "risky": verdicts.get("会出错(误判闲聊)", 0) + verdicts.get("会出错(工具不符)", 0),
    }


def markdown(summary: dict, pairs: list[dict]) -> str:
    v = summary["verdicts"]
    lines = [
        "# 本地 Router shadow 数据分析",
        "",
        f"配对运行数 **{summary['n_runs']}**；"
        f"既有规则路由判定分布 {summary['rule_mode_distribution']}",
        "",
        "## 结论口径",
        "",
        "| 后果 | 条数 | 占比 |",
        "| --- | --- | --- |",
    ]
    for key in ("可省", "可省(部分)", "会出错(误判闲聊)", "会出错(工具不符)",
                "中性(升级)", "待确认(反问)", "一致(闲聊)", "模型不可用", "无结果(未配对)"):
        if key in v:
            lines.append(f"| {key} | {v[key]} | {summary['verdict_ratio'].get(key, 0):.1%} |")
    lines += [
        "",
        f"- **接管可省 planner 调用**：{summary['savable']} 条",
        f"- **接管会出错**：{summary['risky']} 条  ← 决定是否开 guard 档的关键数字",
        "",
        "## 按规则路由判定拆分",
        "",
        "| rules.mode | " + " | ".join(sorted({k for c in summary["verdict_by_rule_mode"].values() for k in c})) + " |",
    ]
    sub_keys = sorted({k for c in summary["verdict_by_rule_mode"].values() for k in c})
    lines.append("| --- | " + " | ".join("---" for _ in sub_keys) + " |")
    for mode, counter in summary["verdict_by_rule_mode"].items():
        lines.append(f"| {mode} | " + " | ".join(str(counter.get(k, 0)) for k in sub_keys) + " |")

    conf = summary["confidence"]
    lines += [
        "",
        "## 置信度分布（用于选门控阈值 θ）",
        "",
        f"均值 {conf['mean']}（n={conf['n']}）",
        "",
        "| 带宽 | " + " | ".join(conf["bands"]) + " |",
        "| --- | " + " | ".join("---" for _ in conf["bands"]) + " |",
        "| 条数 | " + " | ".join(str(x) for x in conf["bands"].values()) + " |",
        "",
        "### 门控模拟（θ 以下一律转升级）",
        "",
        "| θ | 被拦条数 | 占比 | 其中本可省 | 其中本会出错 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary["gate_simulation"]:
        lines.append(f"| {row['theta']} | {row['gated']} | {row['gated_ratio']:.1%} | "
                     f"{row['gated_savable']} | {row['gated_wrong']} |")

    lines += ["", "## 逐条明细（草稿，供人工核对）", "",
              "| run_id | 用户说了什么 | 规则判定 | 本地 Router | 置信度 | 实际执行 | 后果 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for pair in pairs[:80]:
        router = pair.get("router") or {}
        outcome = pair.get("outcome") or {}
        utterance = str(pair.get("utterance") or "")[:28].replace("|", "/")
        lines.append(
            f"| {pair.get('run_id')} | {utterance} | {((pair.get('rules') or {}).get('mode'))} | "
            f"{router.get('route')} | {router.get('confidence')} | "
            f"{','.join(outcome.get('executed_tools') or [])} | {verdict(pair)} |"
        )
    if len(pairs) > 80:
        lines.append(f"| … | 其余 {len(pairs) - 80} 条见 json | | | | | |")
    lines += [
        "",
        "## 注意",
        "",
        "- shadow 阶段**没有 `available_columns`**；对照数字请用 `train_runtime_l1.py` 的"
        "「shadow 口径」列，不要用默认列。",
        "- 样本量小的阶段不要看百分比，只看条数；**「会出错(误判闲聊)」应当是 0 才考虑开 guard 档**。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="分析本地 Router shadow trace")
    parser.add_argument("--path", default="", help="trace 文件（默认取配置里的 shadow.jsonl）")
    parser.add_argument("--out", default="", help="报告输出目录（默认与 trace 同目录）")
    args = parser.parse_args()

    path = Path(args.path) if args.path else T.trace_path()
    if not path.exists():
        print(f"未找到 trace：{path}")
        print("先打开 shadow 档（backend/.env 里设 LOCAL_ROUTER_MODE=shadow）并跑几条对话。")
        return 1

    records = list(T.iter_records(path))
    pairs = join(records)
    if not pairs:
        print(f"trace 存在但无有效记录：{path}（原始行 {len(records)}）")
        return 1

    summary = summarize(pairs)
    out_dir = Path(args.out) if args.out else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shadow_analysis.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "shadow_analysis.md").write_text(markdown(summary, pairs), encoding="utf-8")

    print(f"配对运行 {summary['n_runs']} 条 → {out_dir / 'shadow_analysis.md'}")
    for key in ("可省", "可省(部分)", "会出错(误判闲聊)", "会出错(工具不符)", "中性(升级)", "待确认(反问)"):
        if key in summary["verdicts"]:
            print(f"  {key:16s} {summary['verdicts'][key]:4d}  "
                  f"({summary['verdict_ratio'][key]:.1%})")
    print(f"  可省 {summary['savable']} / 会出错 {summary['risky']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
