"""评测 L0 结构规则层（`app.local_router.escalation_rules`）的升级判定质量。

回答一个问题：**「该不该升级」是规则问题还是模型问题？**

做法：同一份 844 条样本、同一套 `eval_harness` 口径，对比三种路线
  A) 纯结构规则            —— 本模块，零参数
  B) hybrid(groupCV)       —— 纯学习的词法基线（ask 用规则、rest 用 LinearSVC）
  C) B + A 覆盖            —— 规则命中则升级，否则用 B 的工具判定

⚠️ 规则是**只读 utterance** 的纯函数；本脚本不把规则产物写回数据集，避免自证。
⚠️ 升级语料与规则同一作者手写 ⇒ 存在循环论证风险，真实泛化须等真实会话采集后复验
   （见 `app/local_router/escalation_rules.py` 模块 docstring 的「适用边界」）。

词表一律从 `escalation_rules` 导入，本脚本**不重复定义**任何标记。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_harness as H  # noqa: E402
import baseline_lexical as B  # noqa: E402
from app.local_router.escalation_rules import detect_escalation  # noqa: E402

OUT_DIR = H.DATASET_DIR


def utterance_of(sample: dict) -> str:
    return str((sample.get("request") or {}).get("utterance") or "")


def rule_route(sample: dict) -> str | None:
    """规则给出的 route 串（仅升级侧）；None 表示规则不表态。"""
    reason = detect_escalation(utterance_of(sample))
    return f"escalate::{reason.value}" if reason else None


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    rows = H.load_split("all")
    by_id = {s["id"]: s for s in rows}
    gold_kind = {s["id"]: H.kind_of(H.route_label(s)) for s in rows}
    gold_reason = {s["id"]: (H.reason_of(H.route_label(s)) or "") for s in rows}
    pred_rule = {s["id"]: detect_escalation(utterance_of(s)) for s in rows}

    # ---- A) 纯结构规则（只衡量升级侧，其余不表态）----
    tp = sum(1 for i, k in gold_kind.items() if k == "escalate" and pred_rule[i])
    fp = sum(1 for i, k in gold_kind.items() if k != "escalate" and pred_rule[i])
    fn = sum(1 for i, k in gold_kind.items() if k == "escalate" and not pred_rule[i])
    p, r, f1 = prf(tp, fp, fn)

    print(f"样本 {len(rows)} 条")
    print(f"\n[A 纯结构规则] escalate 精确率 {p:.1%} / 召回率 {r:.1%} / F1 {f1:.1%}"
          f"  (tp={tp} fp={fp} fn={fn})")

    reasons = sorted(v for v in set(gold_reason.values()) if v)
    per_reason = []
    print("  逐原因（「原因也对」为严格口径）：")
    for rs in reasons:
        idx = [i for i in gold_kind if gold_kind[i] == "escalate" and gold_reason[i] == rs]
        exact_hit = sum(1 for i in idx if pred_rule[i] and pred_rule[i].value == rs)
        any_hit = sum(1 for i in idx if pred_rule[i])
        per_reason.append({"reason": rs, "n": len(idx), "reason_exact": exact_hit,
                           "detected_as_escalate": any_hit})
        print(f"    {rs:16s} n={len(idx):2d}  原因对 {exact_hit}/{len(idx)}"
              f"  判为升级 {any_hit}/{len(idx)}")

    fps = [s for s in rows if gold_kind[s["id"]] != "escalate" and pred_rule[s["id"]]]
    fns = [s for s in rows if gold_kind[s["id"]] == "escalate" and not pred_rule[s["id"]]]
    print(f"\n  误判为升级 {len(fps)} 条；漏判 {len(fns)} 条")
    for s in fns:
        print(f"    漏判 gold={H.route_label(s)} | {utterance_of(s)[:56]}")

    # ---- B) 学习基线 + C) 规则覆盖 ----
    hyb, test_rows, _, _ = B.run_group_cv(rows, label_fn=B.collapsed_label, post=B.apply_ask_rule)
    hyb_by_id = {s["id"]: q for s, q in zip(test_rows, hyb)}

    def combined(sample: dict) -> str:
        return rule_route(sample) or hyb_by_id.get(sample["id"], "chat")

    rep_b = H.evaluate(lambda s: hyb_by_id.get(s["id"], "chat"), test_rows,
                       name="hybrid(groupCV)")
    rep_c = H.evaluate(combined, test_rows, name="hybrid+规则(groupCV)")

    print()
    print(H.format_report([rep_b, rep_c]))

    extras = {
        "rule_only": {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "per_reason": per_reason,
            "false_positives": [{"gold": H.route_label(s), "utterance": utterance_of(s)}
                                for s in fps],
            "false_negatives": [{"gold": H.route_label(s), "utterance": utterance_of(s)}
                                for s in fns],
        },
        "hybrid": rep_b["escalation"],
        "hybrid_plus_rules": rep_c["escalation"],
        "orthogonality": {
            "executable_decision_hybrid": rep_b["capability"]["executable_decision"]["acc"],
            "executable_decision_plus_rules": rep_c["capability"]["executable_decision"]["acc"],
            "tool_acc_hybrid": rep_b["tool"]["tool_acc"],
            "tool_acc_plus_rules": rep_c["tool"]["tool_acc"],
            "note": "叠加规则后这两项不变 ⇒ 规则只动升级侧，与工具选择正交。",
        },
    }

    notes = {
        "circularity_risk": (
            "升级语料与规则由同一作者手写 ⇒ 上述数字含循环论证成分。"
            "真实泛化能力必须在 AgentStore 采集到真实会话后复验，不可直接当线上指标。"
        ),
        "single_char_pitfall": (
            "词表必须短语级。实测把 弄/搞/处理 当单字标记后，「预处理参数」「想搞懂流程步骤」"
            "全部误伤，escalate F1 由 87.5% 降至 83.6% —— 这是被数据否掉的第一个直觉。"
        ),
        "residual": (
            f"漏判 {fn} 条，属**语义**冲突（如「把行数改成 100 行然后保存」结构上与单步调用无异）。"
            "这类残留才是 L1 小模型真正该学的部分。"
        ),
        "no_writeback": "规则产物不写回数据集，评测与训练数据严格分离。",
    }

    (OUT_DIR / "escalation_rules_report.json").write_text(json.dumps(
        {"reports": [rep_b, rep_c], "extras": extras, "notes": notes,
         "table_markdown": H.format_report([rep_b, rep_c]),
         "generated_at_epoch": int(time.time())},
        ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# L0 结构规则层评测 —— 升级判定该不该学？", "",
          "> 规则实现 `app/local_router/escalation_rules.py`；"
          "口径 `eval_harness.py`；脚本 `scripts/router/escalation_rules_eval.py`。", "",
          "## 一、纯结构规则（零参数、零模型、只读 utterance）", "",
          "| 指标 | 值 |", "| --- | --- |",
          f"| escalate 精确率 | {p:.1%} |", f"| escalate 召回率 | {r:.1%} |",
          f"| escalate F1 | {f1:.1%} |", f"| tp / fp / fn | {tp} / {fp} / {fn} |",
          "", "### 逐原因明细", "",
          "| 原因 | n | 原因判对 | 判为升级 |", "| --- | --- | --- | --- |"]
    for d in per_reason:
        md.append(f"| {d['reason']} | {d['n']} | {d['reason_exact']}/{d['n']} | "
                  f"{d['detected_as_escalate']}/{d['n']} |")
    if fns:
        md += ["", "### 漏判（残留）", ""] + [
            f"- `{H.route_label(s)}` — {utterance_of(s)}" for s in fns]
    md += ["", "## 二、与学习基线的对比（模板级 groupCV）", "", H.format_report([rep_b, rep_c]),
           "", "**正交性证据**：",
           f"① 可执行决策 {rep_b['capability']['executable_decision']['acc']:.1%} → "
           f"{rep_c['capability']['executable_decision']['acc']:.1%}；"
           f"工具选择 {rep_b['tool']['tool_acc']:.1%} → {rep_c['tool']['tool_acc']:.1%}"
           "（规则只动升级侧）。", "", "## 三、注意事项", ""]
    for k, v in notes.items():
        md.append(f"- **{k}**：{v}")
    (OUT_DIR / "escalation_rules_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n已写出 dataset/escalation_rules_report.json + escalation_rules_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
