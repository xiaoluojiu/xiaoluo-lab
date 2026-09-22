"""Router 评估口径 —— 任何 Router 实现（规则 / 词法 / 神经网络）共用同一套指标。

只依赖标准库 + `dataset/*.jsonl`，不 import 任何模型框架。

标签空间（route 串）**从样本自身派生**，不硬编码任何工具名：

    chat                  intent=chat，不调工具
    escalate::<reason>    该交给更强模型（L2）
    call::<tool>          本地可直接执行
    ask::<tool>           工具已定但缺必填槽位 ⇒ 反问用户（零成本，≠升级）

两条刻意的口径设计：

1. **模板级（template-grouped）评估才是泛化真值。**
   样本级 holdout 与训练集共享 `meta.template`，只有槽位取值不同 ⇒ 表层 / 词法模型
   会因模板重叠而虚高。必须同时报告 group CV 数字，否则指标无意义。

2. **升级指标按「业务代价」拆开**，不给单一 accuracy：
   - `unnecessary_escalation_rate`：本该本地做却升级 ⇒ 多花 API 钱（用户最在意）
   - `missed_escalation_rate`：本该升级却本地硬做 ⇒ 降质量
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

KINDS = ("chat", "call", "ask", "escalate")


# ------------------------------------------------------------------ 载入

def load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(split: str = "train") -> list[dict]:
    """split ∈ {train, eval, holdout, all}。

    注意：`samples.jsonl` 已含全部条目（其 `split` 字段标了 train/eval/holdout），
    `holdout.jsonl` 是其中 47 条的副本 ⇒ holdout 只从副本读，避免重复计数。
    """
    if split == "holdout":
        return load_jsonl(DATASET_DIR / "holdout.jsonl")
    rows = load_jsonl(DATASET_DIR / "samples.jsonl")
    if split == "all":
        return rows
    return [r for r in rows if r.get("split") == split]


def load_boundary() -> list[dict]:
    return load_jsonl(DATASET_DIR / "boundary.jsonl")


def load_combo_violations() -> list[dict]:
    return load_jsonl(DATASET_DIR / "combo_violations.jsonl")


def load_tool_confusions() -> list[dict]:
    """`hard_negatives.jsonl` 实为「工具混淆对诊断」，不是判别训练样本。"""
    return load_jsonl(DATASET_DIR / "hard_negatives.jsonl")


# ------------------------------------------------------------------ 标签

def route_label(sample: dict) -> str:
    t = sample.get("target") or {}
    if t.get("escalate"):
        return "escalate::" + str(t.get("escalate_reason") or "unspecified")
    tool = t.get("tool")
    if tool:
        return ("ask::" if t.get("missing") else "call::") + str(tool)
    return "chat"


def kind_of(label: str) -> str:
    return str(label).split("::", 1)[0]


def tool_of(label: str) -> str | None:
    s = str(label)
    return s.split("::", 1)[1] if s.startswith(("call::", "ask::")) else None


def reason_of(label: str) -> str | None:
    s = str(label)
    return s.split("::", 1)[1] if s.startswith("escalate::") else None


def request_text(req: dict) -> str:
    """把 RouterRequest 摊平成模型输入文本。

    utterance 是主信号；「是否已绑定数据集 / 可见列 / 近期工具」都是 Router 真实可得的
    结构化信号，一并编码 —— 这是 implicit / follow_up 类样本能被判对的前提。

    ⚠️ 绑定状态必须**双向**编码：`ask::<tool>`（工具已定但缺 dataset_id）与
    `call::<tool>`（可直接执行）的差别**不在措辞里**，只在 `bound_dataset_id` 是否为
    null。若只写「已绑定」而不写「未绑定」，这两类在词面上完全同分布 ⇒ 模型不可能学会。
    """
    req = req or {}
    parts = [str(req.get("utterance") or "")]
    cols = req.get("available_columns") or []
    parts.append(f"可见列数={len(cols)}")
    if cols:
        parts.append("列 " + " ".join(map(str, cols)))
    parts.append("已绑定数据集" if req.get("bound_dataset_id") is not None else "未绑定数据集")
    recent = req.get("recent_tools") or []
    if recent:
        parts.append("近期 " + " ".join(map(str, recent)))
    return " ".join(parts).strip()


# ------------------------------------------------------------------ 分组

def group_key(sample: dict) -> str:
    """模板级分组的组名。

    - 有 `meta.template`（routing / slot_missing）⇒ 按模板分组：同模板只能落在
      train 或 test 之一，杜绝「换个槽位值」造成的近乎重复的泄漏（这是本数据集
      样本级 holdout 虚高的根因）。
    - 无模板（chat / escalate / implicit / follow_up 都是**逐条手写**的独立句子）
      ⇒ 每条自成一組。它们本就不存在可背的模板，样本级切分才是对「换一种说法」
      的诚实测试；若按类目整体成组会把整类从训练集抹掉，那是失真而非严格。
    """
    meta = sample.get("meta") or {}
    tpl = meta.get("template")
    if tpl:
        return "tpl::" + str(tpl)
    return "uniq::" + str(sample.get("id"))


def group_folds(samples: list[dict], n_folds: int = 5, seed: int = 20260922):
    """按 `group_key` 做 n 折切分（组不跨折），返回 [(train, test), ...]。"""
    groups = sorted({group_key(s) for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        buckets[i % n_folds].append(g)
    folds = []
    for i in range(n_folds):
        test_groups = set(buckets[i])
        train = [s for s in samples if group_key(s) not in test_groups]
        test = [s for s in samples if group_key(s) in test_groups]
        folds.append((train, test))
    return folds


# ------------------------------------------------------------------ 评估

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ★ 比值一律只做「高精度收尾」，不要先舍到 4 位再交给显示层：
#   显示层用的是 `:.1%`（还会再舍一次）⇒ 双重舍入会造出假数字。
#   实测：α=0.5 融合的 706/844 = 0.8364929，先舍 4 位变 0.8365，显示成 **83.7%**
#   而真值是 **83.6%**（同一份概率在另一个脚本里显示 83.6%，两份报告自相矛盾）。
ROUND_DP = 6


def _ratio(num: int, den: int) -> float:
    """比值 = num/den，保留 6 位小数（显示层再舍一次即可，误差 5e-7 远小于 0.1% 显示粒度）。"""
    return round(num / den, ROUND_DP) if den else 0.0


def evaluate(predict, samples: list[dict], name: str = "") -> dict:
    """`predict(sample) -> route 串`。返回可序列化的指标字典。"""
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    for s in samples:
        gold = route_label(s)
        try:
            pred = str(predict(s))
        except Exception as exc:  # noqa: BLE001
            pred = "__error__"
            errors.append(f"{type(exc).__name__}: {exc}")
        pairs.append((gold, pred))

    n = len(pairs)
    if not n:
        return {"name": name, "n": 0}

    exact = sum(1 for g, p in pairs if g == p)
    kind_hits = sum(1 for g, p in pairs if kind_of(g) == kind_of(p))

    by_kind: dict[str, dict] = {}
    for k in KINDS:
        tp = sum(1 for g, p in pairs if kind_of(p) == k and kind_of(g) == k)
        fp = sum(1 for g, p in pairs if kind_of(p) == k and kind_of(g) != k)
        fn = sum(1 for g, p in pairs if kind_of(g) == k and kind_of(p) != k)
        by_kind[k] = _prf(tp, fp, fn)

    # 工具选择：只在金标**有工具**（call / ask）的样本上衡量，chat/escalate 不参与
    tool_rows = [(g, p) for g, p in pairs if tool_of(g) is not None]
    tool_hits = sum(1 for g, p in tool_rows if tool_of(p) == tool_of(g))
    tool_hits_kind_ok = sum(1 for g, p in tool_rows
                            if kind_of(p) == kind_of(g) and tool_of(p) == tool_of(g))

    n_local = sum(1 for g, _ in pairs if kind_of(g) != "escalate")
    n_esc = sum(1 for g, _ in pairs if kind_of(g) == "escalate")
    unnecessary = sum(1 for g, p in pairs if kind_of(g) != "escalate" and kind_of(p) == "escalate")
    missed = sum(1 for g, p in pairs if kind_of(g) == "escalate" and kind_of(p) != "escalate")

    esc_rows = [(g, p) for g, p in pairs if kind_of(g) == "escalate"]
    reason_hits = sum(1 for g, p in esc_rows if reason_of(p) == reason_of(g))

    # 按升级原因分解：只看「有没有识别出该升级」（kind 级），
    # 因为理由细分准确率在样本量小的时候噪声极大，不适合当第一诊断指标。
    reason_recall: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g, p in esc_rows:
        r = reason_of(g) or "-"
        reason_recall[r][0] += 1
        reason_recall[r][1] += int(kind_of(p) == "escalate")

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s, (g, p) in zip(samples, pairs):
        c = s.get("category") or "?"
        by_cat[c][0] += 1
        by_cat[c][1] += int(g == p)

    conf_kind = Counter((kind_of(g), kind_of(p)) for g, p in pairs if kind_of(g) != kind_of(p))
    conf_route = Counter((g, p) for g, p in pairs if g != p and kind_of(g) == kind_of(p))

    # 能力分层：本任务其实是两种能力叠在一起，必须分开看
    #   ① 可执行决策 = 从 utterance 定位到具体工具（call / ask）
    #   ② 语义闸门   = 判断该闲聊还是该升级（chat / escalate）
    exec_rows = [(g, p) for g, p in pairs if kind_of(g) in ("call", "ask")]
    gate_rows = [(g, p) for g, p in pairs if kind_of(g) in ("chat", "escalate")]

    return {
        "name": name,
        "n": n,
        "route_acc": _ratio(exact, n),
        "kind_acc": _ratio(kind_hits, n),
        "capability": {
            "executable_decision": {"n": len(exec_rows), "acc": _ratio(
                sum(1 for g, p in exec_rows
                    if kind_of(p) == kind_of(g) and tool_of(p) == tool_of(g)), len(exec_rows))},
            "semantic_gate": {"n": len(gate_rows), "acc": _ratio(
                sum(1 for g, p in gate_rows if kind_of(p) == kind_of(g)), len(gate_rows))},
        },
        "by_kind": by_kind,
        "tool": {
            "n": len(tool_rows),
            "tool_acc": _ratio(tool_hits, len(tool_rows)),
            "tool_acc_given_kind": _ratio(tool_hits_kind_ok, len(tool_rows)),
        },
        "escalation": {
            "gold_escalate": n_esc,
            "gold_local": n_local,
            "unnecessary_escalation_rate": _ratio(unnecessary, n_local),
            "missed_escalation_rate": _ratio(missed, n_esc),
            "reason_acc": _ratio(reason_hits, n_esc),
            "by_reason_recall": {r: {"n": t, "recall": _ratio(h, t)}
                                 for r, (t, h) in sorted(reason_recall.items())},
        },
        "by_category": {c: {"n": t, "route_acc": _ratio(h, t)}
                        for c, (t, h) in sorted(by_cat.items())},
        "top_kind_confusions": [[a, b, c] for (a, b), c in conf_kind.most_common(8)],
        "top_route_confusions": [[a, b, c] for (a, b), c in conf_route.most_common(8)],
        "errors": errors[:5],
    }


def format_report(reports: list[dict]) -> str:
    """把多个 report 拼成一张横向对比表（markdown）。"""
    if not reports:
        return "（无报告）"
    cols = [r.get("name") or f"#{i}" for i, r in enumerate(reports)]
    out = ["| 指标 | " + " | ".join(cols) + " |",
           "| --- | " + " | ".join("---" for _ in cols) + " |"]

    def row(label, fn):
        out.append(f"| {label} | " + " | ".join(fn(r) for r in reports) + " |")

    row("样本数", lambda r: str(r.get("n", 0)))
    row("**整条决策准确率**", lambda r: f"{r.get('route_acc', 0):.1%}")
    row("决策类型准确率", lambda r: f"{r.get('kind_acc', 0):.1%}")
    row("① 可执行决策准确率", lambda r: f"{r['capability']['executable_decision']['acc']:.1%}")
    row("② 语义闸门准确率", lambda r: f"{r['capability']['semantic_gate']['acc']:.1%}")
    row("工具选择准确率", lambda r: f"{r['tool']['tool_acc']:.1%}")
    row("· 类型对时工具对", lambda r: f"{r['tool']['tool_acc_given_kind']:.1%}")
    row("chat F1", lambda r: f"{r['by_kind']['chat']['f1']:.1%}")
    row("call F1", lambda r: f"{r['by_kind']['call']['f1']:.1%}")
    row("ask F1", lambda r: f"{r['by_kind']['ask']['f1']:.1%}")
    row("escalate F1", lambda r: f"{r['by_kind']['escalate']['f1']:.1%}")
    row("无谓升级率 ↓", lambda r: f"{r['escalation']['unnecessary_escalation_rate']:.1%}")
    row("漏升级率 ↓", lambda r: f"{r['escalation']['missed_escalation_rate']:.1%}")
    row("升级理由准确率", lambda r: f"{r['escalation']['reason_acc']:.1%}")
    return "\n".join(out)


if __name__ == "__main__":
    for sp in ("train", "eval", "holdout"):
        rows = load_split(sp)
        buckets = Counter(route_label(s).split("::", 1)[0] for s in rows)
        print(f"{sp:8s} n={len(rows):4d}  {dict(buckets)}")
    print(f"group 数（模板级）: {len({group_key(s) for s in load_split('all')})}")
    print(f"工具混淆对（诊断文件）: {len(load_tool_confusions())}")
