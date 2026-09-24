"""Phase 0 · 统一本地模型基线评估 + 评估集冻结。

任务定位（见仓库顶层 long-text 指令）：
  在改动任何 Agent 架构代码之前，先用**真实可复现**的基线评估证明现状。
  本次不改模型、不改架构，只回答一个问题：**本地 Router 现在到底多准、成本多少。**

三项数据来源（诚实标注，禁止伪造）：
--------------------------------------------------------------------
1. `held_out_eval`  —— 仓库已有的**冻结评测集**（`dataset/samples.jsonl` 844 条，
   `dataset/holdout.jsonl` 132 条）。它有明确 gold label（route 串），
   模板级 groupCV 口径是**泛化真值**。这是本阶段的主评估集。
2. `real_user_logs`  —— 从 `data/agent_store.json` 提取的真实运行轨迹（有真实工具
   调用的请求）。用于 shadow 分析「如果当时让本地 Router 接管会怎样」。
   ⚠️ 诚实标注：这批日志去重后仅 5 种独特请求，多样性不足，只能作**补充证据**，
   不能替代 held_out_eval。
3. `historical_reports` —— 仓库已有的 Neural L1（`l1_report_*.md`）与 Fusion
   （`fusion_report_L4H256.md`）报告，由**系统 Python + torch** 训练产出。
   当前 backend/.venv 无 torch，无法在此复现 ⇒ 原样引用并标注 `frozen`，
   不伪造「重新训练过」。

为什么「不能只报告 held_out_eval」：held_out_eval 是模板合成数据，真实对话措辞分布
未知；real_user_logs 是真实分布但缺 gold label 且多样性差。两者互补，缺一不可。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_lexical as B  # noqa: E402
import eval_harness as H  # noqa: E402
import l1_data as D  # noqa: E402

FROZEN_DIR = _HERE / "eval_frozen"
DATASET_DIR = _HERE / "dataset"

# 真实工具前缀：过滤掉 test.* / wait.* 等演示工具，只保留真实能力。
REAL_PREFIX = ("dataset.", "ml.", "eda.", "report.", "data.", "workflow.", "agent.", "connector.")

# 历史报告的 frozen 数字（Neural L1 / Fusion，需 torch，本环境无法复现）。
# 全部来自仓库已有报告，不在此重算。
HISTORICAL = {
    "neural_l1": {
        "source": "dataset/l1_report_chroberta-L4H256.md",
        "route_acc_pure": 0.730,
        "route_acc_with_rules": 0.795,
        "tool_acc": 0.790,
        "semantic_gate_acc": 0.915,
        "params": 8773663,
        "infer_ms_per_sample": 4.19,
        "note": "chroberta-L4H256，cuda，需 torch；本环境无 torch，仅引用历史报告（frozen）。",
    },
    "fusion": {
        "source": "dataset/fusion_report_L4H256.md",
        "route_acc_alpha_0_5": 0.836,
        "route_acc_alpha_0_75": 0.857,
        "route_acc_alpha_0": 0.795,
        "route_acc_alpha_1": 0.807,
        "tool_acc": 0.833,
        "semantic_gate_acc": 0.966,
        "disagreement_route_level": 203,
        "perfect_selector_upper": 0.890,
        "note": "词法+神经融合，需 torch；本环境无 torch，仅引用历史报告（frozen）。",
    },
}


# ---------------------------------------------------------------------------
# 一、held_out_eval 评估（TF-IDF + LinearSVC，本环境可复现）
# ---------------------------------------------------------------------------


def eval_lexical_heldout() -> dict:
    """在冻结 holdout 与 groupCV 上复现词法基线。"""
    rows = H.load_split("all")
    holdout = H.load_split("holdout")

    # groupCV（泛化真值）
    preds, test_rows = _cv_predictions(rows)
    routes = {s["id"]: D.assemble_route(s, preds.get(s["id"])) for s in test_rows}
    rep_groupcv = H.evaluate(lambda s: routes[s["id"]], test_rows, name="tfidf_svc(groupCV)")

    # holdout（样本级，虚高，仅供对照）
    vec, clf = B.lexical_model(H.load_split("train"))
    rep_holdout = _eval_split(holdout, vec, clf, name="tfidf_svc(holdout)")

    return {
        "groupcv": rep_groupcv,
        "holdout": rep_holdout,
        "n_groupcv": rep_groupcv["n"],
        "n_holdout": rep_holdout["n"],
    }


def _cv_predictions(rows: list[dict], n_folds: int = 5, seed: int = 20260922):
    """与 train_runtime_l1.py 同口径的 groupCV 折外预测。"""
    preds: dict[str, str] = {}
    test_rows: list[dict] = []
    for train, test in H.group_folds(rows, n_folds=n_folds, seed=seed):
        fit_rows = D.trainable_rows(train)
        if not fit_rows or not test:
            continue
        vec, clf = B.lexical_model(fit_rows)
        matrix = vec.transform([H.request_text(s.get("request") or {}) for s in test])
        for sample, label in zip(test, (str(p) for p in clf.predict(matrix))):
            preds[sample["id"]] = label
        test_rows.extend(test)
    return preds, test_rows


def _eval_split(samples: list[dict], vec, clf, name: str) -> dict:
    matrix = vec.transform([H.request_text(s.get("request") or {}) for s in samples])
    preds = {s["id"]: str(p) for s, p in zip(samples, clf.predict(matrix))}
    routes = {s["id"]: D.assemble_route(s, preds[s["id"]]) for s in samples}
    return H.evaluate(lambda s: routes[s["id"]], samples, name=name)


# ---------------------------------------------------------------------------
# 二、EDA 子意图混淆矩阵（held_out_eval 上的 EDA 样本）
# ---------------------------------------------------------------------------


def eda_subintent_confusion() -> dict:
    """在 held_out_eval 里，对 EDA 能力域样本输出「子意图」混淆矩阵。

    子意图 = 金标工具（eda.describe / eda.distribution / eda.correlation /
    eda.outlier / eda.visualize / dataset.profile 等）vs 预测工具。
    任务要求的子意图：describe / distribution / correlation / outlier / profile / 其他。
    """
    rows = H.load_split("all")
    preds, test_rows = _cv_predictions(rows)

    eda_tools = {"eda.describe", "eda.distribution", "eda.correlation", "eda.outlier",
                 "eda.visualize", "dataset.profile"}

    def subintent(tool: str | None) -> str:
        if tool is None:
            return "(其他)"
        if tool == "eda.describe":
            return "describe"
        if tool == "eda.distribution":
            return "distribution"
        if tool == "eda.correlation":
            return "correlation"
        if tool == "eda.outlier":
            return "outlier"
        if tool == "dataset.profile":
            return "profile"
        if tool.startswith("eda."):
            return f"eda:{tool.split('.', 1)[1]}"
        return "(其他)"

    labels = ["describe", "distribution", "correlation", "outlier", "profile", "其他"]
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    n = 0
    for s in test_rows:
        gold = D.assemble_route(s, None)  # 只为拿 gold tool
        gold_label = H.route_label(s)
        gold_tool = H.tool_of(gold_label)
        if gold_tool is None:
            continue
        # 只统计 gold 是 EDA 相关工具的样本
        if gold_tool not in eda_tools and not gold_tool.startswith("eda."):
            continue
        pred_label = preds.get(s["id"])
        pred_tool = H.tool_of(pred_label) if pred_label else None
        g = subintent(gold_tool)
        p = subintent(pred_tool)
        if g not in labels:
            g = "其他"
        if p not in labels:
            p = "其他"
        matrix[g][p] += 1
        n += 1

    return {"n": n, "labels": labels, "matrix": matrix,
            "note": "金标为 EDA 能力域样本（groupCV 折外预测），gold 与 pred 均为工具名归一化后的子意图。"}


# ---------------------------------------------------------------------------
# 三、real_user_logs shadow 分析（真实日志）
# ---------------------------------------------------------------------------


def extract_real_logs(agent_store_path: Path) -> list[dict]:
    """从 agent_store.json 提取有真实工具调用的请求。

    每条记录：request（真实用户文本）+ executed_tool（当时系统实际执行的第一步真实工具）。
    注意：executed_tool 是**当时规则路由/LLM 规划**的产物，不是 ground truth ——
    它可能本身就有错（例如「看看这批数据的分布」被路由到 dataset.inspect）。
    因此这里只能做「如果让本地 Router 接管会怎样」的 shadow 对照，不能当准确率金标。
    """
    if not agent_store_path.exists():
        return []
    data = json.loads(agent_store_path.read_text(encoding="utf-8"))
    samples: list[dict] = []
    for r in data.get("runs", []):
        req = str(r.get("user_request") or "").strip()
        if not req:
            continue
        tools = [tc.get("tool") for tc in r.get("tool_calls", [])
                 if str(tc.get("tool", "")).startswith(REAL_PREFIX)]
        if not tools:
            continue
        samples.append({"request": req, "executed_tool": tools[0],
                        "status": r.get("status"), "run_id": r.get("id")})
    return samples


def shadow_real_logs(samples: list[dict]) -> dict:
    """用本地 Router 对真实请求预测，与真实日志实际行为对照。"""
    from app.local_router.router import route_request

    if not samples:
        return {"n": 0}

    verdicts = Counter()
    confidences = []
    detail = []
    for s in samples:
        try:
            decision = route_request({"utterance": s["request"],
                                      "bound_dataset_id": None})
            route = _decision_route(decision)
            conf = float(decision.confidence)
        except Exception as exc:  # noqa: BLE001
            route = "__error__"
            conf = 0.0
            decision = None
        confidences.append(conf)
        executed = s["executed_tool"]
        verdict = _shadow_verdict(route, executed)
        verdicts[verdict] += 1
        detail.append({"request": s["request"], "executed_tool": executed,
                       "router_route": route, "confidence": round(conf, 4),
                       "verdict": verdict})

    n = len(samples)
    return {
        "n": n,
        "n_unique_requests": len({s["request"] for s in samples}),
        "verdicts": dict(verdicts),
        "verdict_ratio": {k: round(v / n, 4) for k, v in verdicts.items()},
        "confidence_mean": round(sum(confidences) / n, 4),
        "detail": detail,
        "note": "executed_tool 是当时规则路由的产物（可能本身有错），非 ground truth；"
                "此表只回答「本地 Router 若接管会怎样」，不是准确率。",
    }


def _decision_route(decision) -> str:
    from app.local_router.router import decision_to_route
    try:
        return decision_to_route(decision)
    except Exception:  # noqa: BLE001
        return "__error__"


def _shadow_verdict(route: str, executed: str) -> str:
    kind = route.split("::", 1)[0]
    if kind == "escalate":
        return "中性(升级)"
    if kind == "ask":
        return "待确认(反问)"
    if kind == "chat":
        return "会出错(误判闲聊)"
    if kind == "call":
        tool = route.split("::", 1)[1] if "::" in route else ""
        return "可省" if tool == executed else "会出错(工具不符)"
    return "异常"


# ---------------------------------------------------------------------------
# 四、冻结评估集
# ---------------------------------------------------------------------------


def freeze_eval_set() -> dict:
    """把本阶段用的评估集冻结：held_out_eval 签名 + real_user_logs 快照。"""
    import hashlib

    frozen: dict = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "held_out_eval": {},
        "real_user_logs": {},
    }
    for name in ("samples.jsonl", "holdout.jsonl"):
        path = DATASET_DIR / name
        if path.exists():
            blob = path.read_bytes()
            frozen["held_out_eval"][name] = {
                "bytes": len(blob),
                "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
                "lines": blob.count(b"\n"),
            }
    store = _BACKEND_ROOT / "data" / "agent_store.json"
    if store.exists():
        blob = store.read_bytes()
        frozen["real_user_logs"]["agent_store.json"] = {
            "bytes": len(blob),
            "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
        }
    # 写快照
    snapshot_path = FROZEN_DIR / "eval_set_manifest.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    frozen["_written"] = str(snapshot_path)
    return frozen


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.1%}"


def build_report(lexical: dict, eda: dict, shadow: dict, frozen: dict) -> str:
    g = lexical["groupcv"]
    h = lexical["holdout"]
    n = HISTORICAL["neural_l1"]
    f = HISTORICAL["fusion"]

    L: list[str] = []
    L += [
        "# Phase 0 · 本地模型基线评估报告",
        "",
        "> 脚本 `scripts/router/baseline_phase0.py`；口径 `eval_harness.py` + `l1_data.assemble_route`。",
        f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}。",
        "",
        "## 数据来源（诚实标注）",
        "",
        "| 来源 | 标记 | 规模 | 说明 |",
        "| --- | --- | --- | --- |",
        f"| 冻结评测集 | `held_out_eval` | {lexical['n_groupcv']} 条(groupCV) + {lexical['n_holdout']} 条(holdout) | 模板合成，有 gold label，主评估集 |",
        f"| 真实运行轨迹 | `real_user_logs` | {shadow['n']} 条（去重 {shadow.get('n_unique_requests', 0)} 种） | 真实分布，但多样性不足，仅作补充 |",
        f"| 历史 Neural/Fusion 报告 | `frozen` | 844 条 | 需 torch，本环境无，引用历史产物 |",
        "",
        "> ⚠️ **real_user_logs 诚实标注**：`data/agent_store.json` 去重后仅 "
        f"{shadow.get('n_unique_requests', 0)} 种独特请求（如「查看数据」「检查一下数据质量」「看看这批数据的分布」），"
        "高度重复、多样性不足，不能替代 held_out_eval。它只能回答「本地 Router 若在真实线上接管会怎样」。",
        "",
        "## 一、TF-IDF + LinearSVC（本环境可复现）",
        "",
        "| 指标 | groupCV(泛化真值) | holdout(样本级·虚高) |",
        "| --- | --- | --- |",
        f"| 样本数 | {g['n']} | {h['n']} |",
        f"| **route accuracy（整条决策）** | {_pct(g['route_acc'])} | {_pct(h['route_acc'])} |",
        f"| 决策类型准确率 | {_pct(g['kind_acc'])} | {_pct(h['kind_acc'])} |",
        f"| ① 可执行决策准确率 | {_pct(g['capability']['executable_decision']['acc'])} | {_pct(h['capability']['executable_decision']['acc'])} |",
        f"| ② 语义闸门准确率 | {_pct(g['capability']['semantic_gate']['acc'])} | {_pct(h['capability']['semantic_gate']['acc'])} |",
        f"| **tool accuracy** | {_pct(g['tool']['tool_acc'])} | {_pct(h['tool']['tool_acc'])} |",
        f"| · 类型对时工具对 | {_pct(g['tool']['tool_acc_given_kind'])} | {_pct(h['tool']['tool_acc_given_kind'])} |",
        f"| 无谓升级率 ↓ | {_pct(g['escalation']['unnecessary_escalation_rate'])} | {_pct(h['escalation']['unnecessary_escalation_rate'])} |",
        f"| 漏升级率 ↓ | {_pct(g['escalation']['missed_escalation_rate'])} | {_pct(h['escalation']['missed_escalation_rate'])} |",
        "",
        "### 高置信度阈值下的 precision / coverage",
        "",
        "（门控阈值扫描：把置信度低于 θ 的「可本地执行」转为升级，看 precision 与 coverage 的权衡。"
        "详细操作点见 `dataset/baseline_report.md` 的「弃权阈值扫描」。）",
        "",
    ]

    # 高置信度 precision/coverage：用 groupCV 的逐条置信度重算（近似，用 ConstantModel 注入）
    L += _confidence_coverage_section(lexical)

    L += [
        "",
        "## 二、Neural L1（frozen，历史报告）",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 来源 | `{n['source']}` |",
        f"| route accuracy（纯模型） | {_pct(n['route_acc_pure'])} |",
        f"| route accuracy（+规则） | {_pct(n['route_acc_with_rules'])} |",
        f"| tool accuracy | {_pct(n['tool_acc'])} |",
        f"| 语义闸门准确率 | {_pct(n['semantic_gate_acc'])} |",
        f"| 参数量 | {n['params']:,} |",
        f"| 推理延迟 | {n['infer_ms_per_sample']} ms·条⁻¹ |",
        f"| 备注 | {n['note']} |",
        "",
        "## 三、Fusion（frozen，历史报告）",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 来源 | `{f['source']}` |",
        f"| route accuracy（α=0.5 融合） | {_pct(f['route_acc_alpha_0_5'])} |",
        f"| route accuracy（α=0.75） | {_pct(f['route_acc_alpha_0_75'])} |",
        f"| route accuracy（α=0 纯神经） | {_pct(f['route_acc_alpha_0'])} |",
        f"| route accuracy（α=1 纯词法） | {_pct(f['route_acc_alpha_1'])} |",
        f"| tool accuracy | {_pct(f['tool_acc'])} |",
        f"| 语义闸门准确率 | {_pct(f['semantic_gate_acc'])} |",
        f"| 路线级分歧 | {f['disagreement_route_level']} / 844 |",
        f"| 完美选择器上限（不可达） | {_pct(f['perfect_selector_upper'])} |",
        f"| 备注 | {f['note']} |",
        "",
        "## 四、TF-IDF vs Neural vs Fusion 对比",
        "",
        "| 方案 | route accuracy | tool accuracy | 语义闸门 | 部署代价 |",
        "| --- | --- | --- | --- | --- |",
        f"| TF-IDF + LinearSVC | {_pct(g['route_acc'])} | {_pct(g['tool']['tool_acc'])} | {_pct(g['capability']['semantic_gate']['acc'])} | 3.5MB / 0.008ms / 零新依赖 |",
        f"| Neural L1（+规则） | {_pct(n['route_acc_with_rules'])} | {_pct(n['tool_acc'])} | {_pct(n['semantic_gate_acc'])} | 需 torch ≈2.5GB |",
        f"| Fusion（α=0.5） | {_pct(f['route_acc_alpha_0_5'])} | {_pct(f['tool_acc'])} | {_pct(f['semantic_gate_acc'])} | 需 torch 或 ONNX |",
        "",
        "### 差值",
        "",
        f"- Fusion − TF-IDF = {f['route_acc_alpha_0_5'] - g['route_acc']:+.1%}（route accuracy）",
        f"- Fusion − Neural = {f['route_acc_alpha_0_5'] - n['route_acc_with_rules']:+.1%}（route accuracy）",
        f"- Neural − TF-IDF = {n['route_acc_with_rules'] - g['route_acc']:+.1%}（route accuracy）",
        "",
        "## 五、EDA 子意图混淆矩阵（held_out_eval groupCV）",
        "",
        _eda_confusion_table(eda),
        "",
        "## 六、real_user_logs shadow 分析（本地 Router 若接管会怎样）",
        "",
        _shadow_table(shadow),
        "",
        "## 七、评估集冻结",
        "",
        f"- 冻结清单已写：`{frozen.get('_written', '')}`",
        f"- held_out_eval 指纹：samples.jsonl sha256[:16] = "
        f"{frozen['held_out_eval'].get('samples.jsonl', {}).get('sha256_16', 'N/A')}",
        f"- real_user_logs 指纹：agent_store.json sha256[:16] = "
        f"{frozen['real_user_logs'].get('agent_store.json', {}).get('sha256_16', 'N/A')}",
        "",
        "## 结论",
        "",
        f"- 本地 Router 现状（TF-IDF groupCV）：route accuracy **{_pct(g['route_acc'])}**，"
        f"tool accuracy **{_pct(g['tool']['tool_acc'])}**。",
        f"- 语义闸门（{_pct(g['capability']['semantic_gate']['acc'])}）与升级识别是当前主要短板（见 baseline_report 的 escalate F1 51.3%）。",
        f"- 真实日志里「看看这批数据的分布」被路由到 `dataset.inspect`（应为 EDA 分布类），"
        "佐证了「关键词规则 → 单一 Tool」路径的局限 —— 这正是后续 TaskSpec 架构要修复的。",
        "",
    ]
    return "\n".join(L)


def _confidence_coverage_section(lexical: dict) -> list[str]:
    """高置信度阈值下的 precision / coverage（groupCV 折外预测，记录 label + 置信度）。

    precision = 保留（conf ≥ θ）样本里「最终 route 全对」的比例；
    coverage/retention = 保留样本占「可本地执行（非升级）」样本的比例。
    """
    import numpy as np
    from app.local_router.scoring import proba_from_scores, top1

    rows = H.load_split("all")
    records: list[dict] = []  # id, gold, label, conf
    for train, test in H.group_folds(rows, n_folds=5, seed=20260922):
        fit_rows = D.trainable_rows(train)
        if not fit_rows or not test:
            continue
        vec, clf = B.lexical_model(fit_rows)
        label_space = D.l1_label_space()
        for sample in test:
            text = H.request_text(sample.get("request") or {})
            matrix = vec.transform([text])
            scores = np.asarray(clf.decision_function(matrix), dtype=float)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            classes = [str(c) for c in getattr(clf, "classes_", [])]
            proba = proba_from_scores(classes, scores, label_space)
            label, conf = top1(proba, label_space)
            gold = H.route_label(sample)
            records.append({"gold": gold, "label": label, "conf": float(conf)})

    local_records = [r for r in records if H.kind_of(r["gold"]) != "escalate"]
    lines: list[str] = []
    for theta in (0.3, 0.5, 0.7):
        kept = [r for r in local_records if r["conf"] >= theta]
        correct = 0
        for r in kept:
            # 用三层合成还原最终 route（gold 与 pred 都是 assemble_route 之后的口径）
            # 这里 gold 已经是 route_label；pred 需要经 assemble_route，但缺 sample 上下文。
            # 简化口径：比较「工具 + 是否闲聊」这两层（L0 升级已在上游用规则兜住，等效）。
            pred_kind = H.kind_of(r["label"])
            gold_kind = H.kind_of(r["gold"])
            if pred_kind == "chat" and gold_kind == "chat":
                correct += 1
            elif pred_kind == "call" and gold_kind in ("call", "ask"):
                correct += int(H.tool_of(r["label"]) == H.tool_of(r["gold"]))
        precision = correct / len(kept) if kept else 0.0
        coverage = len(kept) / max(len(local_records), 1)
        lines.append(f"| θ={theta} | {precision:.1%} | {coverage:.1%} | {len(kept)} |")
    return [
        "| 阈值 θ | 高置信度 precision | coverage/retention | 保留样本数 |",
        "| --- | --- | --- | --- |",
        "（precision 按「闲聊/工具选择」两层口径；L0 升级规则先于模型，故升级样本不计入本地执行）",
    ] + lines


def _eda_confusion_table(eda: dict) -> str:
    labels = eda["labels"]
    lines = ["| 金标\\预测 | " + " | ".join(labels) + " |",
             "| --- | " + " | ".join("---" for _ in labels) + " |"]
    m = eda["matrix"]
    for g in labels:
        lines.append(f"| {g} | " + " | ".join(str(m[g].get(p, 0)) for p in labels) + " |")
    lines.append("")
    lines.append(f"样本数 {eda['n']}；{eda['note']}")
    return "\n".join(lines)


def _shadow_table(shadow: dict) -> str:
    if not shadow.get("n"):
        return "（无真实日志）"
    lines = ["| 后果 | 条数 | 占比 |",
             "| --- | --- | --- |"]
    for k in ("可省", "可省(部分)", "会出错(误判闲聊)", "会出错(工具不符)", "中性(升级)", "待确认(反问)", "异常"):
        if k in shadow["verdicts"]:
            lines.append(f"| {k} | {shadow['verdicts'][k]} | {shadow['verdict_ratio'].get(k, 0):.1%} |")
    lines += [
        "",
        f"- 置信度均值 {shadow.get('confidence_mean')}",
        f"- 去重后独特请求 {shadow.get('n_unique_requests')} 种",
        "",
        "### 逐条明细（前 30）",
        "",
        "| 请求 | 当时实际执行 | 本地 Router 判定 | 置信度 | 后果 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in shadow["detail"][:30]:
        lines.append(f"| {d['request'][:24]} | {d['executed_tool']} | {d['router_route']} | "
                     f"{d['confidence']} | {d['verdict']} |")
    lines.append("")
    lines.append(shadow.get("note", ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 统一基线评估 + 评估集冻结")
    parser.add_argument("--no-shadow", action="store_true", help="跳过 real_user_logs shadow 分析")
    parser.add_argument("--store", default="", help="agent_store.json 路径（默认 backend/data/）")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("== Phase 0 基线评估 ==")

    print("\n[1/4] TF-IDF + LinearSVC（held_out_eval）…")
    lexical = eval_lexical_heldout()

    print("[2/4] EDA 子意图混淆矩阵…")
    eda = eda_subintent_confusion()

    print("[3/4] real_user_logs shadow 分析…")
    store = Path(args.store) if args.store else (_BACKEND_ROOT / "data" / "agent_store.json")
    shadow = shadow_real_logs(extract_real_logs(store)) if not args.no_shadow else {"n": 0}

    print("[4/4] 冻结评估集…")
    frozen = freeze_eval_set()

    report = build_report(lexical, eda, shadow, frozen)
    out_md = FROZEN_DIR / "baseline_report_phase0.md"
    out_md.write_text(report, encoding="utf-8")

    # JSON 版
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {
            "held_out_eval": {"n_groupcv": lexical["n_groupcv"], "n_holdout": lexical["n_holdout"]},
            "real_user_logs": {"n": shadow.get("n", 0), "n_unique": shadow.get("n_unique_requests", 0)},
            "historical": HISTORICAL,
        },
        "tfidf": {"groupcv": lexical["groupcv"], "holdout": lexical["holdout"]},
        "eda_subintent": eda,
        "shadow": shadow,
        "frozen": frozen,
    }
    (FROZEN_DIR / "baseline_report_phase0.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n报告已写：{out_md}")
    print(report)
    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
