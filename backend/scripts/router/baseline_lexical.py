"""词法基线 Router —— 用客观数字回答「这个任务到底需不需要小模型」。

基线一览（共用 `eval_harness` 口径）：

  majority              恒预测训练集众数 route（绝对下限）
  tfidf_svc/split       样本级 split 拟合 → holdout 测试
  tfidf_svc/groupCV     按 `meta.template` 分组 5 折 CV ⇒ **泛化真值**
  hybrid/groupCV        同上，但把 `ask`（反问用户）从学习目标里摘掉，改由确定性规则算
  abstain/groupCV       再把「无工具可匹配」的升级（ambiguous / out_of_scope）改为**弃权**判定

为什么必须看 groupCV：样本级 holdout 与训练集共享模板（只有槽位取值不同），表层/词法
模型会「背模板」而虚高（实测 85% vs 48%）。不报 groupCV 的数字没有意义。

为什么要把三类决策拆开：本任务其实是三种性质不同的决策叠在一起 ——
  ① 语言问题：这句话在说什么？（工具选择）—— 词法模型能做得不错
  ② 语言问题：这句话该闲聊还是该升级？（语义闸门）—— 需要语义能力
  ③ **符号问题**：选定工具后必填槽位是否齐备？（执行还是反问）
     `missing = required_params(tool) ∖ resolved` 是纯函数，当学习目标时 F1 会随
     分组口径在 8.9%~47% 间剧烈摆动 ⇒ 摘出去用规则算
  ★ 第四类：`ambiguous` / `out_of_scope` 这两个升级原因**按定义就是「没有任何工具
     匹配得上」**，本质是**检索拒识决策**而非语句分类。实测按分类学识别率只有
     17% / 22%，而能从措辞识别的 `conflict` 有 80% ⇒ 改用「top1 决策分数 < θ ⇒
     弃权」。θ 逐折在**内层 group CV 的 out-of-fold 分数**上取 binary F1 最优点，
     绝不触碰外层 test。

零 GPU、零 torch，模型体积几 MB。**若它已足够好，L1 小模型的意义就要重新评估**。
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_harness as H  # noqa: E402
# 契约走 router_contract（实时优先/快照兜底）：训练环境（系统 Python，有 torch 无 polars）
# 里也能重算基线，从而支持「同一次运行内对比」——消除 sklearn 版本差异。
from router_contract import CONTRACT as C  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

OUT_DIR = H.DATASET_DIR

# 规则能直接判定的槽位 —— 只有能从 RouterRequest 无条件读出的那些。
_RULE_RESOLVABLE = {"dataset_id"}

# 「没有工具匹配得上」的升级原因 —— 由弃权处理，不作为学习标签。
NO_TOOL_ESCALATION_REASONS = {"ambiguous", "out_of_scope"}


# ------------------------------------------------------------------ 标签变换

def collapse_ask(label: str) -> str:
    """route 串层面：把 `ask::X` 并回 `call::X`。"""
    return "call::" + label.split("::", 1)[1] if label.startswith("ask::") else label


def collapsed_label(sample: dict) -> str:
    """样本层面：只让模型学「选哪个工具」，不问「槽位是否齐备」。"""
    return collapse_ask(H.route_label(sample))


def abstain_label(sample: dict) -> str:
    """弃权变体的学习标签：ask 并回 call，escalate 合成单一类（理由不参与学习）。"""
    lbl = collapsed_label(sample)
    return "escalate" if H.kind_of(lbl) == "escalate" else lbl


def is_no_tool_escalation(sample: dict) -> bool:
    g = H.route_label(sample)
    return (H.kind_of(g) == "escalate"
            and (H.reason_of(g) or "") in NO_TOOL_ESCALATION_REASONS)


def apply_ask_rule(sample: dict, label: str, score=None) -> str:
    """确定性反问规则：工具已定，但 RouterRequest 里读得出必填槽位缺失 ⇒ 改为 ask。

    这一步替代了让模型直接预测 `missing` —— 与 `contract.decide_route()` 的职责一致。
    """
    kind, _, tool = str(label).partition("::")
    if kind != "call" or not tool:
        return label
    req = sample.get("request") or {}
    resolved = {"dataset_id"} if req.get("bound_dataset_id") is not None else set()
    missing = [p for p in C.required_params(tool)
               if p not in resolved and p in _RULE_RESOLVABLE]
    return ("ask::" + tool) if missing else label


def make_abstain_post(theta: float):
    """弃权后置：top1 分数低于 θ ⇒ 交 L2（理由占位 ambiguous）。"""
    def _post(sample: dict, label: str, score=None) -> str:
        if score is not None and score < theta:
            return "escalate::ambiguous"
        return apply_ask_rule(sample, label, score)
    return _post


# ------------------------------------------------------------------ 拟合

def _abstain_score(df, mode: str) -> list[float]:
    """「有多确信有工具适用」的分数 —— 越小越该弃权。

    top1   = 最高决策分数（绝对量；受各类别边际尺度影响，可能偏弱）
    margin = 最高与次高之差（相对量；对「哪个都不像」更敏感）
    """
    if getattr(df, "ndim", 1) == 1:
        df = df.reshape(-1, 1)
    if mode == "margin" and df.shape[1] >= 2:
        out = []
        for row in df:
            top2 = sorted(float(v) for v in row)[-2:]
            out.append(top2[1] - top2[0])
        return out
    return [float(v) for v in df.max(axis=1)]


def lexical_model(train: list[dict], text_fn=None):
    """拟合词法模型，返回 (vec, clf)。

    **融合实验必须复用这个函数**，否则抄一份 TfidfVectorizer / LinearSVC 的配置，
    两边一旦漂移，融合的对比就失去意义。

    `text_fn` 缺省用 `H.request_text`。只有做「输入口径消融」（例如线上 shadow 拿不到
    `available_columns`）时才传别的 —— 消融与主实验因此共用同一套超参，不会各写一份。
    """
    to_text = text_fn or H.request_text
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True)
    clf = LinearSVC(C=1.0, random_state=0)
    x = vec.fit_transform([to_text(s.get("request") or {}) for s in train])
    clf.fit(x, [collapsed_label(s) for s in train])
    return vec, clf


def lexical_score_matrix(vec, clf, rows: list[dict]):
    """返回 (类别顺序, 决策分数矩阵)。

    ⚠️ `clf.classes_` 只含**训练折里出现过**的类别，可能少于全部 31 类；
    融合时必须按完整标签空间对齐（缺失列填 -inf），否则下标会错位。
    """
    import numpy as np

    x = vec.transform([H.request_text(s.get("request") or {}) for s in rows])
    df = clf.decision_function(x)
    if getattr(df, "ndim", 1) == 1:
        df = df.reshape(-1, 1)
    return [str(c) for c in clf.classes_], np.asarray(df, dtype=float)


def fit_predict(train, test, label_fn=None, post=None, score_mode="top1"):
    """返回 (预测列表, 弃权分数, 模型 bytes, 训练秒, 推理 ms/条, 特征维数)。"""
    label_fn = label_fn or H.route_label
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True)
    clf = LinearSVC(C=1.0, random_state=0)

    t0 = time.perf_counter()
    x_tr = vec.fit_transform([H.request_text(s.get("request") or {}) for s in train])
    clf.fit(x_tr, [label_fn(s) for s in train])
    train_s = time.perf_counter() - t0

    x_te = vec.transform([H.request_text(s.get("request") or {}) for s in test])
    t1 = time.perf_counter()
    df = clf.decision_function(x_te)
    scores = _abstain_score(df, score_mode)
    raw = [str(p) for p in clf.predict(x_te)]
    infer_ms = (time.perf_counter() - t1) * 1000.0 / max(len(test), 1)

    if post is not None:
        raw = [post(s, p, sc) for s, p, sc in zip(test, raw, scores)]
    return raw, scores, len(pickle.dumps((vec, clf))), train_s, infer_ms, x_tr.shape[1]


def evaluate_preds(preds, test, name):
    table = {s["id"]: p for s, p in zip(test, preds)}
    return H.evaluate(lambda s: table.get(s["id"], "chat"), test, name=name)


def calibrate_theta(rows, label_fn, drop_fn=None, n_folds=3, seed=777, n_grid=40,
                    score_mode="top1") -> float:
    """内层 group CV 的 OOF 分数上取 binary F1 最优的弃权阈值 —— 不碰外层 test。"""
    pairs: list[tuple[float, bool]] = []
    for tr, va in H.group_folds(rows, n_folds=n_folds, seed=seed):
        tr_fit = [s for s in tr if not (drop_fn and drop_fn(s))] if drop_fn else tr
        if not tr_fit or not va:
            continue
        _, sc, _, _, _, _ = fit_predict(tr_fit, va, label_fn=label_fn, score_mode=score_mode)
        pairs.extend((v, H.kind_of(H.route_label(s)) == "escalate") for s, v in zip(va, sc))
    if not pairs:
        return 0.0
    lo = min(v for v, _ in pairs)
    hi = max(v for v, _ in pairs)
    cands = [lo - 1e-6] + [lo + (hi - lo) * q / n_grid for q in range(1, n_grid + 1)]
    best = (lo - 1e-6, -1.0)
    for th in cands:
        tp = sum(1 for v, g in pairs if g and v < th)
        fp = sum(1 for v, g in pairs if not g and v < th)
        fn = sum(1 for v, g in pairs if g and v >= th)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best[1]:
            best = (th, f1)
    return best[0]


def run_group_cv(rows, label_fn=None, post=None, post_factory=None, drop_fn=None,
                 n_folds=5, seed=20260922, score_mode="top1"):
    """返回 (预测, 测试样本, 各折信息, 弃权分数)。θ 逐折在折内训练集上校准。"""
    preds, test_rows, folds, scores = [], [], [], []
    for i, (tr, te) in enumerate(H.group_folds(rows, n_folds=n_folds, seed=seed), 1):
        tr_fit = [s for s in tr if not (drop_fn and drop_fn(s))] if drop_fn else tr
        p_fn = post_factory(tr_fit) if post_factory else post
        pr, sc, _, _, _, _ = fit_predict(tr_fit, te, label_fn=label_fn, post=p_fn,
                                        score_mode=score_mode)
        preds.extend(pr)
        test_rows.extend(te)
        scores.extend(sc)
        folds.append({"fold": i, "train": len(tr_fit), "test": len(te),
                      "test_groups": len({H.group_key(s) for s in te})})
    return preds, test_rows, folds, scores


def is_actionable(sample: dict) -> bool:
    """平台是否**有工具能接**这句话。这是「拒识」问题，不是语句分类问题。"""
    return H.kind_of(H.route_label(sample)) in ("call", "ask")


def _fit(train_rows, label_fn):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True)
    clf = LinearSVC(C=1.0, random_state=0)
    x = vec.fit_transform([H.request_text(s.get("request") or {}) for s in train_rows])
    clf.fit(x, [label_fn(s) for s in train_rows])
    return vec, clf


def _pred(vec, clf, rows) -> list[str]:
    if not rows:
        return []
    x = vec.transform([H.request_text(s.get("request") or {}) for s in rows])
    return [str(p) for p in clf.predict(x)]


def run_hier_group_cv(rows, n_folds=5, seed=20260922):
    """层级式三段：① 门（有没有工具能接？）→ ② 工具分类 / ③ chat-or-escalate。

    与「一个 32 类分类器顺便学出拒识」的关键差别：**门是专门的二分类任务**，
    正样本＝有工具可接（call/ask），负样本＝闲聊与升级。这是在验证一个假设：
    拒识学不好，是因为**表述错了**（被塞进多分类），而不是信号本身不存在。

    返回 (预测, 测试样本, 各折信息, 门判断对 [gold_actionable, pred_actionable])。
    """
    preds, test_rows, folds, gate = [], [], [], []
    for i, (tr, te) in enumerate(H.group_folds(rows, n_folds=n_folds, seed=seed), 1):
        tr_a = [s for s in tr if is_actionable(s)]
        tr_n = [s for s in tr if not is_actionable(s)]
        if not tr_a or not tr_n:
            continue
        v_gate, c_gate = _fit(tr, lambda s: "actionable" if is_actionable(s) else "none")
        v_tool, c_tool = _fit(tr_a, abstain_label)
        v_none, c_none = _fit(tr_n, lambda s: "chat" if H.kind_of(H.route_label(s)) == "chat"
                              else "escalate")

        gate_hat = _pred(v_gate, c_gate, te)
        gate.extend((is_actionable(s), p == "actionable") for s, p in zip(te, gate_hat))

        idx_a = [k for k, p in enumerate(gate_hat) if p == "actionable"]
        idx_n = [k for k, p in enumerate(gate_hat) if p != "actionable"]
        out: list[str | None] = [None] * len(te)
        for k, p in zip(idx_a, _pred(v_tool, c_tool, [te[k] for k in idx_a])):
            out[k] = apply_ask_rule(te[k], p)
        for k, p in zip(idx_n, _pred(v_none, c_none, [te[k] for k in idx_n])):
            out[k] = p
        preds.extend([p if p is not None else "chat" for p in out])
        test_rows.extend(te)
        folds.append({"fold": i, "train": len(tr), "test": len(te),
                      "test_groups": len({H.group_key(s) for s in te})})
    return preds, test_rows, folds, gate


def escalate_sweep(test_rows, scores, quantiles=(0, 5, 10, 15, 20, 25, 30, 40, 50)) -> list[dict]:
    """把弃权阈值当操作点扫描：量化「多拦一点升级」与「误拦」的取舍。"""
    if not scores:
        return []
    ordered = sorted(scores)
    n = len(ordered)
    out = []
    for q in quantiles:
        th = ordered[min(int(q / 100.0 * n), n - 1)]
        tp = fp = fn = 0
        for s, v in zip(test_rows, scores):
            gold = H.kind_of(H.route_label(s)) == "escalate"
            pred = v < th
            if gold and pred:
                tp += 1
            elif gold:
                fn += 1
            elif pred:
                fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"theta_quantile": q, "theta": round(th, 3),
                    "escalate_precision": round(prec, 4), "escalate_recall": round(rec, 4),
                    "escalate_f1": round(f1, 4),
                    "abstain_rate": round((fp + tp) / max(len(scores), 1), 4)})
    return out


# ------------------------------------------------------------------ 主流程

def main() -> int:
    all_rows = H.load_split("all")
    train = H.load_split("train")
    holdout = H.load_split("holdout")

    print(f"samples: all={len(all_rows)} train={len(train)} holdout={len(holdout)}")

    reports, extras = [], {}

    # ---- 1) majority 下限 ----
    mode = Counter(H.route_label(s) for s in train).most_common(1)[0][0]
    reports.append(H.evaluate(lambda s: mode, holdout, name="majority(holdout)"))
    extras["majority_label"] = mode

    # ---- 2) 纯词法：样本级 holdout ----
    p_hold, _, size, train_s, infer_ms, dim = fit_predict(train, holdout)
    reports.append(evaluate_preds(p_hold, holdout, name="tfidf_svc(holdout)"))

    # ---- 3) 纯词法 + 反问规则：样本级 holdout ----
    p_hold_h, _, _, _, _, _ = fit_predict(train, holdout,
                                          label_fn=collapsed_label, post=apply_ask_rule)
    reports.append(evaluate_preds(p_hold_h, holdout, name="hybrid(holdout)"))

    # ---- 4) 纯词法：模板级 groupCV ----
    p_cv, cv_test, folds, _ = run_group_cv(all_rows)
    reports.append(evaluate_preds(p_cv, cv_test, name="tfidf_svc(groupCV)"))

    # ---- 5) 纯词法 + 反问规则：模板级 groupCV ----
    p_cvh, _, _, _ = run_group_cv(all_rows, label_fn=collapsed_label, post=apply_ask_rule)
    reports.append(evaluate_preds(p_cvh, cv_test, name="hybrid(groupCV)"))

    # ---- 6) 再加弃权：模板级 groupCV（两种弃权信号对照）----
    ab_details: dict[str, dict] = {}
    for mode in ("top1", "margin"):
        thetas: list[float] = []

        def post_factory(tr_fit, _mode=mode):
            th = calibrate_theta(tr_fit, abstain_label, drop_fn=is_no_tool_escalation,
                                 score_mode=_mode)
            thetas.append(th)
            return make_abstain_post(th)

        p_ab, ab_test, _, ab_scores = run_group_cv(
            all_rows, label_fn=abstain_label, post_factory=post_factory,
            drop_fn=is_no_tool_escalation, score_mode=mode)
        reports.append(evaluate_preds(p_ab, ab_test, name=f"abstain-{mode}(groupCV)"))
        ab_details[mode] = {
            "theta_per_fold": [round(t, 3) for t in thetas],
            "sweep": escalate_sweep(ab_test, ab_scores),
        }

    extras["abstain"] = {
        "dropped_from_training": sorted(NO_TOOL_ESCALATION_REASONS),
        "dropped_count": sum(1 for s in all_rows if is_no_tool_escalation(s)),
        "note": "理由细分不是本变体目标：学习到的 escalate 不带 reason，弃权一律标 ambiguous。",
        "per_mode": ab_details,
        "best_escalate_precision_by_mode": {
            m: max((r["escalate_precision"] for r in d["sweep"]), default=0.0)
            for m, d in ab_details.items()},
    }

    # ---- 7) 层级式：专门的拒识门 + 工具分类 + chat/escalate ----
    p_hi, hi_test, _, gate_pairs = run_hier_group_cv(all_rows)
    reports.append(evaluate_preds(p_hi, hi_test, name="hier(groupCV)"))

    g_tp = sum(1 for g, p in gate_pairs if g and p)
    g_fp = sum(1 for g, p in gate_pairs if not g and p)
    g_fn = sum(1 for g, p in gate_pairs if g and not p)
    g_prec = g_tp / (g_tp + g_fp) if g_tp + g_fp else 0.0
    g_rec = g_tp / (g_tp + g_fn) if g_tp + g_fn else 0.0
    extras["hier_gate"] = {
        "n": len(gate_pairs),
        "precision": round(g_prec, 4), "recall": round(g_rec, 4),
        "f1": round(2 * g_prec * g_rec / (g_prec + g_rec), 4) if g_prec + g_rec else 0.0,
        "tp": g_tp, "fp": g_fp, "fn": g_fn,
        "note": "门＝「平台有工具能接吗」的二分类；与弃权阈值是两种不同的拒识实现。",
    }

    extras["model"] = {
        "vectorizer": "TfidfVectorizer(char_wb, 1-4gram, sublinear)",
        "classifier": "LinearSVC(C=1.0)",
        "n_features": dim,
        "model_bytes": size,
        "train_seconds": round(train_s, 3),
        "infer_ms_per_sample": round(infer_ms, 3),
        "learned_classes_full": len(set(collapsed_label(s) for s in all_rows)),
        "learned_classes_abstain": len(set(abstain_label(s) for s in all_rows
                                           if not is_no_tool_escalation(s))),
    }
    extras["group_cv"] = {"n_folds": 5, "seed": 20260922, "folds": folds}

    table = H.format_report(reports)

    all_calls = [s for s in all_rows if H.kind_of(H.route_label(s)) == "call"]
    calls_unbound = sum(1 for s in all_calls
                        if (s.get("request") or {}).get("bound_dataset_id") is None)
    eval_n = sum(1 for s in all_rows if s.get("split") == "eval")

    notes = {
        "leakage_warning": (
            "样本级 holdout 与训练集共享 meta.template（仅槽位取值不同）⇒ 表层模型虚高。"
            "**泛化真值请看 *(groupCV) 三列**。"
        ),
        "ask_is_symbolic": (
            "`ask`（反问用户）不该由模型学：它是 required_params 的确定性函数。"
            f"证据：{len(all_calls)} 条 call 样本里有 {calls_unbound} 条 bound_dataset_id 也是 None"
            " ⇒「未绑定」不是判别特征。"
        ),
        "abstain_rationale": (
            "`ambiguous` / `out_of_scope` 按定义是「没有工具匹配得上」⇒ 用**弃权**处理，"
            "而不是当分类标签学；θ 逐折在内层 group CV 的 OOF 分数上取 binary F1 最优点。"
        ),
        "escalate_is_bottleneck": (
            "扩数据后工具选择已到 80%，剩下瓶颈是语义闸门；误判方向集中为 escalate→call。"
        ),
        "eval_split_note": f"eval split 仅 {eval_n} 条且类别极不均衡 ⇒ 不作主指标（报告里已省略）。",
        "label_space": "route 串由 eval_harness.route_label 派生，工具名零硬编码 ⇒ 不可能输出不存在的工具。",
        "hard_negatives_note": "dataset/hard_negatives.jsonl 是工具混淆对诊断，未参与训练/评估。",
    }

    (OUT_DIR / "baseline_report.json").write_text(json.dumps(
        {"reports": reports, "extras": extras, "notes": notes,
         "table_markdown": table, "generated_at_epoch": int(time.time())},
        ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# 词法基线 Router 报告", "",
          "> 生成脚本 `backend/scripts/router/baseline_lexical.py`；口径 `eval_harness.py`。", "",
          table, "",
          "## 模型规模 / 速度", "",
          "| 项 | 值 |", "| --- | --- |"]
    for k, v in extras["model"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## 弃权阈值扫描（操作点）", "",
           f"训练集中被移出学习的「无工具可匹配」升级样本："
           f"{extras['abstain']['dropped_count']} 条 "
           f"({', '.join(extras['abstain']['dropped_from_training'])})"]
    for _m, _d in extras["abstain"]["per_mode"].items():
        md += ["", f"### 弃权信号 = {_m}", "",
               f"逐折校准出的 θ：{_d['theta_per_fold']}", "",
               "| θ 分位 | θ | escalate 精确率 | escalate 召回率 | escalate F1 | 弃权率 |",
               "| --- | --- | --- | --- | --- | --- |"]
        for r in _d["sweep"]:
            md.append(f"| {r['theta_quantile']}% | {r['theta']} | {r['escalate_precision']:.1%} | "
                      f"{r['escalate_recall']:.1%} | {r['escalate_f1']:.1%} | {r['abstain_rate']:.1%} |")

    _g = extras["hier_gate"]
    md += ["", "## 拒识门（hier 第一段，专门二分类）", "",
           "| 指标 | 值 |", "| --- | --- |",
           f"| 样本数 | {_g['n']} |",
           f"| 精确率 | {_g['precision']:.1%} |",
           f"| 召回率 | {_g['recall']:.1%} |",
           f"| F1 | {_g['f1']:.1%} |",
           f"| tp / fp / fn | {_g['tp']} / {_g['fp']} / {_g['fn']} |",
           "", _g["note"]]

    md += ["", "## 分组 5 折明细", "",
           "| fold | train | test | test 组数 |", "| --- | --- | --- | --- |"]
    for f in extras["group_cv"]["folds"]:
        md.append(f"| {f['fold']} | {f['train']} | {f['test']} | {f['test_groups']} |")
    md += ["", "## 按类目明细（groupCV 三列）", ""]
    for r in reports:
        if "groupCV" not in r["name"]:
            continue
        md += [f"### {r['name']}", "", "| category | n | route_acc |", "| --- | --- | --- |"]
        for c, d in r.get("by_category", {}).items():
            md.append(f"| {c} | {d['n']} | {d['route_acc']:.1%} |")
        conf = r.get("top_kind_confusions") or []
        if conf:
            md += ["", "主要类型混淆（金标→预测）：" +
                   "；".join(f"{a}→{b}×{c}" for a, b, c in conf[:6])]
        by_reason = (r.get("escalation") or {}).get("by_reason_recall") or {}
        if by_reason:
            md += ["", "升级识别率（按原因，kind 级）：" +
                   "；".join(f"{k} {v['recall']:.0%}(n={v['n']})" for k, v in by_reason.items())]
        md.append("")
    md += ["## 注意事项", ""]
    for k, v in notes.items():
        md.append(f"- **{k}**：{v}")
    (OUT_DIR / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print(table)
    print()
    print(f"模型 {size} bytes / 特征 {dim} 维 / 训练 {train_s:.3f}s / 推理 {infer_ms:.3f} ms·条⁻¹")
    for _m, _d in extras["abstain"]["per_mode"].items():
        print(f"  弃权信号 {_m:6s} θ（逐折）: {_d['theta_per_fold']}  "
              f"最高 escalate 精确率: {max((r['escalate_precision'] for r in _d['sweep']), default=0):.1%}")
    _g = extras["hier_gate"]
    print(f"  拒识门（二分类）: 精确率 {_g['precision']:.1%} / 召回率 {_g['recall']:.1%} / F1 {_g['f1']:.1%}")
    print("已写出 dataset/baseline_report.json + baseline_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
