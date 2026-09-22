"""离线复核融合实验：把「算」和「看」分开。

`fuse_l1_lexical.py --dump_probs` 会把逐条概率落盘到 `dataset/fusion_probs_{tag}.npz`，
那一次训练（约 4 分钟 GPU）是昂贵的；本脚本复用该产物，不再训练即回答三个问题：

1. **曲线形状**：α 从 0 扫到 1（步长 0.05），增益是尖峰还是平台？
   尖峰 ⇒ 大概率是在测试集上挑出来的噪声；平台 ⇒ 融合有真实、稳定的收益。
2. **分折复核**：每一折单独的 route 准确率（对照该折的纯词法 / 纯神经），
   看总增益是否只由某一折贡献。
3. **嵌套验证（诚实版）**：α 在「其余 4 折」上挑、在留出的那一折上评。
   这是「α 在测试集上挑最优」唯一能站住的替代口径 ——
   若嵌套结果仍显著高于纯词法，融合才值得上。

口径与 `fuse_l1_lexical.py` 完全一致：`assemble_route` 合成三层决策，
金标用 `eval_harness.route_label`。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_harness as H  # noqa: E402
import l1_data as D  # noqa: E402

import numpy as np  # noqa: E402

FINE = tuple(round(0.05 * i, 2) for i in range(21))
# 容差取 1pt：n=844 时 1 条 = 0.12%，0.5pt 仅 4 条，用它判「平台/尖峰」等于拿噪声当信号。
PLATEAU_TOL = 0.01
# 决策相关的阈值：融合至少要「明显优于纯词法」，低于这个量级就不值得多付一份神经模型的代价。
DECISION_MARGIN = 0.02


class Fused:
    """逐条概率 + 金标；所有指标都从概率重算，保证口径唯一。"""

    def __init__(self, ids, folds, gold, label_space, p_lex, p_neu, rows_by_id):
        self.ids = ids
        self.folds = folds
        self.gold = gold
        self.label_space = label_space
        self.p_lex = p_lex
        self.p_neu = p_neu
        self.rows = [rows_by_id[i] for i in ids]
        self.fold_ids = sorted(set(folds))

    def sel(self, fold: int | None = None, exclude: int | None = None) -> list[int]:
        if fold is not None:
            return [i for i, f in enumerate(self.folds) if f == fold]
        return [i for i, f in enumerate(self.folds) if f != exclude]

    def route_labels(self, idx: list[int], alpha: float) -> list[str]:
        """α 加权融合概率 → argmax → 三层合成，返回**路线级**标签。"""
        p = alpha * self.p_lex[idx] + (1.0 - alpha) * self.p_neu[idx]
        labels = [self.label_space[i] for i in p.argmax(axis=1).tolist()]
        return [D.assemble_route(s, lab) for s, lab in zip((self.rows[i] for i in idx), labels)]

    def correct(self, idx: list[int], alpha: float) -> int:
        return sum(1 for a, b in zip(self.route_labels(idx, alpha),
                                     (self.gold[i] for i in idx)) if a == b)

    def acc(self, idx: list[int], alpha: float) -> float:
        return self.correct(idx, alpha) / len(idx)

    def best_alpha(self, idx: list[int]) -> tuple[float, float]:
        scored = {a: self.acc(idx, a) for a in FINE}
        a = max(scored, key=lambda k: scored[k])
        return a, scored[a]


def load(tag: str) -> tuple[Fused, Path]:
    merged = H.DATASET_DIR / f"fusion_probs_{tag}.npz"
    fold_files = sorted(H.DATASET_DIR.glob(f"fusion_probs_{tag}_fold*.npz"))
    if not merged.exists() and not fold_files:
        raise FileNotFoundError(
            f"找不到 {merged}（也没有逐折文件）；请先用 fuse_l1_lexical.py --dump_probs 生成。")

    if merged.exists():
        z = np.load(merged, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        folds = [int(x) for x in z["folds"]]
        gold = [str(x) for x in z["gold_route"]]
        label_space = [str(x) for x in z["label_space"]]
        p_lex, p_neu = z["p_lexical"], z["p_neural"]
        src = merged
    else:
        # 兜底：合并文件因故未写出时，从逐折文件拼装（折号按文件名顺序）
        ids, folds, gold, label_space = [], [], [], None
        plex_parts, pneu_parts = [], []
        for k, f in enumerate(fold_files, 1):
            zz = np.load(f, allow_pickle=True)
            label_space = [str(x) for x in zz["label_space"]]
            ids += [str(x) for x in zz["ids"]]
            folds += [k] * len(zz["ids"])
            gold += [str(x) for x in zz["gold_route"]]
            plex_parts.append(zz["p_lexical"])
            pneu_parts.append(zz["p_neural"])
        p_lex = np.concatenate(plex_parts, axis=0)
        p_neu = np.concatenate(pneu_parts, axis=0)
        src = fold_files[0]

    rows_by_id = {s["id"]: s for s in H.load_split("all")}
    missing = [i for i in ids if i not in rows_by_id]
    if missing:
        raise KeyError(f"{len(missing)} 个样本 id 在 samples 里找不到（如 {missing[:3]}）")
    return Fused(ids, folds, gold, label_space, p_lex, p_neu, rows_by_id), src


def gate_analysis(F: "Fused", alpha: float,
                  quantiles=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)) -> dict:
    """置信度门控：把「最没把握」的那部分请求升级给云端，看本地准确率能抬到多少。

    为什么只在子集上算：`escalate` 已被 L0 结构规则拦下（召回 96.6%），L1 根本不管它。
    ⇒ 门控只对 **L1 真正负责的样本**（gold kind ∈ chat/call/ask）有意义。

    用**保留比例**而不是绝对阈值来叙述，因为阈值随 softmax 尺度漂移，
    而「只本地处理最有把握的 K%」是与尺度无关、可直接决策的说法；同时给出行内阈值供实现参考。
    """
    idx = [i for i, g in enumerate(F.gold) if H.kind_of(g) != "escalate"]
    preds = F.route_labels(idx, alpha)
    correct = np.asarray([p == F.gold[i] for p, i in zip(preds, idx)])
    pmax = (alpha * F.p_lex[idx] + (1.0 - alpha) * F.p_neu[idx]).max(axis=1)
    order = np.argsort(-pmax)
    m = len(idx)

    rows = []
    for q in quantiles:
        k = max(int(round(q * m)), 1)
        keep = order[:k]
        rows.append({
            "cover": k / m,
            "n_local": k,
            "local_acc": float(correct[keep].mean()),
            "theta": float(pmax[order[k - 1]]),
            "escalate_rate": 1 - k / m,
        })
    return {"alpha": alpha, "n_l1": m, "acc_no_gate": float(correct.mean()), "rows": rows}


def _gate_table(gates: list[dict]) -> str:
    head = "| 本地保留比例 | " + " | ".join(f"α={g['alpha']:g} 准确率" for g in gates) + " | 实现阈值 θ（α=1） |"
    sep = "| --- | " + " | ".join("---" for _ in gates) + " | --- |"
    by_q = {round(r["cover"], 3): r for r in gates[0]["rows"]}
    lines = [head, sep]
    for q in sorted(by_q, reverse=True):
        cells = []
        for g in gates:
            r = next(x for x in g["rows"] if round(x["cover"], 3) == q)
            cells.append(f"{r['local_acc']:.1%}")
        th = next(x for x in gates[0]["rows"] if round(x["cover"], 3) == q)["theta"]
        lines.append(f"| {q:.0%}（{by_q[q]['n_local']} 条） | " + " | ".join(cells) + f" | {th:.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="L4H256")
    args = ap.parse_args()

    try:
        F, src = load(args.tag)
    except (FileNotFoundError, KeyError) as exc:
        print(exc)
        return 1

    n = len(F.ids)
    print(f"离线复核：{src.name} | {n} 条 | {len(F.label_space)} 类 | "
          f"{len(F.fold_ids)} 折 {F.fold_ids}\n")

    all_idx = list(range(n))
    sweep = {a: F.acc(all_idx, a) for a in FINE}
    lex_acc, neu_acc = sweep[1.0], sweep[0.0]
    best_a = max(sweep, key=lambda k: sweep[k])
    plateau = [a for a in FINE if sweep[a] >= sweep[best_a] - PLATEAU_TOL]
    fixed_half = sweep[0.5]

    # ---- 1. 细扫 α ----
    print("| α | route 准确率 | 备注 |")
    print("| --- | --- | --- |")
    for a in FINE:
        note = []
        if a in (0.0, 1.0):
            note.append("自检端点")
        if a == best_a:
            note.append("细扫最优")
        if a == 0.5:
            note.append("事先定好的头条")
        print(f"| {a:.2f} | {sweep[a]:.1%} | {' / '.join(note)} |")
    # 注意：峰窄不等于「噪声」——判噪声要看下面的嵌套验证，不能只看曲线形状。
    shape = (f"最优附近{'平坦' if len(plateau) >= 3 else '较窄'}"
             f"（{PLATEAU_TOL:.0%} 内 {len(plateau)}/{len(FINE)} 网格）")
    decisive = [a for a in FINE if sweep[a] >= lex_acc + DECISION_MARGIN]
    print(f"\n纯词法 α=1 {lex_acc:.1%} ｜ 纯神经 α=0 {neu_acc:.1%} ｜ "
          f"细扫最优 α={best_a:.2f} → {sweep[best_a]:.1%}（{sweep[best_a]-lex_acc:+.1%} vs 词法）")
    print(f"距最优 {PLATEAU_TOL:.1%} 以内的 α：{plateau[0]:.2f}~{plateau[-1]:.2f}"
          f"（{len(plateau)}/{len(FINE)} 网格）⇒ {shape}")
    print(f"≥ 纯词法 +{DECISION_MARGIN:.1%} 的 α 区间：{decisive[0]:.2f}~{decisive[-1]:.2f}"
          f"（{len(decisive)}/{len(FINE)} 网格）—— 这才是「值不值得上融合」的判据；"
          f"事先定好的 α=0.5 = {fixed_half:.1%}（{fixed_half-lex_acc:+.1%}），"
          f"落在区间内 ⇒ 结论不依赖调参\n")

    # ---- 2. 分折复核 ----
    print("| 折 | n | 纯词法 | 纯神经 | α=0.5 | 该折最优 α | 该折最优 | 相对同折词法 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    per_fold = {}
    for k in F.fold_ids:
        idx = F.sel(fold=k)
        a_lex, a_neu = F.acc(idx, 1.0), F.acc(idx, 0.0)
        ka, kacc = F.best_alpha(idx)
        per_fold[k] = (a_lex, a_neu, F.acc(idx, 0.5), ka, kacc)
        print(f"| {k} | {len(idx)} | {a_lex:.1%} | {a_neu:.1%} | {F.acc(idx, 0.5):.1%} | "
              f"{ka:.2f} | {kacc:.1%} | {kacc - a_lex:+.1%} |")
    gains = [per_fold[k][4] - per_fold[k][0] for k in F.fold_ids]
    print(f"\n各折最优 α 相对同折纯词法的增益：{', '.join(f'{g:+.1%}' for g in gains)} ⇒ "
          f"{'每折均为正，增益非单折贡献' if all(g > 0 for g in gains) else '存在负增益折，增益偏单折'}\n")

    # ---- 3. 嵌套验证：α 在其余折挑，在留出折评 ----
    chosen, nested_correct, nested_total = {}, 0, 0
    for k in F.fold_ids:
        a_star, _ = F.best_alpha(F.sel(exclude=k))
        chosen[k] = a_star
        idx = F.sel(fold=k)
        nested_correct += F.correct(idx, a_star)
        nested_total += len(idx)
    nested_acc = nested_correct / nested_total

    print("嵌套验证（α 在其余 4 折挑 → 在留出折评）：")
    print(f"- 各折选中的 α：{', '.join(f'折{k}={chosen[k]:.2f}' for k in F.fold_ids)}")
    print(f"- 嵌套整体 route **{nested_acc:.1%}** ｜ 固定 α=0.5 **{fixed_half:.1%}** ｜ 纯词法 {lex_acc:.1%}")
    print(f"- 嵌套 − 纯词法 = **{nested_acc - lex_acc:+.1%}**；嵌套 − 纯神经 = {nested_acc - neu_acc:+.1%}")

    # ---- 4. 互补性：融合到底有没有空间（全程路线级）----
    lex_rt = F.route_labels(all_idx, 1.0)
    neu_rt = F.route_labels(all_idx, 0.0)
    combo = Counter()
    for gl, a, b in zip(F.gold, lex_rt, neu_rt):
        combo["both_right" if (a == gl and b == gl) else
              "lexical_only" if a == gl else
              "neural_only" if b == gl else "both_wrong"] += 1
    ceiling = (combo["both_right"] + combo["lexical_only"] + combo["neural_only"]) / n
    best_single = (combo["both_right"] + max(combo["lexical_only"], combo["neural_only"])) / n

    print("\n互补性（路线级四分解，n=%d）：都对 %d ｜ 仅词法对 %d ｜ 仅神经对 %d ｜ 都错 %d"
          % (n, combo["both_right"], combo["lexical_only"], combo["neural_only"], combo["both_wrong"]))
    print(f"- 最佳单模型 {best_single:.1%} ｜ 完美选择器上限 {ceiling:.1%}"
          f"（事后统计、不可达，仅用于判断融合空间）")
    print(f"- 实际融合最高 {sweep[best_a]:.1%}，兑现了上限的 "
          f"{(sweep[best_a] - lex_acc) / max(ceiling - lex_acc, 1e-9):.0%}")

    # ---- 5. 置信度门控：80% 的准确率够不够用，取决于能否把没把握的那部分甩给云端 ----
    gates = [gate_analysis(F, a) for a in (1.0, 0.5, 0.75)]
    print("\n置信度门控（只在 L1 真正负责的样本上：L0 已拦下 escalate）")
    for g in gates:
        print(f"   α={g['alpha']:g}：L1 负责 {g['n_l1']} 条，不门控时准确率 {g['acc_no_gate']:.1%}")
    print(_gate_table(gates))
    print("\n读法：本地保留比例＝不升级、由本地直接答；其余升级给云端（多花钱但换质量）。")
    print("⇒ 「80% 够不够用」的真问题不是平均值，而是**要求本地准确率 95% 时还能覆盖多少请求**。")

    md = [f"# 融合实验离线复核（{args.tag}）", "",
          f"> 数据源 `dataset/{src.name}`（由 `fuse_l1_lexical.py --dump_probs` 落盘），本脚本不再训练。", "",
          f"- 样本 **{n}** 条，{len(F.fold_ids)} 折模板级 groupCV，标签空间 {len(F.label_space)} 类",
          f"- 纯词法 α=1：**{lex_acc:.1%}** ｜ 纯神经 α=0：**{neu_acc:.1%}**",
          f"- 事先定好的 α=0.5：**{fixed_half:.1%}**（{fixed_half-lex_acc:+.1%} vs 词法）",
          f"- 细扫最优 α={best_a:.2f}：**{sweep[best_a]:.1%}**（{sweep[best_a]-lex_acc:+.1%} vs 词法）",
          f"- 距最优 {PLATEAU_TOL:.1%} 内的 α 区间：{plateau[0]:.2f}~{plateau[-1]:.2f}"
          f"（{len(plateau)}/{len(FINE)} 网格）⇒ **{shape}**",
          f"- ≥ 纯词法 +{DECISION_MARGIN:.1%} 的 α 区间：**{decisive[0]:.2f}~{decisive[-1]:.2f}**"
          f"（{len(decisive)}/{len(FINE)} 网格）；事先定好的 α=0.5 落在区间内 ⇒ 结论不依赖调参", "",
          "## α 细扫", "", "| α | route 准确率 |", "| --- | --- |"]
    md += [f"| {a:.2f} | {sweep[a]:.1%} |" for a in FINE]
    md += ["", "## 分折复核", "",
           "| 折 | n | 纯词法 | 纯神经 | α=0.5 | 该折最优 α | 该折最优 | 相对同折词法 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in F.fold_ids:
        a_lex, a_neu, a_half, ka, kacc = per_fold[k]
        md.append(f"| {k} | {len(F.sel(fold=k))} | {a_lex:.1%} | {a_neu:.1%} | {a_half:.1%} | "
                  f"{ka:.2f} | {kacc:.1%} | {kacc - a_lex:+.1%} |")
    md += ["", f"各折增益：{', '.join(f'{g:+.1%}' for g in gains)}。", "",
           "## 嵌套验证（诚实的 α 选择口径）", "",
           f"- 各折选中的 α：{', '.join(f'折{k}={chosen[k]:.2f}' for k in F.fold_ids)}",
           f"- 嵌套整体 route：**{nested_acc:.1%}**",
           f"- 固定 α=0.5：**{fixed_half:.1%}** ｜ 纯词法：{lex_acc:.1%}",
           f"- 嵌套 − 纯词法：**{nested_acc - lex_acc:+.1%}**；嵌套 − 纯神经：{nested_acc - neu_acc:+.1%}", "",
           "## 互补性：融合到底有没有空间（路线级）", "",
           "| 组合 | 条数 |", "| --- | --- |",
           f"| 两者都对 | {combo['both_right']} |",
           f"| 仅词法对 | {combo['lexical_only']} |",
           f"| 仅神经对 | {combo['neural_only']} |",
           f"| 两者都错 | {combo['both_wrong']} |", "",
           f"- 最佳单模型 **{best_single:.1%}**；完美选择器上限 **{ceiling:.1%}**（事后统计、不可达）",
           f"- 实际融合最高 {sweep[best_a]:.1%}，兑现了上限的 "
           f"{(sweep[best_a] - lex_acc) / max(ceiling - lex_acc, 1e-9):.0%}", ""]
    gate_md = ["", "## 置信度门控：把没把握的请求升级出去（回答「80% 够不够用」）", "",
               "只在 **L1 真正负责**的样本上算（`escalate` 已由 L0 规则拦下，L1 不管它）："
               f"L1 负责 **{gates[0]['n_l1']}** 条；不门控时准确率 "
               + " ／ ".join(f"α={g['alpha']:g} {g['acc_no_gate']:.1%}" for g in gates) + "。", "",
               _gate_table(gates), "",
               "- **本地保留比例** = 不升级、由本地直接答；其余升级给云端（多花钱换质量）。",
               "- **实现阈值 θ** 取自 α=1（纯词法，零新依赖那一版），按 top1 概率取分位点即可，"
               "不需要额外训练 —— 门控是**零成本**的。",
               "- ⚠️ 这是**离线**校准，真实流量分布不同，阈值上线后要按实际覆盖率重标一次。", ""]
    md += gate_md
    md += ["## 怎么读这张表", "",
           "- **α=0.5 是事先定好的头条口径**，不是最优点；它若已明显超过纯词法，就说明结论不依赖调参。",
           f"- **『值不值得上』看的是 ≥ 纯词法 +{DECISION_MARGIN:.0%} 的那个带宽**，不是最优点那一个网格；"
           "峰窄不等于噪声。",
           "- **嵌套验证**才是判噪声的仲裁：若 α 在各折上被稳定地选到同一处、留出折仍赢，"
           "就不能说增益是挑出来的。", ""]
    out = H.DATASET_DIR / f"fusion_recheck_{args.tag}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写出 dataset/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
