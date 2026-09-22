"""融合实验：词法基线 + 神经 L1 —— 互补性上限 89%，实际能拿多少？

为什么值得做
------------
单模型对比（groupCV，844 条，同 seed / 同 5 折）：

| 方案 | 参数 | 单条延迟 | route | 工具 |
| --- | --- | --- | --- | --- |
| 词法基线（TF-IDF char_wb + LinearSVC） | — / 3.5MB | 0.008 ms | 80.7% | 80.1% |
| L1 字符 Transformer 从零训 339k | 1.4MB | 1.45 ms | 73.1% | 72.6% |
| L1 预训练中文 RoBERTa L2H128 | 3.19M / 12.8MB | 3.61 ms | 75.9% | 75.3% |
| L1 预训练中文 RoBERTa L4H256 | 8.77M / 35.1MB | 4.19 ms | 79.5% | 79.0% |

神经 L1 随规模单调追平但**始终未超过**一个 3.5MB 的 CPU 词法模型。然而互补性诊断（本脚本末段重算）显示：
两者的错误**高度不重叠**、完美选择器上限明显高于任一单模型 ⇒ 融合可能有真实空间。

本脚本回答：**把两者的概率平均起来，实际能拿多少？** 并给出代价（延迟/体积）。

融合方式（刻意保持朴素、不调参到过拟合）
----------------------------------------
1. 两个模型的原始分数**尺度完全不同**（LinearSVC 决策函数 ±1 量级，神经 logits ±10 量级）。
   直接平均等于让「更尖锐」的那个说了算 ⇒ 先**逐行 z-score 标准化**再 softmax，把两者尺度拉平。
2. 权重 α ∈ {0, .25, .5, .75, 1}：`P = α·P_词法 + (1-α)·P_神经`。
   **α=1 必须复现词法基线、α=0 必须复现神经 L1** —— 这是本脚本的内部一致性自检。
3. 权重扫描**只是操作曲线**，不是「选最优」的结果；报头条时用**事先定好的 α=0.5**。

⚠️ 标签空间对齐是这里最容易错的地方：`clf.classes_` 只含训练折里出现过的类，
必须按完整 31 类标签空间对齐（缺失列填 -1e9），否则列错位而指标看起来还挺正常。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_lexical as B  # noqa: E402
import eval_harness as H  # noqa: E402
import l1_data as D  # noqa: E402
import train_l1 as TL  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HEADLINE_ALPHA = 0.5
NEG = -1e9


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def rownorm(z: np.ndarray) -> np.ndarray:
    """逐行 z-score：把两个模型的分数尺度拉到可比，否则平均由更尖锐的一方主导。"""
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    return (z - mu) / np.where(sd < 1e-9, 1.0, sd)


def align(classes: list[str], matrix: np.ndarray, label_space: list[str]) -> np.ndarray:
    """把 `clf.classes_` 的列对齐到完整标签空间；缺席的类填 NEG（softmax 后≈0）。"""
    out = np.full((matrix.shape[0], len(label_space)), NEG, dtype=float)
    index = {lab: i for i, lab in enumerate(label_space)}
    for j, name in enumerate(classes):
        if name in index:
            out[:, index[name]] = matrix[:, j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in TL.DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--tag", type=str, default="fuse")
    ap.add_argument("--dump_probs", action="store_true",
                    help="把逐条概率落盘（npz），供离线扫 α / 分折复核，避免重复训练")
    args = ap.parse_args()

    kind = "pretrained" if args.pretrained else "scratch"
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    rows = H.load_split("all")
    label_space = D.l1_label_space()
    print(f"词法 + 神经融合实验 | 神经后端 {kind}"
          + (f"（{args.pretrained.split('/')[-1]}）" if kind == "pretrained" else ""))
    print(f"设备 {device} | 标签空间 {len(label_space)} 类 | 权重 α ∈ {ALPHAS}"
          f"（头条用 α={HEADLINE_ALPHA}）")
    print(f"契约来源 {D.contract_source()}\n")

    routes = {a: {} for a in ALPHAS}      # α -> {id: route}
    agree_ids, disagree_ids = [], []
    combo: Counter = Counter()            # both_right / baseline_only / l1_only / both_wrong
    test_rows: list[dict] = []
    dump_ids: list[int] = []
    dump_folds: list[int] = []
    dump_gold: list[str] = []
    dump_plexi: list[np.ndarray] = []
    dump_pneu: list[np.ndarray] = []
    t_start = time.perf_counter()

    for i, (tr, te) in enumerate(H.group_folds(rows, n_folds=args.folds, seed=args.seed), 1):
        # ① 词法
        vec, clf = B.lexical_model(tr)
        classes, mat = B.lexical_score_matrix(vec, clf, te)
        p_lex = softmax(rownorm(align(classes, mat, label_space)))

        # ② 神经
        out = TL.train_fold(tr, args, device, label_space, kind)
        with torch.no_grad():
            logits = out["backend"].logits(te).float().cpu().numpy()
        p_neu = softmax(rownorm(logits))

        # ⚠️ 分歧分析必须全程用**路线级**标签：原先拿原始 argmax 标签算分歧、
        # 却拿路线标签算一致率，两个尺度混用得出的分解（都对/仅词法/仅神经/都错）不成立。
        lex_arg = np.asarray(label_space)[p_lex.argmax(1)]
        neu_arg = np.asarray(label_space)[p_neu.argmax(1)]
        fold_agree = 0
        for s, la, na in zip(te, lex_arg.tolist(), neu_arg.tolist()):
            g = H.route_label(s)
            lr = D.assemble_route(s, la)
            nr = D.assemble_route(s, na)
            if lr == nr:
                agree_ids.append(s["id"])
                fold_agree += 1
            else:
                disagree_ids.append(s["id"])
            combo["both_right" if (lr == g and nr == g) else
                  "baseline_only" if lr == g else
                  "l1_only" if nr == g else "both_wrong"] += 1
            lex_only_win = combo["baseline_only"]
            neu_only_win = combo["l1_only"]

        dump_ids.extend(s["id"] for s in te)
        dump_folds.extend([i] * len(te))
        dump_gold.extend(H.route_label(s) for s in te)
        dump_plexi.append(p_lex)
        dump_pneu.append(p_neu)
        if args.dump_probs:
            # 逐折增量落盘：样本 id 是 `chat-xxxx` 这类字符串，必须 dtype=object；
            # 且训练要 3~4 分钟，万一后面出错也不该丢掉已算好的折。
            np.savez_compressed(
                H.DATASET_DIR / f"fusion_probs_{args.tag}_fold{i}.npz",
                ids=np.asarray([s["id"] for s in te], dtype=object),
                label_space=np.asarray(label_space, dtype=object),
                gold_route=np.asarray([H.route_label(s) for s in te], dtype=object),
                p_lexical=p_lex, p_neural=p_neu)

        for a in ALPHAS:
            p = a * p_lex + (1.0 - a) * p_neu
            labels = np.asarray(label_space)[p.argmax(1)]
            for s, lab in zip(te, labels.tolist()):
                routes[a][s["id"]] = D.assemble_route(s, lab)
        test_rows.extend(te)
        print(f"  fold {i}: test={len(te):3d} 神经训练 {out['train_s']:.1f}s "
              f"两者路线一致 {fold_agree}/{len(te)}")

    # ---- 评估 ----
    reports = []
    for a in ALPHAS:
        name = f"α={a:g} " + ("(纯词法)" if a == 1 else "(纯神经)" if a == 0
                              else "(融合)" if a == HEADLINE_ALPHA else "")
        reports.append(H.evaluate(lambda s, a=a: routes[a][s["id"]], test_rows, name=name.strip()))

    print()
    print(H.format_report(reports))

    # ---- 一致性自检 ----
    gold = {s["id"]: H.route_label(s) for s in test_rows}
    lex_acc = sum(1 for i in gold if routes[1.0][i] == gold[i]) / len(gold)
    neu_acc = sum(1 for i in gold if routes[0.0][i] == gold[i]) / len(gold)
    fused_acc = sum(1 for i in gold if routes[HEADLINE_ALPHA][i] == gold[i]) / len(gold)
    print(f"\n一致性自检：α=1 复现词法 {lex_acc:.1%}；α=0 复现神经 {neu_acc:.1%}；"
          f"α={HEADLINE_ALPHA:g} 融合 {fused_acc:.1%}")

    # ---- 分歧分析（全程路线级）：两者不一致时谁对；一致时两者必然同对/同错 ----
    n_agree, n_dis = len(agree_ids), len(disagree_ids)
    agree_acc = combo["both_right"] / max(n_agree, 1)
    print(f"\n两模型一致 {n_agree}/{len(gold)} = {n_agree/len(gold):.1%}"
          f"（其中路线正确 {agree_acc:.1%}）")
    print(f"两模型分歧 {n_dis} 条：词法对 {combo['baseline_only']} ｜ "
          f"神经对 {combo['l1_only']} ｜ 都错 {n_dis - combo['baseline_only'] - combo['l1_only']}")
    ceiling = (combo["both_right"] + combo["baseline_only"] + combo["l1_only"]) / len(gold)
    best_single = (combo["both_right"] + max(combo["baseline_only"], combo["l1_only"])) / len(gold)
    print(f"路线级四分解：都对 {combo['both_right']} ｜ 仅词法对 {combo['baseline_only']} ｜ "
          f"仅神经对 {combo['l1_only']} ｜ 都错 {combo['both_wrong']}"
          f"（合计 {sum(combo.values())}/{len(gold)}）")
    print(f"⇒ 最佳单模型 {best_single:.1%} ｜ 完美选择器上限 {ceiling:.1%}"
          f"（上限是事后统计，不可达；只用来判断融合有没有空间）")

    extras = {
        "backend_kind": kind,
        "config": {k: getattr(args, k) for k in TL.DEFAULTS},
        "alphas": list(ALPHAS),
        "headline_alpha": HEADLINE_ALPHA,
        "route_acc_by_alpha": {f"{a:g}": round(sum(1 for i in gold if routes[a][i] == gold[i])
                                              / len(gold), 4) for a in ALPHAS},
        "agreement": {
            "level": "route",
            "n_agree": n_agree, "n_disagree": n_dis,
            "agree_route_acc": round(agree_acc, 4),
            "lex_wins_on_disagreement": combo["baseline_only"],
            "neural_wins_on_disagreement": combo["l1_only"],
            "both_wrong_on_disagreement": n_dis - combo["baseline_only"] - combo["l1_only"],
            "both_right": combo["both_right"],
            "both_wrong_total": combo["both_wrong"],
            "best_single_model": round(best_single, 4),
            "perfect_selector_ceiling": round(ceiling, 4),
        },
        "selfcheck": {"alpha1_matches_lexical": round(lex_acc, 4),
                      "alpha0_matches_neural": round(neu_acc, 4)},
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    notes = {
        "scale_handling": "两个模型分数尺度不同 ⇒ 先逐行 z-score 再 softmax，否则平均由更尖锐的一方主导。",
        "alpha_sweep": "α 扫描是**操作曲线**，不是「选最优」；头条数字用事先定好的 α=0.5。",
        "selfcheck": "α=1/α=0 必须分别复现词法/神经单模型的结果 —— 不自检就没有理由相信融合数字。",
        "alignment": "clf.classes_ 只含训练折出现过的类，必须按完整标签空间对齐（缺失填 -1e9）。",
        "disagreement_level": "一致/分歧与四分解**全程按路线级标签**（assemble_route 之后）统计；"
                              "若一致率用路线标签、分歧用原始 argmax 标签，两个尺度混用会得出不成立的分解。",
        "deploy_cost": "融合要同时跑两个模型：词法 0.008ms + 神经数 ms，且神经权重体积照付。",
    }

    out_dir = H.DATASET_DIR
    if args.dump_probs:
        np.savez_compressed(
            out_dir / f"fusion_probs_{args.tag}.npz",
            ids=np.asarray(dump_ids, dtype=object),
            folds=np.asarray(dump_folds, dtype=np.int64),
            gold_route=np.asarray(dump_gold, dtype=object),
            label_space=np.asarray(label_space, dtype=object),
            p_lexical=np.concatenate(dump_plexi, axis=0),
            p_neural=np.concatenate(dump_pneu, axis=0),
        )
        print(f"已写出 dataset/fusion_probs_{args.tag}.npz（逐条概率，供离线扫 α / 分折复核）")
    (out_dir / f"fusion_report_{args.tag}.json").write_text(json.dumps(
        {"reports": reports, "extras": extras, "notes": notes,
         "table_markdown": H.format_report(reports),
         "generated_at_epoch": int(time.time())}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    md = [f"# 词法 + 神经 L1 融合实验（{args.tag}）", "",
          "> 脚本 `scripts/router/fuse_l1_lexical.py`；口径 `eval_harness.py`；"
          "三层合成 `l1_data.assemble_route`。", "",
          f"- 神经后端：{kind}"
          + (f"（`{args.pretrained}`）" if kind == "pretrained" else "（字符级从零训练）"),
          f"- 融合权重 α ∈ {list(ALPHAS)}，头条用 **α={HEADLINE_ALPHA}**", "",
          "## 指标对比（模板级 groupCV，844 条）", "", H.format_report(reports), "",
          "## 一致性自检与分歧分析", "",
          "| 项 | 值 |", "| --- | --- |",
          f"| α=1 应复现词法 | {lex_acc:.1%} |",
          f"| α=0 应复现神经 | {neu_acc:.1%} |",
          f"| 融合（α={HEADLINE_ALPHA}） | {fused_acc:.1%} |",
          f"| 两模型路线一致 | {n_agree}/{len(gold)}（{n_agree/len(gold):.1%}），其中正确 {agree_acc:.1%} |",
          f"| 路线级分歧 | {n_dis} |",
          f"| └ 分歧时词法对 | {combo['baseline_only']} |",
          f"| └ 分歧时神经对 | {combo['l1_only']} |",
          f"| └ 分歧时都错 | {n_dis - combo['baseline_only'] - combo['l1_only']} |",
          f"| 路线级四分解 | 都对 {combo['both_right']} ｜ 仅词法对 {combo['baseline_only']} ｜ "
          f"仅神经对 {combo['l1_only']} ｜ 都错 {combo['both_wrong']} |",
          f"| 最佳单模型 | {best_single:.1%} |",
          f"| 完美选择器上限（事后，不可达） | {ceiling:.1%} |", "",
          "## 口径说明", ""] + [f"- **{k}**：{v}" for k, v in notes.items()]
    (out_dir / f"fusion_report_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n已写出 dataset/fusion_report_{args.tag}.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
