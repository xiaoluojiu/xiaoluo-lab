"""L1 基线对照实验：神经 L1 到底能不能打过词法基线？

要回答的唯一问题
----------------
词法基线（`baseline_lexical.py`：TF-IDF char_wb 1-4gram + LinearSVC，约 3.5MB、0.008 ms·条⁻¹）
在模板级 groupCV 上拿到 工具 80.1% / 整条 80.7%。**L1 神经模型必须超过它才值得存在。**

两种后端，**同一条代码路径**（`--pretrained` 切换）：

| 后端 | 说明 |
| --- | --- |
| `scratch`（默认） | 字符级 Transformer **从零训练**（`TinyCharEncoder`） |
| `pretrained` | 预训练中文编码器微调，如 `uer/chinese_roberta_L-2_H-128` |

刻意做四件事保证结论可信：

1. **同口径**：直接 import `eval_harness.group_folds`（同 seed / 同 5 折 / 同 `group_key`），
   并用 `l1_data.assemble_route` 走与线上一致的三层合成（L0 规则 → L1 → 反问规则）。
2. **同版本**：在同一次运行里把词法基线也重跑一遍 ⇒ 排除 sklearn 版本差异
   （系统 python 是 1.5.2，后端 venv 是 1.9.1）。
3. **不偷看测试折**：词表 / 权重 / 最佳 epoch 全部只用**训练折**（最佳 epoch 由训练折内部再切出的
   验证折选），测试折只在前向推理时被触碰一次。
4. **两个口径并列**：`纯模型` 与 `+规则`。前者衡量模型自己的语义能力（与词法基线比），
   后者是线上真实质量。

环境
----
训练要 GPU ⇒ 走**系统 Python**（有 torch，无 polars）；契约因此从
`router_contract` 读（实时优先、`dataset/contract_snapshot.json` 兜底）。

`pretrained` 后端需要 `transformers`；HF 官方被墙时用镜像
（`HF_ENDPOINT=https://hf-mirror.com`，脚本内已默认设置），缓存放 `backend/.hf-cache`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ⚠️ 必须在 import transformers / huggingface_hub 之前设好端点与缓存位置。
# hf-mirror 实测可达，huggingface.co 在本机超时；缓存放项目内便于查看与删除。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HOME", str(_BACKEND_ROOT / ".hf-cache"))

import eval_harness as H  # noqa: E402
import l1_data as D  # noqa: E402
from app.local_router.escalation_rules import detect_escalation  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

PAD, UNK = 0, 1

DEFAULTS = dict(
    d=128, layers=2, heads=4, ffn=256, dropout=0.10, max_len=128,
    batch=32, lr=3e-3, wd=0.01, epochs=60, label_smoothing=0.10,
    folds=5, seed=20260922, inner_seed=1, device="cuda",
    pretrained="", enc_lr=5e-5,
)


def utterance_of(sample: dict) -> str:
    return str((sample.get("request") or {}).get("utterance") or "")


# ==================================================================== 模型

class TinyCharEncoder(nn.Module):
    """字符嵌入 + 位置嵌入 + Transformer 编码器 + 掩码平均池化 + 线性头。"""

    def __init__(self, vocab: int, n_labels: int, *, d: int, layers: int, heads: int,
                 ffn: int, max_len: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len + 1, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False：norm_first=True 时它本来就不生效，关掉以免刷警告
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_labels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        h = self.drop(self.emb(x) + self.pos.weight[:length].unsqueeze(0))
        h = self.enc(h, src_key_padding_mask=~mask)
        # 掩码平均池化：忽略 padding，避免短句被 <pad> 稀释
        w = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.head(self.norm(pooled))


# ==================================================================== 后端

class ScratchBackend:
    """字符级 Transformer，从零训练。词表只能从**训练折**构建。"""

    name = "scratch"

    def __init__(self, train_rows: list[dict], label_space: list[str], args, device: str):
        self.args, self.device = args, device
        counter = Counter(ch for s in train_rows for ch in D.text_of(s))
        self.itos = ["<pad>", "<unk>"] + [c for c, _ in counter.most_common()]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.module = TinyCharEncoder(
            len(self.itos), len(label_space), d=args.d, layers=args.layers, heads=args.heads,
            ffn=args.ffn, max_len=args.max_len, dropout=args.dropout).to(device)
        self.meta = {"vocab": len(self.itos)}

    def logits(self, samples: list[dict]) -> torch.Tensor:
        seqs = [[self.stoi.get(ch, UNK) for ch in D.text_of(s)[:self.args.max_len]] or [UNK]
                for s in samples]
        width = max(len(q) for q in seqs)
        x = torch.full((len(seqs), width), PAD, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, q in enumerate(seqs):
            x[i, :len(q)] = torch.tensor(q, dtype=torch.long)
            mask[i, :len(q)] = True
        return self.module(x.to(self.device), mask.to(self.device))

    def optimizer(self):
        return torch.optim.AdamW(self.module.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.wd)


class PretrainedBackend:
    """预训练中文编码器微调（含自带分词器）。"""

    name = "pretrained"

    def __init__(self, train_rows: list[dict], label_space: list[str], args, device: str):
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # 每折加载都会打一份 LOAD REPORT + 进度条，5 折就把日志淹了；这两句压掉噪音。
        # （分类头/pooler 必然是 MISSING 的：源权重是 MLM 任务，本任务要新建 31 类头。）
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()

        self.args, self.device = args, device
        name = args.pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.module = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=len(label_space)).to(device)
        self.meta = {
            "pretrained_name": name,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "hf_home": os.environ.get("HF_HOME"),
            "tok_vocab": self.tokenizer.vocab_size,
        }

    def logits(self, samples: list[dict]) -> torch.Tensor:
        batch = self.tokenizer(
            [D.text_of(s) for s in samples],
            padding=True, truncation=True, max_length=self.args.max_len,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.module(**batch).logits

    def optimizer(self):
        # 编码器用小学习率、分类头用大 20 倍 —— 微调小模型的常规做法。
        head, enc = [], []
        for n, p in self.module.named_parameters():
            (head if n.startswith(("classifier", "pooler")) else enc).append(p)
        return torch.optim.AdamW(
            [{"params": enc, "lr": self.args.enc_lr},
             {"params": head, "lr": self.args.enc_lr * 20}],
            weight_decay=self.args.wd)


def build_backend(kind: str, train_rows, label_space, args, device):
    cls = PretrainedBackend if kind == "pretrained" else ScratchBackend
    return cls(train_rows, label_space, args, device)


# ==================================================================== 训练 / 推理

@torch.no_grad()
def predict(backend, samples: list[dict], index_to_label: list[str],
            batch: int = 128) -> list[str]:
    backend.module.eval()
    out: list[str] = []
    for i in range(0, len(samples), batch):
        logits = backend.logits(samples[i:i + batch])
        out.extend(index_to_label[j] for j in logits.argmax(-1).tolist())
    return out


def accuracy(pred: list[str], truth: list[str]) -> float:
    return sum(p == t for p, t in zip(pred, truth)) / len(truth) if truth else 0.0


def train_fold(rows: list[dict], args, device: str, label_space: list[str], kind: str) -> dict:
    """训练一折。词表 / 权重 / 最佳 epoch 全部只用训练折（内含验证折选 epoch）。"""
    label_index = {lab: i for i, lab in enumerate(label_space)}
    index_to_label = list(label_space)

    pool = D.trainable_rows(rows)
    (tr, va) = H.group_folds(pool, n_folds=args.folds, seed=args.inner_seed)[0]
    gold_va = [D.l1_label(s) for s in va]

    torch.manual_seed(args.seed)
    backend = build_backend(kind, tr, label_space, args, device)
    opt = backend.optimizer()
    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    rng = np.random.default_rng(args.seed)
    best = {"acc": -1.0, "epoch": 0, "state": None}

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        backend.module.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), args.batch):
            chunk = [tr[j] for j in order[i:i + args.batch]]
            y = torch.tensor([label_index[D.l1_label(s)] for s in chunk],
                             dtype=torch.long, device=device)
            loss = lossf(backend.logits(chunk), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(backend.module.parameters(), 1.0)
            opt.step()
        acc = accuracy(predict(backend, va, index_to_label), gold_va)
        if acc >= best["acc"]:
            best = {"acc": acc, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in backend.module.state_dict().items()}}
    train_s = time.perf_counter() - t0

    backend.module.load_state_dict(best["state"])
    return {"backend": backend, "n_params": sum(p.numel() for p in backend.module.parameters()),
            "best_epoch": best["epoch"], "val_acc": best["acc"], "train_s": train_s}


@torch.no_grad()
def latency_ms(backend, samples: list[dict], device: str, batch_size: int,
               repeats: int = 20) -> float:
    """一次前向的端到端延迟（**含分词/编码**），按调用次数摊薄 ⇒ 单位 ms·次⁻¹。

    ⚠️ 必须同时报 `batch_size=1`：线上 Router 就是一次一句话，
    只报大批次的数字会把 Python/核函数启动开销摊薄掉，属于**不诚实的乐观**。
    """
    backend.module.eval()
    probe = samples[:batch_size]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(3):  # 预热
        backend.logits(probe)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        backend.logits(probe)
    sync()
    return (time.perf_counter() - t0) * 1000.0 / repeats


# ==================================================================== 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    kind = "pretrained" if args.pretrained else "scratch"
    if not args.tag:
        args.tag = args.pretrained.split("/")[-1] if kind == "pretrained" else "scratch"

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    rows = H.load_split("all")
    label_space = D.l1_label_space()
    print(f"后端 {kind}" + (f" | {args.pretrained}" if kind == "pretrained" else ""))
    print(f"设备 {device}" + (f" | {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    print(f"样本 {len(rows)} 条 ⇒ L1 可训练 {len(D.trainable_rows(rows))} 条；"
          f"标签空间 {len(label_space)} 类（chat + {len(label_space) - 1} 工具）")
    print(f"契约来源 {D.contract_source()}"
          + ("（读工具注册表）" if D.contract_source() == "live" else "（读冻结快照）"))
    print(f"配置 max_len={args.max_len} batch={args.batch} epochs={args.epochs} "
          f"lr={args.lr if kind == 'scratch' else args.enc_lr} dropout={args.dropout}")
    if kind == "pretrained":
        print(f"HF 端点 {os.environ.get('HF_ENDPOINT')} ｜ 缓存 {os.environ.get('HF_HOME')}")

    # ---- 词法基线：本次运行内重算，保证同 sklearn 版本 ----
    baseline, baseline_fn = None, None
    if not args.skip_baseline:
        import baseline_lexical as B
        t0 = time.perf_counter()
        p_b, tb, _, _ = B.run_group_cv(rows, label_fn=B.collapsed_label, post=B.apply_ask_rule)
        b_by_id = {s["id"]: q for s, q in zip(tb, p_b)}

        def baseline_fn(s):
            r = detect_escalation(utterance_of(s))
            return f"escalate::{r.value}" if r else b_by_id.get(s["id"], "chat")

        baseline = H.evaluate(baseline_fn, tb, name="词法基线(groupCV)")
        print(f"词法基线本次重算 {time.perf_counter() - t0:.1f}s")

    # ---- 分组 CV ----
    print()
    raw_preds: dict[str, str] = {}
    routed: dict[str, str] = {}
    test_rows: list[dict] = []
    folds = []
    last = None
    for i, (tr, te) in enumerate(H.group_folds(rows, n_folds=args.folds, seed=args.seed), 1):
        out = train_fold(tr, args, device, label_space, kind)
        last = out
        l1_pred = predict(out["backend"], te, list(label_space))
        for s, lp in zip(te, l1_pred):
            raw_preds[s["id"]] = lp
            routed[s["id"]] = D.assemble_route(s, lp)
        test_rows.extend(te)
        folds.append({"fold": i, "train_l1": len(D.trainable_rows(tr)), "test": len(te),
                      "best_epoch": out["best_epoch"], "val_acc": round(out["val_acc"], 4),
                      "train_s": round(out["train_s"], 2)})
        print(f"  fold {i}: train_l1={folds[-1]['train_l1']:3d} test={len(te):3d} "
              f"best_epoch={out['best_epoch']:3d} val={out['val_acc']:.1%} {out['train_s']:.1f}s")

    # ---- 两个口径 ----
    rep_pure = H.evaluate(lambda s: raw_preds[s["id"]], test_rows, name=f"L1-{args.tag} 纯模型")
    rep_full = H.evaluate(lambda s: routed[s["id"]], test_rows, name=f"L1-{args.tag}+规则")
    reports = ([baseline] if baseline else []) + [rep_pure, rep_full]

    ms1 = latency_ms(last["backend"], test_rows, device, 1)
    batch_n = len(test_rows)
    ms_batch = latency_ms(last["backend"], test_rows, device, batch_n, repeats=5)
    ms_amortized = ms_batch / batch_n
    n_params = last["n_params"]
    size_mb = n_params * 4 / 1e6

    print()
    print(H.format_report(reports))
    print(f"\n参数量 {n_params:,}（fp32 约 {size_mb:.2f} MB） | 设备 {device}")
    print(f"推理延迟：单条 {ms1:.2f} ms·次⁻¹ ｜ 整批 {ms_batch:.1f} ms·{batch_n}条⁻¹"
          f"（= {ms_amortized:.3f} ms·条⁻¹）。**线上问的是单条，用前者**。")
    print(f"训练合计 {sum(f['train_s'] for f in folds):.1f}s（{len(folds)} 折）")

    # ---- L1 纯模型混淆：残留错在哪 ----
    conf = Counter()
    for s in test_rows:
        gold = D.l1_label(s)
        if gold is not None and raw_preds[s["id"]] != gold:
            conf[(gold, raw_preds[s["id"]])] += 1
    print("\nL1 纯模型主要混淆（金标→预测，前 10）：")
    for (g, p), c in conf.most_common(10):
        print(f"  {g:26s} → {p:26s} ×{c}")

    # ---- 误差互补性：对比对象必须是**部署形态**基线，不是裸 SVC 输出 ----
    complement = None
    if baseline_fn is not None:
        gold = {s["id"]: H.route_label(s) for s in test_rows}
        base = {s["id"]: baseline_fn(s) for s in test_rows}
        n = len(gold)
        both = sum(1 for i in gold if base[i] == gold[i] and routed[i] == gold[i])
        b_only = sum(1 for i in gold if base[i] == gold[i] and routed[i] != gold[i])
        l_only = sum(1 for i in gold if base[i] != gold[i] and routed[i] == gold[i])
        neither = sum(1 for i in gold if base[i] != gold[i] and routed[i] != gold[i])
        complement = {
            "both_right": both, "baseline_only": b_only, "l1_only": l_only,
            "both_wrong": neither, "n": n,
            "l1_fixes_share_of_baseline_errors": round(l_only / max(l_only + neither, 1), 4),
            "best_single_model": round((both + max(b_only, l_only)) / n, 4),
            "perfect_selector_ceiling": round((both + b_only + l_only) / n, 4),
        }
        print(f"\n误差互补性（整条 route，n={n}）：都对 {both} ｜ 仅词法对 {b_only} ｜ "
              f"仅 L1 对 {l_only} ｜ 都错 {neither}")
        print(f"  ⇒ L1 补上了词法基线的 {l_only / max(l_only + neither, 1):.1%} 的错误；"
              f"最佳单模型 {(both + max(b_only, l_only)) / n:.1%}；"
              f"完美选择器上限 {(both + b_only + l_only) / n:.1%}（融合空间）")

    extras = {
        "backend_kind": kind,
        "backend_meta": last["backend"].meta,
        "config": {k: getattr(args, k) for k in DEFAULTS},
        "device": device,
        "n_params": n_params,
        "size_mb_fp32": round(size_mb, 2),
        "infer_ms_single": round(ms1, 3),
        "infer_ms_amortized": round(ms_amortized, 4),
        "contract_source": D.contract_source(),
        "train_seconds_total": round(sum(f["train_s"] for f in folds), 2),
        "folds": folds,
        "label_space_size": len(label_space),
        "trainable_rows": len(D.trainable_rows(rows)),
        "top_confusions": [[g, p, c] for (g, p), c in conf.most_common(15)],
        "complementarity": complement,
    }
    notes = {
        "purpose": "回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。",
        "fairness": "同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。",
        "no_test_peek": "词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。",
        "escalate_excluded": "升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。",
    }

    out_dir = H.DATASET_DIR
    (out_dir / f"l1_report_{args.tag}.json").write_text(json.dumps(
        {"reports": reports, "extras": extras, "notes": notes,
         "table_markdown": H.format_report(reports),
         "generated_at_epoch": int(time.time())}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    md = [f"# L1 基线对照（{args.tag}）", "",
          "> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；"
          "三层合成 `l1_data.assemble_route`。", "",
          f"- 后端：{kind}" + (f"（`{args.pretrained}`）" if kind == "pretrained"
                              else "（字符级从零训练）"),
          f"- 设备：{device}" + (f"（{torch.cuda.get_device_name(0)}）" if device == "cuda" else ""),
          f"- 参数量：{n_params:,}（fp32 ≈ {size_mb:.2f} MB）",
          f"- 推理：单条 {ms1:.2f} ms·次⁻¹ / 整批 {ms_batch:.1f} ms·{batch_n}条⁻¹"
          f"（= {ms_amortized:.3f} ms·条⁻¹）",
          f"- 训练合计 {extras['train_seconds_total']}s / {len(folds)} 折",
          f"- 标签空间：{len(label_space)} 类；L1 可训练样本 {extras['trainable_rows']} 条",
          f"- 契约来源：{D.contract_source()}", "",
          "## 指标对比", "", H.format_report(reports), "", "## 分组折明细", "",
          "| fold | train_l1 | test | best_epoch | val_acc | train_s |",
          "| --- | --- | --- | --- | --- | --- |"]
    for f in folds:
        md.append(f"| {f['fold']} | {f['train_l1']} | {f['test']} | {f['best_epoch']} | "
                  f"{f['val_acc']:.1%} | {f['train_s']} |")
    md += ["", "## L1 纯模型混淆（金标→预测）", "",
           "| 金标 | 预测 | 次数 |", "| --- | --- | --- |"]
    for g, p, c in extras["top_confusions"]:
        md.append(f"| {g} | {p} | {c} |")
    if complement:
        md += ["", "## 误差互补性（L1 vs 词法基线，整条 route）", "",
               "| 组合 | 条数 |", "| --- | --- |",
               f"| 两者都对 | {complement['both_right']} |",
               f"| 仅词法基线对 | {complement['baseline_only']} |",
               f"| 仅 L1 对 | {complement['l1_only']} |",
               f"| 两者都错 | {complement['both_wrong']} |", "",
               f"L1 补上了词法基线 **{complement['l1_fixes_share_of_baseline_errors']:.1%}** 的错误；"
               f"最佳单模型 **{complement['best_single_model']:.1%}**；"
               f"完美选择器上限 **{complement['perfect_selector_ceiling']:.1%}**。", "",
               "> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；"
               "它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。"]
    md += ["", "## 口径说明", ""] + [f"- **{k}**：{v}" for k, v in notes.items()]
    (out_dir / f"l1_report_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n已写出 dataset/l1_report_{args.tag}.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
