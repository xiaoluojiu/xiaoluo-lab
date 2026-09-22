"""本地 Router 的**分数归一到概率**的那一步 —— 运行时与离线标定必须共用同一份。

为什么单独成模块：这一步决定 `confidence`，而 `confidence` 决定门控（低置信度就升级）。
如果训练侧与运行侧各写一份，阈值就不可迁移 —— 标定出来的 θ 到线上会失效。
本模块**只依赖 numpy**，因此实验脚本（`.venv-l1`，无 pydantic）也能安全引用，
不会被迫加载 `app.local_router.__init__` 里的契约层。

两步处理的缘由（与 `fuse_l1_lexical.py` 的离线实验一致）
------------------------------------------------------
1. `LinearSVC.decision_function` 是多类 one-vs-rest 的原始分数，**尺度取决于边距**，
   直接 softmax 会得到极端尖锐或平坦的分布；
2. 所以先**逐行 z-score**（`rownorm`）把尺度拉平，再 softmax —— 得到的才是可比较的置信度。
3. 分类器只见过训练折里出现过的类，必须按**完整标签空间**对齐（`align_scores`），
   缺席的类填 `NEG`（softmax 后≈0），否则列错位而指标看起来还挺正常。
"""

from __future__ import annotations

import numpy as np

# softmax 前给「缺席类」的填充值。不用 -inf 是为了避免极端情况下的 nan。
NEG = -1e9

__all__ = ["NEG", "align_scores", "proba_from_scores", "rownorm", "softmax", "top1"]


def rownorm(z: np.ndarray) -> np.ndarray:
    """逐行 z-score：把多类边距分数拉到可比尺度。标准差为 0 的行保持 0。"""
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    return (z - mu) / np.where(sd < 1e-9, 1.0, sd)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_scores(classes: list[str], matrix: np.ndarray, label_space: list[str]) -> np.ndarray:
    """把 `classes` 的列对齐到完整 `label_space`；缺席的类填 `NEG`。"""
    out = np.full((matrix.shape[0], len(label_space)), NEG, dtype=float)
    index = {label: i for i, label in enumerate(label_space)}
    for j, name in enumerate(classes):
        if name in index:
            out[:, index[name]] = matrix[:, j]
    return out


def proba_from_scores(classes: list[str], scores: np.ndarray, label_space: list[str]) -> np.ndarray:
    """原始分数 → 归一化概率（逐行 z-score → 对齐 → softmax）。"""
    return softmax(rownorm(align_scores(classes, scores, label_space)))


def top1(proba: np.ndarray, label_space: list[str]) -> tuple[str, float]:
    """取最可能的标签与其置信度。"""
    index = int(np.argmax(proba, axis=1)[0])
    return label_space[index], float(proba[0, index])
