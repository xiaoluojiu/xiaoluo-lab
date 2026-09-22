"""本地 Router 的模型侧：输入口径、加载、预测、置信度。

选型：**TF-IDF char_wb(1-4gram) + LinearSVC**（词法模型）
--------------------------------------------------------
这是离线对照实验的结论，不是随手选的（详见 `scripts/router/L1_EXPERIMENT_SUMMARY.md`）：

| 方案 | 部署态 route | 代价 |
| --- | --- | --- |
| **本模块（词法）** | **80.7%** | 3.5MB / 0.008ms / **零新依赖** |
| 字符 Transformer 从零训 | 73.1% | 需 torch |
| 预训练中文 RoBERTa L4H256 | 79.5% | 需 torch（≈2.5GB） |
| 词法 + 神经 融合（α=0.75） | 85.7% | 需 torch，或先做 ONNX 导出 |

⇒ 单神经模型**打不过**它；融合更强但后端要背 2.5GB 的 torch。
所以运行时先上零新依赖的这一版，融合留作后续（ONNX 路线）。

失败模式（刻意选择）
--------------------
模型缺失 / 过期 / 加载失败 ⇒ `get_model()` 返回 **None**，由上层保守处理（升级给云端），
**绝不猜**。宁可在 shadow 模式里记录一条 `model_unavailable`，也不要静默给出错误路由。
"""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.local_router import contract as C
from app.local_router.scoring import proba_from_scores, top1

logger = logging.getLogger(__name__)

# L1 把「闲聊」当作与 30 个工具并列的一个类。
CHAT_LABEL = "chat"

# 工具名标签前缀：标签空间是 `chat` 或 `call::<tool>`（不加前缀说明就是这个工具名）。
CALL_PREFIX = "call::"

__all__ = [
    "CALL_PREFIX",
    "CHAT_LABEL",
    "ARTIFACT_NAME",
    "LexicalRouterModel",
    "artifact_path",
    "get_model",
    "l1_label_space",
    "request_text",
    "reset_model_cache",
    "tool_of_label",
]


# ---------------------------------------------------------------------------
# 输入口径 —— 训练与推理必须完全一致，否则特征分布漂移
# ---------------------------------------------------------------------------


def request_text(req: Any) -> str:
    """把 RouterRequest 摊平成模型输入文本（**训练与推理的唯一口径**）。

    utterance 是主信号；「是否已绑定数据集 / 可见列 / 近期工具」都是 Router 真实可得的
    结构化信号，一并编码 —— 这是 implicit / follow_up 类样本能被判对的前提。

    ⚠️ 绑定状态必须**双向**编码：`ask::<tool>`（工具已定但缺 dataset_id）与
    `call::<tool>`（可直接执行）的差别**不在措辞里**，只在 `bound_dataset_id` 是否为 null。
    若只写「已绑定」而不写「未绑定」，这两类在词面上完全同分布 ⇒ 模型不可能学会。

    `req` 接受 dict 或 `contract.RouterRequest`（两者字段同名）。
    """
    get = (lambda k: getattr(req, k, None)) if not isinstance(req, dict) else req.get
    utterance = str(get("utterance") or "")
    cols = list(get("available_columns") or [])
    bound = get("bound_dataset_id")
    recent = list(get("recent_tools") or [])

    parts = [utterance, f"可见列数={len(cols)}"]
    if cols:
        parts.append("列 " + " ".join(map(str, cols)))
    parts.append("已绑定数据集" if bound is not None else "未绑定数据集")
    if recent:
        parts.append("近期 " + " ".join(map(str, recent)))
    return " ".join(parts).strip()


def l1_label_space() -> list[str]:
    """L1 的闭合标签空间：`chat` + 30 个工具。

    工具名**动态取自注册表** ⇒ 模型在结构上不可能输出平台不存在的工具。
    平台增删工具后模型会失效，由 `LexicalRouterModel.staleness()` 检出并要求重训。
    """
    return [CHAT_LABEL] + [CALL_PREFIX + name for name in C.tool_label_space()]


def tool_of_label(label: str | None) -> str | None:
    """`call::eda.distribution` → `eda.distribution`；`chat` 或异常值 → None。"""
    if not label:
        return None
    text = str(label)
    if text.startswith(CALL_PREFIX):
        return text[len(CALL_PREFIX):] or None
    if text == CHAT_LABEL or "::" in text:
        return None
    return text  # 容错：历史产物里可能直接存工具名


# ---------------------------------------------------------------------------
# 产物路径与加载
# ---------------------------------------------------------------------------

ARTIFACT_NAME = "lexical_l1.pkl"


def artifact_path() -> Path:
    """产物路径：`{MODEL_ROOT}/local_router/lexical_l1.pkl`（`models/` 已在 .gitignore）。

    延迟 import settings —— 本模块会被离线脚本引用，不希望它们被迫拉起整个配置层。
    """
    from app.core.config import settings

    root = Path(getattr(settings, "LOCAL_ROUTER_MODEL_DIR", "") or (settings.model_root_path / "local_router"))
    return root if root.suffix else (root / ARTIFACT_NAME)


@dataclass
class LexicalRouterModel:
    """加载好的词法 Router 模型。

    只承载「预测」这一件事：把文本映射到 `chat` / `call::<tool>` 与置信度。
    三层合成（L0 规则 / 反问规则 / 门控）在 `router.py`，职责分离便于各自单测。
    """

    vectorizer: Any
    classifier: Any
    label_space: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def staleness(self) -> str | None:
        """返回过期原因；`None` 表示与当前平台能力一致。

        为什么必须查：标签空间是从工具注册表派生的。平台加了一个工具后，
        旧模型少一个类 —— 它永远不会输出这个新工具，表现为「新功能用不了」，
        而且**不报错**。这是一类必须主动探测的静默失效。
        """
        live_tools = sorted(C.tool_label_space())
        trained_tools = sorted(self.meta.get("tool_label_space") or [])
        if not trained_tools:
            return "产物未记录训练时的工具清单，无法判断是否过期"
        if trained_tools != live_tools:
            added = sorted(set(live_tools) - set(trained_tools))
            removed = sorted(set(trained_tools) - set(live_tools))
            return f"平台工具清单已变化（新增 {added or '—'}，移除 {removed or '—'}），需重训"
        live_space = l1_label_space()
        if list(self.label_space) != live_space:
            return "标签空间与当前契约不一致，需重训"
        return None

    # -- 预测 ---------------------------------------------------------------

    def proba(self, text: str) -> np.ndarray:
        """单条文本 → 完整标签空间的概率向量（形状 `(1, n_classes)`）。"""
        matrix = self.vectorizer.transform([text])
        scores = np.asarray(self.classifier.decision_function(matrix), dtype=float)
        if scores.ndim == 1:  # 二分类退化为 (n,)，补一列以便统一处理
            scores = np.column_stack([-scores, scores])
        classes = [str(name) for name in getattr(self.classifier, "classes_", [])]
        return proba_from_scores(classes, scores, list(self.label_space))

    def predict(self, text: str) -> tuple[str, float]:
        """返回 `(label, confidence)`；label ∈ `chat` / `call::<tool>`。"""
        return top1(self.proba(text), list(self.label_space))

    def predict_for(self, req: Any) -> tuple[str, float]:
        return self.predict(request_text(req))


def _load_from(path: Path) -> LexicalRouterModel:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "vectorizer" not in payload:
        raise ValueError(f"{path} 不是预期的 Router 产物结构")
    return LexicalRouterModel(
        vectorizer=payload["vectorizer"],
        classifier=payload["classifier"],
        label_space=[str(x) for x in payload.get("label_space") or []],
        meta=dict(payload.get("meta") or {}),
        path=path,
    )


_MODEL_LOCK = threading.Lock()
_MODEL: LexicalRouterModel | None = None
_MODEL_FAILURE: str | None = None  # 记住失败原因，避免每条请求都打日志


def get_model(*, refresh: bool = False, path: Path | None = None) -> LexicalRouterModel | None:
    """取进程级共享的模型（懒加载 + 锁）。取不到返回 None 并只记一次日志。

    「只记一次」很重要：shadow 模式下每来一条请求都会问一次，
    若每次都打 traceback，日志会被淹掉，反而看不出真正的问题。
    """
    global _MODEL, _MODEL_FAILURE
    with _MODEL_LOCK:
        if _MODEL is not None and not refresh:
            return _MODEL
        target = path or artifact_path()
        if not target.exists():
            if _MODEL_FAILURE != f"missing:{target}":
                _MODEL_FAILURE = f"missing:{target}"
                logger.warning("本地 Router 模型不存在：%s（请先跑 train_runtime_l1.py）", target)
            return None
        try:
            model = _load_from(target)
        except Exception as exc:  # noqa: BLE001 — 加载失败不该拖垮请求链路
            _MODEL_FAILURE = f"load:{target}"
            logger.warning("本地 Router 模型加载失败（%s）：%s", target, exc)
            return None
        stale = model.staleness()
        if stale:
            _MODEL_FAILURE = f"stale:{target}"
            logger.warning("本地 Router 模型已过期：%s", stale)
            return None
        _MODEL, _MODEL_FAILURE = model, None
        return _MODEL


def reset_model_cache() -> None:
    """仅供测试使用：清掉缓存，让下一次 `get_model()` 重新读盘。"""
    global _MODEL, _MODEL_FAILURE
    with _MODEL_LOCK:
        _MODEL, _MODEL_FAILURE = None, None
