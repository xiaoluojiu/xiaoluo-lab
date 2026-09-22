"""L1（本地小模型）的数据装配 —— 训练与评测共用的**单一事实源**。

L1 到底学什么
-------------
把整条决策拆到三层之后，L1 的职责被压到最小：**只学「这句话在说什么」，即工具选择**。

| 决策 | 谁来做 | 为什么 |
| --- | --- | --- |
| 该不该升级 | **L0 结构规则** `detect_escalation()` | 五类原因都是结构/词表信号；规则 F1 98.2% ≫ 学习版 51.3% |
| 选哪个工具 / 是否闲聊 | **L1 本模块** | 唯一真正需要语义（换一种说法仍认得）的部分 |
| 工具已定但缺必填槽位 | **反问规则**（纯函数） | `required_params ∖ resolved` 是符号计算，学习版 F1 8.9~47% |

因此 L1 的标签空间 = **1（chat）+ 30（工具）= 31 类**，`escalate` 样本**不参与训练**
（它们由 L0 规则兜住）。这与线上真实调用顺序一致：先过规则，再到模型。

⚠️ 评测集仍然是**全部 844 条**：升级样本也出现在测试折里，但它们的路线由规则给出。
这样得到的数字与 `escalation_rules_eval.py` 的 `hybrid+规则` 列**完全同口径可比**。

契约来源（为什么有一个「快照」分支）
------------------------------------
训练要用 GPU ⇒ 走**系统 Python**（有 torch 2.7.1+cu118）；而 `contract.tool_label_space()`
会触发 `register_builtin_tools()` → `app.analysis` → `import polars`，
**系统 Python 没有 polars** ⇒ 实时契约在训练环境里取不到。

于是 `export_contract.py`（在 venv 下跑）把契约冻结成 `dataset/contract_snapshot.json`，
由 `router_contract.py` 统一按「实时优先、快照兜底」解析，本模块只是转出它；
`contract_source()` 可如实报告实际用的是哪一个。快照与实时契约的一致性由
`tests/test_local_router_contract_snapshot.py` 断言。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_harness as H  # noqa: E402
from app.local_router.escalation_rules import detect_escalation  # noqa: E402
from router_contract import CONTRACT, CONTRACT_SOURCE, contract_source  # noqa: E402

__all__ = [
    "CHAT_LABEL",
    "CONTRACT",
    "CONTRACT_SOURCE",
    "RULE_RESOLVABLE",
    "assemble_route",
    "contract_source",
    "l1_label",
    "l1_label_space",
    "text_of",
    "trainable_rows",
]

# chat 作为 L1 的一个类，与 30 个工具并列。
CHAT_LABEL = "chat"

# 规则能无条件判定的槽位（与 `contract.decide_route` 的职责一致）。
RULE_RESOLVABLE = {"dataset_id"}


def text_of(sample: dict) -> str:
    """模型输入文本 —— 与词法基线**完全同一口径**，否则对比不公平。"""
    return H.request_text(sample.get("request") or {})


def l1_label_space() -> list[str]:
    """L1 的闭合标签空间。工具名动态取自注册表 ⇒ 永不输出不存在的工具。"""
    return [CHAT_LABEL] + ["call::" + name for name in CONTRACT.tool_label_space()]


def l1_label(sample: dict) -> str | None:
    """L1 的学习标签；返回 None 表示**不归 L1 管**（升级交给 L0 规则）。"""
    label = H.route_label(sample)
    kind = H.kind_of(label)
    if kind == "chat":
        return CHAT_LABEL
    if kind in ("call", "ask"):
        # ask 与 call 在 L1 眼里是同一件事（选哪个工具），是否反问由规则决定。
        return "call::" + str(H.tool_of(label))
    return None


def trainable_rows(rows: list[dict]) -> list[dict]:
    """筛出真正参与 L1 训练的样本（升级样本被剔除）。"""
    return [s for s in rows if l1_label(s) is not None]


def _missing_required(sample: dict, tool: str) -> list[str]:
    """工具已定，但 RouterRequest 里读得出必填槽位缺失 ⇒ 应当反问用户。"""
    req = sample.get("request") or {}
    resolved = {"dataset_id"} if req.get("bound_dataset_id") is not None else set()
    return [p for p in CONTRACT.required_params(tool)
            if p not in resolved and p in RULE_RESOLVABLE]


def assemble_route(sample: dict, l1_pred: str | None) -> str:
    """把三层合成最终 route 串 —— 线上推理与离线评测必须共用这一个函数。

    顺序即线上顺序：① L0 升级规则 > ② L1 工具判定 > ③ 反问规则。

    `l1_pred` 必须取自 `l1_label_space()`（`chat` 或 `call::<tool>`）；函数自身会做归一化，
    不会重复套用反问规则。
    """
    utterance = str((sample.get("request") or {}).get("utterance") or "")
    reason = detect_escalation(utterance)
    if reason is not None:
        return "escalate::" + reason.value

    if not l1_pred:
        # L1 未表态属异常；保守升级（宁可多花一次远程调用，也不要硬做错）。
        return "escalate::ambiguous"

    kind, _, tool = str(l1_pred).partition("::")
    if kind == "ask" and tool:      # 容错：已带 ask 前缀的输入按 call 处理，不重复套规则
        kind = "call"
    if kind == CHAT_LABEL:
        return CHAT_LABEL
    if kind != "call" or not tool:
        return "escalate::ambiguous"
    if _missing_required(sample, tool):
        return "ask::" + tool
    return "call::" + tool
