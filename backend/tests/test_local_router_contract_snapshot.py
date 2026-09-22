"""钉住两件事：契约快照不漂移、三层路由合成的不变式。

背景
----
训练 L1 要用 GPU ⇒ 走**系统 Python**（有 torch、无 polars）；而后端工具注册需要 polars。
于是契约被冻结成 `scripts/router/dataset/contract_snapshot.json`（由 `export_contract.py`
在 venv 下生成），训练侧读快照。**快照一旦与实时契约脱节，训练用的标签空间就可能是错的**
—— 这会导致「训练时 30 个工具、推理时 29 个」这类极难排查的错配。所以这里把一致性钉死。

同时钉住 `l1_data.assemble_route` 的三层合成语义（L0 规则 → L1 → 反问规则）：
它既是离线评测口径，也是线上推理口径，**必须是同一个函数**，因此它的每个分支都值得测。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROUTER_DIR = _BACKEND / "scripts" / "router"
if str(_ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROUTER_DIR))

from app.local_router import contract as C  # noqa: E402

SNAPSHOT = _ROUTER_DIR / "dataset" / "contract_snapshot.json"


def _snapshot() -> dict:
    if not SNAPSHOT.exists():
        pytest.skip("contract_snapshot.json 未生成（先跑 scripts/router/export_contract.py）")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


# --------------------------------------------------------- 快照一致性

def test_snapshot_tool_list_matches_live_registry():
    snap = _snapshot()
    assert snap["tools"] == C.tool_label_space()


def test_snapshot_required_params_match_live_registry():
    snap = _snapshot()
    live = {t: C.required_params(t) for t in C.tool_label_space()}
    assert snap["required_params"] == live


def test_snapshot_label_space_matches_contract():
    snap = _snapshot()
    assert snap["label_space"] == ["chat"] + ["call::" + t for t in C.tool_label_space()]


def test_snapshot_source_hash_is_current():
    """改过 `contract.py` 就必须重跑 `export_contract.py`，否则快照已过期。"""
    snap = _snapshot()
    live_sha1 = hashlib.sha1(Path(C.__file__).read_bytes()).hexdigest()
    assert snap["source_sha1"] == live_sha1, (
        "contract.py 已变更但契约快照未重新导出 —— 请运行 "
        "`backend/.venv/Scripts/python.exe backend/scripts/router/export_contract.py`"
    )


def test_live_registry_wins_when_available():
    """venv 里（有 polars）必须走实时契约，快照只是兜底。"""
    from router_contract import contract_source

    assert contract_source() == "live"


# --------------------------------------------------------- 三层合成

def _tool_needing_dataset_id() -> str:
    for name in C.tool_label_space():
        if C.required_params(name) == ["dataset_id"]:
            return name
    pytest.skip("没有仅以 dataset_id 为必填的工具，本用例不适用")


def _sample(utterance: str, bound: int | None = None) -> dict:
    return {
        "id": "probe",
        "request": {
            "utterance": utterance,
            "bound_dataset_id": bound,
            "available_columns": ["a", "b"],
            "recent_tools": [],
        },
        "target": {"escalate": False, "tool": None, "params": {}, "missing": []},
    }


def _assemble():
    from l1_data import assemble_route

    return assemble_route


def test_escalation_rule_has_priority_over_l1():
    """L0 规则优先：哪怕 L1 说了个工具，规则判升级就必须升级。"""
    s = _sample("把这个数据集同步到我们的 S3 存储桶")
    assert _assemble()(s, "call::dataset.quality").startswith("escalate::")


def test_unbound_dataset_turns_call_into_ask():
    """必填槽位缺失时是**反问**，不是执行、更不是升级。"""
    tool = _tool_needing_dataset_id()
    s = _sample("对这批数据做质量检查", bound=None)
    assert _assemble()(s, "call::" + tool) == "ask::" + tool


def test_bound_dataset_executes_directly():
    tool = _tool_needing_dataset_id()
    s = _sample("对这批数据做质量检查", bound=7)
    assert _assemble()(s, "call::" + tool) == "call::" + tool


def test_ask_prefix_is_idempotent_not_double_wrapped():
    """已带 ask 前缀的输入不得被二次套用规则而变成升级（曾出现的隐患）。"""
    tool = _tool_needing_dataset_id()
    s = _sample("对这批数据做质量检查", bound=None)
    assert _assemble()(s, "ask::" + tool) == "ask::" + tool


def test_chat_and_missing_prediction_paths():
    s = _sample("你好")
    assert _assemble()(s, "chat") == "chat"
    # L1 未表态属异常，保守升级优于硬做错
    assert _assemble()(s, None) == "escalate::ambiguous"


def test_label_space_never_contains_unknown_tool():
    """L1 的标签空间只能来自注册表 ⇒ 结构上不可能输出不存在的工具。"""
    from l1_data import l1_label_space

    registered = set(C.tool_label_space())
    for label in l1_label_space():
        if label == "chat":
            continue
        assert label.startswith("call::")
        assert label.split("::", 1)[1] in registered
