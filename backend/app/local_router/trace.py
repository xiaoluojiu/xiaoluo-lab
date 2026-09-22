"""本地 Router 的 **shadow 埋点** —— 并排记录「本地怎么判」与「线上实际怎么走」。

为什么必须先 shadow，而不是直接接管
------------------------------------
1. 离线指标（route 80.7%）是在**合成数据集**上算的，真实对话的措辞分布未知；
   直接接管等于用一个未在真实分布上验证过的模型替换掉一个已知稳定的规则路由。
2. 真实数据里最有价值的是**纠正信号**（用户追问「不是这个意思」、手动选了别的工具），
   这类信号只有在真实运行中才会产生，且必须与「当时 Router 判了什么」一一对应才有用。
3. shadow 对线上**零行为改动**：模型缺失、加载失败、写盘失败都只影响这条日志，
   不影响这次运行的结果。

数据形状（jsonl，一次运行两行，用 `run_id` 关联）
-------------------------------------------------
- `kind="route"`  —— 路由发生时记录：用户说了什么 + 给了哪些结构化信号 +
  本地 Router 的判定（route / tool / confidence）+ **既有规则路由**的判定（对照基线）。
- `kind="outcome"` —— 运行结束时记录：实际走的模式、计划里的工具、真正执行成功的工具、状态。

离线分析（`scripts/router/analyze_shadow.py`）按 `run_id` join 两行，即可回答
「如果当时让本地 Router 接管，会有多少条与真人实际做法不同」。

写入策略
--------
- 追加写 + 线程锁；超过 `LOCAL_ROUTER_TRACE_MAX_BYTES` 轮转为 `shadow.jsonl.1`（只留一代）。
- **任何异常都吞掉**（只记一次 warning）：埋点失败绝不能打断用户请求。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "SHADOW_FILE",
    "enabled",
    "iter_records",
    "record_outcome",
    "shadow_route",
    "trace_path",
]

SHADOW_FILE = "shadow.jsonl"

DEFAULT_MAX_BYTES = 5 * 1024 * 1024

_LOCK = threading.Lock()
_WARNED: set[str] = set()


# ---------------------------------------------------------------------------
# 配置读取（延迟 import：本模块会被离线脚本引用，不该强制拉起配置层）
# ---------------------------------------------------------------------------


def _config(name: str, default: Any) -> Any:
    try:
        from app.core.config import settings

        return getattr(settings, name, default)
    except Exception:  # noqa: BLE001
        return default


def enabled() -> bool:
    """本地 Router 是否激活（`off` 档表示完全不碰这条链路）。"""
    return str(_config("LOCAL_ROUTER_MODE", "off") or "off").strip().lower() != "off"


def trace_path() -> Path:
    """trace 与模型产物同目录，便于「一个目录看全部 Router 状态」。"""
    from app.local_router.model import artifact_path

    return artifact_path().parent / SHADOW_FILE


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message)


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------


def _rotate_if_needed(path: Path) -> None:
    """超限轮转。只保留一代历史 —— trace 是「攒数据」用的，不是审计日志。"""
    try:
        cap = int(_config("LOCAL_ROUTER_TRACE_MAX_BYTES", DEFAULT_MAX_BYTES) or DEFAULT_MAX_BYTES)
        if path.exists() and path.stat().st_size >= cap:
            path.replace(path.with_name(path.name + ".1"))
    except OSError:
        pass


def _append(record: dict[str, Any]) -> None:
    if not enabled():
        return
    try:
        path = trace_path()
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — 埋点失败不得影响业务
        _warn_once("append", f"本地 Router shadow 写入失败（不影响运行）：{exc}")


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def shadow_route(
    *,
    run_id: str,
    session_id: str,
    request: Any,
    rules_mode: str,
    rules_reason: str,
) -> None:
    """跑一次本地 Router 并落盘一条 `route` 记录。

    `request` 是 `contract.RouterRequest`（或任何字段同名的对象）。
    `rules_mode` / `rules_reason` 是**既有规则路由**给出的判定 —— 它是本次对照的基线：
    没有它，事后无法区分「本地 Router 判错」与「规则本来就判错了」。
    """
    if not enabled():
        return

    utterance = str(getattr(request, "utterance", "") or "")
    bound = getattr(request, "bound_dataset_id", None)
    columns = list(getattr(request, "available_columns", None) or [])
    recent = list(getattr(request, "recent_tools", None) or [])

    record: dict[str, Any] = {
        "kind": "route",
        "ts": time.time(),
        "run_id": run_id,
        "session_id": session_id,
        "utterance": utterance,
        "bound_dataset_id": bound,
        "n_columns": len(columns),
        "recent_tools": recent,
        "rules": {"mode": rules_mode, "reason": rules_reason},
    }

    try:
        from app.local_router.router import decision_to_route, route_request

        decision = route_request(request)
        record["router"] = {
            "available": True,
            "route": decision_to_route(decision),
            "tool": decision.tool,
            "intent": decision.intent.value if decision.intent is not None else None,
            "confidence": round(float(decision.confidence), 6),
            "escalate": bool(decision.escalate),
            "escalate_reason": (
                decision.escalate_reason.value if decision.escalate_reason is not None else None
            ),
            "missing": list(decision.missing),
        }
    except Exception as exc:  # noqa: BLE001 — Router 内部异常也不能影响运行
        record["router"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    _append(record)


def record_outcome(
    *,
    run_id: str,
    session_id: str,
    status: str,
    executed_tools: list[str | None],
    planned_tools: list[str | None],
    error: str = "",
    elapsed: float | None = None,
) -> None:
    """运行结束时落盘一条 `outcome` 记录（与同 `run_id` 的 `route` 记录 join）。"""
    if not enabled():
        return
    _append(
        {
            "kind": "outcome",
            "ts": time.time(),
            "run_id": run_id,
            "session_id": session_id,
            "status": status,
            "planned_tools": [t for t in planned_tools if t],
            "executed_tools": [t for t in executed_tools if t],
            "error": error,
            "elapsed": elapsed,
        }
    )


def iter_records(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """逐行读 trace；坏行跳过（追加写 + 轮转下可能出现半行）。"""
    target = path or trace_path()
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item
