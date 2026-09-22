"""契约解析层：**实时优先，冻结快照兜底**。离线脚本都从这里拿契约，不要各自 import。

为什么需要
----------
两个环境天然分裂，而两边都需要工具清单 / 必填参数：

| 环境 | torch（训练要 GPU） | polars（工具注册要） |
| --- | --- | --- |
| `backend/.venv/Scripts/python.exe` | ✗ | ✓ |
| 系统 Python 3.12 | ✓ | ✗ |

`contract.tool_label_space()` / `required_params()` 都会触发 `register_builtin_tools()`
→ `app.tools.eda_tools` → `app.analysis` → `import polars`，
所以**训练环境里实时契约取不到**（`ModuleNotFoundError: No module named 'polars'`）。

解法是 `export_contract.py`（在 venv 下跑）把契约冻结成
`dataset/contract_snapshot.json`，本模块按「实时优先、快照兜底」解析。

⚠️ 漂移风险与防线
-----------------
快照会与实时契约脱节。防线三条：
1. `contract_source()` 如实返回 `live` / `snapshot`，**评测报告里必须写明**用的是哪个。
2. `tests/test_local_router_contract_snapshot.py`（在 venv 下跑）断言两者逐字段相等。
3. 实时契约可用时**永远优先实时**，快照只是「没有 polars 时的替代」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.local_router import contract as _LIVE  # noqa: E402

__all__ = ["CONTRACT", "CONTRACT_SOURCE", "SNAPSHOT_PATH", "contract_source"]

SNAPSHOT_PATH = _HERE / "dataset" / "contract_snapshot.json"


class _SnapshotContract:
    """离线契约快照，只实现离线脚本真正用到的两件事。"""

    def __init__(self, data: dict):
        self._data = data
        self._required = data.get("required_params") or {}

    def tool_label_space(self) -> list[str]:
        return list(self._data.get("tools") or [])

    def required_params(self, tool: str) -> list[str]:
        return list(self._required.get(tool) or [])


def _pick():
    try:
        _LIVE.tool_label_space()  # 触发内置工具注册；缺依赖时在这里抛
        return _LIVE, "live"
    except Exception as exc:  # noqa: BLE001
        if not SNAPSHOT_PATH.exists():
            raise RuntimeError(
                f"实时契约不可用（{type(exc).__name__}: {exc}），且找不到快照 {SNAPSHOT_PATH}；"
                "请先在 venv 下运行 scripts/router/export_contract.py"
            ) from exc
        return _SnapshotContract(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))), "snapshot"


CONTRACT, CONTRACT_SOURCE = _pick()


def contract_source() -> str:
    """`live` = 直接读工具注册表；`snapshot` = 读冻结快照。"""
    return CONTRACT_SOURCE
