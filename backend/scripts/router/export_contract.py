"""把平台契约**冻结**成离线快照，供训练环境使用（在 venv 下运行）。

为什么需要这个文件
------------------
两个环境天然分裂：

| 环境 | 有 torch | 有 polars |
| --- | --- | --- |
| `backend/.venv/Scripts/python.exe` | ✗ | ✓ |
| 系统 Python 3.12（训练用 GPU torch） | ✓ | ✗ |

而 `contract.tool_label_space()` 会触发 `register_builtin_tools()` → `app.tools.eda_tools`
→ `app.analysis` → `import polars` ⇒ **训练环境里拿不到工具清单与必填参数**。

三个可选方案，选了第三个：
1. 往系统 python 装 polars —— 污染用户环境，且可能与已有版本冲突。✗
2. 在 venv 里装 torch —— 要下 ~2.5GB，只为跑一个 40 万参数的模型。✗
3. **把契约导成 JSON 快照** —— 训练侧只读文件，零后端依赖；且快照本身可进版本库，
   让 L1 训练在别的机器上也能复现（契合「可移植」）。✓

⚠️ 快照与实时契约存在**漂移风险**。防线：
- `dataset/contract_snapshot.json` 带 `source_sha1`（contract.py 内容摘要）。
- `l1_data.contract_source()` 报告实际用的是快照还是实时注册表。
- `tests/test_local_router_contract_snapshot.py` 断言两者**逐字段相等**（在 venv 下跑）。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.local_router import contract as C  # noqa: E402

OUT = _HERE / "dataset" / "contract_snapshot.json"


def main() -> int:
    tools = C.tool_label_space()
    snapshot = {
        "tools": tools,
        "intent_of_tool": {
            t: (C.intent_of_tool(t).value if C.intent_of_tool(t) is not None else None)
            for t in tools
        },
        "required_params": {t: C.required_params(t) for t in tools},
        "optional_params": {t: C.optional_params(t) for t in tools},
        "label_space": ["chat"] + ["call::" + t for t in tools],
        "category_inconsistencies": C.category_consistency_report(),
        "source_sha1": hashlib.sha1(
            (Path(C.__file__)).read_bytes()
        ).hexdigest(),
        "generated_at_epoch": int(time.time()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

    n_req = sum(len(v) for v in snapshot["required_params"].values())
    print(f"工具 {len(tools)} 个 / 必填参数 {n_req} 个 / "
          f"标签空间 {len(snapshot['label_space'])} 类")
    print(f"category 不一致 {len(snapshot['category_inconsistencies'])} 个")
    print(f"已写出 {OUT.relative_to(_BACKEND_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
