"""契约探测：确认工具标签空间与参数面能正常派生。

只读，不调用任何 LLM，不启动服务。用于在写生成器之前先验证契约层自洽。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.local_router import contract as C  # noqa: E402


def main() -> None:
    names = C.tool_label_space()
    print(f"tools={len(names)}")
    for name in names:
        spec = C.tool_spec(name) or {}
        intent = C.intent_of_tool(name)
        print(
            f"  {name:26s} cat={str(spec.get('category')):9s} "
            f"intent={(intent.value if intent else 'NONE'):15s} "
            f"req={C.required_params(name)} opt={len(C.optional_params(name))}"
        )

    print()
    intents = [C.intent_of_tool(n) for n in names]
    print("intent 分布:", dict(Counter(i.value for i in intents if i is not None)))
    print("工具参数键总数:", sum(len(C.known_params(n)) for n in names))
    print("必填参数键总数:", sum(len(C.required_params(n)) for n in names))
    print("无法映射 intent 的工具:", [n for n in names if C.intent_of_tool(n) is None])
    print("无必填参数的工具:", [n for n in names if not C.required_params(n)])
    print("schema 为空的工具:", [n for n in names if not C.known_params(n)])

    print()
    inconsistent = C.category_consistency_report()
    print(f"category 声明不一致的工具: {len(inconsistent)}/{len(names)}")
    for item in inconsistent:
        print(
            f"  {item['tool']:26s} declared={item['declared_category']:10s} "
            f"expected={item['expected_category_from_name']:9s} "
            f"hint_lookup_hit={item['hint_lookup_hit']}"
        )


if __name__ == "__main__":
    main()
