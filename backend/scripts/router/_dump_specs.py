"""导出平台能力快照（工具 schema + ML 参数约束）。

只读，不调用任何 LLM。产出 `_specs.json` 供人工核对与生成器复用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.local_router import contract as C  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "_specs.json"


def main() -> None:
    tools: list[dict] = []
    for name in C.tool_label_space():
        spec = C.tool_spec(name) or {}
        intent = C.intent_of_tool(name)
        tools.append(
            {
                "name": name,
                "description": spec.get("description"),
                "declared_category": spec.get("category"),
                "intent": intent.value if intent else None,
                "permission": spec.get("permission"),
                "risk_level": spec.get("risk_level"),
                "requires_confirmation": spec.get("requires_confirmation"),
                "required": C.required_params(name),
                "optional": C.optional_params(name),
                "params": C.known_params(name),
            }
        )

    payload: dict = {"tools": tools}

    try:
        from app.ml_engine import metadata as md

        payload["param_combos"] = md.PARAM_COMBOS
        payload["ml_model_params"] = {
            model: [p.get("name") for p in (spec.get("params") or [])]
            for model, spec in md.MODEL_PARAMS.items()
        }
        payload["ml_preprocessing_params"] = [
            p.get("name") for p in (md.PREPROCESSING_PARAMS or [])
        ]
        payload["ml_training_params"] = [p.get("name") for p in (md.TRAINING_PARAMS or [])]
    except Exception as exc:  # noqa: BLE001
        payload["ml_error"] = repr(exc)

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {OUT_PATH} tools={len(tools)} bytes={OUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
