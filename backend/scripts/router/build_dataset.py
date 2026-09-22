"""从平台元数据反向生成本地 Router 的数据集骨架。

**全程不调用任何 LLM**。样本来源只有两个：平台自身的工具 schema / 参数约束，
以及人工维护的表达模板。这样做的直接好处是标签空间天然等于平台真实能力 ——
生成器不可能产出「平台不存在的工具调用」。

产出（写入 `scripts/router/dataset/`）
------------------------------------
- `samples.jsonl`          全部训练 / 评测样本（含 `split` 字段）
- `holdout.jsonl`          **冻结**测试集，独立成文件以便后续锁死
- `hard_negatives.jsonl`   易混淆工具对诊断（不是训练样本，用于补数据）
- `boundary.jsonl`         确定性校验器的回归测试集
- `combo_violations.jsonl` ML 参数约束知识条目
- `taxonomy.json`          标签空间 + 参数面（推理与评测的基础）
- `manifest.json`          统计 / 覆盖率 / 缺口 / 可复现指纹

三种校验结局，对应三条出路
--------------------------
1. `ok`      —— 入集，作为正常路由样本。
2. `missing` —— 模板没覆盖某个必填槽位。**不丢弃**：它恰好等价于
   「用户这句话没说清关键信息」，正是"反问用户"的天然样本，
   转为 `slot_missing` 类别，并把上下文绑定清空（无上下文时槽位才算真缺）。
3. `invalid` —— 结构性错误（参数名不认识、类型不符、intent 不自洽）。
   这类必须暴露进 `manifest.rejected`，因为它说明**模板或槽位表写错了**。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.local_router import contract as C  # noqa: E402
from scripts.router import utterance_templates as T  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "dataset"
_SLOT_RE = re.compile(r"\{([a-z_0-9]+)\}")
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)(?::([a-z_]+))?\}")

HOLDOUT_BUCKET = 15
EVAL_BUCKET = 25


# ---------------------------------------------------------------------------
# 槽位定义
# ---------------------------------------------------------------------------

DATASET_ID_DISPLAY = "{}号数据集"

SLOT_SPEC: dict[str, dict[str, Any]] = {
    "dataset_id": {"param": "dataset_id", "values": T.DATASET_IDS,
                   "display": DATASET_ID_DISPLAY},
    "left_dataset_id": {"param": "left_dataset_id", "values": [1, 3, 7],
                        "display": DATASET_ID_DISPLAY},
    "right_dataset_id": {"param": "right_dataset_id", "values": [2, 5, 9],
                         "display": DATASET_ID_DISPLAY},
    "run_id": {"param": "run_id", "values": T.RUN_IDS, "display": "第{}次训练"},
    "workflow_id": {"param": "workflow_id", "values": T.WORKFLOW_IDS,
                    "display": "{}号工作流"},
    "column": {"param": "column", "values": T.NUMERIC_COLUMNS + T.CATEGORY_COLUMNS},
    "category_column": {"param": "group_by", "values": T.CATEGORY_COLUMNS},
    "chart": {"param": "chart", "values": list(T.CHART_WORDS), "encode": T.CHART_WORDS},
    "model": {"param": "model", "values": list(T.MODEL_WORDS), "encode": T.MODEL_WORDS},
    "mltopic": {"param": "topic", "values": list(T.TOPIC_WORDS), "encode": T.TOPIC_WORDS},
}


def display_of(slot: str, value: Any) -> str:
    """槽位在 utterance 里的显示形式。

    参数值和人话不是一回事：`dataset_id=3` 在句子里要读作「3号数据集」，
    否则会渲染出「3 哪些列有离群点」这种机器腔。参数真值另行回填。
    """
    pattern = SLOT_SPEC.get(slot, {}).get("display")
    return pattern.format(value) if pattern else str(value)

# (tool, slot) -> 参数名；显式 None 表示「只出现在句子里，不产出参数键」。
# data.filter 的列名已经表达在 conditions 内部，不能再单独成为一个键。
SLOT_PARAM_OVERRIDE: dict[tuple[str, str], str | None] = {
    ("data.filter", "column"): None,
}


def resolve_slot_param(tool: str, slot: str) -> str | None:
    if (tool, slot) in SLOT_PARAM_OVERRIDE:
        return SLOT_PARAM_OVERRIDE[(tool, slot)]
    spec = SLOT_SPEC.get(slot)
    return spec["param"] if spec else None


# 结构化必填参数的 Gold 取值。`{slot:xxx}` 会替换成模板里实际填过的槽位值，
# 保证 utterance 里说的列与 gold 里的列一致（否则是自相矛盾的标签）。
_STRUCTURED: dict[tuple[str, str], Any] = {
    ("data.filter", "conditions"): [
        {"column": "{slot:column}", "op": ">", "value": 100}
    ],
    ("data.aggregate", "aggregations"): [{"column": "{numeric}", "func": "mean"}],
    ("data.transform", "name"): "new_column",
    ("data.transform", "expression"): {
        "type": "math",
        "op": "add",
        "left": {"type": "column", "name": "{numeric}"},
        "right": {"type": "value", "value": 1},
    },
    ("workflow.create", "name"): "自动分析流程",
    ("workflow.create", "nodes"): [
        {"id": "n1", "type": "data.load", "config": {"dataset_id": T.DATASET_IDS[0]}},
        {"id": "n2", "type": "data.quality_check", "config": {}},
    ],
    ("workflow.create", "edges"): [{"source": "n1", "target": "n2"}],
    ("workflow.build_and_run", "name"): "自动分析流程",
    ("workflow.build_and_run", "nodes"): [
        {"id": "n1", "type": "data.load", "config": {"dataset_id": T.DATASET_IDS[0]}},
        {"id": "n2", "type": "data.statistics", "config": {}},
    ],
    ("workflow.build_and_run", "edges"): [{"source": "n1", "target": "n2"}],
}


def _substitute(node: Any, context: dict[str, Any]) -> Any:
    if isinstance(node, str):
        match = _PLACEHOLDER_RE.fullmatch(node)
        if match:
            kind, key = match.group(1), match.group(2)
            if kind == "slot":
                return context.get(key) or T.NUMERIC_COLUMNS[0]
            if kind == "numeric":
                return T.NUMERIC_COLUMNS[0]
            if kind == "category":
                return T.CATEGORY_COLUMNS[0]
        return node
    if isinstance(node, dict):
        return {k: _substitute(v, context) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, context) for v in node]
    return node


def structured_value(tool: str, param: str, context: dict[str, Any]) -> Any:
    """结构化必填参数的 Gold 取值（极简版）。

    这些槽位一句话说不清（`conditions` / `nodes` / `expression` …），
    人工模板覆盖不到，但它们都是 `required`。不补的话整批样本会退化成
    「缺槽位」。取值刻意保持最小合法，参数结构的复杂度留给二级模型。
    """
    template = _STRUCTURED.get((tool, param))
    if template is None:
        return None
    return _substitute(template, context)


def build_params(
    tool: str,
    *,
    filled: dict[str, str] | None = None,
    direct: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`direct` 优先（上下文补齐的参数），其次模板槽位，最后结构化必填。"""
    params: dict[str, Any] = dict(direct or {})

    # `direct` 与模板槽位是**同一个参数面的两种来源**，归一化必须一致。
    # 原先只有下面 `filled` 分支做标量→数组转换，`direct` 不做 ⇒ 传
    # `group_by="city"` 会被 validate_decision 判 invalid（实测踩到）。
    for key, value in list(params.items()):
        if C.param_type(tool, key) == "array" and not isinstance(value, list):
            params[key] = [value]

    for slot, raw in (filled or {}).items():
        param = resolve_slot_param(tool, slot)
        if param is None:
            continue
        value: Any = SLOT_SPEC[slot].get("encode", {}).get(raw, raw) if slot in SLOT_SPEC else raw
        if C.param_type(tool, param) == "array" and not isinstance(value, list):
            value = [value]
        params[param] = value

    context = {"column": params.get("column")}
    for name in C.required_params(tool):
        if name in params:
            continue
        fallback = structured_value(tool, name, context)
        if fallback is not None:
            params[name] = fallback
    return params


# ---------------------------------------------------------------------------
# 样本骨架
# ---------------------------------------------------------------------------


def sample_id(category: str, payload: str) -> str:
    return f"{category}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def split_of(sid: str) -> str:
    bucket = int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < HOLDOUT_BUCKET:
        return "holdout"
    if bucket < EVAL_BUCKET:
        return "eval"
    return "train"


def make_decision(
    *,
    intent: str,
    tool: str | None,
    params: dict[str, Any] | None = None,
    missing: list[str] | None = None,
    escalate: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "tool": tool,
        "params": params or {},
        "missing": missing or [],
        "escalate": escalate,
        "escalate_reason": reason,
        "confidence": 1.0,
    }


def classify(target: dict[str, Any]) -> tuple[str, list[str]]:
    """返回 (status, detail)；status ∈ {ok, missing, invalid}。"""
    try:
        decision = C.RouterDecision(**target)
    except Exception as exc:  # noqa: BLE001
        return "invalid", [f"构造失败：{exc}"]
    outcome = C.validate_decision(decision)
    if outcome.problems:
        return "invalid", outcome.problems
    if outcome.missing_required:
        return "missing", outcome.missing_required
    return "ok", []


def make_sample(
    *,
    category: str,
    utterance: str,
    target: dict[str, Any],
    bound_dataset_id: int | None = None,
    recent_tools: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    params = target.get("params") or {}
    columns = [params["column"]] if isinstance(params.get("column"), str) else []
    sid = sample_id(category, f"{utterance}|{suffix}")
    return {
        "id": sid,
        "category": category,
        "split": split_of(sid),
        "request": {
            "utterance": utterance,
            "bound_dataset_id": bound_dataset_id,
            "available_columns": columns,
            "recent_tools": recent_tools or [],
        },
        "target": target,
        "meta": meta or {},
    }


# ---------------------------------------------------------------------------
# 各类样本
# ---------------------------------------------------------------------------


def gen_routing(rejected: list[dict[str, str]]) -> list[dict[str, Any]]:
    """单工具路由样本：模板 × 槽位取值笛卡尔积。

    校验结果决定去向：`ok` 入 routing，`missing` 转 slot_missing（并把上下文
    绑定清空，因为无上下文时槽位才算真的缺），`invalid` 记入 rejected。
    """
    samples: list[dict[str, Any]] = []
    for tool in C.tool_label_space():
        templates = T.ROUTING_TEMPLATES.get(tool)
        if not templates:
            rejected.append({"tool": tool, "reason": "缺少人工模板", "kind": "routing"})
            continue
        intent = C.intent_of_tool(tool)
        if intent is None:
            rejected.append({"tool": tool, "reason": "无法派生 intent", "kind": "routing"})
            continue

        for template in templates:
            slots = _SLOT_RE.findall(template)
            unknown = [s for s in slots if s not in SLOT_SPEC]
            if unknown:
                rejected.append({"tool": tool, "reason": f"未知槽位 {unknown}", "kind": "routing"})
                continue

            choices = [SLOT_SPEC[s]["values"][:2] for s in slots]
            for combo in _product(choices):
                filled = dict(zip(slots, combo, strict=True))
                # utterance 用显示形式（「3号数据集」），params 用真值（3）
                utterance = template.format(
                    **{k: display_of(k, v) for k, v in filled.items()}
                )
                params = build_params(tool, filled=filled)
                target = make_decision(intent=intent.value, tool=tool, params=params)
                status, detail = classify(target)

                if status == "invalid":
                    rejected.append(
                        {"tool": tool, "reason": "；".join(detail[:2]), "kind": "routing",
                         "utterance": utterance}
                    )
                    continue

                if status == "missing":
                    target["missing"] = detail
                    samples.append(
                        make_sample(
                            category="slot_missing",
                            utterance=utterance,
                            target=target,
                            bound_dataset_id=None,
                            # ⚠️ 必须带上 template：slot_missing 与它的 routing 兄弟是**同一
                            # 模板的两种情形**（只是必填槽位漏说）。若这里丢掉 template，
                            # 分组评估会把同模板的两兄弟拆到 train/test 两侧 ⇒ 模型被迫在
                            # 没见过「有 dataset_id」的样子时判断「没有 dataset_id」，call/ask
                            # 会双向混淆（实测 call→ask×13、ask→call×12 正是此因）。
                            meta={"tool": tool, "missing": detail,
                                  "derived_from": "routing", "template": template},
                            suffix=f"missing|{tool}",
                        )
                    )
                    continue

                samples.append(
                    make_sample(
                        category="routing",
                        utterance=utterance,
                        target=target,
                        bound_dataset_id=params.get("dataset_id"),
                        meta={"tool": tool, "template": template, "filled": filled},
                        suffix=tool,
                    )
                )
    return samples


def gen_slot_missing_curated(rejected: list[dict[str, str]]) -> list[dict[str, Any]]:
    """人工设计的「信息不全」场景 —— 期望动作是反问，不是升级。"""
    samples: list[dict[str, Any]] = []
    for case in T.SLOT_MISSING_TEMPLATES:
        tool = case["tool"]
        intent = C.intent_of_tool(tool)
        if intent is None:
            rejected.append({"reason": f"未知工具 {tool}", "kind": "slot_missing"})
            continue
        target = make_decision(intent=intent.value, tool=tool, params={})
        status, detail = classify(target)
        if status != "missing":
            rejected.append(
                {"tool": tool, "reason": f"期望缺槽位但得到 {status}", "kind": "slot_missing",
                 "utterance": case["utterance"]}
            )
            continue
        target["missing"] = detail
        samples.append(
            make_sample(
                category="slot_missing",
                utterance=case["utterance"],
                target=target,
                bound_dataset_id=None,
                meta={"tool": tool, "missing": detail, "derived_from": "curated"},
                suffix=f"curated|{tool}",
            )
        )
    return samples


def gen_chat() -> list[dict[str, Any]]:
    """闲聊样本 —— 挡住它们等于省掉一次远程调用。"""
    return [
        make_sample(
            category="chat",
            utterance=utterance,
            target=make_decision(intent=C.Intent.CHAT.value, tool=None),
            meta={},
            suffix="chat",
        )
        for utterance in T.CHAT_UTTERANCES
    ]


def gen_escalation() -> list[dict[str, Any]]:
    """确实超出平台能力的请求 —— 应当升级。"""
    samples: list[dict[str, Any]] = []
    for case in T.ESCALATION_CASES:
        utterance = (
            case["utterance"].format(dataset_id=T.DATASET_IDS[0])
            if "{dataset_id}" in case["utterance"]
            else case["utterance"]
        )
        samples.append(
            make_sample(
                category="escalate",
                utterance=utterance,
                target=make_decision(
                    intent=C.Intent.CHAT.value,
                    tool=None,
                    escalate=True,
                    reason=case["reason"],
                ),
                bound_dataset_id=T.DATASET_IDS[0],
                meta={"escalate_reason": case["reason"]},
                suffix=case["reason"],
            )
        )
    return samples


def gen_implicit(rejected: list[dict[str, str]]) -> list[dict[str, Any]]:
    """说得含糊但平台能处理 —— **不该升级**。

    没有这批数据，模型会把一切「没说清楚」都判成升级，反而制造更多远程调用。
    """
    samples: list[dict[str, Any]] = []
    for case in T.RESOLVABLE_IMPLICIT:
        tool = case["tool"]
        intent = C.intent_of_tool(tool)
        if intent is None:
            rejected.append({"reason": f"未知工具 {tool}", "kind": "implicit"})
            continue
        params = build_params(tool, direct=case.get("slots") or {})
        target = make_decision(intent=intent.value, tool=tool, params=params)
        status, detail = classify(target)
        if status != "ok":
            rejected.append(
                {"tool": tool, "reason": f"{status}：{'；'.join(detail[:2])}",
                 "kind": "implicit", "utterance": case["utterance"]}
            )
            continue
        samples.append(
            make_sample(
                category="implicit",
                utterance=case["utterance"],
                target=target,
                bound_dataset_id=T.DATASET_IDS[0],
                meta={"tool": tool},
                suffix=tool,
            )
        )
    return samples


def gen_follow_up(rejected: list[dict[str, str]]) -> list[dict[str, Any]]:
    """多轮指代：宾语被省略，要靠 recent_tools 消解。"""
    samples: list[dict[str, Any]] = []
    for case in T.FOLLOW_UP_TEMPLATES:
        tool = case["tool"]
        intent = C.intent_of_tool(tool)
        if intent is None:
            rejected.append({"reason": f"未知工具 {tool}", "kind": "follow_up"})
            continue
        params = build_params(tool, direct=case.get("slots") or {})
        target = make_decision(intent=intent.value, tool=tool, params=params)
        status, detail = classify(target)
        if status != "ok":
            rejected.append(
                {"tool": tool, "reason": f"{status}：{'；'.join(detail[:2])}",
                 "kind": "follow_up", "utterance": case["utterance"]}
            )
            continue
        samples.append(
            make_sample(
                category="follow_up",
                utterance=case["utterance"],
                target=target,
                bound_dataset_id=T.DATASET_IDS[0],
                recent_tools=[case["after"]],
                meta={"tool": tool, "resolved_from": case["after"]},
                suffix=f"{case['after']}|{tool}",
            )
        )
    return samples


# ---------------------------------------------------------------------------
# 诊断类产出
# ---------------------------------------------------------------------------


def build_hard_negatives() -> list[dict[str, Any]]:
    """同能力域内易混淆的工具对 —— 补数据时应优先覆盖这些方向。"""
    rows: list[dict[str, Any]] = []
    by_intent: dict[str, list[str]] = defaultdict(list)
    for tool in C.tool_label_space():
        intent = C.intent_of_tool(tool)
        if intent is not None:
            by_intent[intent.value].append(tool)

    for intent, tools in sorted(by_intent.items()):
        for i, left in enumerate(tools):
            for right in tools[i + 1 :]:
                d_left = str((C.tool_spec(left) or {}).get("description", ""))
                d_right = str((C.tool_spec(right) or {}).get("description", ""))
                rows.append(
                    {
                        "intent": intent,
                        "tools": [left, right],
                        "overlap": round(_jaccard(d_left, d_right), 3),
                        "left_desc": d_left,
                        "right_desc": d_right,
                    }
                )
    rows.sort(key=lambda r: -r["overlap"])
    return rows


def build_taxonomy() -> dict[str, Any]:
    tools = []
    for name in C.tool_label_space():
        spec = C.tool_spec(name) or {}
        intent = C.intent_of_tool(name)
        tools.append(
            {
                "name": name,
                "description": spec.get("description"),
                "intent": intent.value if intent else None,
                "declared_category": spec.get("category"),
                "risk_level": spec.get("risk_level"),
                "requires_confirmation": spec.get("requires_confirmation"),
                "required": C.required_params(name),
                "optional": C.optional_params(name),
                "param_types": {k: v.get("type") for k, v in C.known_params(name).items()},
            }
        )
    return {
        "intents": [i.value for i in C.Intent],
        "escalation_reasons": [r.value for r in C.EscalationReason],
        "tool_count": len(tools),
        "tools": tools,
        "category_inconsistency": C.category_consistency_report(),
        "decision_schema": C.decision_json_schema(),
    }


def build_boundary() -> list[dict[str, Any]]:
    """确定性校验器的回归测试集：每条给一个决策 + 期望结论。"""
    cases: list[tuple[str, dict[str, Any], bool, str]] = [
        (
            "合法：带必填参数的工具调用",
            make_decision(intent="dataset", tool="dataset.quality", params={"dataset_id": 1}),
            True,
            "基线：应当放行",
        ),
        (
            "非法：工具不存在",
            make_decision(intent="dataset", tool="dataset.nonexistent", params={}),
            False,
            "不在注册表内",
        ),
        (
            "非法：intent 与 tool 能力域不一致",
            make_decision(intent="ml", tool="dataset.quality", params={"dataset_id": 1}),
            False,
            "intent/tool 不自洽",
        ),
        (
            "非法：参数类型错误（整数给字符串）",
            make_decision(intent="dataset", tool="dataset.quality", params={"dataset_id": "abc"}),
            False,
            "类型粗校验拦截",
        ),
        (
            "非法：schema 未声明的参数",
            make_decision(
                intent="dataset",
                tool="dataset.quality",
                params={"dataset_id": 1, "unknown_key": 1},
            ),
            False,
            "未知参数键",
        ),
        (
            "非法：升级但未给出原因",
            {
                "intent": "chat", "tool": None, "params": {}, "missing": [],
                "escalate": True, "escalate_reason": None, "confidence": 1.0,
            },
            False,
            "escalate 必须带 reason",
        ),
        ("合法：闲聊无需工具", make_decision(intent="chat", tool=None), True,
         "chat 允许 tool 为空"),
        (
            "合法：主动升级",
            make_decision(intent="chat", tool=None, escalate=True, reason="out_of_scope"),
            True,
            "带 reason 的升级应放行",
        ),
    ]
    rows = []
    for label, decision, expect_ok, note in cases:
        status, detail = classify(decision)
        rows.append(
            {
                "label": label,
                "decision": decision,
                "expect_well_formed": expect_ok,
                "actual_status": status,
                "actual_detail": detail,
                "passed": (status != "invalid") == expect_ok,
                "note": note,
            }
        )
    return rows


def build_combo_violations() -> list[dict[str, Any]]:
    """ML 参数约束知识条目（来自 `ml_engine.metadata.PARAM_COMBOS`）。"""
    try:
        from app.ml_engine import metadata as md
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "model": combo.get("model"),
            "when": combo.get("when"),
            "require": combo.get("require"),
            "severity": combo.get("severity", "error"),
            "message": combo.get("message"),
        }
        for combo in md.PARAM_COMBOS
    ]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _product(choices: list[list[Any]]) -> list[tuple[Any, ...]]:
    if not choices:
        return [()]
    out: list[tuple[Any, ...]] = [()]
    for pool in choices:
        out = [existing + (item,) for existing in out for item in pool]
    return out


def _jaccard(left: str, right: str) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def dedupe(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一 (类别, utterance, 工具) 只保留一条。

    类别必须参与去重键：同一句 utterance 在「参数齐全 → 直接执行」与
    「缺槽位 → 反问用户」两种情况下都是合法样本，不能互相覆盖。
    """
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[dict[str, Any]] = []
    for item in samples:
        key = (item["category"], item["request"]["utterance"], item["target"]["tool"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rejected: list[dict[str, str]] = []

    samples: list[dict[str, Any]] = []
    samples += gen_routing(rejected)
    samples += gen_slot_missing_curated(rejected)
    samples += gen_chat()
    samples += gen_escalation()
    samples += gen_implicit(rejected)
    samples += gen_follow_up(rejected)

    samples = dedupe(samples)
    samples.sort(key=lambda x: (x["category"], x["id"]))

    holdout = [s for s in samples if s["split"] == "holdout"]
    hard_negatives = build_hard_negatives()
    boundary = build_boundary()
    combos = build_combo_violations()

    files = {
        "samples.jsonl": write_jsonl(OUT_DIR / "samples.jsonl", samples),
        "holdout.jsonl": write_jsonl(OUT_DIR / "holdout.jsonl", holdout),
        "hard_negatives.jsonl": write_jsonl(OUT_DIR / "hard_negatives.jsonl", hard_negatives),
        "boundary.jsonl": write_jsonl(OUT_DIR / "boundary.jsonl", boundary),
        "combo_violations.jsonl": write_jsonl(OUT_DIR / "combo_violations.jsonl", combos),
    }
    (OUT_DIR / "taxonomy.json").write_text(
        json.dumps(build_taxonomy(), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    tools_covered = {s["target"]["tool"] for s in samples if s["target"]["tool"]}
    seen_param_keys = {
        (s["target"]["tool"], key)
        for s in samples
        if s["target"]["tool"]
        for key in (s["target"]["params"] or {})
    }
    all_required = {(t, p) for t in C.tool_label_space() for p in C.required_params(t)}
    all_optional = {(t, p) for t in C.tool_label_space() for p in C.optional_params(t)}
    manifest = {
        "sample_total": len(samples),
        "files": files,
        "by_category": dict(sorted(Counter(s["category"] for s in samples).items())),
        "by_split": dict(sorted(Counter(s["split"] for s in samples).items())),
        "holdout_ids_sha1": hashlib.sha1(
            "|".join(sorted(s["id"] for s in holdout)).encode("utf-8")
        ).hexdigest(),
        "coverage": {
            "tools_total": len(C.tool_label_space()),
            "tools_with_samples": len(tools_covered),
            "tools_without_samples": sorted(set(C.tool_label_space()) - tools_covered),
            "tools_without_templates": sorted(
                t for t in C.tool_label_space() if not T.ROUTING_TEMPLATES.get(t)
            ),
            "templates_total": sum(len(v) for v in T.ROUTING_TEMPLATES.values()),
            "required_params_total": sum(
                len(C.required_params(t)) for t in C.tool_label_space()
            ),
            "boundary_passed": sum(1 for r in boundary if r["passed"]),
            "boundary_total": len(boundary),
            "required_param_coverage": {
                "total": len(all_required),
                "covered": len(all_required & seen_param_keys),
                "missing": sorted(f"{t}.{p}" for t, p in all_required - seen_param_keys),
            },
            "optional_param_coverage": {
                "total": len(all_optional),
                "covered": len(all_optional & seen_param_keys),
                "note": "可选参数未覆盖属预期：本阶段只教工具选择与必填槽位",
            },
        },
        "escalation": {
            "positive": sum(1 for s in samples if s["target"]["escalate"]),
            "negative": sum(1 for s in samples if not s["target"]["escalate"]),
            "reasons_covered": sorted(
                {s["target"]["escalate_reason"] for s in samples if s["target"]["escalate_reason"]}
            ),
        },
        "known_gaps": [
            "多轮上下文样本仅 8 条 —— 真实会话接入后需大幅扩充",
            "多意图（一句话里含两个任务）尚无样本",
            "可选参数（bins / top_n / test_size / params 等）未覆盖：本阶段只教必填槽位",
            "hard negative 目前只有混淆对诊断，尚未生成对应的判别样本",
            "真实会话轨迹（AgentStore）尚未接入，当前全部为模板合成",
            "参数越界值（超出声明 range）样本未生成",
        ],
        "rejected": rejected,
        "rejected_total": len(rejected),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("sample_total", "by_category", "by_split", "coverage",
                          "escalation", "rejected_total")
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
