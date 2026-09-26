"""确定性层（Layer 1）：零模型、零 LLM 的流程判定与动作模板。

本模块是重构后唯一保留的「确定性决策」落点，收敛了旧体系四处分散的机制：

* 取消/停止命令（旧 ``decision/rule.py``）；
* 必填槽位检查（旧 ``decision/rule.py`` + ``local_router.contract``）；
* 多步句首分句（旧 ``task_spec_builder._split_first_clause``）；
* 客观数据信号 → 只读工具白名单（旧 ``decision/signal.py``）；
* 九条确定性 playbook（旧 ``planner.py::_rule_plan``）。

禁止在这里增加任何「用户说了什么词 → 选哪个工具」的新语义表；能力域关键词
的唯一真源仍是 :mod:`app.agent.intent`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.intent import (
    COMPREHENSIVE_KEYWORDS,
    QUALITY_KEYWORDS,
    Intent,
    hits,
    model_hint,
    wants_merge,
    wants_modeling,
)
from app.agent.state import PendingAction, TaskState

#: 明确的系统命令（取消 / 停止 / 退出）。
CANCEL_UTTERANCES = frozenset({"取消", "停止", "退出", "不做了", "停下"})

#: 多步编排的分句连词；只切「后置分句开头」，句首「先」单独剥除。
SEQUENCE_SPLIT_MARKERS: tuple[str, ...] = ("再", "然后", "接着", "之后", "最后", "同时")

#: 客观数据信号 → 数据集级只读/处置工具（平台能力边界，非自然语言语义）。
#: 不收录需要 column 参数的工具（信号层没有列名，硬排必然缺参失败）。
SIGNAL_TO_TOOL: dict[str, str] = {
    "outliers_detected": "eda.outlier",
    "high_skew": "eda.outlier",
    "high_missing": "data.clean",
    "strong_correlation": "eda.correlation",
}

#: 信号触发 data.clean 时的安全默认：与旧规则链一致（看不到列类型，
#: mean/median 在字符串列必失败），用户可改要求为填充（state.apply_constraint_change）。
_SIGNAL_CLEAN_ARGS = {"missing": {"strategy": "drop"}}


@dataclass
class PlaybookSelection:
    """一次确定性选择的结果（name 用于 trace/source 标注）。"""

    name: str
    actions: list[PendingAction] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------- 纯函数

def is_cancel_command(text: str) -> bool:
    raw = (text or "").strip()
    return raw in CANCEL_UTTERANCES or (len(raw) <= 3 and raw in {"取消", "停止", "退出"})


def split_first_clause(user_request: str) -> str:
    """「先 X 再 Y」→ 首分句 X。切不出则返回原句（保守切分）。"""
    text = (user_request or "").strip()
    if not text:
        return ""
    cut = len(text)
    for marker in SEQUENCE_SPLIT_MARKERS:
        idx = text.find(marker)
        if 0 < idx < cut:
            cut = idx
    if cut < len(text):
        text = text[:cut]
    if text.startswith("先"):
        text = text[1:].strip()
    text = text.rstrip("，,。；;、：: ")
    return text or (user_request or "").strip()


def missing_required_slots(tool: str, dataset_ids: list[int]) -> list[str]:
    """规则可无条件判定的缺失槽位（目前只有 dataset_id）。空列表 = 齐备。"""
    if not tool:
        return []
    from app.local_router.contract import required_params  # 懒加载：contract 可能带 schema 资源

    if "dataset_id" in required_params(tool) and not dataset_ids:
        return ["dataset_id"]
    return []


def action_for_signal(
    signals: list[str],
    *,
    dataset_id: int | None,
    last_tool: str | None = None,
    done_tools: set[str] | None = None,
    pending_tools: set[str] | None = None,
    missing_strategy: str | None = None,
) -> PendingAction | None:
    """按白名单把最新客观信号映射为下一步动作；已做/在队/刚做过则不表态。

    ``missing_strategy``：用户已在任务约束里改过缺失值处置（如「不要删列，改成
    均值填充」）时，信号触发的 data.clean 必须沿用同一策略。
    """
    done_tools = done_tools or set()
    pending_tools = pending_tools or set()
    for sig in signals or []:
        tool = SIGNAL_TO_TOOL.get(str(sig))
        if not tool or tool == last_tool or tool in done_tools or tool in pending_tools:
            continue
        if dataset_id is None:
            continue
        args: dict[str, Any] = {"dataset_id": dataset_id}
        if tool == "data.clean":
            args.update(_SIGNAL_CLEAN_ARGS)
            if missing_strategy:
                args["missing"] = {"strategy": missing_strategy}
        return PendingAction(tool=tool, arguments=args, source="signal",
                             rationale=f"数据信号 {sig} 提示深入 {tool}")
    return None


#: 缺失值处置关键词 → data.clean.missing.strategy（与 TaskState 填充词表同口径）。
_MISSING_STRATEGY_WORDS: tuple[tuple[str, str], ...] = (
    ("中位数", "median"), ("median", "median"),
    ("众数", "mode"), ("mode", "mode"),
    ("固定值", "constant"), ("常数", "constant"),
    ("均值", "mean"), ("平均值", "mean"), ("平均", "mean"),
    ("填充", "mean"), ("补上", "mean"), ("mean", "mean"),
)


def missing_fill_strategy(text: str, state: TaskState | None) -> str | None:
    """从当前请求与任务约束中识别「缺失值用填充而非删除」的策略；无填充意图返回 None。"""
    blob = str(text or "")
    if state is not None:
        blob = blob + " " + " ".join(state.constraints[-3:])
    if not any(w in blob for w in ("填充", "补上", "均值", "平均", "中位数", "众数", "固定值", "mean", "median", "mode")):
        return None
    for word, strategy in _MISSING_STRATEGY_WORDS:
        if word in blob:
            return strategy
    return "mean"


def _missing_strategy(text: str, state: TaskState | None) -> str:
    """缺失值处置策略：用户/约束指定了填充就用填充，否则沿用旧安全默认 drop。"""
    return missing_fill_strategy(text, state) or "drop"


# ---------------------------------------------------------------- playbook 选择

def select_playbook(user_request: str, state: TaskState | None, dataset_ids: list[int], *, base_index: int = 0) -> PlaybookSelection:
    """按与旧 ``_rule_plan`` 一致的优先级选择确定性动作链，并按 state 去重/修引用。

    ``base_index``：本批动作在当前 Run 里的起始绝对步号（0 基），多轮续跑时
    ``{{stepN.field}}`` 引用必须按它偏移（Run 内 step_index 是跨批次累计的）。
    """
    text = (user_request or "").lower()
    if not dataset_ids:
        return PlaybookSelection("need_dataset", [], reason="未绑定数据集，需要先向用户确认")

    intent_hits = hits(text)
    matched = lambda intent: intent_hits.get(intent) or []  # noqa: E731

    wants_train = wants_modeling(text)
    wants_quality = any(k in text for k in QUALITY_KEYWORDS)
    wants_corr = "相关" in matched(Intent.EDA)
    wants_workflow = bool(intent_hits.get(Intent.WORKFLOW))
    do_merge = wants_merge(text)
    wants_report = bool(intent_hits.get(Intent.REPORT))
    wants_comprehensive = any(k in text for k in COMPREHENSIVE_KEYWORDS)
    wants_eda = bool(intent_hits.get(Intent.EDA))
    wants_evaluate = any(k in matched(Intent.ML) for k in ("评估", "效果", "指标", "准确率"))
    transform_hits = matched(Intent.DATA_TRANSFORM)
    wants_dedupe = any(k in transform_hits for k in ("去重", "重复"))
    wants_missing = any(k in transform_hits for k in ("缺失", "填充")) or any(
        k in text for k in ("空值", "missing", "null")
    )
    wants_column_op = any(k in transform_hits for k in ("筛选", "聚合", "排序"))

    ds_id = dataset_ids[0] if len(dataset_ids) == 1 else None

    def args(extra: dict[str, Any] | None = None, *, dataset: int | None = ds_id) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if dataset is not None:
            out["dataset_id"] = dataset
        if extra:
            out.update(extra)
        return out

    if do_merge and len(dataset_ids) >= 2:
        chain = [
            PendingAction(tool="dataset.inspect", arguments={"dataset_id": dataset_ids[0]}, source="playbook"),
            PendingAction(
                tool="data.merge",
                arguments={
                    "left_dataset_id": dataset_ids[0],
                    "right_dataset_id": dataset_ids[1],
                    "join_type": "left" if any(k in text for k in ("左", "保留左", "left")) else "inner",
                },
                source="playbook",
            ),
        ]
        return PlaybookSelection("merge", _filter_done(chain, state))

    if wants_workflow:
        chain = [
            PendingAction(tool="dataset.inspect", arguments=args(), source="playbook"),
            PendingAction(
                tool="workflow.build_and_run",
                arguments={
                    "name": (user_request or "Agent 自动编排")[:24],
                    "nodes": [
                        {"id": "n1", "type": "dataset.read", "config": {"dataset_id": ds_id}},
                        {"id": "n2", "type": "data.quality_check", "config": {}},
                        {"id": "n3", "type": "data.statistics", "config": {}},
                        {"id": "n4", "type": "report.summary", "config": {}},
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2"}, {"source": "n1", "target": "n3"},
                        {"source": "n2", "target": "n4"}, {"source": "n3", "target": "n4"},
                    ],
                },
                source="playbook",
            ),
        ]
        return PlaybookSelection("workflow", _filter_done(chain, state))

    if wants_train:
        return PlaybookSelection("modeling", _modeling_chain(user_request, args, state, wants_evaluate, base_index=base_index))

    if wants_dedupe or wants_missing or wants_column_op:
        chain = [PendingAction(tool="dataset.inspect", arguments=args(), source="playbook")]
        if wants_dedupe or wants_missing:
            clean_args: dict[str, Any] = {}
            if wants_missing:
                clean_args["missing"] = {"strategy": _missing_strategy(user_request, state)}
            if wants_dedupe:
                clean_args["deduplicate"] = {"keep": "first"}
            chain.append(PendingAction(tool="data.clean", arguments=args(clean_args), source="playbook"))
        else:
            # 筛选/聚合/排序必须先有真实列名（规划期禁止臆造）。
            chain += [
                PendingAction(tool="dataset.schema", arguments=args(), source="playbook"),
                PendingAction(tool="dataset.profile", arguments=args(), source="playbook"),
            ]
        return PlaybookSelection("transform", _filter_done(chain, state))

    if wants_comprehensive:
        chain = [
            PendingAction(tool="dataset.inspect", arguments=args(), source="playbook"),
            PendingAction(tool="dataset.schema", arguments=args(), source="playbook"),
            PendingAction(tool="dataset.quality", arguments=args(), source="playbook"),
            PendingAction(tool="dataset.profile", arguments=args(), source="playbook"),
        ]
        if wants_corr:
            chain.append(PendingAction(tool="eda.correlation", arguments=args(), source="playbook"))
        selection = PlaybookSelection("comprehensive", _filter_done(chain, state))

    elif wants_quality:
        chain = [
            PendingAction(tool="dataset.inspect", arguments=args(), source="playbook"),
            PendingAction(tool="dataset.quality", arguments=args(), source="playbook"),
        ]
        selection = PlaybookSelection("quality", _filter_done(chain, state))

    elif wants_eda:
        chain = [
            PendingAction(tool="dataset.inspect", arguments=args(), source="playbook"),
            PendingAction(tool="eda.describe", arguments=args(), source="playbook"),
        ]
        if wants_corr:
            chain.append(PendingAction(tool="eda.correlation", arguments=args(), source="playbook"))
        selection = PlaybookSelection("eda", _filter_done(chain, state))
    else:
        chain = [
            PendingAction(tool="dataset.inspect", arguments=args(), source="playbook"),
            PendingAction(tool="dataset.profile", arguments=args(), source="playbook"),
        ]
        selection = PlaybookSelection("intake", _filter_done(chain, state))

    if wants_report:
        tail = PendingAction(tool="report.generate", arguments=args(), source="playbook")
        done = state.completed_tools() if state else set()
        if tail.tool not in done and not any(a.tool == tail.tool for a in selection.actions):
            selection.actions.append(tail)
    return selection


# ---------------------------------------------------------------- 内部工具

def _filter_done(chain: list[PendingAction], state: TaskState | None) -> list[PendingAction]:
    """去掉已完成或已在队的动作（state 驱动去重，多轮不重复只读步骤）。"""
    if state is None:
        return chain
    done = state.completed_tools()
    pending = {a.tool for a in state.pending_actions}
    return [a for a in chain if a.tool not in done and a.tool not in pending]


def _modeling_chain(user_request: str, args_fn, state: TaskState | None, wants_evaluate: bool, *, base_index: int = 0) -> list[PendingAction]:
    """建模链：inspect/schema/profile → detect_task → prepare → train [→ evaluate]。

    引用步号按**过滤后批次**内的实际位置生成；detect 已在历史 Run 完成且事实中
    带 target 时直接用字面量，避免重复只读/推理步骤。``base_index`` 是本批在
    当前 Run 内的起始绝对步号（0 基）。
    """
    goal = (user_request or "")[:200]
    chain = [
        PendingAction(tool="dataset.inspect", arguments=args_fn(), source="playbook"),
        PendingAction(tool="dataset.schema", arguments=args_fn(), source="playbook"),
        PendingAction(tool="dataset.profile", arguments=args_fn(), source="playbook"),
        PendingAction(tool="ml.detect_task", arguments=args_fn({"infer_target": True, "goal": goal}), source="playbook"),
    ]
    chain = _filter_done(chain, state)

    target_ref = _target_from_facts(state)
    if target_ref is None:
        detect_pos = next((i + 1 for i, a in enumerate(chain) if a.tool == "ml.detect_task"), 0)
        if detect_pos:
            target_ref = "{{step%d.target}}" % (base_index + detect_pos)

    if target_ref:
        chain.append(PendingAction(tool="ml.prepare", arguments=args_fn({"target": target_ref}), source="playbook"))
    model = model_hint(user_request)
    train_args = {"model": model, "goal": goal}
    if target_ref:
        train_args["target"] = target_ref
    chain.append(PendingAction(tool="ml.train", arguments=args_fn(train_args), source="playbook"))
    if wants_evaluate:
        chain.append(
            PendingAction(
                tool="ml.evaluate",
                arguments={"run_id": "{{step%d.run_id}}" % (base_index + len(chain))},
                source="playbook",
            )
        )
    # detect 之后的 prepare/train/evaluate 是新执行动作，不做「已完成工具」粗粒度去重；
    # 参数级去重仍由 TaskState.queue 在入队时兜底。
    return chain


def _target_from_facts(state: TaskState | None) -> Any:
    if state is None:
        return None
    fact = state.facts.get("ml.detect_task")
    if isinstance(fact, dict) and fact.get("target"):
        return fact["target"]
    return None
