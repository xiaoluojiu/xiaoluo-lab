"""本地 AI Router 的 I/O 契约（单一事实源）。

本模块只做一件事：把「平台当前真实具备的能力」翻译成一份**封闭、可校验、可训练**的
输入输出契约。运行时（Router 推理）与离线（数据集构建 / 评测）共用它，
避免训练与推理两套定义漂移。

三条设计原则
------------
1. **标签空间动态派生**。工具清单与参数约束一律从 `TOOL_REGISTRY` 读取，
   本文件不硬编码任何工具名或参数名。平台增删工具时契约自动跟随 ——
   这是「不让模型学习平台不存在的能力」的**结构性保证**，而不是靠人工审查。
2. **输出封闭**。`RouterDecision` 的字段都是枚举或受限类型，可直接用于约束解码
   （constrained decoding），小模型也不容易吐出非法结构。
3. **「缺槽位」与「要升级」是两件事**。前者由 `missing` 表达，正确处置是**反问用户**
   （零成本，体验也更好）；后者由 `escalate` 表达，才需要交给上级模型。
   把两者混为一谈会造成大量无效升级 —— 这是 Router 设计里最常见的浪费源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 一、封闭枚举
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """Router 判定的能力域（粗粒度）。

    刻意**不等同于工具名**：intent 说「用户想干什么」，tool 说「平台用哪个工具干」。
    两者必须自洽（见 `intent_of_tool`）—— 不自洽本身就是一次免费的置信度信号。
    """

    CHAT = "chat"
    DATASET = "dataset"
    DATA_TRANSFORM = "data_transform"
    EDA = "eda"
    ML = "ml"
    WORKFLOW = "workflow"
    REPORT = "report"


class EscalationReason(str, Enum):
    """升级到上级模型的原因。

    封闭枚举的好处：可统计「为什么升级」，从而针对性补数据。
    例如升级原因长期集中在 `multi_step`，说明该补的是多步编排样本，
    而不是继续加大意图分类的数据量。
    """

    MULTI_STEP = "multi_step"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"
    PARAM_DEPENDENCY = "param_dependency"
    CONFLICT = "conflict"


# 工具 category -> intent。category 是 `Tool` 基类的既有字段，不新增概念。
_CATEGORY_TO_INTENT: dict[str, Intent] = {
    "dataset": Intent.DATASET,
    "data": Intent.DATA_TRANSFORM,
    "eda": Intent.EDA,
    "ml": Intent.ML,
    "workflow": Intent.WORKFLOW,
    "report": Intent.REPORT,
    # connector 是「外部数据源接入」，归属数据集能力域（数据接入）。
    "connector": Intent.DATASET,
}

# 工具名前缀 -> intent。作为 category 缺失时的回退来源。
# 工具命名统一是 `前缀.动作`，前缀比 category 更稳定：category 会被中间基类吞掉，
# 前缀不会（见 `category_consistency_report` 暴露的那 12 个工具）。
_INTENT_BY_PREFIX: dict[str, Intent] = {
    "dataset": Intent.DATASET,
    "data": Intent.DATA_TRANSFORM,
    "eda": Intent.EDA,
    "ml": Intent.ML,
    "workflow": Intent.WORKFLOW,
    "report": Intent.REPORT,
    "connector": Intent.DATASET,
}


# ---------------------------------------------------------------------------
# 二、平台能力（全部动态派生，零硬编码）
# ---------------------------------------------------------------------------


def _ensure_registered() -> None:
    """确保内置工具已注册。

    工具注册是 `app.tools.builtin` 的模块级副作用，而契约层不 import 它
    （否则任何 import contract 的地方都会连带加载全部 30 个工具实现）。
    这里在真正需要工具清单时按需触发一次 —— `register_builtin_tools` 自身
    有 `_REGISTERED` 守卫，重复调用无副作用。
    """
    from app.tools.builtin import register_builtin_tools

    register_builtin_tools()


def tool_specs() -> list[dict[str, Any]]:
    """当前平台真实注册的工具描述（含 input_schema）。"""
    _ensure_registered()
    from app.tools.registry import TOOL_REGISTRY

    return TOOL_REGISTRY.list()


#: 不进入 Router 标签空间的工具 category。这些是 Agent **内部**工具，
#: 用户不会主动请求它们（例如 ``agent.clarify`` 是计划执行中的反问通道），
#: 让 Router 学「预测反问」会把内部编排动作误当成用户意图 —— 语义上无解。
#: 排除后模型标签空间只含「用户真实可请求」的能力，staleness 也不会因
#: 新增内部工具而无谓失效。
_NON_ROUTABLE_CATEGORIES = frozenset({"agent"})


def is_routable_tool(name: str) -> bool:
    """该工具是否属于「用户可请求」能力（决定是否进入 Router 标签空间）。"""
    spec = tool_spec(name)
    if spec is None:
        return False
    return str(spec.get("category", "")) not in _NON_ROUTABLE_CATEGORIES


def tool_label_space() -> list[str]:
    """工具标签空间（已排序，训练与评测必须共用同一份）。

    只含「用户可请求」的工具（排除 ``agent.clarify`` 这类内部反问通道），
    避免模型把内部编排动作当成用户意图去学。
    """
    _ensure_registered()
    from app.tools.registry import TOOL_REGISTRY

    return [name for name in TOOL_REGISTRY.names() if is_routable_tool(name)]


def tool_spec(name: str) -> dict[str, Any] | None:
    """按名取单个工具描述；不存在返回 None（而不是抛异常，生成器要容忍空值）。"""
    for spec in tool_specs():
        if spec.get("name") == name:
            return spec
    return None


def intent_of_tool(name: str) -> Intent | None:
    """工具所属能力域。

    派生顺序：优先用平台显式声明的 `category`，缺失或未映射时回退到工具名前缀。
    两条路都走不通才返回 None。

    回退是必要的：有 12 个工具经中间基类继承，漏掉了 `category` 声明，
    实际值是基类默认的 `"general"`（详见 `category_consistency_report`）。
    """
    spec = tool_spec(name)
    if spec is not None:
        mapped = _CATEGORY_TO_INTENT.get(str(spec.get("category", "")))
        if mapped is not None:
            return mapped
    return _INTENT_BY_PREFIX.get(name.split(".", 1)[0])


def tools_of_intent(intent: Intent) -> list[str]:
    """某能力域下的全部工具（生成数据、算召回、做覆盖度检查都用它）。"""
    return sorted(name for name in tool_label_space() if intent_of_tool(name) == intent)


def category_consistency_report() -> list[dict[str, Any]]:
    """列出 `category` 声明与工具名前缀不一致（含缺失）的工具。

    平台侧补齐 `category` 后此列表应为空 —— 它同时是一份回归检查。
    存在的意义：`ToolRegistry.retrieve_with_scores` 的类目词加分按 `category`
    查表，声明缺失即等于该类目词加分整段失效，会拖低工具召回质量。
    """
    inconsistent: list[dict[str, Any]] = []
    for name in tool_label_space():
        spec = tool_spec(name) or {}
        declared = str(spec.get("category", ""))
        expected_prefix = _INTENT_BY_PREFIX.get(name.split(".", 1)[0])
        if _CATEGORY_TO_INTENT.get(declared) == expected_prefix:
            continue
        inconsistent.append(
            {
                "tool": name,
                "declared_category": declared or "(empty)",
                "expected_category_from_name": name.split(".", 1)[0],
                "hint_lookup_hit": declared in _CATEGORY_TO_INTENT,
            }
        )
    return inconsistent


def input_schema(name: str) -> dict[str, Any]:
    """工具的输入 JSON Schema。"""
    spec = tool_spec(name)
    if spec is None:
        return {}
    schema = spec.get("input_schema")
    return schema if isinstance(schema, dict) else {}


def known_params(name: str) -> dict[str, dict[str, Any]]:
    """工具 schema 声明的全部参数（含可选）。"""
    props = input_schema(name).get("properties")
    return props if isinstance(props, dict) else {}


def required_params(name: str) -> list[str]:
    """必填参数名。缺这些应当**反问用户**，而不是升级。"""
    required = input_schema(name).get("required")
    return [str(item) for item in required] if isinstance(required, list) else []


def optional_params(name: str) -> list[str]:
    """可选参数名。"""
    required = set(required_params(name))
    return [key for key in known_params(name) if key not in required]


def param_type(name: str, param: str) -> str | None:
    """参数声明的 JSON 类型（integer / number / string / boolean / array / object）。"""
    spec = known_params(name).get(param)
    if not isinstance(spec, dict):
        return None
    declared = spec.get("type")
    return str(declared) if declared else None


# ---------------------------------------------------------------------------
# 三、输入 / 输出模型
# ---------------------------------------------------------------------------


class RouterRequest(BaseModel):
    """Router 的一次输入。

    **刻意保持极小**，且**不包含工具清单**。
    输入长度直接决定本地推理能否跑得快：把 30 个工具的 schema 塞进 prompt
    会让小模型 prefill 上万 token，本地反而比云端 API 慢得多。
    工具选择依赖模型对封闭标签空间的记忆（30 类），而不是 prompt 里的枚举。
    """

    utterance: str
    bound_dataset_id: int | None = None
    available_columns: list[str] = Field(default_factory=list)
    recent_tools: list[str] = Field(default_factory=list)
    #: 已绑定数据集的**名字**。词法模型用不到（它只关心「绑没绑」这个二值信号），
    #: 但神经模型必须按训练口径把「已绑定数据集 {name}（dataset_id={id}）」写进提示词
    #: —— 训练语料里就是这个形态，不写会造成输入分布漂移。
    bound_dataset_name: str | None = None
    #: 当前可见数据集，形如 `[{"name": "sales", "id": 2}]`。
    #: 神经模型靠它把「sales 表」解析成具体 dataset_id；解析层同时用它做**幻觉校验**
    #: （模型吐出的 id 不在本表里 ⇒ 丢弃，不采信）。
    available_datasets: list[dict[str, Any]] = Field(default_factory=list)


class RouterDecision(BaseModel):
    """Router 的输出。字段全部受限，可直接约束解码。

    `intent` 允许为空，且**只在升级决策时为空**：`escalate` 由 L0 结构规则先于模型触发
    （见 `escalation_rules.detect_escalation`），此时「用户想干什么」**尚未判定**。
    填一个猜测值（例如强行写 `chat`）会让下游与日志把它读成真实结论 —— 空值表达「无信息」比假值诚实。
    """

    intent: Intent | None = None
    tool: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    escalate: bool = False
    escalate_reason: EscalationReason | None = None
    confidence: float = 0.0


def decision_json_schema() -> dict[str, Any]:
    """给约束解码用的 schema：把 `tool` 的取值收紧到当前注册表。"""
    schema = RouterDecision.model_json_schema()
    properties = schema.setdefault("properties", {})
    properties["tool"] = {
        "anyOf": [{"type": "string", "enum": tool_label_space()}, {"type": "null"}]
    }
    return schema


# ---------------------------------------------------------------------------
# 四、校验与路由决策
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    """一次决策的确定性校验结果。不含任何模型自评成分。"""

    problems: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    unknown_params: list[str] = field(default_factory=list)

    @property
    def well_formed(self) -> bool:
        return not self.problems


def _type_ok(value: Any, json_type: str | None) -> bool:
    if json_type is None:
        return True
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


def validate_decision(decision: RouterDecision) -> ValidationOutcome:
    """纯确定性校验：结构、标签、参数类型、必填槽位。

    这里**不做升级判断** —— 只回答「这份决策本身是否自洽」。
    升级策略见 `decide_route`，两者分开是为了能独立地评测「校验器」和「策略」。
    """
    outcome = ValidationOutcome()

    if decision.escalate:
        # 主动升级的决策不需要工具，但必须给出原因。
        if decision.escalate_reason is None:
            outcome.problems.append("escalate=true 但缺少 escalate_reason")
        return outcome

    # 1) 闲聊/无需工具的能力域允许 tool 为空。
    tool_free_intents = {Intent.CHAT}
    if decision.tool is None:
        if decision.intent not in tool_free_intents:
            # intent 可能为 None（升级路径），这里不能直接取 .value。
            shown = decision.intent.value if decision.intent is not None else "(未判定)"
            outcome.problems.append(f"intent={shown} 必须给出 tool")
        return outcome

    # 2) tool 必须在当前注册表内。
    label_space = set(tool_label_space())
    if decision.tool not in label_space:
        outcome.problems.append(f"tool={decision.tool!r} 不在当前工具注册表中")
        return outcome

    # 3) intent 与 tool 的 category 必须自洽。
    expected_intent = intent_of_tool(decision.tool)
    if expected_intent is not None and expected_intent != decision.intent:
        outcome.problems.append(
            f"intent={decision.intent.value} 与 tool={decision.tool} 所属"
            f"能力域 {expected_intent.value} 不一致"
        )

    # 4) 参数键必须是 schema 声明过的。
    declared = known_params(decision.tool)
    for key in decision.params:
        if key not in declared:
            outcome.unknown_params.append(key)
    if outcome.unknown_params:
        outcome.problems.append(
            f"tool={decision.tool} 不认识的参数：{sorted(outcome.unknown_params)}"
        )

    # 5) 参数值类型粗校验。
    for key, value in decision.params.items():
        if value is None:
            continue
        declared_type = param_type(decision.tool, key)
        if not _type_ok(value, declared_type):
            outcome.problems.append(
                f"参数 {key} 期望 {declared_type}，实际 {type(value).__name__}"
            )

    # 6) 必填槽位缺失 —— 这是「反问用户」的依据，不是升级依据。
    given = {key for key, value in decision.params.items() if value is not None}
    outcome.missing_required = [
        name for name in required_params(decision.tool) if name not in given
    ]

    return outcome


RouteAction = Literal["execute", "ask_user", "escalate"]


def decide_route(
    outcome: ValidationOutcome,
    *,
    confidence: float,
    confidence_threshold: float = 0.55,
) -> tuple[RouteAction, str]:
    """把校验结果 + 置信度翻译成一条动作。

    顺序即优先级：结构不合法 > 置信度不足 > 缺槽位 > 执行。
    「缺槽位」优先级最低，因为它是**可以零成本挽回**的一类 —— 反问一句即可。
    """
    if not outcome.well_formed:
        return "escalate", "结构校验未通过：" + "；".join(outcome.problems[:2])
    if confidence < confidence_threshold:
        return "escalate", f"置信度 {confidence:.2f} 低于阈值 {confidence_threshold:.2f}"
    if outcome.missing_required:
        return "ask_user", "缺少必填参数：" + "、".join(outcome.missing_required)
    return "execute", ""
