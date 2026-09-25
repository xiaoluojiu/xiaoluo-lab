"""TaskSpecBuilder —— 把「用户语言」转成「任务描述」的本地理解入口。

这是「用户语言 → 本地理解 → TaskSpec」链路的落点。它组合两类信号：

* ``intent.classify`` —— 粗粒度能力域（兼容层，规则关键词）；
* ``LocalModelDecisionProvider`` —— 本地 Router 模型的工具选择（真正参与理解）。

产物是 :class:`~app.agent.task_spec.TaskUnderstanding`：TaskSpec + 备选 + 依据，
供 DecisionProvider 决策、供 DecisionTrace 记录、供后续训练。

不做什么
--------
* 不执行 Tool（那是 AgentExecutor 的职责）；
* 不承载执行状态（那是 AgentContext 的职责）；
* 不臆造列名 / 数据集 —— entities 只放**真实上下文**里拿得到的东西。
"""

from __future__ import annotations

from typing import Any

from app.agent.decision.provider import DecisionAction, DecisionContext
from app.agent.decision.router import DecisionRouter
from app.agent.task_spec import (
    Complexity,
    TaskDomain,
    TaskScope,
    TaskSpec,
    TaskUnderstanding,
)


# Intent 值 → TaskDomain（复用 contract.Intent，不新造枚举）。
_INTENT_TO_DOMAIN: dict[str, TaskDomain] = {
    "chat": TaskDomain.CHAT,
    "dataset": TaskDomain.DATASET,
    "data_transform": TaskDomain.DATA_TRANSFORM,
    "eda": TaskDomain.EDA,
    "ml": TaskDomain.ML,
    "workflow": TaskDomain.WORKFLOW,
    "report": TaskDomain.REPORT,
}


class TaskSpecBuilder:
    """本地理解：用户语言 → TaskSpec。"""

    def __init__(self, *, router: DecisionRouter | None = None) -> None:
        self.router = router or DecisionRouter(llm=None)

    def understand(
        self,
        user_request: str,
        *,
        bound_dataset_id: int | None = None,
        bound_dataset_name: str | None = None,
        available_datasets: list[dict[str, Any]] | None = None,
        available_columns: list[str] | None = None,
        recent_tools: list[str] | None = None,
        intent_decision: Any = None,
    ) -> TaskUnderstanding:
        """理解一条用户请求，产出 TaskSpec + 备选 + 依据。"""
        from app.agent.intent import classify

        # 1) 粗粒度能力域（兼容层）。
        intent = intent_decision if intent_decision is not None else \
            classify(user_request, has_datasets=bound_dataset_id is not None)
        domain = _INTENT_TO_DOMAIN.get(str(intent.value) if hasattr(intent, "value") else str(intent))

        # 2) 本地 Router 模型选工具（真正参与理解）。
        decision = self.router.route(
            DecisionContext(
                user_request=user_request,
                bound_dataset_id=bound_dataset_id,
                bound_dataset_name=bound_dataset_name,
                available_datasets=list(available_datasets or []),
                available_columns=list(available_columns or []),
                recent_tools=list(recent_tools or []),
            ),
            allow_remote=False,
        )

        # ★ multi_step 是「本地多步编排」信号，不是「远程复杂」信号（控制权归位核心）。
        # 本地 Router 对「先X再Y」这类连词句会升级为 escalate::multi_step 且不选工具
        # （L0 规则先于 L1 模型触发）。但这类任务是**本地可循环推进**的：首步工具能从
        # 第一个分句识别出来，后续步骤由 ToolResult.signals 驱动（_run_dynamic）。
        # 若原样交给下游，会被当成 complexity=complex → 走旧 Rule Planner 覆盖语义结果。
        # 这里把 multi_step 拆成「首步工具 + medium 复杂度」，让本地动态循环接管，
        # 而不是升级远程 / 退回关键词规划。
        if decision.action == DecisionAction.ESCALATE and \
                str(decision.evidence.get("escalate_reason")) == "multi_step":
            first_clause = _split_first_clause(user_request)
            if first_clause and first_clause != user_request:
                first_decision = self.router.route(
                    DecisionContext(
                        user_request=first_clause,
                        bound_dataset_id=bound_dataset_id,
                        available_columns=list(available_columns or []),
                    ),
                    allow_remote=False,
                )
                if first_decision.action == DecisionAction.EXECUTE_TOOL and first_decision.tool:
                    decision = first_decision
                    # 标记为本地多步：首步工具 + 循环推进（_run_dynamic 消费）。
                    decision.evidence["multi_step"] = True
                    decision.evidence["first_clause"] = first_clause

        # 3) 组装 TaskSpec（静态任务描述）。
        #    domain 优先从 Router 预测的工具反推（工具→能力域比关键词更可靠；
        #    实测「看看这批数据的分布」被关键词规则误判为 DATASET，而 Router 正确给出
        #    eda.distribution）。intent 分类仅作兜底。
        domain = _domain_from_tool(decision.tool) or domain

        spec = TaskSpec(
            goal=user_request.strip(),
            domain=domain,
            sub_goal=_infer_sub_goal(decision, domain),
            scope=_infer_scope(decision, user_request),
            constraints=["只读分析时不得修改数据", "列名必须来自真实 schema，不得臆造"],
            confidence=decision.confidence,
            source=decision.source,
        )

        # 工具信息放进 entities（决策层要用），而不是 TaskSpec 直接执行。
        if decision.tool:
            spec.entities["tool"] = decision.tool
            spec.entities["router_route"] = decision.evidence.get("route", "")
        spec.entities["intent"] = str(intent.value) if hasattr(intent, "value") else str(intent)
        spec.complexity_hint = _infer_complexity(decision, user_request)

        return TaskUnderstanding(
            spec=spec,
            alternatives=[a for a in _alternatives(decision)],
            evidence={
                "router_decision": decision.to_dict(),
                "intent": str(intent.value) if hasattr(intent, "value") else str(intent),
            },
            source=decision.source,
            needs_clarification=decision.action == DecisionAction.ASK_USER,
        )


def _domain_from_tool(tool: str | None) -> TaskDomain | None:
    """从工具名反推能力域（工具前缀比关键词更稳定）。"""
    if not tool:
        return None
    prefix = tool.split(".", 1)[0]
    return {
        "dataset": TaskDomain.DATASET,
        "data": TaskDomain.DATA_TRANSFORM,
        "eda": TaskDomain.EDA,
        "ml": TaskDomain.ML,
        "workflow": TaskDomain.WORKFLOW,
        "report": TaskDomain.REPORT,
        "connector": TaskDomain.DATASET,
    }.get(prefix)


def _infer_sub_goal(decision: Any, domain: TaskDomain | None) -> str:
    """从 Router 预测的工具推断子目标（如 eda.distribution → distribution）。"""
    tool = decision.tool or ""
    if "." in tool:
        return tool.split(".", 1)[1]
    return ""


def _infer_scope(decision: Any, user_request: str) -> TaskScope:
    """从决策推断作用域：数据集级 vs 单列级 vs 跨数据集。"""
    if decision.action == DecisionAction.CHAT:
        return TaskScope.UNKNOWN
    tool = decision.tool or ""
    # 单列工具 → column；数据集级工具 → dataset。
    if tool in ("eda.distribution", "eda.visualize", "data.filter", "data.aggregate"):
        return TaskScope.COLUMN
    if tool.startswith("data.merge"):
        return TaskScope.MULTI
    return TaskScope.DATASET


def _infer_complexity(decision: Any, user_request: str) -> Complexity:
    """复杂度 hint：决定决策路径（本地直接 / 本地循环 / 远程升级）。"""
    if decision.action == DecisionAction.CHAT:
        return Complexity.SIMPLE
    if decision.action == DecisionAction.ESCALATE:
        return Complexity.COMPLEX
    # ★ 本地多步编排（multi_step 拆解后）：medium → 本地动态循环推进，不升级远程。
    if decision.evidence.get("multi_step"):
        return Complexity.MEDIUM
    # 置信度低的本地决定 → medium（需要更谨慎的循环决策）。
    if decision.confidence < 0.5:
        return Complexity.MEDIUM
    return Complexity.SIMPLE


# 多步编排的分句连词（与 local_router/escalation_rules.SEQUENCE_MARKERS 同源，但只取
# 「后置分句开头」这类能稳定切分首句的标记；「先」通常位于句首，不用于切分）。
_SEQUENCE_SPLIT_MARKERS: tuple[str, ...] = (
    "再", "然后", "接着", "之后", "最后", "同时",
)


def _split_first_clause(user_request: str) -> str:
    """把「先X再Y」拆出第一个分句 X。

    用于 multi_step 任务的本地拆解：首步工具从 X 识别，后续由 signals 驱动。
    只做**保守切分**：找到第一个分句连词，取它之前的部分；切不出或切后为空
    则返回原句（调用方据此判断「未拆解成功」）。
    """
    text = (user_request or "").strip()
    if not text:
        return ""
    # 逐个连词找最早出现位置，取最小值切分。
    cut = len(text)
    for marker in _SEQUENCE_SPLIT_MARKERS:
        idx = text.find(marker)
        if 0 < idx < cut:
            cut = idx
    if cut < len(text):
        text = text[:cut]
    text = text.strip()
    # 去掉句首的「先」前缀（「先看看数据分布」→「看看数据分布」），并清理尾部分句标点。
    if text.startswith("先"):
        text = text[1:].strip()
    text = text.rstrip("，,。；;、：: ")
    return text if text else user_request.strip()


def _alternatives(decision: Any) -> list[TaskSpec]:
    """把 Router 候选转成备选 TaskSpec（供 trace 与分歧检测）。"""
    return []
