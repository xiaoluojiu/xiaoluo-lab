"""DecisionProvider —— 统一的「这一步该怎么走」决策入口。

这是本次架构的核心抽象。**只新增这一层**，不建立
``SemanticProvider / DecisionProvider / ReasoningProvider / PlanningProvider /
StrategyProvider`` 五套 Provider。

要解决的问题
------------
「谁来 Decide」过去散在三个地方，各自为政：

* ``intent.classify`` 用关键词判能力域；
* ``planner._rule_plan`` 用另一套关键词拼计划；
* ``replanner`` 用错误文案正则决定重试/跳过/终止。

结果是「同一条请求，三处可能做出三个不同的判断」，而且没有统一的证据链
（谁判的 / 凭什么 / 多可信），未来训练无从下手。

本模块把「每一步选择哪个动作」收敛到**一个可替换入口**：

    decide(context) -> Decision

三个实现（**不是同等优先级**）：

====================  ====================================================
RuleDecisionProvider      确定性系统命令 / 取消确认 / 状态校验 / 必填参数检查
LocalModelDecisionProvider 本地 Router 模型（词法 / 神经 / 融合）选工具
RemoteLLMDecisionProvider  远程大模型做**战略级**指导（复杂任务升级）
====================  ====================================================

推荐优先级（由调用方组合，不在本模块硬编码调用链）：

    LocalModel  →  必要时 Rule fallback  →  必要时 Remote escalation

关键约束（任务 5.4）
--------------------
**Decide ≠ 每一步都调模型。** 三条路径对应三种复杂度：

* 简单任务：本地直接决定，**零额外模型调用**；
* 一般任务：Local Router → Tool → Result → Local Decision；
* 复杂任务：本地判断复杂 → Remote 战略级指导 → 本地 Agent 执行。

因此 DecisionProvider 必须可替换，但不是要求所有 Decide 都调 LLM。
最终目标是「统一决策入口」，而不是「模型调用泛滥」。

Rule 的职责被**严格限制**（任务 5.3）：只承担无歧义的安全/流程控制，
禁止继续增加「分布→xxx」「相关性→xxx」这类**语义映射**（那些交给本地模型）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.contracts import Decision


#: 决策来源 code —— 与 answer_source 区分：answer_source 标记「回答是谁给的」，
#: 这里标记「**这一步的决策**是谁给的」。两者都进 DecisionTrace，供训练闭环用。
class DecisionSource:
    RULE = "rule"
    #: 词法 Router（TF-IDF + LinearSVC，见 `app.local_router.model`）。
    LOCAL_ROUTER = "local_router"
    #: 神经 Router（Qwen3-0.6B + LoRA，见 `app.local_router.qwen`）。
    #: 与 LOCAL_ROUTER 分开是为了在 DecisionTrace 里能分别评估两层；
    #: 任何「只认 local_router」的白名单都必须同时接受这一个。
    LOCAL_QWEN = "local_qwen"
    LOCAL_MODEL = "local_model"
    LOCAL_AGENT = "local_agent"
    REMOTE_LLM = "remote_llm"
    FALLBACK = "fallback"


#: 决策动作 —— 这一步「做什么」。不是工具名，是决策层的动作语义。
class DecisionAction:
    EXECUTE_TOOL = "execute_tool"      # 执行某工具
    ASK_USER = "ask_user"              # 反问用户（缺必填参数 / 需澄清）
    ESCALATE = "escalate"              # 升级远程
    CHAT = "chat"                      # 直接对话
    STOP = "stop"                      # 终止
    REPLAN = "replan"                  # 重规划（当前步骤失败）


@dataclass
class AgentDecision:
    """「这一步为什么这么做」的结构化判定。

    它是对 ``core.contracts.Decision`` 在 Agent 决策层的一次具体化：
    ``value`` = 要执行的动作；``evidence`` 携带 Router 候选、ToolResult signals 等
    客观依据 —— 这些正是 DecisionTrace 要记录、后续训练要消费的东西。
    """

    action: str                       # DecisionAction 之一
    tool: str | None = None           # action == execute_tool 时的工具名
    params: dict[str, Any] = field(default_factory=dict)
    source: str = ""                  # DecisionSource 之一
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    #: 备选动作（Router 候选 / 其他可行工具），供 trace 与「分歧检测」用。
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    #: 客观依据：Router 候选得分、ToolResult signals、规则命中等。
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool": self.tool,
            "params": dict(self.params),
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "reasons": list(self.reasons),
            "alternatives": list(self.alternatives),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_decision(cls, action: str, decision: Decision | None, **extra: Any) -> "AgentDecision":
        """从通用 ``Decision`` 构造（复用其 value/source/confidence/reasons）。"""
        if decision is None:
            return cls(action=action, **extra)
        return cls(
            action=action,
            source=decision.source,
            confidence=decision.confidence,
            reasons=list(decision.reasons),
            alternatives=[{"value": a} for a in decision.alternatives],
            evidence=dict(decision.evidence),
            **extra,
        )


class DecisionProvider(ABC):
    """统一决策入口。任何「这一步怎么走」的判定都通过它，保证可替换 + 可记录。"""

    @abstractmethod
    def decide(self, context: "DecisionContext") -> AgentDecision:
        """给定决策上下文，返回这一步的动作判定。"""


@dataclass
class DecisionContext:
    """决策所需的输入快照（刻意保持扁平，避免把整个 AgentContext 拖进来）。"""

    #: 用户请求原文。
    user_request: str = ""
    #: 任务理解（TaskSpec）。
    task_spec: Any = None
    #: 当前已绑定的数据集 id（None 表示未绑定）。
    bound_dataset_id: int | None = None
    #: 已绑定数据集的**名字**。神经 Router 要按训练口径把
    #: 「已绑定数据集 {name}（dataset_id={id}）」写进提示词。
    bound_dataset_name: str | None = None
    #: 可见数据集 `[{"name": ..., "id": ...}]`。
    #: 神经 Router 靠它把「sales 表」解析成具体 dataset_id；解析层同时用它做
    #: **幻觉校验**（模型吐出的 id 不在本表里 ⇒ 丢弃）。
    available_datasets: list[dict[str, Any]] = field(default_factory=list)
    #: 可见列名（来自真实 schema，禁止臆造）。
    available_columns: list[str] = field(default_factory=list)
    #: 近期工具（多轮指代消解用）。
    recent_tools: list[str] = field(default_factory=list)
    #: 上一次 ToolResult 的 signals（驱动「Observe→Decide→Act」的下一跳）。
    signals: dict[str, Any] = field(default_factory=dict)
    #: 已尝试/失败的历史（供 RuleDecisionProvider 做终止判定）。
    attempts: int = 0
    #: 其他结构化信号（问答答案、Preflight 结果等）。
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "task_spec": self.task_spec.to_dict() if hasattr(self.task_spec, "to_dict") else self.task_spec,
            "bound_dataset_id": self.bound_dataset_id,
            "available_columns": list(self.available_columns),
            "recent_tools": list(self.recent_tools),
            "signals": dict(self.signals),
            "attempts": self.attempts,
            "extra": dict(self.extra),
        }
