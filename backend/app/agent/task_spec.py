"""TaskSpec —— 「用户到底要完成什么」的最小结构化描述。

定位（与既有结构的边界）
------------------------
仓库已有 ``Plan`` / ``AgentContext`` / ``Decision`` / ``Intent``，本模块**不新增平行结构**，
只补一个缺口：现有结构里没有「用户任务」这一层 —— ``Intent`` 是粗粒度入口分类（兼容层），
``Plan`` 是「准备执行的动作集合」，二者之间缺一层「用户真正想完成什么」的**任务语义**。

职责分工（单一事实源，禁止互相重复）：

====================  ====================================================
Intent                 粗粒度入口分类 / 兼容层（chat / eda / ml / …）
TaskSpec               用户要完成什么（**静态**：goal / domain / sub_goal / …）
AgentContext           当前执行上下文 / **可变状态**（数据集、工具、已答反问…）
Plan                   当前准备执行的动作集合（steps）
Decision               某一步**为什么**选这个动作（value / source / confidence）
====================  ====================================================

TaskSpec 只做一件事：**描述任务**。它不执行 Tool，不承载执行进度，不保存可变状态。
执行过程中的任何可变状态都应放进 ``AgentContext``（其职责就是「执行上下文」），
**不新建一个与 AgentContext 重复职责的 ExecutionState**。

为什么需要 TaskSpec 而不是继续用 Intent
----------------------------------------
``Intent`` 只回答「属于哪类」，回答不了「具体做什么」：
「看看这批数据的分布」与「算一列的相关性」都是 ``Intent.EDA``，
但前者是**数据集级**分布总览（scope=dataset，无需 column），后者是**单列**相关性
（scope=column，需要 column）。把两者压成一个 Intent 就会逼着下游去猜列名
（见 PoC ①「数据集级分布」被错误路由到 dataset.inspect 的真实案例）。

TaskSpec 用 ``sub_goal`` + ``scope`` + ``entities`` 表达这一层差异，让
``DecisionProvider`` 有足够信息做正确的工具选择，而不是靠关键词硬猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskDomain(StrEnum):
    """任务能力域（与 ``Intent`` 对应，但语义是「任务」，不是「入口」）。"""

    CHAT = "chat"
    DATASET = "dataset"
    DATA_TRANSFORM = "data_transform"
    EDA = "eda"
    ML = "ml"
    WORKFLOW = "workflow"
    REPORT = "report"


class TaskScope(StrEnum):
    """任务作用域 —— 决定工具是否需要列级参数。"""

    UNKNOWN = "unknown"
    DATASET = "dataset"   # 数据集级（如「看看这批数据的分布」）
    COLUMN = "column"     # 单列级（如「这一列的分布」）
    MULTI = "multi"       # 跨数据集 / 跨步骤


class Complexity(StrEnum):
    """任务复杂度 hint —— 决定走「本地直接决定 / 本地决策 / 远程升级」哪条路径。"""

    SIMPLE = "simple"     # 本地直接决定，零额外模型调用
    MEDIUM = "medium"     # 本地 Router 决策 + Tool，循环本地决策
    COMPLEX = "complex"   # 本地理解后升级远程做战略级规划，本地执行


@dataclass
class TaskSpec:
    """用户任务的**静态**描述。不承载任何执行状态。"""

    goal: str = ""
    #: 能力域。缺省时由 TaskSpec 的构造路径（Router intent）决定，不强求。
    domain: TaskDomain | None = None
    #: 子目标（如 EDA 下的 distribution_overview / correlation / outlier）。
    sub_goal: str = ""
    #: 作用域（数据集级 / 单列级 / 跨数据集）。
    scope: TaskScope = TaskScope.UNKNOWN
    #: 任务约束（只读、必须真实列名、不得臆造数据等）。
    constraints: list[str] = field(default_factory=list)
    #: 期望产出（一段回答 / 一份报告文件 / 一个模型 / 一次导入 …）。
    expected_output: str = ""
    #: 涉及的实体（列名、数据集、模型名…）。**来自真实上下文，禁止臆造**。
    entities: dict[str, Any] = field(default_factory=dict)
    #: 复杂度 hint（供 DecisionProvider 选择决策路径）。
    complexity_hint: Complexity = Complexity.SIMPLE
    #: 任务理解的置信度（由 Router / 本地模型给出，非自评）。
    confidence: float = 0.0
    #: 任务理解依据（router / local_model / rule / remote …）。
    source: str = ""

    @property
    def resolved(self) -> bool:
        """是否已形成可执行的任务描述（goal 非空）。"""
        return bool(self.goal and self.goal.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "domain": self.domain.value if self.domain else None,
            "sub_goal": self.sub_goal,
            "scope": self.scope.value,
            "constraints": list(self.constraints),
            "expected_output": self.expected_output,
            "entities": dict(self.entities),
            "complexity_hint": self.complexity_hint.value,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass
class TaskUnderstanding:
    """「本地理解」的完整产出：TaskSpec + 备选 + 依据。

    这是 ``Decision`` 契约在任务层的一次实例化 —— TaskSpec 是 value，
    备选与依据补齐「为什么这么理解」的证据链，供 DecisionTrace 记录、供后续训练。
    """

    spec: TaskSpec
    #: 备选理解（如「distribution_overview」vs「单列 distribution」）。
    alternatives: list[TaskSpec] = field(default_factory=list)
    #: 判定依据（命中的信号、Router 候选、规则命中等）。
    evidence: dict[str, Any] = field(default_factory=dict)
    #: 判定来源（router / rule / local_model / remote）。
    source: str = ""
    #: 是否需要澄清（信息不足，无法唯一确定）。
    needs_clarification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "evidence": dict(self.evidence),
            "source": self.source,
            "needs_clarification": self.needs_clarification,
        }
