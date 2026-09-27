"""Agent 工具权限模型。

设计取舍
--------
权限只回答三个问题：**能不能做**、**要不要人确认**、**为什么**。

旧架构在此之上叠了「角色 × 风险等级 × 工具白名单 × 一次性凭据」的组合判定，
生效路径分散在 rules / manager / executor / loop 四处，实际哪个分支生效很难预测。
这里收敛为**一个纯函数**，全部规则写在本文件内，可一眼读完。

依赖方向
--------
本模块是**叶子模块**：不导入 app.tools，也不导入 app.agent 的其他子模块。
工具层（app/tools）依赖这里的类型，Agent 引擎也依赖它，但反向依赖不存在，
因此不存在循环导入。改动本文件时不要引入对工具层的 import。

安全默认
--------
``Tool.risk_level`` 的缺省值是 MEDIUM 而非 LOW：忘记声明风险的工具
不应该被静默放行。显式声明为 LOW 的才走免确认路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Permission(str, Enum):
    """工具所需权限。

    采用最小权限：工具只声明它真正需要的那一项，执行上下文（ToolExecutionContext）
    携带当前用户持有的权限集合，两者做包含判断。

    取值必须与工具层实际声明的字符串**逐字一致**。
    历史上这里定义过一组「看起来更规整」的取值（write_data 等），
    与工具层声明的 modify_data / analyze_data / create_version 对不上，
    结果是除 dataset.list 之外的工具全部被判 DENY —— 而因为 Agent 会
    正常返回「失败」，这个错误在冒烟测试里并不显眼。
    新增权限时必须同时在工具层使用同名字符串。
    """

    #: 读数据集内容与元信息
    READ_DATA = "read_data"
    #: 对数据做统计 / 画像 / 探索性分析，不写回
    ANALYZE_DATA = "analyze_data"
    #: 在现有数据集上做筛选 / 清洗 / 变换
    MODIFY_DATA = "modify_data"
    #: 产出新的数据集版本
    CREATE_VERSION = "create_version"
    #: 训练模型
    TRAIN_MODEL = "train_model"
    #: 执行工作流
    EXECUTE_WORKFLOW = "execute_workflow"
    #: 导出报告等产物
    EXPORT_REPORT = "export_report"
    #: 管理外部数据源连接器
    MANAGE_CONNECTOR = "manage_connector"

    def __str__(self) -> str:
        return self.value


class RiskLevel(str, Enum):
    """风险等级。只有 HIGH / CRITICAL 会触发人工确认（human-in-the-loop）。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @property
    def needs_confirmation(self) -> bool:
        """是否必须经用户确认后才可执行。"""
        return self in (RiskLevel.HIGH, RiskLevel.CRITICAL)


@dataclass(frozen=True)
class PermissionDecision:
    """一次权限裁决的结果。三者互斥且完备：ALLOW / DENY / NEEDS_CONFIRMATION。"""

    allowed: bool
    needs_confirmation: bool
    reason: str = ""

    @property
    def denied(self) -> bool:
        """真正的拒绝：**不允许，且不是「等确认」**。

        三态里 ALLOW 与 NEEDS_CONFIRMATION 的 ``allowed`` 并不互补 ——
        「等确认」的 ``allowed`` 同样是 False（还没拿到授权就不许执行）。
        如果把它也算进 ``denied``，调用方会按「权限不足」直接判失败，
        高风险工具于是永远拿不到确认弹窗，只会报一句「需要你确认后才会执行」。
        这是真实发生过的缺陷：``ToolRegistry.execute`` 曾写成
        ``if decision.denied: raise ToolPermissionError``，
        结果每次确认之后仍以同一句话失败。

        想判断「现在能不能直接跑」请用 :attr:`allowed`；
        想判断「是不是被规则挡死」才用本属性。
        """
        return not self.allowed and not self.needs_confirmation

    @classmethod
    def allow(cls, reason: str = "") -> PermissionDecision:
        return cls(allowed=True, needs_confirmation=False, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls(allowed=False, needs_confirmation=False, reason=reason)

    @classmethod
    def confirm(cls, reason: str) -> PermissionDecision:
        return cls(allowed=False, needs_confirmation=True, reason=reason)

    def as_label(self) -> str:
        if self.needs_confirmation:
            return "REQUIRE_CONFIRMATION"
        return "DENY" if self.denied else "ALLOW"


class PermissionManager:
    """权限裁决器。

    判定顺序（顺序本身就是策略，不要调换）：

    1. 权限缺失 → DENY。用户没有工具要求的权限，确认也没有用。
    2. 高风险或工具显式要求确认 → NEEDS_CONFIRMATION。
    3. 其余 → ALLOW。

    数据集访问范围（``context.can_access_dataset``）不在本类判定：
    它由 ``Tool.assert_dataset_access`` 在工具内部按 dataset_id 逐个校验，
    因为「能不能访问这个数据集」依赖参数值，权限层拿不到。
    """

    def check(
        self,
        user_id: str,
        tool: Any,
        context: Any = None,
        params: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """裁决一次工具调用。

        :param user_id: 调用者标识（当前为单租户，仅用于审计日志）
        :param tool: Tool 实例（需有 permission / risk_level / requires_confirmation）
        :param context: ToolExecutionContext（持有用户权限集合）
        :param params: 工具入参（本实现不读取，保留参数位以便将来做参数级策略）
        """
        required = getattr(tool, "permission", None) or Permission.READ_DATA
        held = _as_permission_values(getattr(context, "permissions", None))
        if _as_permission_value(required) not in held:
            return PermissionDecision.deny(
                f"当前上下文不具备执行该工具所需的权限（{_as_permission_value(required)}）"
            )

        risk = _as_risk_level(getattr(tool, "risk_level", None))
        tool_name = getattr(tool, "name", "") or "unknown"
        # 优先读工具自己合并过的结论（Tool.needs_confirmation）；
        # 拿不到（鸭子类型的假工具）时退回「显式开关 or 风险等级」。
        explicit = getattr(tool, "requires_confirmation", False)
        if getattr(tool, "needs_confirmation", explicit) or risk.needs_confirmation:
            return PermissionDecision.confirm(
                f"「{tool_name}」属于{risk.value}风险操作，需要你确认后才会执行"
            )

        return PermissionDecision.allow("")


def risk_needs_confirmation(value: Any) -> bool:
    """这个风险等级要不要人工确认（对外版本）。

    工具自描述（``Tool.describe``）要靠它回答前端「这个工具是不是高风险」。
    早期版本直接把类属性 ``requires_confirmation`` 原样下发，而它几乎没人显式
    声明过 —— ``ml.train`` 明明写着 ``risk_level = "high"``，对外却自称
    ``requires_confirmation = false``。前端据此默认自动放行，
    用户于是**永远看不到确认弹窗**，只看到一句执行失败。
    """
    return _as_risk_level(value).needs_confirmation


def _as_permission_value(value: Any) -> str:
    """归一化成权限字符串。

    工具层既有写 ``Permission.ANALYZE_DATA`` 的，也有直接写 ``"analyze_data"`` 的。
    两种写法必须等价 —— 依赖 str 枚举的哈希与字符串相同是不可靠的
    （``set(Permission)`` 里的成员按枚举名哈希，混用会导致漏判）。
    """
    return str(getattr(value, "value", value))


def _as_permission_values(values: Any) -> set[str]:
    return {_as_permission_value(v) for v in (values or ())}


def _as_risk_level(value: Any) -> RiskLevel:
    """归一化风险等级。

    与权限同理：工具层既有 ``RiskLevel.HIGH`` 也有 ``"high"``。
    无法识别的取值一律按 **MEDIUM** 处理 —— 认不出来的风险不许按 LOW 放行。
    """
    if isinstance(value, RiskLevel):
        return value
    try:
        return RiskLevel(str(getattr(value, "value", value)).lower())
    except (ValueError, AttributeError):
        return RiskLevel.MEDIUM
