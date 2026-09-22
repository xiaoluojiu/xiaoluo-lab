"""Prompt 116：PermissionManager。

输入 user + tool + context，输出 ALLOW / DENY / REQUIRE_CONFIRMATION。
绝对禁止 Agent 绕过 PermissionManager：ToolRegistry.execute 是唯一执行入口，
并在执行前强制调用本管理器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.permission.models import Decision
from app.agent.permission.rules import risk_requires_confirmation
from app.tools.base import Tool


@dataclass
class PermissionDecision:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == Decision.DENY

    @property
    def needs_confirmation(self) -> bool:
        return self.decision == Decision.REQUIRE_CONFIRMATION


# 工具参数中的数据集引用键（用于数据范围校验）
DATASET_PARAM_KEYS = (
    "dataset_id",
    "left_dataset_id",
    "right_dataset_id",
)


class PermissionManager:
    """权限管理器：唯一裁决入口。"""

    def check(
        self,
        user_id: str,
        tool: Tool,
        context: Any,  # ToolExecutionContext（避免循环导入用 Any）
        params: dict | None = None,
    ) -> PermissionDecision:
        params = params or {}
        perm_value = getattr(tool.permission, "value", tool.permission)
        risk_value = getattr(tool.risk_level, "value", tool.risk_level)

        # 1. 权限校验
        if not context.has_permission(tool.permission):
            return PermissionDecision(
                Decision.DENY,
                reason=f"用户 {user_id!r} 缺少权限 {perm_value}",
            )

        # 2. 数据范围校验：参数引用的数据集必须在上下文允许范围内
        for key in DATASET_PARAM_KEYS:
            if key in params and params[key] is not None:
                if not context.can_access_dataset(params[key]):
                    return PermissionDecision(
                        Decision.DENY,
                        reason=f"数据集 {params[key]} 不在允许范围",
                    )

        # 3. 风险确认：高风险 / 工具显式要求确认
        if risk_requires_confirmation(tool.risk_level):
            return PermissionDecision(
                Decision.REQUIRE_CONFIRMATION,
                reason=f"工具 {tool.name} 风险等级为 {risk_value}（高风险），需要确认后执行",
            )
        if tool.requires_confirmation:
            return PermissionDecision(
                Decision.REQUIRE_CONFIRMATION,
                reason=f"工具 {tool.name} 会改动数据或执行编排流程，需要确认后执行",
            )

        return PermissionDecision(Decision.ALLOW, reason="allowed")
