"""Prompt 088：Tool 抽象。

每个 Tool 必须声明：
- name / description / category
- input_schema / output_schema（JSON Schema 风格 dict）
- permission（所需权限）/ risk_level（风险等级）/ requires_confirmation

核心方法 execute()：只做业务动作；权限校验一律由 ToolRegistry + PermissionManager
在注册表入口统一执行（Tool 不能绕过 Registry 执行）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.agent.permission.models import Permission
from app.agent.permission.rules import RiskLevel
from app.core.exceptions import AppException
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult


class ToolPermissionError(AppException):
    """权限不足（PermissionManager 判定 DENY）。"""

    http_status = 403
    default_code = "TOOL_PERMISSION_DENIED"
    default_message = "Tool permission denied"


class ToolConfirmationRequired(AppException):
    """需要用户确认后才能执行（高风险操作）。"""

    http_status = 202
    default_code = "TOOL_CONFIRMATION_REQUIRED"
    default_message = "Tool requires user confirmation"


@dataclass
class ToolServices:
    """工具依赖的服务集合（按需注入，未注入的服务工具不应使用）。"""

    dataset_service: Any | None = None  # DatasetService
    data_engine_service: Any | None = None  # DataEngineService
    experiment_service: Any | None = None  # ExperimentService
    connector_service: Any | None = None  # ConnectorService（外部数据库接入）
    db: Any | None = None  # SQLAlchemy Session（版本化操作需要）

    def require(self, name: str) -> Any:
        service = getattr(self, name)
        if service is None:
            raise AppException(
                f"工具所需服务 {name} 未注入",
                code="TOOL_SERVICE_MISSING",
            )
        return service


class Tool(ABC):
    """Agent 工具抽象基类。"""

    name: str = ""
    description: str = ""
    category: str = "general"
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    permission: Permission = Permission.READ_DATA
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False

    @abstractmethod
    def execute(
        self,
        params: dict[str, Any],
        context: ToolExecutionContext,
        services: ToolServices,
    ) -> ToolResult:
        """执行工具（入参 / 上下文 / 服务集合）。"""

    # ---- 共用：数据集访问范围校验 ----
    def assert_dataset_access(
        self, context: ToolExecutionContext, dataset_id: int
    ) -> None:
        if not context.can_access_dataset(dataset_id):
            raise ToolPermissionError(
                "无权访问该数据集",
                details={"dataset_id": dataset_id, "allowed": sorted(context.dataset_ids)},
            )

    def describe(self) -> dict[str, Any]:
        """工具自描述（供 Agent 选择工具 / LLM function calling）。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permission": str(self.permission),
            "risk_level": str(self.risk_level),
            "requires_confirmation": self.requires_confirmation,
        }
