"""统一业务异常。"""

from __future__ import annotations

from typing import Any


class AppException(Exception):
    """业务异常基类。"""

    http_status = 500
    default_code = "INTERNAL_ERROR"
    default_message = "Internal Server Error"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: Any = None,
    ) -> None:
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationException(AppException):
    http_status = 422
    default_code = "VALIDATION_ERROR"
    default_message = "Validation failed"


class NotFoundException(AppException):
    http_status = 404
    default_code = "NOT_FOUND"
    default_message = "Resource not found"


class PermissionDeniedException(AppException):
    http_status = 403
    default_code = "PERMISSION_DENIED"
    default_message = "Permission denied"


class ToolExecutionException(AppException):
    http_status = 500
    default_code = "TOOL_EXECUTION_ERROR"
    default_message = "Tool execution failed"


class DatasetException(AppException):
    http_status = 400
    default_code = "DATASET_ERROR"
    default_message = "Dataset error"


class StorageException(AppException):
    http_status = 500
    default_code = "STORAGE_ERROR"
    default_message = "Storage error"


class WorkflowException(AppException):
    http_status = 500
    default_code = "WORKFLOW_ERROR"
    default_message = "Workflow error"


class AgentException(AppException):
    http_status = 500
    default_code = "AGENT_ERROR"
    default_message = "Agent error"
