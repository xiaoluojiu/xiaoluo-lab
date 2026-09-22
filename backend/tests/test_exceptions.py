"""Prompt 007 测试：统一异常体系。"""

from __future__ import annotations

import pytest
from app.core.exceptions import (
    AgentException,
    AppException,
    DatasetException,
    NotFoundException,
    PermissionDeniedException,
    StorageException,
    ToolExecutionException,
    ValidationException,
    WorkflowException,
)


@pytest.mark.parametrize(
    "exc_cls,status,code",
    [
        (AppException, 500, "INTERNAL_ERROR"),
        (ValidationException, 422, "VALIDATION_ERROR"),
        (NotFoundException, 404, "NOT_FOUND"),
        (PermissionDeniedException, 403, "PERMISSION_DENIED"),
        (ToolExecutionException, 500, "TOOL_EXECUTION_ERROR"),
        (DatasetException, 400, "DATASET_ERROR"),
        (StorageException, 500, "STORAGE_ERROR"),
        (WorkflowException, 500, "WORKFLOW_ERROR"),
        (AgentException, 500, "AGENT_ERROR"),
    ],
)
def test_exception_defaults(exc_cls, status, code):
    exc = exc_cls()
    assert exc.http_status == status
    assert exc.code == code
    assert exc.message
    assert exc.details is None


def test_custom_fields():
    exc = NotFoundException("dataset missing", details={"dataset_id": 42})
    assert exc.message == "dataset missing"
    assert exc.details == {"dataset_id": 42}
    assert exc.to_dict() == {
        "code": "NOT_FOUND",
        "message": "dataset missing",
        "details": {"dataset_id": 42},
    }
