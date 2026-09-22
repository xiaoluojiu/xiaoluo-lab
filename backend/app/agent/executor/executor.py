"""AgentExecutor：PlanStep → Permission → ToolRegistry → Tool → Result。"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.planner.models import PlanStep
from app.tools.base import ToolConfirmationRequired, ToolPermissionError, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.registry import TOOL_REGISTRY, ToolRegistry
from app.tools.result import ToolResult

logger = logging.getLogger(__name__)


class ToolCallRecord:
    """一次工具调用的完整记录。"""
    def __init__(self, *, step_index: int, tool: str, arguments: dict[str, Any], attempt: int = 1) -> None:
        self.step_index = step_index
        self.tool = tool
        self.arguments = arguments
        self.attempt = attempt
        self.status = "pending"
        self.result: ToolResult | None = None
        self.error: str = ""
        self.started_at = time.time()
        self.finished_at: float | None = None

    def finish(self, status: str, *, result: ToolResult | None = None, error: str = "") -> None:
        self.status, self.result, self.error, self.finished_at = status, result, error, time.time()

    @property
    def elapsed_ms(self) -> float:
        return round(((self.finished_at or time.time()) - self.started_at) * 1000, 2)

    def to_llm_dict(self, *, max_items: int = 20, max_chars: int = 4000) -> dict[str, Any]:
        return {"step_index": self.step_index, "tool": self.tool, "status": self.status, "attempt": self.attempt, "result": self.result.for_llm(max_items=max_items, max_chars=max_chars) if self.result else None, "error": self.error, "elapsed_ms": self.elapsed_ms}

    def to_dict(self) -> dict[str, Any]:
        return {"step_index": self.step_index, "tool": self.tool, "arguments": self.arguments, "attempt": self.attempt, "status": self.status, "result": self.result.to_dict() if self.result else None, "error": self.error, "elapsed_ms": self.elapsed_ms}


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """轻量 JSON-Schema 子集校验，无第三方依赖；覆盖 Agent 工具最常见契约错误。"""
    if not schema:
        return []
    errors: list[str] = []
    typ = schema.get("type")
    type_ok = True
    if typ == "object": type_ok = isinstance(value, dict)
    elif typ == "array": type_ok = isinstance(value, list)
    elif typ == "string": type_ok = isinstance(value, str)
    elif typ == "integer": type_ok = isinstance(value, int) and not isinstance(value, bool)
    elif typ == "number": type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif typ == "boolean": type_ok = isinstance(value, bool)
    if typ and not type_ok:
        return [f"{path} 类型错误：期望 {typ}，实际 {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} 不在允许值范围：{schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value or value[key] in (None, ""):
                errors.append(f"{path}.{key} 缺少必要参数")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}.{key} 不是工具支持的参数")
        for key, sub in props.items():
            if key in value:
                errors.extend(_schema_errors(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]): errors.append(f"{path} 至少需要 {schema['minItems']} 项")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]): errors.append(f"{path} 最多允许 {schema['maxItems']} 项")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value[:100]): errors.extend(_schema_errors(item, item_schema, f"{path}[{i}]"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]): errors.append(f"{path} 长度不足")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]): errors.append(f"{path} 长度超过限制")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]: errors.append(f"{path} 小于最小值 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]: errors.append(f"{path} 大于最大值 {schema['maximum']}")
    return errors


class AgentExecutor:
    """计划步骤执行器（强制经过 ToolRegistry）。"""
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or TOOL_REGISTRY

    def execute_step(self, step: PlanStep, context: ToolExecutionContext, services: ToolServices, *, confirmed: bool = False, attempt: int = 1, step_index: int = 0) -> ToolCallRecord:
        record = ToolCallRecord(step_index=step_index, tool=step.tool, arguments=dict(step.arguments), attempt=attempt)
        try:
            tool = self.registry.get(step.tool)
            errors = _schema_errors(step.arguments, tool.input_schema or {})
            if errors:
                record.finish("failed", error="工具参数预检失败：" + "；".join(errors[:8]))
                return record
            result = self.registry.execute(step.tool, step.arguments, context, services, confirmed=confirmed)
        except ToolConfirmationRequired as exc:
            record.finish("needs_confirmation", error=exc.message)
        except ToolPermissionError as exc:
            record.finish("denied", error=exc.message)
        except Exception as exc:
            # 记录完整 traceback 到日志，便于溯源（此前只 str(exc) 塞进 record.error）。
            logger.exception(
                "执行步骤 %s 时工具 %s 抛出未捕获异常",
                step_index, step.tool,
            )
            record.finish("failed", error=str(exc))
        else:
            record.finish("ok" if result.success else "failed", result=result)
        return record
