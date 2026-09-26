"""Prompt 127：Agent 结果验证。

验证维度：
1. Schema        —— 工具 output_schema 声明的必需字段是否存在
2. Tool Result   —— success 与 errors
3. Expected Output —— 步骤声明的期望输出关键词是否出现
   （提示性信号：期望输出是规划期 LLM 生成的自然语言描述，
   未命中只记 warning，不做硬性失败，避免误伤正常结果）
4. Data Quality  —— 数值中不得出现 NaN / Inf

失败（valid=False）时 Runtime 必须阻止该结果继续传播
（不进入最终答案、不作为后续步骤的依据）。
"""

from __future__ import annotations

import math
import re
from typing import Any

from app.agent.state import PendingAction
from app.agent.validator.models import ValidationResult
from app.tools.result import ToolResult


def _scan_bad_floats(value: Any, path: str = "$") -> list[str]:
    """递归扫描 NaN / Inf 数值。"""
    bad: list[str] = []
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            bad.append(path)
    elif isinstance(value, dict):
        for k, v in value.items():
            bad.extend(_scan_bad_floats(v, f"{path}.{k}"))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bad.extend(_scan_bad_floats(v, f"{path}[{i}]"))
    return bad


_EXPECTED_SPLIT = re.compile(r"[\s,，。：:；;、！!？?·…/|()（）\[\]【】\"'“”‘’]+")


def _expected_keywords(expected_output: str) -> list[str]:
    """把期望输出描述切成关键词（兼容中文标点；丢弃单字噪声）。"""
    return [w for w in _EXPECTED_SPLIT.split(expected_output) if len(w) >= 2]


class AgentResultValidator:
    """Agent 结果验证器。"""

    def validate(
        self,
        step: PendingAction,
        result: ToolResult,
        output_schema: dict[str, Any] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {"tool": step.tool}

        # 1. Tool Result 基础状态
        if not result.success:
            errors.extend(result.errors or ["工具执行返回失败"])
        checks["tool_success"] = result.success

        # 2. NaN / Inf（数据质量底线：指标永远不能带 NaN/Inf）
        bad_paths = _scan_bad_floats(result.data)
        if bad_paths:
            errors.append(f"结果包含 NaN/Inf：{bad_paths[:5]}")
        checks["nan_inf_paths"] = bad_paths

        # 3. Schema 必需字段
        schema = output_schema or {}
        if schema.get("type") == "object" and isinstance(result.data, dict):
            required = schema.get("required", []) or []
            missing = [k for k in required if k not in result.data]
            if missing:
                errors.append(f"输出缺少必需字段：{missing}")
            checks["schema_required_missing"] = missing

        # 4. Expected Output：期望输出是规划期 LLM 生成的自然语言描述，
        #    命中与否只记入 evidence 与 warnings，不做硬性失败——
        #    结果可信性由工具成功状态、Schema、无 NaN/Inf 兜底。
        if step.expected_output:
            keywords = _expected_keywords(step.expected_output)
            haystack = result.summary + " " + str(result.data)
            hits = [w for w in keywords if w in haystack]
            checks["expected_hits"] = hits
            if keywords and not hits:
                checks["expected_missed"] = True
                warnings.append(
                    f"期望输出未命中（提示，不阻断）：expected={step.expected_output!r} "
                    f"summary={result.summary[:60]!r}"
                )

        evidence = {"checks": checks, "summary": result.summary[:200]}
        if errors:
            return ValidationResult.fail(errors, evidence=evidence, warnings=warnings)
        return ValidationResult.ok(evidence=evidence, warnings=warnings)
