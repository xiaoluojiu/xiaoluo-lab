"""Prompt 126：ValidationResult。

Agent 结果验证的统一返回：valid / errors / warnings / evidence。
失败时（valid=False）结果不得进入最终答案 —— 由 Runtime 强制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """验证结果。"""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        *,
        evidence: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> ValidationResult:
        return cls(valid=True, errors=[], warnings=warnings or [], evidence=evidence or {})

    @classmethod
    def fail(
        cls,
        errors: list[str] | str,
        *,
        evidence: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> ValidationResult:
        if isinstance(errors, str):
            errors = [errors]
        return cls(valid=False, errors=errors, warnings=warnings or [], evidence=evidence or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "evidence": self.evidence,
        }
