"""统一的工具结果结构与低 Token 摘要能力。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _compact_value(value: Any, *, max_items: int = 20, max_chars: int = 4000) -> Any:
    """把大型工具结果压缩为适合进入 LLM 上下文的表示。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "…"
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        result = {str(k): _compact_value(v, max_items=max_items, max_chars=max_chars // 2) for k, v in items}
        if len(value) > max_items:
            result["_truncated"] = True
            result["_total_keys"] = len(value)
        return result
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        result = [_compact_value(v, max_items=max_items, max_chars=max_chars // 2) for v in values[:max_items]]
        if len(values) > max_items:
            result.append({"_truncated": True, "_total_items": len(values)})
        return result
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _rough_tokens(value: Any) -> int:
    """无第三方 tokenizer 依赖的保守估算；真实 provider usage 仍以 API 返回为准。"""
    text = str(value)
    if not text:
        return 0
    # 中文通常接近字符级 token，英文/数字则更接近 4 字符/token；取较保守估计。
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    non_cjk = len(text) - cjk
    return max(1, cjk + (non_cjk + 3) // 4)


@dataclass
class ToolResult:
    """工具执行结果。data 保留原始结果，compact_data 用于 LLM 上下文。

    ``signals``（任务 7）：结构化、可解析的下一步信号，例如 ``high_missing`` /
    ``high_skew`` / ``outliers_detected`` / ``strong_correlation`` /
    ``target_missing`` / ``classification_candidate`` / ``regression_candidate``。
    它们不是自然语言，是 DecisionProvider 能直接消费的标记 —— 让
    Observe→Decide→Act 的「下一步」由数据驱动，而不是靠关键词硬猜。
    这些 signals 未来可直接成为训练数据。
    """

    success: bool
    data: Any = None
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    compact_data: Any = None
    signals: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: Any = None, summary: str = "", *, warnings: list[str] | None = None, metadata: dict[str, Any] | None = None, compact_data: Any = None, signals: list[str] | None = None) -> ToolResult:
        return cls(success=True, data=data, summary=summary, warnings=warnings or [], metadata=metadata or {}, compact_data=compact_data, signals=signals or [])

    @classmethod
    def fail(cls, errors: list[str] | str, *, data: Any = None, metadata: dict[str, Any] | None = None, compact_data: Any = None) -> ToolResult:
        if isinstance(errors, str):
            errors = [errors]
        return cls(success=False, data=data, errors=errors, metadata=metadata or {}, compact_data=compact_data)

    def for_llm(self, *, max_items: int = 20, max_chars: int = 4000) -> dict[str, Any]:
        """返回轻量结果给 LLM；不会修改或丢失原始 data。"""
        compact = self.compact_data
        if compact is None:
            compact = _compact_value(self.data, max_items=max_items, max_chars=max_chars)
        original_tokens = _rough_tokens(self.data)
        compact_tokens = _rough_tokens(compact)
        return {
            "success": self.success,
            "summary": self.summary,
            "data": compact,
            "warnings": self.warnings[:10],
            "errors": self.errors[:10],
            "metadata": self.metadata,
            "signals": list(self.signals),
            "_token": {
                "estimated_original": original_tokens,
                "estimated_compact": compact_tokens,
                "estimated_saved": max(0, original_tokens - compact_tokens),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """完整结果，用于 API/运行记录；保持原有接口兼容。"""
        return {"success": self.success, "data": self.data, "summary": self.summary, "warnings": self.warnings, "errors": self.errors, "metadata": self.metadata, "signals": list(self.signals)}
