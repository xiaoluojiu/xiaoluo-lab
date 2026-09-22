"""Prompt 123：PlanStep / AgentPlan。

计划是纯数据结构：goal + steps。
每个 PlanStep 只声明：执行哪个 tool、参数、期望输出、所需权限。
Plan 不携带任何执行能力 —— 执行只能发生在 AgentExecutor。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 工具名规范：category.action（小写字母数字下划线 + 一个点）
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class PlanStep(BaseModel):
    """计划中的一步。"""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    permission: str = ""

    @field_validator("tool")
    @classmethod
    def _tool_name_valid(cls, value: str) -> str:
        value = value.strip()
        if not TOOL_NAME_RE.match(value):
            raise ValueError(f"非法工具名 {value!r}（必须形如 category.action）")
        return value

    @field_validator("arguments")
    @classmethod
    def _arguments_is_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("arguments 必须是对象")
        # 禁止把可执行代码塞进参数（防御式：参数只能是数据）
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("arguments 的键必须是字符串")
            if callable(item) or isinstance(item, (type,)):
                raise ValueError(f"参数 {key!r} 不能是可执行对象")
        return value


class AgentPlan(BaseModel):
    """Agent 计划。"""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    notes: str = ""  # Replanner 分析备注等
    retry: bool = False  # Replanner 显式标记：本计划首步是否为失败步骤的原样重试
    cache_hit: bool = False  # Planner 标记：本计划是否来自 plan cache（不参与 LLM 输出解析）

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("goal 不能为空")
        return value.strip()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
