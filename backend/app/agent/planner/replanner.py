"""Agent Replanner：只在可恢复错误上重试，不跳过会破坏依赖链的步骤。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.planner.models import AgentPlan
from app.core.exceptions import AgentException


class AgentLimitExceeded(AgentException):
    """Agent 超出执行限制（步数 / 调用次数 / 超时）。"""
    http_status = 400
    default_code = "AGENT_LIMIT_EXCEEDED"
    default_message = "Agent limit exceeded"


@dataclass
class ReplanLimits:
    max_steps: int = 6
    max_tool_calls: int = 12
    timeout_seconds: float = 180.0
    # ★ 一次运行允许的重规划次数上限（2026-09-22 r-22 P0 的死循环总闸）。
    # 单步尝试上限（MAX_ATTEMPTS_PER_STEP）只在「尝试次数被正确记账」时才拦得住循环；
    # 一旦记账漏掉（历史上就是这样），唯一的兜底就是这个计数。
    # 24 对「≤12 步 × 每步 ≤2 次」的计划足够宽松，又能在失控时果断停下。
    max_replans: int = 24


MAX_ATTEMPTS_PER_STEP = 2
_DEPENDENCY_RE = re.compile(r"\{\{\s*(?:steps?\.)?step(\d+)\.")


def analyze_failure(errors: list[str]) -> str:
    text = " ".join(errors)
    lower = text.lower()
    if "依赖" in text or "第 " in text and "步输出" in text:
        return "前置步骤没有产生后续步骤所需的结构化输出"
    if "需要确认" in text or "confirmation" in lower:
        return "需要用户确认（高风险操作）"
    if "权限" in text or "denied" in lower or "permission" in lower:
        return "权限不足，无法执行该工具"
    if "不存在" in text or "not found" in lower or "not_found" in lower:
        return "引用了不存在的对象（数据集 / 列 / 工具）"
    if "dataset_id" in lower or "参数" in text or "缺少必需字段" in text or "parameter" in lower or "required" in lower:
        return "工具参数错误，不应盲目重复执行"
    if "NaN" in text or "Inf" in text:
        return "结果包含非法数值（NaN/Inf）"
    if "期望输出未命中" in text:
        return "结果与期望输出不符"
    return "工具执行失败（可能为瞬时错误）"


def _is_parameter_error(errors: list[str]) -> bool:
    text = " ".join(errors).lower()
    markers = ("dataset_id", "参数", "缺少必需字段", "parameter", "required")
    return any(marker in text for marker in markers)


def _step_depends_on(step, failed_step_abs_index: int) -> bool:
    """判断后续步骤是否显式消费失败步骤的输出。

    参数必须是**绝对**下标（0 基）。`{{stepN}}` 里的 N 是 1 基绝对步号，
    所以比对的是 `failed_step_abs_index + 1`。
    """
    needle = failed_step_abs_index + 1
    raw = str(step.arguments or {})
    return any(int(match.group(1)) == needle for match in _DEPENDENCY_RE.finditer(raw))


class Replanner:
    """重规划器：错误可恢复才重试；依赖链断裂直接终止，禁止制造级联错误。"""

    def __init__(self, limits: ReplanLimits | None = None) -> None:
        self.limits = limits or ReplanLimits()

    def replan(self, plan: AgentPlan, *, failed_step_index: int, errors: list[str], attempts: int, base_index: int = 0) -> AgentPlan:
        """返回重规划后的计划。

        `base_index` ＝ 本计划在当前运行里的**绝对起始下标**（`_run_plan` 的 `offset`）。
        `{{stepN}}` 里写的是**绝对**步号（1 基），而 `failed_step_index` 是**相对本切片**的下标；
        `resume()` 会把计划切成 `plan.steps[step_index:]` 再继续，此时两者不再相等。
        不把它传进来 ⇒ 依赖判定永远对不上 ⇒ 「上游失败、下游等它的输出」这种情况
        不会被识别成依赖链断裂，反被当成瞬时错误反复重试（r-22 死循环的成因之一）。
        """
        cause = analyze_failure(errors)
        failed_abs = base_index + failed_step_index
        remaining = list(plan.steps[failed_step_index + 1 :])

        # 关键安全规则：workflow.create 失败时，不得继续 workflow.run；
        # dataset.prepare 失败时，也不能让后续依赖其产物的步骤继续执行。
        if any(_step_depends_on(step, failed_abs) for step in remaining):
            return AgentPlan(
                goal=plan.goal,
                steps=[],
                notes=f"第 {failed_abs} 步失败且存在依赖它输出的后续步骤（{cause}），为避免级联错误，停止执行。",
            )

        if _is_parameter_error(errors):
            notes = f"第 {failed_step_index} 步参数校验失败（{cause}），不重复执行相同参数。"
            if not remaining:
                notes += "已无剩余步骤，任务失败。"
            return AgentPlan(goal=plan.goal, steps=remaining, notes=notes)

        if attempts < MAX_ATTEMPTS_PER_STEP:
            return AgentPlan(
                goal=plan.goal,
                steps=list(plan.steps[failed_step_index:]),
                notes=f"第 {failed_step_index} 步失败（{cause}），第 {attempts + 1} 次尝试。",
                retry=True,
            )

        notes = f"第 {failed_step_index} 步重试 {attempts} 次后仍失败（{cause}），跳过。"
        if not remaining:
            notes += "已无剩余步骤，任务失败。"
        if len(remaining) > self.limits.max_steps:
            raise AgentLimitExceeded(f"剩余步数 {len(remaining)} 超过上限 {self.limits.max_steps}")
        return AgentPlan(goal=plan.goal, steps=remaining, notes=notes)

    def assert_replan_budget(self, replans: int) -> None:
        """重规划次数熔断 —— 与单步尝试上限互补的最后一道闸。

        单步上限靠「调用方正确记账」才生效，属于**协作式**约束；这条是**结构性**的：
        只要 `_run_plan` 每次重规划都 +1 并在这里校验，任何原因导致的循环都会被终止。
        """
        if replans >= self.limits.max_replans:
            raise AgentLimitExceeded(f"重规划次数 {replans} 达到上限 {self.limits.max_replans}，判定为不可恢复的循环")

    def assert_limits(self, *, tool_calls: int, elapsed_seconds: float, total_tokens: int | None = None, max_total_tokens: int | None = None) -> None:
        if tool_calls > self.limits.max_tool_calls:
            raise AgentLimitExceeded(f"工具调用次数 {tool_calls} 超过上限 {self.limits.max_tool_calls}")
        if elapsed_seconds > self.limits.timeout_seconds:
            raise AgentLimitExceeded(f"运行时长 {elapsed_seconds:.0f}s 超过上限 {self.limits.timeout_seconds:.0f}s")
        # Token 预算纳入重规划熔断：避免重试循环在 LLM 层持续烧 token。
        if max_total_tokens and total_tokens is not None and total_tokens >= int(max_total_tokens):
            raise AgentLimitExceeded(f"Token 用量 {total_tokens} 已达到本次任务预算上限 {int(max_total_tokens)}")
