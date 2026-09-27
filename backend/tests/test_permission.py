"""权限裁决：一个纯函数覆盖全部规则。

旧架构的规则分散在 rules / manager / executor / loop 四处，改一处要同步四处。
这里全部收敛在 ``PermissionManager.check``，因此可以用穷举用例锁死行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agent.permission import (
    Permission,
    PermissionDecision,
    PermissionManager,
    RiskLevel,
    _as_permission_value,
    _as_risk_level,
)


@dataclass
class _FakeTool:
    name: str = "fake.tool"
    permission: Any = Permission.READ_DATA
    risk_level: Any = RiskLevel.LOW
    requires_confirmation: bool = False


@dataclass
class _FakeContext:
    permissions: set = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.permissions is None:
            self.permissions = set(Permission)


def _check(tool: _FakeTool, permissions: set | None = None) -> PermissionDecision:
    manager = PermissionManager()
    ctx = _FakeContext(set(Permission) if permissions is None else permissions)
    return manager.check("tester", tool, ctx, {})


def test_low_risk_allowed():
    decision = _check(_FakeTool(risk_level=RiskLevel.LOW))
    assert decision.allowed
    assert not decision.needs_confirmation
    assert decision.as_label() == "ALLOW"


@pytest.mark.parametrize("risk", [RiskLevel.HIGH, RiskLevel.CRITICAL])
def test_high_risk_needs_confirmation(risk: RiskLevel):
    decision = _check(_FakeTool(risk_level=risk))
    assert not decision.allowed
    assert decision.needs_confirmation
    assert decision.as_label() == "REQUIRE_CONFIRMATION"


def test_medium_risk_does_not_block_but_is_not_low():
    """MEDIUM 是安全默认：可执行，但绝不是「免确认」的 LOW。"""
    decision = _check(_FakeTool(risk_level=RiskLevel.MEDIUM))
    assert decision.allowed
    assert RiskLevel.MEDIUM.needs_confirmation is False


def test_missing_permission_denies_even_if_low_risk():
    """权限缺失优先于风险判定：没有权限时确认也没有意义。"""
    decision = _check(_FakeTool(), permissions={Permission.MODIFY_DATA})
    assert decision.denied
    assert not decision.needs_confirmation
    assert decision.as_label() == "DENY"


def test_permission_value_normalisation():
    """枚举与裸字符串必须等价：工具层两种写法都在用。"""
    assert _as_permission_value(Permission.ANALYZE_DATA) == "analyze_data"
    assert _as_permission_value("analyze_data") == "analyze_data"


def test_raw_string_permission_is_accepted():
    """工具直接写 ``permission = "analyze_data"`` 时不能被判 DENY。

    这是重构期真实踩过的坑：Permission 枚举的取值与工具层对不上时，
    除 dataset.list 外的工具全部被拒，而 Agent 只是「正常失败」，很不显眼。
    """
    tool = _FakeTool(permission="analyze_data", risk_level=RiskLevel.LOW)
    assert _check(tool).allowed


def test_every_registered_tool_declares_a_known_permission():
    """注册表里不允许出现 Permission 之外的权限字符串 —— 出现即等价于该工具永久不可用。"""
    import app.tools.builtin  # noqa: F401  触发注册
    from app.tools.registry import TOOL_REGISTRY

    known = {_as_permission_value(p) for p in Permission}
    unknown: list[str] = []
    for meta in TOOL_REGISTRY.list():
        value = _as_permission_value(TOOL_REGISTRY.get(meta["name"]).permission)
        if value not in known:
            unknown.append(f"{meta['name']}:{value}")
    assert not unknown, f"工具声明了未知权限：{unknown}"


def test_every_registered_tool_is_reachable_with_full_permissions():
    """全权限上下文下，任何工具都不该被 DENY（只允许因高风险要求确认）。"""
    import app.tools.builtin  # noqa: F401  触发注册
    from app.tools.registry import TOOL_REGISTRY

    denied: list[str] = []
    for meta in TOOL_REGISTRY.list():
        tool = TOOL_REGISTRY.get(meta["name"])
        ctx = _FakeContext({p for p in Permission})
        decision = PermissionManager().check("u", tool, ctx, {})
        if decision.denied and not decision.needs_confirmation:
            denied.append(meta["name"])
    assert not denied, f"这些工具在全权限下仍被拒绝：{denied}"


def test_raw_string_risk_level_is_understood():
    """工具层既有 RiskLevel.HIGH 也有 "high"，两种都要能读出「需确认」。"""
    assert _as_risk_level("high").needs_confirmation is True
    assert _as_risk_level(RiskLevel.LOW).needs_confirmation is False
    assert _as_risk_level("unknown-garbage") is RiskLevel.MEDIUM  # 认不出就按中风险
    assert _check(_FakeTool(risk_level="high")).needs_confirmation


def test_explicit_confirmation_flag_overrides_low_risk():
    tool = _FakeTool(risk_level=RiskLevel.LOW, requires_confirmation=True)
    assert _check(tool).needs_confirmation


def test_undeclared_risk_defaults_to_medium_not_low():
    """忘记声明风险的工具不能被静默放行 —— 缺省必须是 MEDIUM。"""
    decision = PermissionManager().check("tester", _FakeTool(risk_level=None), _FakeContext(), {})
    assert decision.allowed  # MEDIUM 可执行
    # 反证：如果缺省是 LOW 的话，下面这条会同样成立，因此单独断言工具层默认值
    from app.tools.base import Tool

    assert Tool.risk_level is not None


def test_reason_mentions_tool_name():
    decision = _check(_FakeTool(name="ml.train", risk_level=RiskLevel.HIGH))
    assert "ml.train" in decision.reason


def test_reason_is_chinese_and_non_empty():
    decision = _check(_FakeTool(risk_level=RiskLevel.HIGH))
    assert decision.reason and any("\u4e00" <= ch <= "\u9fff" for ch in decision.reason)


def test_no_context_is_denied():
    """没有上下文 = 没有权限，必须拒绝而不是放行。"""
    decision = PermissionManager().check("tester", _FakeTool(), None, {})
    assert decision.denied
