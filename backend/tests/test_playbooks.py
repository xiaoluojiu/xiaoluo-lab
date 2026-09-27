"""Playbook：意图 → 固定工具链。

旧架构靠模型现场编排工具顺序，每一步都要把工具 schema 发过去。新架构把链条
写死，代价是失去灵活性，收益是**可预测 + 零 Token**。这个文件保证写死的链条
不会指向不存在的工具 —— 那是重构期最容易留下来的一类哑弹。
"""

from __future__ import annotations

import pytest

from app.agent.intents import Intent
from app.agent.playbooks import PLAYBOOKS, Playbook, get_playbook


def test_every_non_chat_intent_has_playbook():
    for intent in Intent:
        if intent is Intent.CHAT:
            continue
        assert intent in PLAYBOOKS, f"意图 {intent.value} 缺少 playbook"


def test_get_playbook_never_returns_none():
    """调用方不做 None 判断，所以这里必须保证永远有值（含 CHAT）。"""
    for intent in Intent:
        playbook = get_playbook(intent)
        assert isinstance(playbook, Playbook)


def test_playbook_steps_are_bounded():
    """链条长度必须远小于 MAX_STEPS，否则「固定链条」失去意义。"""
    from app.agent.engine import MAX_STEPS

    for intent, playbook in PLAYBOOKS.items():
        assert len(playbook.steps) <= MAX_STEPS, f"{intent.value} 步数超出上限"


def test_all_playbook_tools_exist_in_registry():
    """链条里引用的工具必须在注册表中真实存在。"""
    from app.tools.registry import TOOL_REGISTRY

    missing: list[str] = []
    for intent, playbook in PLAYBOOKS.items():
        for step in playbook.steps:
            for name in (step.tool, step.fallback_tool):
                if not name:
                    continue
                try:
                    TOOL_REGISTRY.get(name)
                except Exception:  # noqa: BLE001
                    missing.append(f"{intent.value}->{name}")
    assert not missing, f"playbook 引用了不存在的工具：{missing}"


def test_step_titles_are_readable():
    """title 直接显示在前端时间线上，不能是空串或工具名原样复制。"""
    for intent, playbook in PLAYBOOKS.items():
        for step in playbook.steps:
            assert step.title, f"{intent.value} 的步骤缺少标题"


def test_ml_train_carries_target_to_next_step():
    """目标列由 detect_task 推断后必须传给 train，否则每步都要重新问一遍。"""
    steps = get_playbook(Intent.ML_TRAIN).steps
    train_step = next(s for s in steps if s.tool == "ml.train")
    assert "target" in (train_step.carry or {})


def test_required_slots_match_known_questions():
    """需要反问的槽位必须有中文问法，否则前端会弹出一个 code 而不是问题。"""
    from app.agent.playbooks import SLOT_QUESTIONS

    for intent, playbook in PLAYBOOKS.items():
        for step in playbook.steps:
            for slot in step.required_slots or ():
                assert slot in SLOT_QUESTIONS, f"{intent.value} 的槽位 {slot} 缺少问法"


def test_visualize_chart_slot_can_override_its_default():
    """用户点名「画成柱状图」时必须画柱状图。

    VISUALIZE 步骤带 ``defaults={"chart": "histogram"}``，而 chart 只挂在
    required_slots 上时，抽取结果**覆盖不了**一个非空默认值 —— 实测
    「把刚才的结果画成柱状图」画出来的是直方图，接着因为那一列不是数值列，
    整次运行以 ``could not convert string to float`` 失败。
    """
    step = get_playbook(Intent.VISUALIZE).steps[0]

    assert step.defaults.get("chart") == "histogram"
    assert "chart" in step.tuned_slots, "chart 必须是可微调槽位，否则默认值永远赢"


def test_visualize_chart_slot_can_override_its_default():
    """用户点名「画成柱状图」时必须画柱状图。

    VISUALIZE 步骤带 ``defaults={"chart": "histogram"}``，而 chart 只挂在
    required_slots 上时，抽取结果**覆盖不了**一个非空默认值 —— 实测
    「把刚才的结果画成柱状图」画出来的是直方图，接着因为那一列不是数值列，
    整次运行以 ``could not convert string to float`` 失败。
    """
    step = get_playbook(Intent.VISUALIZE).steps[0]

    assert step.defaults.get("chart") == "histogram"
    assert "chart" in step.tuned_slots, "chart 必须是可微调槽位，否则默认值永远赢"
