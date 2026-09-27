"""Agent 存储：三条防线（事件上限 / 原子写 / 损坏自愈）。

这三条对应旧架构真实踩过的坑：store 文件涨到 15 MB（内含约 20 万条
replanning 事件）、半截 JSON 让后端起不来。
"""

from __future__ import annotations

import json

import pytest

from app.agent.models import EventType, RunStatus
from app.agent.store import MAX_EVENTS_PER_RUN, AgentStore


def test_create_and_get_session(store: AgentStore):
    session = store.create_session(user_id="u1", title="标题")
    assert store.get_session(session.id) is not None
    assert store.get_session(session.id).title == "标题"


def test_list_sessions_filters_by_user_and_archived(store: AgentStore):
    store.create_session(user_id="u1")
    other = store.create_session(user_id="u2")
    other.archived = True
    store.update_session(other)

    assert len(store.list_sessions(user_id="u1")) == 1
    assert len(store.list_sessions(user_id="u2")) == 0
    assert len(store.list_sessions(user_id="u2", include_archived=True)) == 1


def test_run_is_bound_to_session(store: AgentStore, session):
    run = store.create_run(session.id, "看看质量")
    assert store.get_run(run.id).session_id == session.id
    assert run.id in store.get_session(session.id).run_ids


def test_add_event_assigns_increasing_seq_and_run_id(store: AgentStore, session):
    run = store.create_run(session.id, "x")
    first = store.add_event(run, EventType.ROUTE)
    second = store.add_event(run, EventType.PLANNING)
    assert first.seq == 1 and second.seq == 2
    # run_id 缺失会让前端的确认按钮点不动，必须逐条写死
    assert first.run_id == run.id and second.run_id == run.id


def test_events_are_capped(store: AgentStore, session):
    """事件上限是防御性的：任何新增的循环都不许把磁盘写满。"""
    run = store.create_run(session.id, "x")
    for _ in range(MAX_EVENTS_PER_RUN + 50):
        store.add_event(run, EventType.TOOL_CALL)
    assert len(run.events) == MAX_EVENTS_PER_RUN
    # 保留最近的，seq 必须仍然递增
    assert run.events[-1].seq > run.events[0].seq


def test_delete_session_rejects_when_run_active(store: AgentStore, session):
    run = store.create_run(session.id, "x")
    run.status = RunStatus.RUNNING
    store.update_run(run)

    ok, reason = store.delete_session(session.id)
    assert ok is False and reason
    assert store.get_session(session.id) is not None


def test_delete_session_ok_when_all_terminal(store: AgentStore, session):
    run = store.create_run(session.id, "x")
    run.status = RunStatus.COMPLETED
    store.update_run(run)

    ok, _ = store.delete_session(session.id)
    assert ok is True
    assert store.get_session(session.id) is None
    assert store.get_run(run.id) is None


@pytest.mark.parametrize("status", [RunStatus.WAITING_CONFIRMATION, RunStatus.WAITING_CLARIFICATION])
def test_delete_session_ok_when_run_suspended(store: AgentStore, session, status):
    """挂起态不算「未结束」：引擎线程已在挂起点退出，拦下来就是永久死锁。

    历史缺陷：守卫写成 `not status.terminal`，把 waiting 也拦了 ⇒ 会话永远删不掉。
    """
    run = store.create_run(session.id, "x")
    run.status = status
    store.update_run(run)

    ok, reason = store.delete_session(session.id)
    assert ok is True, reason
    assert store.get_session(session.id) is None
    assert store.get_run(run.id) is None


def test_delete_missing_session(store: AgentStore):
    ok, reason = store.delete_session("s-nope")
    assert ok is False and "不存在" in reason


def test_history_is_capped(store: AgentStore, session):
    for i in range(260):
        store.append_history(session.id, "user", f"m{i}")
    assert len(store.get_session(session.id).history) <= 200


def test_persists_across_instances(tmp_path):
    """整存整取：重新打开 store 必须看到之前写的东西。"""
    path = tmp_path / "agent_store.json"
    first = AgentStore(path)
    session = first.create_session(user_id="u1", title="持久化")
    run = first.create_run(session.id, "看看质量")
    first.add_event(run, EventType.ROUTE)

    second = AgentStore(path)
    assert second.get_session(session.id).title == "持久化"
    assert len(second.get_run(run.id).events) == 1


def test_corrupt_file_self_heals(tmp_path):
    """坏文件不能让后端起不来：备份后以空存储启动。"""
    path = tmp_path / "agent_store.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")

    store = AgentStore(path)
    assert store.sessions == {} and store.runs == {}
    assert list(path.parent.glob("*.corrupt.*.json"))


def test_atomic_write_leaves_no_tmp(tmp_path):
    store = AgentStore(tmp_path / "agent_store.json")
    store.create_session()
    store.save()
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "agent_store.json").read_text(encoding="utf-8"))["version"] == 2


def test_listener_receives_events(store: AgentStore, session):
    run = store.create_run(session.id, "x")
    seen: list[EventType] = []
    store.add_listener(run.id, lambda e: seen.append(e.type))

    store.add_event(run, EventType.ROUTE)
    assert seen == [EventType.ROUTE]

    callback = lambda e: None  # noqa: E731
    store.add_listener(run.id, callback)
    store.remove_listener(run.id, callback)
    store.add_event(run, EventType.PLANNING)
    assert seen == [EventType.ROUTE, EventType.PLANNING]


def test_broken_listener_does_not_break_store(store: AgentStore, session):
    """监听者回调异常不得中断落盘 —— SSE 端写错了不能拖垮 Agent。"""
    run = store.create_run(session.id, "x")

    def boom(_event):
        raise RuntimeError("订阅方炸了")

    store.add_listener(run.id, boom)
    event = store.add_event(run, EventType.ROUTE)
    assert event.seq == 1
    assert store.get_run(run.id).events


def test_unknown_legacy_event_type_is_skipped_not_crashing(tmp_path):
    """旧 store 里有 preflight / replanning 事件：跳过，不能让启动失败。"""
    path = tmp_path / "agent_store.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [{"id": "s-1", "user_id": "u"}],
                "runs": [
                    {"id": "r-bad", "session_id": "s-1", "status": "completed",
                     "events": [{"seq": 1, "run_id": "r-bad", "type": "preflight"}]},
                    {"id": "r-ok", "session_id": "s-1", "status": "completed",
                     "events": [{"seq": 1, "run_id": "r-ok", "type": "route"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    store = AgentStore(path)
    assert store.get_run("r-ok") is not None
    assert store.get_run("r-bad") is None


def test_run_resume_state_survives_reload(tmp_path):
    """挂起中的运行必须完整保存恢复点：resume_step / authorized_key /
    clarification_answers / finished_at。

    真实缺陷：to_dict 漏序列化这四个字段，重启后确认/澄清恢复会从第 0 步重跑、
    已授权凭据失效、用户补充的答案消失，finished_at 丢失使 elapsed_seconds
    退化成 now - started_at（曾出现 2.5 小时这种荒谬数字）。
    """
    store = AgentStore(tmp_path / "agent_store.json")
    session = store.create_session()
    run = store.create_run(session.id, "训练一个模型")
    run.status = RunStatus.WAITING_CONFIRMATION
    run.resume_step = 1
    run.authorized_key = "1:ml.train"
    run.clarification_answers = {"target": "DepDelay"}
    run.finished_at = 12345.678
    store.update_run(run)

    reopened = AgentStore(store.path).get_run(run.id)
    assert reopened.resume_step == 1
    assert reopened.authorized_key == "1:ml.train"
    assert reopened.clarification_answers == {"target": "DepDelay"}
    assert reopened.finished_at == 12345.678


@pytest.mark.parametrize("status", list(RunStatus))
def test_status_roundtrip(tmp_path, status: RunStatus):
    store = AgentStore(tmp_path / f"{status.value}.json")
    session = store.create_session()
    run = store.create_run(session.id, "x")
    run.status = status
    store.update_run(run)

    reopened = AgentStore(store.path).get_run(run.id)
    assert reopened.status is status


def test_new_run_has_nonzero_budget(store: AgentStore):
    """预算必须在建 run 时写进去。

    用默认 TokenUsage() 时 max_llm_calls / max_total_tokens 恒为 0，
    前端 Token 面板就会显示「剩余 0 次」，而 /capabilities 明明写着上限 ——
    两处自相矛盾，用户只会以为额度已经用光了。
    """
    session = store.create_session()
    run = store.create_run(session.id, "看看质量")
    budget = run.token_usage.to_dict()["budget"]
    assert budget["max_llm_calls"] > 0
    assert budget["max_total_tokens"] > 0
    assert budget["remaining_llm_calls"] == budget["max_llm_calls"]


def test_budget_survives_reopen(store: AgentStore):
    """预算要能持久化：重开后不能退回 0。"""
    session = store.create_session()
    run = store.create_run(session.id, "看看质量")
    expected = run.token_usage.to_dict()["budget"]["max_llm_calls"]
    reopened = AgentStore(store.path).get_run(run.id)
    assert reopened.token_usage.to_dict()["budget"]["max_llm_calls"] == expected
