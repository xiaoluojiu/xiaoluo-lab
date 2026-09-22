from __future__ import annotations

import pytest

from app.agent.runtime.models import AgentRun, AgentSession, AgentStore, RunStatus


def test_session_context_is_explicit_and_persistent(tmp_path):
    store = AgentStore(tmp_path / "agent.json")
    session = store.add_session(AgentSession(id="s-1", user_id="u", dataset_ids=[1]))
    assert session.dataset_ids == [1]
    store.update_session_context("s-1", [3, 3, 2])
    assert store.get_session("s-1").dataset_ids == [3, 2]


def test_only_one_active_turn_per_session(tmp_path):
    store = AgentStore(tmp_path / "agent.json")
    session = store.add_session(AgentSession(id="s-1", user_id="u", dataset_ids=[3]))
    run = store.add_run(AgentRun(id="r-1", session_id=session.id, user_id="u", user_request="analyze"))
    assert run.status == RunStatus.PENDING
    with pytest.raises(ValueError, match="活动运行"):
        store.update_session_context(session.id, [4])
    with pytest.raises(ValueError, match="一次只执行一个 Turn"):
        store.add_run(AgentRun(id="r-2", session_id=session.id, user_id="u", user_request="another"))


def test_interrupted_turn_is_recovered_as_failed(tmp_path):
    path = tmp_path / "agent.json"
    store = AgentStore(path)
    session = store.add_session(AgentSession(id="s-1", user_id="u", dataset_ids=[3]))
    store.add_run(AgentRun(id="r-1", session_id=session.id, user_id="u", user_request="analyze"))
    store.persist()

    recovered = AgentStore(path)
    run = recovered.get_run("r-1")
    assert run.status == RunStatus.FAILED
    assert "重启" in run.error
    assert recovered.active_run(session.id) is None
