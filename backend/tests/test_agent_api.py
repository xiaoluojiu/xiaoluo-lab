"""Agent HTTP 层：端点与事件名照旧，前端零改动。

这个文件同时是「前端契约」的守卫：字段名、事件名、状态码一旦漂移，
前端不会报错，只会静默显示空白 —— 所以必须在这里断言。
"""

from __future__ import annotations

import json

import pytest

from app.agent.models import EventType, RunStatus


def _ok(resp) -> dict:
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True, body
    return body["data"]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """按前端的口径切帧：``\\n\\n`` 分帧，读 event: 与 data:。"""
    events: list[tuple[str, dict]] = []
    for raw in text.split("\n\n"):
        chunk = raw.strip()
        if not chunk:
            continue
        name = ""
        data = ""
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if name:
            events.append((name, json.loads(data) if data else {}))
    return events


# ---------------------------------------------------------------- 会话


def test_create_list_patch_delete_session(client):
    session = _ok(client.post("/api/v1/agent/sessions", json={"title": "A", "dataset_ids": []}))
    assert session["id"].startswith("s-")

    sessions = _ok(client.get("/api/v1/agent/sessions"))
    assert any(s["id"] == session["id"] for s in sessions)

    patched = _ok(client.patch(f"/api/v1/agent/sessions/{session['id']}", json={"archived": True}))
    assert patched["archived"] is True

    assert client.delete(f"/api/v1/agent/sessions/{session['id']}").status_code == 200
    assert client.get(f"/api/v1/agent/sessions/{session['id']}/runs").status_code in (200, 404)


def test_delete_session_while_waiting_cancels_then_deletes(client, api_session):
    """挂起等确认的会话必须能删掉。

    历史缺陷：挂起态曾被当成「未结束」，DELETE 恒返回 409；而那个「停止」按钮只对
    当前正在看的 run 有效，从列表 / 归档视图够不着 —— 会话永远删不掉。正确行为是
    删除时把挂起运行收成终态再删。
    """
    client.patch(f"/api/v1/agent/sessions/{api_session}/context", json={"dataset_ids": [1]})
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "清洗一下数据", "stream": False},
    ))
    assert run["status"] == "waiting_confirmation", run["status"]

    resp = client.delete(f"/api/v1/agent/sessions/{api_session}")
    assert resp.status_code == 200, resp.text
    assert _ok(resp)["deleted"] is True
    # 会话与它的运行记录都清掉了
    assert client.get(f"/api/v1/agent/runs/{run['id']}").status_code == 404


def test_delete_session_with_in_flight_run_returns_409(client, api_store, api_session):
    """真正在跑的运行（pending / planning / running）线程还活着，仍拒绝删除。"""
    run = api_store.create_run(api_session, "跑个训练")
    run.status = RunStatus.RUNNING
    api_store.update_run(run)

    resp = client.delete(f"/api/v1/agent/sessions/{api_session}")
    assert resp.status_code == 409
    assert "未结束" in resp.text
    # 会话仍在
    assert client.get("/api/v1/agent/sessions").json()["data"]


def test_session_context_patch(client, api_session):
    data = _ok(client.patch(f"/api/v1/agent/sessions/{api_session}/context", json={"dataset_ids": [1, 2]}))
    assert data["dataset_ids"] == [1, 2]


def test_saying_forget_it_cancels_the_pending_run(client, api_session):
    """等确认时改口说「算了，不用清洗了」，系统不能再洗一遍。

    实测事故：这句话被当成新指令，真的执行了 data.clean（数据被改），
    而原来那条运行永远停在「等待确认」。
    """
    client.patch(f"/api/v1/agent/sessions/{api_session}/context", json={"dataset_ids": [1]})
    pending = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "清洗一下数据", "stream": False},
    ))
    assert pending["status"] == "waiting_confirmation"

    second = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "算了，不用清洗了", "stream": False},
    ))
    # 改口不是新指令：不跑工具，直接收尾
    assert second["status"] == "completed"
    assert second["tool_calls"] == []
    assert "不做了" in (second["final_answer"] or "")

    # 原来挂起的那条也被收成终态，不会永远停在等待确认
    old = _ok(client.get(f"/api/v1/agent/runs/{pending['id']}"))
    assert old["status"] != "waiting_confirmation"


def test_a_new_request_supersedes_a_pending_run(client, api_session):
    """挂起的运行遇到新指令要被收掉，否则界面上永远留着一张点不动的确认卡。"""
    client.patch(f"/api/v1/agent/sessions/{api_session}/context", json={"dataset_ids": [1]})
    pending = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "清洗一下数据", "stream": False},
    ))
    assert pending["status"] == "waiting_confirmation"

    _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "这个数据集有多少行", "stream": False},
    ))
    old = _ok(client.get(f"/api/v1/agent/runs/{pending['id']}"))
    assert old["status"] != "waiting_confirmation"


def test_session_not_found(client):
    assert client.get("/api/v1/agent/sessions/s-nope/runs").status_code in (404, 200)
    assert client.patch("/api/v1/agent/sessions/s-nope/context", json={"dataset_ids": []}).status_code == 404


# ---------------------------------------------------------------- 目录与能力


def test_tools_catalog(client):
    tools = _ok(client.get("/api/v1/agent/tools"))
    assert len(tools) > 20
    for tool in tools:
        assert {"name", "description", "permission", "risk_level"} <= set(tool)


def test_capabilities_shape(client):
    """CapabilitiesPanel 按三层结构取值，缺一层就显示「—」。"""
    caps = _ok(client.get("/api/v1/agent/capabilities"))
    assert set(caps) >= {"agent", "llm", "tools"}
    assert caps["agent"]["max_steps"] == 8
    assert caps["agent"]["llm_budget"]["max_calls"] == 4
    assert caps["agent"]["context_sections"]
    assert caps["agent"]["agent_policy"]
    assert caps["llm"]["api_key_set"] is False or caps["llm"]["api_key_set"] is True
    assert caps["tools"]["count"] > 0


# ---------------------------------------------------------------- 非流式


def test_message_non_stream_chat(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "你好", "stream": False},
    ))
    assert run["status"] == "completed"
    assert run["final_answer"]
    assert run["token_usage"]["llm_calls"] == 0
    assert "elapsed_seconds" in run
    assert "tool_call_count" in run


def test_message_non_stream_tool(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": False},
    ))
    assert run["status"] == "completed"
    assert run["tool_calls"][0]["tool"] == "dataset.list"
    assert run["tool_calls"][0]["status"] == "ok"


def test_empty_message_rejected(client, api_session):
    resp = client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "   ", "stream": False},
    )
    assert resp.status_code == 400


def test_get_run_and_trace(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": False},
    ))
    assert _ok(client.get(f"/api/v1/agent/runs/{run['id']}"))["id"] == run["id"]

    trace = client.get(f"/api/v1/agent/runs/{run['id']}/trace?format=md")
    assert trace.status_code == 200
    assert "Agent 运行轨迹" in trace.text
    assert client.get("/api/v1/agent/runs/r-nope").status_code == 404


# ---------------------------------------------------------------- SSE


def test_sse_stream_emits_expected_frames(client, api_session):
    resp = client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers.get("x-accel-buffering") == "no"

    events = _parse_sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "route"
    assert "tool_call" in names and "tool_result" in names
    assert "completed" in names
    assert names[-1] == "done"  # 前端靠它停止读取

    for name, payload in events:
        if name != "done":
            assert payload.get("run_id"), f"{name} 帧缺 run_id"
            assert isinstance(payload.get("seq"), int)


def test_sse_chat_stream(client, api_session):
    resp = client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "你好", "stream": True},
    )
    names = [name for name, _ in _parse_sse(resp.text)]
    assert names == ["route", "chat", "usage", "completed", "done"]


# ---------------------------------------------------------------- 确认 / 拒绝 / 澄清


def _high_risk_run(client, api_session) -> dict:
    """发一条会触发高风险确认的请求。

    必须先在会话上绑定数据集，否则引擎会先反问「哪个数据集」，走不到权限闸门。
    """
    client.patch(f"/api/v1/agent/sessions/{api_session}/context", json={"dataset_ids": [1]})
    return _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "清洗一下数据", "stream": False},
    ))


def _wait_terminal(client, run_id: str, timeout: float = 15.0) -> dict:
    """confirm / clarify 是在后台线程里续跑的，返回值可能还是 running。"""
    import time

    deadline = time.time() + timeout
    data = {}
    while time.time() < deadline:
        data = _ok(client.get(f"/api/v1/agent/runs/{run_id}"))
        if data["status"] not in ("pending", "planning", "running"):
            return data
        time.sleep(0.05)
    return data


def test_confirm_flow(client, api_session):
    run = _high_risk_run(client, api_session)
    assert run["status"] == "waiting_confirmation"
    assert run["pending_confirmation"]["tool"] == "data.clean"

    confirmed = _ok(client.post(f"/api/v1/agent/runs/{run['id']}/confirm"))
    assert confirmed["status"] != "waiting_confirmation"

    final = _wait_terminal(client, run["id"])
    assert final["status"] in ("completed", "failed")
    assert final["tool_calls"], "确认后必须真的执行了那一步"
    assert final["pending_confirmation"] is None

    # 重复确认必须幂等
    again = _ok(client.post(f"/api/v1/agent/runs/{run['id']}/confirm"))
    assert again["status"] == final["status"]


def test_deny_flow(client, api_session):
    run = _high_risk_run(client, api_session)
    assert run["status"] == "waiting_confirmation"

    denied = _ok(client.post(f"/api/v1/agent/runs/{run['id']}/deny"))
    assert denied["status"] == "completed"
    assert "已拒绝" in denied["final_answer"]
    assert denied["tool_calls"] == []


def test_clarify_flow(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "合并数据集", "stream": False},
    ))
    assert run["status"] == "waiting_clarification"
    assert run["pending_clarification"]["code"] == "slot.right_dataset_id"

    answered = _ok(client.post(f"/api/v1/agent/runs/{run['id']}/clarify", json={"answer": "1"}))
    assert answered["status"] != "waiting_clarification"


def test_clarify_rejects_non_numeric_run_id(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "合并数据集", "stream": False},
    ))
    resp = client.post(f"/api/v1/agent/runs/{run['id']}/clarify", json={"answer": "abc"})
    assert resp.status_code == 400


def test_clarify_rejects_empty_answer(client, api_session):
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "合并数据集", "stream": False},
    ))
    assert client.post(f"/api/v1/agent/runs/{run['id']}/clarify", json={"answer": " "}).status_code == 400


def test_cancel_flow(client, api_session, api_store):
    """取消的语义按状态分三种，界面表现完全不同：

    - 运行中：只打标记，等当前步骤结束（不能强杀正在跑的工具）
    - 挂起中：当场收成终态（引擎线程已退出，标记没人会读）
    - 已结束：原样返回，不许把 final_answer 改掉
    """
    from app.agent.models import RunStatus

    running = api_store.create_run(api_session, "有哪些数据集")
    running.status = RunStatus.RUNNING
    api_store.update_run(running)
    flagged = _ok(client.post(f"/api/v1/agent/runs/{running.id}/cancel"))
    assert flagged["cancel_requested"] is True
    assert flagged["status"] == "running"

    finished = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": False},
    ))
    assert finished["status"] == "completed"
    again = _ok(client.post(f"/api/v1/agent/runs/{finished['id']}/cancel"))
    assert again["cancel_requested"] is False
    assert again["final_answer"] == finished["final_answer"]


# ---------------------------------------------------------------- 事件类型契约


def test_all_frontend_event_names_are_reachable(client, api_session):
    """前端定义了 11 个事件类型；这里确认后端不会产出前端不认识的名字。

    刻意用一个「能一次跑完」的请求：挂起中的流会一直开着（设计如此），
    用它做断言会把测试卡死。
    """
    from app.agent.models import EventType as ET

    frontend_known = {e.value for e in ET} | {"done"}
    resp = client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": True},
    )
    for name, _ in _parse_sse(resp.text):
        assert name in frontend_known, f"出现了前端不认识的事件：{name}"


def test_no_preflight_or_replanning_events(client, api_session):
    """旧架构的两个事件已随重规划一起删除，不许再有。"""
    resp = client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "有哪些数据集", "stream": True},
    )
    names = [n for n, _ in _parse_sse(resp.text)]
    assert "preflight" not in names
    assert "replanning" not in names
    assert EventType.ROUTE.value in names


def test_clarify_emits_answered_event(client, api_session):
    """回答澄清后必须留下一条 stage=answered 的事件，时间线才连续。"""
    run = _ok(client.post(
        f"/api/v1/agent/sessions/{api_session}/messages",
        json={"content": "评估模型", "stream": False},
    ))
    client.post(f"/api/v1/agent/runs/{run['id']}/clarify", json={"answer": "1"})

    events = _ok(client.get(f"/api/v1/agent/runs/{run['id']}"))["events"]
    stages = [e["payload"].get("stage") for e in events if e["type"] == "clarification"]
    assert "asked" in stages
    assert "answered" in stages
