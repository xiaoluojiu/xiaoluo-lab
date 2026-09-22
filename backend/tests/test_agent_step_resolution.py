from app.agent.planner.models import PlanStep
from app.agent.runtime.models import AgentRun, AgentSession
from app.agent.runtime.step_resolution import resolve_arguments
from app.tools.result import ToolResult


def test_resolve_workflow_id_from_previous_step():
    run = AgentRun(id="r-1", session_id="s-1", user_id="u", user_request="test")
    record = __import__("app.agent.executor.executor", fromlist=["ToolCallRecord"]).ToolCallRecord(
        step_index=0,
        tool="workflow.create",
        arguments={},
    )
    # workflow.create 的真实契约：同时返回 id 与 workflow_id 两个同义字段（见 workflow_tools.py）
    record.finish("ok", result=ToolResult.ok(data={"id": 17, "workflow_id": 17, "name": "orders-check", "nodes": [], "edges": []}, summary="created"))
    run.tool_calls.append(record)
    session = AgentSession(id="s-1", user_id="u", dataset_ids=[2])
    context = type("Context", (), {"dataset_ids": lambda self: [2]})()
    step = PlanStep(tool="workflow.run", arguments={"workflow_id": "{{step1.workflow_id}}"})
    resolved = resolve_arguments(
        step,
        run,
        session,
        context,
        {"type": "object", "properties": {"workflow_id": {"type": "integer"}}, "required": ["workflow_id"]},
    )
    assert resolved.arguments["workflow_id"] == 17


def test_missing_dependency_is_explicit():
    run = AgentRun(id="r-1", session_id="s-1", user_id="u", user_request="test")
    session = AgentSession(id="s-1", user_id="u", dataset_ids=[2])
    context = type("Context", (), {"dataset_ids": lambda self: [2]})()
    step = PlanStep(tool="workflow.run", arguments={"workflow_id": "{{step1.workflow_id}}"})
    try:
        resolve_arguments(step, run, session, context, {"type": "object", "properties": {"workflow_id": {"type": "integer"}}, "required": ["workflow_id"]})
    except Exception as exc:
        assert "第 1 步" in str(exc)
    else:
        raise AssertionError("missing dependency should fail clearly")
