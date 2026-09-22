"""Workflow Agent Tools：让 Agent 能发现、设计、检查并执行平台 Workflow。"""
from __future__ import annotations

import logging
from typing import Any

from app.agent.permission.models import Permission
from app.agent.permission.rules import RiskLevel
from app.core.exceptions import AppException
from app.tools.base import Tool, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult
from app.workflow.runners import sanitize_output

logger = logging.getLogger(__name__)


# LLM 规划工作流时常用的别名 → 平台真实节点类型。
# 别名直接映射而不是让这一次调用失败：模型往往只是用了「更像自然语言」的命名，
# 语义与真实节点一一对应，静默纠偏比让整次工具调用失败更有价值（修复会记在 repairs 里）。
_NODE_TYPE_ALIASES: dict[str, str] = {
    "dataset.load": "dataset.read",
    "dataset.read_data": "dataset.read",
    "data.read": "dataset.read",
    "load_data": "dataset.read",
    "data.quality": "data.quality_check",
    "quality_check": "data.quality_check",
    "data.profile": "data.statistics",
    "data.describe": "data.statistics",
    "eda.describe": "data.statistics",
    "data.summary": "data.statistics",
    "statistics": "data.statistics",
    "report.generate": "report.summary",
    "summary": "report.summary",
}


def _normalize_nodes(nodes: list[Any]) -> tuple[list[Any], list[str]]:
    """把别名单点类型纠偏为真实类型，返回（纠偏后的节点，修复记录）。"""
    repairs: list[str] = []
    normalized: list[Any] = []
    for node in nodes or []:
        if isinstance(node, dict) and isinstance(node.get("type"), str):
            raw = node["type"]
            target = _NODE_TYPE_ALIASES.get(raw.strip().lower())
            if target and target != raw:
                fixed = dict(node)
                fixed["type"] = target
                normalized.append(fixed)
                repairs.append(f"{node.get('id', '?')}: {raw} → {target}")
                continue
        normalized.append(node)
    return normalized, repairs


def _run_workflow(workflow_id: int, services: ToolServices, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行 workflow 并整理成 Agent 友好的结构化输出。

    关键：node_logs（含每个失败节点的真实异常信息）必须回传给 Agent。
    过去只回传 node_states，Agent 只看到「某节点 failed」却不知道为什么，
    只能原样重试，表现为「Workflow 工具流执行不了」且无法自愈。

    返回 ``{"ok": bool, "payload": {...}}``；被运行前预检拦下时返回
    ``{"ok": False, "blocked": <AppException>}``——此时 **没有 payload**，
    调用方必须先判 ``blocked`` 再取 payload。
    """
    from app.api.deps import WORKFLOW_SERVICE
    run_context = dict(context or {})
    run_context.update(
        {
            "dataset_service": services.dataset_service,
            "data_engine_service": services.data_engine_service,
        }
    )
    try:
        handle = WORKFLOW_SERVICE.run(workflow_id, context=run_context)
    except AppException as exc:
        # 运行前预检未通过（如必需参数缺失）：明确回答「未执行」，并带上逐节点原因。
        # 以前这种情况会以裸异常冒泡成一次失败的工具调用，Agent 只能原样重试；
        # 现在回传结构化 details，模型可以直接补齐参数再调一次。
        logger.info("workflow %s 运行前预检未通过：%s", workflow_id, exc.message)
        return {"ok": False, "blocked": exc}
    result = handle.result.to_dict() if handle.result else {}
    sanitized_outputs = sanitize_output(result.get("outputs") or {})
    raw_logs = result.get("logs") or []
    errors = {
        str(log.get("node_id")): str(log.get("error"))
        for log in raw_logs
        if log.get("error")
    }
    payload = {
        "run_id": handle.run_id,
        "workflow_id": handle.workflow_id,
        "status": str(handle.status.value),
        "node_states": result.get("node_states") or {},
        "outputs": sanitized_outputs,
        "logs": raw_logs,
        "errors": errors,
        "result": result,
    }
    ok = str(handle.status.value) == "success"
    return {"ok": ok, "payload": payload}


def _blocked_result(workflow_id: int, exc: AppException, workflow: Any = None) -> ToolResult:
    """把运行前预检失败翻译成 Agent 可自纠错的失败结果。"""
    details = exc.details if isinstance(exc.details, dict) else {}
    errors = [str(item) for item in (details.get("errors") or [])] or [exc.message]
    label = f"「{workflow.name}」" if workflow is not None else ""
    return ToolResult.fail(
        f"Workflow {workflow_id}{label} 未执行：{exc.message}（{'；'.join(errors[:6])}）",
        data={
            "workflow_id": workflow_id,
            "executed": False,
            "errors": errors,
            "warnings": details.get("warnings") or [],
        },
        metadata={"preflight": True, "workflow_id": workflow_id, "errors": errors},
    )


def _summarize_failures(node_states: dict[str, Any], errors: dict[str, str]) -> str:
    """把「哪些节点没成功 + 为什么」压缩成一句话，供 LLM 阅读与自我纠错。"""
    bad = {k: v for k, v in (node_states or {}).items() if v in {"failed", "skipped", "cancelled"}}
    if not bad:
        return ""
    parts = []
    for node_id, state in list(bad.items())[:6]:
        reason = errors.get(node_id)
        parts.append(f"{node_id}={state}" + (f"（{reason}）" if reason else ""))
    return "；".join(parts)


class WorkflowListTool(Tool):
    name = "workflow.list"
    description = "列出当前平台已有 Workflow，返回 ID、名称和节点数量。"
    category = "workflow"
    input_schema = {"type": "object", "properties": {}}
    output_schema = {"type": "object"}
    permission = Permission.EXECUTE_WORKFLOW

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.api.deps import WORKFLOW_SERVICE
        data = WORKFLOW_SERVICE.list()
        return ToolResult.ok(data=data, summary=f"共 {len(data)} 个 Workflow")


class WorkflowCreateTool(Tool):
    name = "workflow.create"
    description = "根据 Agent 设计创建并校验一个平台 Workflow；只创建不执行。Agent 可使用语义节点 dataset.read、data.quality_check、data.statistics、report.summary，以及平台原生 data.load/data.clean/data.filter/data.aggregate、ml.*、ai.analyze。返回 id 与 workflow_id 两个同义字段，后续 workflow.run 应使用 workflow_id。"
    category = "workflow"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
            "metadata": {"type": "object"},
        },
        "required": ["name", "nodes", "edges"],
    }
    output_schema = {
        "type": "object",
        "required": ["id", "workflow_id", "name", "nodes", "edges"],
    }
    permission = Permission.EXECUTE_WORKFLOW
    requires_confirmation = True

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.api.deps import WORKFLOW_SERVICE
        nodes, repairs = _normalize_nodes(list(params.get("nodes") or []))
        workflow = WORKFLOW_SERVICE.create(
            name=str(params["name"]),
            nodes=nodes,
            edges=list(params.get("edges") or []),
            metadata=dict(params.get("metadata") or {}),
        )
        workflow_id = int(workflow.metadata["id"])
        data = workflow.to_dict()
        # Agent 工具之间的依赖必须使用稳定、明确的结构化输出。
        # 同时保留 id 与 workflow_id，兼容模型常见的语义命名，避免把自然语言猜测带入下一步。
        data["id"] = workflow_id
        data["workflow_id"] = workflow_id
        data["name"] = workflow.name
        data["nodes"] = [n.to_dict() if hasattr(n, "to_dict") else n for n in workflow.nodes]
        data["edges"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in workflow.edges]
        return ToolResult.ok(data=data, summary=f"Workflow {workflow_id}「{workflow.name}」创建并校验成功")


class WorkflowInspectTool(Tool):
    name = "workflow.inspect"
    description = "查看指定 Workflow 的节点、边和元数据，用于 Agent 规划和校验。"
    category = "workflow"
    input_schema = {"type": "object", "properties": {"workflow_id": {"type": "integer"}}, "required": ["workflow_id"]}
    output_schema = {"type": "object"}
    permission = Permission.EXECUTE_WORKFLOW

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.api.deps import WORKFLOW_SERVICE
        workflow = WORKFLOW_SERVICE.get(int(params["workflow_id"]))
        return ToolResult.ok(data=workflow.to_dict(), summary=f"Workflow {params['workflow_id']} 已读取")


class WorkflowBuildAndRunTool(Tool):
    name = "workflow.build_and_run"
    description = (
        "一步完成「创建 Workflow + 执行 Workflow 」，是 Agent 跑流程的首选工具（比 workflow.create + workflow.run 更可靠，"
        "因为它不需要跨步骤传递 workflow_id）。nodes 为节点数组 [{id,type,config}]，edges 为 [{source,target}]。"
        "可用节点类型：dataset.read（或 data.load，config.dataset_id）、data.quality_check、data.statistics、report.summary、"
        "data.clean/data.filter/data.transform/data.aggregate/data.duplicate/data.pivot/data.melt、ml.train/ml.predict/ml.evaluate/ml.cluster/ml.pca、ai.analyze。"
        "dataset_id 从会话上下文或 dataset.inspect 结果获取。"
    )
    category = "workflow"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
            "metadata": {"type": "object"},
            "context": {"type": "object"},
        },
        "required": ["name", "nodes", "edges"],
    }
    output_schema = {"type": "object"}
    permission = Permission.EXECUTE_WORKFLOW
    risk_level = RiskLevel.HIGH
    requires_confirmation = True

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.api.deps import WORKFLOW_SERVICE
        nodes, repairs = _normalize_nodes(list(params.get("nodes") or []))
        try:
            workflow = WORKFLOW_SERVICE.create(
                name=str(params["name"]),
                nodes=nodes,
                edges=list(params.get("edges") or []),
                metadata=dict(params.get("metadata") or {}),
            )
        except Exception as exc:  # noqa: BLE001 - 校验失败要变成可自纠错的提示，而不是裸异常
            logger.warning("workflow.build_and_run 创建失败：%s", exc)
            supported = sorted(WORKFLOW_SERVICE.node_runners)
            return ToolResult.fail(
                f"Workflow 创建失败：{exc}。可用节点类型：{', '.join(supported)}",
                data={
                    "requested_nodes": nodes,
                    "repairs": repairs,
                    "supported_node_types": supported,
                },
                metadata={"errors": getattr(exc, "details", {}).get("errors", [str(exc)])},
            )
        workflow_id = int(workflow.metadata["id"])
        outcome = _run_workflow(workflow_id, services, params.get("context"))
        if outcome.get("blocked") is not None:
            return _blocked_result(workflow_id, outcome["blocked"], workflow)
        payload = outcome["payload"]
        status = payload["status"]
        node_states = payload.get("node_states") or {}
        succeeded = [k for k, v in node_states.items() if v == "success"]
        # 分母必须取节点总数（node_states 覆盖全部节点）。此前用的是 logs 条数，
        # 而失败/跳过的链路里同一节点可能留下多条日志 ⇒ 「n/m 个节点成功」失真，
        # 模型据此判断进度会得到错误的完成度。
        node_total = len(node_states)
        failed = {k: v for k, v in node_states.items() if v in {"failed", "skipped", "cancelled"}}
        errors = payload.get("errors") or {}
        data = {
            "id": workflow_id,
            "workflow_id": workflow_id,
            "name": workflow.name,
            "run_id": payload["run_id"],
            "status": status,
            "node_states": node_states,
            "outputs": payload["outputs"],
            "errors": errors,
            "nodes": [n.to_dict() for n in workflow.nodes],
            "edges": [e.to_dict() for e in workflow.edges],
            "repairs": repairs,
        }
        if not failed:
            return ToolResult.ok(
                data=data,
                summary=f"Workflow {workflow_id}「{workflow.name}」已执行成功：{len(succeeded)}/{node_total} 个节点成功（{status}）",
            )
        detail = _summarize_failures(node_states, errors)
        return ToolResult.fail(
            f"Workflow {workflow_id} 执行未完全成功：{detail}",
            data=data,
            metadata={"workflow_id": workflow_id, "failed_nodes": failed, "node_errors": errors},
        )


class WorkflowRunTool(Tool):
    name = "workflow.run"
    description = "执行指定 Workflow。workflow_id 必须是整数，通常来自前一步 workflow.create 的结构化输出。执行类操作需要 Agent 权限；高风险节点仍由 Workflow/工具权限规则控制。"
    category = "workflow"
    input_schema = {"type": "object", "properties": {"workflow_id": {"type": "integer"}, "context": {"type": "object"}}, "required": ["workflow_id"]}
    output_schema = {"type": "object"}
    permission = Permission.EXECUTE_WORKFLOW
    requires_confirmation = True

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        workflow_id = int(params["workflow_id"])
        outcome = _run_workflow(workflow_id, services, params.get("context"))
        if outcome.get("blocked") is not None:
            return _blocked_result(workflow_id, outcome["blocked"])
        payload = outcome["payload"]
        node_states = payload.get("node_states") or {}
        failed = {k: v for k, v in node_states.items() if v in {"failed", "skipped", "cancelled"}}
        detail = _summarize_failures(node_states, payload.get("errors") or {})
        return ToolResult.ok(
            data=payload,
            summary=(
                f"Workflow {workflow_id} 执行完成：{payload['status']}"
                + (f"，失败/跳过节点 {detail}" if failed else "")
            ),
        )
