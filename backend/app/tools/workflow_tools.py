"""Workflow Agent Tools：让 Agent 能发现、设计、检查并执行平台 Workflow。"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.permission import Permission, RiskLevel
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

    ★ 节点产出必须**进 summary**。实测：三个节点全部 success，
    但 outputs 里是 ``{"_df": "shape: ...\n┌───┬..."}`` 这种深层嵌套 + 上千字表格，
    事实抽取器既到不了那一层、也不收超长字符串，于是模型只能写
    「本次未获取描述性统计的具体数值结果 / 未获取报告正文」——
    用户跑完工作流拿到的只有一句「执行成功」。
    工具最清楚自己的输出形状，压缩的责任在这里，不在通用的事实抽取器。

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


def _short(value: Any, limit: int = 200) -> str:
    """把任意产出压成一行可读文本。"""
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    raw = " ".join((raw or "").split())
    return raw if len(raw) <= limit else raw[:limit] + "…"


#: 产出里真正有信息量的键（按信息量排序）。命中第一个就够。
_DIGEST_KEYS = (
    "report_text", "markdown", "text", "content", "summary",
    "sections", "issues", "key_statistics", "metrics", "shape",
    "rows", "columns", "row_count", "column_count",
)


def _output_digest(value: Any) -> str:
    """把单个节点的产出压成一句话，供 summary / LLM 事实摘要使用。

    ``_df`` 是纯表格转储（实测单节点 4~8 KB），除了首行的 ``shape:`` 之外
    对答案毫无价值，必须挡掉 —— 否则它会把上下文和展示位全占满。
    """
    if isinstance(value, dict):
        for key in _DIGEST_KEYS:
            if key in value:
                return f"{key}={_short(value[key])}"
        raw = value.get("_df")
        if isinstance(raw, str) and raw.strip().startswith("shape:"):
            return " ".join(raw.strip().splitlines()[0].split())
        for key, inner in value.items():
            if not str(key).startswith("_"):
                return f"{key}={_short(inner)}"
        return "（表格数据）"
    return _short(value)


def _outputs_digest(node_states: dict[str, Any], outputs: dict[str, Any]) -> str:
    """把「每个成功节点产出了什么」压成一段话。"""
    parts: list[str] = []
    for node_id, state in (node_states or {}).items():
        if state != "success":
            continue
        digest = _output_digest((outputs or {}).get(node_id))
        parts.append(f"{node_id}：{digest}")
    return "；".join(parts)


class WorkflowRecommendTool(Tool):
    """基于数据集真实字段自动设计候选 Workflow。

    ★ 历史缺陷：用户说「创建一个叫 X 的 Workflow」时，链路只会跑
      ``workflow.list`` 然后反问「触发方式？流程内容？先做哪一步？」——
      手上有数据集、有列名、有行数，却把设计的活推回给用户（挤牙膏）。

    这里先由**数据**说话：读 schema（列名/类型/唯一值数/行数），
    按「能不能真的跑通」排出 1~3 个方案，并把每个方案的节点与边直接给出，
    用户回一个数字就能接着执行。
    """

    name = "workflow.recommend"
    description = (
        "基于数据集的真实字段（列名、类型、唯一值数、行数）自动设计 1~3 个可执行的 Workflow 方案，"
        "每个方案都带可直接执行的 nodes/edges。用户要求「设计/创建一个 Workflow」时**必须先调用它**，"
        "不要反问用户要包含哪些步骤。返回 needs_choice=True 表示需要用户从中选一个。"
    )
    category = "workflow"
    permission = Permission.EXECUTE_WORKFLOW
    input_schema = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "integer"},
            # ★ 与 dataset.inspect 同理：不给 version 就会读 latest，
            #   而 latest 可能已被一次空结果操作顶成 0 行的表 —— 在 0 行上设计出的
            #   方案全是跑不通的（第十二轮实测）。引擎会填会话基线版本。
            "version": {"type": "integer", "description": "要基于哪个版本设计；不给则读最新版本"},
            "goal": {"type": "string", "description": "用户的原始诉求或期望目标，用于排序方案"},
        },
        "required": ["dataset_id"],
    }
    output_schema = {"type": "object"}

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        dataset_id = int(params["dataset_id"])
        self.assert_dataset_access(context, dataset_id)
        goal = str(params.get("goal") or "")
        version = params.get("version")
        columns, dtypes, unique, row_count, name = self._profile(
            dataset_id, services, int(version) if version else None
        )

        if not columns:
            return ToolResult.fail(
                f"读不到数据集 {dataset_id} 的字段结构，无法设计 Workflow",
                data={"dataset_id": dataset_id},
            )

        from app.workflow.recommend import plan_options, plan_question, recommend_plans

        plans = recommend_plans(
            dataset_id=dataset_id,
            columns=columns,
            dtypes=dtypes,
            unique=unique,
            row_count=row_count,
            goal=goal,
            dataset_name=name,
        )
        return ToolResult.ok(
            data={
                "dataset_id": dataset_id,
                "dataset_name": name,
                "row_count": row_count,
                "columns": columns,
                "plans": plans,
                # 原样回传：下一步执行时要按**同一个 goal** 还原方案，
                # 否则排序一变，用户选的序号就指到别的方案上了。
                "goal": goal,
                # 机器可读的「必须人来选」标记：下游据此挂起问人，
                # 而不是自己挑一个默认方案往下跑。
                "needs_choice": True,
                "plan_options": plan_options(plans),
                "question": plan_question(plans, columns, row_count),
            },
            summary=(
                f"基于数据集 {dataset_id}（{row_count:,} 行 × {len(columns)} 列）"
                f"设计了 {len(plans)} 个 Workflow 方案，等待用户选择："
                + "；".join(f"{p['id']}.{p['name']}" for p in plans)
            ),
        )

    @staticmethod
    def _profile(
        dataset_id: int, services: ToolServices, version: int | None = None
    ) -> tuple[list[str], dict[str, str], dict[str, int], int, str]:
        """读 schema + 规模。只读元信息与一次聚合，不加载数据行。"""
        import polars as pl

        ds = services.require("dataset_service")
        name = ""
        try:
            name = str(ds.get(dataset_id).name or "")
        except Exception:  # noqa: BLE001 - 名字只是修饰
            name = ""
        frame = ds.scan_version(dataset_id, version)
        schema = frame.collect_schema()
        columns = list(schema.names())
        dtypes = {c: str(schema[c]) for c in columns}
        try:
            row_count = int(frame.select(pl.len()).collect().item())
        except Exception:  # noqa: BLE001
            row_count = 0
        # 唯一值数只算非数值列（数值列的分布靠类型判断即可），
        # 且最多算前 30 列 —— 宽表上逐列 n_unique 会拖慢整次调用。
        unique: dict[str, int] = {}
        targets = [c for c in columns if not str(dtypes.get(c, "")).lower().startswith(("int", "uint", "float"))]
        for col in targets[:30]:
            try:
                unique[col] = int(frame.select(pl.col(col).n_unique()).collect().item())
            except Exception:  # noqa: BLE001
                continue
        return columns, dtypes, unique, row_count, name


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


#: 用户没给名字时，槽位抽取经常把「工作流」这个**通名**当成名字回填回来
#: —— 于是流程列表里出现一条叫「工作流」的工作流，用户根本分不清是哪一条。
#: 判定方式：把这些通名全部去掉后如果什么都不剩，说明它只是个占位符。
_GENERIC_NAME_TOKENS = ("工作流", "流程", "workflow", "pipeline", "新建", "我的", "这个", "一个", "新的")


def _is_placeholder_name(name: str) -> bool:
    """这个名字是不是只是「工作流」这类通名的占位符。"""
    stripped = (name or "").strip().lower()
    for token in _GENERIC_NAME_TOKENS:
        stripped = stripped.replace(token, "")
    return not stripped.strip(" 的了个-_")


class WorkflowBuildAndRunTool(Tool):
    name = "workflow.build_and_run"
    description = (
        "一步完成「创建 Workflow + 执行 Workflow 」，是 Agent 跑流程的首选工具（比 workflow.create + workflow.run 更可靠，"
        "因为它不需要跨步骤传递 workflow_id）。nodes 为节点数组 [{id,type,config}]，edges 为 [{source,target}]。"
        "可用节点类型：dataset.read（或 data.load，config.dataset_id）、data.quality_check、data.statistics、report.summary、"
        "data.clean/data.filter/data.transform/data.aggregate/data.duplicate/data.pivot/data.melt、ml.train/ml.predict/ml.evaluate/ml.cluster/ml.pca、ai.analyze。"
        "dataset_id 从会话上下文或 dataset.inspect 结果获取。"
        "★ 也可只给 plan（用户在 workflow.recommend 给的序号 1/2/3 或方案名）+ dataset_id："
        "工具会用与推荐时相同的输入把方案还原成 nodes/edges 再执行，"
        "这样「先推荐再选择」不需要跨步骤传状态。"
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
            "plan": {"type": "string", "description": "用户选择的方案序号（1/2/3）或方案名；给了它就不需要 nodes/edges"},
            "dataset_id": {"type": "integer", "description": "用 plan 还原方案时必填"},
            "version": {"type": "integer", "description": "用 plan 还原方案时所用的版本，必须与推荐时一致"},
            "goal": {"type": "string", "description": "原始诉求，用于让方案还原与推荐时排序一致"},
        },
        "required": ["name"],
    }
    output_schema = {"type": "object"}
    permission = Permission.EXECUTE_WORKFLOW
    risk_level = RiskLevel.HIGH
    requires_confirmation = True

    @staticmethod
    def _resolve_plan(
        params: dict[str, Any], services: ToolServices
    ) -> tuple[list[Any], list[Any], str] | ToolResult:
        """把用户选的序号/方案名还原成 nodes/edges。

        推荐是 dataset_id + goal 的纯函数，这里用**同样的输入**再算一遍，
        因此不需要把方案存进会话状态 —— 也就不存在「状态丢了方案就找不回来」。
        """
        from app.workflow.recommend import recommend_plans, resolve_plan

        dataset_id = params.get("dataset_id")
        if dataset_id is None:
            return ToolResult.fail(
                "要用 plan 还原方案必须同时给出 dataset_id。",
                data={"plan": params.get("plan")},
            )
        dataset_id = int(dataset_id)
        goal = str(params.get("goal") or "")
        version = params.get("version")
        columns, dtypes, unique, row_count, name = WorkflowRecommendTool._profile(
            dataset_id, services, int(version) if version else None
        )
        plans = recommend_plans(
            dataset_id=dataset_id,
            columns=columns,
            dtypes=dtypes,
            unique=unique,
            row_count=row_count,
            goal=goal,
            dataset_name=name,
        )
        plan = resolve_plan(plans, params.get("plan"))
        if plan is None:
            # 认不出就必须明确说，绝不偷偷选第一个 —— 那会让用户以为
            # 自己选的 A 被执行了，实际跑的是 B。
            available = "；".join(f"{p['id']}.{p['name']}" for p in plans)
            return ToolResult.fail(
                f"没有匹配到方案「{params.get('plan')}」。当前可选：{available}",
                data={"plans": plans},
            )
        return list(plan["nodes"]), list(plan["edges"]), str(plan.get("name") or "")

    def execute(self, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices) -> ToolResult:
        from app.api.deps import WORKFLOW_SERVICE

        nodes = list(params.get("nodes") or [])
        edges = list(params.get("edges") or [])
        plan_name = ""
        # 「推荐 → 用户选序号 → 执行」的收口：方案是 dataset_id 的纯函数，
        # 用同一份输入再算一次即可还原，不必把方案存进跨步骤状态。
        if not nodes and str(params.get("plan") or "").strip():
            resolved = self._resolve_plan(params, services)
            if isinstance(resolved, ToolResult):
                return resolved
            nodes, edges, plan_name = resolved
        if not nodes:
            return ToolResult.fail(
                "缺少流程内容：请给出 nodes/edges，或先调用 workflow.recommend 拿到方案后回传 plan 序号。"
                "可用节点类型：dataset.read、data.quality_check、data.statistics、data.clean、"
                "data.filter、data.aggregate、data.transform、ml.train、ml.evaluate、report.summary。",
                data={"supported_node_types": sorted(WORKFLOW_SERVICE.node_runners)},
            )
        nodes, repairs = _normalize_nodes(nodes)
        name = str(params.get("name") or "").strip()
        if not name or _is_placeholder_name(name):
            # 宁可用方案名（「churn 预测建模」）也不要「工作流」：
            # 后者在列表里毫无区分度，用户事后根本认不出自己建的是哪一条。
            name = plan_name or f"方案{params.get('plan') or ''}流程"
        try:
            workflow = WORKFLOW_SERVICE.create(
                name=name,
                nodes=nodes,
                edges=edges,
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
                summary=(
                    f"Workflow {workflow_id}「{workflow.name}」已执行成功："
                    f"{len(succeeded)}/{node_total} 个节点成功（{status}）。"
                    f"产出：{_outputs_digest(node_states, payload.get('outputs'))}"
                ),
            )
        detail = _summarize_failures(node_states, errors)
        return ToolResult.fail(
            f"Workflow {workflow_id} 执行未完全成功：{detail}。"
            f"已完成节点的产出：{_outputs_digest(node_states, payload.get('outputs'))}",
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
                + f"。产出：{_outputs_digest(node_states, payload.get('outputs'))}"
            ),
        )
