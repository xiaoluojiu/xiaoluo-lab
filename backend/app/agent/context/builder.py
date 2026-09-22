"""Agent ContextBuilder。

Context 层只负责提供低成本元数据、权限、历史与候选工具。
禁止在这里读取 DataFrame、schema、profile、quality 或 sample。
所有真实数据读取必须经过 Tool -> DataEngine，以便可观测、可审计并控制 Token。
"""

from __future__ import annotations

from typing import Any

from app.agent.context.budget import ContextBudget
from app.agent.context.models import AgentContext
from app.agent.permission.models import ROLE_PERMISSIONS
from app.core.config import settings
from app.data_engine.service import DataEngineService
from app.tools.registry import TOOL_REGISTRY, ToolRegistry


class ContextBuilder:
    def __init__(self, data_engine: DataEngineService, *, max_chars: int | None = None, budget: ContextBudget | None = None) -> None:
        self.engine = data_engine
        self.max_chars = max_chars or settings.AGENT_CONTEXT_MAX_CHARS
        self.budget = budget or ContextBudget(
            max_chars=self.max_chars,
            user_request=settings.AGENT_CONTEXT_USER_REQUEST_CHARS,
            dataset=settings.AGENT_CONTEXT_DATASET_CHARS,
            task=settings.AGENT_CONTEXT_TASK_CHARS,
            permissions=settings.AGENT_CONTEXT_PERMISSION_CHARS,
            tools=settings.AGENT_CONTEXT_TOOL_CHARS,
            history=settings.AGENT_CONTEXT_HISTORY_CHARS,
            history_messages=settings.AGENT_CONTEXT_HISTORY_MESSAGES,
        )

    def build(self, user_request: str, *, dataset_ids: list[int] | None = None, role: str = "analyst", history: list[dict[str, str]] | None = None, task_context: dict[str, Any] | None = None) -> AgentContext:
        context = AgentContext(user_request=user_request.strip(), budget=self.budget)
        if dataset_ids:
            for ds_id in dict.fromkeys(int(x) for x in dataset_ids if int(x) > 0):
                context.dataset_context[str(ds_id)] = self._dataset_metadata(int(ds_id))
        context.task_context = task_context or {}
        perms = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["viewer"])
        context.permission_context = {"role": role, "permissions": sorted(str(p) for p in perms)}
        if history:
            context.conversation_history = list(history)[-self.budget.history_messages :]
        return context

    def _dataset_metadata(self, dataset_id: int) -> dict[str, Any]:
        """只读取数据库元信息；这里严禁 load_version。"""
        service = self.engine.dataset_service
        item = service.get(dataset_id)
        version = service.get_version_row(dataset_id, None)
        return {
            "dataset_id": int(dataset_id),
            "name": item.name,
            "description": item.description or "",
            "version": int(version.version),
            "rows": int(version.row_count),
            "columns": int(version.column_count),
        }

    def with_tools(self, context: AgentContext, tool_describes: list[dict[str, Any]], *, registry: ToolRegistry | None = None) -> AgentContext:
        """本地召回候选工具，并对高频复合分析任务加入确定性的核心工具集合。"""
        active_registry = registry or TOOL_REGISTRY
        selected_by_name: dict[str, dict[str, Any]] = {}
        retrieval_scores: dict[str, dict[str, Any]] = {}
        if settings.AGENT_ENABLE_TOOL_RETRIEVAL and tool_describes:
            retrieved = active_registry.retrieve_with_scores(
                context.user_request,
                top_k=settings.AGENT_TOOL_RETRIEVAL_TOP_K,
                min_score=settings.AGENT_TOOL_RETRIEVAL_MIN_SCORE,
            )
            for item in retrieved:
                tool = item.get("tool") or {}
                tool_name = str(tool.get("name") or "")
                if tool_name:
                    selected_by_name[tool_name] = tool
                    retrieval_scores[tool_name] = {"score": item.get("score", 0.0), "reason": item.get("reason", "")}
        else:
            selected_by_name.update({str(x.get("name")): x for x in tool_describes if x.get("name")})

        # ---- 确定性注入 ------------------------------------------------
        # 背景：纯检索式召回（top_k + min_score）会把 report.generate / eda.visualize /
        # workflow.* 挡在候选集合外，Planner 一旦规划它们就触发 PlanInvalidError，
        # 表现为「工具已注册但 Agent 不调用 / 调用失败」。
        # 因此这里保证：任何数据分析任务都能看到一组「能力核心」工具，
        # 再按请求关键词叠加工作流 / 建模 / 清洗能力。
        _CORE_TOOLS = (
            "dataset.inspect", "dataset.schema", "dataset.quality", "dataset.profile",
            "eda.describe", "eda.correlation", "eda.visualize",
            "report.generate",
        )
        _ML_TOOLS = ("ml.detect_task", "ml.prepare", "ml.train", "ml.evaluate", "ml.explain")
        _WORKFLOW_TOOLS = ("workflow.build_and_run", "workflow.create", "workflow.run", "workflow.list", "workflow.inspect")
        _DATA_TOOLS = ("data.clean", "data.filter", "data.aggregate", "data.transform")
        _MERGE_TOOLS = ("data.merge",)

        for item in tool_describes:
            if item.get("name") in _CORE_TOOLS:
                selected_by_name[item["name"]] = item

        text = context.user_request.lower()
        comprehensive = any(k in text for k in ("智能分析", "全面分析", "完整分析", "关键统计", "问题摘要", "综合"))
        quality = any(k in text for k in ("质量", "缺失", "重复", "异常", "quality"))
        train = any(k in text for k in ("训练", "模型", "预测", "分类", "回归", "聚类", "train", "model"))
        eda = any(k in text for k in ("相关", "相关性", "correlation", "分布", "探索", "描述", "统计", "直方图", "散点", "热力图", "可视化", "图表"))
        workflow = any(k in text for k in ("workflow", "工作流", "流程", "编排", "流水线", "pipeline", "节点"))
        cleaning = any(k in text for k in ("清洗", "去重", "过滤", "筛选", "缺失值", "转换", "聚合"))
        report = any(k in text for k in ("报告", "汇报", "导出", "结论", "总结", "report", "pdf"))
        # 多表关联：说法很多（关联 / 拼接 / 宽表 / join / merge），缺一个词就会让
        # data.merge 落选候选集，Planner 规划它时直接被判非法。
        merge = any(k in text for k in ("合并", "关联", "拼接", "连接", "宽表", "join", "merge"))

        wanted: set[str] = set()
        if train:
            wanted |= set(_ML_TOOLS)
        if workflow:
            wanted |= set(_WORKFLOW_TOOLS)
        if cleaning:
            wanted |= set(_DATA_TOOLS)
        if merge:
            wanted |= set(_MERGE_TOOLS)
        # 「全面分析 / 生成报告」默认需要建模与图表能力，避免计划被判非法
        if comprehensive:
            wanted |= set(_ML_TOOLS) | set(_DATA_TOOLS)
        if report:
            wanted |= {"report.generate", "eda.visualize"}
        # 兜底：若意图完全没命中，仍然补齐 EDA 与报告，保证最小可用能力面
        if not (train or workflow or cleaning or eda or quality or comprehensive or report):
            wanted |= {"eda.describe", "eda.correlation", "report.generate"}

        for item in tool_describes:
            if item.get("name") in wanted:
                selected_by_name[item["name"]] = item

        # ---- 候选集合封顶：保证 prompt 不会无限膨胀 --------------------
        max_tools = max(int(settings.AGENT_TOOL_RETRIEVAL_TOP_K), len(_CORE_TOOLS))
        max_tools = min(max_tools + 8, len(tool_describes) or 1)
        if len(selected_by_name) > max_tools:
            priority = list(_CORE_TOOLS) + sorted(wanted)
            kept: dict[str, dict[str, Any]] = {}
            for name in priority:
                if name in selected_by_name and len(kept) < max_tools:
                    kept[name] = selected_by_name[name]
            # 剩余名额按检索分数补齐
            for name in sorted(
                (n for n in selected_by_name if n not in kept),
                key=lambda n: -float((retrieval_scores.get(n) or {}).get("score", 0.0)),
            ):
                if len(kept) >= max_tools:
                    break
                kept[name] = selected_by_name[name]
            selected_by_name = kept

        selected = list(selected_by_name.values())
        if not selected:
            selected = tool_describes[: max(settings.AGENT_TOOL_RETRIEVAL_TOP_K, 1)]
        context.tool_context = {
            "retrieval_scores": retrieval_scores,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "params": list((t.get("input_schema") or {}).get("properties", {})),
                    "required": list((t.get("input_schema") or {}).get("required", [])),
                    "risk_level": t.get("risk_level", "low"),
                }
                for t in selected
            ]
        }
        return context
