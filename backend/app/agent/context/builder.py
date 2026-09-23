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
            max_tokens=settings.AGENT_CONTEXT_MAX_TOKENS,
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
        retrieved: list[dict[str, Any]] = []
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
        elif not settings.AGENT_ENABLE_TOOL_RETRIEVAL:
            selected_by_name.update({str(x.get("name")): x for x in tool_describes if x.get("name")})

        # 意图判定与关键词表统一走 app.agent.intent（唯一真源）。
        # 改之前这里有一套自己的关键词，与运行时路由各写一份 ——
        # 「路由放行了、候选集没注入」就是这么来的。
        from app.agent.intent import CORE_TOOLS, hits, tool_domains_of

        intent_hits = hits(context.user_request)
        wanted_domains: set[str] = set()
        for intent in intent_hits:
            wanted_domains |= set(tool_domains_of(intent))
        if not intent_hits:
            # 意图完全没命中：仍补齐 EDA 与报告，保证最小可用能力面
            wanted_domains |= {"eda", "report"}

        for item in tool_describes:
            name = str(item.get("name") or "")
            if not name:
                continue
            if name in CORE_TOOLS:
                selected_by_name[name] = item
                continue
            # 工具名前缀（dataset / data / eda / ml / workflow / report）落在
            # 命中的能力域里就注入 —— 由工具名决定归属，不再维护第二份工具名单。
            if name.split(".", 1)[0] in wanted_domains:
                selected_by_name[name] = item

        # ---- 候选集合封顶：保证 prompt 不会无限膨胀 --------------------
        max_tools = max(int(settings.AGENT_TOOL_RETRIEVAL_TOP_K), len(CORE_TOOLS))
        max_tools = min(max_tools + 8, len(tool_describes) or 1)
        if len(selected_by_name) > max_tools:
            # 优先级：能力核心 → 按意图注入的域工具 → 检索分数补齐
            priority = list(CORE_TOOLS) + [n for n in selected_by_name if n not in CORE_TOOLS]
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
