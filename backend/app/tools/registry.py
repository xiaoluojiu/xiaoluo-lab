"""ToolRegistry：注册、发现、权限裁决与受控执行。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.permission.manager import PermissionManager
from app.core.config import TOOL_CATEGORY_HINTS
from app.core.exceptions import AppException
from app.core.registry import Registry
from app.tools.base import Tool, ToolConfirmationRequired, ToolPermissionError, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.result import ToolResult

logger = logging.getLogger(__name__)


class UnknownToolError(AppException):
    http_status = 404
    default_code = "TOOL_NOT_FOUND"
    default_message = "Tool not found"


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]*|[\u4e00-\u9fff]")


class ToolRegistry(Registry[Tool]):
    """工具注册表 + 受控执行入口。

    键值存储、重复/缺失报错、遍历与自描述汇总全部复用
    :class:`app.core.registry.Registry`（标准化内核），本类只保留工具层特有的
    **语义检索** 与 **受控执行** 两件事，不再重复实现一份 dict 管理。
    """

    def __init__(self, permission_manager: PermissionManager | None = None) -> None:
        super().__init__(label="工具")
        self.permission_manager = permission_manager or PermissionManager()

    # ---- Registry 错误钩子：沿用工具层既有异常体系 ----------------------
    def _conflict_error(self, key: str) -> AppException:
        return AppException(f"工具 {key!r} 已注册", code="TOOL_ALREADY_REGISTERED")

    def _missing_error(self, key: str) -> AppException:
        return UnknownToolError(details={"tool": key, "available": self.keys()})

    # ---- 注册（保持既有「传 Tool 实例」的调用方式） ---------------------
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if not tool.name:
            raise AppException("工具必须声明 name", code="TOOL_NAME_REQUIRED")
        return super().register(tool.name, tool, replace=replace)

    def list(self) -> list[dict[str, Any]]:
        """全部工具自描述（顺序稳定，供 Planner 上下文与 /agent/tools 使用）。"""
        return self.describe_all()

    def names(self) -> list[str]:
        """已注册工具名（排序，供安全校验与文档使用）。"""
        return self.keys()

    def retrieve(self, query: str, *, top_k: int = 8, min_score: float = 0.15) -> list[dict[str, Any]]:
        """本地、零 Token 工具召回（不含得分，供 Planner 上下文使用）。"""
        return [item["tool"] for item in self.retrieve_with_scores(query, top_k=top_k, min_score=min_score)]

    def retrieve_with_scores(self, query: str, *, top_k: int = 8, min_score: float = 0.15) -> list[dict[str, Any]]:
        """本地、零 Token 工具召回（带得分与命中原因，供可观测性事件使用）。

        除 name/description/category 外加入稳定的任务词映射。
        这层只负责扩大候选集合，不替代 Planner 的最终决策。
        """
        if not len(self):
            return []
        top_k = max(int(top_k), 1)
        min_score = max(float(min_score), 0.0)
        query = (query or "").strip().lower()
        if not query:
            return [{"tool": desc, "score": 0.0, "reason": "空查询，取前 top_k"} for desc in self.list()[:top_k]]

        # 关键词表来自 app.core.config.TOOL_CATEGORY_HINTS（补关键词不需要改这里的
        # 打分逻辑）。工具自身的 name/description/category 本来就参与匹配
        # （已拼进 searchable），这里只是补「用户会这么说、描述里没写到」的说法。
        category_hints = TOOL_CATEGORY_HINTS
        query_tokens = set(_TOKEN_RE.findall(query))
        scored: list[tuple[float, str, dict[str, Any], list[str]]] = []
        for tool in self.values():
            desc = tool.describe()
            name = str(desc.get("name", "")).lower()
            description = str(desc.get("description", "")).lower()
            category = str(desc.get("category", "")).lower()
            searchable = f"{name} {description} {category}".lower()
            tool_tokens = set(_TOKEN_RE.findall(searchable))
            score = 0.0
            reasons: list[str] = []
            if name and name in query:
                score += 1.0; reasons.append("工具名完整命中")
            if name and any(part and part in query for part in name.replace(".", " ").split()):
                score += 0.35; reasons.append("工具名部分命中")
            if query and query in description:
                score += 0.75; reasons.append("查询出现在描述中")
            overlap = len(query_tokens & tool_tokens)
            if query_tokens:
                ratio = overlap / len(query_tokens)
                score += 0.55 * ratio
                if overlap:
                    reasons.append(f"关键词重合 {overlap} 个")
            for hint in category_hints.get(category, ()):
                if hint in query:
                    score += 0.18; reasons.append(f"命中类目词「{hint}」")
            if score >= min_score:
                scored.append((score, name, desc, reasons))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"tool": item[2], "score": round(item[0], 3), "reason": "；".join(item[3][:3]) or "低分召回"}
            for item in scored[:top_k]
        ]

    def execute(self, name: str, params: dict[str, Any], context: ToolExecutionContext, services: ToolServices | None = None, *, confirmed: bool = False) -> ToolResult:
        """执行工具：权限裁决 -> 工具执行。"""
        tool = self.get(name)
        services = services or ToolServices()
        decision = self.permission_manager.check(context.user_id, tool, context, params)
        if decision.denied:
            raise ToolPermissionError(decision.reason, details={"tool": name})
        if decision.needs_confirmation and not confirmed:
            raise ToolConfirmationRequired(decision.reason, details={"tool": name})
        try:
            return tool.execute(params, context, services)
        except (ToolPermissionError, ToolConfirmationRequired):
            raise
        except AppException as exc:
            # 业务异常（如 MergeError）：记录 details 到日志便于溯源，再上抛给执行器。
            logger.warning(
                "工具 %s 抛出业务异常: %s | details=%s",
                name, str(exc), getattr(exc, "details", None),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            # 记录完整 traceback 到日志（此前只 str(exc) 返回给 Agent，异常堆栈被丢弃，
            # 服务重启后无从溯源）。脱敏参数，避免把用户数据写进日志。
            logger.exception(
                "工具 %s 执行失败（参数已脱敏）: %s",
                name,
                _redact_params(params),
            )
            return ToolResult.fail(f"工具执行失败：{exc}", metadata={"tool": name})


TOOL_REGISTRY = ToolRegistry()


def _redact_params(params: dict[str, Any], max_chars: int = 500) -> str:
    """把工具参数压成一行可入日志的字符串，截断长值、脱敏敏感键。

    目的：日志里能定位「这次失败是哪个工具、什么参数导致的」，但不把整张数据集
    或用户数据刷进日志。大对象（dict/list 很长）只保留结构概览。
    """
    def _shrink(value: Any, depth: int = 0) -> Any:
        if depth > 2:
            return "…"
        if isinstance(value, dict):
            return {k: _shrink(v, depth + 1) for k, v in list(value.items())[:20]}
        if isinstance(value, (list, tuple)):
            return [_shrink(v, depth + 1) for v in list(value)[:10]]
        if isinstance(value, str) and len(value) > 200:
            return value[:200] + "…"
        return value

    try:
        # 先脱敏（api_key/token/password 等敏感键的值 → ***），再截断，最后序列化。
        from app.core.logging import redact
        text = json.dumps(redact(_shrink(params)), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = repr(params)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text
