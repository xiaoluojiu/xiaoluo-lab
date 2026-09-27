"""Agent 执行引擎。

控制流（一条直线，没有分叉，没有回头路）
----------------------------------------
::

    路由（规则，0 Token）
      → 未命中意图：对话渲染 → 结束
      → 命中：取出固定 playbook → 逐步执行
          ├─ 缺必填槽位 → 有 fallback 就换粗粒度工具，否则挂起等用户补充
          ├─ 高风险 → 挂起等用户确认
          ├─ 工具失败 → 可跳过就记警告继续，否则终止
          └─ 正常 → 记录结果，进入下一步
      → 渲染答案（模板 or LLM 润色）→ 结束

旧架构的病根在这里：**失败后重规划**。它会把不确定性顶上来、再发起一轮远程决策，
实测出现过单 run 近 20 万条 ``replanning`` 事件。新架构**没有重规划**：
一步失败要么跳过（该步骤声明了 ``skippable``），要么如实终止并报告原因。

有界性（四条硬闸，缺一不可）
----------------------------
1. **步数上限** ``MAX_STEPS``：playbook 本身通常 1~3 步，这是防御性上限
2. **LLM 调用上限** ``MAX_LLM_CALLS``：超出后所有 LLM 调用自动省略，退回规则
3. **墙钟上限** ``WALL_CLOCK_SECONDS``：超时即终止，避免 SSE 挂死
4. **停止条件**：完成 / 失败 / 挂起（等人）/ 取消 / 超预算，五者必居其一

人工接管点
----------
高风险工具与缺失槽位都会**挂起**而不是猜测。
一次性授权凭据 ``authorized_key`` 只对被确认的那一个 ``(step_index, tool)`` 生效，
用掉即失效 —— 写成 ``authorized_key is None or == key`` 会让一次确认放行整条链，
这是旧架构踩过并修掉的真实缺陷。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from app.agent.answer import render, render_chat
from app.agent.intents import _looks_like_payload, route
from app.agent.limits import MAX_LLM_CALLS, MAX_STEPS, WALL_CLOCK_SECONDS  # noqa: F401 - 对外保留同名常量
from app.agent.llm import LLMProvider
from app.agent.models import (
    ANSWER_SOURCE_RULE,
    AgentEvent,
    AgentRun,
    AgentSession,
    ClarificationOption,
    ClarificationRequest,
    EventType,
    PermissionRequest,
    RunStatus,
    ToolCall,
    ToolCallStatus,
)
from app.agent.permission import Permission
from app.agent.playbooks import SLOT_QUESTIONS, Playbook, PlaybookStep, get_playbook
from app.agent.slots import extract_slots
from app.agent.store import AgentStore
from app.core.exceptions import AppException
from app.tools.base import ToolConfirmationRequired, ToolServices
from app.tools.context import ToolExecutionContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: 有界性的三个上限定义在 :mod:`app.agent.limits`（引擎与 store / capabilities 共用，
#: 避免「闸门 4 次、面板写 12 次」的自相矛盾）。这里再导出一次，保持旧引用可用。
#: 候选分组键的中文标签（用于反问选项的 note）。
_GROUP_LABELS = {
    "regression": "适合做回归（预测数值）",
    "classification": "适合做分类（预测类别）",
}

#: 单步工具软超时提示阈值（秒）；仅用于日志，工具自身的超时由实现方控制
SLOW_STEP_SECONDS = 60.0

#: 对话路径最多带多少条历史进 LLM。太少接不上话，太多会挤掉事实摘要 ——
#: 用户的追问通常只跟最近一两轮有关。
_CHAT_HISTORY_MESSAGES = 8
#: 对话上下文最多描述几个数据集（会话绑定的可能很多，全列会撑爆预算）
_CHAT_CONTEXT_MAX_DATASETS = 2
#: 单个数据集在上下文里最多列多少个列名
_CHAT_CONTEXT_COLUMNS = 24
#: 上一次分析结论带进上下文的字符上限
_CHAT_CONTEXT_ANSWER_CHARS = 1200

#: 哪些写操作会**推进会话基线版本**（确实改变了「用户当前的数据状态」）。
#: 反过来，``aggregate`` / ``merge`` 只是把产出**另存**成新版本，数据本身的状态
#: 没变 —— 它们不推进基线。否则「按 purpose 分组求平均」会把 1220×14 的数据集
#: 顶成一张 7 行 × 2 列的聚合小表，之后每一步分析都跑在这张废表上
#: （实测：相关性报「至少需要 2 个数值字段」、质量检查说「7 行 2 列」、
#: 用户点名的 age 列根本不存在）。
_BASELINE_ADVANCING_OPS = frozenset({"filter", "transform"})

#: 一轮运行最多往会话历史里带几张图。图是内联 SVG，单张可达数十 KB，
#: 不设上限会把 agent_store.json 撑大，也会拖慢前端重建对话。
_MAX_CHARTS_PER_TURN = 6

#: 并集型列参数（如相关性的 columns）合并后的上限，防止无限膨胀
_UNION_MAX_COLUMNS = 8

#: 一次调用里哪些参数里嵌着列名 —— 用来找回「本会话最近在聊哪一列」
_FOCUS_COLUMN_KEYS = ("column", "x", "y", "columns", "group_by", "target")

#: 「这两个字段」「它们」这类回指：本轮没点名任何列，说的是上一轮那两列
_TWO_THINGS_RE = re.compile(r"两个字段|这两个|那两个|它们|他俩|两者|两列")

#: 建模参数的回指：用户没再点名目标列，说的是上一轮那个
_MODEL_REFERENCE_RE = re.compile(r"刚才那个|上一个|那个目标|之前那个|上次那个|同一个目标|其他不变|模型不变|还是那个")


def _collect_charts(run: Any) -> list[dict]:
    """从本轮工具调用里收集真正渲染成功的图表（SVG）。

    只认 ``metadata["chart"]["rendered"]`` 为真的那些：渲染失败的条目已经在
    工具层写了 warning，不能再当作「已经出图」塞进历史。
    """
    charts: list[dict] = []
    for call in list(run.tool_calls or []):
        # ToolCall 没有 metadata 字段，图表挂在 result["metadata"]["chart"] 上
        # （result 就是 ToolResult.to_dict() 的输出）。
        result = getattr(call, "result", None)
        if not isinstance(result, dict):
            continue
        meta = result.get("metadata")
        if not isinstance(meta, dict):
            continue
        chart = meta.get("chart")
        if isinstance(chart, dict) and chart.get("rendered") and chart.get("svg"):
            charts.append(chart)
        if len(charts) >= _MAX_CHARTS_PER_TURN:
            break
    return charts
_BASELINE_ADVANCING_TOOLS = frozenset({"data.clean"})

#: 结构化参数：用户补的是**一句话**，不是能直接塞给工具的值。
#: 这些键的澄清答案必须交给抽取器解析，不能原样写进 params ——
#: 「credit_amount 大于 5000 且 age 大于 60」原样当 conditions 传下去，
#: filter 拿到字符串直接抛异常（实测：澄清之后必然执行失败）。
_STRUCTURED_PARAMS = frozenset(
    {"conditions", "aggregations", "nodes", "edges", "group_by", "columns", "expression"}
)


def _needs_parsing(param: str, value: Any) -> bool:
    """这个澄清答案是不是「一句自然语言」，必须先解析才能当参数用。

    已经结构化的值（列表 / 字典）直接照用 —— 调用方比我们更清楚结构。
    """
    return param in _STRUCTURED_PARAMS and isinstance(value, str)


def _advances_baseline(tool: Any) -> bool:
    """该工具执行成功后，会话基线版本是否应推进到它产出的新版本。"""
    name = str(getattr(tool, "name", "") or "")
    op = str(getattr(tool, "op_type", "") or "")
    return name in _BASELINE_ADVANCING_TOOLS or op in _BASELINE_ADVANCING_OPS


def _produced_version(result: Any) -> int | None:
    """从工具结果里取出它产出的新版本号（没有则 None）。"""
    data = getattr(result, "data", None)
    if not isinstance(data, dict):
        return None
    produced = data.get("new_version")
    if isinstance(produced, bool):
        return None
    if isinstance(produced, int):
        return int(produced)
    # data.clean 回的是「每一步各产出一个版本」的列表，取最后一个
    versions = data.get("versions")
    if isinstance(versions, list) and versions:
        try:
            return int(versions[-1])
        except (TypeError, ValueError):
            return None
    return None


class AgentCancelled(Exception):
    """用户请求停止。内部信号，不对外暴露。"""


class AgentTimeout(Exception):
    """超过墙钟上限。内部信号，不对外暴露。"""


#: 报错 → 人能读懂的下一步。只登记真实出现过的报错，不做通用兜底话术。
_FAILURE_HINTS: tuple[tuple[str, str], ...] = (
    ("aggregation func", "聚合函数没认出来。换一种说法试试，例如「按 purpose 分组求 credit_amount 的平均值」。"),
    ("字段不存在", "列名可能对不上。先说一句「有哪些列」确认字段名，再重试。"),
    ("超时", "这一步超过了 10 分钟的墙钟上限。可以先筛选/抽样缩小数据量再跑。"),
    ("权限", "当前账号没有执行该操作的权限，请联系管理员开通。"),
    ("未注册", "这个能力当前没有挂载，换一个分析诉求试试。"),
)


def _failure_answer(error: str, run: Any) -> str:
    """把技术报错翻成人能读的中文。

    实测：``data.aggregate`` 因为槽位没给出 func 而失败时，用户看到的只有
    ``unsupported aggregation func: None`` —— 既不知道哪一步挂了，也不知道
    下一步该怎么问。这里不调 LLM（省预算），用规则补一句可执行的建议。
    """
    text = " ".join((error or "").split())
    failed = [c.tool for c in getattr(run, "tool_calls", []) if str(c.status) == "failed"]
    where = f"（{failed[-1]}）" if failed else ""
    lines = [f"这次没能跑通{where}：{text or '未知原因'}。"]
    for pattern, hint in _FAILURE_HINTS:
        if pattern in text:
            lines.append(hint)
            break
    else:
        lines.append("可以把需求说得更具体一点（点名列或字段），或换个说法重试。")
    return "\n".join(lines)


@dataclass
class _RestoredResult:
    """从持久化的 tool_call 恢复出的轻量结果（供答案渲染使用）。"""

    success: bool = False
    data: Any = None
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_call(cls, call: ToolCall) -> _RestoredResult | None:
        if call.result is None or not isinstance(call.result, dict):
            return None
        return cls(
            success=bool(call.result.get("success")),
            data=call.result.get("data"),
            summary=str(call.result.get("summary") or ""),
            warnings=list(call.result.get("warnings") or []),
            errors=list(call.result.get("errors") or []),
        )


class AgentEngine:
    """执行一次 Agent 运行。同步执行；由 API 层放进线程池。"""

    def __init__(
        self,
        store: AgentStore,
        *,
        tools: ToolRegistry | None = None,
        llm: LLMProvider | None = None,
        session_factory: Callable[[], Any] | None = None,
        dataset_service: Any | None = None,
        data_engine: Any | None = None,
        experiment_service: Any | None = None,
        connector_service: Any | None = None,
        db: Any | None = None,
    ) -> None:
        """构建引擎。

        引擎在 SSE / 确认恢复场景里跑在**后台线程**，而 SQLAlchemy Session
        与请求级依赖都不是线程安全的。因此默认形态是「只给会话工厂，内部自建
        全部服务，用完自己关闭」—— 调用方无需感知依赖链。
        注入式参数仅供测试替换个别服务。
        """
        self.store = store
        self.tools = tools or _default_registry()
        self.llm = llm
        self._session_factory = session_factory
        self._injected = ToolServices(
            dataset_service=dataset_service,
            data_engine_service=data_engine,
            experiment_service=experiment_service,
            connector_service=connector_service,
            db=db,
        )
        self._services_cache: ToolServices | None = None
        self._owns_db = False

    # ------------------------------------------------------------------
    def _services(self) -> ToolServices:
        """按需自建依赖链。同一实例内只建一次，运行结束时由 :meth:`_close` 回收。"""
        if self._services_cache is not None:
            return self._services_cache

        injected = self._injected
        if injected.dataset_service is not None:
            self._services_cache = injected
            return injected

        factory = self._session_factory or _default_session_factory()
        db = factory()
        self._owns_db = True
        from app.connectors.service import ConnectorService
        from app.data_engine.service import DataEngineService
        from app.experiments.service import ExperimentService
        from app.services.dataset_service import DatasetService
        from app.storage.service import get_storage

        dataset_service = DatasetService(db, get_storage())
        self._services_cache = ToolServices(
            dataset_service=dataset_service,
            data_engine_service=DataEngineService(dataset_service),
            experiment_service=ExperimentService(db, dataset_service),
            connector_service=ConnectorService(db, dataset_service),
            db=db,
        )
        return self._services_cache

    # ==================================================================
    # 主入口
    # ==================================================================
    def run(self, run: AgentRun, session: AgentSession) -> AgentRun:
        """执行或恢复一次运行。直接修改并返回传入的 run。"""
        if run.status.terminal:
            return run

        started = time.time()
        # 「是不是第一次执行」不能靠 resume_step 判断：
        # 首步就要确认时 resume_step 恰好是 0，恢复后会被当成全新运行 ——
        # ROUTE / PLANNING 再发一遍（前端时间线上出现两条一模一样的计划），
        # started_at 也被重置，elapsed_seconds 会丢掉用户思考的那几分钟。
        if self._is_fresh(run):
            run.started_at = started
            run.status = RunStatus.PLANNING
            self.store.update_run(run)

        try:
            return self._execute(run, session, started)
        except AgentCancelled:
            return self._finish(run, RunStatus.COMPLETED, "已停止本次运行。", None, started)
        except AgentTimeout:
            return self._finish(
                run, RunStatus.FAILED, "", None, started,
                error=f"运行超时（超过 {WALL_CLOCK_SECONDS:.0f} 秒），已终止",
            )
        except Exception as exc:  # noqa: BLE001
            # 兜底：任何未预期异常都必须让 run 进入终态，
            # 否则前端会一直轮询一个永远不结束的运行。
            logger.exception("Agent 运行异常终止（run=%s）", run.id)
            return self._finish(run, RunStatus.FAILED, "", None, started, error=str(exc))
        finally:
            self._close()

    def _close(self) -> None:
        """只关闭自己创建的会话；外部注入的由注入方负责。"""
        if not self._owns_db:
            return
        services = self._services_cache
        if services is not None and services.db is not None:
            try:
                services.db.close()
            except Exception:  # noqa: BLE001
                logger.debug("关闭 Agent 数据库会话失败", exc_info=True)
        self._services_cache = None
        self._owns_db = False

    # ==================================================================
    # 执行主体
    # ==================================================================
    @staticmethod
    def _is_fresh(run: AgentRun) -> bool:
        """本次是不是这条 run 的**第一次**执行。

        判据是「有没有发过 route 事件」，而不是 ``resume_step == 0``。
        """
        return not any(e.type is EventType.ROUTE for e in run.events)

    def _execute(self, run: AgentRun, session: AgentSession, started: float) -> AgentRun:
        fresh = self._is_fresh(run)
        route_result = route(run.user_request)
        if fresh:
            self._emit(
                run,
                EventType.ROUTE,
                {
                    "mode": "chat" if route_result.is_chat else "agent",
                    "reason": route_result.reason,
                    "intent": str(route_result.intent),
                },
            )

        playbook = get_playbook(route_result.intent)
        if playbook.chat_only:
            # ★ 用户贴的是一段参数/JSON，不是需求。交给模型自由发挥会把它当成
            #   需求去猜 —— 实测把半截 JSON 里的 ``"chart": "histogram"`` 读成
            #   「想画直方图」，回了一段「直方图适合看…你想看哪个字段」的话，
            #   用户压根没提过直方图（第十一轮）。这里直接说清楚并要求重说。
            if _looks_like_payload(run.user_request):
                return self._finish(
                    run,
                    RunStatus.COMPLETED,
                    "你发的内容看起来是一段参数（JSON / 键值对），不是一句分析需求，"
                    "所以我先没有执行任何操作。直接用话说就行，"
                    "例如「画 monthly_charges 的直方图」「看看数据质量」。",
                    ANSWER_SOURCE_RULE,
                    started,
                )
            usage = run.token_usage
            answer, source = render_chat(
                run.user_request,
                provider=self._llm_if_budget(run),
                usage=usage,
                # ★ 对话不是「无上下文的一问一答」。用户追问「你推荐哪个目标列」
                # 时，模型必须看得到上一轮跑出来的候选列 —— 少了这两项，它只能
                # 回答「没有看到你的数据字段」。
                history=self._chat_history(session, run),
                context=self._chat_context(session, run),
            )
            self._emit(
                run,
                EventType.CHAT,
                {"stage": "direct_chat", "final_answer": answer, "mode": "chat"},
            )
            return self._finish(run, RunStatus.COMPLETED, answer, source, started)

        steps = playbook.steps
        if run.plan is None:
            run.plan = {
                "goal": playbook.title,
                "steps": [{"title": s.title, "tool": s.tool, "action": s.tool} for s in steps],
                "notes": route_result.reason,
            }
            self._emit(
                run,
                EventType.PLANNING,
                {
                    "stage": "plan_ready",
                    "goal": playbook.title,
                    "steps": len(steps),
                    "tools": [s.tool for s in steps],
                },
            )

        run.status = RunStatus.RUNNING
        self.store.update_run(run)

        context = self._context(session, run)
        services = self._services()
        previous = self._restore_previous(run)
        collected: list[tuple[PlaybookStep, Any]] = self._restore_collected(run, steps)

        index = run.resume_step
        while index < len(steps) and index < MAX_STEPS:
            self._check_budget(run, started)
            if run.cancel_requested:
                raise AgentCancelled()

            step = steps[index]
            tool = self._get_tool(step.tool)
            if tool is None:
                return self._finish(
                    run, RunStatus.FAILED, "", None, started,
                    error=f"工具 {step.tool} 未注册，无法执行该分析",
                )

            params = self._build_params(run, session, step, tool, previous)
            missing = self._missing_slots(step, tool, params)

            if missing and step.fallback_tool:
                # 退化到粗粒度工具（如「哪一列」答不出就改看全表分布概览）。
                # 退化后必须重新判缺 —— 否则会用缺参数的兜底工具去撞 KeyError。
                step = PlaybookStep(
                    tool=step.fallback_tool,
                    title=step.title,
                    defaults=step.defaults,
                )
                tool = self._get_tool(step.tool)
                if tool is None:
                    return self._finish(
                        run, RunStatus.FAILED, "", None, started,
                        error=f"工具 {step.tool} 未注册，无法执行该分析",
                    )
                params = self._build_params(run, session, step, tool, previous)
                missing = self._missing_slots(step, tool, params)

            if missing:
                return self._ask(run, step, missing, index)

            # ---- 上一步跑成功了，但结论里有一个必须人来定的选择 ----
            # 先清掉「看起来填了、其实不是选项」的值：LLM 会把「创建一个叫
            # 客户流失预警 的 Workflow」里的**名字**填进 plan 槽，
            # _pending_choice 见它有值就放行，结果 build_and_run 报
            # 「没有匹配到方案」—— 用户压根没被问过选哪个。
            self._drop_invalid_choice(step, previous, params)
            if step.ask_if and self._pending_choice(step, previous, params):
                options, hint = _choice_options(previous, step)
                # 上一步显式给了问句时直接用它。``workflow.recommend`` 给出的是
                # 「基于当前数据集（…字段），我为你设计了 3 个方案：1… 2… 3…」，
                # 比通用的「需要补充参数：plan」有用得多 —— 用户要看着方案才能选。
                question = str(previous.get("question") or "") if isinstance(previous, dict) else ""
                if not question:
                    question = SLOT_QUESTIONS.get(step.ask_slot) or f"需要补充参数：{step.ask_slot}"
                    if hint:
                        question = f"{question}（候选：{hint}）"
                return self._ask(
                    run, step, [step.ask_slot], index,
                    options=options, question=question,
                )

            # ---- 权限裁决 ----
            key = f"{index}:{step.tool}"
            decision = self.tools.permission_manager.check(
                session.user_id, tool, context, params
            )
            if decision.denied:
                self._record_call(run, index, step, params, ToolCallStatus.FAILED, None, decision.reason)
                return self._finish(run, RunStatus.FAILED, "", None, started, error=decision.reason)
            if decision.needs_confirmation and run.authorized_key != key:
                return self._request_confirmation(run, step, params, index, decision.reason)

            # 凭据用掉即失效，避免一次确认放行整条高风险链
            run.authorized_key = None
            run.resume_step = index + 1
            self.store.update_run(run)

            # ---- 执行 ----
            self._emit(run, EventType.TOOL_CALL, {"tool": step.tool, "step_index": index})
            call_started = time.time()
            try:
                result = self.tools.execute(step.tool, params, context, services, confirmed=True)
            except ToolConfirmationRequired as exc:
                # 纵深防御：注册表仍要求确认 ⇒ 这一凭据没对上，回到挂起而不是判失败。
                # 「高风险操作」最坏的表现就是变成一句执行失败 —— 用户连选择权都没有。
                return self._request_confirmation(run, step, params, index, str(exc))
            except AppException as exc:
                result = _failed_result(str(exc))
            elapsed_ms = int((time.time() - call_started) * 1000)
            if elapsed_ms > SLOW_STEP_SECONDS * 1000:
                logger.warning("工具 %s 耗时 %.1fs（run=%s）", step.tool, elapsed_ms / 1000, run.id)

            ok = bool(getattr(result, "success", False))
            self._record_call(
                run, index, step, params,
                ToolCallStatus.OK if ok else ToolCallStatus.FAILED,
                result, "" if ok else _first_error(result),
            )
            self._emit(
                run,
                EventType.TOOL_RESULT,
                {
                    "tool": step.tool,
                    "step_index": index,
                    "status": "ok" if ok else "failed",
                    "summary": getattr(result, "summary", "") or "",
                },
            )

            if not ok:
                # 只有显式 false 才算校验失败 —— 前端同样按这个口径判断
                self._emit(
                    run,
                    EventType.VALIDATION,
                    {"valid": False, "step_index": index, "errors": list(getattr(result, "errors", []) or [])},
                )
                if not step.skippable:
                    return self._finish(
                        run, RunStatus.FAILED, "", None, started,
                        error=_first_error(result) or f"步骤「{step.title}」执行失败",
                    )
                # ★ 可跳过的步骤失败了也要进结果集。不进的话答案里根本看不到
                #   这次失败，模型只能把「没做成」包装成「可以按以下顺序搭建」
                #   —— 实测：工作流创建失败（ml.train 缺 target_column），
                #   答案却写「清洗与训练工作流可这样搭：1… 2…」，用户以为搭好了。
                collected.append((step, result))
            else:
                self._emit(run, EventType.VALIDATION, {"valid": True, "step_index": index})
                previous = getattr(result, "data", None) or {}
                collected.append((step, result))
                # 只有「改写数据状态」的写操作才推进基线；聚合/合并另存结果，不动基线。
                self._advance_baseline(session, tool, params, result)

            index += 1

        # ---- 渲染答案 ----
        answer, source = render(
            run.user_request,
            collected,
            provider=self._llm_if_budget(run),
            usage=run.token_usage,
            columns=self._known_columns(session),
        )
        return self._finish(run, RunStatus.COMPLETED, answer, source, started)

    # ==================================================================
    # 挂起点
    # ==================================================================
    def _ask(
        self,
        run: AgentRun,
        step: PlaybookStep,
        missing: list[str],
        index: int,
        *,
        options: list[ClarificationOption] | None = None,
        question: str = "",
    ) -> AgentRun:
        """必填槽位缺失，挂起等用户补充。"""
        slot = missing[0]
        request = ClarificationRequest(
            question=question or SLOT_QUESTIONS.get(slot) or f"需要补充参数：{'、'.join(missing)}",
            code=f"slot.{slot}",
            step_index=index,
            tool=step.tool,
            options=options or [],
        )
        run.pending_clarification = request
        run.status = RunStatus.WAITING_CLARIFICATION
        run.resume_step = index
        self._emit(
            run,
            EventType.CLARIFICATION,
            {**request.to_dict(), "stage": "asked", "missing": missing},
        )
        self.store.update_run(run)
        return run

    def _request_confirmation(
        self, run: AgentRun, step: PlaybookStep, params: dict[str, Any], index: int, reason: str
    ) -> AgentRun:
        """高风险操作，挂起等用户确认。"""
        run.pending_confirmation = PermissionRequest(
            tool=step.tool,
            arguments=params,
            step_index=index,
            reason=reason,
        )
        run.status = RunStatus.WAITING_CONFIRMATION
        run.resume_step = index
        self._emit(
            run,
            EventType.PERMISSION,
            {**run.pending_confirmation.to_dict()},
        )
        self.store.update_run(run)
        return run

    def _remember_answer(
        self, run: AgentRun, status: RunStatus, charts: list[dict] | None = None
    ) -> None:
        """把助手这一轮的话写进会话历史。

        只写用户消息、不写助手消息，是「去别的模块逛一圈回来，AI 的回复全没了、
        只剩下我的提问」的直接原因：会话历史是前端重建对话的**唯一数据源**
        （``useAiSession.activate`` 直接把 ``session.history`` 映射成消息列表），
        而这一半从来没被写进去过。

        失败与成功都要写：用户需要知道这一步为什么没结果，
        而不是看到一句孤零零的提问。
        """
        if not run.session_id:
            return
        text = run.final_answer if status is RunStatus.COMPLETED else (run.error or "")
        if not text.strip():
            return
        try:
            self.store.append_history(
                run.session_id,
                "assistant",
                text.strip(),
                charts=charts if charts is not None else _collect_charts(run),
            )
        except Exception:  # noqa: BLE001 - 历史写入失败不该让运行失败
            logger.warning("写入助手消息到会话历史失败（run=%s）", run.id, exc_info=True)

    @staticmethod
    def _drop_invalid_choice(step: PlaybookStep, previous: Any, params: dict[str, Any]) -> None:
        """把不是合法选项的选择值清掉，让链路回到「挂起问人」。

        上一步给了候选选项时，「有值」不等于「选过了」：值可能是模型从同一句话里
        顺手填进来的别的槽位（最典型的是把 Workflow 的名字填成方案序号）。
        留着它，用户就永远等不到那个本该出现的选择题。
        """
        if not step.ask_if or not isinstance(previous, dict):
            return
        if not previous.get(step.ask_if):
            return
        value = params.get(step.ask_slot)
        if value in (None, "", []):
            return
        raw = previous.get(step.ask_options_key)
        valid: set[str] = set()
        if isinstance(raw, dict):
            for group_values in raw.values():
                if isinstance(group_values, (list, tuple)):
                    valid.update(str(v) for v in group_values)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    if item.get("value") is not None:
                        valid.add(str(item["value"]))
                else:
                    valid.add(str(item))
        if not valid:
            return
        if str(value) not in valid:
            del params[step.ask_slot]

    @staticmethod
    def _pending_choice(step: PlaybookStep, previous: Any, params: dict[str, Any]) -> bool:
        """上一步标记了「需要人来选」，而这一步还没拿到答案。

        答案已存在时必须放行 —— 否则用户答完 target 仍会被反复追问同一个问题。
        """
        if not isinstance(previous, dict):
            return False
        if not previous.get(step.ask_if):
            return False
        return params.get(step.ask_slot) in (None, "", [])

    # ------------------------------------------------------------------
    # 对话路径的上下文（历史 + 数据概况 + 上一次结论）
    # ------------------------------------------------------------------
    def _chat_history(self, session: AgentSession, run: AgentRun) -> list[dict[str, str]]:
        """会话里已经发生过的对话，不含本轮请求（它作为 user 单独进 prompt）。"""
        items = list(session.history or [])
        # send_message 先把本轮请求追加进历史，这里要剔除，否则用户的话会出现两次
        if items and items[-1].get("content") == run.user_request:
            items = items[:-1]
        return [dict(item) for item in items[-_CHAT_HISTORY_MESSAGES:]]

    def _chat_context(self, session: AgentSession, run: AgentRun) -> str:
        """对话可用的事实：绑定了哪些数据集、上一次分析得出了什么。

        它们是**工具跑出来的真实结果**，不是模型可以编的；把它们放进上下文，
        追问才有据可依。数据集概况只读 Parquet schema，不加载数据行。
        """
        lines: list[str] = []
        for dataset_id in list(session.dataset_ids or [])[:_CHAT_CONTEXT_MAX_DATASETS]:
            brief = self._dataset_brief(dataset_id, session)
            if brief:
                lines.append(f"- {brief}")
        previous = self._last_answer(session, run)
        if previous:
            # 结论是**当时绑定的数据集**的。不标注的话，用户中途换数据集之后
            # 模型会把上一张表的行列数说成当前这张的。
            lines.append(
                f"- 本会话上一次分析的结论（针对**当时**绑定的数据集，"
                f"不一定等于上面列出的当前数据集）：{previous}"
            )
        return "\n".join(lines)

    def _dataset_brief(self, dataset_id: int, session: AgentSession | None = None) -> str:
        """一行数据集概况：名称 + id + 规模 + 列名（取前若干个）。"""
        if dataset_id is None:
            return ""
        name = ""
        try:
            service = self._services().dataset_service
            if service is not None:
                name = str(service.get(dataset_id).name or "")
        except Exception:  # noqa: BLE001 - 名字只是修饰，读不到不影响回答
            name = ""
        label = f"{name}（id={dataset_id}）" if name else f"数据集 id={dataset_id}"
        shape = self._dataset_shape(dataset_id, session)
        columns = self._schema_columns(dataset_id)
        if not columns:
            return f"{label}，{shape}" if shape else label
        shown = "、".join(columns[:_CHAT_CONTEXT_COLUMNS])
        tail = f" 等 {len(columns)} 列" if len(columns) > _CHAT_CONTEXT_COLUMNS else ""
        return f"{label}，{shape}，列名：{shown}{tail}".strip("，")

    def _dataset_shape(self, dataset_id: int, session: AgentSession | None = None) -> str:
        """当前基线版本的真实「vN，R 行 × C 列」。

        对话路径此前只有列名没有行数，模型就把**上一次**分析的行数搬到了
        刚换过来的数据集上（实测：换成 id=12 后问「那这个呢」，答
        「它是 21 行 18 列」—— 21×18 是换表之前那张表的数）。
        """
        version = self.baseline_version(session, dataset_id) if session is not None else None
        service = self._services().dataset_service
        if service is None:
            return ""
        try:
            frame = service.scan_version(dataset_id, version)
            rows = int(frame.select(pl.len()).collect().item())
            cols = len(frame.collect_schema().names())
        except Exception:  # noqa: BLE001
            return ""
        return f"v{version or '最新'}，{rows} 行 × {cols} 列"

    def _last_answer(self, session: AgentSession, run: AgentRun) -> str:
        """本会话上一次成功运行给出的结论（截断后）。"""
        for run_id in reversed(list(session.run_ids or [])):
            if run_id == run.id:
                continue
            previous = self.store.get_run(run_id)
            if previous is None or not (previous.final_answer or "").strip():
                continue
            return " ".join(previous.final_answer.split())[:_CHAT_CONTEXT_ANSWER_CHARS]
        return ""

    # ==================================================================
    # 收尾
    # ==================================================================
    def _finish(
        self,
        run: AgentRun,
        status: RunStatus,
        answer: str,
        source: Any,
        started: float,
        *,
        error: str = "",
    ) -> AgentRun:
        run.status = status
        run.finished_at = time.time()
        run.pending_confirmation = None
        run.pending_clarification = None
        run.authorized_key = None
        if error:
            run.error = error
        if status is RunStatus.COMPLETED:
            run.final_answer = answer or "已完成，但没有产生可展示的结论。"
            run.answer_source = source
        else:
            # 失败也要有话可说：只有一句英文/技术报错（如
            # 「unsupported aggregation func: None」）时，用户既不知道哪一步挂了，
            # 也不知道下一步该怎么问。
            run.final_answer = _failure_answer(error, run)
            run.answer_source = source

        # 本轮画出来的图（内联 SVG）。必须随 completed 事件一起下发：
        # 前端靠这条事件把助手消息追加到对话框，图不跟着走就永远显示不出来。
        charts = _collect_charts(run)
        self._remember_answer(run, status, charts=charts)
        self._emit(run, EventType.USAGE, {"token_usage": run.token_usage.to_dict()})
        if status is RunStatus.COMPLETED:
            self._emit(
                run,
                EventType.COMPLETED,
                {
                    "final_answer": run.final_answer,
                    "answer_source": source.to_dict() if source else None,
                    "charts": charts,
                },
            )
        else:
            # final_answer 一起带出去：前端 ``failed`` 分支优先显示它，
            # 否则用户看到的是一句原始技术报错。
            self._emit(
                run,
                EventType.FAILED,
                {"error": run.error or "未知错误", "final_answer": run.final_answer, "charts": charts},
            )
        self.store.update_run(run)
        return run

    # ==================================================================
    # 辅助
    # ==================================================================
    def _emit(self, run: AgentRun, event_type: EventType, payload: dict[str, Any] | None = None) -> AgentEvent:
        return self.store.add_event(run, event_type, payload)

    def _check_budget(self, run: AgentRun, started: float) -> None:
        if time.time() - started > WALL_CLOCK_SECONDS:
            raise AgentTimeout()

    def _llm_if_budget(self, run: AgentRun) -> LLMProvider | None:
        """预算内才返回 provider；超预算返回 None，链路自动退回规则路径。"""
        if self.llm is None:
            return None
        if run.token_usage.llm_calls >= MAX_LLM_CALLS:
            return None
        return self.llm

    @staticmethod
    def _missing_slots(step: PlaybookStep, tool: Any, params: dict[str, Any]) -> list[str]:
        """必填但还没定下来的参数。

        除了 playbook 显式声明的 ``required_slots``，工具 schema 里声明的
        ``dataset_id`` / ``left_dataset_id`` 也算必填：会话没绑定数据集时若直接
        传 None，工具内部 ``int(params["dataset_id"])`` 会 KeyError，
        用户只看到一次莫名其妙的失败，不如直接问。
        """
        missing = [s for s in step.required_slots if params.get(s) in (None, "", [])]
        props = ((tool.input_schema or {}).get("properties") or {}) if tool else {}
        for implicit in ("dataset_id", "left_dataset_id", "right_dataset_id"):
            if implicit in props and params.get(implicit) is None and implicit not in missing:
                missing.append(implicit)
        return missing

    def _get_tool(self, name: str) -> Any | None:
        try:
            return self.tools.get(name)
        except AppException:
            return None

    def _context(self, session: AgentSession, run: AgentRun) -> ToolExecutionContext:
        # 单租户本地平台：权限全给，真正的闸门是风险等级（HIGH 需人工确认）。
        # dataset_ids 三态必须保留：空集合表示「一个都不许访问」，
        # 与 None（不限制）语义不同，混在一起会让 Agent 静默读到任意数据集。
        return ToolExecutionContext(
            user_id=session.user_id,
            session_id=session.id,
            dataset_ids=set(session.dataset_ids) if session.dataset_ids else set(),
            permissions=set(Permission),
            extra={"user_request": run.user_request},
        )

    def baseline_version(self, session: AgentSession, dataset_id: int | None) -> int | None:
        """该数据集在本会话里的**基线版本**（首次用到时把当前 latest 固化下来）。

        固化的意义：写类工具随时可能在数据集上追加新版本，而新版本即 latest。
        不固化，下一次请求就会读到「别人产出的中间结果」。
        """
        if dataset_id is None:
            return None
        key = int(dataset_id)
        cached = session.baseline_versions.get(key)
        if cached:
            return int(cached)
        service = self._services().dataset_service
        if service is None:
            return None
        try:
            row = service.get_version_row(key, None)
        except Exception:  # noqa: BLE001 - 数据集不可读时退回「不指定版本」
            logger.debug("解析数据集 %s 的基线版本失败", key, exc_info=True)
            return None
        if row is None:
            return None
        version = int(getattr(row, "version", 0) or 0)
        if version <= 0:
            return None
        # ★ latest 是 0 行时不能当基线。一次没匹配上的筛选就会产出 0 行的新版本，
        #   之后**开新会话**也锚在它上面 —— 用户看到的是「这份数据 0 行，
        #   画不出图」，而数据明明还在（第十二轮实测）。往前找最近的非空版本。
        if int(getattr(row, "row_count", 0) or 0) == 0:
            fallback = self._latest_non_empty_version(key)
            if fallback:
                logger.warning(
                    "数据集 %s 的最新版本 v%s 行数为 0，基线回退到最近的非空版本 v%s",
                    key, version, fallback,
                )
                version = fallback
        session.baseline_versions[key] = version
        self.store.update_session(session)
        return version

    def _latest_non_empty_version(self, dataset_id: int) -> int | None:
        """最近一个**有数据行**的版本号（从新往旧找）。找不到返回 None。"""
        service = self._services().dataset_service
        if service is None:
            return None
        try:
            versions, _total = service.get_versions(dataset_id, page=1, page_size=50)
        except Exception:  # noqa: BLE001
            return None
        for row in sorted(versions, key=lambda v: int(getattr(v, "version", 0) or 0), reverse=True):
            if int(getattr(row, "row_count", 0) or 0) > 0:
                return int(getattr(row, "version", 0) or 0) or None
        return None

    def _advance_baseline(
        self,
        session: AgentSession,
        tool: Any,
        params: dict[str, Any],
        result: Any,
    ) -> None:
        """写操作成功后推进基线（只限真正改写数据状态的那几类操作）。"""
        if not _advances_baseline(tool):
            return
        produced = _produced_version(result)
        if produced is None:
            return
        raw = params.get("dataset_id")
        if raw is None:
            return
        try:
            dataset_id = int(raw)
        except (TypeError, ValueError):
            return
        # ★ 结果为空的新版本不能当基线。「筛选 customer_id > 99999999」产出 0 行后
        #   基线被切到那张空表，之后用户问「有多少行」答 0、画图也全空 ——
        #   数据明明还在，却被一次没匹配上的筛选「清空」了（第十一轮实测）。
        if self._version_rows(dataset_id, produced) == 0:
            logger.warning("写操作 %s 产出的 v%s 行数为 0，不推进基线", getattr(tool, "name", tool), produced)
            warnings = getattr(result, "warnings", None)
            if isinstance(warnings, list):
                warnings.append(
                    "本次结果为空（0 行），因此没有把当前版本切换过去，后续分析仍基于操作前的版本"
                )
            return
        if session.baseline_versions.get(dataset_id) == produced:
            return
        session.baseline_versions[dataset_id] = produced
        self.store.update_session(session)

    def _version_rows(self, dataset_id: int, version: int) -> int | None:
        """某版本的行数。读不出来返回 None（此时不做任何判断）。"""
        service = self._services().dataset_service
        if service is None:
            return None
        try:
            frame = service.scan_version(dataset_id, version)
            return int(frame.select(pl.len()).collect().item())
        except Exception:  # noqa: BLE001
            return None

    def _schema_columns(self, dataset_id: int | None, version: int | None = None) -> list[str] | None:
        """读列名（只读 Parquet schema，不加载数据行）。失败返回 None。

        ``version`` 必须传基线版本：不传就读 latest，而 latest 可能已经是某次
        聚合产出的中间结果表（7 行 × 2 列）—— 槽位抽取器拿到的候选列名会全是
        聚合结果的列名，用户点名的原始列一个都不在候选里，只能瞎挑一个。
        """
        if dataset_id is None:
            return None
        service = self._services().dataset_service
        if service is None:
            return None
        try:
            frame = service.scan_version(dataset_id, version)
            return list(frame.collect_schema().names())
        except Exception:  # noqa: BLE001
            logger.debug("读取数据集 %s 的列结构失败", dataset_id, exc_info=True)
            return None

    def _build_params(
        self,
        run: AgentRun,
        session: AgentSession,
        step: PlaybookStep,
        tool: Any,
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        """组装工具入参：defaults → 自动槽位 → carry → 用户补充 → 规则/LLM 抽取。"""
        props = ((tool.input_schema or {}).get("properties") or {}) if tool else {}

        auto: dict[str, Any] = {}
        primary = session.dataset_ids[0] if session.dataset_ids else None
        secondary = session.dataset_ids[1] if len(session.dataset_ids) > 1 else None
        if "dataset_id" in props and primary is not None:
            auto["dataset_id"] = primary
        if "left_dataset_id" in props and primary is not None:
            auto["left_dataset_id"] = primary
        # 用户说「这两个数据集」时，会话里已经绑好了第二个 —— 不填就是逼他手输 id。
        # 只绑了一个时才留空，由 _missing_slots 反问。
        if "right_dataset_id" in props and secondary is not None:
            auto["right_dataset_id"] = secondary
        # ★ 版本一律锚在会话基线上，绝不留空去读 latest。
        #   latest 可能已被上一次聚合/合并顶成中间结果表，读它等于在废表上做分析。
        if "version" in props:
            base = self.baseline_version(session, primary)
            if base is not None:
                auto["version"] = base
        for param, key in (step.carry or {}).items():
            if isinstance(previous, dict) and key in previous:
                auto[param] = previous[key]
        # 用户此前针对本步骤补充的答案（键可能带 slot. 前缀）
        for param in step.required_slots:
            # 结构化参数（conditions / aggregations / nodes …）在这里也要挡：
            # 上一版只挡住了下面的 params 循环，漏了这里，于是自然语言答案
            # 照样被原样塞进 conditions —— 澄清之后依然必失败。
            if any(
                _needs_parsing(param, run.clarification_answers.get(key))
                for key in (param, f"slot.{param}")
            ):
                continue
            for key in (param, f"slot.{param}"):
                if key in run.clarification_answers:
                    auto[param] = run.clarification_answers[key]
                    break

        params: dict[str, Any] = dict(step.defaults or {})
        # 回指复用：「那个图换成 30 个分箱」里的图表名与列名来自上一轮同名工具。
        # 放在 defaults 之后、auto/extracted 之前 —— 它只补**没说到的**参数，
        # 用户这轮点名的新值必须能盖掉它。
        previous_args: dict[str, Any] = {}
        if getattr(step, "reuse_previous_args", False):
            previous_args = self._previous_tool_args(session, run, step.tool)
            for key, value in previous_args.items():
                params.setdefault(key, value)
        params.update(auto)
        # 澄清答案的键可能是「slot.xxx」或「xxx」——两种都要认，
        # 认错一种的代价是同一个问题被反复问下去。
        #
        # ★ 但**结构化参数不能原样照搬**：用户补充的是一句自然语言
        #   （「credit_amount 大于 5000 且 age 大于 60」），直接塞进 conditions
        #   就变成了字符串，filter 拿到它直接抛异常 —— 实测澄清后必失败，
        #   用户被反问两次最后还是没跑成。自然语言要交给抽取器解析。
        clarify_text = ""
        for key, value in (run.clarification_answers or {}).items():
            name = key.split(".", 1)[1] if key.startswith("slot.") else key
            if _needs_parsing(name, value):
                clarify_text += f" {value}"
                continue
            params[name] = value
        clarify_text = clarify_text.strip()

        # 列名类参数（column / conditions / group_by / x / y / target / name）内部都嵌列名，
        # 必须把真实列名传给抽取器：否则模型只能凭「出发延误」猜出 departure_delay，
        # 而真实列名是 DepDelay（真实事故：filter 条件列名填错 → 筛选静默失效）。
        _col_params = ("column", "columns", "conditions", "group_by", "x", "y", "target", "name", "expression")
        needs_schema = any(s == "column" for s in step.required_slots) or any(k in props for k in _col_params)
        columns = self._schema_columns(primary, self.baseline_version(session, primary)) if needs_schema else None
        # 模型看不到会话状态：编排工作流时它不知道 dataset_id 该填几，
        # 只能凭空写一个。把已绑定的数据集 id 明确告诉它。
        hint = ""
        if primary is not None:
            hint = f"会话绑定的数据集 id = {primary}"
            if secondary is not None:
                hint += f"，第二个数据集 id = {secondary}"
        extracted = extract_slots(
            f"{run.user_request} {clarify_text}".strip() if clarify_text else run.user_request,
            step,
            schema_columns=columns,
            provider=self._llm_if_budget(run),
            usage=run.token_usage,
            # ★ 让抽取器看见 enum / 描述。少了这个，模型只拿到 ["model"] 这种
            # 光秃秃的键名，只能填 auto 或省略 —— 用户点名的模型与超参一个都不会生效。
            tool_schema=(tool.input_schema or {}) if tool else {},
            # ★ 编排类工具（workflow.build_and_run）的合法节点类型写在 description 里，
            # 不给它，模型只能自创节点类型，创建必失败。
            tool_description=str(getattr(tool, "description", "") or ""),
            hint=hint,
        )
        # 抽取结果只补空缺，不覆盖已经确定的自动值；
        # 但**可微调槽位**例外：它是「用户点名了就照办」的参数，
        # 必须能盖掉 defaults（否则 defaults 的 model="auto" 永远赢）。
        tuned = set(getattr(step, "tuned_slots", ()))
        # ★ 用户**已经选定**的槽位，不许被重新抽取覆盖。
        #   否则会出现死循环：用户选了「1」→ 抽取器又从原始请求里抽出
        #   「客户流失预警」填进 plan → 不是合法选项 → 再问一遍。
        #
        #   只保护**直接采用**的那些答案。自然语言答案（如「credit_amount 大于
        #   5000 且 age 大于 60」）本来就要交给抽取器解析成 conditions，
        #   把它们也算进 answered 会让解析结果被挡掉 —— 那才是澄清后必失败的老毛病。
        answered = {
            (k.split(".", 1)[1] if k.startswith("slot.") else k)
            for k, v in (run.clarification_answers or {}).items()
            if not _needs_parsing(k.split(".", 1)[1] if k.startswith("slot.") else k, v)
        }
        for key, value in extracted.items():
            if key in answered:
                continue
            if key in tuned or params.get(key) in (None, "", []):
                params[key] = value

        # ★ 回指合并必须放在**抽取之后**：columns 是可微调槽位，抽取器会用本轮的
        #   「总费用」把合并结果覆盖掉，回指就白做了（第十三轮实测）。
        if getattr(step, "reuse_previous_args", False):
            self._merge_previous_lists(
                step, previous_args, params,
                fallback=self._recent_columns(session, run) if getattr(step, "union_previous", ()) else None,
            )
            # ★ 「画个图看看这两个字段的关系」：图表类型还停在默认的直方图，x/y 也抽不到
            #   —— 用户说的是**上一轮那两个字段**，链路上却当成一次全新的单列直方图
            #   （第十三轮实测：画出来的是 tenure_months 单列，跟「两个字段」毫无关系）。
            if step.tool == "eda.visualize":
                recent: list[str] = []
                if _TWO_THINGS_RE.search(run.user_request or "") and str(
                    params.get("chart") or ""
                ) == "histogram":
                    recent = self._recent_columns(session, run, limit=2)
                    if len(recent) >= 2 and not params.get("x") and not params.get("y"):
                        params["chart"] = "scatter"
                        params["x"], params["y"] = recent[0], recent[1]
                # ★ 双列图表（散点/折线）缺 x/y：用户说的是上一轮那两个字段，
                #   直接报「散点图需要两个数值列」等于什么都没画（100 条多轮压测实测）。
                elif str(params.get("chart") or "") in ("scatter", "line"):
                    if not (params.get("x") and params.get("y")):
                        recent = self._recent_columns(session, run, limit=2)
                        if len(recent) >= 2:
                            params["x"], params["y"] = recent[0], recent[1]
            # ★ 「刚才那个目标列，换成随机森林再跑一次」：用户没再点名，
            #   不记住上一次的 target，这一轮会退化成 detect 推断的另一个列，
            #   跑出来的模型和上一轮完全对不上（Telco 压测实测）。
            if step.tool in ("ml.detect_task", "ml.train") and not params.get("target"):
                if _MODEL_REFERENCE_RE.search(run.user_request or ""):
                    remembered = self._last_target(session, run)
                    if remembered:
                        params["target"] = remembered
        return params

    def _record_call(
        self,
        run: AgentRun,
        index: int,
        step: PlaybookStep,
        params: dict[str, Any],
        status: ToolCallStatus,
        result: Any,
        error: str,
    ) -> None:
        run.tool_calls = [c for c in run.tool_calls if c.step_index != index]
        run.tool_calls.append(
            ToolCall(
                step_index=index,
                tool=step.tool,
                arguments=dict(params),
                attempt=1,
                status=status,
                result=result.to_dict() if result is not None else None,
                error=error,
            )
        )
        self.store.update_run(run)

    def _previous_tool_args(self, session: Any, run: AgentRun, tool: str) -> dict[str, Any]:
        """本会话上一次**成功**调用 ``tool`` 时用的参数（不含本轮）。

        「那个图换成 30 个分箱」「再画一遍」这类回指，缺的不是权限也不是数据，
        而是**上一轮那张图是什么**。这里从会话历史里把它找回来。

        只认成功调用：失败的参数本身就是错的，复用它等于把同一个坑踩两遍。
        """
        for run_id in reversed(list(getattr(session, "run_ids", None) or [])):
            if run_id == run.id:
                continue
            previous = self.store.get_run(run_id)
            if previous is None:
                continue
            for call in reversed(list(previous.tool_calls or [])):
                if call.tool != tool or call.status is not ToolCallStatus.OK:
                    continue
                if isinstance(call.arguments, dict) and call.arguments:
                    return dict(call.arguments)
        return {}

    def _recent_columns(self, session: Any, run: Any, limit: int = 2) -> list[str]:
        """本会话**最近被点名过的列名**（最近的在前）。

        ★ 回指不一定落在同一个工具上：「那个月费的分布」走的是 ``eda.distribution``，
          下一句「再看看它跟总费用的关系」走的是 ``eda.correlation`` ——
          只看「上一次同名调用」什么都拿不到，而用户指的明明就是上一句那列
          （第十三轮实测）。这里跨工具找最近聊到的列。
        """
        found: list[str] = []
        for run_id in reversed(list(getattr(session, "run_ids", None) or [])):
            if run_id == getattr(run, "id", None):
                continue
            previous = self.store.get_run(run_id)
            if previous is None:
                continue
            for call in reversed(list(previous.tool_calls or [])):
                if call.status is not ToolCallStatus.OK:
                    continue
                args = call.arguments or {}
                for key in _FOCUS_COLUMN_KEYS:
                    value = args.get(key)
                    if isinstance(value, str) and value:
                        found.append(value)
                    elif isinstance(value, list):
                        found.extend(v for v in value if isinstance(v, str) and v)
            if len(found) >= limit:
                break
        return list(dict.fromkeys(found))[:limit]

    def _last_target(self, session: Any, run: Any) -> str | None:
        """本会话上次**实际用过**的目标列（只看成功调用）。"""
        for run_id in reversed(list(getattr(session, "run_ids", None) or [])):
            if run_id == getattr(run, "id", None):
                continue
            previous = self.store.get_run(run_id)
            if previous is None:
                continue
            for call in reversed(list(previous.tool_calls or [])):
                if call.status is not ToolCallStatus.OK:
                    continue
                value = (call.arguments or {}).get("target")
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _merge_previous_lists(
        step: Any,
        previous_args: dict[str, Any],
        params: dict[str, Any],
        fallback: list[str] | None = None,
    ) -> None:
        """并集型参数：本轮点名的新值要和上一轮的**合并**，不是替换。

        ★ 相关性是「多列两两比较」：「那个月费的分布 → 再看看它跟总费用的关系」
          里本轮只抽得到 total_charges，替换掉上一轮的 monthly_charges 就只剩
          一列，工具直接报「至少需要 2 个数值字段」（第十二轮实测）。

        只作用于 :attr:`PlaybookStep.union_previous` 登记的多值参数；
        标量参数（bins / chart）替换才是对的。

        ``fallback`` 是跨工具找回的「最近聊到的列」：上一次**同名**调用不存在时用它，
        否则「分布 → 再看看它跟 X 的关系」这类跨工具回指拿不到第二列。
        """
        for key in getattr(step, "union_previous", ()) or ():
            old = previous_args.get(key)
            if not isinstance(old, list):
                old = list(fallback or [])
            new = params.get(key) if isinstance(params.get(key), list) else []
            merged = list(dict.fromkeys([*old, *new]))[:_UNION_MAX_COLUMNS]
            if merged:
                params[key] = merged

    def _known_columns(self, session: Any) -> list[str]:
        """给答案渲染用的真实列名。

        ★ 答案里出现过「画 credit_amount 的直方图」这种建议 —— 那个字段根本不在
          这份数据里，是模型从训练语料带出来的（第十轮实测）。用户照着说一句，
          只会得到一次失败。把真实列名交给渲染层钉死可用范围，比在 prompt 里
          喊一百遍「不许编造列名」有效。

        复用 :meth:`_schema_columns`（它按**基线版本**读，避免读到聚合中间表的列名）。
        """
        dataset_id = next(iter(list(getattr(session, "dataset_ids", None) or [])), None)
        if dataset_id is None:
            return []
        try:
            version = self.baseline_version(session, dataset_id)
            return list(self._schema_columns(dataset_id, version) or [])
        except Exception:  # noqa: BLE001 - 列名只是给 prompt 用的提示，拿不到就不约束
            return []

    def _restore_previous(self, run: AgentRun) -> dict[str, Any]:
        """恢复上一步结果（用于 carry）。只取 resume_step 之前的成功调用。"""
        for call in reversed(run.tool_calls):
            if call.step_index < run.resume_step and call.status is ToolCallStatus.OK:
                data = (call.result or {}).get("data")
                if isinstance(data, dict):
                    return data
        return {}

    def _restore_collected(self, run: AgentRun, steps: tuple[PlaybookStep, ...]) -> list[tuple[PlaybookStep, Any]]:
        """恢复已完成步骤的结果，供答案渲染使用（跨挂起恢复时不能丢）。"""
        by_index = {i: s for i, s in enumerate(steps)}
        collected: list[tuple[PlaybookStep, Any]] = []
        for call in sorted(run.tool_calls, key=lambda c: c.step_index):
            if call.step_index >= run.resume_step:
                continue
            if call.status is not ToolCallStatus.OK:
                continue
            step = by_index.get(call.step_index)
            restored = _RestoredResult.from_call(call)
            if step is not None and restored is not None:
                collected.append((step, restored))
        return collected


#: 反问时最多给多少个候选选项。选项太多会让选择器变成一个下拉长列表，
#: 而「拿不定就自己输」仍然可行（前端保留自由输入）。
_MAX_CHOICE_OPTIONS = 8


def _choice_options(previous: dict[str, Any], step: PlaybookStep) -> tuple[list[ClarificationOption], str]:
    """把上一步结果里的候选值整理成可点选的选项 + 一句候选提示。

    候选既可能是列表（``["a", "b"]``），也可能是按类别分组的字典
    （``{"regression": [...], "classification": [...]}``）。后者要展平，
    并把类别写进 note —— 用户看到的是「fare_amount（适合回归）」，
    比一串裸列名更容易决定。
    """
    raw = previous.get(step.ask_options_key) if step.ask_options_key else None
    options: list[ClarificationOption] = []
    labels: list[str] = []

    if isinstance(raw, dict):
        for group, values in raw.items():
            if not isinstance(values, (list, tuple)):
                continue
            note = _GROUP_LABELS.get(str(group), str(group))
            for value in list(values)[:4]:
                text = str(value)
                if text in labels:
                    continue
                labels.append(text)
                options.append(ClarificationOption(value=text, label=text, note=note))
    elif isinstance(raw, (list, tuple)):
        for value in raw:
            # 方案类候选自带 value/label/note（如 workflow.recommend 的 1/2/3），
            # 直接展平会变成一串 dict 的字符串化 —— 必须按字段取。
            if isinstance(value, dict):
                text = str(value.get("value") or "")
                if not text:
                    continue
                options.append(
                    ClarificationOption(
                        value=text,
                        label=str(value.get("label") or text),
                        note=str(value.get("note") or ""),
                    )
                )
                labels.append(text)
                continue
            text = str(value)
            if text in labels:
                continue
            labels.append(text)
            options.append(ClarificationOption(value=text, label=text, note=""))

    options = options[:_MAX_CHOICE_OPTIONS]
    hint = "、".join(labels[:5])
    return options, hint


def _first_error(result: Any) -> str:
    errors = getattr(result, "errors", None) or []
    return str(errors[0]) if errors else ""


def _failed_result(message: str) -> Any:
    from app.tools.result import ToolResult

    return ToolResult.fail(message)


_REGISTRY: ToolRegistry | None = None


def _default_registry() -> ToolRegistry:
    """默认注册表（内置工具）。延迟导入，避免 import 期就触发注册。"""
    global _REGISTRY
    if _REGISTRY is None:
        from app.tools.builtin import TOOL_REGISTRY

        _REGISTRY = TOOL_REGISTRY
    return _REGISTRY


_SESSION_FACTORY: Callable[[], Any] | None = None


def _default_session_factory() -> Callable[[], Any]:
    """默认会话工厂。延迟导入，避免在模块导入期就建立数据库引擎。"""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        from app.core.database import SessionLocal

        _SESSION_FACTORY = SessionLocal
    return _SESSION_FACTORY
