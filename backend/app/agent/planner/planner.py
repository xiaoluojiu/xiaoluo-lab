"""Agent Planner。"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from pydantic import ValidationError

from app.agent.context.budget import ContextBudget
from app.agent.context.models import AgentContext
from app.agent.llm.base import LLMProvider
from app.agent.planner.models import AgentPlan, PlanStep
from app.core.config import settings
from app.core.exceptions import AgentException


class PlanInvalidError(AgentException):
    """计划非法。"""
    http_status = 400
    default_code = "AGENT_PLAN_INVALID"
    default_message = "Agent plan invalid"


class AgentPlanner:
    """只负责规划，不读取数据、不执行工具。"""

    def __init__(self, llm: LLMProvider | None = None, *, max_steps: int = 6) -> None:
        self.llm = llm
        self.max_steps = max_steps
        # 进程级共享（deps 注入单例）后存在多线程并发访问，plan cache 必须加锁。
        self._plan_cache: OrderedDict[str, AgentPlan] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.plan_cache_hits = 0
        self.plan_cache_misses = 0

    def build_plan(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> AgentPlan:
        if self.llm is not None:
            cache_key = self._cache_key(user_request, context, tools)
            if settings.AGENT_ENABLE_PLAN_CACHE:
                cached = self._cache_get(cache_key)
                if cached is not None:
                    with self._cache_lock:
                        self.plan_cache_hits += 1
                    cached.cache_hit = True
                    return self._validate(cached.model_copy(deep=True), tools, context)
                with self._cache_lock:
                    self.plan_cache_misses += 1
            plan = self._llm_plan(user_request, context, tools)
            plan = self._validate(plan, tools, context)
            if settings.AGENT_ENABLE_PLAN_CACHE:
                self._cache_put(cache_key, plan)
            return plan
        return self._validate(self._rule_plan(user_request, context), tools, context)

    def build_plan_resilient(
        self,
        user_request: str,
        context: AgentContext,
        tools: list[dict[str, Any]],
        *,
        all_tools: list[dict[str, Any]] | None = None,
    ) -> AgentPlan:
        """带降级的规划入口。

        LLM 规划的失败模式是「计划引用了候选集外的工具」→ PlanInvalidError，
        过去会直接让整次 Agent 运行失败（用户看到的却是「工具调用失败」）。
        这里做三级降级：
        1. 候选工具集规划；
        2. 失败则放开到全量工具集重新规划（工具已注册 ≠ 被召回，这一步直接消除该类错误）；
        3. 再失败则用规则规划兜底，保证数据分析请求至少能跑出真实结果。
        """
        try:
            return self.build_plan(user_request, context, tools)
        except PlanInvalidError:
            pass
        if all_tools and len(all_tools) > len(tools):
            try:
                return self.build_plan(user_request, context, all_tools)
            except PlanInvalidError:
                pass
        return self._validate(self.rule_plan(user_request, context), all_tools or tools, context)

    def rule_plan(self, user_request: str, context: AgentContext) -> AgentPlan:
        """公开的规则规划入口（降级兜底时使用）。"""
        return self._rule_plan(user_request, context)

    def _cache_get(self, key: str) -> AgentPlan | None:
        with self._cache_lock:
            cached = self._plan_cache.get(key)
            if cached is not None:
                self._plan_cache.move_to_end(key)
            return cached

    def _cache_put(self, key: str, plan: AgentPlan) -> None:
        with self._cache_lock:
            self._plan_cache[key] = plan.model_copy(deep=True)
            self._plan_cache.move_to_end(key)
            while len(self._plan_cache) > settings.AGENT_PLAN_CACHE_MAX_ITEMS:
                self._plan_cache.popitem(last=False)

    def _context_budget(self) -> ContextBudget:
        return ContextBudget(
            max_chars=settings.AGENT_CONTEXT_MAX_CHARS,
            user_request=settings.AGENT_CONTEXT_USER_REQUEST_CHARS,
            dataset=settings.AGENT_CONTEXT_DATASET_CHARS,
            task=settings.AGENT_CONTEXT_TASK_CHARS,
            permissions=settings.AGENT_CONTEXT_PERMISSION_CHARS,
            tools=settings.AGENT_CONTEXT_TOOL_CHARS,
            history=settings.AGENT_CONTEXT_HISTORY_CHARS,
            history_messages=settings.AGENT_CONTEXT_HISTORY_MESSAGES,
        )

    def _cache_key(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> str:
        # 缓存键刻意排除会话历史：对同一请求 + 同一工具集 + 同一数据集，
        # 最优计划不变；把历史掺进 key 会导致缓存永远 miss。
        tool_fingerprint = json.dumps(
            [{"name": t.get("name"), "schema": t.get("input_schema", {})} for t in tools],
            ensure_ascii=False, sort_keys=True, default=str,
        )
        provider = getattr(self.llm, "name", "unknown")
        model = getattr(self.llm, "model", "")
        dataset_ids = ",".join(str(x) for x in sorted(context.dataset_ids()))
        raw = "\n".join((provider, model, str(self.max_steps), tool_fingerprint, user_request.strip(), dataset_ids))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_plan(self, user_request: str, context: AgentContext, tools: list[dict[str, Any]]) -> AgentPlan:
        text = user_request.lower()
        plan_intent = []
        for kw in ("报告", "汇报", "导出报告", "总结報告", "report"):
            if kw in text:
                plan_intent.append("report.generate")
                break
        for kw in ("工作流", "workflow", "流程", "pipeline", "编排"):
            if kw in text:
                plan_intent.append("workflow.build_and_run")
                break
        intent_hint = ""
        if plan_intent:
            intent_hint = f"本次请求的意图工具已被识别为：{'、'.join(plan_intent)}，计划中应当包含它们。"

        system = (
            "你是小洛实验室的数据 Agent Planner。你的职责只有一件事：根据用户请求制定真实工具执行计划。"
            "严禁读取数据、严禁假装已经分析、严禁直接给出数据结论。"
            "你只能使用下面提供的候选工具，不能创造工具名。"
            "需要 dataset_id 的工具，如果上下文只有一个关联数据集，必须填写该 ID；多个数据集时不要猜。"
            "普通聊天不需要工具时输出空 steps。"
            "跨步骤传值必须用结构化引用 {{stepN.字段}}（N 从 1 开始，等于步骤序号），例如 {{step5.target}}、{{step3.workflow_id}}；"
            "禁止把上一步的内容用自然语言复述成参数。"
            + intent_hint +
            "【数据分析任务的标准交付路径】用户要求分析/智能分析/全面分析时，计划必须覆盖："
            "① 读取结构 dataset.inspect（必要时 dataset.schema）；"
            "② 数据质量 dataset.quality；"
            "③ 统计画像 dataset.profile；"
            "④ 分布与关系 eda.describe / eda.correlation；"
            "⑤ 结论沉淀 report.generate（它会基于数据集自动产出并嵌入：分布直方图、类别柱状图、"
            "相关系数热力图、相关性散点图、正态 Q-Q 图、累积分布图，因此一般不需要再用 eda.visualize 重复画这些标准图；"
            "只有用户明确指定某一张图、或需要 report.generate 之外的特殊图表时才加 eda.visualize，"
            "且其 column/x/y/columns 必须来自 dataset.schema / dataset.profile 的真实字段名，不确定就不要画）；"
            "除非用户明确只要聊天，否则不要省略 report.generate。"
            "workflow 相关请求请用 workflow.build_and_run 一步完成（内部已包含创建与执行），不要拆成 workflow.create + workflow.run。"
            "建模请求固定顺序：dataset.inspect → dataset.profile → ml.detect_task(infer_target=true) → ml.prepare(target={{stepN.target}}) → ml.train(target={{stepN.target}}) → report.generate。"
            "计划的最后一步通常是 report.generate。"
            "每个步骤的 expected_output 用不超过 20 字的一句话概括，不要写长句。"
            "输出 JSON：{\"goal\": str, \"steps\": [{\"tool\": str, \"arguments\": object, \"expected_output\": str, \"permission\": str}]}。"
        )
        context_text = context.to_prompt_text(max_chars=settings.AGENT_CONTEXT_MAX_CHARS, budget=self._context_budget())
        tool_specs = json.dumps(
            [
                {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {}),
                    "permission": t.get("permission", ""),
                }
                for t in tools
            ], ensure_ascii=False, default=str,
        )
        user = f"候选工具：\n{tool_specs}\n\n上下文（仅元数据，不含原始数据）：\n{context_text}\n\n用户请求：{user_request}\n最多 {self.max_steps} 步。"
        from app.agent.llm.base import LLMMessage
        data = self.llm.structured_output(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            AgentPlan,
            max_tokens=settings.AGENT_LLM_MAX_OUTPUT_TOKENS,
        )
        try:
            return AgentPlan.model_validate(data)
        except ValidationError as exc:
            raise PlanInvalidError("LLM 计划未通过校验", details={"errors": exc.errors(include_url=False)[:5]}) from exc

    def _rule_plan(self, user_request: str, context: AgentContext) -> AgentPlan:
        text = user_request.lower()
        ds_ids = context.dataset_ids()
        ds_id = ds_ids[0] if len(ds_ids) == 1 else None
        steps: list[PlanStep] = []

        def _args(extra: dict[str, Any] | None = None) -> dict[str, Any]:
            args: dict[str, Any] = {}
            if ds_id is not None:
                args["dataset_id"] = ds_id
            if extra:
                args.update(extra)
            return args

        wants_train = any(k in text for k in ("训练", "模型", "预测", "分类", "回归", "train"))
        wants_quality = any(k in text for k in ("质量", "缺失", "重复", "异常", "quality"))
        wants_corr = any(k in text for k in ("相关", "corr"))
        wants_workflow = any(k in text for k in ("workflow", "工作流", "流程", "编排", "pipeline"))
        wants_merge = any(k in text for k in ("合并", "关联", "拼接", "连接", "宽表", "join", "merge"))
        wants_report = any(k in text for k in ("报告", "汇报", "导出", "report", "pdf"))
        wants_chart = any(k in text for k in ("图", "可视化", "chart", "直方图", "散点", "热力图", "分布图"))
        # 注意：这里用到的工具必须在 ContextBuilder.with_tools 的确定性注入集合里，
        # 否则规则规划会引用未被注入的候选工具而被 _validate 拒绝。
        wants_comprehensive = any(k in text for k in ("智能分析", "全面分析", "完整分析", "关键统计", "问题摘要", "综合"))
        wants_eda = wants_corr or any(k in text for k in ("eda", "分布", "探索", "描述", "统计"))
        if not ds_ids:
            return AgentPlan(goal=f"需要数据集才能执行：{user_request[:80]}", steps=[])

        if wants_merge and len(ds_ids) >= 2:
            # 多表关联：keys 不填，由 data.merge 按同名列自动推断并写进执行结果的 plan 里。
            # 规划阶段拿不到列名（ContextBuilder 明确禁止读 Schema），以前只能硬猜或整条失败。
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments={"dataset_id": ds_ids[0]}),
                PlanStep(tool="data.merge", arguments={
                    "left_dataset_id": ds_ids[0],
                    "right_dataset_id": ds_ids[1],
                    "join_type": "left" if any(k in text for k in ("左", "保留左", "left")) else "inner",
                }),
            ])
            return AgentPlan(goal=f"关联多张表：{user_request[:80]}", steps=steps[: self.max_steps])

        if wants_workflow:
            # 一条「读数据 → 质量 → 统计 → 汇总」的编排：build_and_run 内部完成创建与执行
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="workflow.build_and_run", arguments={
                    "name": user_request[:24] or "Agent 自动编排",
                    "nodes": [
                        {"id": "n1", "type": "dataset.read", "config": {"dataset_id": ds_id}},
                        {"id": "n2", "type": "data.quality_check", "config": {}},
                        {"id": "n3", "type": "data.statistics", "config": {}},
                        {"id": "n4", "type": "report.summary", "config": {}},
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2"},
                        {"source": "n1", "target": "n3"},
                        {"source": "n2", "target": "n4"},
                        {"source": "n3", "target": "n4"},
                    ],
                }),
                PlanStep(tool="report.generate", arguments=_args()),
            ])
            return AgentPlan(goal=f"编排并执行工作流：{user_request[:80]}", steps=steps[: self.max_steps])

        if wants_train:
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="dataset.schema", arguments=_args()),
                PlanStep(tool="dataset.profile", arguments=_args()),
                PlanStep(tool="ml.detect_task", arguments=_args({"infer_target": True})),
                PlanStep(tool="ml.prepare", arguments=_args({"target": "{{step4.target}}"})),
            ])
            # target 不再靠猜：ml.detect_task（第 4 步）会按命名约定推断目标列，
            # ml.train 通过结构化引用消费其输出；无目标列时 detect_task 判为聚类，
            # train 会给出明确错误而不是盲目执行。
            model = "linear_regression" if "回归" in text else "logistic_regression"
            train_args = _args({"model": model, "target": "{{step4.target}}"})
            steps.append(PlanStep(tool="ml.train", arguments=train_args))
            # 「训练并评估」是常见说法：训练步自带指标，但显式要求评估时应补 ml.evaluate，
            # 这样链路才闭环（此前规则规划只训练不评估，用户看到的结果少一截）。
            if any(k in text for k in ("评估", "效果", "指标", "准确率", "evaluate")):
                steps.append(
                    PlanStep(tool="ml.evaluate", arguments={"run_id": "{{step%d.run_id}}" % len(steps)})
                )
        elif wants_comprehensive:
            steps.extend([
                PlanStep(tool="dataset.inspect", arguments=_args()),
                PlanStep(tool="dataset.schema", arguments=_args()),
                PlanStep(tool="dataset.quality", arguments=_args()),
                PlanStep(tool="dataset.profile", arguments=_args()),
            ])
            if wants_corr:
                steps.append(PlanStep(tool="eda.correlation", arguments=_args()))
        elif wants_quality:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="dataset.quality", arguments=_args())])
        elif wants_eda:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="eda.describe", arguments=_args())])
            if wants_corr:
                steps.append(PlanStep(tool="eda.correlation", arguments=_args()))
        else:
            steps.extend([PlanStep(tool="dataset.inspect", arguments=_args()), PlanStep(tool="dataset.profile", arguments=_args())])

        # 交付闭环：任何数据分析分支都以 report.generate 收尾。
        # report.generate 内部会自动产出「分布直方图 + 相关系数热力图 + 相关性散点图 +
        # 类别柱状图 + Q-Q 图 + CDF」并写进报告中心，保证无 LLM 时也有图文并茂的报告。
        if not any(s.tool == "report.generate" for s in steps):
            steps.append(PlanStep(tool="report.generate", arguments=_args()))
        return AgentPlan(goal=f"完成用户请求：{user_request[:80]}", steps=steps[: self.max_steps])

    def _validate(self, plan: AgentPlan, tools: list[dict[str, Any]], context: AgentContext) -> AgentPlan:
        known = {t["name"]: t for t in tools}
        unknown = [s.tool for s in plan.steps if s.tool not in known]
        if unknown:
            raise PlanInvalidError("计划引用了未被工具检索选中的工具", details={"unknown_tools": unknown, "available": sorted(known)})

        dataset_ids = context.dataset_ids()
        if len(dataset_ids) == 1:
            dataset_id = dataset_ids[0]
            for step in plan.steps:
                schema = known[step.tool].get("input_schema") or {}
                properties = schema.get("properties") or {}
                required = schema.get("required") or []
                if "dataset_id" in properties and "dataset_id" in required and "dataset_id" not in step.arguments:
                    step.arguments["dataset_id"] = dataset_id
        elif len(dataset_ids) > 1:
            for step in plan.steps:
                schema = known[step.tool].get("input_schema") or {}
                required = schema.get("required") or []
                if "dataset_id" in required and "dataset_id" not in step.arguments:
                    raise PlanInvalidError("当前会话关联多个数据集，请明确指定要操作的数据集", details={"tool": step.tool, "dataset_ids": dataset_ids})

        # 步数超限不再整体判非法（LLM 常略超上限，直接失败会让整次运行空手而归）；
        # 截断到上限并保留前缀，保证「先产出结果」而不是「什么都不做」。
        if len(plan.steps) > self.max_steps:
            # report.generate 是交付物，超限时也不能被截掉
            tail_keep: list = []
            for step in reversed(plan.steps):
                if step.tool == "report.generate":
                    tail_keep.insert(0, step)
                    break
            plan.steps = plan.steps[: max(0, self.max_steps - len(tail_keep))] + tail_keep
        return plan
