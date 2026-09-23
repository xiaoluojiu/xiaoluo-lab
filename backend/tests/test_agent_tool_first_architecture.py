from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_context_builder_never_loads_dataframe():
    source = (ROOT / "app/agent/context/builder.py").read_text(encoding="utf-8")
    forbidden = ("load_version(", "engine.schema(", "engine.profile(", "engine.quality(", ".head(")
    assert not any(token in source for token in forbidden)


def test_runtime_short_circuits_chat_before_context_build():
    source = (ROOT / "app/agent/runtime/runtime.py").read_text(encoding="utf-8")
    chat_guard = 'if plan_override is None and mode == "chat":'
    context_build = "context=self._build_context(run,session)"
    route_event = 'self._emit(run,"route",{"mode":mode,"reason":route_reason,"intent":explain_intent(intent)},on_event)'
    assert chat_guard in source
    assert route_event in source
    # 路由事件必须先于对话短路与上下文构建发出（透明性：路由依据对用户可见）
    assert source.index(route_event) < source.index(chat_guard) < source.index(context_build)


def _compact(source: str) -> str:
    """去掉所有空白再比对，避免源码换行/空格调整就让架构约束测试失效。"""
    return "".join(source.split())


def test_runtime_uses_retrieved_candidates_for_planner():
    source = _compact((ROOT / "app/agent/runtime/runtime.py").read_text(encoding="utf-8"))
    candidates = "candidate_tools=self._candidate_tools(context,all_tools)"
    assert candidates in source
    # 计划必须优先只用「被检索命中的候选工具」；
    # build_plan_resilient 是在候选集不足以成计划时的兜底（放宽到全量工具、再降级规则计划），
    # 绝不能反过来一开始就把全量工具塞给规划器（那会让工具召回形同虚设）。
    assert "self.planner.build_plan_resilient(run.user_request,context,candidate_tools" in source
    # all_tools 只能作为放宽重试的兜底参数传入
    assert "all_tools=all_tools" in source
    assert source.index(candidates) < source.index("self.planner.build_plan_resilient(")


def test_planner_forbids_unretrieved_tools():
    source = (ROOT / "app/agent/planner/planner.py").read_text(encoding="utf-8")
    assert "计划引用了未被工具检索选中的工具" in source
