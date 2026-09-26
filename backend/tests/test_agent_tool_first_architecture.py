"""架构不变式守护（源码级 + 运行时级断言）。

重构后（统一 Agent Loop）钉住五条不变式：
① ContextBuilder 不加载 DataFrame（元数据-only 禁令）；
② 唯一控制流循环在 loop 模块，runtime 不再按 mode 切两条流程；
③ 无 Chat/Agent 硬分流；
④ 远程决策载荷只含紧凑状态与当前阶段候选工具（不发送完整历史 / ToolResult）；
⑤ 无 shell / eval 类代码执行工具。
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_context_builder_never_loads_dataframe():
    source = (ROOT / "app/agent/context/builder.py").read_text(encoding="utf-8")
    forbidden = ("load_version(", "engine.schema(", "engine.profile(", "engine.quality(", ".head(")
    assert not any(token in source for token in forbidden)


def test_single_loop_in_loop_module():
    """唯一迭代循环位于 loop 模块；runtime 不保留任何旧决策/规划/执行方法。"""
    loop_source = (ROOT / "app/agent/loop.py").read_text(encoding="utf-8")
    runtime_source = (ROOT / "app/agent/runtime/runtime.py").read_text(encoding="utf-8")
    assert "class AgentLoop" in loop_source
    assert "while True" in loop_source  # 唯一迭代体
    # runtime 瘦身：不含旧方法定义
    for banned in ("_run_plan", "_run_dynamic", "_build_plan", "_local_direct_plan", "_task_spec_plan", "_understand", "_escalate", "_direct_chat", "_compose_answer"):
        assert f"def {banned}" not in runtime_source, f"runtime.py 不得残留 {banned}"


def test_no_chat_agent_hard_split():
    """无 Chat/Agent 硬分流：不存在按前缀切流的 _route / _needs_data_tools。"""
    runtime_source = (ROOT / "app/agent/runtime/runtime.py").read_text(encoding="utf-8")
    assert "def _route" not in runtime_source
    assert "def _needs_data_tools" not in runtime_source


def test_remote_decision_payload_is_compact():
    """远程决策输入只含紧凑状态 + 阶段候选工具，不含完整历史与完整 ToolResult。"""
    import re

    loop_source = (ROOT / "app/agent/loop.py").read_text(encoding="utf-8")
    # 远程载荷走 compact_for_remote（紧凑状态视图）
    assert "compact_for_remote()" in loop_source
    # 候选工具只注入当前阶段（_tool_brief），不把全部 schema 塞给远程
    assert "_tool_brief" in loop_source
    # 远程决策方法体不得直接引用完整历史 / 完整 ToolResult
    m = re.search(r"def _remote_decide\(.*?(?=\n    def )", loop_source, re.S)
    assert m, "_remote_decide 方法不存在"
    body = m.group(0)
    assert "ctx.session.history" not in body, "远程决策载荷不得包含完整会话历史"
    assert "run.tool_calls" not in body, "远程决策载荷不得包含完整 ToolResult"


def test_no_shell_or_eval_tool():
    """工具注册表不存在 shell / eval 类代码执行通道。"""
    from app.tools.builtin import register_builtin_tools
    from app.tools.registry import TOOL_REGISTRY

    register_builtin_tools()
    names = set(TOOL_REGISTRY.names())
    for banned in ("system.shell", "python.eval", "os.system", "subprocess.run"):
        assert banned not in names, f"不得注册 {banned}"
