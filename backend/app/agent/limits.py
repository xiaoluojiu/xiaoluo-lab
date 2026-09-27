"""Agent 运行期的硬上限。

为什么单独放一个文件
--------------------
这些常量**同时被三处消费**：

- ``engine`` 用它做闸门（超了就停）
- ``store.create_run`` 用它初始化 ``TokenUsage`` 的预算
- ``api/v1/agent`` 的 ``/capabilities`` 把它展示给用户

引擎 import store，store 不能反向 import 引擎（会成环）。
放在这里三方都能取，且不会出现「闸门是 4、面板写 12」这种自相矛盾的展示。

改这里的数必须同步看 ``WALL_CLOCK_SECONDS`` 与前端 ``useAgentRun`` 的超时：
前端先超时、后端还在跑，是用户最能感知的一类「卡住」。
"""

from __future__ import annotations

#: 单次运行最多执行多少步（防御性上限；实际 playbook 通常 1~3 步）
MAX_STEPS = 8
#: 单次运行最多调用多少次 LLM（参数抽取 + 答案润色）
MAX_LLM_CALLS = 4
#: 墙钟上限（秒）。超过即终止，避免长任务把 SSE 与线程占死。
#: 必须留足合法长任务的余量：ml.train 同步执行，训练时长随数据规模/模型/超参
#: 波动（实测 random_forest 在 20 万行上曾到 342s）。若上限太短，训练结果
#: 已落库、run 却被判「运行超时」失败，用户看到超时而非指标。
WALL_CLOCK_SECONDS = 600.0


def max_total_tokens() -> int:
    """单次运行的 Token 总预算。读 settings，允许在运行期被设置页改。"""
    from app.core.config import settings

    return int(getattr(settings, "AGENT_LLM_MAX_TOTAL_TOKENS", 0) or 0)
