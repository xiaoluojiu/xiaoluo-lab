"""钉住「这条回答到底是谁给的」——答案来源标记（`app/agent/answer_source.py`）。

为什么值得单独钉
----------------
用户的原话是「我不知道在开了 API 的情况下我的回答究竟是谁回答的」。这句话背后是
**三种在界面上长得一模一样、但含义完全不同**的来路：

| 场景 | 真实来源 | 应标记 |
| --- | --- | --- |
| 远程开着，调用成功 | 远程大模型 | `REMOTE_LLM_*` |
| 远程关着 | 平台内置规则 | `PLATFORM_RULES_*` |
| **远程开着，但这一次调用失败** | **仍是内置规则兜底** | `LLM_ERROR_FALLBACK` |

第三行是唯一「看起来像模型答的、其实是规则答的」的情况，也是最容易误判的一行。
本文件的核心用例就是钉死它：**凡是没走通 LLM 的分支，一律不许标成 `by_llm=True`**。

其余不变式：
- `describe()` 对未知/空值必须优雅退化（`NO_ANSWER`），不能让前端为未知值崩溃；
- 模型名只在「确实由模型产出」时附带 —— 降级时不带，避免界面显示误导性的模型标识；
- 来源标记必须随 `run.summary()` 下发，否则前端拿不到。

全部用例毫秒级、不启服务、不联网、**不调用任何真实大模型**。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent import answer_source as A
from app.agent.runtime.models import AgentRun, AgentSession, AgentTokenLedger
from app.agent.runtime.runtime import AgentRuntime


# ---------------------------------------------------------------------------
# 一、describe() 的契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,by_llm",
    [
        (A.REMOTE_LLM_CHAT, True),
        (A.REMOTE_LLM_SUMMARY, True),
        (A.PLATFORM_RULES_CHAT, False),
        (A.PLATFORM_RULES_NOTICE, False),
        (A.PLATFORM_RULES_SUMMARY, False),
        (A.LLM_ERROR_FALLBACK, False),
        (A.NO_ANSWER, False),
    ],
)
def test_describe_marks_by_llm_correctly(source, by_llm):
    d = A.describe(source)
    assert d["source"] == source
    assert d["by_llm"] is by_llm
    assert d["label"] and d["detail"], "每个来源都要有可展示的标签与说明"


@pytest.mark.parametrize("source", ["", None, "totally-unknown", 123])
def test_describe_degrades_for_unknown(source):
    """未知 / 空值统一落到 NO_ANSWER，避免前端为每个取值补兜底。"""
    d = A.describe(source)
    assert d["source"] == A.NO_ANSWER
    assert d["by_llm"] is False
    assert d["model"] == ""


def test_model_name_only_attached_when_llm_really_answered(monkeypatch):
    """降级场景绝不许挂模型名 —— 那会让用户以为「DeepSeek 答的这句」。"""
    monkeypatch.setattr(A, "llm_model_name", lambda: "deepseek-chat")
    assert A.describe(A.REMOTE_LLM_CHAT)["model"] == "deepseek-chat"
    assert A.describe(A.LLM_ERROR_FALLBACK)["model"] == ""
    assert A.describe(A.PLATFORM_RULES_CHAT)["model"] == ""


def test_describe_survives_settings_failure(monkeypatch):
    """模型名读不到只影响展示，不能让整条链路挂掉。"""
    def _boom():
        raise RuntimeError("配置不可用")

    monkeypatch.setattr(A, "llm_model_name", _boom)
    assert A.describe(A.REMOTE_LLM_CHAT)["model"] == ""
    assert A.describe(A.REMOTE_LLM_CHAT)["by_llm"] is True


# ---------------------------------------------------------------------------
# 四、对外下发：summary() 必须带上来源
# ---------------------------------------------------------------------------


def _run(text: str = "你好") -> AgentRun:
    return AgentRun(id="r-src", session_id="s-src", user_id="u-src", user_request=text)


def _session() -> AgentSession:
    return AgentSession(id="s-src", user_id="u-src")


def test_summary_exposes_answer_source():
    run = _run()
    run.answer_source = A.REMOTE_LLM_SUMMARY
    data = run.summary()
    assert data["answer_source"]["source"] == A.REMOTE_LLM_SUMMARY
    assert data["answer_source"]["by_llm"] is True
    assert "label" in data["answer_source"]


def test_summary_of_unmarked_run_is_no_answer():
    """还没产出回答的运行（失败 / 取消）如实标成「未产出回答」，不留空让前端猜。"""
    run = _run()
    assert run.summary()["answer_source"]["source"] == A.NO_ANSWER


def test_failed_run_is_marked_no_answer():
    run = _run()
    events: list[str] = []
    AgentRuntime._fail(
        SimpleNamespace(_emit=lambda r, t, p, e: events.append(t), _emit_usage=lambda *a, **k: None),
        run,
        "第 3 步失败",
        None,
    )
    assert run.answer_source == A.NO_ANSWER
    assert A.describe(run.answer_source)["by_llm"] is False


# ---------------------------------------------------------------------------
# 五、Token 账本快照（`usage` 事件）：运行过程中就要能看到花了多少 / 省了多少
# ---------------------------------------------------------------------------


def test_emit_usage_publishes_a_snapshot_without_touching_behaviour():
    run = _run()
    run.token_ledger = AgentTokenLedger(llm_calls=2, actual_input_tokens=100, actual_output_tokens=40, actual_total_tokens=140, estimated_result_saved_tokens=900)
    captured: list[tuple[str, dict]] = []
    AgentRuntime._emit_usage(
        SimpleNamespace(_emit=lambda r, t, p, e: captured.append((t, p))),
        run,
        None,
    )
    assert [t for t, _ in captured] == ["usage"]
    payload = captured[0][1]["token_usage"]
    assert payload["actual"]["total_tokens"] == 140
    assert payload["optimization"]["estimated_saved_tokens"] == 900
    assert payload["budget"]["max_total_tokens"] >= 0


def test_emit_usage_never_breaks_the_run():
    """账本上报属于可观测性增强：即使抛异常也不能把主流程带崩。"""
    run = _run()

    def _boom(*_a, **_k):
        raise RuntimeError("事件流断了")

    # 不应抛出
    AgentRuntime._emit_usage(SimpleNamespace(_emit=_boom), run, None)
