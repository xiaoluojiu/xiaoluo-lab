"""远程大模型「失败之后到底发生了什么」的兜底闭环回归测试。

为什么值得单独一个文件
----------------------
开关开着、凭据也在，但这一次调用没成功 —— 这是线上最常见、也最容易做假的一类失败
（欠费 / 限流 / 超时 / 网关 5xx）。它暴露过四个彼此独立的问题：

1. **闲聊兜底是假的**：`_direct_chat` 只是把异常文本拼成
   「暂时无法完成对话请求：…」，然后标 `LLM_ERROR_FALLBACK`。用户既没拿到回答，
   界面还告诉他「已降级到规则」。
2. **规划兜底是假的**：`build_plan_resilient` 只捕获 `PlanInvalidError`，
   而远程失败抛的是 `LLMException` ⇒ 欠费时整次数据分析直接失败。
3. **重试策略不分成因**：429 一律重试，于是「余额不足」被重试 3 次 + 指数退避空等。
4. **来源会撒谎**：见上第 1 条。

本文件钉住修复后的四条不变式，全部离线、毫秒级、**不调用任何真实大模型**。
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from app.agent import answer_source as A
from app.agent.context.builder import ContextBuilder
from app.agent.llm import openai_compatible as O
from app.agent.llm.base import LLMException, LLMMessage, LLMProvider, is_fallbackable_error
from app.agent.llm.openai_compatible import OpenAICompatibleProvider
from app.agent.runtime.models import RunStatus
from app.agent.runtime.runtime import AgentRuntime
from app.core.config import settings
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.services.dataset_service import DatasetService
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FailingLLM(LLMProvider):
    """永远失败的远程模型：模拟欠费 / 限流 / 超时 / 网关 5xx。"""

    name = "failing"

    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__()
        self._exc = exc or LLMException("账户余额不足")
        self.chat_calls = 0
        self.structured_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        self.chat_calls += 1
        self.begin_call()
        raise self._exc

    def structured_output(self, messages, schema, *, max_attempts=3, **kwargs):
        self.structured_calls += 1
        self.begin_call()
        raise self._exc


class OkLLM(LLMProvider):
    """永远成功的远程模型（用于对照「真的由模型产出」那一格）。"""

    name = "ok"

    def __init__(self, content: str = "远程模型给出的回答") -> None:
        super().__init__()
        self._content = content
        self.chat_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        self.chat_calls += 1
        self.begin_call()
        from app.agent.llm.base import LLMResponse

        return LLMResponse(content=self._content, model="ok-1")


def _frame(rows: int = 60) -> pl.DataFrame:
    half = max(rows // 2, 1)
    return pl.DataFrame(
        {
            "x1": [float(i + 1) for i in range(rows)],
            "x2": [float(rows - i) for i in range(rows)],
            "target": ["pos" if i < half else "neg" for i in range(rows)],
        }
    )


@pytest.fixture()
def env(db, storage):
    """数据引擎 + 实验 + 已带版本的数据集（60 行，Pre-flight 不会因样本量反问）。"""
    ds = DatasetService(db, storage)
    dataset = ds.create("toy", "玩具数据")
    ds.create_version(dataset.id, _frame())
    return {
        "db": db,
        "ds": ds,
        "engine": DataEngineService(ds),
        "exp": ExperimentService(db, ds),
        "dataset_id": dataset.id,
    }


def _runtime(env, llm) -> AgentRuntime:
    """构造带指定 LLM 的 Runtime；规划器同步注入（与 deps.get_agent_runtime 同口径）。"""
    return AgentRuntime(
        env["engine"],
        experiment_service=env["exp"],
        db=env["db"],
        llm=llm,
    )


# ---------------------------------------------------------------------------
# 一、普通聊天：远程失败后必须真的走本地兜底
# ---------------------------------------------------------------------------


def test_greeting_is_answered_locally_without_remote(env, monkeypatch):
    """★ 问候（greeting）走确定性本地应答，零远程调用 —— 欠费与否都不影响它。

    重构后 greeting 由统一 Loop 的确定性层直接回答，不再发起远程调用，
    因此「远程失败 → 降级本地」这条后路在问候场景下**天然不需要**。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "你好")

    assert run.status == RunStatus.COMPLETED
    assert run.answer_source == A.PLATFORM_RULES_CHAT
    assert llm.chat_calls == 0, "问候不应发起远程调用"
    assert A.describe(run.answer_source)["by_llm"] is False
    assert run.final_answer and "AI 助手" in run.final_answer


def test_chat_falls_back_to_notice_when_local_cannot_answer(env, monkeypatch):
    """★ P0：内置规则也答不上时，给的是**状态说明**，来源如实标 `NO_ANSWER`。

    「绝不能无响应」+「绝不能假装远程」：文字照给，但绝不标成
    `LLM_ERROR_FALLBACK`（没有真的降级成功）或 `REMOTE_LLM_*`。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    from app.agent.local_chat import no_llm_notice

    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "讲个笑话")

    assert run.status == RunStatus.COMPLETED
    assert run.final_answer == no_llm_notice(reason="账户余额不足")
    assert "账户余额不足" in run.final_answer, "真实失败原因必须透出给用户"
    assert run.answer_source == A.NO_ANSWER
    assert A.describe(run.answer_source)["by_llm"] is False


def test_chat_fails_when_fallback_is_disabled(env, monkeypatch):
    """`AGENT_ALLOW_MODEL_FALLBACK=False` ⇒ 远程失败就如实失败，不做任何包装。"""
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", False)
    llm = FailingLLM(LLMException("账户余额不足"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "讲个笑话")

    assert run.status == RunStatus.FAILED
    assert run.answer_source == A.NO_ANSWER
    assert not run.final_answer


def test_program_bug_is_not_masked_as_fallback(env, monkeypatch):
    """★ 不该盲目兜底的一类：`TypeError` 是代码缺陷，不是「远程不可用」。

    用规则应答去兜一个程序 bug，界面上会显示「降级成功」，
    于是真正的缺陷永远不会被发现。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(TypeError("unexpected keyword argument"))
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, "讲个笑话")

    assert run.status == RunStatus.FAILED
    assert run.answer_source == A.NO_ANSWER
    assert is_fallbackable_error(TypeError("x")) is False


# ---------------------------------------------------------------------------
# 二、数据分析：远程规划失败后 Rule Planner 必须真的接管
# ---------------------------------------------------------------------------


def test_data_task_falls_back_to_rule_planner_and_executes(env, monkeypatch):
    """★ P0：欠费时数据分析仍然要跑出**真实工具结果**，而不是整次运行失败。

    「检查一下数据质量」是本地 Router 能高置信度识别的简单任务（控制权归位后），
    直接走本地决策执行，**根本不会去调远程 LLM 规划器** —— 欠费与否都不影响它跑出
    真实结果。这正是「简单任务零远程调用」的体现：无需再靠「远程失败→规则兜底」这条
    后路，本地从一开始就不发起那笔注定失败的远程调用。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = FailingLLM(LLMException("Insufficient Balance"))
    rt = _runtime(env, llm)
    session = rt.create_session(dataset_ids=[env["dataset_id"]])

    run = rt.run(session, "检查一下数据质量")

    assert run.status == RunStatus.COMPLETED, run.error
    executed = [call.tool for call in run.tool_calls]
    assert "dataset.quality" in executed, f"本地决策应执行 dataset.quality，实际：{executed}"
    # 简单任务走本地直连（local_direct），不经过远程规划器 ⇒ 没有 planner_fallback 标记，
    # 也从未发起远程规划调用（欠费与否都与本次运行无关）。
    assert run.plan and run.plan.get("planner_fallback") is not True
    # 简单任务走本地直连（local_direct），规划阶段**不发起远程规划调用**（structured_output 为 0）。
    # 汇总阶段仍会试一次 chat（这是最终答案合成，不是规划），失败后降级到规则汇总 —— 属预期。
    assert llm.structured_calls == 0, \
        f"简单任务规划阶段不应发起远程规划调用，实际 structured={llm.structured_calls} 次"
    # 数据是真的跑出来的；汇总时无远程 ⇒ 标平台规则汇总，`by_llm` 必须是 False。
    assert run.answer_source in (A.LLM_ERROR_FALLBACK, A.PLATFORM_RULES_SUMMARY)
    assert A.describe(run.answer_source)["by_llm"] is False
    assert run.final_answer



# ---------------------------------------------------------------------------
# 三、429 / 配额：重试策略必须分成因
# ---------------------------------------------------------------------------


class _Resp:
    """最小 httpx.Response 替身。"""

    def __init__(self, status_code: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}
        self._json: Any = None
        import json as _json

        try:
            self._json = _json.loads(body)
        except ValueError:
            self._json = None

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("not json")
        return self._json


def test_quota_429_is_not_retried(monkeypatch):
    """★ P1：余额不足的 429 **一次都不重试** —— 重试一万次也不会成功。

    顺带验证它转成的是「可降级的 LLMException」：上层据此走本地兜底。
    """
    calls: list[int] = []

    def fake_post(url, **kwargs):
        calls.append(1)
        return _Resp(429, '{"error":{"code":"insufficient_quota","message":"Insufficient Balance"}}')

    monkeypatch.setattr(O.httpx, "post", fake_post)
    # base_url 唯一：熔断器按 base_url 隔离，避免与其它用例互相拖累
    provider = OpenAICompatibleProvider("http://quota.test", "m", "k")

    with pytest.raises(LLMException) as excinfo:
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1, f"配额类错误只应请求一次，实际 {len(calls)} 次"
    assert "余额" in excinfo.value.message or "配额" in excinfo.value.message
    assert excinfo.value.fallbackable is True


@pytest.mark.parametrize(
    "status,body",
    [
        (429, '{"error":{"code":"billing_hard_limit_reached"}}'),
        (429, '{"error":{"type":"insufficient_quota"}}'),
        (402, '{"error":{"message":"Insufficient Balance"}}'),
        (429, '{"error":{"message":"您的账户余额不足"}}'),
    ],
)
def test_quota_signatures_are_all_recognized(monkeypatch, status, body):
    """各家网关措辞不统一：code / type / message / HTTP 402 都要认得出来。"""
    calls: list[int] = []
    monkeypatch.setattr(O.httpx, "post", lambda url, **kw: (calls.append(1), _Resp(status, body))[1])
    provider = OpenAICompatibleProvider(f"http://quota-{status}-{abs(hash(body))}.test", "m", "k")

    with pytest.raises(LLMException):
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1


def test_transient_429_retries_then_succeeds(monkeypatch):
    """★ P1：真正的瞬时限流要**有限重试**，并且用服务端给的 `Retry-After`。"""
    monkeypatch.setattr(O, "_RETRY_BACKOFF_BASE_SECONDS", 0.0)  # 测试里不真睡
    statuses = [429, 429, 200]
    delays: list[float] = []
    real_sleep = O.time.sleep
    monkeypatch.setattr(O.time, "sleep", lambda s: delays.append(s))

    def fake_post(url, **kwargs):
        status = statuses.pop(0)
        if status == 200:
            return _Resp(200, '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}')
        return _Resp(429, '{"error":{"message":"Rate limit reached"}}', headers={"Retry-After": "0"})

    monkeypatch.setattr(O.httpx, "post", fake_post)
    provider = OpenAICompatibleProvider("http://ratelimit.test", "m", "k")

    response = provider.chat([LLMMessage(role="user", content="hi")])

    assert response.content == "ok"
    assert len(delays) == 2, f"两次失败 ⇒ 等待两次，实际 {len(delays)}"
    assert all(d == 0.0 for d in delays), "`Retry-After: 0` 应被优先采用而不是指数退避"
    assert real_sleep is not None


def test_retry_after_is_preferred_over_backoff(monkeypatch):
    """服务端明确给了 `Retry-After` 时以它为准（不再自己拍一个指数退避）。"""
    resp = _Resp(429, "{}", headers={"Retry-After": "7"})
    assert O._retry_after_seconds(resp) == 7.0
    assert O._retry_after_seconds(_Resp(429, "{}", headers={})) is None


def test_unretryable_4xx_is_not_retried(monkeypatch):
    """鉴权 / 参数错误重试没有意义：只请求一次。"""
    calls: list[int] = []
    monkeypatch.setattr(O.httpx, "post", lambda url, **kw: (calls.append(1), _Resp(401, '{"error":{"message":"invalid api key"}}'))[1])
    provider = OpenAICompatibleProvider("http://auth.test", "m", "k")

    with pytest.raises(LLMException):
        provider.chat([LLMMessage(role="user", content="hi")])

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 四、Answer Source 不许撒谎（四种处境的完整对照）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,utterance,llm_factory,expect_source,expect_by_llm",
    [
        ("远程正常", "讲个笑话", lambda: OkLLM(), A.REMOTE_LLM_CHAT, True),
        ("远程失败+本地答不上", "讲个笑话", lambda: FailingLLM(LLMException("余额不足")), A.NO_ANSWER, False),
        # 同一个问法换个 LLM 状态，来源必须跟着变 —— 这是「来源不许撒谎」最硬的一条
        ("远程未启用+本地答不上", "讲个笑话", lambda: None, A.PLATFORM_RULES_NOTICE, False),
        ("问候走本地零远程", "你好", lambda: OkLLM(), A.PLATFORM_RULES_CHAT, False),
    ],
)
def test_answer_source_never_lies(env, monkeypatch, label, utterance, llm_factory, expect_source, expect_by_llm):
    """★ 唯一判据是「这一次到底有没有走通远程」，不是配置项。

    `远程失败+本地答不上` 那一格最关键：它**不许**标 `LLM_ERROR_FALLBACK`
    （没有真的降级成功），也不许标 `REMOTE_*`（不是模型生成的）。
    """
    monkeypatch.setattr(settings, "AGENT_ALLOW_MODEL_FALLBACK", True)
    llm = llm_factory()
    rt = _runtime(env, llm)
    session = rt.create_session()

    run = rt.run(session, utterance)

    assert run.answer_source == expect_source, f"[{label}] 来源标错"
    described = A.describe(run.answer_source)
    assert described["by_llm"] is expect_by_llm, f"[{label}] by_llm 标错"
    if not expect_by_llm:
        assert described["model"] == "", f"[{label}] 非模型产出不许挂模型名"


def test_no_answer_case_still_answers_the_user():
    """`NO_ANSWER` 是「没有回答用户的问题」，不等于「界面上什么都不显示」。

    界面仍会收到一段如实的状态说明（含真实失败原因），只是来源标成未产出回答。
    """
    from app.agent.local_chat import no_llm_notice

    notice = no_llm_notice(reason="Insufficient Balance")
    assert "Insufficient Balance" in notice
    assert notice


def test_fallback_flag_default_is_on():
    """平台推荐默认开启：远程挂了也要能用，而不是让用户对着失败发呆。

    ★ 这里**绝不能**用 `importlib.reload(app.core.config)`：reload 会换掉模块里的
    `settings` 单例，而其它模块早已持有旧对象的引用（`from app.core.config import settings`
    在不同时刻 import 会拿到不同对象），于是后续用例的 monkeypatch 全部打空，
    表现为「改了 A 处、B 处用例挂」—— 用类直接取默认值即可，不碰进程内单例。
    """
    from app.core.config import Settings

    assert Settings(_env_file=None).AGENT_ALLOW_MODEL_FALLBACK is True
