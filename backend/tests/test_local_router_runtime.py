"""钉住本地 Router 的**线上推理实现**（`app/local_router/{scoring,model,router,trace}.py`）。

为什么必须单测这几个文件
------------------------
线上路由漂移的症状是「偶发把数据请求当闲聊 / 把闲聊当数据任务」这类**静默错误**：
不报错、不抛异常，用户只觉得「它没听懂」。所以把线上实现与离线口径**绑死**，
让任何一边的改动立刻在测试里失败：

| 不变式 | 为什么值得钉 |
| --- | --- |
| L0 规则优先于模型 | 顺序错了会把「平台做不了的事」喂给只会选工具的小模型 |
| **线上合成 == 离线合成（逐条）** | 否则论文里 80.7% 的数字与线上行为无关 |
| 模型缺失 ⇒ 保守升级，绝不猜 | 静默给出错误路由比多花一次远程调用昂贵得多 |
| 工具清单变化 ⇒ 判过期 | 平台加工具后旧模型**不报错**，只是新功能永远用不了 |
| 概率归一 + 类别对齐 | `LinearSVC` 只见过训练折出现过的类，列错位也不会报错 |
| shadow 只记录、不改行为、失败不抛 | 埋点绝不能成为新的故障源 |

全部用例都是毫秒级离线计算：不训练、不启服务、不联网。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROUTER_DIR = _BACKEND / "scripts" / "router"
if str(_ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROUTER_DIR))

from app.core.config import Settings, settings  # noqa: E402
from app.local_router import contract as C  # noqa: E402
from app.local_router import trace as T  # noqa: E402
from app.local_router.contract import (  # noqa: E402
    EscalationReason,
    RouterDecision,
    RouterRequest,
    validate_decision,
)
from app.local_router.model import (  # noqa: E402
    CALL_PREFIX,
    CHAT_LABEL,
    LexicalRouterModel,
    l1_label_space,
    request_text,
    tool_of_label,
)
from app.local_router.router import (  # noqa: E402
    ConstantModel,
    decision_to_route,
    route_request,
)
from app.local_router.scoring import proba_from_scores, top1  # noqa: E402

# L0 规则能稳定命中「平台做不了的事」的句子（来自既有用例，已验证）。
ESCALATION_UTTERANCE = "把这个数据集同步到我们的 S3 存储桶"


def _req(utterance: str = "看看这批数据的分布", bound: int | None = None,
         columns: list[str] | None = None) -> RouterRequest:
    return RouterRequest(
        utterance=utterance,
        bound_dataset_id=bound,
        available_columns=columns if columns is not None else ["a", "b"],
    )


def _tool_needing_dataset_id() -> str:
    for name in C.tool_label_space():
        if C.required_params(name) == ["dataset_id"]:
            return name
    pytest.skip("没有仅以 dataset_id 为必填的工具，本用例不适用")


# ---------------------------------------------------------------------------
# 一、三层顺序
# ---------------------------------------------------------------------------


def test_escalation_wins_over_model():
    """L0 规则优先：哪怕模型信心十足地说了一个工具，规则判升级就必须升级。"""
    decision = route_request(_req(ESCALATION_UTTERANCE), model=ConstantModel("call::dataset.quality"))
    assert decision.escalate is True
    assert decision.escalate_reason is not None
    assert decision.intent is None, "升级路径尚未判定能力域，填猜测值会被下游读成结论"
    assert decision_to_route(decision).startswith("escalate::")


def test_escalation_reason_comes_from_rules():
    decision = route_request(_req(ESCALATION_UTTERANCE), model=None)
    assert decision.escalate_reason.value in {r.value for r in EscalationReason}


def test_missing_model_escalates_conservatively():
    """模型拿不到时**绝不猜**：宁多花一次远程调用，也不要静默做错。"""
    decision = route_request(_req(), model=None)
    assert decision.escalate is True
    assert decision.tool is None
    assert decision_to_route(decision) == "escalate::ambiguous"


def test_unbound_dataset_turns_call_into_ask():
    """必填槽位缺失是**反问**，不是升级、更不是执行。"""
    tool = _tool_needing_dataset_id()
    decision = route_request(_req(bound=None), model=ConstantModel(CALL_PREFIX + tool))
    assert decision.tool == tool
    assert decision.missing == ["dataset_id"]
    assert decision.escalate is False
    assert decision_to_route(decision) == "ask::" + tool


def test_bound_dataset_executes_directly():
    tool = _tool_needing_dataset_id()
    decision = route_request(_req(bound=7), model=ConstantModel(CALL_PREFIX + tool))
    assert decision.missing == []
    assert decision_to_route(decision) == "call::" + tool


def test_chat_and_ask_prefix_tolerance():
    assert decision_to_route(route_request(_req("你好"), model=ConstantModel(CHAT_LABEL))) == "chat"
    tool = _tool_needing_dataset_id()
    # 上游给到 `ask::X` 时不得被二次套用规则（否则会重复计数、甚至变成升级）
    bare = _req(bound=7)
    assert decision_to_route(
        route_request(bare, model=ConstantModel("ask::" + tool))
    ) == "call::" + tool


def test_request_accepts_plain_dict():
    """离线样本里 request 是 dict，线上是 RouterRequest —— 两条路必须等价。"""
    payload = {"utterance": "你好", "bound_dataset_id": None, "available_columns": []}
    from_dict = decision_to_route(route_request(payload, model=ConstantModel(CHAT_LABEL)))
    from_obj = decision_to_route(
        route_request(RouterRequest(**payload), model=ConstantModel(CHAT_LABEL))
    )
    assert from_dict == from_obj == "chat"


# ---------------------------------------------------------------------------
# 二、线上 == 离线（本文件最重要的一条）
# ---------------------------------------------------------------------------


def _dataset_rows() -> list[dict]:
    import eval_harness as H

    try:
        return H.load_split("all")
    except Exception:  # noqa: BLE001 — 数据集缺失/损坏不应让单测失败
        return []


@pytest.mark.parametrize("bound", [None, 7])
@pytest.mark.parametrize("kind", ["chat", "empty", "call", "ask"])
def test_online_synthesis_matches_offline_on_handwritten_cases(kind, bound):
    """合成口径逐条比对：`route_request`（线上） vs `assemble_route`（离线）。

    四种输入覆盖三层合成的全部分支：闲聊 / 无表态 / 可执行或反问。
    """
    from l1_data import assemble_route

    tool = _tool_needing_dataset_id()
    label = {"chat": CHAT_LABEL, "empty": None,
             "call": CALL_PREFIX + tool, "ask": "ask::" + tool}[kind]

    req = _req("看看这批数据的分布", bound=bound)
    sample = {"id": "probe", "request": req.model_dump()}
    assert decision_to_route(route_request(req, model=ConstantModel(label))) == \
        assemble_route(sample, label)


def test_online_rejects_unknown_tool_while_offline_passes_it_through():
    """**有意保留的差异**（不是 bug）：产物标签空间闭合，线上不可能拿到不存在的工具；
    若真的拿到（产物被篡改/标签空间过期），线上选择升级而不是照抄一个不存在的工具。
    离线口径没有这层防御，因为它的职责只是复现评测算法。
    """
    from l1_data import assemble_route

    bogus = "call::不存在的工具"
    req = _req(bound=7)
    sample = {"id": "probe", "request": req.model_dump()}
    assert assemble_route(sample, bogus) == bogus
    decision = route_request(req, model=ConstantModel(bogus))
    assert decision.escalate is True
    assert decision_to_route(decision) == "escalate::ambiguous"


def test_online_synthesis_matches_offline_on_full_dataset():
    """全量数据集上逐条比对 —— 覆盖率比手写用例高几个数量级。

    金标标签直接当模型预测注入：这一条与「模型准不准」无关，
    只验证**两条合成路径实现相同**（这才是评测数字能否代表线上的前提）。
    """
    import eval_harness as H
    from l1_data import assemble_route, l1_label

    rows = _dataset_rows()
    if not rows:
        pytest.skip("路由数据集不存在（先跑 scripts/router/build_dataset.py）")

    mismatches = []
    for sample in rows:
        label = l1_label(sample)
        offline = assemble_route(sample, label)
        online = decision_to_route(
            route_request(sample.get("request") or {}, model=ConstantModel(label))
        )
        if offline != online:
            mismatches.append((sample["id"], label, offline, online))
    assert not mismatches, f"线上/离线合成不一致 {len(mismatches)} 条，前 5 条：{mismatches[:5]}"


def test_offline_and_online_input_text_are_identical():
    """输入文本口径必须逐字符相同，否则线上分布 ≠ 被验证过的分布（指标看不出来）。"""
    import eval_harness as H

    rows = _dataset_rows()
    if not rows:
        pytest.skip("路由数据集不存在")
    for sample in rows:
        req = sample.get("request") or {}
        assert H.request_text(req) == request_text(req), f"输入口径不一致：{sample['id']}"


# ---------------------------------------------------------------------------
# 三、模型加载 / 过期
# ---------------------------------------------------------------------------


def test_missing_artifact_returns_none(tmp_path):
    from app.local_router.model import get_model

    assert get_model(refresh=True, path=tmp_path / "nope.pkl") is None


def test_staleness_detects_registry_change():
    """平台加/删工具后旧模型**不报错**，只是新工具永远选不出来 —— 必须主动探测。"""
    live = list(C.tool_label_space())
    assert live, "工具注册表为空，测试前提不成立"

    no_meta = LexicalRouterModel(vectorizer=None, classifier=None, meta={})
    assert no_meta.staleness() is not None

    shrunk = LexicalRouterModel(
        vectorizer=None, classifier=None, label_space=l1_label_space(),
        meta={"tool_label_space": live[:-1]},
    )
    reason = shrunk.staleness()
    assert reason is not None and "变化" in reason

    fresh = LexicalRouterModel(
        vectorizer=None, classifier=None, label_space=l1_label_space(),
        meta={"tool_label_space": live},
    )
    assert fresh.staleness() is None


def test_tool_of_label():
    assert tool_of_label("chat") is None
    assert tool_of_label(None) is None
    assert tool_of_label("call::eda.distribution") == "eda.distribution"
    assert tool_of_label("escalate::ambiguous") is None
    # 容错：历史产物里可能直接存工具名
    assert tool_of_label("eda.distribution") == "eda.distribution"


def test_label_space_never_contains_unknown_tool():
    registered = set(C.tool_label_space())
    for label in l1_label_space():
        if label == CHAT_LABEL:
            continue
        assert label.startswith(CALL_PREFIX)
        assert label[len(CALL_PREFIX):] in registered


# ---------------------------------------------------------------------------
# 四、分数 → 概率
# ---------------------------------------------------------------------------


def test_proba_from_scores_aligns_and_normalizes():
    space = ["chat", "call::a", "call::b"]
    scores = np.array([[0.2, 3.0, 1.0]])
    proba = proba_from_scores(["call::b", "chat"], scores[:, [2, 0]], space)
    assert proba.shape == (1, 3)
    assert abs(float(proba.sum()) - 1.0) < 1e-9
    label, confidence = top1(proba, space)
    assert label == "call::b" and 0.0 < confidence <= 1.0


def test_proba_handles_missing_class_without_index_shift():
    """缺席的类必须落在**它自己的位置**上，而不是把后面的列整体前移。

    判据：`LinearSVC` 没见过 `call::a` ⇒ 它必须拿到全场最低概率；
    若对齐写错（把两列直接按顺序铺开），`call::a` 会顶着 `chat` 的分数。
    """
    space = ["chat", "call::a", "call::b"]
    scores = np.array([[5.0, -5.0]])
    proba = proba_from_scores(["call::b", "chat"], scores, space)
    absent = proba[0, 1]
    assert absent < proba[0, 0] and absent < proba[0, 2]
    assert int(np.argmax(proba, axis=1)[0]) == 2, "top1 应是分数最高的 call::b"


def test_binary_svc_decision_function_is_reshaped():
    """真实 LinearSVC 只有两个类时 `decision_function` 退化为一维 —— 这条路径必须能跑。

    用真 sklearn 拟合（而不是手造分数矩阵），因为「哪天换分类器/换类别数」正是这里
    最可能被踩到的地方；`classes_` 的顺序也由 sklearn 决定，不由我们假设。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    texts = ["做分布图", "画个直方图", "看看相关性", "你好呀", "谢谢", "闲聊一下"]
    labels = ["call::eda.distribution", "call::eda.distribution", "call::eda.distribution",
              "chat", "chat", "chat"]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
    clf = LinearSVC(C=1.0, random_state=0)
    clf.fit(vec.fit_transform(texts), labels)
    assert len(clf.classes_) == 2
    assert np.asarray(clf.decision_function(vec.transform(["你好呀"]))).ndim == 1, \
        "前提变了：本次 sklearn 版本下二分类不再返回一维数组，用例需要重写"

    space = ["chat"] + [CALL_PREFIX + n for n in C.tool_label_space()]
    model = LexicalRouterModel(vectorizer=vec, classifier=clf, label_space=space,
                               meta={"tool_label_space": list(C.tool_label_space())})
    label, confidence = model.predict("帮我看看分布")
    assert label in set(space)
    assert 0.0 < confidence <= 1.0
    assert abs(float(model.proba("你好").sum()) - 1.0) < 1e-9
    assert model.staleness() is None


# ---------------------------------------------------------------------------
# 五、契约：intent 允许为空（仅升级路径）
# ---------------------------------------------------------------------------


def test_decision_intent_optional_only_for_escalation():
    escalate = RouterDecision(escalate=True, escalate_reason=EscalationReason.OUT_OF_SCOPE)
    assert validate_decision(escalate).well_formed

    # 非升级 + 无 tool + 无 intent ⇒ 不自洽（必须报问题，不能静默通过）
    bogus = RouterDecision()
    outcome = validate_decision(bogus)
    assert not outcome.well_formed
    assert any("未判定" in p for p in outcome.problems)


def test_escalate_without_reason_is_rejected():
    outcome = validate_decision(RouterDecision(escalate=True))
    assert not outcome.well_formed


# ---------------------------------------------------------------------------
# 六、shadow 埋点：只记录、不改行为、失败不抛
# ---------------------------------------------------------------------------


@pytest.fixture()
def shadow_env(tmp_path, monkeypatch):
    """打开 shadow 档，并把产物/trace 都锚到 tmp_path（绝不碰工作区真产物）。"""
    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", "shadow", raising=False)
    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODEL_DIR",
                        str(tmp_path / "router_artifacts"), raising=False)
    T._WARNED.clear()
    return tmp_path / "router_artifacts" / T.SHADOW_FILE


def test_config_default_is_off():
    assert Settings.model_fields["LOCAL_ROUTER_MODE"].default == "off"
    assert Settings.model_fields["LOCAL_ROUTER_CONFIDENCE_THRESHOLD"].default == 0.0
    assert Settings(LOCAL_ROUTER_MODE="off").local_router_summary()["active"] is False
    assert Settings(LOCAL_ROUTER_MODE="shadow").local_router_summary()["active"] is True


def test_trace_disabled_writes_nothing(shadow_env, monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", "off", raising=False)
    assert T.enabled() is False
    T.shadow_route(run_id="r-1", session_id="s-1", request=_req(),
                   rules_mode="agent", rules_reason="测试")
    T.record_outcome(run_id="r-1", session_id="s-1", status="completed",
                     executed_tools=["eda.distribution"], planned_tools=["eda.distribution"])
    assert not shadow_env.exists(), "off 档下不应产生任何文件"


def test_trace_records_route_and_outcome_joined_by_run_id(shadow_env):
    T.shadow_route(run_id="r-7", session_id="s-3", request=_req("看看这批数据的分布", bound=7),
                   rules_mode="agent", rules_reason="命中数据任务关键词：分布")
    T.record_outcome(run_id="r-7", session_id="s-3", status="completed",
                     executed_tools=["eda.distribution", None], planned_tools=["eda.distribution"],
                     elapsed=1.25)

    records = list(T.iter_records())
    assert [r["kind"] for r in records] == ["route", "outcome"]
    assert {r["run_id"] for r in records} == {"r-7"}

    route = records[0]
    assert route["rules"]["mode"] == "agent"
    assert route["utterance"] == "看看这批数据的分布"
    assert route["bound_dataset_id"] == 7
    assert route["n_columns"] == 2
    assert route["router"]["available"] is True
    assert route["router"]["route"].startswith(("chat", "call::", "ask::", "escalate::"))

    outcome = records[1]
    assert outcome["executed_tools"] == ["eda.distribution"], "空工具名应被剔除"
    assert outcome["elapsed"] == 1.25


def test_trace_survives_router_failure(shadow_env, monkeypatch):
    """Router 内部炸了也必须留下一条 `available=False` 记录，而不是把异常抛给调用方。"""
    import app.local_router.router as R

    def _boom(*_args, **_kwargs):
        raise RuntimeError("模拟模型加载失败")

    monkeypatch.setattr(R, "route_request", _boom)
    T.shadow_route(run_id="r-9", session_id="s-9", request=_req(),
                   rules_mode="chat", rules_reason="未命中数据任务关键词")
    records = list(T.iter_records())
    assert len(records) == 1
    assert records[0]["router"]["available"] is False
    assert "模拟模型加载失败" in records[0]["router"]["error"]


def test_trace_never_raises_on_bad_input(shadow_env):
    T.shadow_route(run_id="r-x", session_id="s-x", request=object(),
                   rules_mode="agent", rules_reason="对象没有 utterance 属性")
    assert len(list(T.iter_records())) == 1, "异常输入也必须留下记录，而不是静默丢失"


def test_runtime_hooks_swallow_everything():
    """`AgentRuntime._trace_*` 的兜底：传入完全非法的对象也不得抛异常。

    直接以未绑定方式调用，避免为了测埋点而构造整个 AgentRuntime（需要数据库/引擎）。
    """
    from app.agent.runtime.runtime import AgentRuntime

    AgentRuntime._trace_route(None, None, None, "agent", "reason")  # type: ignore[arg-type]
    AgentRuntime._trace_outcome(None, None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 七、可观测性：只读状态端点
# ---------------------------------------------------------------------------


def test_settings_endpoint_exposes_local_router_state(client):
    """「打开了 shadow 却没数据」是最常见的困惑 —— 状态端点必须能直接回答。

    它同时暴露「产物不存在 / 已过期」，而不是让用户去猜。

    断言按**当前档位**分支，不写死 `off`：本机 `backend/.env` 一旦把
    `LOCAL_ROUTER_MODE` 设成 shadow（正式跑数据的必要前提），端点就会真的加载产物，
    写死 `artifact_loaded is False` 会让用例随 `.env` 红绿。
    """
    body = client.get("/api/v1/settings/local_router").json()
    assert body["success"] is True
    data = body["data"]
    assert data["mode"] in {"off", "shadow", "guard"}
    assert data["active"] is (data["mode"] != "off")
    for key in ("artifact", "artifact_exists", "artifact_loaded", "artifact_staleness"):
        assert key in data, f"状态端点应暴露 {key}，否则用户无法自查产物是否就位"
    assert isinstance(data["artifact_exists"], bool)

    if data["active"]:
        # active 档会真的去加载产物：loaded 由产物是否就位/可用决定，
        # 但不允许「文件在却静默加载不了」——那种静默失效必须给出人话原因。
        if data["artifact_exists"] and not data["artifact_loaded"]:
            assert data["artifact_staleness"], "产物存在却加载不了，必须说明原因而不是留空"
    else:
        # off 档刻意不碰模型，避免无谓的磁盘读取
        assert data["artifact_loaded"] is False
        assert data["artifact_staleness"] is None
