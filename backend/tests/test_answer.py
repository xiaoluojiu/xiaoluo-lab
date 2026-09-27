"""答案渲染：模板是底线，LLM 只是可选润色。

「模板渲染不能标成大模型回答」是硬要求 —— 前端靠 answer_source 显示来源标签，
标错等于欺骗用户。
"""

from __future__ import annotations

from typing import Any

from app.agent.answer import (
    _FACTS_MAX_CHARS,
    _MAX_FACTS_PER_STEP,
    _advice,
    _collect_facts,
    _fmt_scalar,
    _facts_for_llm,
    render,
    render_chat,
    render_template,
)
from app.agent.models import ANSWER_SOURCE_RULE
from app.agent.playbooks import PlaybookStep


class _Result:
    def __init__(self, success: bool = True, summary: str = "", data: Any = None,
                 errors: list[str] | None = None, warnings: list[str] | None = None):
        self.success = success
        self.summary = summary
        self.data = data
        self.errors = errors or []
        self.warnings = warnings or []


def _step(tool: str = "dataset.quality", title: str = "数据质量检查") -> PlaybookStep:
    return PlaybookStep(tool=tool, title=title)


# ---------------------------------------------------------------- 格式化


def test_fmt_scalar_trims_float_noise():
    assert _fmt_scalar(0.8364929) == "0.8365"
    assert _fmt_scalar(1.0) == "1"
    assert _fmt_scalar(True) == "是"
    assert _fmt_scalar(42) == "42"


def test_collect_facts_handles_dict_list_scalar():
    data = {"total": 3, "items": [{"message": "缺少年龄"}, {"message": "重复邮箱"}], "ok": False}
    facts = _collect_facts(data)
    joined = "\n".join(facts)
    assert "total：3" in joined
    assert "ok：否" in joined
    assert "缺少年龄" in joined and "重复邮箱" in joined


def test_conditions_keep_their_operator():
    """筛选条件里的比较符不能丢。

    只给 column + value 时，模型会把它读成等式 —— 实测答案写
    「筛选条件：credit_amount = 5000；age = 60」，而真实条件是「大于」，
    语义整个反了。
    """
    facts = _collect_facts(
        {
            "params": {
                "conditions": [
                    {"column": "credit_amount", "op": "gt", "value": 5000},
                    {"column": "age", "op": "gte", "value": 60},
                ]
            }
        }
    )
    joined = "\n".join(facts)
    assert "credit_amount > 5000" in joined
    assert "age ≥ 60" in joined


def test_collect_facts_is_bounded():
    """工具可能返回上百个键，展示必须封顶，否则答案比原始数据还长。

    上限值写在 ``_MAX_FACTS_PER_STEP`` 里（调过：12 会在双侧结果里把右半边整段切掉），
    这里断言「有上限」而不是「恰好 12」—— 断言具体数字会让每次调参都变成一次改测试。
    """
    data = {f"k{i}": i for i in range(500)}
    assert len(_collect_facts(data)) <= _MAX_FACTS_PER_STEP


def test_collect_facts_on_none():
    assert _collect_facts(None) == []


# ---------------------------------------------------------------- 模板渲染


def test_render_template_shows_summary_and_facts():
    result = _Result(summary="共 2 个问题", data={"total": 2})
    text = render_template("检查质量", [(_step(), result)])
    assert "数据质量检查" in text
    assert "共 2 个问题" in text
    assert "total：2" in text


def test_render_template_reports_failure_honestly():
    result = _Result(success=False, errors=["数据集不存在"])
    text = render_template("检查质量", [(_step(), result)])
    assert "未成功" in text and "数据集不存在" in text


def test_render_template_empty_steps_has_fallback():
    assert render_template("检查质量", []) == "本次没有产生可展示的结果。"


def test_render_template_skips_none_results():
    assert "本次没有产生可展示的结果" in render_template("x", [(_step(), None)])


# ---------------------------------------------------------------- 零 Token


def test_render_without_provider_is_rule_source():
    text, source = render("检查质量", [(_step(), _Result(summary="ok"))], provider=None)
    assert source is ANSWER_SOURCE_RULE
    assert source.by_llm is False
    assert text


def test_render_chat_without_provider_is_honest():
    text, source = render_chat("你好", provider=None)
    assert source is ANSWER_SOURCE_RULE
    assert "API Key" in text  # 必须说明自己答不了，而不是装作回答了


# ---------------------------------------------------------------- LLM 降级


class _ExplodingProvider:
    def chat(self, *args, **kwargs):
        raise RuntimeError("服务不可用")


def test_render_falls_back_when_llm_explodes():
    text, source = render("检查质量", [(_step(), _Result(summary="ok"))], provider=_ExplodingProvider())
    assert source is ANSWER_SOURCE_RULE
    assert "ok" in text


def test_render_chat_falls_back_when_llm_explodes():
    text, source = render_chat("你好", provider=_ExplodingProvider())
    assert source is ANSWER_SOURCE_RULE
    assert "无法调用大模型" in text


class _EmptyProvider:
    def chat(self, *args, **kwargs):
        class _Resp:
            content = "   "
            usage = type("U", (), {"input_tokens": 0, "output_tokens": 0})()

        return _Resp()


def test_render_falls_back_on_empty_content():
    text, source = render("x", [(_step(), _Result(summary="ok"))], provider=_EmptyProvider())
    assert source is ANSWER_SOURCE_RULE
    assert "ok" in text


def test_facts_for_llm_is_capped():
    """喂给模型的摘要必须封顶，否则一次润色就能吃掉全部 Token 预算。

    边界直接钉在 ``_FACTS_MAX_CHARS`` 上加一点截断后缀的余量，
    而不是写死一个数字——预算调大时这个测试不该变成谎话。
    """
    big = _Result(summary="很" * 10000)
    facts = _facts_for_llm([(_step(), big)])
    assert len(facts) <= _FACTS_MAX_CHARS + 64


def test_facts_for_llm_keeps_enough_budget_for_field_lists():
    """★ 事故：预算只有 3000 字时，dataset.relation 的字段清单被截到只剩结论，
    模型于是写出「共有 3 个同名字段（具体字段名本次未列出）」这种废话。
    字段名是这类工具的全部价值，预算必须容得下它们。"""
    relation = _Result(
        summary="两个数据集有 3 个同名字段",
        data={
            "shared_columns": [
                {"column": f"field_{i}", "dtype": "Utf8"} for i in range(20)
            ]
        },
    )
    facts = _facts_for_llm([(_step("dataset.relation"), relation)])
    assert "field_19" in facts
    assert _FACTS_MAX_CHARS >= 6000


# ---------------------------------------------------------------- 指标解读


def test_weak_silhouette_is_interpreted_not_just_printed():
    """★ 事故：答案原样列出 silhouette 0.1316，用户无从判断好坏。"""
    result = _Result(
        summary="训练成功：kmeans",
        data={"metrics": {"cluster_count": 8, "silhouette": 0.1316}},
    )
    text = render_template("做个聚类", [(_step("ml.train", "训练模型"), result)])
    assert "0.1316" in text
    assert "偏低" in text
    assert "不足以支撑结论" in text


def test_strong_silhouette_is_not_doomed():
    result = _Result(data={"metrics": {"silhouette": 0.72}})
    text = render_template("做个聚类", [(_step("ml.train", "训练模型"), result)])
    assert "较清晰" in text
    assert "偏低" not in text


def test_needs_target_says_no_model_was_trained():
    """没定下目标列就是没建模，答案必须说清楚，不能让用户以为跑完了。"""
    result = _Result(
        summary="任务类型：clustering",
        data={
            "needs_target": True,
            "target_candidates": {
                "regression": ["fare_amount", "trip_distance"],
                "classification": ["payment_type"],
            },
        },
    )
    text = render_template("做个机器学习分析", [(_step("ml.detect_task", "识别任务"), result)])
    assert "没有" in text and "训练任何模型" in text
    assert "fare_amount" in text


def test_large_row_count_suggests_sampling_first():
    result = _Result(data={"row_count": 2_964_624, "task": "clustering"})
    text = render_template("做个分析", [(_step("ml.detect_task", "识别任务"), result)])
    assert "2,964,624" in text
    assert "抽样" in text


def test_no_advice_section_when_nothing_to_say():
    """没有可解读的信号时不要硬凑一段「建议」。"""
    text = render_template("检查质量", [(_step(), _Result(summary="共 2 个问题", data={"total": 2}))])
    assert "解读与下一步" not in text


def test_advice_reaches_the_llm_prompt():
    """润色时也要带上解读，否则模型会把「训练成功」讲得更顺却仍不提 0.13 的含义。"""
    result = _Result(summary="训练成功", data={"metrics": {"silhouette": 0.1316}})
    facts = _facts_for_llm([(_step("ml.train", "训练模型"), result)])
    assert "解读与下一步" in facts
    assert "偏低" in facts


def test_relation_facts_keep_both_sides():
    """双侧结果必须两边都进事实摘要。

    dataset.relation 返回 left / right 两段。旧的 8 条上限刚好在右表列名之前用完，
    模型于是如实回答「右表列名本次未获取」—— 不是模型编造，是我们先把事实掐掉了。
    """
    data = {
        "left": {"dataset_id": 1, "name": "flights", "rows": 500, "column_count": 6,
                 "column_names": ["id", "DepDelay", "carrier"]},
        "right": {"dataset_id": 2, "name": "orders", "rows": 500, "column_count": 7,
                  "column_names": ["id", "amount", "status"]},
        "shared_columns": ["id"],
    }
    facts = _collect_facts(data)
    joined = "\n".join(facts)
    assert "flights" in joined
    assert "orders" in joined, "右表整个被截掉了"
    assert "amount" in joined, "右表列名被截掉了"


def test_facts_for_llm_matches_template_budget():
    """喂给 LLM 的事实不能比模板答案还少，否则有 LLM 时答案反而更空。"""
    data = {f"k{i}": i for i in range(40)}
    step = _step()
    result = _Result(summary="s", data=data)
    llm_facts = len(_facts_for_llm([(step, result)]).splitlines())
    template_facts = len(_collect_facts(data))
    assert llm_facts >= template_facts


def test_needs_target_advice_is_dropped_when_training_succeeded():
    """「没有训练任何模型」这句话不能和训练指标同时出现。

    真实链路：detect_task 推断不出目标列（needs_target=True）→ 用户补 target →
    ml.train 成功。旧逻辑照抄 detect 的结论，答案于是先说「本次没有训练任何模型」，
    紧接着列出 MAE / R² —— 用户第一眼看到「没训」，就会以为下面那串指标是假的。
    """
    detect = _step("ml.detect_task", "识别任务")
    train = _step("ml.train", "训练模型")
    steps = [
        (detect, _Result(data={"needs_target": True, "target_candidates": {"regression": ["value"]}})),
        (train, _Result(data={"run_id": 13, "metrics": {"r2": 0.8}})),
    ]
    advice = _advice(steps)
    assert not any("目标列未确定" in line for line in advice), advice


def test_needs_target_advice_is_kept_when_no_training():
    """反过来：真的没训就必须说清楚，否则用户以为模型已经跑完了。"""
    detect = _step("ml.detect_task", "识别任务")
    steps = [(detect, _Result(data={"needs_target": True}))]
    assert any("目标列未确定" in line for line in _advice(steps))


# ---------------------------------------------------------------- 相关性矩阵


def test_correlation_matrix_is_summarized_into_top_pairs():
    """7×7 矩阵原样展开是 49 个数字 → 占满事实条数，模型一个系数都读不到。

    实测答案：「相关性矩阵已计算完成……但事实摘要中未给出任何相关系数数值」——
    明明算出来了，却告诉用户没算。
    """
    cols = ["a", "b", "c"]
    matrix = {
        "a": {"a": 1.0, "b": 0.9123, "c": -0.2},
        "b": {"a": 0.9123, "b": 1.0, "c": 0.05},
        "c": {"a": -0.2, "b": 0.05, "c": 1.0},
    }
    data = {
        "method": "auto",
        "columns": cols,
        # 这两个键每列一条、每条都是 "pearson"：没有它们，20 条事实会被它们吃光
        "methods_used": {c: "pearson" for c in cols},
        "pair_methods": {f"{x}|{y}": "pearson" for x in cols for y in cols},
        "matrix": matrix,
    }
    joined = "\n".join(_collect_facts(data))
    assert "0.9123" in joined, joined
    assert "a × b" in joined, joined
    # 自相关（1.0）不该被当成「强相关对」推给用户
    assert "a × a" not in joined
    # 低价值的「方法名」字典不该逐条占位置
    assert "pair_methods" not in joined


def test_aggregate_preview_rows_reach_the_answer():
    """聚合/筛选回执的明细行必须进事实摘要。

    实测：``data.aggregate`` 跑成功了，但答案写「结果共 8 行……没有显示具体的
    均值、笔数数值」—— preview 是一串裸记录（没有 column/outlier_count 这类
    约定键），旧逻辑只报条数，用户算完了却看不到数。
    """
    data = {
        "rows": 8,
        "preview": [
            {"purpose": "car", "credit_amount_mean": 5372.5, "credit_amount_count": 337},
            {"purpose": "radio/tv", "credit_amount_mean": 3120.25, "credit_amount_count": 280},
        ],
    }
    joined = "\n".join(_collect_facts(data))
    assert "car" in joined
    assert "5372.5" in joined, joined
    assert "337" in joined, joined


# ---------------------------------------------------------------- 文案口径


def test_render_prompt_forbids_internal_jargon():
    """「本次未获取该信息 / 事实摘要中只有……」是内部口径，不能出现在答案里。

    实测：用户要两张图，答案写「散点图本次未获取该信息。原因：事实摘要中只包含
    直方图结果……如需散点图，请重新执行」—— 既暴露内部数据结构，又把活推回用户。
    """
    captured: dict[str, str] = {}

    class _Provider:
        model = "fake"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            captured["system"] = messages[0].content
            from app.agent.llm import LLMResponse

            return LLMResponse(content="答案", usage=type("U", (), {"input_tokens": 0, "output_tokens": 0})())

    render("看看相关性", [(_step("eda.correlation"), _Result(summary="s", data={"matrix": {}}))], provider=_Provider())
    system = captured["system"]
    for banned in ("本次未获取", "未获取该信息", "重新执行"):
        assert banned in system, f"提示词里必须明令禁止：{banned}"
    assert "禁止出现" in system
    # 另一个变体：模型用「本次只读取了…没有执行任何…操作」描述自己干了什么，
    # 实测「筛选后还剩多少行」答成「无法给出：本次只读取了数据集基本信息，
    # 没有执行任何筛选操作」，紧接着又把 473 行写出来 —— 自相矛盾。
    assert "禁止描述系统自己做了什么" in system


def test_render_prompt_forbids_leading_with_a_negation():
    """答案的第一句不能是「无法 / 还没 / 没有」。

    实测「筛选后还剩多少行」得到「筛选条件还没给，无法算出剩余行数。当前数据集
    最新版本（version 8）共 164 行、18 列。」—— 第一句否定、第二句又把数字
    给了，自相矛盾，用户第一眼读到的是「我白问了」。
    """
    captured: dict[str, str] = {}

    class _Provider:
        model = "fake"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            captured["system"] = messages[0].content
            from app.agent.llm import LLMResponse

            return LLMResponse(content="答案", usage=type("U", (), {"input_tokens": 0, "output_tokens": 0})())

    render("还剩多少行", [(_step("dataset.inspect"), _Result(summary="s", data={"rows": 164}))], provider=_Provider())
    system = captured["system"]
    assert "第一句必须是已拿到的事实或数字" in system
    for banned in ("无法 / 不能 / 还没 / 尚未 / 没有", "如果你想…可以说…", "version 8"):
        assert banned in system, f"提示词里必须写清楚：{banned}"


def test_render_strips_result_reference_before_it_reaches_the_model():
    """喂给模型的用户请求要先剥掉「筛选后 / 清洗后」这类回指状语。

    模型看到字面上的「筛选」、手里却没有筛选这一步的事实，就会先写一句
    「无法算出」，再报出真实行数。剥掉之后它只会老老实实报数。
    """
    captured: dict[str, str] = {}

    class _Provider:
        model = "fake"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            captured["user"] = messages[-1].content
            from app.agent.llm import LLMResponse

            return LLMResponse(content="答案", usage=type("U", (), {"input_tokens": 0, "output_tokens": 0})())

    render(
        "筛选后还剩多少行",
        [(_step("dataset.inspect"), _Result(summary="s", data={"rows": 164}))],
        provider=_Provider(),
    )
    assert "筛选" not in captured["user"], captured["user"]
    assert "还剩多少行" in captured["user"]


def test_facts_tell_the_model_a_step_failed():
    """失败的步骤必须进事实摘要，并且要求模型**明说**失败。

    实测：工作流创建失败（ml.train 缺 target_column），模型答成
    「清洗与训练工作流可按以下顺序搭建：1… 2… 3…」—— 只字不提没建成，
    用户以为流程已经搭好了。
    """
    steps = [
        (_step("dataset.inspect", "查看数据集概况"), _Result(summary="50 行 × 18 列", data={"rows": 50})),
        (
            _step("workflow.build_and_run", "创建并执行工作流"),
            _Result(success=False, errors=["节点 'train' 缺少必需参数：['target_column']"]),
        ),
    ]
    facts = _facts_for_llm(steps)
    assert "没有成功" in facts
    assert "workflow.build_and_run" in facts
    assert "target_column" in facts
    assert "不要把它包装成" in facts


def test_render_prompt_forbids_plan_disguise_and_version_dumping():
    """两个实测出来的噪音：把失败写成计划、把版本年表整个列出来。"""
    captured: dict[str, str] = {}

    class _Provider:
        model = "fake"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            captured["system"] = messages[0].content
            from app.agent.llm import LLMResponse

            return LLMResponse(content="答案", usage=type("U", (), {"input_tokens": 0, "output_tokens": 0})())

    render("还剩多少行", [(_step("dataset.inspect"), _Result(summary="s", data={"rows": 50}))], provider=_Provider())
    system = captured["system"]
    assert "不要把它包装成" in system
    assert "不要罗列各个历史版本" in system
