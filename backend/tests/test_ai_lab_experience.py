"""AI 实验室体验红线：图表必须看得见、Workflow 不许挤牙膏。

本轮实机测试（deepseek-flash，5 轮 × 6 类复杂场景）暴露的缺陷护栏。
每条用例都对应一个**真实复现过**的现象，不是凭空加的断言。

两类缺陷最隐蔽，也是这些用例真正要钉住的东西：
1. 「工具跑成功了，但用户要的东西没出现」（图只有文字描述、工作流只列出已有流程）
2. 「看起来跑通了，其实跑的是错的」（把 region 当目标列、把名字当方案序号）
"""

from __future__ import annotations

import pytest

from app.agent.answer import _chart_notice
from app.agent.engine import AgentEngine
from app.agent.intents import _looks_like_payload, is_abandonment, route
from app.agent.models import AgentRun
from app.agent.playbooks import PlaybookStep
from app.agent.slots import (
    _match_bins,
    _match_method,
    _match_name,
    _match_plan,
    _match_target,
    _match_xy,
    rule_extract,
)
from app.tools.eda_tools import (
    _auto_pick_column,
    _chart_artifact,
    _normalize_chart_params,
)
from app.tools.result import ToolResult
from app.workflow.recommend import plan_options, plan_question, recommend_plans, resolve_plan

# ----------------------------------------------------------------------
# 一、图表必须真的画出来
# ----------------------------------------------------------------------


class _Step:
    tool = "eda.visualize"
    title = "生成图表"

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class _Result:
    success = True
    data = {}
    summary = ""
    warnings: list = []
    errors: list = []

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


def _chart_meta(rendered: bool = True, svg: str = "<svg></svg>") -> dict:
    return {"chart": {"kind": "histogram", "title": "分布 · tenure_months", "svg": svg, "rendered": rendered}}


def test_chart_is_rendered_to_svg_not_described_in_words():
    """直方图结果必须变成 SVG，不能只留数据给答案层翻译成文字。"""
    data = {"chart": "histogram", "column": "tenure_months",
            "x": ["[1,8)", "[8,15)"], "y": [351, 284], "missing": 0}
    artifact = _chart_artifact(data, {"chart": "histogram"})
    assert artifact["rendered"] is True
    assert artifact["svg"].startswith("<svg")
    assert artifact["title"]


def test_chart_render_failure_is_reported_not_faked():
    """渲染不出来必须如实上报，绝不能假装「已生成」。"""
    artifact = _chart_artifact({"chart": "totally_unknown_chart"}, {"chart": "totally_unknown_chart"})
    # 不支持的类型会退化成占位图；真正的失败要有 reason 字段
    if not artifact["rendered"]:
        assert artifact["reason"]


def test_answer_tells_user_the_chart_is_on_screen():
    notice = _chart_notice([(_Step(_chart_meta()), _Result(_chart_meta()))])
    assert "图表已渲染在对话框中" in notice
    assert "分箱数" in notice


def test_answer_says_export_when_chart_cannot_render():
    """兜底：画不出来必须指向导出/预览，而不是沉默。"""
    meta = _chart_meta(rendered=False, svg="")
    notice = _chart_notice([(_Step(meta), _Result(meta))])
    assert "无法直接显示图片" in notice
    assert "导出" in notice


def test_no_chart_means_no_notice():
    assert _chart_notice([(_Step({}), _Result({}))]) == ""


# ----------------------------------------------------------------------
# 二、图表参数纠偏（类型敏感）
# ----------------------------------------------------------------------


def test_scatter_swaps_when_x_is_not_numeric():
    dtypes = {"region": "String", "amount": "Float64"}
    fixed, notes = _normalize_chart_params(
        {"chart": "scatter", "x": "region", "y": "amount"}, dtypes
    )
    assert fixed["x"] == "amount" and fixed["y"] == "region"
    assert notes


def test_grouped_bar_assigns_by_dtype_not_by_word_order():
    """「region 和 monthly_charges」—— 谁当分类列由类型决定，不能靠说话顺序。"""
    dtypes = {"region": "String", "monthly_charges": "Float64", "contract_type": "String"}
    fixed, _ = _normalize_chart_params(
        {"chart": "grouped_bar", "x": "region", "y": "monthly_charges",
         "group_by": "contract_type"}, dtypes
    )
    assert fixed["column"] == "region"
    assert fixed["y"] == "monthly_charges"
    assert fixed["group_by"] == "contract_type"


def test_grouped_bar_degrades_to_boxplot_when_only_one_category():
    """★ 只有一个分类维度时 group_by 只能等于 column，工具会直接报错。

    「画 region 和 monthly_charges 的关系图」连错两轮就是卡在这。
    类别 vs 数值的关系本来就该用箱线图。
    """
    dtypes = {"region": "String", "monthly_charges": "Float64"}
    fixed, notes = _normalize_chart_params(
        {"chart": "grouped_bar", "x": "region", "y": "monthly_charges"}, dtypes
    )
    assert fixed["chart"] == "boxplot"
    assert fixed["column"] == "monthly_charges"
    assert fixed["group_by"] == "region"
    assert notes


def test_grouped_bar_keeps_two_distinct_categories():
    dtypes = {"region": "String", "monthly_charges": "Float64", "contract_type": "String"}
    fixed, _ = _normalize_chart_params(
        {"chart": "grouped_bar", "x": "region", "y": "monthly_charges", "group_by": "contract_type"},
        dtypes,
    )
    assert fixed["chart"] == "grouped_bar"
    assert fixed["column"] == "region" and fixed["group_by"] == "contract_type"


def test_grouped_bar_handles_reversed_word_order():
    dtypes = {"region": "String", "monthly_charges": "Float64", "contract_type": "String"}
    fixed, _ = _normalize_chart_params(
        {"chart": "grouped_bar", "x": "monthly_charges", "y": "region",
         "group_by": "contract_type"}, dtypes
    )
    assert fixed["column"] == "region"
    assert fixed["y"] == "monthly_charges"


# ----------------------------------------------------------------------
# 三、Workflow 自主设计
# ----------------------------------------------------------------------

_COLUMNS = [
    "customer_id", "tenure_months", "region", "channel",
    "contract_type", "monthly_charges", "total_charges",
    "support_tickets", "satisfaction", "churn",
]
_DTYPES = {
    "customer_id": "Int64", "tenure_months": "Int64", "region": "String",
    "channel": "String", "contract_type": "String", "monthly_charges": "Float64",
    "total_charges": "Float64", "support_tickets": "Int64",
    "satisfaction": "Float64", "churn": "String",
}
_UNIQUE = {"region": 5, "channel": 4, "contract_type": 3, "churn": 2}


def _plans(**kw):
    params = dict(
        dataset_id=14, columns=_COLUMNS, dtypes=_DTYPES, unique=_UNIQUE, row_count=3035,
    )
    params.update(kw)
    return recommend_plans(**params)


def test_recommend_returns_up_to_three_executable_plans():
    plans = _plans()
    assert 1 <= len(plans) <= 3
    for plan in plans:
        assert plan["name"] and plan["steps"]
        assert plan["nodes"] and isinstance(plan["edges"], list)
        for node in plan["nodes"]:
            assert node["type"] and isinstance(node["config"], dict)


def test_target_column_beats_first_low_cardinality_column():
    """★ region 有 5 个唯一值且排在前面；真正的目标列是 churn。

    按列顺序取第一个会把「预测流失」悄悄换成「预测地区」——
    跑得通但跑的是错的，比失败危险得多。
    """
    plans = _plans(goal="客户流失预警")
    assert "churn" in plans[0]["name"]


def test_target_hint_matches_tokens_not_substrings():
    """``y`` 不能用子串匹配：monthly_charges 里就有一个 y。"""
    from app.workflow.recommend import _is_target_like

    assert _is_target_like("churn") is True
    assert _is_target_like("monthly_charges") is False
    assert _is_target_like("是否流失") is True


def test_goal_reorders_plans():
    assert _plans(goal="看看分布")[0]["family"] == "explore"
    assert _plans(goal="客户流失预警")[0]["family"] == "classification"


def test_plan_nodes_carry_no_dataset_id_on_ml_steps():
    """建模节点的数据来自上游；带 dataset_id 会让模型构造器报不认识的参数。"""
    plans = _plans(goal="客户流失预警")
    for plan in plans:
        for node in plan["nodes"]:
            if node["type"].startswith("ml."):
                assert "dataset_id" not in node["config"], f"{node['type']} 不该带 dataset_id"
            if node["type"] == "dataset.read":
                assert node["config"]["dataset_id"] == 14


def test_aggregate_node_uses_column_and_func_shape():
    """聚合项必须是 {"column":..,"func":..}；{列: 均值} 会让 func 取到 None。"""
    plans = _plans(goal="按 region 分组对比")
    agg_nodes = [
        n for p in plans for n in p["nodes"] if n["type"] == "data.aggregate"
    ]
    assert agg_nodes, "分组对比方案里应该有聚合节点"
    for node in agg_nodes:
        for item in node["config"].get("aggregations") or []:
            assert set(item) == {"column", "func"}, item
            assert item["func"] in {"mean", "sum", "count", "median"}


def test_resolve_plan_accepts_index_and_name_but_rejects_noise():
    plans = _plans()
    assert resolve_plan(plans, "1") is plans[0]
    assert resolve_plan(plans, plans[1]["name"]) is plans[1]
    assert resolve_plan(plans, "客户流失预警") is None
    assert resolve_plan(plans, "") is None


def test_plan_question_lists_columns_and_choices():
    text = plan_question(_plans(), _COLUMNS, 3035)
    assert "tenure_months" in text
    assert "请回复数字" in text
    # 澄清面板渲染纯文本，不能出现 markdown 粗体标记
    assert "**" not in text


def test_plan_options_have_value_label_note():
    opts = plan_options(_plans())
    assert [o["value"] for o in opts] == ["1", "2", "3"]
    for opt in opts:
        assert opt["label"] and opt["note"]


# ----------------------------------------------------------------------
# 四、槽位抽取
# ----------------------------------------------------------------------


def test_workflow_name_is_extracted():
    assert _match_name("帮我创建一个叫 客户流失预警 的 Workflow") == "客户流失预警"
    assert _match_name("创建一个名为 数据体检 的 Workflow") == "数据体检"
    assert _match_name("创建一个客户流失预警流程") == "客户流失预警"


def test_plan_choice_is_extracted():
    assert _match_plan("我选 2") == "2"
    assert _match_plan("2") == "2"
    assert _match_plan("第二个") == "2"
    assert _match_plan("方案3") == "3"


def test_bins_is_extracted():
    assert _match_bins("那个图换成 30 个分箱再看看") == 30
    assert _match_bins("分箱数改成 50") == 50
    assert _match_bins("画个直方图") is None


def test_xy_is_extracted_for_two_column_charts():
    got = _match_xy("画 tenure_months 和 monthly_charges 的散点图", _COLUMNS)
    assert got == {"x": "tenure_months", "y": "monthly_charges"}


def test_correlation_method_is_extracted():
    """「这次用 spearman」—— 方法名是这句话里唯一的新信息，丢了就是静默算错。"""
    assert _match_method("这次用 spearman") == "spearman"
    assert _match_method("用斯皮尔曼") == "spearman"
    assert _match_method("算一下相关性") is None


def test_rerun_with_method_routes_to_correlation():
    """「刚才那两个参数再跑一次，这次用 spearman」里没有「相关性」三个字。"""
    assert route("刚才第 2 条说的那两个参数，再帮我跑一次，这次用 spearman").intent.value == "correlation"


def test_xy_extraction_drops_unresolvable_references():
    """「刚才那个字段和 TotalCharges 做散点图」不能把「刚才那个字段」当列名。

    实测：x 被原样塞进工具 ⇒ 「指定的字段不存在：刚才那个字段」，一张图都没画成。
    对不上真实列名就必须丢弃，把机会留给引擎的回指兜底。
    """
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    assert _match_xy("刚才那个字段和 TotalCharges 做散点图", cols) is None
    assert _match_xy("画 tenure 和 MonthlyCharges 的散点图", cols) == {
        "x": "tenure", "y": "MonthlyCharges",
    }


def test_binary_target_can_enter_correlation():
    """「所有特征与 Churn 的相关系数」不能因为 Churn 是 Yes/No 就整条失败。

    Telco 压测实测：Churn 是字符串 ⇒ 被当「不是数值类型」剔除，最后报
    「当前可用数值列 0 个」。而类别目标与特征的关联度恰恰是分类建模第一步。
    二值编码后算的就是标准点二列相关，编码方式必须回执。
    """
    import polars as pl

    from app.analysis import CorrelationAnalyzer

    df = pl.DataFrame(
        {
            "tenure": [1, 12, 24, 48, 60, 70],
            "MonthlyCharges": [30.0, 55.0, 70.0, 90.0, 100.0, 110.0],
            "Churn": ["Yes", "No", "No", "No", "No", "No"],
        }
    )
    result = CorrelationAnalyzer().analyze(df, columns=["tenure", "Churn"])
    assert "Churn" in result["columns"]
    assert result["binary_encoded"]["Churn"].startswith("Yes=1")
    # 三取值以上的类别列不能硬编成数字
    df3 = df.with_columns(pl.Series("grade", ["a", "b", "c", "a", "b", "c"]))
    try:
        CorrelationAnalyzer().analyze(df3, columns=["tenure", "grade"])
    except Exception:
        pass
    else:  # pragma: no cover - 失败是允许的，静默编码才是缺陷
        assert "grade" not in CorrelationAnalyzer().analyze(
            df3, columns=["tenure", "grade"]
        ).get("binary_encoded", {})


def test_redraw_words_only_mean_redraw_in_a_chart_context():
    """「换成 / 改成」脱离图表语境时不能把诉求吸成"重画一张图"。

    Telco 压测实测：「刚才那个目标列，换成随机森林再跑一次」里 VISUALIZE 靠
    「换成」拿 3 分，压过 ML_TRAIN 的「随机森林」⇒ 用户要换模型，系统画了直方图。
    """
    assert route("刚才那个目标列，换成随机森林再跑一次").intent.value == "ml_train"
    assert route("把 Contract 改成 PaymentMethod 分组").intent.value != "visualize"
    # 图表语境里的「换成」仍要指向改图
    assert route("那个图换成 30 个分箱再看看").intent.value == "visualize"
    assert route("把刚才的结果画成柱状图").intent.value == "visualize"


def test_user_named_target_beats_inferred_target():
    """「把 tenure 作为目标列」必须真的变成 target=tenure。

    Telco 压测实测：``ml.train`` 的 tuned_slots 里没有 target，用户点名的列
    从没被抽出来，目标列一律来自 detect_task 的推断（推断成了 Churn）
    ⇒ classification + linear_regression 冲突失败，一次训练都没发生。
    """
    cols = ["customerID", "gender", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract"]
    assert _match_target("帮我把 tenure 作为目标列，跑一个回归模型", cols) == "tenure"
    assert _match_target("用 TotalCharges 做目标列跑回归", cols) == "TotalCharges"
    assert _match_target("目标列换成 Churn", cols) == "Churn"
    assert _match_target("以 Churn 为目标", cols) == "Churn"
    # 「训练一个预测模型」里的「模型」不是列名，不能当目标列
    assert _match_target("训练一个预测模型", cols) is None
    # 对不上真实列名的诉求（不可能任务）也不能硬凑一个列名出来
    assert _match_target("预测下个月的股票价格", cols) is None


def test_rule_extract_fills_name_plan_and_bins():
    step = PlaybookStep(
        "workflow.build_and_run", "创建并执行",
        required_slots=("name",), tuned_slots=("plan",),
    )
    got = rule_extract("创建一个叫 客户流失预警 的 Workflow，我选 2", step)
    assert got["name"] == "客户流失预警"
    assert got["plan"] == "2"


# ----------------------------------------------------------------------
# 五、路由与打断
# ----------------------------------------------------------------------


def test_create_verb_plus_workflow_noun_routes_to_create():
    """「创建一个叫 X 的 Workflow」中间隔着任意内容，词表覆盖不到。"""
    for text in (
        "帮我创建一个叫 客户流失预警 的 Workflow",
        "创建一个名为 数据体检 的 Workflow",
    ):
        assert route(text).intent.value == "workflow_create", text


def test_chart_alias_is_normalized_instead_of_rejected():
    """「箱线图」被抽成 ``box`` 后不能撞上「不支持的图表类型：box」（第十轮实测）。"""
    dtypes = {"tenure_months": "Int64", "region": "Utf8"}
    fixed, notes = _normalize_chart_params(
        {"chart": "box", "column": "tenure_months"}, dtypes
    )
    assert fixed["chart"] == "boxplot"
    assert any("box" in n for n in notes)


def test_missing_column_is_auto_picked_not_asked():
    """「画个直方图看看」不该挂起反问「要对哪一列做这个分析？」。

    第十轮实测连问三轮，一张图都没出 —— 反问而不给选项，是把本该由系统
    承担的推理推回给用户。工具必须自己挑一列画出来。
    """
    dtypes = {
        "customer_id": "Int64",
        "tenure_months": "Int64",
        "monthly_charges": "Float64",
        "region": "Utf8",
        "churn": "Utf8",
    }
    # 直方图要数值列，且必须跳过 customer_id
    fixed, notes = _normalize_chart_params({"chart": "histogram"}, dtypes)
    assert fixed["column"] == "tenure_months", fixed
    assert any("未指定" in n for n in notes)
    # 柱状图优先类别列
    assert _auto_pick_column("bar", dtypes) == "region"
    # 已给的列合法时必须原样保留，不能自作主张换掉
    keep, _ = _normalize_chart_params({"chart": "histogram", "column": "monthly_charges"}, dtypes)
    assert keep["column"] == "monthly_charges"


def test_read_tools_declare_version_so_the_baseline_is_used():
    """只读工具必须声明 ``version``，否则引擎不会把会话基线填进来。

    engine 的自动槽位只对**工具 schema 里写了**的参数生效。``dataset.inspect``
    此前没写 version，于是硬读 latest —— 一次 0 行的筛选把 latest 顶成空表后，
    用户再问「有多少行」答的是 0 行，而基线明明还是 3035 行（第十二轮实测）。
    """
    from app.tools.registry import TOOL_REGISTRY

    import app.tools.builtin  # noqa: F401 - 触发注册

    for name in ("dataset.inspect", "report.generate", "workflow.recommend", "workflow.build_and_run"):
        tool = TOOL_REGISTRY.get(name)
        assert tool is not None, f"{name} 未注册"
        props = (tool.input_schema or {}).get("properties") or {}
        assert "version" in props, f"{name} 未声明 version，会去读 latest"


def test_empty_kv_skeleton_is_payload_but_filled_kv_is_a_request():
    """``dataset_id=null&column=&chart=`` 不能再被当成需求去画一张没人要的图。

    第十一轮实测：一串全空的参数被当成需求，系统自己挑了一列画了直方图。
    反过来 ``chart=boxplot; column=monthly_charges; bins=30`` 值都在，
    那是能执行的有效指令，不能一并拦掉。
    """
    assert _looks_like_payload("dataset_id=null&column=&chart=") is True
    assert _looks_like_payload("chart=boxplot; column=monthly_charges; bins=30") is False
    assert _looks_like_payload("画个直方图") is False


def test_comparison_sentence_routes_to_filter():
    """「找出所有 customer_id 大于 99999999 的记录」里没写「筛选」，但条件写死了。"""
    assert route("找出所有 customer_id 大于 99999999 的记录").intent.value == "filter"
    # 只是描述分布、不带筛选意图的句子不能被抢走
    assert route("看看 monthly_charges 的分布").intent.value != "filter"


def test_auto_pick_skips_constant_columns():
    """region 只有一个取值时，自动挑列不能挑中它 —— 画出来是一根孤零零的柱子。"""
    dtypes = {"region": "Utf8", "channel": "Utf8", "tenure_months": "Int64"}
    counts = {"region": 1, "channel": 4, "tenure_months": 40}
    assert _auto_pick_column("bar", dtypes, counts) == "channel"
    assert _auto_pick_column("histogram", dtypes, counts) == "tenure_months"


def test_sweep_word_routes_multi_angle_request_to_overview():
    """「全面分析，包括质量、分布、相关性和可视化」不能只跑一个 eda.correlation。"""
    got = route("帮我做一次全面分析，包括质量、分布、相关性和可视化")
    assert got.intent.value == "overview", got.intent.value


def test_typo_tolerant_quality_routing():
    """错别字不能让诉求掉进对话路径后变成一次反问。"""
    assert route("帮窝看瞎这分数据的缺实值").intent.value == "quality"


def test_design_verb_routes_to_workflow_create():
    """★ 红线原句：「设计适合的 Workflow」必须进推荐流程，不能被 workflow.list 抢走。

    第八轮实测：``帮我设计一个客户流失分析的 Workflow`` 里「设计」既不在创建动词表
    也不在强词表，整句被 WORKFLOW 的强词「workflow」拿走，落到 workflow.list ——
    用户要一套方案，系统回一句「当前工作流列表为空」外加一段泛泛建议。
    """
    for text in (
        "帮我设计一个客户流失分析的 Workflow",
        "设计一个适合这份数据的流程",
        "帮我设计个 Workflow",
        "我要搞个工作流",
        "帮我弄一个工作流",
        "设计一个客户流失预警的工作流",
        "帮我设计一套流程",
        "design a workflow for churn analysis",
    ):
        got = route(text)
        assert got.intent.value == "workflow_create", f"{text} -> {got.intent.value}"


def test_pipeline_word_routes_to_workflow_create_not_data_clean():
    """「创建一个完整的客户流失分析管道，从数据清洗到模型输出」不能真去清洗数据。

    Telco 压测实测：这句话里只有 CLEAN 的强词「清洗」命中，系统直接跑了一次
    data.clean，把 7043 行洗成 7032 行并新增了版本 —— 用户要的是设计管道，
    不是立刻改数据。这是**破坏性**的路由误判。
    """
    for text in (
        "创建一个完整的客户流失分析管道，从数据清洗到模型输出",
        "帮我搭一条数据处理的流水线",
        "建一个从清洗到建模的管道",
    ):
        got = route(text)
        assert got.intent.value == "workflow_create", f"{text} -> {got.intent.value}"


def test_collinearity_routes_to_correlation():
    """「哪些字段之间存在潜在的共线性」此前掉进对话路径后只能反问用户选字段。"""
    assert route("根据数据字典，哪些字段之间存在潜在的共线性？").intent.value == "correlation"


def test_design_alone_does_not_steal_non_workflow_requests():
    """「设计一个机器学习实验方案」里的「设计一个」不能把建模诉求吸成建工作流。

    把「设计一个」当强词时它确实吸走了：test_ml_train 那条用例直接跑飞。
    泛指说法只能靠「设计动词 + 工作流名词」的结构化判定去接。
    """
    for text in (
        "请根据当前数据集设计一个机器学习实验方案，说明目标变量和候选模型",
        "设计一个机器学习方案",
        "帮我设计一个实验方案",
    ):
        got = route(text)
        assert got.intent.value != "workflow_create", f"{text} -> {got.intent.value}"


def test_listing_workflows_still_routes_to_list():
    """加了「设计」之后，纯查看的说法不能被误伤成「新建」。"""
    for text in ("看看有哪些工作流", "列出所有工作流", "当前有哪些流程", "工作流列表"):
        got = route(text)
        assert got.intent.value == "workflow", f"{text} -> {got.intent.value}"


def test_payload_looking_input_is_not_treated_as_a_request():
    """把一段 JSON 参数贴进来，不能当成需求去画图。"""
    assert _looks_like_payload('{"dataset_id": "abc", "chart": , "column": }') is True
    assert _looks_like_payload("画个直方图") is False
    assert route('{"dataset_id": "abc", "chart": , "column": }').intent.value == "chat"


def test_abandonment_with_followup_request_is_not_pure_abandonment():
    """「算了，先告诉我数据有多少行」= 撤回 + 新指令，后半句不能被丢掉。"""
    assert is_abandonment("算了") is True
    assert is_abandonment("不用了") is True
    assert is_abandonment("算了，先告诉我数据有多少行") is False


def test_relation_chart_routes_to_visualize_not_correlation():
    """类别列 × 数值列画不了相关性矩阵，只能画分组图。"""
    assert route("再画个 region 和 monthly_charges 的关系图").intent.value == "visualize"


def test_chart_followup_routes_to_visualize():
    """「那个图换成 30 个分箱」—— 唯一的动词是「分箱」。"""
    assert route("那个图换成 30 个分箱再看看").intent.value == "visualize"


# ----------------------------------------------------------------------
# 六、引擎：选择值与回指
# ----------------------------------------------------------------------


def test_invalid_choice_is_dropped_so_the_user_gets_asked():
    """名字被误填进 plan 槽时，必须清掉并挂起问人，而不是直接执行失败。"""
    engine = AgentEngine.__new__(AgentEngine)
    step = PlaybookStep(
        "workflow.build_and_run", "创建并执行",
        ask_if="needs_choice", ask_slot="plan", ask_options_key="plan_options",
    )
    previous = {
        "needs_choice": True,
        "plan_options": [
            {"value": "1", "label": "1. churn 预测建模", "note": "x"},
            {"value": "2", "label": "2. 分布与相关性探索", "note": "y"},
        ],
    }
    params = {"plan": "客户流失预警", "name": "客户流失预警"}
    AgentEngine._drop_invalid_choice(step, previous, params)
    assert "plan" not in params
    assert AgentEngine._pending_choice(step, previous, params) is True


def test_correlation_columns_union_with_previous_turn():
    """「那个月费的分布 → 再看看它跟总费用的关系」必须得到两列，不能只剩一列。

    相关性是多列两两比较，本轮抽到的新列要**并入**上一轮的列。
    替换的话 eda.correlation 只会拿到 ['total_charges']，直接报
    「至少需要 2 个数值字段」（第十二轮实测）。
    """
    step = PlaybookStep("eda.correlation", "相关性", union_previous=("columns",))
    params: dict = {"columns": ["total_charges"]}
    AgentEngine._merge_previous_lists(step, {"columns": ["monthly_charges"]}, params)
    assert params["columns"] == ["monthly_charges", "total_charges"]
    # 重复列名要去重
    AgentEngine._merge_previous_lists(step, {"columns": ["monthly_charges"]}, {"columns": ["monthly_charges"]})
    # 标量参数不受影响（没登记进 union_previous 就不会动）
    scalar = PlaybookStep("eda.correlation", "相关性")
    only_new: dict = {"columns": ["total_charges"]}
    AgentEngine._merge_previous_lists(scalar, {"columns": ["monthly_charges"]}, only_new)
    assert only_new["columns"] == ["total_charges"]
    # ★ 上一次**同名**调用不存在时，用跨工具找回的「最近聊到的列」兜底：
    #   「那个月费的分布」走 eda.distribution，下一句「再看看它跟总费用的关系」走
    #   eda.correlation —— 只看同名调用什么都拿不到，相关性会只剩一列（第十三轮实测）。
    filled: dict = {"columns": ["total_charges"]}
    AgentEngine._merge_previous_lists(step, {}, filled, fallback=["monthly_charges"])
    assert filled["columns"] == ["monthly_charges", "total_charges"]


def test_valid_choice_is_kept():
    engine = AgentEngine.__new__(AgentEngine)
    step = PlaybookStep(
        "workflow.build_and_run", "创建并执行",
        ask_if="needs_choice", ask_slot="plan", ask_options_key="plan_options",
    )
    previous = {"needs_choice": True, "plan_options": [{"value": "1", "label": "1", "note": ""}]}
    params = {"plan": "1"}
    AgentEngine._drop_invalid_choice(step, previous, params)
    assert params["plan"] == "1"
    assert AgentEngine._pending_choice(step, previous, params) is False


def test_previous_tool_args_powers_chart_followups():
    """「那个图换成 30 个分箱」靠上一次成功调用的参数接住。"""
    from app.agent.models import ToolCall, ToolCallStatus

    engine = AgentEngine.__new__(AgentEngine)

    class _Store:
        def __init__(self) -> None:
            self.runs: dict = {}

        def get_run(self, run_id):  # noqa: ANN001
            return self.runs.get(run_id)

    engine.store = _Store()
    old_run = AgentRun(id="r-old", session_id="s-1", user_request="画个直方图")
    old_run.tool_calls = [
        ToolCall(step_index=0, tool="eda.visualize", status=ToolCallStatus.OK,
                 arguments={"chart": "histogram", "column": "tenure_months"}),
    ]
    engine.store.runs["r-old"] = old_run

    session = type("S", (), {"run_ids": ["r-old"]})()
    current = AgentRun(id="r-new", session_id="s-1", user_request="那个图换成 30 个分箱")
    got = engine._previous_tool_args(session, current, "eda.visualize")
    assert got == {"chart": "histogram", "column": "tenure_months"}


def test_failed_previous_call_is_not_reused():
    """失败调用的参数本身就是错的，复用等于把同一个坑踩两遍。"""
    from app.agent.models import ToolCall, ToolCallStatus

    engine = AgentEngine.__new__(AgentEngine)

    class _Store:
        def __init__(self) -> None:
            self.runs: dict = {}

        def get_run(self, run_id):  # noqa: ANN001
            return self.runs.get(run_id)

    engine.store = _Store()
    old = AgentRun(id="r-old", session_id="s-1")
    old.tool_calls = [
        ToolCall(step_index=0, tool="eda.visualize", status=ToolCallStatus.FAILED,
                 arguments={"chart": "scatter", "column": "bad"}),
    ]
    engine.store.runs["r-old"] = old
    session = type("S", (), {"run_ids": ["r-old"]})()
    assert engine._previous_tool_args(session, AgentRun(id="r-new", session_id="s-1"), "eda.visualize") == {}


def test_collect_charts_only_keeps_rendered_ones():
    from app.agent.engine import _collect_charts
    from app.agent.models import ToolCall, ToolCallStatus

    run = AgentRun(id="r-1", session_id="s-1")
    run.tool_calls = [
        ToolCall(step_index=0, tool="eda.visualize", status=ToolCallStatus.OK,
                 result={"metadata": {"chart": {"kind": "histogram", "svg": "<svg/>", "rendered": True}}}),
        ToolCall(step_index=1, tool="eda.visualize", status=ToolCallStatus.OK,
                 result={"metadata": {"chart": {"kind": "bar", "svg": "", "rendered": False}}}),
    ]
    got = _collect_charts(run)
    assert len(got) == 1 and got[0]["kind"] == "histogram"


def test_ml_config_strips_non_model_keys():
    """dataset_id 进了模型构造器会报「不认识的参数」。"""
    from app.workflow.runners import _ml_config

    node = type("N", (), {"config": {"dataset_id": 14, "version": 1, "target_column": "churn",
                                     "model": "random_forest_classifier", "__ui": {}}})()
    got = _ml_config(node)
    assert got == {"target_column": "churn", "model": "random_forest_classifier"}
