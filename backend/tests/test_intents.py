"""意图路由：确定性、零 Token、无网络。

旧架构在这里有六层决策（确定性长链 / 本地神经路由 / 单步直连 / 短链兜底 /
Remote 结构化升级 / 纯对话），一次决策要发几千 Token。新架构是一张表 + 一次
打分。这个文件的价值就是**把「表」钉死**：说法变了要显式改表，不能靠模型兜。
"""

from __future__ import annotations

import pytest

from app.agent.intents import MIN_SCORE, Intent, route, strip_result_reference

#: (期望意图, 用户原话)。原话取自真实提问习惯，不是从关键词表倒推出来的。
CASES: list[tuple[str, str]] = [
    ("list_datasets", "有哪些数据集"),
    ("list_datasets", "列出所有数据集"),
    ("inspect_dataset", "这个数据集的基本信息"),
    ("inspect_dataset", "看下数据集详情"),
    ("preview_dataset", "预览前10行"),
    ("preview_dataset", "给我看几行数据"),
    ("schema", "这个表有哪些列"),
    ("schema", "字段名是什么"),
    ("profile", "做一份数据画像"),
    ("profile", "看看整体概览"),
    ("quality", "看看这个数据集的质量"),
    ("quality", "有多少缺失值"),
    ("quality", "数据质量怎么样"),
    # 四个角度一起提：缺失 / 重复 / 异常值 / 字段可用性。
    # 表里曾只有「缺失值」「重复值」，裸词「缺失、重复」不命中，
    # 于是 3 分的 OUTLIER 赢过 1 分的 QUALITY，四个角度只跑了异常值一个。
    ("quality", "从缺失、重复、异常值、字段可用性四个角度评估这份数据"),
    ("quality", "这份数据的完整度和一致性怎么样"),
    ("overview", "全面分析一下这份数据"),
    ("overview", "帮我分析"),
    ("describe", "描述统计"),
    ("describe", "做个统计摘要"),
    ("distribution", "分析 age 列的分布"),
    ("distribution", "看下取值分布"),
    ("visualize", "画个直方图"),
    ("correlation", "计算各字段的相关性"),
    ("correlation", "相关性矩阵"),
    # 「看看 duration 和 credit_amount 的关系」没有「相关性」三个字，
    # 此前只能掉进对话路径，Agent 回一段「请确认用哪种方式」的空话。
    ("correlation", "看看 duration 和 credit_amount 的关系"),
    ("correlation", "这两列有什么关系"),
    ("outlier", "有没有异常值"),
    ("outlier", "检测离群点"),
    ("visualize", "画个图看看"),
    ("visualize", "做个可视化"),
    ("filter", "筛选 age > 30 的数据"),
    ("filter", "只保留北京的记录"),
    ("clean", "清洗一下数据"),
    ("clean", "去重"),
    # 动词在名词**后面**的写法：「把缺失值填充掉」里 CLEAN 的「填充缺失」匹配不上，
    # 而 QUALITY 的「缺失值」命中 ⇒ 用户要改数据，拿到一份体检报告，数据一行没变。
    ("clean", "把这份数据里的缺失值填充掉"),
    ("clean", "缺失值补一下"),
    ("clean", "把重复值删除掉"),
    ("inspect_dataset", "这份数据有多少行多少列"),
    ("inspect_dataset", "这个数据集有多少条记录"),
    ("inspect_dataset", "数据量多大"),
    # 「筛选后还剩多少行」里的「筛选」是回指上一轮的时间状语，不是本轮诉求。
    # 不剥离就会被 FILTER 抢走，然后反问「筛选条件是什么」—— 一步工具都没跑。
    ("inspect_dataset", "筛选后还剩多少行"),
    ("inspect_dataset", "清洗后有多少行"),
    ("distribution", "看看筛选后 credit_amount 的分布"),
    # 反过来：真要筛选时不能因为这条规则被误剥离
    ("filter", "筛选出 age 大于 50 的数据"),
    ("transform", "新增一列"),
    ("transform", "加一个派生列"),
    ("aggregate", "按城市分组统计"),
    ("aggregate", "求平均"),
    # 「按 X 分组求 Y 的总和」：旧表只有「分组聚合」「分组统计」这类紧邻写法，
    # 真实说法中间隔着「求 credit_amount 的」，于是一句明确的聚合诉求掉进对话
    # 路径 —— Agent 回一段「请确认用哪个版本」的空话，一个数都没算。
    ("aggregate", "按 job 分组求 credit_amount 的总和"),
    ("aggregate", "分组求总和"),
    ("aggregate", "按 purpose 分组求 credit_amount 的平均值"),
    ("merge", "合并两个数据集"),
    ("merge", "把这两个数据集按 customer_id 合并"),
    # 「两个数据集」是主语不是诉求：它若算强词，会压过「合并」——
    # 实测用户说要合并，系统只跑了一次关系探索，还在结尾提醒他「如需合并请说一句…」。
    ("merge", "按 customer_id 合并这两个数据集"),
    ("relation", "看看这两个数据集有什么联系"),
    ("merge", "join 一下"),
    ("ml_train", "训练一个预测模型"),
    ("ml_train", "建模"),
    # 点名目标列 / 算法名同样是明确的建模诉求。
    # 旧表只有「预测模型」这种组合词命中，于是「预测 survived」「用随机森林」
    # 全部掉进对话路径，用户拿到一句「我没有理解你想做的具体分析」。
    ("ml_train", "预测 survived"),
    ("ml_train", "用随机森林训练"),
    ("ml_train", "做一次聚类"),
    ("ml_evaluate", "评估模型"),
    ("ml_compare", "对比模型"),
    ("ml_explain", "特征重要性"),
    ("report", "生成一份分析报告"),
    # 「生成一份数据质量报告」：REPORT 与 QUALITY 都命中强词，但用户要的是
    # **一份报告文件**，不是再看一遍体检结论 —— 此前被 QUALITY 抢走，
    # 跑的是 dataset.quality，report.generate 永远调不到。
    ("report", "生成一份数据质量报告"),
    ("report", "导出一份数据质量报告"),
    ("report", "要一份数据报告"),
    # 反过来：只说体检、没说报告时仍然要走质量检查
    ("quality", "检查一下这份数据的质量"),
    ("quality", "缺失值多吗"),
    ("workflow", "跑个工作流"),
    ("workflow", "看看有哪些工作流"),
    # 「创建工作流」里往往夹着别的具体动作（描述性统计 / 训练模型）。
    # 这些从属步骤的分数更高，主意图被抢走过 —— 用户要建流程，系统只跑了一次统计。
    ("workflow_create", "创建工作流：读取数据集 → 生成描述性统计 → 生成报告"),
    ("workflow_create", "帮我新建一个工作流，先清洗再训练模型"),
    ("workflow_create", "搭个工作流把这串步骤固化下来"),
    ("connector", "配置连接器"),
]


@pytest.mark.parametrize("expected,text", CASES, ids=[f"{e}:{t}" for e, t in CASES])
def test_route_hits_expected_intent(expected: str, text: str):
    assert route(text).intent.value == expected


@pytest.mark.parametrize("text", ["你好", "今天天气怎么样", "", "   "])
def test_chat_fallback(text: str):
    result = route(text)
    assert result.intent is Intent.CHAT
    assert result.is_chat


def test_route_is_pure_function():
    """同一输入必须永远同一输出：路由不能有随机性，否则 trace 不可复现。"""
    first = route("计算各字段的相关性")
    for _ in range(20):
        again = route("计算各字段的相关性")
        assert (again.intent, again.score, again.reason) == (first.intent, first.score, first.reason)


def test_specific_beats_generic():
    """「分析一下相关性」不能被 OVERVIEW 抢走 —— 具体意图优先级更高。"""
    assert route("分析一下相关性").intent is Intent.CORRELATION


def test_score_never_below_threshold_when_not_chat():
    for expected, text in CASES:
        result = route(text)
        assert result.score >= MIN_SCORE, text


def test_row_limit_slot():
    assert route("预览前10行").slots.get("limit") == 10
    assert route("预览数据").slots.get("limit") is None


def test_reason_is_human_readable():
    """reason 会直接显示在前端 route 事件里，不能是裸数据结构。"""
    reason = route("有多少缺失值").reason
    assert isinstance(reason, str) and reason
    assert "缺失值" in reason


@pytest.mark.parametrize(
    "text,kept",
    [
        ("筛选后还剩多少行", "还剩多少行"),
        ("清洗后有多少行", "有多少行"),
        ("合并后 credit_amount 的分布", "credit_amount 的分布"),
        ("训练完了效果怎么样", "效果怎么样"),
    ],
)
def test_strip_result_reference_keeps_the_actual_question(text: str, kept: str):
    """「动作 + 后」是回指状语，剥掉之后剩下的才是本轮真正要问的。"""
    stripped = strip_result_reference(text)
    assert "筛选" not in stripped or "筛选" in kept
    assert kept.strip() in stripped


def test_strip_result_reference_does_not_touch_a_real_request():
    """「筛选出 age>50 的数据」是真请求，中间的宾语不能被剥掉。"""
    assert strip_result_reference("筛选出 age 大于 50 的数据") == "筛选出 age 大于 50 的数据"
