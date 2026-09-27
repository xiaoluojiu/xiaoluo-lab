"""意图 → 固定工具链。

为什么是固定链，而不是让模型编排
--------------------------------
数据分析的工具链是**有限且稳定的**：「看看质量」就是 ``dataset.quality``，
「训练模型」就是 ``detect_task → train → evaluate``。让模型每次重新编排，
付出 Token 换来的只是不确定性 —— 而且一旦编排错，失败后的重规划会再烧一轮。

固定链的代价是「表里没有的诉求答不了」。这是可接受的取舍：
答不了会**明确说暂不支持**，而不是硬凑一条跑不通的链把用户的额度烧光。

槽位的三种来源（优先级从高到低）
--------------------------------
1. **自动**：``dataset_id`` 取会话绑定的数据集，不需要用户说
2. **抽取**：规则从文本里读（如「前 10 行」）；规则读不出时，若配置了 LLM 则由 LLM 抽一次
3. **澄清**：以上都拿不到且该槽位必填 → 挂起等用户回答，不猜

``fallback_tool`` 用于「缺槽位时不问用户，改用一个更粗粒度的工具」：
用户说「看看分布」却没说哪一列，直接给全表分布概览，比反问「请问要看哪一列」体验好。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.intents import Intent


@dataclass(frozen=True)
class PlaybookStep:
    """工具链中的一步。"""

    tool: str
    title: str
    #: 固定默认值（会被引擎的自动槽位与抽取结果覆盖）
    defaults: dict[str, Any] = field(default_factory=dict)
    #: 必填槽位（工具参数名）。缺失时先看有无 fallback_tool，都没有才向用户澄清。
    required_slots: tuple[str, ...] = ()
    #: **可微调槽位**：用户在请求里点名了就按他说的来，没点名就用 :attr:`defaults`，
    #: 抽不到也**绝不反问**。
    #:
    #: 为什么必填槽位之外还要单独一类：``model`` / ``params`` 这种参数
    #: 从来不该反问用户（「请问用什么模型」是废话，默认 auto 就够了），
    #: 但用户真的说了「用随机森林、200 棵树」时必须照办。
    #: 只挂在 required_slots 上会导致要么一直问、要么永远拿默认值 ——
    #: 这正是「Agent 只会调工具、不会按诉求调参数」的根因。
    tuned_slots: tuple[str, ...] = ()
    #: 从上一步结果的 ``data`` 里取值：{工具参数名: 上一步 data 的键名}
    carry: dict[str, str] = field(default_factory=dict)
    #: 必填槽位补齐失败时改用这个工具（它需要的参数更少）
    fallback_tool: str = ""
    #: 失败时是否可跳过继续下一步。False ⇒ 整条链终止（默认保守）
    skippable: bool = False
    #: 上一步结果里该键为真时，先挂起问用户再执行本步（见 ``ask_slot``）。
    #:
    #: 用来表达「工具跑成功了，但结论里有一个必须人来定的选择」。
    #: 典型场景：``ml.detect_task`` 推断不出目标列 —— 它返回 success + 候选列，
    #: 但「到底预测哪一列」只有用户知道。没有这个机制时，链路只能自己挑一个
    #: 默认值往下跑（实测就是默认按聚类开训，产出一份没人要的结果）。
    ask_if: str = ""
    #: :attr:`ask_if` 触发时反问的槽位名（同时也是工具参数名）
    ask_slot: str = ""
    #: :attr:`ask_if` 触发时，从上一步 ``data`` 的哪个键取候选选项（列表或字典）
    ask_options_key: str = ""
    #: 开了 :attr:`reuse_previous_args` 时，这些**多值参数**与上一轮的值取并集
    #: 而不是替换。只用于「本来就是一堆列一起算」的参数（相关性的 columns）；
    #: 标量参数（bins / chart）替换才是对的。
    union_previous: tuple[str, ...] = ()
    #: 本步参数缺失时，复用**本会话上一次同名工具调用**的参数。
    #:
    #: 用来接住「那个图换成 30 个分箱」「再画一遍」这类**回指**。
    #: 用户说的是上一轮那张图，链路上却把它当成一次全新的请求：
    #: 没有列名、没有图表类型 ⇒ 退化成反问「请确认是否用当前数据集」，
    #: 用户要的图重画一次都没发生。
    reuse_previous_args: bool = False


@dataclass(frozen=True)
class Playbook:
    """一个意图对应的完整工具链。"""

    intent: Intent
    title: str
    steps: tuple[PlaybookStep, ...] = ()
    #: 无工具（纯对话）
    chat_only: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)


def _p(intent: Intent, title: str, *steps: PlaybookStep) -> Playbook:
    return Playbook(intent=intent, title=title, steps=steps)


PLAYBOOKS: dict[Intent, Playbook] = {
    # ---------------- 数据集 ----------------
    Intent.LIST_DATASETS: _p(
        Intent.LIST_DATASETS, "列出数据集",
        PlaybookStep("dataset.list", "列出可用数据集", defaults={"page_size": 50}),
    ),
    Intent.INSPECT_DATASET: _p(
        Intent.INSPECT_DATASET, "查看数据集概况",
        PlaybookStep("dataset.inspect", "读取数据集基本信息"),
    ),
    Intent.PREVIEW_DATASET: _p(
        Intent.PREVIEW_DATASET, "预览数据",
        PlaybookStep("dataset.preview", "预览数据行", defaults={"page_size": 10}),
    ),
    Intent.SCHEMA: _p(
        Intent.SCHEMA, "查看字段结构",
        PlaybookStep("dataset.schema", "读取字段与类型"),
    ),
    Intent.PROFILE: _p(
        Intent.PROFILE, "生成数据画像",
        PlaybookStep("dataset.profile", "计算列级统计画像"),
    ),
    Intent.QUALITY: _p(
        Intent.QUALITY, "检查数据质量",
        PlaybookStep("dataset.quality", "检查缺失/重复/异常"),
    ),
    Intent.RELATION: _p(
        Intent.RELATION, "探索两个数据集的联系",
        # 概况只是背景，拿不到也不该让整条链停下来 —— 关系分析本身才是用户要的
        PlaybookStep("dataset.inspect", "读取主数据集信息", skippable=True),
        # 只读的关系探索：共同字段、重叠度、可关联键建议。
        # 刻意不做 data.merge —— 那是高风险写操作，且用户只说了「探索联系」。
        PlaybookStep("dataset.relation", "分析两个数据集的共同字段与可关联键"),
        # 只识别「能做哪种机器学习」并给出候选目标列，不开训：
        # 没有目标列的建模一定是猜，这里把选择权交回用户。
        PlaybookStep(
            "ml.detect_task", "判断可做哪种机器学习分析",
            defaults={"infer_target": True},
            skippable=True,
        ),
    ),
    Intent.OVERVIEW: _p(
        Intent.OVERVIEW, "综合数据分析",
        PlaybookStep("dataset.profile", "计算列级统计画像"),
        PlaybookStep("dataset.quality", "检查数据质量"),
        PlaybookStep("eda.describe", "生成描述性统计"),
        # ★ 相关性是全表分析里最常被点名的第四项。少了它，用户说
        #   「包括质量、分布、相关性和可视化」时答案只能自己承认
        #   「相关性没做」（第十轮实测）。没有数值列会失败 ⇒ 必须可跳过。
        PlaybookStep("eda.correlation", "计算相关性矩阵", skippable=True),
        # ★ 同理补一张图。「全面分析，包括…和可视化」里可视化是点名要的，
        #   却一张图都没有 —— 答案只能写「可视化结果未包含在内」（第十轮实测）。
        #   列由工具按类型自动挑，挑不出就跳过，绝不反问用户。
        PlaybookStep("eda.visualize", "生成分布图", defaults={"chart": "histogram"}, skippable=True),
    ),
    # ---------------- EDA ----------------
    Intent.DESCRIBE: _p(
        Intent.DESCRIBE, "描述性统计",
        # columns 是可微调槽位：用户点名了哪几列就只统计哪几列，没点名就全表。
        # 不放进 required_slots —— 「请问要统计哪几列」是废话，全表才是合理默认。
        PlaybookStep("eda.describe", "计算描述性统计", tuned_slots=("columns",)),
    ),
    Intent.DISTRIBUTION: _p(
        Intent.DISTRIBUTION, "分布分析",
        PlaybookStep(
            "eda.distribution", "分析指定列的分布",
            required_slots=("column",),
            # 没说哪一列 → 给全表分布概览，而不是反问
            fallback_tool="eda.distribution_overview",
        ),
    ),
    Intent.CORRELATION: _p(
        Intent.CORRELATION, "相关性分析",
        # 同理：点名「customer_id 和 age」就只算这一对（哪怕 customer_id 是 id 列，
        # 用户要看就得给他看）；没点名才由工具自动挑连续数值列。
        PlaybookStep(
            "eda.correlation", "计算相关性矩阵",
            tuned_slots=("columns", "method"),
            # ★ 「刚才那两个参数，再帮我跑一次，这次用 spearman」里的列来自上一轮。
            #   不开这个开关，用户点名的两列拿不到，他只说得出「那两个参数」。
            reuse_previous_args=True,
            # ★ columns 取并集：「那个月费的分布 → 再看看它跟总费用的关系」里
            #   本轮只抽得到 total_charges，替换掉上一轮的 monthly_charges 就只剩
            #   一列，工具直接报「至少需要 2 个数值字段」（第十二轮实测）。
            union_previous=("columns",),
        ),
        # ★ 「生成所有数值字段的相关性热力图」此前只给一串相关系数，末尾还要用户
        #   再说一遍「画热力图」（Telco 压测实测）—— 用户已经点名要图了。
        #   没有数值列会失败 ⇒ 必须可跳过，不能让整条相关性链挂在这里。
        PlaybookStep(
            "eda.visualize", "生成相关性热力图",
            defaults={"chart": "heatmap"},
            skippable=True,
        ),
    ),
    Intent.OUTLIER: _p(
        Intent.OUTLIER, "异常值检测",
        PlaybookStep("eda.outlier", "检测异常值", tuned_slots=("columns",)),
    ),
    Intent.VISUALIZE: _p(
        Intent.VISUALIZE, "生成图表",
        PlaybookStep(
            "eda.visualize", "生成可视化图表",
            defaults={"chart": "histogram"},
            # ★ column **刻意不是必填槽位**。「画个直方图看看」里没有列名，
            #   挂起反问「要对哪一列做这个分析？」等于把本该由系统承担的推理
            #   推回给用户 —— 实测连问三轮，一张图都没出（第十轮）。
            #   正确做法是让工具按列类型自己挑一列画出来，并在回答里说明
            #   「已按 X 绘制，要看其他列直接说」，用户改口比填空轻松得多。
            required_slots=("chart",),
            # ★ chart 必须同时是**可微调槽位**：用户点名「画成柱状图」时，
            #   defaults 的 histogram 不该赢 —— 实测「把刚才的结果画成柱状图」
            #   画出来的是直方图，接着又因列不是数值列而整次失败。
            # x / y 是双列图表（散点、折线、分组柱状图）的必备项：
            # 不从文本里抽，用户点名的两列一个都用不上。
            tuned_slots=("chart", "column", "bins", "x", "y"),
            # ★ 「那个图换成 30 个分箱」里没有图表名也没有列名 ——
            #   它们说的是**上一轮那张图**。不开这个开关，系统只能反问
            #   「请确认是否用当前数据集」，等于什么都没做。
            reuse_previous_args=True,
        ),
    ),
    # ---------------- 数据处理 ----------------
    Intent.FILTER: _p(
        Intent.FILTER, "筛选数据",
        PlaybookStep("data.filter", "按条件筛选行", required_slots=("conditions",)),
    ),
    Intent.CLEAN: _p(
        Intent.CLEAN, "清洗数据",
        # data.clean 是 HIGH 风险，执行前必须经用户确认（由权限层拦截）
        PlaybookStep("data.clean", "处理缺失值与重复行", defaults={"missing": {"strategy": "drop"}}),
    ),
    Intent.TRANSFORM: _p(
        Intent.TRANSFORM, "派生新列",
        PlaybookStep("data.transform", "新增派生列", required_slots=("name", "expression")),
    ),
    Intent.AGGREGATE: _p(
        Intent.AGGREGATE, "分组聚合",
        # ★ aggregations 不再是必填：「按 Contract 分组统计」此前挂起反问
        #   「要对哪些列做什么聚合？」且**一个候选都不给**，用户改口后再问一遍
        #   （Telco 压测实测）。没点名聚合方式时按分组计数走，这比反问有用得多。
        PlaybookStep(
            "data.aggregate", "分组聚合统计",
            required_slots=("group_by",),
            tuned_slots=("aggregations",),
            reuse_previous_args=True,
            # ★ 没说「按什么分组」时不能反问「按哪些列分组？」。
            #   实测「统计 tenure 均值」「报告每折分数和均值」「最后汇总对比」
            #   都被这句反问拦下 —— 它们压根不是分组诉求，只是撞上了「均值」
            #   这个 AGGREGATE 强词。退化成全表描述统计，比反问有用得多。
            fallback_tool="eda.describe",
        ),
    ),
    Intent.MERGE: _p(
        Intent.MERGE, "合并数据集",
        # 左表取会话数据集；右表必须由用户指出，猜错代价太大
        PlaybookStep("data.merge", "合并两个数据集", required_slots=("right_dataset_id",)),
    ),
    # ---------------- 建模 ----------------
    Intent.ML_TRAIN: _p(
        Intent.ML_TRAIN, "训练模型",
        PlaybookStep(
            "ml.detect_task", "识别任务类型与目标列",
            defaults={"infer_target": True},
            # ★ 用户点名的目标列必须传到 detect：任务类型（回归/分类）由它决定。
            #   不传的话，detect 会按自己的推断给出一个完全不同的目标列，
            #   于是「用 tenure 做目标列跑回归」被判成 classification。
            tuned_slots=("target",),
        ),
        PlaybookStep(
            "ml.train", "训练模型",
            defaults={"model": "auto"},
            # ★ 用户点名了模型 / 超参 / 测试集比例就必须照办。没有 tuned_slots 时，
            # 「用随机森林训练，200 棵树」也会被 defaults 的 model="auto" 吃掉，
            # 永远跑成 logistic_regression —— 看起来是「Agent 不会调参数」，
            # 实际上是引擎压根没给它一个能填参数的位置。
            # ★ target 必须是可微调槽位：用户点名的目标列要能盖过 detect 的推断值
            #   （「把 tenure 作为目标列」此前完全不生效，Telco 压测实测）。
            tuned_slots=("model", "params", "test_size", "target"),
            # 刻意**不** carry task：目标列若由用户后补，task 必须重新判定。
            # 带着 detect 阶段的 clustering 去训一个用户新选的数值目标列，
            # 会把回归问题硬跑成聚类。
            carry={"target": "target"},
            # 推断不出目标列时挂起问用户，绝不默认按聚类开训
            ask_if="needs_target",
            ask_slot="target",
            ask_options_key="target_candidates",
        ),
        PlaybookStep(
            "ml.evaluate", "评估模型指标",
            carry={"run_id": "run_id"},
            skippable=True,
        ),
    ),
    Intent.ML_EVALUATE: _p(
        Intent.ML_EVALUATE, "评估模型",
        # run_id 不再必填：用户说「评估最近一次」时由工具从最近实验自动推断
        PlaybookStep("ml.evaluate", "读取模型指标"),
    ),
    Intent.ML_COMPARE: _p(
        Intent.ML_COMPARE, "对比模型",
        PlaybookStep("ml.compare", "对比多次运行的指标"),
    ),
    Intent.ML_EXPLAIN: _p(
        Intent.ML_EXPLAIN, "解释模型",
        PlaybookStep("ml.explain", "输出特征重要性"),
    ),
    # ---------------- 编排与产出 ----------------
    Intent.REPORT: _p(
        Intent.REPORT, "生成分析报告",
        PlaybookStep("report.generate", "生成中文分析报告"),
    ),
    Intent.WORKFLOW: _p(
        Intent.WORKFLOW, "查看工作流",
        PlaybookStep("workflow.list", "列出工作流"),
    ),
    # ★ 两代缺陷，同一类根因：**把设计工作推回给用户**。
    #  1. 最早只挂 workflow.list —— 用户说「创建工作流」时系统只会列出现有流程，
    #     而 workflow.create / build_and_run 根本没被任何 playbook 引用（写了调不到）。
    #  2. 补上创建链路后，又因为没有 nodes/edges 而反问「流程内容：先做哪一步？」
    #     用户明明给了数据集，系统却要他手写步骤 —— 这就是「挤牙膏」。
    #
    # 现在：先由 workflow.recommend 读真实字段（列名/类型/唯一值数/行数）
    # 主动设计 1~3 个可跑的方案，再用 ask_* 挂起让用户回一个数字即可。
    # 推荐是 dataset_id 的纯函数，用户选完之后 build_and_run 用同一份输入还原方案，
    # 不需要跨步骤传递状态。
    Intent.WORKFLOW_CREATE: _p(
        Intent.WORKFLOW_CREATE, "设计并执行工作流",
        # 先读一下数据集概况：后续节点要用到 dataset_id，也让答案有背景
        PlaybookStep("dataset.inspect", "读取数据集信息", skippable=True),
        PlaybookStep(
            "workflow.recommend", "基于数据字段自动设计候选方案",
            # 用户的原始诉求参与方案排序（说「流失预警」就把分类建模排第一）。
            tuned_slots=("goal",),
        ),
        PlaybookStep(
            "workflow.build_and_run", "创建并执行选中的工作流",
            # recommend 总是返回 needs_choice=True：方案是系统给的，
            # 但「跑哪一个」必须人来定 —— 这一步挂起问人，不自己挑。
            ask_if="needs_choice",
            ask_slot="plan",
            ask_options_key="plan_options",
            # ★ name **刻意不是必填**：用户说「设计一个挽留 Workflow」时并没起名，
            #   反问「给这个工作流起个名字？」就是在挤牙膏（100 条多轮压测实测）。
            #   工具层会用方案名兜底（`_is_placeholder_name`）。
            required_slots=(),
            tuned_slots=("plan", "dataset_id"),
            # ★ goal 必须从上一步**原样带过来**：方案排序由它决定，
            #   若这里重新抽一次得到不同的 goal，方案会重排，用户选的「1」
            #   就可能指向另一个方案 —— 那是「跑通了但跑错了」，比失败更危险。
            carry={"goal": "goal"},
            skippable=True,
        ),
    ),
    Intent.CONNECTOR: _p(
        Intent.CONNECTOR, "查看外部数据源",
        PlaybookStep("connector.list", "列出数据库连接器"),
    ),
    # ---------------- 兜底 ----------------
    Intent.CHAT: Playbook(intent=Intent.CHAT, title="对话", chat_only=True),
}


def get_playbook(intent: Intent) -> Playbook:
    """取意图对应的工具链；未登记时按对话处理（绝不返回 None 让调用方猜）。"""
    return PLAYBOOKS.get(intent) or PLAYBOOKS[Intent.CHAT]


#: 需要「用户明确指出」而无法自动补齐的槽位说明（供澄清文案使用）。
SLOT_QUESTIONS: dict[str, str] = {
    "column": "要对哪一列做这个分析？",
    "chart": "想看什么类型的图？（如直方图 / 散点图 / 箱线图）",
    "conditions": "筛选条件是什么？（如 age > 30）",
    "name": "新列的名字叫什么？",
    "expression": "新列的计算表达式是什么？（如 price * quantity）",
    "group_by": "按哪些列分组？",
    "aggregations": "要对哪些列做什么聚合？（如 求和 / 平均值 / 计数）",
    # 「合并」与「探索联系」共用这个槽位，文案必须对两者都成立 ——
    # 写成「要和哪个数据集合并？」会让关系探索的用户以为要改数据。
    "right_dataset_id": "要和哪个数据集关联？请给出数据集 id",
    "run_id": "要操作哪一次训练运行？请给出 run id",
    "target": "要预测（或作为分析对象）的是哪一列？",
    # 工作流编排：这些槽位不该靠反问补齐（用户答不出「edges 是什么」），
    # 文案只在抽取彻底失败、确实需要人来补时才用得上。
    "name": "给这个工作流起个名字？",
    "nodes": "工作流要包含哪些步骤？（如：读取数据 → 质量检查 → 生成统计）",
    "edges": "这些步骤的执行顺序是什么？",
    # 会话没绑定数据集时才会问：工具需要 dataset_id，传 None 只会把它打崩
    "dataset_id": "要对哪个数据集做这个分析？请给出数据集 id",
    "left_dataset_id": "左侧用哪个数据集？请给出数据集 id",
}
