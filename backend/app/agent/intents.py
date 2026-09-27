"""意图路由：零 LLM、零网络、确定性。

为什么是规则而不是模型
----------------------
这是整个 Agent **Token 消耗的主战场**，也是旧架构最复杂的地方。

旧架构在这里叠了六层决策：确定性长链 → 本地 Router（词法/Qwen 神经路由）→
Router 单步直连 → 确定性短链兜底 → Remote 结构化升级 → 纯对话，
层与层之间还有若干「让位条件」。实际走哪条路很难预测，而一旦升级到 Remote，
就要把候选工具的 schema 一起发给模型 —— 一次决策就是几千 Token。

新架构是一张关键词表 + 一次打分：

- **命中** → 交给固定的 playbook（工具链写死，不需要模型编排）
- **未命中** → 走对话，**不发起任何 LLM 调用**（除非配置了 Key 且用户确实在聊天）

代价是「说法没写进关键词表就识别不了」。这是可接受的取舍：
关键词表是一份**可读、可审计、可增量补充**的显式知识，
而神经网络路由器的行为只能靠重新训练改变。

打分规则
--------
- 强关键词（如「相关性」）3 分：出现即可判定意图
- 弱关键词（如「统计」）1 分：只用于同分时的细微倾斜
- 阈值 3 分：低于阈值一律判为 ``CHAT``

同分时按 :data:`_PRIORITY` 取靠前者（越具体的意图越靠前）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Intent(str, Enum):
    """用户意图。新增意图三处都要改：本枚举、``_STRONG``/``_WEAK``、``playbooks.PLAYBOOKS``。"""

    # ---- 数据集 ----
    LIST_DATASETS = "list_datasets"
    INSPECT_DATASET = "inspect_dataset"
    PREVIEW_DATASET = "preview_dataset"
    SCHEMA = "schema"
    PROFILE = "profile"
    QUALITY = "quality"
    OVERVIEW = "overview"
    #: 两个数据集之间的关系探索（共同字段 / 关联键），不是合并
    RELATION = "relation"

    # ---- EDA ----
    DESCRIBE = "describe"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    OUTLIER = "outlier"
    VISUALIZE = "visualize"

    # ---- 数据处理 ----
    FILTER = "filter"
    CLEAN = "clean"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    MERGE = "merge"

    # ---- 建模 ----
    ML_TRAIN = "ml_train"
    ML_EVALUATE = "ml_evaluate"
    ML_COMPARE = "ml_compare"
    ML_EXPLAIN = "ml_explain"

    # ---- 编排与产出 ----
    REPORT = "report"
    WORKFLOW = "workflow"
    #: 明确要**新建**一个工作流。与 WORKFLOW 分开是刻意的：
    #: 「看看有哪些工作流」只能列，不能让模型顺手编一个出来。
    WORKFLOW_CREATE = "workflow_create"
    CONNECTOR = "connector"

    # ---- 兜底 ----
    CHAT = "chat"

    def __str__(self) -> str:
        return self.value


#: 强关键词：命中即 +3 分。写「用户真的会这么说」的完整短语，
#: 单个泛词（如「分析」「数据」）不要放进来，否则会把所有请求都吸走。
_STRONG: dict[Intent, tuple[str, ...]] = {
    Intent.LIST_DATASETS: (
        "有哪些数据集", "数据集列表", "列出数据集", "所有数据集", "我的数据集",
        "list datasets", "show datasets",
    ),
    Intent.INSPECT_DATASET: (
        "数据集概况", "数据集信息", "基本信息", "数据集详情",
        # 「这份数据有多少行多少列」此前一个词都命中不了 —— 它既不是「质量」
        # 也不是「画像」，于是掉进对话路径，只能靠上下文瞎答（实测答得出列数、
        # 答不出行数）。规模类问法一律交给 dataset.inspect，那是它唯一的正解。
        # 刻意不收「行数」「条数」：它们是「行数据 / 条数据」的子串，
        # 「给我看几行数据」会被判成查规模而不是预览（预览才是对的）。
        "多少行", "有多少行", "多少条", "有多少条", "多少行数据", "记录数",
        "多少列", "有多少列", "数据量", "数据规模", "规模多大", "多大", "几行", "几条",
    ),
    Intent.PREVIEW_DATASET: (
        "预览数据", "预览一下", "前几行", "看几行", "样例数据", "样本数据", "看看数据",
        "预览", "preview", "head rows",
    ),
    Intent.SCHEMA: (
        "有哪些列", "字段名", "列名", "字段类型", "列类型", "数据结构", "表结构", "字段结构", "字段列表",
        "schema", "columns", "dtypes",
    ),
    Intent.PROFILE: (
        "数据画像", "画像", "整体概览", "数据概览", "总览", "profile",
    ),
    Intent.QUALITY: (
        "数据质量", "质量问题", "数据集的质量", "数据的质量", "质量怎么样", "质量如何",
        "质量情况", "缺失值", "缺失情况", "空值", "重复值", "重复行", "重复数据",
        # 「从缺失、重复、异常值、字段可用性四个角度评估」里前两项是**裸词**：
        # 表里只有「缺失值」「重复值」，于是整句话只命中「异常值」，
        # 用户要的四个角度里三个没跑，答案却写得像个完整评估。
        "字段可用性", "可用字段", "完整度", "数据完整性",         "缺失率", "重复率", "一致性",
        "quality", "missing value", "null count",
        # ★ 错别字容错：真实输入里「缺实值」「确失值」很常见（第十轮实测
        #   「帮窝看瞎这分数据的缺实值」一个词都命中不了，直接掉进对话路径，
        #   回答变成「需要实际跑一遍，你确认一下吗」—— 反问了一次本可以直接做的事）。
        "缺实值", "确失值", "缺矢值", "缺失数据", "丢值",
    ),
    Intent.DESCRIBE: (
        "描述统计", "描述性统计", "统计摘要", "统计描述", "基本统计",
        "describe", "summary statistics",
    ),
    Intent.DISTRIBUTION: (
        "分布", "频次", "分布分析", "取值分布",
        "distribution",
    ),
    Intent.CORRELATION: (
        "相关性", "相关系数", "相关分析", "关联分析", "相关矩阵",
        "correlation", "heatmap",
        # ★ 「再帮我跑一次，这次用 spearman」里既没有「相关性」也没有「相关系数」，
        #   只有方法名 —— 不认它，这句会掉进对话路径，用户要的重算根本没发生。
        "spearman", "pearson", "斯皮尔曼", "皮尔逊",
        # 「看看 duration 和 credit_amount 的关系」此前一个词都命中不了：它没有
        # 「相关性」三个字，于是掉进对话路径，Agent 回一段「请确认用哪种方式」
        # 的空话 —— 明明一次 eda.correlation 就能给出答案。
        "的关系", "有什么关系", "之间有什么关系",
        # ★ 「哪些字段之间存在潜在的共线性」此前一个词都命中不了 —— 掉进对话路径
        #   后只能回一句「需要看实际数据才能判断…你希望我对哪些字段做计算？」，
        #   把一次本来可以直接算的相关性推回给用户（Telco 压测实测）。
        "共线性", "多重共线性", "collinearity", "vif",
    ),
    Intent.OUTLIER: (
        "异常值", "离群", "异常点", "异常检测", "离群点",
        "outlier", "anomaly",
    ),
    Intent.VISUALIZE: (
        "画图", "可视化", "图表", "画个图", "柱状图", "折线图", "散点图", "箱线图", "饼图",
        "直方图", "histogram",
        "plot", "chart", "visualize",
        # ★ 「画个 region 和 monthly_charges 的关系图」此前掉进 CORRELATION
        #   （「的关系」是它的强词），而 region 是类别列 ⇒ eda.correlation 直接报
        #   「不是数值类型」，用户要的图一张没出。类别列 × 数值列只能画分组图。
        "关系图", "对比图", "趋势图", "占比图", "分组柱状图", "分组图",
        "分箱",
        # ★ 「画 MonthlyCharges 的小提琴图」「画三个子图」此前都不进这条链：
        #   「小提琴图」「子图」不在词表里，整句只有弱词 ⇒ 掉进对话路径，
        #   然后模型**凭空说自己画了**（「小提琴图我直接按 Churn 分组画…」），
        #   实际一张图都没有（Telco 压测实测）。图表名一律要能触发可视化链路。
        "小提琴图", "子图", "多子图", "累积分布图", "面积图", "qq图", "雷达图", "桑基图",
    ),
    Intent.FILTER: (
        "筛选", "过滤", "条件查询", "筛选出", "只保留", "筛出",
        "filter",
    ),
    Intent.CLEAN: (
        "清洗", "去重", "去除重复", "删除重复", "填充缺失", "处理缺失", "补全缺失", "清理数据",
        "clean", "drop duplicates", "fill missing",
        # 「把缺失值填充掉」「缺失值补一下」里的动词在**名词后面**，
        # 表里只有「填充缺失」（动词在前）匹配不上，于是整句被 QUALITY 的
        # 「缺失值」拿走 —— 用户要的是改数据，拿到的却是一份体检报告。
        "缺失值填充", "缺失值填补", "缺失值补", "缺失值处理", "空值填充", "空值填补",
        "填补缺失", "补全空值", "填充空值", "空值补全", "重复行删除", "重复行去掉", "重复值删除",
    ),
    Intent.TRANSFORM: (
        "新增列", "新增一列", "添加一列", "增加一列", "派生列", "添加列", "加一列",
        "计算列", "新增字段", "转换列",
        "transform",
    ),
    Intent.AGGREGATE: (
        "分组聚合", "分组统计", "聚合", "按组", "统计每个", "求和", "求平均", "平均", "平均值", "均值", "计数",
        # 「按 job 分组求 credit_amount 的**总和**」此前一个词都命中不了：
        # 表里只有「分组聚合」「分组统计」这类紧邻写法，而真实说法是「按 X 分组求…
        # 的总和」。结果是明确的聚合诉求掉进对话路径，Agent 回一段
        # 「请确认用哪个版本」的空话，一个数都没算。
        "分组", "总和", "总计", "汇总", "分组求", "每组的",
        "透视", "透视表", "pivot", "groupby", "group by", "aggregate",
    ),
    Intent.MERGE: (
        "合并数据集", "合并两个", "合并这两个", "关联两个", "合并表", "连接两个", "合并", "join", "merge",
    ),
    # 「预测」与算法名必须算强信号：
    # 「预测 survived」「用随机森林」「做聚类」都是明确的建模诉求，
    # 旧表里只有「预测模型」这种组合词命中，于是它们全掉进对话路径 ——
    # 用户看到的是一句「我没有理解你想做的具体分析」，而他要的模型根本没跑。
    Intent.ML_TRAIN: (
        "训练模型", "建模", "建个模型", "跑个模型", "训练一个", "机器学习", "预测模型",
        "train model", "train a model", "build model",
        "预测", "回归模型", "分类模型", "聚类", "分群", "分簇",
        "随机森林", "逻辑回归", "线性回归", "决策树", "knn", "kmeans", "dbscan",
        "random forest", "random_forest", "logistic regression", "linear regression",
        "decision tree",
    ),
    Intent.ML_EVALUATE: (
        "评估模型", "模型评估", "评估这次", "评估一下", "看看效果", "效果如何", "效果怎么样", "evaluate",
    ),
    Intent.ML_COMPARE: (
        "对比模型", "比较模型", "模型对比", "对比实验", "compare models", "compare",
    ),
    Intent.ML_EXPLAIN: (
        "特征重要性", "解释模型", "模型解释", "可解释", "feature importance", "explain",
    ),
    Intent.REPORT: (
        "生成报告", "导出报告", "实验报告", "分析报告", "出一份报告", "写报告",
        "report",
        # 「生成一份数据质量报告」此前一个词都命中不了：表里只有「生成报告」这种
        # 紧邻写法，而真实说法中间永远隔着「一份数据质量」。结果 QUALITY 靠
        # 「数据质量」拿到 3 分抢走整句 —— 用户要的是一份可交付的报告文件，
        # 拿到的却是一段体检结论。
        # 只加「质量报告」还不够：QUALITY 靠「数据质量」(强) +「质量」(弱) 拿 4 分，
        # REPORT 只有 3 分，同分都轮不到。这条完整短语让「一份数据质量报告」直接
        # 拿到 6 分，不必依赖弱词与优先级兜底。
        "数据质量报告", "质量报告", "数据报告", "总结报告", "做个报告", "出个报告", "报告文档",
        "给我报告", "报告给我", "生成一份报告", "要一份报告",
    ),
    Intent.WORKFLOW: (
        "工作流", "流程编排", "编排", "workflow", "pipeline",
        "执行工作流", "运行工作流", "跑一个工作流",
        # 「当前有哪些流程」此前 0 分掉进对话路径：WORKFLOW 的强词只有「工作流」，
        # 而用户这句里是「流程」。看列表的诉求拿不到列表，只能收到一段空话。
        # 刻意只收**列举式**短语，不收裸词「流程」（它会吸走大量无关句子）。
        "有哪些流程", "流程列表", "所有流程", "list workflows", "show workflows",
    ),
    #: 「创建工作流」里的「创建」此前没有任何权重：一句话里只要顺带提到
    #: 「描述性统计」，DESCRIBE 就拿到 3+1=4 分压过 WORKFLOW 的 3 分，
    #: 结果是用户要建流程，系统只跑了一次描述统计。
    Intent.WORKFLOW_CREATE: (
        "创建工作流", "新建工作流", "建个工作流", "建一个工作流", "搭个工作流", "搭建工作流",
        "创建流程", "新建流程", "自动化流程", "把流程固化", "固化成工作流",
        "create workflow", "build workflow", "new workflow",
        # ★ 「帮我设计一个客户流失分析的 Workflow」此前一个词都命中不了：
        #   「设计」既不在创建动词表里，也不在下面任何强词里，于是整句被 WORKFLOW
        #   的强词「workflow」拿走，落到 workflow.list —— 用户要一套方案，
        #   系统回一句「当前工作流列表为空」外加一段泛泛建议（第八轮实测）。
        #   这正是红线里点名的说法：「设计适合的 Workflow」必须进推荐流程。
        # ★ 刻意**不带名词的写法一概不收**。「设计一个机器学习实验方案」里
        #   也有「设计一个」，把它当强词会把建模诉求吸成建工作流（实测踩到：
        #   test_ml_train_suspends_then_completes_after_confirmation 直接跑飞）。
        #   泛指说法交给下面的「设计动词 + 工作流名词」结构化判定去接。
        "设计工作流", "设计一个工作流", "设计个工作流", "设计一套工作流",
        "设计流程", "设计一个流程", "设计一套流程", "设计适合的工作流", "设计适合的流程",
        "design workflow", "design a workflow",
    ),
    Intent.CONNECTOR: (
        "连接器", "外部数据源", "数据库接入", "外部数据库", "导入数据库", "connector",
    ),
    Intent.RELATION: (
        # 「探索这两个数据集间的联系」这类说法此前没有任何关键词命中：
        # 它既不含「合并」也不含「相关性」，于是只能被同一句话里的「机器学习」
        # 三个字吸到训练链路上，最后跑出一个谁也没要求的 kmeans。
        "数据集间的联系", "数据集之间的联系", "数据集之间的关系", "数据集间的关系",
        "有什么联系", "有什么关系", "之间的关系", "之间的联系",
        "数据集关系", "跨数据集", "数据集关联",
        # ★ 「两个数据集/这两个数据集」已降级到 _WEAK：它是**主语**不是诉求。
        #   挂在强词上时，「把这两个数据集按 customer_id 合并」里它一句就命中
        #   两条拿到 6 分，压过「合并」的 3 分 —— 用户说要合并，系统只做了一次
        #   关系探索，还在结尾提醒他「如需合并请说一句…」。
        "relation between", "how are the datasets",
    ),
    Intent.OVERVIEW: (
        "全面分析", "综合分析", "整体分析", "做个分析", "分析一下", "帮我分析",
        "分析这个数据集", "分析这份数据", "分析数据", "看看有什么",
    ),
}

#: 弱关键词：命中 +1 分。只用于在强词打平时做倾斜，不足以单独判定意图。
_WEAK: dict[Intent, tuple[str, ...]] = {
    Intent.PROFILE: ("概览", "总体情况"),
    Intent.DESCRIBE: ("统计",),
    Intent.QUALITY: ("质量",),
    # 刻意不放「分析」：它太泛，会给 OVERVIEW 平白加分，
    # 把「分析一下相关性」这类请求从具体意图手里抢走。
    Intent.OVERVIEW: ("洞察", "发现"),
    Intent.ML_TRAIN: ("训练", "预测"),
    # ★ 「改成 / 换成 / 重画」**不再是强词**。挂在强词上时它一个词就抢走整句：
    #   「刚才那个目标列，换成随机森林再跑一次」里 VISUALIZE 靠「换成」拿 3 分，
    #   压过 ML_TRAIN 的「随机森林」3 分（VISUALIZE 在 _PRIORITY 里更靠前）
    #   ⇒ 用户要换模型，系统画了一张直方图（Telco 压测实测）。
    #   它们只有在**图表语境**里才算改图（见 route 里的结构化判定）。
    Intent.VISUALIZE: ("图", "改成", "换成", "重画", "重新画"),
    # 「创建工作流：先做描述性统计再出报告」里，创建是**动词**，
    # 描述性统计只是流程中的一个节点。只靠 3 分强词会被 4 分的 DESCRIBE 压过去，
    # 用户要建流程却只拿到一份描述统计。给创建动词 1 分把它顶上去。
    Intent.WORKFLOW_CREATE: ("创建", "新建", "搭建", "固化", "编排成"),
    # 「两个数据集」只是主语，不是诉求：给它 1 分而不是 3 分，
    # 才能让「按 customer_id 合并这两个数据集」里的「合并」说了算。
    Intent.RELATION: ("两个数据集", "这两个数据集", "数据集之间", "两个数据集间"),
}

#: 同分时的优先级。越靠前越具体 —— 「相关性」比「分析」具体，
#: 因此用户说「分析一下相关性」时不应被 OVERVIEW 抢走。
#: 复合意图里「动作」优先于「对象」：「清洗掉缺失值」里清洗是动作、缺失值是对象，
#: 所以 CLEAN 在 QUALITY 之前；「创建工作流：训练模型」里创建工作流是主意图，
#: 所以 WORKFLOW 在 ML_TRAIN 之前。
_PRIORITY: tuple[Intent, ...] = (
    Intent.CORRELATION,
    # 「创建工作流：…先做描述性统计…」里创建是主意图、描述统计是从属步骤，
    # 所以它必须排在 DESCRIBE 之前 —— 否则同分时又被描述统计抢回去。
    Intent.WORKFLOW_CREATE,
    # 动作类意图整体前移：复合句里「动作」优先于「观察」。
    # 「清洗掉缺失值」若被 QUALITY 抢走，用户拿到的是一份体检报告而不是干净数据。
    Intent.CLEAN,
    Intent.FILTER,
    Intent.TRANSFORM,
    Intent.AGGREGATE,
    # REPORT 必须早于 QUALITY：一句「生成一份数据质量报告」里两者都会命中强词，
    # 而用户要的是**产出一份报告文件**，不是再看一遍体检结论 —— 同分时报告优先。
    Intent.REPORT,
    # QUALITY 必须早于 OUTLIER：它是**总纲**，「从缺失、重复、异常值三个角度看质量」
    # 里 dataset.quality 一条工具就覆盖缺失/重复/异常，跑它才是用户要的；
    # 让 OUTLIER 拿走同分，四个角度就只剩异常值一个，而答案仍写得像完整评估。
    Intent.QUALITY,
    Intent.OUTLIER,
    Intent.DISTRIBUTION,
    Intent.DESCRIBE,
    Intent.VISUALIZE,
    Intent.SCHEMA,
    Intent.PROFILE,
    Intent.PREVIEW_DATASET,
    Intent.INSPECT_DATASET,
    Intent.LIST_DATASETS,
    # 真要合并比「先看看有什么联系」更具体，因此 MERGE 在前；
    # 而 RELATION 必须早于建模意图 —— 它是建模的前置步骤，
    # 若被 ML_TRAIN 抢走，就会在没有目标列的情况下直接开训。
    Intent.MERGE,
    Intent.RELATION,
    # 编排（工作流）比单个建模动作更上层：用户说「创建工作流」时主意图是工作流。
    # 注意 WORKFLOW_CREATE 已在上方登记过（排在 DESCRIBE 之前），此处不重复。
    Intent.WORKFLOW,
    Intent.ML_COMPARE,
    Intent.ML_EVALUATE,
    Intent.ML_EXPLAIN,
    Intent.ML_TRAIN,
    # REPORT 已前移到 QUALITY 之前（同分时报告优先于体检），此处不再重复登记。
    Intent.CONNECTOR,
    Intent.OVERVIEW,
    Intent.CHAT,
)

#: 「动作压制对象」：同一句话里动作意图与对象意图**都过了阈值**时，动作胜出，
#: 哪怕对象意图分数更高。
#:
#: 为什么 :data:`_PRIORITY` 解决不了：「把这份数据里的重复行和缺失值清洗掉」
#: 里 CLEAN 只有「清洗」一个强词（3 分），QUALITY 却靠「缺失值」「重复行」
#: 两个强词拿了 6 分。按分数 QUALITY 赢 —— 用户要的是**清洗**（写操作），
#: 拿到的却是一份质量报告：数据一行没变，而答案写得像已经处理完了。
#: 同分优先级只管打平，压不住这种「对象词天然更多」的句式。
#:
#: 只登记真正踩过的组合，不搞通用「动作 > 对象」——
#: 后者会让「先看看质量再决定要不要清洗」也被判成清洗。
_ACTION_OVERRIDES: dict[Intent, tuple[Intent, ...]] = {
    Intent.CLEAN: (Intent.QUALITY,),
}

_STRONG_WEIGHT = 3
_WEAK_WEIGHT = 1
#: 判定阈值：低于它一律按对话处理。
MIN_SCORE = 3

_ROWS_RE = re.compile(r"(?:前|top|TOP)?\s*(\d{1,4})\s*(?:行|条|rows?)", re.IGNORECASE)

#: 总纲词：用户点名了**多个**分析角度（「包括质量、分布、相关性和可视化」）时，
#: 他要的是一次全跑完，不是被最靠前的那个具体意图抢走。
#: 实测「帮我做一次全面分析，包括质量、分布、相关性和可视化」只跑了一个
#: eda.correlation —— CORRELATION 在 _PRIORITY 里排第一，同分压过 OVERVIEW，
#: 于是答案里自己写着「质量与分布结果、可视化图表这一步没做成」。
_SWEEP_RE = re.compile(r"全面|综合|整体|全套|完整|一次性|从头到尾|方方面面|扫一遍|过一遍")

#: 图表语境词：只有和「改成 / 换成 / 重画」一起出现时，才说明用户在改图
_CHART_CONTEXT_RE = re.compile(
    r"图|图表|直方|散点|柱状|条图|箱线|饼图|热力|分箱|可视化|坐标|轴|chart|plot|histogram|scatter|boxplot|heatmap"
)
_REDRAW_RE = re.compile(r"改成|换成|改一下|换一下|重画|重新画|重做|换个|改成分|换成个")

#: 比较表达式 + 行/记录类名词 = 明确的**筛选**诉求。
#: 「找出所有 customer_id 大于 99999999 的记录」里没有「筛选/过滤」任何一个字，
#: 掉进对话路径后只能反问「请确认用当前这份数据」（第十一轮实测）——
#: 条件明明写在句子里，却一次筛选都没发生。
_COMPARE_RE = re.compile(
    r"大于|小于|超过|高于|低于|不低于|不少于|大于等于|小于等于|最多|最少"
    r"|[<>≥≤]\s*\d|\d\s*[<>]"
)
_ROW_NOUN_RE = re.compile(r"记录|行|数据|样本|条|用户|客户|订单")


@dataclass(frozen=True)
class RouteResult:
    """一次路由的结果。"""

    intent: Intent
    score: int
    reason: str
    #: 从文本里直接读到的简单槽位（如预览行数）。不含需要 schema 才能定的值。
    slots: dict[str, Any]
    #: 同时识别到、但本次不会执行的其他意图（分数由高到低）。
    #:
    #: 一句话里常有两个诉求（「探索两个数据集的联系，并做机器学习分析」）。
    #: 旧做法只返回胜出的那一个，另一个被**静默丢弃** —— 用户以为都做了，
    #: 实际只做了一半，而且没有任何提示。这里把它带出来，由引擎在路由事件里
    #: 明说「本次只执行 X，未执行 Y」。
    secondary: tuple[Intent, ...] = ()

    @property
    def is_chat(self) -> bool:
        return self.intent is Intent.CHAT


#: 「筛选后 / 清洗后 / 合并后」这类**动作 + 后**的组合是回指上一轮结果的
#: 时间状语，不是本轮诉求。注意只匹配**紧邻**的写法：
#: 「筛选出 age>50 的数据后再看看分布」中间隔着宾语，不会被误剥离。
_RESULT_REFERENCE_RE = re.compile(
    r"(?:筛选|过滤|清洗|清理|去重|合并|转换|抽样|填充|训练|建模)(?:后|之后|以后|完了|完成)"
)


def strip_result_reference(text: str) -> str:
    """去掉「筛选后 / 清洗后 / 合并后」这类回指状语，返回用户的**本轮**诉求。

    路由用它是为了避免 FILTER 抢走整句；答案渲染也用它 —— 否则模型会读到
    「筛选后还剩多少行」里的「筛选」，手里却没有这一步的事实，于是写出
    「筛选条件还没给，无法算出剩余行数。当前数据集共 164 行」这种
    先否定、后给数、自相矛盾的话。
    """
    return _RESULT_REFERENCE_RE.sub("", text or "")


#: 改口 / 放弃。上一轮还在等确认或等补充信息，用户这一句说「算了、不用了」，
#: 那就是撤回，不是新指令。不接住的话系统会把它当成新诉求重新跑一遍 ——
#: 实测：等确认清洗时说「算了，不用清洗了」，结果真的清洗了一次，
#: 而原来那条运行还永远停在「等待确认」。
#: 只在**确实有挂起运行**时才生效，普通对话不受影响。
_ABANDON_WORDS = (
    "算了", "不用了", "不要了", "不做了", "别做了", "别执行", "别运行",
    "取消", "撤销", "撤回", "停止", "停下来", "先别", "不用管", "跳过",
)
#: 「不用清洗了 / 不用再看了」这类中间夹着动词的写法
_ABANDON_RE = re.compile(r"不(?:用|要|必)[^，。！？；]{0,8}了")


#: 创建动词 / 工作流名词：分开判定，用来接住「创建一个叫 X 的 Workflow」这类
#: 中间隔着任意内容的语序（见 ``route`` 里的结构化判定）。
# 「设计」必须算创建动词：用户说「设计一个客户流失分析的 Workflow」时，
# 他要的是**产出一套流程方案**，和「创建一个」完全同义。缺了它，这句话会被
# 纯名词意图（workflow.list）抢走 —— 红线场景「设计适合的 Workflow」直接失效。
_CREATE_VERB_RE = re.compile(
    r"创建|新建|搭建|建个|建一个|创建个|做一个|生成个|编排"
    r"|设计|设计个|设计一个|设计一套|弄个|搞个|整一个|弄一个|搞一个|写个|配一个|配置个"
    # 「帮我搭一条…流水线」里的「搭一条」此前不认，于是整句 0 分掉进对话路径
    r"|搭个|搭一条|搭一个|搭条|搭套|搭出"
)
# ★ 「管道 / 流水线」必须算工作流名词。实测「创建一个完整的客户流失分析管道，
#   从数据清洗到模型输出」里没有 pipeline 也没有「流程」，只有 CLEAN 的强词「清洗」
#   命中 ⇒ 系统直接**真的清洗了一遍数据集**（7043 → 7032 行，新增版本）。
#   用户要的是设计一条管道，不是立刻改数据 —— 这是破坏性的路由误判。
_WORKFLOW_NOUN_RE = re.compile(r"workflow|工作流|流程|管道|流水线|pipeline|pipe", re.IGNORECASE)

#: 像「一段参数」而不是「一句需求」的输入。
_PAYLOAD_HINT_RE = re.compile(r"[{\[]\s*[\"']?\w+[\"']?\s*[:=]|[\"']\w+[\"']\s*:\s*[\"'\d\[{]")

#: ``k=v`` 参数串（``&`` 或 ``;`` 分隔）
_KV_PAIR_RE = re.compile(r"([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*=\s*([^;&\s]*)")

#: 这些「值」等于没给
_EMPTY_VALUES = frozenset({"", "null", "none", "nan", "undefined", "n/a", "nil", "无"})


def _looks_like_payload(text: str) -> bool:
    """这段输入是不是直接把参数/JSON 贴了进来。

    ★ 历史缺陷：把 ``{"dataset_id": "abc", "chart": , "column": }`` 贴进对话框，
      系统当作正常需求处理 —— 从里面抠出几个词就去画了一张没人要的直方图。
      用户看到的是「我说的是一段 JSON，它却给我画了张图」，完全不知所云。

    判据刻意保守：必须有**键值对的形状**才算，普通带括号的话不受影响。
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if len(raw) > 400:  # 超长输入另有一套处理，这里只管「像参数」的短输入
        return False
    if _PAYLOAD_HINT_RE.search(raw):
        return True
    # ★ ``dataset_id=null&column=&chart=`` 这种**空参数骨架**：形状是参数串，
    #   但一个值都没给。它此前被当成需求执行，系统自己挑了一列画了张图
    #   （第十一轮实测）—— 用户贴的是一串空参数，看到一张图只会更懵。
    #   反过来，``chart=boxplot; column=monthly_charges; bins=30`` 值都在，
    #   那是**能执行**的明确指令，不能拦。
    pairs = _KV_PAIR_RE.findall(raw)
    if len(pairs) >= 2 and all(v.strip().lower() in _EMPTY_VALUES for _, v in pairs):
        return True
    return False


def is_abandonment(text: str) -> bool:
    """这句话是不是在撤回上一步（而不是下达新指令）。

    ★ 只认**纯撤回**。``is_abandonment("算了，先告诉我数据有多少行")`` 曾返回 True，
      于是后半句的新诉求被整句丢掉，回答只有「好的，这一步不做了」——
      用户问的行数一个数都没给。撤回词之后还跟着内容的，那是**撤回 + 新指令**。
    """
    query = (text or "").strip()
    if not query:
        return False
    matched = any(word in query for word in _ABANDON_WORDS) or bool(_ABANDON_RE.search(query))
    if not matched:
        return False
    # 去掉撤回词后还剩有实质内容 ⇒ 是「撤回 + 新指令」，不能当纯撤回处理。
    remainder = query
    for word in _ABANDON_WORDS:
        remainder = remainder.replace(word, "")
    remainder = _ABANDON_RE.sub("", remainder)
    remainder = re.sub(r"[\s，,。.、；;！!？?的了哈吧呢啊]+", "", remainder)
    return len(remainder) < 4


def route(text: str) -> RouteResult:
    """把用户请求映射到一个意图。纯函数，无副作用、无网络、无随机性。"""
    query = (text or "").strip().lower()
    if not query:
        return RouteResult(Intent.CHAT, 0, "空请求", {})
    # 「筛选后还剩多少行」里的「筛选」是**回指上一轮操作**的时间状语，不是本轮
    # 诉求。不剥离的话 FILTER 会拿走整句，然后反问「筛选条件是什么」—— 用户
    # 只是想知道上一次筛选剩了多少行（实测：这句和「看看筛选后 credit_amount
    # 的分布」都卡在等待澄清，一步工具都没跑）。
    query = strip_result_reference(query)

    scores: dict[Intent, int] = {}
    hits: dict[Intent, list[str]] = {}
    for intent, keywords in _STRONG.items():
        for kw in keywords:
            if kw.lower() in query:
                scores[intent] = scores.get(intent, 0) + _STRONG_WEIGHT
                hits.setdefault(intent, []).append(kw)
    for intent, keywords in _WEAK.items():
        for kw in keywords:
            if kw.lower() in query:
                scores[intent] = scores.get(intent, 0) + _WEAK_WEIGHT
                hits.setdefault(intent, []).append(kw)

    # 「前 10 行 / 看 100 条」这类带行数的说法，字面子串穷举「前几行」覆盖不了
    # （「前10行」≠「前几行」），而 _ROWS_RE 能稳定识别。命中即落到预览意图。
    if _ROWS_RE.search(query):
        scores[Intent.PREVIEW_DATASET] = scores.get(Intent.PREVIEW_DATASET, 0) + _STRONG_WEIGHT
        hits.setdefault(Intent.PREVIEW_DATASET, []).append("前N行")

    # ★ 改图判定：「改成 / 换成」只有在**图表语境**里才指向改图。
    #   脱离语境它们是通用动词（换成随机森林、改成分组方式…），
    #   挂在强词上会把完全无关的诉求吸成"重画一张图"。
    if _CHART_CONTEXT_RE.search(query) and _REDRAW_RE.search(query):
        scores[Intent.VISUALIZE] = scores.get(Intent.VISUALIZE, 0) + _STRONG_WEIGHT
        hits.setdefault(Intent.VISUALIZE, []).append("改图")

    # ★ 比较表达式：句子里写死了筛选条件，就别再把它当一句闲聊。
    if _COMPARE_RE.search(query) and _ROW_NOUN_RE.search(query):
        scores[Intent.FILTER] = scores.get(Intent.FILTER, 0) + _STRONG_WEIGHT
        hits.setdefault(Intent.FILTER, []).append("比较条件")

    # ★ 总纲词加权：「全面分析 / 综合看一遍」这类说法后面往往跟着一串具体角度，
    #   只按同分优先级会让排在最前的 CORRELATION 拿走整句，其余角度一个都不跑。
    if _SWEEP_RE.search(query):
        scores[Intent.OVERVIEW] = scores.get(Intent.OVERVIEW, 0) + _STRONG_WEIGHT
        hits.setdefault(Intent.OVERVIEW, []).append("总纲词")

    # ★ 结构化判定：创建动词 + 工作流名词 ⇒ 一定是「新建」，不是「查看」。
    #
    # 关键词表覆盖不了真实语序：「帮我创建一个叫 客户流失预警 的 Workflow」
    # 里既没有紧邻的「创建工作流」，也没有「新建工作流」，于是只有弱词「创建」
    # 拿 1 分，被 WORKFLOW 的强词「workflow」(3 分) 压过 —— 用户要建流程，
    # 系统只列了一圈已有流程（实测两轮都卡在这里）。
    # 「动词 + 名词」的两个部分可以相隔很远，这正是子串词表的盲区。
    if _CREATE_VERB_RE.search(query) and _WORKFLOW_NOUN_RE.search(query):
        scores[Intent.WORKFLOW_CREATE] = scores.get(Intent.WORKFLOW_CREATE, 0) + _STRONG_WEIGHT * 2
        hits.setdefault(Intent.WORKFLOW_CREATE, []).append("创建动词+工作流")

    # ★ 畸形输入：看起像一段参数/JSON，而不是一句需求。
    # 实测把 ``{"dataset_id": "abc", "chart": , "column": }`` 贴进来，
    # 系统从里面抠出几个词就当需求跑了（画了一张没人要的直方图）——
    # 这种「幻觉式执行」比直接说「看不懂」更糟。
    if _looks_like_payload(text):
        return RouteResult(
            Intent.CHAT, 0,
            "输入看起来像一段结构化参数而不是分析需求",
            _extract_simple_slots(query),
        )

    if not scores:
        return RouteResult(Intent.CHAT, 0, "未命中任何意图关键词", _extract_simple_slots(query))

    best_score = max(scores.values())
    if best_score < MIN_SCORE:
        top = _pick(scores, best_score)
        return RouteResult(
            Intent.CHAT,
            best_score,
            f"最高分 {best_score} 低于阈值 {MIN_SCORE}（最接近 {top.value}），按对话处理",
            _extract_simple_slots(query),
        )

    winner = _pick(scores, best_score)
    winner = _action_first(scores, winner)
    keywords = hits.get(winner) or []
    others = [i for i, s in scores.items() if s >= MIN_SCORE and i is not winner]
    others.sort(key=lambda i: (-scores[i], _priority_index(i)))
    reason = f"命中关键词：{_clip_list(keywords)}"
    tied = [i for i in others if scores[i] == best_score]
    weaker = [i for i in others if scores[i] < best_score]
    if tied:
        reason += f"；同分按优先级取胜（另有 {', '.join(i.value for i in tied)}）"
    if weaker:
        reason += f"；同时识别到 {', '.join(i.value for i in weaker)}，本次只执行主意图"
    return RouteResult(winner, best_score, reason, _extract_simple_slots(query), tuple(others))


def _action_first(scores: dict[Intent, int], winner: Intent) -> Intent:
    """见 :data:`_ACTION_OVERRIDES`：动作意图过阈值时压过它作用的对象意图。"""
    for action, objects in _ACTION_OVERRIDES.items():
        if scores.get(action, 0) >= MIN_SCORE and winner in objects:
            return action
    return winner


def _priority_index(intent: Intent) -> int:
    return _PRIORITY.index(intent) if intent in _PRIORITY else len(_PRIORITY)


def _pick(scores: dict[Intent, int], best_score: int) -> Intent:
    """同分取 :data:`_PRIORITY` 中最靠前者；都不在表里时按枚举定义序兜底。"""
    candidates = [i for i, s in scores.items() if s == best_score]
    candidates.sort(key=lambda i: (_priority_index(i), list(Intent).index(i)))
    return candidates[0]


def _clip_list(items: list[str], limit: int = 4) -> str:
    shown = "、".join(items[:limit])
    return shown + ("…" if len(items) > limit else "") or "（无）"


def _extract_simple_slots(query: str) -> dict[str, Any]:
    """抽取不依赖 schema 的简单槽位。

    只取**明确写在文本里**的值。数据集 id 不在这里抽 —— 文本里的数字
    极不可靠（"前10行" 的 10 不是数据集 id），一律由会话绑定的 dataset_ids 决定。
    """
    slots: dict[str, Any] = {}
    match = _ROWS_RE.search(query)
    if match:
        value = int(match.group(1))
        if 0 < value <= 10000:
            slots["limit"] = value
    return slots
