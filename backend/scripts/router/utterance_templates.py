"""路由样本的表达模板（人工维护部分）。

这是整个数据集里**唯一需要人工编写**的内容，其余（参数边界、非法组合、
hard negative、槽位缺失、holdout 划分）全部由 `build_dataset.py` 自动派生。

维护约定
--------
1. 模板用 `{占位符}` 标记槽位。占位符的**显示形式**由 `build_dataset.SLOT_SPEC`
   的 `display` 决定（如 `dataset_id` 显示为「3号数据集」），参数真值另行回填。
   所以占位符要**紧贴后文、不加空格**，否则会渲染出「3号数据集 的字段」这种断句。
2. 同一个工具要给**不同语体**的说法：口语 / 书面 / 省略主语 / 倒装 / 带情绪 /
   中英混用 / 简短祈使。清一色「请帮我 XXX」会让模型学到句式而不是语言。
3. 只说平台**真实具备**的能力。新增工具时先补模板，再跑生成器。
4. 占位符必须与该工具 schema 里**真实存在的参数**对应（required 或 optional），
   否则 `validate_decision` 会判 invalid 并进 `rejected`。
   槽位 → 参数的映射见 `build_dataset.SLOT_SPEC`（`category_column`→`group_by`、
   `mltopic`→`topic` 等）。`nodes`/`edges`/`conditions`/`aggregations`/`expression`
   这类一句话说不清的必填参数由 `_STRUCTURED` 自动补最小合法值，模板不必出现。
5. 刻意保留少量**不带数据集**的说法（如「{column}这一列的分布怎么样」），
   它们会自动变成「反问用户」类样本 —— 这是该类数据的自然来源，
   但**只放在确有必要的工具上**，否则会把 routing 样本整体稀释。

规模基线（2026-09-22 扩写）
--------------------------
每工具 **4 → 11~12 条**。动机：模板级 groupCV 下词法基线的工具准确率只有 57.9%，
而样本级 holdout 有 88.4% —— 差距几乎全部来自「句式只有 4 条，换个说法就认不出」。
**数据多样性才是当前的天花板，不是模型容量。**
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 槽位取值池
# ---------------------------------------------------------------------------

DATASET_IDS = [1, 3, 7, 12, 42]
RUN_IDS = [1, 2, 5, 11]
WORKFLOW_IDS = [1, 2, 3, 8]

NUMERIC_COLUMNS = [
    "age",
    "price",
    "sepal_length",
    "petal_width",
    "income",
    "score",
    "temperature",
    "sales",
]
CATEGORY_COLUMNS = ["city", "category", "species", "gender", "grade", "region", "product_type"]

# 图表类型的中文说法 -> eda.visualize 的 chart 枚举值
CHART_WORDS: dict[str, str] = {
    "直方图": "histogram",
    "柱状图": "bar",
    "折线图": "line",
    "散点图": "scatter",
    "箱线图": "boxplot",
    "热力图": "heatmap",
    "Q-Q 图": "qq",
    "分组柱状图": "grouped_bar",
    "面积图": "area",
}

# 模型的中文/英文说法 -> 平台模型 key（用于 ml.train 的 model 槽位）
MODEL_WORDS: dict[str, str] = {
    "逻辑回归": "logistic_regression",
    "KNN": "knn_classifier",
    "K 近邻": "knn_classifier",
    "决策树": "decision_tree_classifier",
    "随机森林": "random_forest_classifier",
    "线性回归": "linear_regression",
    "KMeans 聚类": "kmeans",
    "DBSCAN": "dbscan",
    "PCA 降维": "pca",
}

# ml.explain_config 的 topic 说法 -> 枚举值
TOPIC_WORDS: dict[str, str] = {
    "流程步骤": "pipeline",
    "预处理参数": "preprocessing",
    "训练参数": "training",
    "评估指标": "metrics",
    "调参": "tuning",
}


# ---------------------------------------------------------------------------
# 单工具路由模板：tool -> [utterance]
# 占位符紧贴后文，不加空格。
# ---------------------------------------------------------------------------

ROUTING_TEMPLATES: dict[str, list[str]] = {
    # ---- dataset：数据集与元信息（6）----
    "dataset.list": [
        "我有哪些数据集",
        "列出所有数据集",
        "平台里现在有什么数据",
        "看看我上传过哪些表",
        "数据集清单给我一份",
        "我现在能分析的数据有哪几个",
        "都有哪些表格",
        "数据列表",
        "我传过的东西列一下",
        "把所有数据集的名字报给我",
        "现在可用的数据集有哪些",
    ],
    "dataset.inspect": [
        "{dataset_id}的详细信息给我看看",
        "{dataset_id}有几个版本",
        "看看{dataset_id}的最新规模",
        "帮我看下{dataset_id}的元信息",
        "{dataset_id}到底多大，多少行多少列",
        "想了解一下{dataset_id}的基本情况",
        "{dataset_id}的信息汇总一下",
        "给我{dataset_id}的概览",
        "{dataset_id}是什么时候建的",
        "把{dataset_id}的详情调出来",
        "{dataset_id}的属性能列一下吗",
    ],
    "dataset.preview": [
        "预览一下{dataset_id}",
        "{dataset_id}前几行长什么样",
        "我想看看{dataset_id}里的实际数据",
        "把{dataset_id}的内容调出来看看",
        "{dataset_id}实际长啥样，给我瞅两眼",
        "抽样看下{dataset_id}的数据",
        "{dataset_id}的头几行",
        "让我看看{dataset_id}的真身",
        "{dataset_id}里的数据能预览下吗",
        "把{dataset_id}打开看看内容",
        "{dataset_id}具体记录了些什么",
    ],
    "dataset.schema": [
        "{dataset_id}有哪些列",
        "{dataset_id}的字段结构是什么",
        "看看{dataset_id}的列名和类型",
        "{dataset_id}的列分别是什么类型",
        "{dataset_id}的表结构给我",
        "{dataset_id}都有哪些字段",
        "想知道{dataset_id}的列定义",
        "{dataset_id}的 schema 是什么",
        "列一下{dataset_id}的字段",
        "{dataset_id}里每列是什么含义什么类型",
        "{dataset_id}的字段清单来一份",
    ],
    "dataset.profile": [
        "给{dataset_id}做个统计画像",
        "{dataset_id}的数值分布和缺失情况怎么样",
        "生成{dataset_id}的数据概况",
        "帮我摸一下{dataset_id}的整体情况",
        "{dataset_id}的画像出一份",
        "概览一下{dataset_id}，各列分布和缺失都要",
        "{dataset_id}整体上是什么样子的",
        "给我{dataset_id}的 profile",
        "看下{dataset_id}的统计概貌",
        "{dataset_id}各列的值大概怎么分布的",
        "出一份{dataset_id}的整体描述",
    ],
    "dataset.quality": [
        "检查{dataset_id}的数据质量",
        "{dataset_id}有没有缺失值和重复行",
        "{dataset_id}干净吗",
        "帮我看下{dataset_id}的质量问题",
        "{dataset_id}质量体检一下",
        "{dataset_id}的数据有没有问题",
        "跑一遍{dataset_id}的质量检查",
        "{dataset_id}数据可靠吗，有没有脏数据",
        "诊断下{dataset_id}的质量",
        "{dataset_id}的重复和空值情况",
        "健康检查一下{dataset_id}",
        "这数据质量怎么样",
    ],
    # ---- data：加工（5）----
    "data.filter": [
        "把{dataset_id}里{column}大于 100 的行筛出来",
        "过滤{dataset_id}，只保留{column}超过 100 的",
        "从{dataset_id}中筛选出符合条件的记录",
        "{dataset_id}按条件过滤一下",
        "{dataset_id}只要{column}超过 100 的那部分",
        "把{dataset_id}里{column}不满足大于 100 的行去掉",
        "想从{dataset_id}里挑出符合要求的子集",
        "{dataset_id}筛一下子集",
        "{dataset_id}按{column}过滤，保留大于 100 的",
        "用条件把{dataset_id}裁一下",
        "{dataset_id}里{column}符合要求的单独拿出来",
    ],
    "data.clean": [
        "清洗一下{dataset_id}",
        "{dataset_id}的缺失值帮我处理掉",
        "把{dataset_id}里的重复行去掉",
        "{dataset_id}数据太脏了，清理一下",
        "{dataset_id}得洗一洗了",
        "帮{dataset_id}做数据清洗",
        "{dataset_id}里的空值和重复都处理掉",
        "{dataset_id}清洗干净再往下用",
        "{dataset_id}有些脏数据，去一下",
        "{dataset_id}缺失的补上，重复的删掉",
        "把{dataset_id}整理干净",
        "帮我把数据清洗一下",
    ],
    "data.transform": [
        "给{dataset_id}加一个新列",
        "{dataset_id}里加一列算出来的值",
        "我想在{dataset_id}上派生一个新字段",
        "基于{dataset_id}的现有列做一个新列",
        "给{dataset_id}新增派生列",
        "{dataset_id}上算一个新字段出来",
        "在{dataset_id}里用现有列组合出新列",
        "{dataset_id}加一列，值由已有列算得",
        "{dataset_id}需要构造一个新变量",
        "帮{dataset_id}造一个新字段",
        "{dataset_id}派生一列出来",
    ],
    "data.aggregate": [
        "把{dataset_id}按{category_column}分组统计",
        "{dataset_id}按{category_column}做个聚合汇总",
        "对{dataset_id}按{category_column}分组求平均值",
        "{dataset_id}按照{category_column}分组算一下",
        "{dataset_id}每个{category_column}的均值是多少",
        "{dataset_id}分组汇总一下，维度是{category_column}",
        "用{category_column}给{dataset_id}做分组聚合",
        "{dataset_id}按{category_column}分组求平均",
        "{dataset_id}按{category_column}拆开统计",
        "{dataset_id}分{category_column}求均值",
        "对{dataset_id}做{category_column}维度的聚合",
    ],
    "data.merge": [
        "把{left_dataset_id}和{right_dataset_id}合并",
        "{left_dataset_id}和{right_dataset_id}按 Key 关联起来",
        "把{left_dataset_id}、{right_dataset_id}合成一张表",
        "帮我 join 一下{left_dataset_id}和{right_dataset_id}",
        "{left_dataset_id}和{right_dataset_id}拼在一起",
        "把{left_dataset_id}与{right_dataset_id}对齐合并",
        "{left_dataset_id}、{right_dataset_id}按主键连表",
        "{left_dataset_id}和{right_dataset_id}做一次关联",
        "将{left_dataset_id}和{right_dataset_id}整合为宽表",
        "{left_dataset_id}和{right_dataset_id}两个表合起来",
        "用{left_dataset_id}去关联{right_dataset_id}",
    ],
    # ---- eda：探索分析（5）----
    "eda.describe": [
        "对{dataset_id}做描述性统计",
        "{dataset_id}的均值和中位数是多少",
        "给我{dataset_id}的统计概览",
        "算一下{dataset_id}各列的基本统计量",
        "{dataset_id}的统计摘要出一份",
        "{dataset_id}各列均值方差都是多少",
        "看一下{dataset_id}的描述统计",
        "{dataset_id}的分位数和均值",
        "{dataset_id}统计信息汇总",
        "把{dataset_id}的基本统计量算出来",
        "{dataset_id}各列的汇总统计",
        "帮我做个描述性统计",
    ],
    "eda.distribution": [
        "{column}这一列的分布怎么样",
        "帮我看看{dataset_id}里{column}的分布",
        "画一下{dataset_id}里{column}的分布情况",
        "{column}的取值均匀吗",
        "{dataset_id}中{column}的分布形态如何",
        "{dataset_id}里{column}的取值集中在哪",
        "看看{dataset_id}的{column}是怎么分布的",
        "{dataset_id}的{column}偏态吗",
        "{dataset_id}里{column}的频次分布",
        "{dataset_id}的{column}这一列取值概况",
        "分析{dataset_id}中{column}的分布",
    ],
    "eda.correlation": [
        "{dataset_id}各列的相关性如何",
        "算一下{dataset_id}的相关性矩阵",
        "{dataset_id}里哪些变量之间相关性强",
        "给我{dataset_id}的相关系数",
        "{dataset_id}的变量之间有没有相关性",
        "{dataset_id}两两相关程度怎么样",
        "{dataset_id}相关矩阵算一个",
        "看下{dataset_id}各列之间的相关关系",
        "{dataset_id}哪些字段是强相关的",
        "{dataset_id}的列间相关性分析",
        "{dataset_id}变量相关性强弱排一下",
        "看看这些列之间有没有关系",
    ],
    "eda.outlier": [
        "{dataset_id}有没有异常值",
        "对{dataset_id}做离群点检测",
        "找出{dataset_id}里的异常数据",
        "{dataset_id}哪些列有离群点",
        "{dataset_id}有没有明显离群的点",
        "检查{dataset_id}的异常值情况",
        "{dataset_id}里哪些点不太正常",
        "{dataset_id}离群点排查一下",
        "{dataset_id}的极值多不多",
        "给{dataset_id}做异常检测",
        "{dataset_id}数据里有没有离谱的取值",
    ],
    "eda.visualize": [
        "给{column}画个{chart}",
        "{dataset_id}生成一张{chart}",
        "我想看{column}的{chart}",
        "用{dataset_id}画个{chart}看看",
        "{dataset_id}画一张{chart}",
        "帮我用{dataset_id}出个{chart}",
        "{dataset_id}的{column}画成{chart}",
        "来一张{dataset_id}的{chart}",
        "{dataset_id}做个{chart}",
        "用{dataset_id}里的{column}画{chart}",
        "{dataset_id}的{chart}给我出一张",
        "画个{chart}看看",
    ],
    # ---- ml：建模（8）----
    "ml.detect_task": [
        "{dataset_id}适合做什么任务",
        "{dataset_id}是分类还是回归问题",
        "判断一下{dataset_id}该用哪类模型",
        "用{dataset_id}能建什么模",
        "{dataset_id}能做分类吗",
        "{dataset_id}该走监督还是无监督",
        "帮{dataset_id}判断一下任务类型",
        "{dataset_id}适合哪种建模路线",
        "{dataset_id}这个数据建模方向是什么",
        "看下{dataset_id}能用什么模型",
        "{dataset_id}的任务类型判一下",
    ],
    "ml.prepare": [
        "给{dataset_id}做预处理配置",
        "{dataset_id}训练前要怎么处理",
        "帮我准备{dataset_id}的预处理方案",
        "{dataset_id}的缺失和编码怎么处理",
        "{dataset_id}建模前先做预处理",
        "{dataset_id}的数据要预处理好",
        "给{dataset_id}配一套预处理流程",
        "{dataset_id}训练前的清洗和编码准备一下",
        "{dataset_id}预处理方案定一下",
        "{dataset_id}的预处理怎么配",
        "为{dataset_id}准备预处理步骤",
    ],
    "ml.train": [
        "用{model}训练{dataset_id}",
        "用{dataset_id}跑个{model}",
        "帮我拿{dataset_id}训练一个{model}",
        "我要在{dataset_id}上训练{model}",
        "{dataset_id}训练一个{model}出来",
        "用{dataset_id}试试{model}",
        "拿{dataset_id}训个{model}",
        "帮我用{model}在{dataset_id}上建模",
        "{dataset_id}跑一遍{model}",
        "用{dataset_id}做{model}训练",
        "{dataset_id}上拿{model}benchmark 一下",
    ],
    "ml.predict": [
        "用{run_id}的模型做预测",
        "拿{run_id}的模型推理一批新数据",
        "{run_id}得到的模型预测一下新数据",
        "用{run_id}的结果做推理",
        "{run_id}训出来的模型跑一下预测",
        "用{run_id}的模型推一批",
        "拿{run_id}的模型对新数据做推理",
        "{run_id}的模型应用一下",
        "用{run_id}做预测",
        "{run_id}模型的预测结果跑出来",
        "调用{run_id}的模型做批量预测",
    ],
    "ml.evaluate": [
        "{run_id}的评估指标是什么",
        "{run_id}这个模型效果怎么样",
        "看看{run_id}的指标",
        "{run_id}的模型表现如何",
        "{run_id}训练结果的评估指标",
        "{run_id}跑出来效果好吗",
        "{run_id}的各项指标报一下",
        "{run_id}的模型准不准",
        "评估一下{run_id}",
        "{run_id}的结果评估一下",
        "看下{run_id}的评测结果",
    ],
    "ml.compare": [
        "把这几次训练的结果对比一下",
        "哪个模型效果最好",
        "比较一下我跑过的这些实验",
        "把所有运行按指标排个序",
        "几次训练的指标放一起比比",
        "哪个运行的分数最高",
        "我这些实验谁表现最好",
        "对比一下所有训练结果",
        "排一下各次运行的排名",
        "各模型效果横向对比一下",
        "我这几次建模哪个更强",
    ],
    "ml.explain": [
        "{run_id}的模型里哪些特征最重要",
        "解释一下{run_id}的模型",
        "{run_id}的特征重要性给我看看",
        "{run_id}为什么这么预测",
        "{run_id}的模型是怎么决策的",
        "{run_id}哪些变量影响最大",
        "帮我解释{run_id}的模型逻辑",
        "{run_id}最重要的特征是哪个",
        "看下{run_id}的特征贡献",
        "{run_id}模型的特征权重",
        "{run_id}靠哪些特征做判断",
    ],
    "ml.explain_config": [
        "平台的建模{mltopic}是什么样的",
        "解释一下{mltopic}",
        "{mltopic}有哪些可选值",
        "我想了解一下{mltopic}",
        "{mltopic}具体指什么",
        "介绍一下平台的{mltopic}",
        "{mltopic}可以怎么设置",
        "平台的{mltopic}说明一下",
        "{mltopic}的取值都有什么",
        "想搞懂{mltopic}",
        "帮我讲讲{mltopic}",
    ],
    # ---- workflow：编排（5）----
    "workflow.list": [
        "平台上有哪些工作流",
        "列出所有工作流",
        "我建过哪些 Workflow",
        "看看已有的流程有哪些",
        "我的工作流清单",
        "现在有哪些编排好的流程",
        "工作流都有哪些",
        "列一下我建的流程",
        "有哪些 Workflow 可以用",
        "看看我保存过哪些流程",
        "流程列表给我一份",
    ],
    "workflow.create": [
        "帮我建一个工作流",
        "创建一个新的 Workflow",
        "设计一个流程并保存下来",
        "搭一个数据分析流程",
        "帮我编排一个新流程",
        "新建一个工作流",
        "我要定义一个流程",
        "组一个分析流程存起来",
        "建一个 Workflow",
        "帮我搭一个流程框架",
        "创建一个可复用的流程",
    ],
    "workflow.inspect": [
        "{workflow_id}里有哪些节点",
        "看看{workflow_id}的结构",
        "{workflow_id}是怎么连的",
        "查看{workflow_id}的定义",
        "{workflow_id}包含哪些步骤",
        "{workflow_id}的节点和连线给我看看",
        "{workflow_id}具体是什么流程",
        "{workflow_id}的定义打开看看",
        "{workflow_id}里面怎么编排的",
        "看下{workflow_id}的内容",
        "{workflow_id}的组成是什么",
    ],
    "workflow.run": [
        "运行{workflow_id}",
        "执行{workflow_id}",
        "把{workflow_id}跑一下",
        "启动{workflow_id}",
        "{workflow_id}跑起来",
        "让{workflow_id}执行一遍",
        "触发{workflow_id}",
        "{workflow_id}运行一下",
        "把{workflow_id}执行了",
        "开始跑{workflow_id}",
        "{workflow_id}走一遍",
    ],
    "workflow.build_and_run": [
        "建个工作流直接跑起来",
        "创建一个流程并立刻执行",
        "帮我编排一个流程然后运行",
        "搭好流程顺便跑一遍",
        "新建流程并马上执行",
        "帮我搭一个流程并且立刻跑",
        "编排一个流程直接运行",
        "建流程加运行一步到位",
        "搭个流程立刻跑起来",
        "创建并执行一个工作流",
        "组一个流程并直接跑",
    ],
    # ---- report（1）----
    "report.generate": [
        "给{dataset_id}生成一份分析报告",
        "{dataset_id}出个报告",
        "帮我写一份{dataset_id}的完整分析报告",
        "基于{dataset_id}生成实验报告",
        "{dataset_id}的分析报告写一份",
        "用{dataset_id}产出报告",
        "{dataset_id}报告生成一下",
        "给我{dataset_id}的报告",
        "{dataset_id}做一份汇总报告",
        "帮{dataset_id}出一份分析结论文档",
        "根据{dataset_id}写个报告",
        "出一份分析报告",
    ],
}


# ---------------------------------------------------------------------------
# 非工具请求（intent = chat）
# 这一类是「省 token」的主战场：挡掉它们等于省掉一次远程调用。
# ---------------------------------------------------------------------------

CHAT_UTTERANCES: list[str] = [
    "你好",
    "在吗",
    "谢谢",
    "你是谁",
    "你能做什么",
    "今天天气怎么样",
    "讲个笑话",
    "帮我把这份报告发给老板",
    "晚上吃什么好",
    "再见",
    "你用的是哪个大模型",
    "这个平台是谁做的",
    "感觉你今天不太聪明",
    "陪我聊会儿天",
    "1+1 等于几",
    "早上好",
    "哈喽",
    "你叫什么名字",
    "你觉得自己聪明吗",
    "帮我点个外卖",
    "推荐一部电影",
    "现在几点了",
    "讲个冷笑话",
    "你累不累",
    "随便聊点什么",
    "今天星期几",
    "辛苦了",
    "在干嘛",
    "夸夸我",
    "晚安",
]


# ---------------------------------------------------------------------------
# 需要升级的请求（escalate = true）
# 每条都必须**真的**超出平台能力，否则会把升级判定带偏。
# ---------------------------------------------------------------------------

ESCALATION_CASES: list[dict[str, str]] = [
    # out_of_scope：平台压根没有这类工具
    {"utterance": "把这个数据集同步到我们的 S3 存储桶", "reason": "out_of_scope"},
    {"utterance": "预测一下明天的股票价格", "reason": "out_of_scope"},
    {"utterance": "帮我按这张照片画出数据分布", "reason": "out_of_scope"},
    {"utterance": "把结果发到我的微信上", "reason": "out_of_scope"},
    {"utterance": "用 GPT-4 微调一个自己的模型", "reason": "out_of_scope"},
    {"utterance": "把这份数据用 3D 立体的方式展示出来", "reason": "out_of_scope"},
    {"utterance": "连上我们公司的 Oracle 数据库同步数据", "reason": "out_of_scope"},
    {"utterance": "生成一段介绍这个数据集的宣传视频", "reason": "out_of_scope"},
    {"utterance": "接入 Kafka 实时流做在线预测", "reason": "out_of_scope"},
    # multi_step：需要的能力平台都有，但一次规划串不起来
    {
        "utterance": "把 A 表和 B 表合并，清洗完再训练三个模型对比，"
        "选最好的一个做预测，最后把全流程写份报告",
        "reason": "multi_step",
    },
    {
        "utterance": "先做数据质量检查，有问题就自动清洗，然后跑 EDA，"
        "再根据结果决定要不要做特征工程",
        "reason": "multi_step",
    },
    {
        "utterance": "根据上个月的销售数据预测下个月，"
        "如果低于目标就自动调整投放策略并生成一份说明",
        "reason": "multi_step",
    },
    {
        "utterance": "先合并两张表，再做特征工程，然后对比三个模型并调参，"
        "最后自动选最优的部署上线",
        "reason": "multi_step",
    },
    {
        "utterance": "持续监控这个模型的表现，指标掉下来就自动重训并替换",
        "reason": "multi_step",
    },
    # ambiguous：信息不足且无法用一次反问补齐
    {"utterance": "处理一下这个", "reason": "ambiguous"},
    {"utterance": "你懂的，就那个", "reason": "ambiguous"},
    {"utterance": "优化一下", "reason": "ambiguous"},
    {"utterance": "按老规矩来一份", "reason": "ambiguous"},
    {"utterance": "那个啥弄一下", "reason": "ambiguous"},
    {"utterance": "帮我弄好点", "reason": "ambiguous"},
    # param_dependency：参数之间有平台未建模的复杂依赖
    {
        "utterance": "训练的时候让学习率随迭代次数衰减，"
        "前一半用余弦退火后一半用常数，再配合 warmup",
        "reason": "param_dependency",
    },
    {
        "utterance": "自动帮我搜索最优超参数组合，"
        "要求验证集 f1 最高同时训练时间不超过 10 秒",
        "reason": "param_dependency",
    },
    {
        "utterance": "自定义一个损失函数让模型对少数类更敏感，"
        "还要支持早停和梯度裁剪",
        "reason": "param_dependency",
    },
    {
        "utterance": "显存不足时自动调小 batch size 并接着往下训",
        "reason": "param_dependency",
    },
    # conflict：与平台约束直接冲突
    {"utterance": "把这个数据集的行数改成 100 行然后保存", "reason": "conflict"},
    {"utterance": "把这个数据集删掉，但它的历史版本都要保留", "reason": "conflict"},
    {"utterance": "把这一列的类型改成整数，但不要产生新版本", "reason": "conflict"},
    {"utterance": "把数据集的列名全部改成中文，但不要影响已经训好的模型", "reason": "conflict"},
    {"utterance": "在不新建版本的前提下把这份数据直接覆盖掉", "reason": "conflict"},
]


# 模糊但**平台能处理**的对照样本：必须 escalate = false。
# 用来压住升级判定的误报率 —— 没有这批数据，模型会把「说得不清楚」
# 全部判成「要升级」，反而制造更多远程调用。
#
# `slots` 是这句话**没说出口、由上下文补齐**的参数：数据集从会话绑定拿，
# run_id / model 延续最近一次操作。它们不进 utterance，但要进 gold，
# 否则这批样本会全部退化成「缺槽位」，失去对照意义。
#
# 注意：这批用例**不允许**出现缺槽位 —— 它们是 slot_missing 的对照组，
# 生成器遇到 status != ok 会直接进 rejected（不会自动转换）。
RESOLVABLE_IMPLICIT: list[dict] = [
    {"utterance": "这个数据干净吗", "tool": "dataset.quality", "slots": {"dataset_id": 1}},
    {"utterance": "帮我看看有多少行", "tool": "dataset.inspect", "slots": {"dataset_id": 1}},
    {"utterance": "这些列都是什么", "tool": "dataset.schema", "slots": {"dataset_id": 1}},
    {
        "utterance": "跑个基线模型试试",
        "tool": "ml.train",
        "slots": {"dataset_id": 1, "model": "logistic_regression"},
    },
    {"utterance": "哪个特征重要一点", "tool": "ml.explain", "slots": {"run_id": 1}},
    {"utterance": "数据长什么样", "tool": "dataset.preview", "slots": {"dataset_id": 1}},
    {"utterance": "有没有特别离谱的值", "tool": "eda.outlier", "slots": {"dataset_id": 1}},
    {"utterance": "它们之间有关系吗", "tool": "eda.correlation", "slots": {"dataset_id": 1}},
    {"utterance": "出份材料", "tool": "report.generate", "slots": {"dataset_id": 1}},
    {"utterance": "从头跑一遍流程", "tool": "workflow.build_and_run", "slots": {}},
    {"utterance": "有几列", "tool": "dataset.schema", "slots": {"dataset_id": 1}},
    {"utterance": "帮我看看数据", "tool": "dataset.preview", "slots": {"dataset_id": 1}},
    {"utterance": "清洗下", "tool": "data.clean", "slots": {"dataset_id": 1}},
    {"utterance": "相关性算一下", "tool": "eda.correlation", "slots": {"dataset_id": 1}},
    {"utterance": "结果怎么样", "tool": "ml.evaluate", "slots": {"run_id": 1}},
    {
        "utterance": "出张图",
        "tool": "eda.visualize",
        "slots": {"dataset_id": 1, "chart": "histogram"},
    },
]


# ---------------------------------------------------------------------------
# 多轮指代模板：上下文省略了宾语，需要靠 recent_tools 消解
# 同 RESOLVABLE_IMPLICIT：必须是完整可执行的决策，缺槽位会进 rejected。
# ---------------------------------------------------------------------------

FOLLOW_UP_TEMPLATES: list[dict] = [
    {"utterance": "再过滤一下", "tool": "data.filter", "after": "data.filter",
     "slots": {"dataset_id": 1}},
    {"utterance": "换一种清洗方式", "tool": "data.clean", "after": "data.clean",
     "slots": {"dataset_id": 1}},
    {"utterance": "这个再画一遍", "tool": "eda.visualize", "after": "eda.visualize",
     "slots": {"dataset_id": 1, "chart": "histogram"}},
    {"utterance": "用随机森林再试试", "tool": "ml.train", "after": "ml.train",
     "slots": {"dataset_id": 1, "model": "random_forest_classifier"}},
    {"utterance": "再训练一次，参数保持不变", "tool": "ml.train", "after": "ml.train",
     "slots": {"dataset_id": 1, "model": "random_forest_classifier"}},
    {"utterance": "刚才那个模型预测一下新数据", "tool": "ml.predict", "after": "ml.train",
     "slots": {"run_id": 1}},
    {"utterance": "刚才那个结果出份报告", "tool": "report.generate", "after": "eda.describe",
     "slots": {"dataset_id": 1}},
    {"utterance": "换成 Spearman 再算一遍", "tool": "eda.correlation",
     "after": "eda.correlation", "slots": {"dataset_id": 1}},
    {"utterance": "换个模型再来一次", "tool": "ml.train", "after": "ml.train",
     "slots": {"dataset_id": 1, "model": "decision_tree_classifier"}},
    {"utterance": "这个再聚合一下", "tool": "data.aggregate", "after": "data.aggregate",
     "slots": {"dataset_id": 1, "group_by": "city"}},
    {"utterance": "把刚才那步再跑一遍", "tool": "workflow.run", "after": "workflow.run",
     "slots": {"workflow_id": 1}},
    {"utterance": "接着再统计一遍", "tool": "eda.describe", "after": "eda.describe",
     "slots": {"dataset_id": 1}},
]


# ---------------------------------------------------------------------------
# 明确的「信息不全」场景：正确动作是**反问用户**，不是升级
#
# 关键约束：这批样本的 `bound_dataset_id` 必须为 None。
# 只要上下文绑定了数据集，dataset_id 就不再算缺失，missing 也就不成立 ——
# 「缺槽位」本质上是**依赖上下文**的判定，不是纯文本判定。
# ---------------------------------------------------------------------------

SLOT_MISSING_TEMPLATES: list[dict] = [
    {"utterance": "清洗一下这批数据", "tool": "data.clean"},
    {"utterance": "帮我做个描述性统计", "tool": "eda.describe"},
    {"utterance": "画一下分布", "tool": "eda.distribution"},
    {"utterance": "生成一份报告", "tool": "report.generate"},
    {"utterance": "把异常值过滤掉", "tool": "data.filter"},
    {"utterance": "看看相关性", "tool": "eda.correlation"},
    {"utterance": "检查一下数据质量", "tool": "dataset.quality"},
    {"utterance": "训练一个模型", "tool": "ml.train"},
    {"utterance": "跑个模型看看效果", "tool": "ml.train"},
    {"utterance": "这个模型评估一下", "tool": "ml.evaluate"},
    {"utterance": "看看特征重要性", "tool": "ml.explain"},
    {"utterance": "预测一下新数据", "tool": "ml.predict"},
    {"utterance": "查看一下数据集的列结构", "tool": "dataset.schema"},
    {"utterance": "做一下预处理", "tool": "ml.prepare"},
    {"utterance": "帮我看看这列的分布", "tool": "eda.distribution"},
    {"utterance": "把缺失值填一下", "tool": "data.clean"},
    {"utterance": "我要建模", "tool": "ml.train"},
    {"utterance": "做个可视化", "tool": "eda.visualize"},
    {"utterance": "看看字段有哪些", "tool": "dataset.schema"},
    {"utterance": "汇总统计一下", "tool": "eda.describe"},
]
