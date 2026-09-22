"""学习中心 · 实验任务目录（单一事实源）。

设计意图
--------
学习中心此前空荡的根本原因：实验任务定义写死在两个前端文件里，
既没有后端来源，也没有任何「判定」能力——学生写完代码只是拿到一段
AI 评语，没有任何东西能回答「我做对了没有」。

本模块把任务定义搬到后端，并且给出**每条要求对应的可执行判定规则**。
判定分两类：
  - ``ast``   ：对提交代码做语法结构检查（不执行用户代码，安全）
  - ``data``  ：对所选真实数据集做统计检查（走 DataEngineService）

两类都是「客观可复核」的：同样的代码与数据必然得到同样的结论，
这正是学习场景需要的确定性，也是论文里可以写成实验指标的东西。

两条主线：
  - ``ml``  ：机器学习全流程（数据理解 → 划分 → 训练 → 评估 → 调优）
  - ``llm`` ：大模型核心机制（Attention / 采样 / 提示工程 / RAG / 微调）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TrackId = Literal["ml", "llm"]
Level = Literal["基础", "进阶", "挑战"]
CheckKind = Literal["ast", "data"]


@dataclass(frozen=True)
class Check:
    """一条可执行判定规则。"""

    # → 对应 requirement 的索引，前端把结果挂回具体某条要求上
    requirement_index: int
    kind: CheckKind
    # 人类可读的名字，直接展示在检查结果里
    label: str
    # ast 类：**全部**都必须命中的符号（如同时要 MAE 与 RMSE）
    require_all_calls: tuple[str, ...] = ()
    # ast 类：命中**任一**即可（同一能力的多种等价写法，如多种评价指标）
    require_any_calls: tuple[str, ...] = ()
    # ast 类：必须出现的属性访问（如 .fit / .predict）
    require_attrs: tuple[str, ...] = ()
    # ast 类：禁止出现（如禁止用现成的 sklearn 高级封装偷懒）
    forbid_calls: tuple[str, ...] = ()
    # ast 类：要求出现矩阵乘法（@ / np.matmul / np.dot）。
    # 用于「算相似度」「用权重加权 V」这类没有具名函数、只能靠运算判定的要求。
    require_matmul: bool = False
    # ast 类：要求这些变量**真的被赋值成了非占位内容**。
    # 这是本引擎最关键的一条约束：骨架里 ``TASK = ...`` 本身就含 ``TASK`` 这个名字，
    # 若只查「名字是否出现」，学习者什么都写也照样通过。所以这里查的是「右边是不是 ...」。
    require_filled: tuple[str, ...] = ()
    # require_filled 的附加门槛：字符串内容至少多少字符（挡住 ``TASK = "分析"`` 这种敷衍）
    min_length: int = 0
    # require_filled 的附加门槛：列表至少多少项（挡住 ``CONSTRAINTS = ["要准确"]``）
    min_items: int = 0
    # ast 类：要求源码里存在非占位的字符串字面量（用于「定义了角色」这类纯内容要求）
    require_string_literal: bool = False
    # ast 类：要求「自己实现了」某个能力——定义名字含关键字的函数，或用了
    # 切片 / 循环这种朴素手段。用于 RAG 切块这类「允许自己写而不调库」的要求。
    require_custom: tuple[str, ...] = ()
    # data 类：要执行的数据探针名（见 reviewer.py 的 _probe）
    probe: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """一个学习实验任务。"""

    key: str
    title: str
    track: TrackId
    level: Level
    # 一句话学什么
    summary: str
    # 为什么值得学（学习动机，放在任务卡片正面）
    why: str
    # 教学目标
    goal: str
    # 实验要求（人类可读，与 Check.requirement_index 一一对应）
    requirements: tuple[str, ...]
    # 起步代码骨架
    starter: str
    checks: tuple[Check, ...]
    # 需要哪类数据（用于前端提示怎么准备数据）
    needs: str = "任意表格数据集"
    # 预计投入
    minutes: int = 20
    # 关键概念（前端渲染成可展开的概念卡，不是课件正文）
    concepts: tuple[dict[str, str], ...] = field(default_factory=tuple)
    # 参考资料（真实外链）
    references: tuple[dict[str, str], ...] = field(default_factory=tuple)
    # 前置实验 key（用于锁/推荐顺序）
    prerequisite: str | None = None


# =====================================================================
# 机器学习主线
# =====================================================================

_ML_IRIS = ExperimentSpec(
    key="ml-classification-baseline",
    title="分类基线：从数据到可复核的准确率",
    track="ml",
    level="基础",
    summary="搭起第一条完整的监督学习链路，并让结果可复现。",
    why="这是所有机器学习项目的骨架。跑通一次，你就拥有了改造任何分类任务的地图。",
    goal="理解特征 / 标签、训练测试划分、模型拟合、测试集评价这四个不可省略的环节。",
    requirements=(
        "明确区分特征 X 与标签 y",
        "划分训练集与测试集（而非在全量数据上评估）",
        "创建并训练一个分类模型",
        "在测试集上计算至少一个评价指标",
    ),
    starter=(
        "# 分类基线：把下面 4 处 ... 换成真正的实现，然后点「检查」\n"
        "# 提示：先用「数据集」面板看一眼真实列名\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score\n"
        "\n"
        "# 1. 特征与标签：把标签列从特征里排除出去\n"
        "TARGET = ...   # 填标签列名\n"
        "X = ...        # 特征矩阵，例如 df.drop(columns=[TARGET])\n"
        "y = ...        # 标签向量，例如 df[TARGET]\n"
        "\n"
        "# 2. 划分训练集 / 测试集\n"
        "X_train, X_test, y_train, y_test = ...\n"
        "\n"
        "# 3. 创建并训练分类模型\n"
        "model = ...\n"
        "...            # 用 X_train / y_train 拟合\n"
        "\n"
        "# 4. 在测试集上评价（不要用训练集）\n"
        "y_pred = ...\n"
        "print('准确率:', ...)\n"
    ),
    checks=(
        Check(0, "ast", "区分了 X 与 y", require_all_calls=("drop",), require_attrs=()),
        Check(1, "ast", "做了训练 / 测试划分", require_all_calls=("train_test_split",)),
        Check(2, "ast", "创建并训练了模型", require_all_calls=("LogisticRegression",), require_attrs=("fit",)),
        Check(3, "ast", "计算了评价指标", require_any_calls=("accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score")),
        Check(3, "data", "标签列类别数 ≥ 2（分类任务前提）", probe="classification_target"),
    ),
    needs="含至少 2 个类别标签列的数据集",
    minutes=20,
    concepts=(
        {"term": "为什么必须留测试集", "body": "在同一批数据上训练又评估，等于考前看了答案。测试集模拟「没见过的新数据」，只有它给出的分数才对未来有效。"},
        {"term": "过拟合长什么样", "body": "训练集准确率接近 1.0、测试集明显偏低，就是模型把训练数据的噪声也背下来了。手段是正则化、减少特征、增加数据。"},
        {"term": "准确率什么时候骗人", "body": "类别严重不平衡时（比如 99% 是负例），全预测负例也有 99% 准确率。这时要看 F1、召回率或 AUC。"},
    ),
    references=(
        {"label": "scikit-learn 模型评估指南", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html"},
        {"label": "Google 机器学习速成课 · 分类", "url": "https://developers.google.com/machine-learning/crash-course/classification"},
    ),
)

_ML_REGRESSION = ExperimentSpec(
    key="ml-regression-error",
    title="回归任务：把「误差」当成主角",
    track="ml",
    level="基础",
    summary="训练回归模型，并用残差判断模型到底哪里错了。",
    why="回归的产出不是「准不准」这一个数，而是「在哪类样本上错得离谱」。看残差才能改模型。",
    goal="掌握连续目标的建模流程，理解 MAE 与 RMSE 的差异及其决策含义。",
    requirements=(
        "识别连续型目标列作为 y",
        "划分训练集与测试集",
        "训练一个回归模型",
        "计算 MAE 与 RMSE 两个误差指标",
    ),
    starter=(
        "# 回归任务：把下面 4 处 ... 换成真正的实现\n"
        "# 关注误差而不只是分数\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error\n"
        "import numpy as np\n"
        "\n"
        "TARGET = ...   # 填连续数值目标列名\n"
        "X = ...\n"
        "y = ...\n"
        "\n"
        "X_train, X_test, y_train, y_test = ...\n"
        "\n"
        "model = ...\n"
        "...            # 拟合\n"
        "\n"
        "y_pred = ...\n"
        "mae = ...      # 平均绝对误差\n"
        "rmse = ...     # 均方根误差（记得对 MSE 开方）\n"
        "print('MAE:', mae, 'RMSE:', rmse)\n"
    ),
    checks=(
        Check(0, "data", "目标列是连续数值列", probe="regression_target"),
        Check(1, "ast", "做了训练 / 测试划分", require_all_calls=("train_test_split",)),
        Check(2, "ast", "创建并训练了回归模型", require_all_calls=("RandomForestRegressor",), require_attrs=("fit",)),
        Check(3, "ast", "同时算了 MAE 与 RMSE", require_all_calls=("mean_absolute_error", "mean_squared_error")),
    ),
    needs="含连续数值目标列的数据集",
    minutes=25,
    prerequisite="ml-classification-baseline",
    concepts=(
        {"term": "MAE vs RMSE", "body": "MAE 是所有误差绝对值的平均，对异常值不敏感；RMSE 先平方再开方，会放大少数大误差。RMSE 远大于 MAE，说明存在个别错得很离谱的样本。"},
        {"term": "残差图怎么读", "body": "残差 = 真实值 - 预测值。理想情况残差随机散布在 0 附近。若呈喇叭形，说明误差随样本增大；若呈弧形，说明模型漏掉了非线性关系。"},
    ),
    references=(
        {"label": "scikit-learn 回归指标", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics"},
    ),
)

_ML_UNSUPERVISED = ExperimentSpec(
    key="ml-clustering-no-leak",
    title="无监督聚类：当数据没有标签",
    track="ml",
    level="进阶",
    summary="在没有 y 的情况下发现结构，并用轮廓系数决定要不要相信它。",
    why="现实中绝大多数数据没有标签。聚类是「先看看数据长什么样」的标准起手式，但没有指标就容易自欺欺人。",
    goal="理解聚类不需要 y，掌握用轮廓系数评估聚类质量而非「凭感觉看图」。",
    requirements=(
        "不使用标签列参与建模",
        "对特征做标准化（距离类算法对量纲敏感）",
        "用肘部法或轮廓系数选定 K",
        "输出至少一个聚类质量指标",
    ),
    starter=(
        "# 无监督聚类：把下面 4 处 ... 换成真正的实现\n"
        "# 没有 y，也要有判断依据\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import silhouette_score\n"
        "\n"
        "# 1. 特征矩阵：注意不要把任何标签列放进来\n"
        "X = ...\n"
        "\n"
        "# 2. 标准化（这一步不能省）\n"
        "X_scaled = ...\n"
        "\n"
        "# 3. 聚类\n"
        "k = 3\n"
        "model = ...\n"
        "labels = ...\n"
        "\n"
        "# 4. 聚类质量指标\n"
        "score = ...\n"
        "print('轮廓系数:', score)\n"
    ),
    checks=(
        Check(0, "ast", "未把标签列当作特征", forbid_calls=("accuracy_score", "f1_score", "train_test_split")),
        Check(1, "ast", "做了特征标准化", require_all_calls=("StandardScaler",)),
        Check(2, "ast", "使用了 KMeans 或等价聚类算法", require_any_calls=("KMeans", "DBSCAN", "AgglomerativeClustering")),
        Check(3, "ast", "计算了聚类质量指标", require_any_calls=("silhouette_score", "davies_bouldin_score", "calinski_harabasz_score")),
        Check(3, "data", "特征列中存在可标准化的数值列", probe="numeric_features"),
    ),
    needs="含多个数值列的数据集（不需要标签）",
    minutes=30,
    prerequisite="ml-classification-baseline",
    concepts=(
        {"term": "为什么必须标准化", "body": "KMeans 用欧氏距离衡量相似。若「年收入」是十万量级、「年龄」是两位数，距离几乎全由年收入决定，年龄等于没参与。标准化让每一列权重相当。"},
        {"term": "轮廓系数怎么读", "body": "取值 -1 到 1。越接近 1 说明样本离自己簇更近、离别的簇更远。0.5 以上通常算结构清晰，0.25 以下基本是硬切出来的。"},
        {"term": "肘部法的局限", "body": "肘部法看的是簇内平方和下降拐点，拐点常常不清晰。它是启发式，不是判决书——两个指标一起看，或结合业务含义定 K。"},
    ),
    references=(
        {"label": "scikit-learn 聚类指南", "url": "https://scikit-learn.org/stable/modules/clustering.html"},
    ),
)


# =====================================================================
# 大模型主线
# =====================================================================

_LLM_ATTENTION = ExperimentSpec(
    key="llm-attention-mechanism",
    title="Attention：亲手算一遍 Q / K / V",
    track="llm",
    level="基础",
    summary="不用任何框架，用矩阵运算把注意力机制从公式变成数字。",
    why="Transformer 的一切都建立在缩放点积注意力上。亲手算一遍，你就再也不会被「注意力」这个词唬住。",
    goal="掌握 Q/K/V 的来源、缩放为什么必要、Softmax 之后权重如何加权 V。",
    requirements=(
        "构造 Q、K、V 三个矩阵",
        "计算相似度矩阵 QKᵀ",
        "除以 √dₖ 做缩放并套用 Softmax",
        "用注意力权重对 V 加权求和",
    ),
    starter=(
        "# Attention：从公式到可运行的数字\n"
        "# 把下面 5 处 ... 换成真正的实现\n"
        "import numpy as np\n"
        "\n"
        "def softmax(x):\n"
        "    # 提示：先减去每行最大值再取 exp，防止数值溢出；最后按行归一化\n"
        "    ...\n"
        "\n"
        "seq_len, d_k = 4, 8\n"
        "rng = np.random.default_rng(42)\n"
        "\n"
        "# 1. 构造 Q / K / V（真实模型里由输入 x 乘可学习权重得到）\n"
        "Q = ...\n"
        "K = ...\n"
        "V = ...\n"
        "\n"
        "# 2. 相似度矩阵：Q 与 K 的转置做矩阵乘\n"
        "scores = ...\n"
        "\n"
        "# 3. 缩放（除以 sqrt(d_k)）+ Softmax\n"
        "scores = ...\n"
        "weights = ...\n"
        "\n"
        "# 4. 用注意力权重对 V 做加权求和\n"
        "out = ...\n"
        "\n"
        "print('权重每行和:', weights.sum(axis=1))\n"
        "print('输出形状:', out.shape)\n"
    ),
    checks=(
        Check(0, "ast", "构造了 Q / K / V", require_all_calls=("normal",)),
        Check(1, "ast", "计算了 Q 与 K 的相似度", require_matmul=True),
        Check(2, "ast", "做了 √dₖ 缩放", require_all_calls=("sqrt",)),
        Check(2, "ast", "套用了 Softmax", require_any_calls=("softmax", "exp")),
        Check(3, "ast", "用权重加权了 V", require_matmul=True),
    ),
    needs="无需数据集",
    minutes=25,
    concepts=(
        {"term": "Q / K / V 各是什么", "body": "Q（Query）是「我在找什么」，K（Key）是「我有什么可供匹配」，V（Value）是「匹配上之后我实际取走的信息」。三者都由同一份输入经不同权重矩阵线性变换得到。"},
        {"term": "为什么必须除以 √dₖ", "body": "维度越高，点积的方差越大，数值会跑到 Softmax 的饱和区——输出接近 one-hot，梯度几乎为零，训练不动。除以 √dₖ 把方差拉回 1 量级。"},
        {"term": "Softmax 前为什么要减最大值", "body": "exp 在数值较大时会溢出成 inf。减去每行最大值不改变结果（分子分母同乘一个常数），却能把指数控制在安全范围。这是工程上必备的细节。"},
    ),
    references=(
        {"label": "Attention Is All You Need（原论文）", "url": "https://arxiv.org/abs/1706.03762"},
        {"label": "The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/"},
    ),
)

_LLM_TOKENIZATION = ExperimentSpec(
    key="llm-tokenization-sampling",
    title="生成控制：温度、Top-k 与 Top-p",
    track="llm",
    level="进阶",
    summary="用真实的概率分布亲手实现解码策略，看清「随机性」是怎么被调出来的。",
    why="为什么同一个提示词每次回答不一样？为什么温度调高会胡言乱语？答案全在解码这一步。",
    goal="掌握温度缩放、Top-k 截断、Top-p 核采样的实现差异与适用场景。",
    requirements=(
        "用 Softmax 把 logits 转成概率分布",
        "实现温度缩放（temperature）",
        "实现 Top-k 截断",
        "实现 Top-p（核采样）截断",
    ),
    starter=(
        "# 解码策略：控制生成的三把旋钮\n"
        "# 补齐 4 个函数体，然后观察同一个 logits 在不同策略下的分布差异\n"
        "import numpy as np\n"
        "\n"
        "def softmax(x):\n"
        "    ...\n"
        "\n"
        "def apply_temperature(logits, temperature=1.0):\n"
        "    # 提示：temperature -> 0 趋于确定性；-> 大 趋于均匀\n"
        "    ...\n"
        "\n"
        "def top_k_filter(probs, k=5):\n"
        "    # 提示：用 np.argsort(probs)[::-1] 取概率最大的 k 个位置，其余置 0，再归一化\n"
        "    ...\n"
        "\n"
        "def top_p_filter(probs, p=0.9):\n"
        "    # 提示：先降序排序，用 np.cumsum 求累积概率，累积到 p 就截断\n"
        "    ...\n"
        "\n"
        "logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -2.0])\n"
        "print('原始:', apply_temperature(logits, 1.0).round(3))\n"
        "print('低温:', apply_temperature(logits, 0.3).round(3))\n"
        "print('Top-k:', top_k_filter(apply_temperature(logits), 3).round(3))\n"
        "print('Top-p:', top_p_filter(apply_temperature(logits), 0.9).round(3))\n"
    ),
    checks=(
        Check(0, "ast", "实现了 Softmax 概率化", require_any_calls=("exp", "softmax")),
        Check(1, "ast", "实现了温度缩放", require_any_calls=("temperature",)),
        Check(2, "ast", "实现了 Top-k 截断", require_any_calls=("argsort", "topk", "partition")),
        Check(3, "ast", "实现了 Top-p 核采样", require_all_calls=("cumsum",)),
    ),
    needs="无需数据集",
    minutes=30,
    prerequisite="llm-attention-mechanism",
    concepts=(
        {"term": "温度到底在做什么", "body": "logits 除以温度 T。T<1 时分布更尖锐（更确定、更重复），T>1 时更平坦（更发散、更容易出错）。它不是「创造力旋钮」，而是「原分布的锐化/平滑程度」。"},
        {"term": "Top-k 与 Top-p 的分工", "body": "Top-k 固定候选数量，简单但对「本就只有 1 个合理候选」和「有 50 个候选」一视同仁。Top-p 按累积概率自适应截断，候选池随分布形状变化，因此更常用。"},
        {"term": "为什么不能只调温度", "body": "温度是全局的，它同时影响头部的合理候选和尾部的垃圾词。截断类方法专门砍掉长尾，两者通常配合使用：先截断再采样。"},
    ),
    references=(
        {"label": "Hugging Face 文本生成策略", "url": "https://huggingface.co/docs/transformers/generation_strategies"},
        {"label": "The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751"},
    ),
)

_LLM_PROMPT = ExperimentSpec(
    key="llm-prompt-structure",
    title="提示工程：把「感觉」变成结构化约束",
    track="llm",
    level="基础",
    summary="用角色、任务、约束、输出格式四段式重构你的提示词。",
    why="提示词写不清楚，模型再强也只能猜。结构化提示是普通人最划算的能力提升。",
    goal="掌握可复用的提示词骨架，并理解「输出格式约束」为什么能显著降低返工。",
    requirements=(
        "定义明确的角色或身份",
        "写清具体任务（而非笼统目标）",
        "给出至少两条显式约束",
        "指定输出格式",
        "提供少样本示例（few-shot）",
    ),
    starter=(
        "# 提示工程：把「感觉」变成结构化约束\n"
        "# 你的任务：为一个真实场景写一条结构化提示词，并补齐末尾的拼装逻辑\n"
        "# 场景建议：让模型根据字段清单判断数据集适合做哪类机器学习任务\n"
        "ROLE = '你是一名严谨的数据分析助手。'\n"
        "\n"
        "# 写清具体任务（而不是笼统目标），例如：根据字段清单判断该数据集适合分类还是回归\n"
        "TASK = ...\n"
        "\n"
        "# 至少两条显式约束。好约束是可检验的，并给出「信息不足时怎么办」的明确退路\n"
        "CONSTRAINTS = [\n"
        "    ...,\n"
        "]\n"
        "\n"
        "# 指定输出格式（例如以 JSON 返回 {'task': ..., 'reason': ...}）\n"
        "OUTPUT_FORMAT = ...\n"
        "\n"
        "# 一两个「输入 -> 输出」的少样本示例\n"
        "FEW_SHOT = ...\n"
        "\n"
        "# 把上面几段拼装成最终提示词（要让 CONSTRAINT 与 OUTPUT_FORMAT 真的出现在结果里）\n"
        "prompt = ...\n"
        "print(prompt)\n"
    ),
    checks=(
        Check(0, "ast", "定义了角色 / 身份", require_string_literal=True),
        Check(1, "ast", "写清了具体任务", require_filled=("TASK",), min_length=20),
        Check(2, "ast", "给出了显式约束", require_filled=("CONSTRAINTS",), min_items=2),
        Check(3, "ast", "指定了输出格式", require_filled=("OUTPUT_FORMAT",)),
        Check(4, "ast", "提供了少样本示例", require_filled=("FEW_SHOT",)),
    ),
    needs="无需数据集",
    minutes=20,
    concepts=(
        {"term": "为什么「输出格式」最值钱", "body": "下游要用程序解析结果时，格式不稳定等于不可用。显式给出 JSON schema 或字段名，能把「自然语言解释」变成「可编程接口」，返工率下降最明显。"},
        {"term": "少样本示例在教什么", "body": "示例不是在补充知识，而是在演示「格式」与「判断尺度」。模型从示例里学的是你想让答案长什么样，而不是事实本身。"},
        {"term": "约束要写「不要什么」", "body": "只说「要准确」没有约束力。写成「字段不足时必须回答信息不足」才是可检验的——它给了模型一条明确的退路，而不是逼它编。"},
    ),
    references=(
        {"label": "OpenAI 提示工程指南", "url": "https://platform.openai.com/docs/guides/prompt-engineering"},
        {"label": "Prompt Engineering Guide", "url": "https://www.promptingguide.ai/"},
    ),
)

_LLM_RAG = ExperimentSpec(
    key="llm-rag-pipeline",
    title="RAG 骨架：让模型回答它没学过的东西",
    track="llm",
    level="挑战",
    summary="实现切块、向量化、检索、拼装上下文的完整链路。",
    why="微调成本高、更新慢。RAG 用「先检索再回答」把外部知识接进模型，是当前落地最广的方案。",
    goal="掌握 RAG 的四个环节，理解切块策略与检索质量对最终答案的决定性影响。",
    requirements=(
        "把文档切分成块（chunk）",
        "把文本块转成向量表示",
        "按相似度检索出与问题最相关的块",
        "把检索结果拼进提示词上下文",
    ),
    starter=(
        "# RAG：从文档到可回答的上下文\n"
        "# 补齐 4 个关键函数，跑通「切块 -> 向量化 -> 检索 -> 拼上下文」\n"
        "import numpy as np\n"
        "\n"
        "def chunk_text(text, size=200, overlap=40):\n"
        "    # 提示：步长 = size - overlap，重叠是为了避免答案正好被切在边界上\n"
        "    ...\n"
        "\n"
        "def embed(texts):\n"
        "    # 真实场景换成 embedding 模型；这里用确定性伪向量演示流程\n"
        "    vecs = []\n"
        "    for t in texts:\n"
        "        rng = np.random.default_rng(abs(hash(t)) % (2**32))\n"
        "        v = rng.normal(size=64)\n"
        "        vecs.append(v / (np.linalg.norm(v) + 1e-9))\n"
        "    return np.array(vecs)\n"
        "\n"
        "def retrieve(question, chunks, top_k=3):\n"
        "    # 提示：先用 embed 把问题和所有块转成向量，算余弦相似度，再用 argsort 取前 top_k\n"
        "    ...\n"
        "\n"
        "doc = '把这里替换成你的一段真实文档文本。' * 20\n"
        "chunks = chunk_text(doc)\n"
        "\n"
        "# 把检索结果拼进提示词上下文\n"
        "context = ...\n"
        "prompt = ...   # 用 context 拼出最终提示词\n"
        "print(prompt[:300])\n"
    ),
    checks=(
        Check(0, "ast", "实现了文本切块", require_custom=("chunk", "split")),
        Check(1, "ast", "实现了向量化", require_any_calls=("embed", "encode", "vectorize")),
        Check(2, "ast", "实现了相似度检索", require_matmul=True),
        Check(3, "ast", "把检索结果拼进了上下文", require_any_calls=("join", "context", "prompt")),
    ),
    needs="无需数据集",
    minutes=45,
    prerequisite="llm-prompt-structure",
    concepts=(
        {"term": "切块为什么要有重叠", "body": "按固定长度硬切，关键句可能刚好被切成两半，两半都不足以回答问题。overlap 让边界内容同时出现在相邻块里，代价是存储与检索量增加。"},
        {"term": "检索对了，答案才对", "body": "RAG 的上限由检索决定。检索召回的是无关内容，模型只能基于无关内容编答案。调优的第一优先级永远是检索质量，不是提示词措辞。"},
        {"term": "相似度不等于有用", "body": "问题和文档用词相似，不代表文档包含答案。所以生产系统常加「重排序（rerank）」环节：先宽召回，再用更强的模型精排。"},
    ),
    references=(
        {"label": "检索增强生成综述", "url": "https://arxiv.org/abs/2312.10997"},
        {"label": "LangChain RAG 教程", "url": "https://python.langchain.com/docs/tutorials/rag/"},
    ),
)


CATALOG: tuple[ExperimentSpec, ...] = (
    _ML_IRIS,
    _ML_REGRESSION,
    _ML_UNSUPERVISED,
    _LLM_ATTENTION,
    _LLM_PROMPT,
    _LLM_TOKENIZATION,
    _LLM_RAG,
)

BY_KEY: dict[str, ExperimentSpec] = {spec.key: spec for spec in CATALOG}

TRACKS: tuple[dict[str, Any], ...] = (
    {
        "id": "ml",
        "label": "机器学习",
        "description": "从数据理解到模型评估的完整链路。每个实验都要求你在一份真实数据集上跑出可复核的结果。",
        "accent": "track-ml",
    },
    {
        "id": "llm",
        "label": "大模型",
        "description": "不调用黑盒 API，用矩阵运算把注意力、采样、检索这些机制亲手算一遍。",
        "accent": "track-llm",
    },
)


def get_spec(key: str) -> ExperimentSpec | None:
    return BY_KEY.get(key)


def spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    """目录项 -> API 字典（不含判定规则细节，前端只需知道有条数与名字）。"""
    return {
        "key": spec.key,
        "title": spec.title,
        "track": spec.track,
        "level": spec.level,
        "summary": spec.summary,
        "why": spec.why,
        "goal": spec.goal,
        "requirements": list(spec.requirements),
        "starter": spec.starter,
        "needs": spec.needs,
        "minutes": spec.minutes,
        "prerequisite": spec.prerequisite,
        "concepts": [dict(c) for c in spec.concepts],
        "references": [dict(r) for r in spec.references],
        "check_count": len(spec.checks),
    }
