"""ML 教学元数据（单一事实源）。

目标：把「模块内部流程 + 关键参数 + 取值范围 + 默认值 + 对结果的影响」结构化地表达出来，
供三处复用，避免文档与代码各写一份：

1. API：`GET /ml/catalog` 返回完整目录，前端「机器学习」页与学习中心直接渲染；
2. 工具：Agent 的 `ml.explain_config` 工具读取同一份数据，回答"这个参数是干什么的"；
3. 文档：`backend/docs/ML_GUIDE.md` 的关键参数表由本文件派生（口径一致）。

设计原则：
- 只描述**真实存在**的参数（与 `MODEL_REGISTRY` / `preprocessing.py` 的校验集合一致），
  不虚构学术名词；无法确定影响时明确写"影响有限"而不是编造。
- **每个参数必须能被 `MODEL_REGISTRY.supports_param()` 签名探测命中** ——
  即该名字确实是底层 sklearn 构造函数的形参，否则前端允许填写、后端却会 TypeError。
- 每个参数给：名称 / 中文名 / 类型 / 默认值 / 真实可回填的默认值 / 取值范围 /
  tier（core 最值得先调 / advanced 进阶）/ 步长 / 常用值 / 依赖约束 / 作用与调整建议。
- 每个流程步骤给：名称 / 输入 / 输出 / 关键行为 / 可核验点（中间结果落在哪里）。

两处超出「参数说明」的增值内容：
- `PARAM_COMBOS`：参数不是彼此独立的（如 penalty=l1 必须配 liblinear/saga），
  在点训练之前就提示无效组合，而不是等 sklearn 抛英文错。
- `TUNING_PLAYBOOK`：把「指标不理想」翻译成「具体该动哪个参数」，
  让调参从试错变成有方向的动作（overfit / underfit / no_signal / imbalanced / cluster_weak / costly）。
"""

from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------------
# 一、流程步骤（数据输入 → 预处理 → 训练/推理 → 结果输出）
#     每一步都标明「中间产物落在哪里」，对应代码里的产物键名。
# ----------------------------------------------------------------------
PIPELINE_STEPS: list[dict[str, Any]] = [
    {
        "id": "load",
        "name": "① 数据输入",
        "summary": "从某个数据集版本快照读取数据（不可变，保证可复现）。",
        "input": "dataset_id + version（缺省最新版本）",
        "output": "Polars DataFrame",
        "key_behaviour": [
            "实验与 DatasetVersion 严格绑定：run 时读的是当时那份快照，后续改动数据不会影响已完成的实验。",
            "数据以 Parquet 存储，读取时整表载入内存（大表需注意内存占用）。",
        ],
        "verify": "run.artifacts.train_rows / test_rows 记录了实际参与训练与测试的行数。",
        "code": "app/experiments/service.py::ExperimentService._load_data",
    },
    {
        "id": "feature_select",
        "name": "② 特征与目标划分",
        "summary": "确定 target（监督任务）并从特征中排除不参与训练的列。",
        "input": "target_column + excluded_columns",
        "output": "X（特征）/ y（目标，聚类为 None）",
        "key_behaviour": [
            "excluded_columns 用于排除 id、主键、高基数字符串等无信息列。",
            "target 不允许同时出现在 excluded_columns 中（会被显式拒绝）。",
        ],
        "verify": "run.artifacts.features 是排除后真正参与训练的原始列清单。",
        "code": "app/experiments/service.py::ExperimentService._execute",
    },
    {
        "id": "split",
        "name": "③ 训练/测试划分",
        "summary": "按比例切分数据；分类任务在每类样本足够时自动分层。",
        "input": "X, y, test_size, seed",
        "output": "X_train / X_test / y_train / y_test",
        "key_behaviour": [
            "test_size 默认 0.2；分类任务若每类样本数 ≥ 2 则按类别分层（stratify），保证类别比例一致。",
            "固定 seed 后切分结果完全一致（可复现的关键）。",
            "聚类任务不做切分：全部样本参与拟合，评估用轮廓系数。",
        ],
        "verify": "run.artifacts.train_rows / test_rows / stratified 记录切分规模与是否分层。",
        "code": "app/ml_engine/preprocessing.py::PreprocessingPipeline.train_test_split",
    },
    {
        "id": "preprocess",
        "name": "④ 预处理（防泄漏）",
        "summary": "缺失填补 → 类别编码 → 数值标准化，统计量只在训练集拟合。",
        "input": "X_train, X_test, preprocessing 配置",
        "output": "Xtr / Xte（模型可直接消费的数值矩阵）",
        "key_behaviour": [
            "**防泄漏**：管道只在 X_train 上 fit_transform，X_test 只 transform，绝不使用测试集统计量。",
            "未显式提供配置时，按数据画像自动生成默认配置（有缺失则填补、有类别列则编码、有数值列则标准化）。",
            "时间列转时间戳、布尔列转 0/1，避免 datetime64 直接进模型报错。",
        ],
        "verify": "run.artifacts.preprocessing_report 含 fitted / features_out / 实际生效的 config；model_features 是编码后的真实特征名。",
        "code": "app/ml_engine/preprocessing.py::PreprocessingPipeline",
    },
    {
        "id": "train",
        "name": "⑤ 模型训练",
        "summary": "在训练集上拟合模型，seed 注入 random_state 保证可复现。",
        "input": "Xtr, y_train, model, params",
        "output": "已训练的 estimator（可序列化）",
        "key_behaviour": [
            "模型参数透传给底层 sklearn 模型；不支持的 random_state 会被自动跳过（seed 仍用于数据切分）。",
            "任务类型由目标列自动推断（推断场景下若模型与任务不匹配会自动换用该任务默认模型，并在 model_adjusted 里记录）。",
        ],
        "verify": "run.artifacts.model_summary 含 name / task / params / n_features / n_samples / fit_seconds。",
        "code": "app/ml_engine/base.py::ModelAdapter.fit",
    },
    {
        "id": "evaluate",
        "name": "⑥ 评估",
        "summary": "在测试集上计算指标，输出可直接解读的评估报告。",
        "input": "Xte / y_test（或全量 X 对聚类）",
        "output": "metrics + 混淆矩阵 / 逐类报告 / 残差统计",
        "key_behaviour": [
            "分类：accuracy / precision / recall / f1（macro）/ roc_auc（需概率输出）。",
            "回归：MAE / MSE / RMSE / R²。",
            "聚类：簇数量 / 轮廓系数（簇数不满足条件时显式置空并说明）。",
            "指标永不返回 NaN/Inf，无法计算时返回 None + 原因文字，避免 JSON 序列化失败。",
        ],
        "verify": "run.metrics + run.artifacts.confusion_matrix / per_class / residual_stats。",
        "code": "app/ml_engine/evaluation.py",
    },
    {
        "id": "persist",
        "name": "⑦ 产物持久化",
        "summary": "模型与预处理管道落盘，供推理与解释复用。",
        "input": "model / pipeline",
        "output": "experiments/{exp_id}/runs/{run_id}/model.pkl + pipeline.pkl",
        "key_behaviour": [
            "推理必须同时加载 pipeline：训练特征是编码/标准化后的列，直接用原始列会列不匹配。",
            "产物缺失时给出可操作提示（'请重新训练一次'），不静默产出结果。",
        ],
        "verify": "run.artifacts.model_key / pipeline_key。",
        "code": "app/experiments/service.py::ExperimentService.run",
    },
    {
        "id": "predict",
        "name": "⑧ 推理",
        "summary": "加载模型与管道，对新数据批量预测。",
        "input": "run_id + dataset_id（或内联数据）",
        "output": "预测列 + 概率列 + 预览",
        "key_behaviour": [
            "严格校验推理数据的特征列是否齐全，缺列即明确报错。",
            "分类模型额外返回各分类的概率列（predict_proba）。",
            "聚类模型对新样本的预测语义有限（DBSCAN 不支持对新样本预测）。",
        ],
        "verify": "predict 返回 feature_columns / model_features / pipeline_applied / runtime。",
        "code": "app/experiments/service.py::ExperimentService.predict",
    },
    {
        "id": "explain",
        "name": "⑨ 解释",
        "summary": "输出特征重要性，回答「模型靠什么做决策」。",
        "input": "run_id",
        "output": "特征重要性排名",
        "key_behaviour": [
            "树模型取 feature_importances_，线性模型取 |coef| 均值。",
            "训练时已落库，解释时优先复用，避免重复反序列化整个模型。",
            "不支持特征重要性的模型返回明确说明，而不是空结果。",
        ],
        "verify": "explain 返回 source（run_artifacts 复用 / model 现算）+ method + importances。",
        "code": "app/ml_engine/explainability.py",
    },
]


# ----------------------------------------------------------------------
# 二、预处理参数说明
#
# 字段约定（与 MODEL_PARAMS 共用同一套规格，供前端渲染控件 + 折叠说明）：
#   name            参数名（点号表示嵌套配置，如 missing.strategy）
#   label           中文名
#   type            int | float | enum | bool | string[] | object
#   default         人类可读默认值（说明用，不可直接回填）
#   range           取值范围；**enum 时为候选数组**，前端据此渲染下拉
#   effect/when_to_change  对结果的影响 / 什么时候该改（折进 ⓘ）
# ----------------------------------------------------------------------
PREPROCESSING_PARAMS: list[dict[str, Any]] = [
    {
        "name": "missing.strategy",
        "label": "缺失值填补策略",
        "type": "enum",
        "default": "median（有数值缺失列时） / mode（仅类别列）",
        "range": ["mean", "median", "mode", "constant"],
        "tunable": True,
        "effect": "决定空值用什么值补齐。median 抗异常值；mean 受极端值影响；mode 用该列出现最多的取值（类别列唯一合理选择）；constant 用固定值（默认 0）。",
        "when_to_change": "数据含极端离群值时优先 median；类别列必须用 mode/constant（数值策略对字符串无效）；缺失本身有含义（如'未填写'）时可用 constant 并保留空值语义。",
        "code": "app/ml_engine/preprocessing.py::MISSING_STRATEGIES",
    },
    {
        "name": "missing.columns",
        "label": "需填补的列（声明用）",
        "type": "string[]",
        "default": "自动检测含空值的列",
        "range": "数据集实际列名",
        "tunable": False,
        "effect": "声明「哪些列需要填补」，用于漏配检测。实际执行时填补按列类型整段生效（数值列、类别列各一段），不会只处理列出的几列。",
        "when_to_change": "一般不必手写：未声明的空值列会被自动补进清单，不会静默漏掉——这个字段的意义是让你能确认「到底哪些列被填补了」。",
        "code": "app/ml_engine/preprocessing.py::default_preprocessing_config",
    },
    {
        "name": "encoding.method",
        "label": "类别编码方式",
        "type": "enum",
        "default": "one_hot",
        "range": ["one_hot", "ordinal"],
        "tunable": True,
        "effect": "one_hot 为每个类别生成一个 0/1 列，不引入虚假大小关系，但列数会随类别数膨胀（高基数类别会让特征数暴涨）；ordinal 用整数编号，列数不变，但会让模型以为类别之间有大小/顺序关系。",
        "when_to_change": "类别取值较少（几十以内）用 one_hot；取值很多（上百）或本身有序（评分等级、星期）用 ordinal，能显著减少特征数、加快训练；KNN / 线性模型对 ordinal 的虚假顺序较敏感。",
        "code": "app/ml_engine/preprocessing.py::ENCODING_METHODS",
    },
    {
        "name": "scaling.method",
        "label": "数值缩放方式",
        "type": "enum",
        "default": "standard",
        "range": ["standard", "min_max"],
        "tunable": True,
        "effect": "standard 把每列变成均值 0、方差 1（对异常值较稳健）；min_max 把每列压到 [0,1]（完全由最小/最大值决定，个别极端值会把其余数据挤在一小段里）。",
        "when_to_change": "KNN / KMeans / 逻辑回归等按距离或梯度优化的模型必须缩放，建议 standard；树模型对缩放不敏感，换它基本没影响；已知数据无极端离群值且需要固定区间时用 min_max。",
        "code": "app/ml_engine/preprocessing.py::SCALING_METHODS",
    },
    {
        "name": "excluded_columns",
        "label": "排除的特征列",
        "type": "string[]",
        "default": "空",
        "range": "数据集实际列名（不得含 target）",
        "tunable": False,
        "effect": "这些列不参与训练；常用来排除 id、主键、高基数标识列。",
        "when_to_change": "当模型把 id 当特征导致指标虚高时，务必排除。",
        "code": "app/experiments/service.py::ExperimentService._execute",
    },
]


# ----------------------------------------------------------------------
# 三、通用训练参数说明
# ----------------------------------------------------------------------
TRAINING_PARAMS: list[dict[str, Any]] = [
    {
        "name": "task",
        "label": "任务类型",
        "type": "enum",
        "default": "自动推断（有目标列时按目标列类型）",
        "range": ["classification", "regression", "clustering"],
        "effect": "决定使用哪类模型与哪套评估指标。",
        "when_to_change": "目标列是类别 → classification；是连续数值 → regression；无目标想分组 → clustering。",
        "code": "app/tools/ml_tools.py::MlDetectTaskTool",
    },
    {
        "name": "target_column",
        "label": "目标列",
        "type": "string",
        "default": "无（聚类任务）",
        "range": "数据集实际列名",
        "effect": "监督学习要预测的那一列；类型决定任务类型。",
        "when_to_change": "务必选择「你希望模型预测的列」，不要选 id。",
        "code": "app/experiments/service.py::ExperimentService.create",
    },
    {
        "name": "test_size",
        "label": "测试集比例",
        "type": "float",
        "default": 0.2,
        "range": "(0, 1)，常用 0.2 ~ 0.3",
        "effect": "划分给测试集的数据比例。越大评估越稳健但训练数据越少；越小训练数据越多但评估波动越大。",
        "when_to_change": "数据量小（<200 行）时可升到 0.3 让评估更稳；数据量大时可降到 0.1。",
        "code": "app/ml_engine/preprocessing.py::PreprocessingPipeline.train_test_split",
    },
    {
        "name": "seed",
        "label": "随机种子",
        "type": "int",
        "default": 42,
        "range": "任意整数",
        "effect": "固定后数据切分与支持 random_state 的模型训练结果完全可复现。",
        "when_to_change": "做对比实验时**必须固定同一个 seed**；想看结果波动时可换不同 seed 观察指标变化。",
        "code": "app/experiments/service.py::ExperimentService._build_model",
    },
    {
        "name": "preprocessing",
        "label": "预处理配置",
        "type": "object",
        "default": "自动按数据画像生成",
        "range": "{missing, encoding, scaling}",
        "effect": "覆盖默认预处理；只传其中一部分时其余部分保持默认。",
        "when_to_change": "当默认策略不适合数据（如类别列应使用 ordinal）时手动指定。",
        "code": "app/ml_engine/preprocessing.py::build_pipeline",
    },
]


# ----------------------------------------------------------------------
# 四、各模型的关键参数（与 MODEL_REGISTRY 的 11 个模型一一对应）
#     只列真实会透传给 sklearn 的常用参数。
# ----------------------------------------------------------------------
# 字段约定（供前端渲染控件、判定依赖、给出「一键应用」的目标值）：
#   name              sklearn 参数名
#   label             中文名
#   type              int | float | enum | bool
#   default           人类可读默认值（说明用）
#   sklearn_default   **真实可回填的默认值**（数字 / 布尔 / 字符串 / None）
#   range             取值范围；**enum 时为候选数组**，前端据此渲染下拉
#   tier              core（常显，最值得先调）| advanced（进阶，折叠后按需展开）
#   step              数字输入的步长（缺省 int=1 / float=0.1）
#   typical           常用候选值，前端给快捷输入提示
#   requires          依赖约束 {其它参数: 期望值}，不满足时前端禁用并说明原因
#   effect/when_to_change  对结果的影响 / 什么时候该改（折进 ⓘ）
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "note": "线性分类基线，可解释性强，输出概率。",
        "params": [
            {
                "name": "max_iter",
                "label": "最大迭代次数",
                "type": "int",
                "default": 1000,
                "sklearn_default": 1000,
                "range": "正整数，常用 200 ~ 5000",
                "tier": "core",
                "step": 100,
                "typical": [200, 500, 1000, 2000, 5000],
                "effect": "优化器最多迭代多少轮。太小会在没收敛时就停下，系数不准（日志里出现 ConvergenceWarning）；太大只是更慢，结果不会更好。",
                "when_to_change": "看到未收敛告警、或同一份数据换 seed 后指标明显抖动时调大。",
            },
            {
                "name": "C",
                "label": "正则强度倒数",
                "type": "float",
                "default": "1.0",
                "sklearn_default": 1.0,
                "range": "正浮点数，常用 0.01 ~ 100",
                "tier": "core",
                "step": 0.1,
                "typical": [0.01, 0.1, 1, 10, 100],
                "effect": "C 越大正则越弱、越贴近训练集（可能过拟合）；C 越小正则越强、模型越保守（可能欠拟合）。它是线性模型里最有效的「过拟合旋钮」。",
                "when_to_change": "训练集很好但测试集差 → 调小 C；两边都差 → 调大 C。",
            },
            {
                "name": "class_weight",
                "label": "类别权重",
                "type": "enum",
                "default": "None",
                "sklearn_default": None,
                "range": "None（每个样本等权） / balanced（按类别样本量反比加权）",
                "options": [
                    {"label": "None（每个样本等权）", "value": None},
                    {"label": "balanced（少数类加权）", "value": "balanced"},
                ],
                "tier": "core",
                "effect": "balanced 按类别样本量反比加权，让少数类获得更高权重，缓解类别不均衡时模型一味偏向多数类；None 不加权（每个样本等权）。",
                "when_to_change": "目标列类别明显不均衡（某类样本远多于其它）时用 balanced；类别大致均衡时用 None。它通常能明显拉高少数类的 recall，代价是多数类 precision 略降。",
            },
            {
                "name": "penalty",
                "label": "正则类型",
                "type": "enum",
                "default": "l2",
                "sklearn_default": "l2",
                "range": "l2（默认） / l1（可归零） / elasticnet（混合） / none（不正则）",
                "options": [
                    {"label": "l2（系数整体收缩，默认）", "value": "l2"},
                    {"label": "l1（把无用特征系数压到 0）", "value": "l1"},
                    {"label": "elasticnet（l1 与 l2 混合）", "value": "elasticnet"},
                    {"label": "none（完全不正则）", "value": "none"},
                ],
                "tier": "advanced",
                "effect": "l2 让系数整体变小；l1 会把不重要的特征系数直接压到 0（自带特征选择效果，适合高维稀疏数据）；elasticnet 是两者混合；None 不正则。",
                "when_to_change": "特征很多、想把无用的直接归零时试 l1；只是怕过拟合用 l2 即可。注意 l1/elasticnet 需要配合 solver（见 solver 说明），否则会直接报错。",
            },
            {
                "name": "solver",
                "label": "优化算法",
                "type": "enum",
                "default": "lbfgs",
                "sklearn_default": "lbfgs",
                "range": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                "tier": "advanced",
                "effect": "决定用什么数值方法求解。lbfgs 是默认且通用的选择；liblinear 只支持 l1/l2 且适合小数据；saga 支持全部正则并适合大数据；sag 速度快但需要标准化。",
                "when_to_change": "选了 l1 正则需要 liblinear 或 saga；选了 elasticnet 只能用 saga；数据量大且已标准化时 saga/sag 更快。改错组合会直接报错，所以这里与 penalty 有联动约束。",
            },
            {
                "name": "l1_ratio",
                "label": "l1 占比（elasticnet）",
                "type": "float",
                "default": "None（仅 elasticnet 生效）",
                "sklearn_default": 0.5,
                "range": "0 ~ 1 的浮点数",
                "tier": "advanced",
                "step": 0.1,
                "typical": [0.1, 0.5, 0.9],
                "requires": {"penalty": "elasticnet"},
                "effect": "在 elasticnet 混合正则里 l1 占多少比重：0 等价纯 l2，1 等价纯 l1。",
                "when_to_change": "只有把 penalty 设为 elasticnet（且 solver=saga）时才有意义，想多剔除特征就调大。",
            },
            {
                "name": "tol",
                "label": "收敛容差",
                "type": "float",
                "default": 0.0001,
                "sklearn_default": 0.0001,
                "range": "正浮点数，常用 1e-3 ~ 1e-5",
                "tier": "advanced",
                "step": 0.0001,
                "effect": "判定「已经收敛」的精度门槛。调小要求更精确（迭代更多、更慢）；调大则更早停下。",
                "when_to_change": "训练太慢且已无未收敛告警时可略微调大；追求数值精度时调小。影响通常远小于 C。",
            },
            {
                "name": "fit_intercept",
                "label": "拟合截距",
                "type": "bool",
                "default": True,
                "sklearn_default": True,
                "range": "true / false",
                "tier": "advanced",
                "effect": "是否给模型一个常数项。关闭后决策边界必须过原点，只适合「特征为 0 时结果必然为 0」的场景。",
                "when_to_change": "业务上确认无截距项时才关掉，否则一般保持开启。",
            },
        ],
    },
    "knn_classifier": {
        "note": "基于距离投票的分类，对特征尺度敏感（建议保留标准化）。",
        "params": [
            {
                "name": "n_neighbors",
                "label": "邻居数量 k",
                "type": "int",
                "default": 5,
                "sklearn_default": 5,
                "range": "正整数，常见 3 ~ 20（且需 < 训练样本数）",
                "tier": "core",
                "typical": [3, 5, 7, 11, 15],
                "effect": "投票时参考几个最近的邻居。k 越小决策边界越贴合局部（易过拟合、对噪声敏感）；k 越大越平滑（易欠拟合，极端情况会把所有点都判成多数类）。",
                "when_to_change": "数据噪声大 → 调大 k；类别边界复杂、样本充足 → 调小 k。它通常是 KNN 唯一需要认真调的参数。",
            },
            {
                "name": "weights",
                "label": "距离权重",
                "type": "enum",
                "default": "uniform",
                "sklearn_default": "uniform",
                "range": ["uniform", "distance"],
                "tier": "core",
                "effect": "uniform 所有邻居等权投票；distance 按距离倒数加权，越近的邻居话语权越大，能提升局部拟合精度，但更易受噪声干扰。",
                "when_to_change": "特征已标准化、局部模式明显时可试 distance；数据噪声大时保持 uniform。",
            },
            {
                "name": "p",
                "label": "距离度量阶数 p",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数，1 = 曼哈顿距离，2 = 欧氏距离",
                "tier": "core",
                "typical": [1, 2],
                "effect": "决定「多远算远」。p=2 是欧氏距离（默认）；p=1 是曼哈顿距离（各维度差值绝对值相加），对单个维度的极端差异更不敏感，在高维或含离群维度时可能更稳。",
                "when_to_change": "欧氏距离效果一般、特征维度多且相互独立时可试 p=1。",
            },
            {
                "name": "metric",
                "label": "距离度量",
                "type": "enum",
                "default": "minkowski",
                "sklearn_default": "minkowski",
                "range": ["minkowski", "euclidean", "manhattan", "chebyshev"],
                "tier": "advanced",
                "effect": "具体的距离定义。minkowski 是通用形式（由参数 p 决定具体距离）；euclidean/manhattan/chebyshev 是直接指定，语义等价于固定 p 的 minkowski。",
                "when_to_change": "一般不用改：想改距离用 p 更直观。设了 metric 后 p 会被忽略。",
            },
            {
                "name": "algorithm",
                "label": "近邻搜索算法",
                "type": "enum",
                "default": "auto",
                "sklearn_default": "auto",
                "range": ["auto", "ball_tree", "kd_tree", "brute"],
                "tier": "advanced",
                "effect": "只是「怎么找邻居」的实现方式，不改变预测结果，只影响速度。auto 会自动挑；brute 是暴力全量距离计算；kd_tree/ball_tree 建树后查询更快。",
                "when_to_change": "样本量大且训练/推理慢时可试 kd_tree 或 ball_tree；维度很高（几百维）时树结构失效，brute 反而更可靠。",
            },
            {
                "name": "leaf_size",
                "label": "树叶子大小",
                "type": "int",
                "default": 30,
                "sklearn_default": 30,
                "range": "正整数，常用 10 ~ 60",
                "tier": "advanced",
                "effect": "仅在使用 kd_tree / ball_tree 时影响建树与查询的平衡点，不改变预测结果。",
                "when_to_change": "只有明显觉得建树慢或查询慢时才微调，对精度无影响。",
            },
        ],
    },
    "decision_tree_classifier": {
        "note": "树形规则，可解释性极强，容易过拟合。",
        "params": [
            {
                "name": "max_depth",
                "label": "最大深度",
                "type": "int",
                "default": "None（不限，长到纯节点为止）",
                "sklearn_default": None,
                "range": "正整数，常用 3 ~ 20",
                "tier": "core",
                "typical": [3, 5, 8, 12, 20],
                "effect": "限制树的层数，是单棵树最有效的抗过拟合旋钮。不限深度会一直分裂到每个叶子都纯净，训练集接近 100% 而测试集明显差。",
                "when_to_change": "训练集指标远好于测试集 → 调小；两边都差、树显得太浅 → 放宽。",
            },
            {
                "name": "min_samples_leaf",
                "label": "叶节点最小样本数",
                "type": "int",
                "default": 1,
                "sklearn_default": 1,
                "range": "正整数，常用 1 ~ 20",
                "tier": "core",
                "typical": [1, 2, 5, 10, 20],
                "effect": "每个叶子至少保留多少样本，越大树越平滑、越抗噪；太大则学不到细节（欠拟合）。",
                "when_to_change": "样本量小、标签有噪声、或出现「某个叶子只有 1~2 个样本却决定了预测」时调大。",
            },
            {
                "name": "min_samples_split",
                "label": "分裂所需最小样本数",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20",
                "tier": "core",
                "effect": "一个节点至少要多少样本才继续分裂，越大树越浅、越抗过拟合。",
                "when_to_change": "与 min_samples_leaf 配合使用，过拟合时一起调大；通常先动 max_depth 和 min_samples_leaf。",
            },
            {
                "name": "criterion",
                "label": "分裂准则",
                "type": "enum",
                "default": "gini",
                "sklearn_default": "gini",
                "range": ["gini", "entropy", "log_loss"],
                "tier": "core",
                "effect": "每次分裂时用哪个指标挑最优切分点。gini 计算快、最常用；entropy / log_loss 用信息增益（两者数学上仅差常数），对更细微的类别差异可能更敏感。",
                "when_to_change": "三者效果通常接近，gini 不佳时可试 entropy；不必反复来回试，收益一般很小。",
            },
            {
                "name": "max_features",
                "label": "每次分裂考虑的特征数",
                "type": "enum",
                "default": "None（全部特征）",
                "sklearn_default": None,
                "range": "None（全部） / sqrt（特征数开方） / log2（特征数取对数）",
                "options": [
                    {"label": "None（全部特征，默认）", "value": None},
                    {"label": "sqrt（特征数开方）", "value": "sqrt"},
                    {"label": "log2（特征数取对数）", "value": "log2"},
                ],
                "tier": "advanced",
                "effect": "每次分裂只从随机抽出的部分特征里选最佳切分。限制它能降低过拟合（树之间/节点之间更不相似），但可能欠拟合。",
                "when_to_change": "特征很多且高度相关、单棵树过拟合明显时试 sqrt / log2。",
            },
            {
                "name": "max_leaf_nodes",
                "label": "最大叶子数",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 8 ~ 64",
                "tier": "advanced",
                "typical": [8, 16, 32, 64],
                "effect": "限制树最多长出几个叶子（相当于限制模型复杂度），与 max_depth 二选一使用效果类似。",
                "when_to_change": "想按「模型最多输出几条规则」来控制复杂度时用它替代 max_depth。",
            },
            {
                "name": "ccp_alpha",
                "label": "后剪枝强度",
                "type": "float",
                "default": "0.0（不剪枝）",
                "sklearn_default": 0.0,
                "range": "0 ~ 1 的浮点数，常用 0.0 ~ 0.01",
                "tier": "advanced",
                "step": 0.001,
                "effect": "训练完再按代价复杂度剪掉「性价比低」的分支。0 表示不剪；稍大于 0 就能明显压缩树规模并缓解过拟合。",
                "when_to_change": "树很庞大、叶子很多、测试集不佳时从 0.001 起逐步试。它是比 max_depth 更精细的抗过拟合手段。",
            },
            {
                "name": "splitter",
                "label": "分裂点选择方式",
                "type": "enum",
                "default": "best",
                "sklearn_default": "best",
                "range": ["best", "random"],
                "tier": "advanced",
                "effect": "best 在每个特征的所有切分点里选最优；random 随机选一个切分点，训练更快、树更多样（单棵树情况下通常更差，但可当随机化正则）。",
                "when_to_change": "特征维度极高、训练太慢时可试 random，其余情况保持 best。",
            },
            {
                "name": "min_impurity_decrease",
                "label": "分裂最小收益",
                "type": "float",
                "default": 0.0,
                "sklearn_default": 0.0,
                "range": "0 ~ 1 的浮点数",
                "tier": "advanced",
                "step": 0.0001,
                "effect": "只有当分裂带来的不纯度下降超过该阈值才允许分裂，相当于给树加了一道「收益门槛」。",
                "when_to_change": "想让模型只保留有实际价值的规则时略微调大（如 0.0001 ~ 0.001），调太大树会退化成几个叶子。",
            },
        ],
    },
    "random_forest_classifier": {
        "note": "多棵树集成，抗过拟合，通常是最稳的默认选择。",
        "params": [
            {
                "name": "n_estimators",
                "label": "树的数量",
                "type": "int",
                "default": 100,
                "sklearn_default": 100,
                "range": "正整数，常用 50 ~ 500",
                "tier": "core",
                "step": 50,
                "typical": [50, 100, 200, 300, 500],
                "effect": "树越多结果越稳定（指标方差变小），但训练/推理耗时近似线性增加。超过一定数量后指标基本不再提升，只是变慢。",
                "when_to_change": "同一配置换 seed 后指标波动大 → 加到 200~300；只想快速验证思路 → 降到 30~50。",
            },
            {
                "name": "max_depth",
                "label": "单树最大深度",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 3 ~ 30",
                "tier": "core",
                "typical": [4, 6, 10, 16, 30],
                "effect": "控制每棵树的复杂度。集成本身有兜底，这里的限制比单棵树宽松：不限深度通常也能用，限制后训练更快、更抗过拟合。",
                "when_to_change": "训练集明显好于测试集 → 调小；树太浅导致两边都差 → 放宽。",
            },
            {
                "name": "min_samples_leaf",
                "label": "叶节点最小样本数",
                "type": "int",
                "default": 1,
                "sklearn_default": 1,
                "range": "正整数，常用 1 ~ 20",
                "tier": "core",
                "typical": [1, 2, 4, 8, 16],
                "effect": "每个叶子至少保留多少样本，越大单棵树越平滑、越抗噪。对随机森林而言，它常比 max_depth 更平滑地控制过拟合。",
                "when_to_change": "过拟合（训练好、测试差）时调大；样本量本来就少时优先动它而不是 max_depth。",
            },
            {
                "name": "max_features",
                "label": "每棵树随机特征数",
                "type": "enum",
                "default": "sqrt",
                "sklearn_default": "sqrt",
                "range": "sqrt（特征数开方，默认） / log2（取对数，更随机） / None（全部特征）",
                "options": [
                    {"label": "sqrt（特征数开方，默认）", "value": "sqrt"},
                    {"label": "log2（取对数，随机性更强）", "value": "log2"},
                    {"label": "None（每棵树用全部特征）", "value": None},
                ],
                "tier": "core",
                "effect": "每棵树分裂时随机考虑多少特征：sqrt 取特征数开方（分类默认），log2 取对数，None 用全部特征。越小树之间差异越大、越抗过拟合，但单棵树更弱、可能欠拟合。",
                "when_to_change": "特征很多且高度相关、想更抗过拟合 → log2 或更小；特征本来就少、欠拟合 → None。它是森林里除树数外最值得试的一项。",
            },
            {
                "name": "class_weight",
                "label": "类别权重",
                "type": "enum",
                "default": "None",
                "sklearn_default": None,
                "range": "None（每个样本等权） / balanced（按类别样本量反比加权）",
                "options": [
                    {"label": "None（每个样本等权）", "value": None},
                    {"label": "balanced（少数类加权）", "value": "balanced"},
                ],
                "tier": "advanced",
                "effect": "balanced 按类别样本量反比加权，让少数类获得更高权重，适合类别不均衡场景；None 每个样本等权。",
                "when_to_change": "目标列类别不均衡、少数类 recall 很低时改用 balanced，通常能明显提升少数类表现。",
            },
            {
                "name": "min_samples_split",
                "label": "分裂所需最小样本数",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20",
                "tier": "advanced",
                "effect": "节点样本数不足该值就不再分裂，越大树越浅。",
                "when_to_change": "与 min_samples_leaf 配合，过拟合时一起调大。",
            },
            {
                "name": "bootstrap",
                "label": "自助采样",
                "type": "bool",
                "default": True,
                "sklearn_default": True,
                "range": "true / false",
                "tier": "advanced",
                "effect": "是否让每棵树只在一份「有放回随机抽样」的子集上训练。这是随机森林「随机」的主要来源之一；关掉后所有树看到同一份数据，多样性下降、更容易一起过拟合。",
                "when_to_change": "一般保持开启。只有想用全部数据训练每棵树且明显欠拟合时才关掉（关掉会同时让 oob_score / max_samples 失效）。",
            },
            {
                "name": "oob_score",
                "label": "袋外评分",
                "type": "bool",
                "default": False,
                "sklearn_default": False,
                "range": "true / false",
                "tier": "advanced",
                "requires": {"bootstrap": True},
                "effect": "用每棵树没抽到的样本来估一个「不占用测试集」的泛化指标。它不改变模型本身的预测，只是多给一个参考分数。",
                "when_to_change": "数据量小、想多一个独立于测试集的评估视角时开启。注意它需要 bootstrap=True。",
            },
            {
                "name": "max_samples",
                "label": "每棵树采样比例",
                "type": "float",
                "default": "None（等于全部训练样本）",
                "sklearn_default": None,
                "range": "0 ~ 1 的浮点数",
                "tier": "advanced",
                "step": 0.1,
                "requires": {"bootstrap": True},
                "effect": "每棵树的抽样量占训练集的比例。调小 → 树之间差异更大、训练更快，但单棵树更弱。",
                "when_to_change": "训练集很大、训练太慢时设 0.5~0.8 可明显提速；数据量小时保持默认。需 bootstrap=True。",
            },
            {
                "name": "max_leaf_nodes",
                "label": "单树最大叶子数",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 16 ~ 256",
                "tier": "advanced",
                "effect": "限制每棵树最多长出几个叶子，与 max_depth 作用方向一致（按叶子数而非层数控制复杂度）。",
                "when_to_change": "想按「模型最多输出多少条规则」控制复杂度时使用。",
            },
            {
                "name": "ccp_alpha",
                "label": "后剪枝强度",
                "type": "float",
                "default": "0.0（不剪枝）",
                "sklearn_default": 0.0,
                "range": "0 ~ 1 的浮点数，常用 0.0 ~ 0.01",
                "tier": "advanced",
                "step": 0.001,
                "effect": "训练后按代价复杂度剪掉低收益分支，能压缩模型并缓解过拟合。",
                "when_to_change": "树数量多、模型体积大且测试集不佳时从 0.001 起逐步试。",
            },
        ],
    },
    "linear_regression": {
        "note": "假设特征与目标线性相关的最简回归，几乎无超参（seed 不适用）。",
        "params": [
            {
                "name": "fit_intercept",
                "label": "拟合截距",
                "type": "bool",
                "default": True,
                "sklearn_default": True,
                "range": "true / false",
                "tier": "core",
                "effect": "是否给模型一个常数项。关闭后回归面必须过原点，只适合「特征全为 0 时目标必然为 0」的场景。",
                "when_to_change": "业务上确认无截距项时才关掉。注意：线性回归**没有正则化参数**，如果它在测试集上明显过拟合，正确做法是换随机森林或改用带正则的模型，而不是调这里。",
            },
            {
                "name": "positive",
                "label": "系数限定为正",
                "type": "bool",
                "default": False,
                "sklearn_default": False,
                "range": "true / false",
                "tier": "advanced",
                "effect": "强制所有回归系数 ≥ 0，即「特征越大预测值只增不减」。",
                "when_to_change": "业务上明确所有特征都与目标正相关时开启，可避免出现反直觉的负系数；否则会引入偏差。",
            },
        ],
    },
    "knn_regressor": {
        "note": "基于邻居均值回归，对尺度敏感。",
        "params": [
            {
                "name": "n_neighbors",
                "label": "邻居数量 k",
                "type": "int",
                "default": 5,
                "sklearn_default": 5,
                "range": "正整数，需 < 训练样本数",
                "tier": "core",
                "typical": [3, 5, 7, 11, 15],
                "effect": "用最近几个邻居的平均值作为预测。k 小 → 贴合局部细节但噪声敏感；k 大 → 曲线平滑，极端情况会退化成「全数据均值预测」。",
                "when_to_change": "预测曲线抖动厉害、对大误差敏感（RMSE 高）→ 调大 k；曲线过度平滑、抓不到局部规律 → 调小。",
            },
            {
                "name": "weights",
                "label": "距离权重",
                "type": "enum",
                "default": "uniform",
                "sklearn_default": "uniform",
                "range": ["uniform", "distance"],
                "tier": "core",
                "effect": "uniform 邻居等权平均；distance 按距离倒数加权，近邻贡献更大，通常能降低预测偏差，但更易受噪声干扰。",
                "when_to_change": "局部趋势明显时可试 distance；噪声大时保持 uniform。",
            },
            {
                "name": "p",
                "label": "距离度量阶数 p",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数，1 = 曼哈顿距离，2 = 欧氏距离",
                "tier": "advanced",
                "typical": [1, 2],
                "effect": "决定「多远算远」。p=2 为欧氏距离（默认）；p=1 为曼哈顿距离，对单个维度的极端差异不敏感。",
                "when_to_change": "高维或含离群维度时可试 p=1。",
            },
            {
                "name": "algorithm",
                "label": "近邻搜索算法",
                "type": "enum",
                "default": "auto",
                "sklearn_default": "auto",
                "range": ["auto", "ball_tree", "kd_tree", "brute"],
                "tier": "advanced",
                "effect": "只影响「怎么找邻居」的速度，不改变预测结果。",
                "when_to_change": "样本量大且训练/推理慢时可试 kd_tree；维度很高（几百维）时 brute 反而更可靠。",
            },
            {
                "name": "leaf_size",
                "label": "树叶子大小",
                "type": "int",
                "default": 30,
                "sklearn_default": 30,
                "range": "正整数，常用 10 ~ 60",
                "tier": "advanced",
                "effect": "仅在使用 kd_tree / ball_tree 时影响性能，不改变结果。",
                "when_to_change": "只有明显觉得慢时才微调，对精度无影响。",
            },
        ],
    },
    "decision_tree_regressor": {
        "note": "树形回归，可捕捉非线性，容易过拟合。",
        "params": [
            {
                "name": "max_depth",
                "label": "最大深度",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 3 ~ 20",
                "tier": "core",
                "typical": [3, 5, 8, 12, 20],
                "effect": "限制树的层数，是单棵树最有效的抗过拟合旋钮。越深越贴合训练集，越浅越平滑。",
                "when_to_change": "训练集 R² 远高于测试集 → 调小；两边都差、树太浅 → 放宽。",
            },
            {
                "name": "min_samples_leaf",
                "label": "叶节点最小样本数",
                "type": "int",
                "default": 1,
                "sklearn_default": 1,
                "range": "正整数，常用 1 ~ 20",
                "tier": "core",
                "typical": [1, 2, 5, 10, 20],
                "effect": "每个叶子至少保留多少样本，越大树越平滑、抗噪。对回归树尤其重要——叶子样本太少时，叶子值会被个别极端样本带偏。",
                "when_to_change": "过拟合、或预测值出现不合理的极端值（残差 max_abs 很大）时调大。",
            },
            {
                "name": "min_samples_split",
                "label": "分裂所需最小样本数",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20",
                "tier": "core",
                "effect": "节点样本数不足该值就不再分裂，越大树越浅。",
                "when_to_change": "过拟合时与 min_samples_leaf 一起调大。",
            },
            {
                "name": "criterion",
                "label": "分裂准则",
                "type": "enum",
                "default": "squared_error",
                "sklearn_default": "squared_error",
                "range": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                "tier": "core",
                "effect": "squared_error 最小化均方误差（最常用，对大误差惩罚重）；absolute_error 用绝对误差，对离群点稳健得多；friedman_mse 面向梯度提升场景；poisson 适合计数型目标。",
                "when_to_change": "目标含明显离群点、或 MAE 与 RMSE 差距很大时试 absolute_error；目标是计数（如订单量）可试 poisson。",
            },
            {
                "name": "max_features",
                "label": "每次分裂考虑的特征数",
                "type": "enum",
                "default": "None（全部特征）",
                "sklearn_default": None,
                "range": "None（全部） / sqrt（特征数开方） / log2（特征数取对数）",
                "options": [
                    {"label": "None（全部特征，默认）", "value": None},
                    {"label": "sqrt（特征数开方）", "value": "sqrt"},
                    {"label": "log2（特征数取对数）", "value": "log2"},
                ],
                "tier": "advanced",
                "effect": "每次分裂只从随机抽出的部分特征里选最佳切分，能降低过拟合。",
                "when_to_change": "特征多且相关、单棵树过拟合明显时试 sqrt / log2。",
            },
            {
                "name": "max_leaf_nodes",
                "label": "最大叶子数",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 8 ~ 64",
                "tier": "advanced",
                "typical": [8, 16, 32, 64],
                "effect": "限制树最多长出几个叶子，与 max_depth 二选一使用。",
                "when_to_change": "想按「模型最多输出几条规则」控制复杂度时用它替代 max_depth。",
            },
            {
                "name": "ccp_alpha",
                "label": "后剪枝强度",
                "type": "float",
                "default": "0.0（不剪枝）",
                "sklearn_default": 0.0,
                "range": "0 ~ 1 的浮点数，常用 0.0 ~ 0.01",
                "tier": "advanced",
                "step": 0.001,
                "effect": "训练后剪掉低收益分支，能压缩树规模并缓解过拟合。",
                "when_to_change": "树很庞大、测试集 R² 明显低于训练集时从 0.001 起逐步试。",
            },
        ],
    },
    "random_forest_regressor": {
        "note": "集成回归，非线性场景的稳健默认选择。",
        "params": [
            {
                "name": "n_estimators",
                "label": "树的数量",
                "type": "int",
                "default": 100,
                "sklearn_default": 100,
                "range": "常用 50 ~ 500",
                "tier": "core",
                "step": 50,
                "typical": [50, 100, 200, 300, 500],
                "effect": "越多越稳（预测方差更小）、越慢。",
                "when_to_change": "追求稳定 → 200+；快速试跑 → 30~50。",
            },
            {
                "name": "max_depth",
                "label": "单树最大深度",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "3 ~ 30",
                "tier": "core",
                "typical": [4, 6, 10, 16, 30],
                "effect": "控制每棵树的复杂度，越小越平滑。",
                "when_to_change": "过拟合（训练 R² 高、测试 R² 低）时调小。",
            },
            {
                "name": "min_samples_leaf",
                "label": "叶节点最小样本数",
                "type": "int",
                "default": 1,
                "sklearn_default": 1,
                "range": "正整数，常用 1 ~ 20",
                "tier": "core",
                "typical": [1, 2, 4, 8, 16],
                "effect": "每个叶子至少保留多少样本，越大越平滑、越抗过拟合。回归场景下它对「个别极端样本带偏预测」的抑制效果很明显。",
                "when_to_change": "训练 R² 高、测试 R² 低时调大；这也是降低 RMSE 尾部的常用手段。",
            },
            {
                "name": "max_features",
                "label": "每棵树随机特征数",
                "type": "enum",
                "default": "1.0（回归默认用全部特征）",
                "sklearn_default": 1.0,
                "range": "1.0（全部特征，默认） / 0.5（一半） / sqrt（开方） / log2（取对数）",
                "options": [
                    {"label": "1.0（每棵树用全部特征，默认）", "value": 1.0},
                    {"label": "0.5（每棵树随机用一半特征）", "value": 0.5},
                    {"label": "sqrt（特征数开方）", "value": "sqrt"},
                    {"label": "log2（特征数取对数）", "value": "log2"},
                ],
                "tier": "core",
                "effect": "每棵树分裂时随机考虑多少特征。回归版 sklearn 默认用全部特征；调小能让树之间更不同、更抗过拟合（相当于更强的随机化），但单棵树更弱。",
                "when_to_change": "特征多且相关、过拟合时试 sqrt 或 0.5；欠拟合时保持 1.0。",
            },
            {
                "name": "min_samples_split",
                "label": "分裂所需最小样本数",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20",
                "tier": "advanced",
                "effect": "节点样本数不足该值就不再分裂，越大树越浅。",
                "when_to_change": "过拟合时与 min_samples_leaf 一起调大。",
            },
            {
                "name": "bootstrap",
                "label": "自助采样",
                "type": "bool",
                "default": True,
                "sklearn_default": True,
                "range": "true / false",
                "tier": "advanced",
                "effect": "是否让每棵树只在一份有放回抽样的子集上训练，是森林「随机性」的主要来源。",
                "when_to_change": "一般保持开启；关掉后 oob_score / max_samples 失效。",
            },
            {
                "name": "oob_score",
                "label": "袋外评分",
                "type": "bool",
                "default": False,
                "sklearn_default": False,
                "range": "true / false",
                "tier": "advanced",
                "requires": {"bootstrap": True},
                "effect": "用每棵树没抽到的样本估一个不占用测试集的泛化分数，不改变预测结果。",
                "when_to_change": "数据量小、想多一个参考视角时开启。需 bootstrap=True。",
            },
            {
                "name": "max_samples",
                "label": "每棵树采样比例",
                "type": "float",
                "default": "None（等于全部训练样本）",
                "sklearn_default": None,
                "range": "0 ~ 1 的浮点数",
                "tier": "advanced",
                "step": 0.1,
                "requires": {"bootstrap": True},
                "effect": "每棵树用多少比例的训练样本。调小 → 更快、树之间差异更大，但单棵树更弱。",
                "when_to_change": "训练集很大、训练慢时设 0.5~0.8 提速。需 bootstrap=True。",
            },
            {
                "name": "max_leaf_nodes",
                "label": "单树最大叶子数",
                "type": "int",
                "default": "None（不限）",
                "sklearn_default": None,
                "range": "正整数，常用 16 ~ 256",
                "tier": "advanced",
                "effect": "限制每棵树最多几个叶子，按叶子数控制复杂度。",
                "when_to_change": "想按规则条数控制模型规模时使用。",
            },
            {
                "name": "ccp_alpha",
                "label": "后剪枝强度",
                "type": "float",
                "default": "0.0（不剪枝）",
                "sklearn_default": 0.0,
                "range": "0 ~ 1 的浮点数，常用 0.0 ~ 0.01",
                "tier": "advanced",
                "step": 0.001,
                "effect": "训练后剪掉低收益分支，压缩模型规模。",
                "when_to_change": "模型体积大且测试集不佳时从 0.001 起逐步试。",
            },
        ],
    },
    "kmeans": {
        "note": "划分式聚类，需指定簇数，适合近似球形、规模相近的簇。",
        "params": [
            {
                "name": "n_clusters",
                "label": "簇数量 k",
                "type": "int",
                "default": 8,
                "sklearn_default": 8,
                "range": "2 ~ min(样本数, 合理上限)，常用 2 ~ 10",
                "tier": "core",
                "typical": [2, 3, 4, 5, 8, 10],
                "effect": "要分成几类，是聚类里唯一真正决定结果的参数。k 太大会把本应同类的样本拆开（轮廓系数通常下降）；太小会把不同类混在一起。",
                "when_to_change": "先用业务含义定一个候选范围，再对比不同 k 的轮廓系数：轮廓系数高且簇规模不畸形的那档通常更合理。注意轮廓系数会偏向 k=2，不能只看它。",
            },
            {
                "name": "init",
                "label": "初始质心选法",
                "type": "enum",
                "default": "k-means++",
                "sklearn_default": "k-means++",
                "range": "k-means++（分散初始化，默认） / random（随机抽点）",
                "options": [
                    {"label": "k-means++（质心尽量分散，默认）", "value": "k-means++"},
                    {"label": "random（随机抽取样本点）", "value": "random"},
                ],
                "tier": "core",
                "effect": "k-means++ 让初始质心彼此尽量远，通常收敛更快、结果更好；random 是原始做法，容易陷入较差的局部最优。",
                "when_to_change": "基本不用改成 random。若想复现「同一数据不同 seed 结果差异大」的现象做对比实验，可以切成 random。",
            },
            {
                "name": "n_init",
                "label": "初始化次数",
                "type": "int",
                "default": "auto（sklearn 默认，等价于 10）",
                "sklearn_default": 10,
                "range": "正整数，常用 10 ~ 20",
                "tier": "core",
                "typical": [1, 10, 20],
                "effect": "算法用多少组不同的初始质心各跑一遍，最后取最好的一组。越大结果越稳定但更慢。",
                "when_to_change": "同一数据换 seed 后簇划分明显不同（结果不稳定）时调大。",
            },
            {
                "name": "max_iter",
                "label": "单次最大迭代次数",
                "type": "int",
                "default": 300,
                "sklearn_default": 300,
                "range": "正整数，常用 100 ~ 1000",
                "tier": "advanced",
                "step": 50,
                "effect": "一组初始质心最多迭代多少轮。必要时可通过此值给「未收敛」一个显式上界（超过则告警）。",
                "when_to_change": "样本量很大、日志出现未收敛告警时调大。",
            },
            {
                "name": "tol",
                "label": "收敛容差",
                "type": "float",
                "default": 0.0001,
                "sklearn_default": 0.0001,
                "range": "正浮点数",
                "tier": "advanced",
                "step": 0.0001,
                "effect": "质心移动幅度小于该值时判定收敛。调小更精确但更慢。",
                "when_to_change": "一般不用改。",
            },
            {
                "name": "algorithm",
                "label": "实现算法",
                "type": "enum",
                "default": "lloyd",
                "sklearn_default": "lloyd",
                "range": "lloyd（经典，通用） / elkan（三角不等式加速）",
                "options": [
                    {"label": "lloyd（经典实现，默认）", "value": "lloyd"},
                    {"label": "elkan（加速版，需球形簇假设）", "value": "elkan"},
                ],
                "tier": "advanced",
                "effect": "只影响速度，不改变目标函数。elkan 在簇结构较清晰、数据量中等时更快。",
                "when_to_change": "数据量大且训练慢时可试 elkan（注意它不支持稀疏数据）。",
            },
        ],
    },
    "dbscan": {
        "note": "密度聚类，自动发现簇数，可识别噪声点（标签 -1），不支持对新样本预测。",
        "params": [
            {
                "name": "eps",
                "label": "邻域半径",
                "type": "float",
                "default": 0.5,
                "sklearn_default": 0.5,
                "range": "正浮点数（**单位是标准化后的尺度**，因此通常在 0.1 ~ 2 之间）",
                "tier": "core",
                "step": 0.05,
                "typical": [0.2, 0.3, 0.5, 0.8, 1.0],
                "effect": "判断「两个点算不算邻居」的距离阈值，是 DBSCAN 最关键也最难定的参数。太大 → 所有点合并成一个大簇；太小 → 大量点被判为噪声（标签 -1）。",
                "when_to_change": "结果里噪声点特别多 → 逐步调大 eps；所有点挤成一簇（簇数=1）→ 调小。建议以 0.05 为步长逐档试并观察簇数与噪声比例。",
            },
            {
                "name": "min_samples",
                "label": "核心点最小邻居数",
                "type": "int",
                "default": 5,
                "sklearn_default": 5,
                "range": "正整数，常见 3 ~ 20",
                "tier": "core",
                "typical": [3, 5, 8, 10, 20],
                "effect": "一个点周围至少要多少邻居才算「核心点」。越大越严格，越容易把稀疏区域的点判为噪声；越小越容易合并成大片。",
                "when_to_change": "噪声过多 → 调小；簇过于碎（很多小簇）→ 调大。经验上可取「特征数 + 1」作为起点。",
            },
            {
                "name": "metric",
                "label": "距离度量",
                "type": "enum",
                "default": "euclidean",
                "sklearn_default": "euclidean",
                "range": "euclidean（欧氏，默认） / manhattan（曼哈顿） / chebyshev（切比雪夫）",
                "options": [
                    {"label": "euclidean（欧氏距离，默认）", "value": "euclidean"},
                    {"label": "manhattan（曼哈顿距离）", "value": "manhattan"},
                    {"label": "chebyshev（各维度最大差值）", "value": "chebyshev"},
                ],
                "tier": "advanced",
                "effect": "邻域半径 eps 是相对于该度量定义的球体，改度量相当于换了一套「距离尺度」。",
                "when_to_change": "改成 manhattan 后需要重新试 eps（原 eps 往往不再合适）。",
            },
            {
                "name": "algorithm",
                "label": "近邻搜索算法",
                "type": "enum",
                "default": "auto",
                "sklearn_default": "auto",
                "range": "auto（自动） / ball_tree / kd_tree / brute",
                "options": [
                    {"label": "auto（自动选择，默认）", "value": "auto"},
                    {"label": "ball_tree", "value": "ball_tree"},
                    {"label": "kd_tree", "value": "kd_tree"},
                    {"label": "brute（暴力计算）", "value": "brute"},
                ],
                "tier": "advanced",
                "effect": "只影响速度，不改变聚类结果。",
                "when_to_change": "样本量大且慢时可试树结构；维度很高时 brute 更可靠。",
            },
            {
                "name": "leaf_size",
                "label": "树叶子大小",
                "type": "int",
                "default": 30,
                "sklearn_default": 30,
                "range": "正整数，常用 10 ~ 60",
                "tier": "advanced",
                "effect": "仅在使用树结构算法时影响性能，不改变结果。",
                "when_to_change": "只影响速度，对聚类质量无影响。",
            },
        ],
    },
    "pca": {
        "note": "主成分分析，降维与可视化，不预测标签。",
        "params": [
            {
                "name": "n_components",
                "label": "主成分数量",
                "type": "int",
                "default": 2,
                "sklearn_default": 2,
                "range": "1 ~ min(样本数, 特征数)",
                "tier": "core",
                "typical": [2, 3, 5, 10],
                "effect": "保留多少个主成分。取 2 可直接二维可视化；取值越大保留信息越多但降维意义越弱（取到特征数即等于没降维）。",
                "when_to_change": "看 explained_variance_ratio 的累计值：达到 85% 以上通常认为保留充分。若只是为了画图，保持 2。",
            },
            {
                "name": "whiten",
                "label": "白化",
                "type": "bool",
                "default": False,
                "sklearn_default": False,
                "range": "true / false",
                "tier": "core",
                "effect": "开启后把各主成分除以自己的标准差，使输出各维度方差一致。对「看主成分重要性」是干扰（方差信息被抹掉），但作为后续模型的输入特征时有时更稳定。",
                "when_to_change": "把 PCA 当作下游模型的预处理特征时（如配合聚类/KNN）可试开启；只为解释方差与画图则保持关闭。",
            },
            {
                "name": "svd_solver",
                "label": "特征分解算法",
                "type": "enum",
                "default": "auto",
                "sklearn_default": "auto",
                "range": "auto（自动） / full（精确，慢） / randomized（随机，快） / arpack（稀疏场景）",
                "options": [
                    {"label": "auto（自动选择，默认）", "value": "auto"},
                    {"label": "full（精确分解，慢但最稳）", "value": "full"},
                    {"label": "randomized（近似快速分解）", "value": "randomized"},
                    {"label": "arpack（大数据稀疏场景）", "value": "arpack"},
                ],
                "tier": "advanced",
                "effect": "不同的数值分解实现，结果理论上一致，差别在速度与数值精度。randomized 用随机近似，大数据下快很多。",
                "when_to_change": "数据量大、PCA 很慢时可试 randomized；结果出现细微不一致或要做精确对比时用 full。",
            },
            {
                "name": "tol",
                "label": "收敛容差",
                "type": "float",
                "default": 0.0,
                "sklearn_default": 0.0,
                "range": "0 或正浮点数",
                "tier": "advanced",
                "step": 0.0001,
                "effect": "仅对 arpack / randomized 求解器有意义，控制迭代停止的精度。0 表示用求解器默认值。",
                "when_to_change": "一般不用改。",
            },
        ],
    },
}


# ----------------------------------------------------------------------
# 五、指标解读（教学衔接：指标是什么、多少算好、看哪个更合适）
# ----------------------------------------------------------------------
METRIC_GUIDE: dict[str, dict[str, str]] = {
    "accuracy": {
        "label": "准确率",
        "task": "classification",
        "meaning": "预测正确的样本占比。",
        "how_to_read": "类别均衡时直观好用；**类别极不均衡时会严重虚高**（全猜多数类也能很高）。",
        "better": "越高越好，接近 1",
    },
    "precision": {
        "label": "精确率（macro）",
        "task": "classification",
        "meaning": "预测为某类中，真正属于该类的比例，各类别取平均。",
        "how_to_read": "关心'误报'代价时看它（如把正常判为异常）。",
        "better": "越高越好",
    },
    "recall": {
        "label": "召回率（macro）",
        "task": "classification",
        "meaning": "真正属于某类的样本中，被成功找出的比例，各类别取平均。",
        "how_to_read": "关心'漏报'代价时看它（如漏掉真正的高风险用户）。",
        "better": "越高越好",
    },
    "f1": {
        "label": "F1（macro）",
        "task": "classification",
        "meaning": "precision 与 recall 的调和平均，兼顾两者。",
        "how_to_read": "两指标冲突时看 F1 折中；类别不均衡时比 accuracy 更可信。",
        "better": "越高越好",
    },
    "roc_auc": {
        "label": "ROC-AUC",
        "task": "classification",
        "meaning": "模型把正类排在负类前面的能力，与阈值无关。",
        "how_to_read": "0.5=随机猜，0.7 尚可，0.8+ 良好，0.9+ 优秀；需模型支持概率输出。",
        "better": "越高越好，0.5 为无信息基线",
    },
    "mae": {
        "label": "平均绝对误差",
        "task": "regression",
        "meaning": "预测值与真实值差的绝对值的平均，单位与目标列一致。",
        "how_to_read": "最直观：'平均差多少个单位'。",
        "better": "越低越好",
    },
    "rmse": {
        "label": "均方根误差",
        "task": "regression",
        "meaning": "误差平方平均后开方，单位与目标列一致。",
        "how_to_read": "对大误差惩罚更重，比 MAE 更能反映'偶尔差很多'。",
        "better": "越低越好",
    },
    "r2": {
        "label": "决定系数 R²",
        "task": "regression",
        "meaning": "模型解释了目标方差的多少比例（相对'只用均值预测'的改进）。",
        "how_to_read": "1=完美，0=与直接猜均值一样，**负数表示还不如猜均值**。",
        "better": "越高越好，>0 才算有信息",
    },
    "cluster_count": {
        "label": "簇数量",
        "task": "clustering",
        "meaning": "算法划分出的簇个数。",
        "how_to_read": "看是否与业务上预期的分组数接近。",
        "better": "无绝对好坏，需结合业务判断",
    },
    "silhouette": {
        "label": "轮廓系数",
        "task": "clustering",
        "meaning": "衡量簇内紧密、簇间分离程度的综合分。",
        "how_to_read": "-1~1：>0.5 结构清晰，0.25~0.5 一般，<0.25 结构弱，负数表示可能分错。",
        "better": "越高越好",
    },
}


# ----------------------------------------------------------------------
# 六、参数组合约束
#
# 有些参数不是独立的：选了 A 就必须把 B 也改成特定值，否则 sklearn 会直接抛错。
# 这类「无效组合」最好在点训练之前就提示，而不是等训练失败回来读英文报错。
#
# 字段：model 适用模型 / when 触发条件 / require 必须同时满足 / message 提示文案。
# 判定语义：when 中每个参数的当前值 ∈ 给定集合 ⇒ 触发该条；
#           此时 require 中任一参数当前值 ∉ 给定集合 ⇒ 组合无效。
# ----------------------------------------------------------------------
PARAM_COMBOS: list[dict[str, Any]] = [
    {
        "model": "logistic_regression",
        "when": {"penalty": ["l1"]},
        "require": {"solver": ["liblinear", "saga"]},
        "message": "penalty=l1 只支持 solver=liblinear 或 saga，当前组合会被 sklearn 直接拒绝。",
    },
    {
        "model": "logistic_regression",
        "when": {"penalty": ["elasticnet"]},
        "require": {"solver": ["saga"]},
        "message": "penalty=elasticnet 只能用 solver=saga。",
    },
    {
        "model": "logistic_regression",
        "when": {"solver": ["liblinear"]},
        "require": {"penalty": ["l1", "l2"]},
        "message": "solver=liblinear 只支持 penalty=l1 或 l2。",
    },
    {
        "model": "logistic_regression",
        "when": {"penalty": ["elasticnet"]},
        "require": {"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        "message": "penalty=elasticnet 需要同时给出 l1_ratio（0~1），否则 elasticnet 混合比例无法确定。",
        "severity": "warning",
    },
    {
        "model": "random_forest_classifier",
        "when": {"bootstrap": [False]},
        "require": {"oob_score": [False], "max_samples": [None]},
        "message": "关闭 bootstrap 后 oob_score 与 max_samples 都会失效（sklearn 会直接报错），请把它们留空或恢复 bootstrap。",
    },
    {
        "model": "random_forest_regressor",
        "when": {"bootstrap": [False]},
        "require": {"oob_score": [False], "max_samples": [None]},
        "message": "关闭 bootstrap 后 oob_score 与 max_samples 都会失效（sklearn 会直接报错），请把它们留空或恢复 bootstrap。",
    },
]


# ----------------------------------------------------------------------
# 七、调参手册（把「指标不理想」翻译成「具体该动哪个参数」）
#
# 这是「训练 → 看结果 → 调参」闭环的规则来源：前端拿 run 的指标与产物判定
# 命中哪条 signal，再把 advice 里「当前模型确实存在」的参数渲染成可一键应用的建议。
#
# advice 项的 op 语义（前端实现，保持一致即可）：
#   halve / double —— 基于当前值（未设置时取 sklearn_default，仍非数字则用 value 兜底）
#   set            —— 直接设为 value
# 数值型 op 的兜底规则：比例类（0~1）不低于 0.01；整数不低于 1。
# ----------------------------------------------------------------------
TUNING_PLAYBOOK: list[dict[str, Any]] = [
    {
        "id": "overfit",
        "title": "过拟合：训练集明显好于测试集",
        "signal": "overfit",
        "detect": "训练集指标高出测试集较多（分类 accuracy / f1 差 > 0.10；回归 R² 差 > 0.15）。",
        "why": "模型把训练数据里的噪声也当成规律记住了，换到没见过的数据就失灵。方向上要做的是「降低模型容量」，而不是加参数。",
        "advice": [
            {"param": "max_depth", "op": "halve", "value": 6,
             "note": "限制树的层数，最直接地压缩模型容量。"},
            {"param": "min_samples_leaf", "op": "double", "value": 4,
             "note": "逼每个叶子保留更多样本，抑制对个别样本的记忆。"},
            {"param": "min_samples_split", "op": "double", "value": 8,
             "note": "抬高分裂门槛，让树不容易长出细碎分支。"},
            {"param": "ccp_alpha", "op": "set", "value": 0.001,
             "note": "训练后剪枝，从很小的强度起步逐步加大。"},
            {"param": "max_features", "op": "set", "value": "log2",
             "note": "让每次分裂只看部分特征，增加随机性、削弱记忆。"},
            {"param": "n_estimators", "op": "set", "value": 300,
             "note": "森林加树数能降低预测方差，把过拟合波动摊平。"},
            {"param": "C", "op": "halve", "value": 0.5,
             "note": "线性模型里等价于加强正则，系数更保守。"},
            {"param": "n_neighbors", "op": "double", "value": 10,
             "note": "KNN 调大 k 让决策边界更平滑。"},
            {"param": "n_clusters", "op": "halve", "value": 3,
             "note": "聚类里合并相邻簇，避免把本应同类的样本拆碎。"},
        ],
        "escalate": "若压到很浅（如 max_depth≤3）后训练集指标也掉下来、测试集仍无改善，说明问题多半在特征或数据量，而不是模型容量。",
    },
    {
        "id": "underfit",
        "title": "欠拟合：训练集和测试集都不理想",
        "signal": "underfit",
        "detect": "训练集与测试集指标都偏低且彼此接近（分类 accuracy < 0.7 且两者差 ≤ 0.05；回归 R² < 0.3 且两者差 ≤ 0.1）。",
        "why": "模型表达能力不够，或者特征本身与目标关系太弱。方向上要「放宽限制 / 提升容量」，和过拟合正好相反——先确认不是过拟合再动这些参数。",
        "advice": [
            {"param": "max_depth", "op": "set", "value": 12,
             "note": "放宽树的深度限制，让模型能表达更复杂的关系。"},
            {"param": "min_samples_leaf", "op": "set", "value": 1,
             "note": "允许叶子只保留很少样本，恢复拟合细节的能力。"},
            {"param": "n_estimators", "op": "set", "value": 300,
             "note": "树多不等于过拟合，森林欠拟合时先加树数是最低风险的一步。"},
            {"param": "max_features", "op": "set", "value": 1.0,
             "note": "让每棵树用全部特征，单棵树更强。"},
            {"param": "C", "op": "double", "value": 10,
             "note": "线性模型放宽正则，允许系数更大。"},
            {"param": "n_neighbors", "op": "set", "value": 3,
             "note": "KNN 调小 k，让决策边界更贴合局部。"},
            {"param": "min_samples", "op": "halve", "value": 3,
             "note": "DBSCAN 放宽「核心点」门槛，让更稀疏的区域也能成簇。"},
        ],
        "escalate": "放宽到不限深度仍无改善，通常意味着特征不足或关系本身是非线性的：建议换模型（线性回归 → 随机森林）或回到数据处理环节补充特征。",
    },
    {
        "id": "no_signal",
        "title": "基本没学到东西：指标接近「瞎猜」水平",
        "signal": "no_signal",
        "detect": "分类 ROC-AUC ≤ 0.55；或回归 R² ≤ 0；或聚类轮廓系数 < 0.1。",
        "why": "这种结果一般不是调参能救回来的——要么特征与目标其实无关，要么关键信息被排除在训练之外（如误把有用列排除、目标列选错）。",
        "advice": [
            {"param": "n_estimators", "op": "set", "value": 200,
             "note": "作为对照：先把模型换成更稳的配置，排除「当前配置太弱」这一可能。"},
        ],
        "escalate": "优先去核对三件事：① 目标列是否选对（不是 id 或与业务无关的列）；② 参与训练的特征里是否漏了关键列、或混进了 id 这类无信息列；③ 目标列与特征在业务上是否真存在关系。确认无误再考虑换模型。",
    },
    {
        "id": "imbalanced",
        "title": "类别不均衡：模型偏向多数类",
        "signal": "imbalanced",
        "detect": "分类任务中多数类样本占比 ≥ 70%，且少数类的 recall 明显低于多数类。",
        "why": "模型只要一直预测多数类就能拿到不错的 accuracy，于是「懒得」去学少数类。这时 accuracy 会骗人，要看 f1 / recall。",
        "advice": [
            {"param": "class_weight", "op": "set", "value": "balanced",
             "note": "按类别样本量反比加权，让少数类获得更高权重——这是最省事也最有效的第一步。"},
            {"param": "n_estimators", "op": "set", "value": 200,
             "note": "配合加权一起用，让提升后的少数类模式更稳定。"},
        ],
        "escalate": "若加权后少数类 recall 上去了但 precision 崩了，说明两类确实高度重叠：此时更该补特征或重新定义目标，而不是继续调权重。",
        # 不重新训练的替代路径：改 class_weight 要重训，改决策阈值不用。
        # 两者目标相同（让少数类更多地被判出来），代价却差一个数量级，所以必须点出来。
        "inference_alternative": "若不想重新训练，可以改「决策阈值」：把阈值从 0.5 调低，效果上就是让少数类更容易被判为正类，同样能抬高少数类 recall。区别在于阈值只改「概率→标签」的映射、不动模型，因此可以立刻反复试；代价是它无法让模型学到新规律，只是重新分配了判错的方向。",
    },
    {
        "id": "cluster_weak",
        "title": "聚类结构弱：簇不清晰或簇数异常",
        "signal": "cluster_weak",
        "detect": "聚类任务中轮廓系数 < 0.25，或簇数量等于 1，或噪声点占比过高。",
        "why": "k-means 假设簇是近似球形且规模相近；DBSCAN 依赖密度。结构弱往往是参数没对上数据的真实形态。",
        "advice": [
            {"param": "n_clusters", "op": "halve", "value": 3,
             "note": "先减少簇数，看轮廓系数是否回升——原 k 可能把同类拆开了。"},
            {"param": "n_init", "op": "set", "value": 20,
             "note": "增加初始化次数，排除「只是初始质心没选好」造成的假象。"},
            {"param": "eps", "op": "double", "value": 0.8,
             "note": "DBSCAN 结果全是噪声时调大邻域半径，让点能互相连上。"},
            {"param": "min_samples", "op": "halve", "value": 3,
             "note": "DBSCAN 放宽核心点门槛，允许更稀疏的区域成簇。"},
            {"param": "whiten", "op": "set", "value": True,
             "note": "如果聚类前先做 PCA，白化能让各主成分等权参与距离计算。"},
        ],
        "escalate": "若各种 k 的轮廓系数都低于 0.25，可能是数据本来就没有明显分组结构，或者需要先做特征变换（如对偏态列取对数）再聚类。",
    },
    {
        "id": "costly",
        "title": "训练/推理成本偏高",
        "signal": "costly",
        "detect": "训练耗时较长（> 30s），或预处理后特征数远大于训练样本数（特征数 > 样本数）。",
        "why": "前者是算力换稳定性的问题，后者是「维度高于样本量」——这种情况下模型很容易找到虚假规律。",
        "advice": [
            {"param": "n_estimators", "op": "halve", "value": 50,
             "note": "树数是耗时的主要来源，减半通常只带来轻微指标损失。"},
            {"param": "max_samples", "op": "set", "value": 0.5,
             "note": "每棵树只用一半样本，训练时间近似减半（需 bootstrap=True）。"},
            {"param": "max_iter", "op": "halve", "value": 500,
             "note": "迭代类模型降低上限，先看指标是否还能接受。"},
            {"param": "leaf_size", "op": "double", "value": 60,
             "note": "近邻搜索建树更快，对结果无影响。"},
            {"param": "min_samples_leaf", "op": "double", "value": 4,
             "note": "叶子更少、树更小，同时顺带缓解高维带来的虚假规律。"},
            {"param": "max_features", "op": "set", "value": "sqrt",
             "note": "限制每次分裂考虑的特征数，直接降低训练开销。"},
        ],
        "escalate": "特征数超过样本数时，靠调参只能缓解；更有效的做法是回到数据处理阶段做特征筛选或降维。",
    },
]


# ----------------------------------------------------------------------
# 八、推理阶段参数说明（训练完成之后才生效的旋钮）
# ----------------------------------------------------------------------
# 与 MODEL_PARAMS 的本质区别：超参数在训练时决定「模型学成什么样」，改一次必须重训；
# 推理参数作用在已经训练好的模型上，改一次立刻能看到新结果。
#
# `affects_result` 必须显式声明：用户问「这个参数会不会影响结果」时，
# 界面要能直接回答，而不是让人自己猜。会改变预测结果的（threshold）与
# 只影响展示/性能的（limit）混在一起不加区分，是这类面板最容易误导人的地方。
#
# 这里条目少是**如实反映**，不是偷懒：训练后的环节本就没有多少能改结果又不需要重训的旋钮。
# 与其凑数塞进几个性能开关，不如把阈值这一个真正有用的讲透。
INFERENCE_PARAMS: list[dict[str, Any]] = [
    {
        "name": "threshold",
        "label": "决策阈值",
        "type": "float",
        "default": "0.5（与 sklearn 默认一致）",
        "sklearn_default": 0.5,
        "range": "0.05 ~ 0.95，步长 0.05",
        "step": 0.05,
        "tier": "core",
        "applies_to": "classification_binary",
        "affects_result": True,
        "tunable": True,
        "effect": "把「正类概率」变成「类别标签」的那条分界线：概率大于阈值才判为正类。默认 0.5 与 sklearn 原来的行为逐样本一致（概率恰为 0.5 时判负类），所以「不动它」等于「什么都没改」。调高＝只有很有把握才判正类，精确率上升、召回率下降；调低则相反。它不改动模型本身，只改概率到标签的映射，所以不需要重新训练。",
        "when_to_change": "漏判（假阴性）代价高时调低：疾病筛查、欺诈初筛、设备故障预警。误判（假阳性）代价高时调高：自动外呼、自动放款、自动告警。类别不均衡时尤其值得调——多数类占比很高时 0.5 会让模型几乎不判正类，F1 最优阈值往往明显低于 0.5。",
        "evidence": "阈值不该凭感觉挑：训练时会在留出测试集上扫一遍 0.05~0.95，把每个阈值对应的精确率 / 召回率 / F1 记进运行产物的 threshold_curve，推理面板据此展示取舍曲线并标出 F1 最优档。曲线口径为**正类**（不是 run.metrics 的 macro），因此不要与指标卡上的数值直接对比。",
        "code": "app/ml_engine/threshold.py",
    },
    {
        "name": "limit",
        "label": "预览行数",
        "type": "int",
        "default": "200",
        "sklearn_default": 200,
        # 两层上限如实写出来：接口收 5000，界面输入框只放到 200（表格再长就没法看了）。
        # 只写 5000 会让用户以为界面上也能填 5000，填进去却被悄悄夹回 200。
        "range": "1 ~ 200（界面输入框）；接口上限 5000",
        "tier": "core",
        "applies_to": "all",
        "affects_result": False,
        "tunable": True,
        "effect": "只决定返回多少行预测明细给你查看。全部样本都会参与推理与统计，预测结果、阈值判定、标签变化比例都不受它影响。",
        "when_to_change": "需要人工核对更多样本时调大；数据量大时调小可以让返回更快、页面更轻。",
        "code": "app/api/v1/experiments.py::PredictRequest.limit",
    },
]


def _params_for_model(model_name: str) -> list[dict[str, Any]]:
    entry = MODEL_PARAMS.get(model_name)
    return list(entry.get("params", [])) if entry else []


def build_catalog(registry_list: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """构造完整教学目录。

    registry_list：来自 MODEL_REGISTRY.list()（[{name, task}]）。传入时会补充每个
    模型的 task / 中文名，避免本文件与注册表漂移。
    """
    registry_list = registry_list or []
    task_of = {m["name"]: m.get("task", "") for m in registry_list}
    models: list[dict[str, Any]] = []
    for name, entry in MODEL_PARAMS.items():
        models.append(
            {
                "name": name,
                "task": task_of.get(name, ""),
                "note": entry.get("note", ""),
                "params": _params_for_model(name),
            }
        )
    return {
        "pipeline_steps": PIPELINE_STEPS,
        "preprocessing_params": PREPROCESSING_PARAMS,
        "training_params": TRAINING_PARAMS,
        "models": models,
        "metrics": METRIC_GUIDE,
        # 「训练 → 看结果 → 调参」闭环的规则来源：前端按 run 指标命中 signal，
        # 再把 advice 中该模型确实存在的参数渲染成一键应用的建议。
        "tuning_playbook": TUNING_PLAYBOOK,
        # 无效参数组合（如 l1 必须配 liblinear/saga），用于在点训练之前就提示
        "param_combos": PARAM_COMBOS,
        # 训练之后才生效的旋钮（决策阈值等）。与 models[].params 分开返回，
        # 是因为它们作用在「已训练好的模型」上，迭代代价与超参数完全不同：
        # 超参数改一次要重训，推理参数改一次立刻见效。混在一起会误导用户。
        "inference_params": INFERENCE_PARAMS,
        "conventions": {
            "target_naming": ["target", "label", "y", "class"],
            "default_test_size": 0.2,
            "default_seed": 42,
            "default_models": {
                "classification": "logistic_regression",
                "regression": "linear_regression",
                "clustering": "kmeans",
            },
        },
    }
