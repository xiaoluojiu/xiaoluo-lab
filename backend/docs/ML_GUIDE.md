# 机器学习模块使用与教学指南

> 面向两类读者：
> - **新手**：用「一键流程」跑出第一条结果并读懂它；
> - **进阶用户**：查第 4~7 节的参数表按需调参，知道"改这个会发生什么"。
>
> **口径来源**：第 4~7 节的全部参数表由 `backend/app/ml_engine/metadata.py` 派生，
> 与 `GET /ml/catalog` 接口、Agent 的 `ml.explain_config` 工具**同源**。
> 也就是说，文档、接口、Agent 回答三者不会各说一套。改参数说明只需改那一个文件。

---

## 1. 模块结构（优化后）

```
backend/app/
├── ml_engine/                     # 算法内核：不依赖 FastAPI，可被任意层调用
│   ├── registry.py                # MODEL_REGISTRY：12 个模型（5 分类 / 4 回归 / 2 聚类 / 1 降维）
│   ├── base.py                    # ModelAdapter 抽象：fit / predict / predict_proba / summary
│   │                              #   + 产物 HMAC 信封 to_signed_bytes / from_signed_bytes
│   ├── classification.py          # 5 个分类模型（含直方图梯度提升）
│   ├── regression.py              # 4 个回归模型
│   ├── clustering.py              # KMeans / DBSCAN
│   ├── dimensionality.py          # PCA（降维与可视化，不预测标签）
│   ├── preprocessing.py           # 预处理管道（missing → encoding → scaling，防泄漏）
│   ├── evaluation.py              # 指标计算（分类/回归/聚类 + 混淆矩阵/残差/逐类报告）
│   ├── explainability.py          # 特征重要性（树模型 / 线性模型两种口径）
│   ├── exceptions.py              # MLEngineException
│   │
│   ├── metadata.py                # 教学元数据单一事实源（流程/参数/指标 + 调参手册 + 组合约束）
│
├── experiments/service.py
│   └── ExperimentService.run()     # 「一键全流程」：切分→预处理→训练→评估→持久化
│
├── api/v1/experiments.py
│   ├── GET  /ml/models             # 模型目录
│   ├── GET  /ml/catalog  ★ 新增   # 教学目录（流程步骤/参数/指标，供前端与学习中心渲染）
│   ├── POST /ml/train              # 训练
│   ├── POST /ml/predict            # 推理（预览前 N 行）
│   ├── GET  /experiments/runs/{id}/predict-export  # 全量推理结果导出（csv / parquet）
│   └── POST /ml/explain /compare    # 解释 / 对比
│
└── tools/ml_tools.py               # Agent 工具（8 个）
    ├── ml.detect_task   ml.prepare   ml.train   ml.predict
    ├── ml.evaluate      ml.compare   ml.explain
    └── ml.explain_config  ★ 新增   # 教学问答：这个参数是干什么的
```

---

## 2. 九个环节与各自的中间产物

每个环节都标注了"中间结果落在哪里"——这是可核验性的基础。

| # | 环节 | 输入 | 输出 | 中间产物落点（可核验） |
|---|---|---|---|---|
| ① | 数据输入 | `dataset_id` + `version` | Polars DataFrame | `artifacts.train_rows` / `test_rows` |
| ② | 特征与目标划分 | `target_column` + `excluded_columns` | X / y | `artifacts.features`（排除后实际参与训练的列） |
| ③ | 训练/测试划分 | `test_size` + `seed` | 四个切分结果 | `artifacts.stratified` |
| ④ | 预处理（防泄漏） | 预处理配置 | 数值矩阵 | `artifacts.preprocessing_report` / `model_features` |
| ⑤ | 模型训练 | 模型名 + 参数 | estimator | `artifacts.model_summary` |
| ⑥ | 评估 | 测试集 | 指标 | `metrics` + `confusion_matrix` / `per_class` / `residual_stats` |
| ⑦ | 产物持久化 | model / pipeline | pkl 文件 | `artifacts.model_key` / `pipeline_key` |
| ⑧ | 推理 | `run_id` + 新数据 | 预测 + 概率 | 返回 `pipeline_applied` / `runtime`；导出走 `predict-export`（csv / parquet 全量） |
| ⑨ | 解释 | `run_id` | 特征重要性 | 返回 `source`（复用产物 / 现算）+ `method` |

**④ 的核心约定（防数据泄漏）**：管道只在训练集 `fit_transform`，测试集只 `transform`。
测试集的均值/方差永远不参与标准化参数的估计——这是评估可信的前提。

---

## 3. 关键参数说明表

### 3.1 训练参数

| 参数 | 中文名 | 类型 | 默认值 | 取值范围 | 调大 / 调小的影响 | 建议 |
|---|---|---|---|---|---|---|
| `task` | 任务类型 | enum | 自动推断 | classification / regression / clustering | 决定用哪类模型与哪套指标 | 目标是类别→分类；连续数值→回归；无目标想分组→聚类 |
| `target_column` | 目标列 | string | 无（聚类） | 数据集实际列名 | 监督学习要预测的那一列 | **务必选"你希望模型预测的列"，绝不要选 id** |
| `test_size` | 测试集比例 | float | `0.2` | `(0, 1)`，常用 0.2~0.3 | ⬆ 评估更稳，但训练数据更少；⬇ 训练数据更多，但评估波动大 | 数据 <200 行可升到 0.3；数据大可降到 0.1 |
| `seed` | 随机种子 | int | `42` | 任意整数 | 固定后切分与支持 `random_state` 的模型完全可复现 | **做对比实验必须固定同一个 seed** |
| `preprocessing` | 预处理配置 | object | 自动按数据画像生成 | `{missing, encoding, scaling}` | 覆盖默认预处理；只传一部分时其余保持默认 | 默认策略不适合时才手动指定 |
| `enable_learning_curve` | 生成学习曲线 | bool | `false` | true / false | true 时在多个训练子集上重训并评估，用于判断「加数据还有没有用」 | 想判断偏差/方差走向时开；数据大时耗时明显 |
| `enable_cv` | 5 折交叉验证 | bool | `false` | true / false | true 时在**切分前的全量数据**上再跑一遍 5 折 CV，给出各折均值 ± 标准差 | 数据少、单次切分波动大时开；代价约为普通训练的 5 倍 |

> **两个开关都默认关闭**：它们都在主训练之外额外跑模型，开着会显著拉长等待。
> 需要「这个 0.87 到底稳不稳」的答案时才开 —— 这正是它们存在的理由。

> **`test_size` 的硬约束**：分层切分要求测试集样本数 ≥ 类别数。
> 例如 8 行、2 类别、`test_size=0.2` → 测试集 2 个样本，刚好够；
> 若数据再小，模块会给出中文提示"测试集样本不足以覆盖 N 个类别"，而不是抛 sklearn 的英文原始错误。

### 3.2 预处理参数

| 参数 | 中文名 | 类型 | 默认值 | 取值范围 | 影响 | 何时改 |
|---|---|---|---|---|---|---|
| `missing.strategy` | 缺失值填补策略 | enum | `median`（有数值缺失列）/ `mode`（仅类别列） | mean / median / mode / constant | 决定空值用什么补齐。median 抗异常值；mean 受极端值影响；mode 用于类别列 | 有极端离群值→median；类别列**必须** mode/constant（数值策略对字符串无效） |
| `missing.columns` | 需填补的列（声明用） | string[] | 自动检测含空值的列 | 数据集实际列名 | 声明「哪些列需要填补」，用于漏配检测；实际填补按列类型整段执行 | 一般不必手写：未声明的空值列会被自动补进清单，不会静默漏掉 |
| `encoding.method` | 类别编码方式 | enum | `one_hot` | one_hot / ordinal | one_hot 为每个类别生成 0/1 列，不引入虚假大小关系，但列数膨胀；ordinal 用整数编号，列数不变但引入"类别大小"关系 | 取值少（几十以内）→ one_hot；取值多且有序（如评分等级）→ ordinal |
| `scaling.method` | 数值缩放方式 | enum | `standard` | standard / min_max | standard 均值 0 方差 1，对异常值较稳健；min_max 压到 [0,1]，受极值影响大 | KNN / KMeans / 逻辑回归等对尺度敏感的模型建议 standard |
| `excluded_columns` | 排除的特征列 | string[] | 空 | 数据集实际列名（**不得含 target**） | 这些列不参与训练，常用来排除 id / 主键 / 高基数标识列 | 模型把 id 当特征导致指标虚高时**务必排除** |

> **兜底行为（两条，都在 `build_pipeline`）**：
> 1. 你只传了 `scaling` 却漏配 `missing` → 自动为未覆盖的空值列补填补；
> 2. 你只传了 `scaling` 却漏配 `encoding` → 自动为未编码的类别列补 one_hot。
>
> 两条都只做"补漏"，**你显式给出的策略一律保留**。这是为了让部分配置也能跑通，
> 而不是静默改掉你的意图。

### 3.3 各模型的可调参数

参数的可调范围取决于底层 sklearn 模型的真实签名——`MODEL_PARAMS` 里每个参数名都经过
`MODEL_REGISTRY.supports_param()` 的签名级探测，因此**界面上能填的每一项都是模型真的接受的**，
不会出现"填了却报 TypeError"。

共 12 个模型 / **82 个可调参数**，其中 **36 个核心参数**。核心参数在界面上常显，
进阶参数在「展开进阶参数」里按需启用（启用后才写进请求；不启用＝交给 sklearn 用它自己的默认值）。

下面列出每个模型的核心参数；进阶参数的完整说明同样可查（界面 ⓘ / `ml.explain_config` 的 `topic=model`）。

#### logistic_regression

线性分类基线，可解释性强，输出概率。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `max_iter` | 最大迭代次数 | 1000 | 正整数，常用 200 ~ 5000 | 200 / 500 / 1000 / 2000 / 5000 | 优化器最多迭代多少轮。太小会在没收敛时就停下，系数不准（日志里出现 ConvergenceWarning）；太大只是更慢，结果不会更好。<br>**什么时候改**：看到未收敛告警、或同一份数据换 seed 后指标明显抖动时调大。 |
| `C` | 正则强度倒数 | 1.0 | 正浮点数，常用 0.01 ~ 100 | 0.01 / 0.1 / 1 / 10 / 100 | C 越大正则越弱、越贴近训练集（可能过拟合）；C 越小正则越强、模型越保守（可能欠拟合）。它是线性模型里最有效的「过拟合旋钮」。<br>**什么时候改**：训练集很好但测试集差 → 调小 C；两边都差 → 调大 C。 |
| `class_weight` | 类别权重 | None | None（每个样本等权） / balanced（按类别样本量反比加权） | — | balanced 按类别样本量反比加权，让少数类获得更高权重，缓解类别不均衡时模型一味偏向多数类。<br>**什么时候改**：类别明显不均衡时用 balanced；它通常能明显拉高少数类 recall，代价是多数类 precision 略降。 |

进阶参数：`penalty`、`solver`、`l1_ratio`、`tol`、`fit_intercept`

#### knn_classifier

基于距离投票的分类，对特征尺度敏感（建议保留标准化）。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_neighbors` | 邻居数量 k | 5 | 正整数，常见 3 ~ 20（且需 < 训练样本数） | 3 / 5 / 7 / 11 / 15 | 投票时参考几个最近的邻居。k 越小决策边界越贴合局部（易过拟合、对噪声敏感）；k 越大越平滑（易欠拟合，极端情况把所有点判成多数类）。<br>**什么时候改**：数据噪声大 → 调大 k；边界复杂、样本充足 → 调小 k。它通常是最值得先调的一项。 |
| `weights` | 距离权重 | uniform | uniform / distance | — | uniform 所有邻居等权投票；distance 按距离倒数加权，越近的邻居话语权越大，能提升局部拟合精度，但更易受噪声干扰。<br>**什么时候改**：特征已标准化、局部模式明显时可试 distance；数据噪声大时保持 uniform。 |
| `p` | 距离度量阶数 p | 2 | 正整数，1 = 曼哈顿距离，2 = 欧氏距离 | 1 / 2 | p=2 是欧氏距离（默认）；p=1 是曼哈顿距离（各维度差值绝对值相加），对单个维度的极端差异更不敏感，在高维或含离群维度时可能更稳。<br>**什么时候改**：欧氏距离效果一般、特征维度多且相互独立时可试 p=1。 |

进阶参数：`metric`、`algorithm`、`leaf_size`

#### decision_tree_classifier

树形规则，可解释性极强，容易过拟合。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `max_depth` | 最大深度 | None（不限，长到纯节点为止） | 正整数，常用 3 ~ 20 | 3 / 5 / 8 / 12 / 20 | 限制树的层数，是单棵树最有效的抗过拟合旋钮。不限深度会一直分裂到每个叶子都纯净，训练集接近 100% 而测试集明显差。<br>**什么时候改**：训练集指标远好于测试集 → 调小；两边都差、树显得太浅 → 放宽。 |
| `min_samples_leaf` | 叶节点最小样本数 | 1 | 正整数，常用 1 ~ 20 | 1 / 2 / 5 / 10 / 20 | 每个叶子至少保留多少样本，越大树越平滑、越抗噪；太大则学不到细节（欠拟合）。<br>**什么时候改**：样本量小、标签有噪声、或出现「某个叶子只有 1~2 个样本却决定了预测」时调大。 |
| `min_samples_split` | 分裂所需最小样本数 | 2 | 正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20 | — | 一个节点至少要多少样本才继续分裂，越大树越浅、越抗过拟合。<br>**什么时候改**：与 min_samples_leaf 配合使用；通常先动 max_depth 和 min_samples_leaf。 |
| `criterion` | 分裂准则 | gini | gini / entropy / log_loss | — | gini 计算快、最常用；entropy / log_loss 用信息增益（两者数学上仅差常数），对更细微的类别差异可能更敏感。<br>**什么时候改**：三者效果通常接近，gini 不佳时可试 entropy；收益一般很小，不必反复来回试。 |

进阶参数：`max_features`、`max_leaf_nodes`、`ccp_alpha`、`splitter`、`min_impurity_decrease`

#### random_forest_classifier

多棵树集成，抗过拟合，通常是最稳的默认选择。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_estimators` | 树的数量 | 100 | 正整数，常用 50 ~ 500 | 50 / 100 / 200 / 300 / 500 | 树越多结果越稳定（指标方差变小），但训练/推理耗时近似线性增加。超过一定数量后指标基本不再提升，只是变慢。<br>**什么时候改**：同一配置换 seed 后指标波动大 → 加到 200~300；只想快速验证思路 → 降到 30~50。 |
| `max_depth` | 单树最大深度 | None（不限） | 正整数，常用 3 ~ 30 | 4 / 6 / 10 / 16 / 30 | 控制每棵树的复杂度。集成本身有兜底：不限深度通常也能用，限制后训练更快、更抗过拟合。<br>**什么时候改**：训练集明显好于测试集 → 调小；树太浅导致两边都差 → 放宽。 |
| `min_samples_leaf` | 叶节点最小样本数 | 1 | 正整数，常用 1 ~ 20 | 1 / 2 / 4 / 8 / 16 | 每个叶子至少保留多少样本，越大单棵树越平滑、越抗噪。对随机森林而言，它常比 max_depth 更平滑地控制过拟合。<br>**什么时候改**：过拟合时调大；样本量本来就少时优先动它而不是 max_depth。 |
| `max_features` | 每棵树随机特征数 | sqrt | sqrt（特征数开方，默认） / log2（取对数，更随机） / None（全部特征） | — | 每棵树分裂时随机考虑多少特征。越小树之间差异越大、越抗过拟合，但单棵树更弱、可能欠拟合。<br>**什么时候改**：特征很多且高度相关、想更抗过拟合 → log2 或更小；特征本来就少、欠拟合 → None。 |

进阶参数：`class_weight`、`min_samples_split`、`bootstrap`、`oob_score`、`max_samples`、`max_leaf_nodes`、`ccp_alpha`

> **类别不均衡时注意**：随机森林同样支持 `class_weight`，它在进阶参数里。
> 如果少数类的 recall 明显偏低，这一项的收益通常大于继续调树数。

#### hist_gradient_boosting_classifier

直方图梯度提升（sklearn 的 LightGBM 同族实现）：先把连续特征分箱再找分裂点，
表格数据上通常比随机森林**更快且略准**。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `learning_rate` | 学习率 | 0.1 | 0 ~ 1 的浮点数，常用 0.01 ~ 0.3 | 0.01 / 0.05 / 0.1 / 0.2 | 每棵树对最终预测的贡献被它缩放。调小需要更多棵树才能达到同等拟合（更慢但常更稳），调大收敛快但容易过冲、卡在次优解。<br>**什么时候改**：它是提升类模型里最有效的一项——测试集不佳时先降到 0.05 并把 `max_iter` 加倍；只想快速试思路时保持 0.1。 |
| `max_iter` | 最大提升轮数 | 100 | 正整数，常用 50 ~ 1000 | 50 / 100 / 200 / 500 / 1000 | 最多叠加多少棵树。轮数不足会欠拟合，过多会过拟合且线性变慢；开启 `early_stopping` 时它只是上限，实际轮数由验证集决定。<br>**什么时候改**：学习率调小时要同步加大（大致翻倍）；开了早停后可以设得宽一些，让模型自己停。 |
| `max_leaf_nodes` | 单棵树最大叶子数 | 31 | 2 ~ 31（内部用 8 位存叶节点索引，超过 31 会直接报错） | 8 / 15 / 31 | 控制单棵树的复杂度，作用与决策树的 `max_depth` 同向，同时决定模型能表达多少交互。<br>**什么时候改**：过拟合时调到 15 或 8；样本量大且欠拟合时保持 31。注意它**不是**越大越好，上限就是 31。 |
| `min_samples_leaf` | 叶节点最小样本数 | 20 | 正整数，常用 5 ~ 100 | 5 / 20 / 50 / 100 | 每个叶子至少保留多少样本。它比 `max_leaf_nodes` 更平滑地控制过拟合，也是直方图方法默认就偏大的一项（20，远大于决策树的 1）。<br>**什么时候改**：数据噪声大或样本少时调大到 50~100；指标明显欠拟合时调小到 5。 |

进阶参数：`l2_regularization`、`max_features`、`early_stopping`、`class_weight`、`max_bins`

> **与随机森林怎么选**：特征基数高（one-hot 之后列很多）且样本上万时，
> 直方图方法的训练耗时通常明显低于随机森林；样本只有几百行时两者差距不大，
> 随机森林的抗过拟合更省心。

#### linear_regression

假设特征与目标线性相关的最简回归，几乎无超参（seed 不适用）。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `fit_intercept` | 拟合截距 | True | true / false | — | 是否给模型一个常数项。关闭后回归面必须过原点，只适合「特征全为 0 时目标必然为 0」的场景。<br>**什么时候改**：业务上确认无截距项时才关掉。注意：线性回归**没有正则化参数**，如果它在测试集上明显过拟合，正确做法是换随机森林，而不是调这里。 |

进阶参数：`positive`

#### knn_regressor

基于邻居均值回归，对尺度敏感。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_neighbors` | 邻居数量 k | 5 | 正整数，需 < 训练样本数 | 3 / 5 / 7 / 11 / 15 | 用最近几个邻居的平均值作为预测。k 小 → 贴合局部细节但噪声敏感；k 大 → 曲线平滑，极端情况退化成「全数据均值预测」。<br>**什么时候改**：预测曲线抖动厉害、大误差多（RMSE 高）→ 调大 k；曲线过度平滑、抓不到局部规律 → 调小。 |
| `weights` | 距离权重 | uniform | uniform / distance | — | uniform 邻居等权平均；distance 按距离倒数加权，近邻贡献更大，通常能降低预测偏差，但更易受噪声干扰。<br>**什么时候改**：局部趋势明显时可试 distance；噪声大时保持 uniform。 |

进阶参数：`p`、`algorithm`、`leaf_size`

#### decision_tree_regressor

树形回归，可捕捉非线性，容易过拟合。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `max_depth` | 最大深度 | None（不限） | 正整数，常用 3 ~ 20 | 3 / 5 / 8 / 12 / 20 | 限制树的层数，是单棵树最有效的抗过拟合旋钮。越深越贴合训练集，越浅越平滑。<br>**什么时候改**：训练集 R² 远高于测试集 → 调小；两边都差、树太浅 → 放宽。 |
| `min_samples_leaf` | 叶节点最小样本数 | 1 | 正整数，常用 1 ~ 20 | 1 / 2 / 5 / 10 / 20 | 每个叶子至少保留多少样本，越大树越平滑、抗噪。对回归树尤其重要——叶子样本太少时，叶子值会被个别极端样本带偏。<br>**什么时候改**：过拟合、或预测值出现不合理的极端值（残差 max_abs 很大）时调大。 |
| `min_samples_split` | 分裂所需最小样本数 | 2 | 正整数或小于 1 的浮点数（表示比例），常用 2 ~ 20 | — | 节点样本数不足该值就不再分裂，越大树越浅。<br>**什么时候改**：过拟合时与 min_samples_leaf 一起调大。 |
| `criterion` | 分裂准则 | squared_error | squared_error / absolute_error / friedman_mse / poisson | — | squared_error 对大误差惩罚重；absolute_error 用绝对误差，对离群点稳健得多；friedman_mse 面向梯度提升场景；poisson 适合计数型目标。<br>**什么时候改**：目标含明显离群点、或 MAE 与 RMSE 差距很大时试 absolute_error；目标是计数（如订单量）可试 poisson。 |

进阶参数：`max_features`、`max_leaf_nodes`、`ccp_alpha`

#### random_forest_regressor

集成回归，非线性场景的稳健默认选择。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_estimators` | 树的数量 | 100 | 常用 50 ~ 500 | 50 / 100 / 200 / 300 / 500 | 越多越稳（预测方差更小）、越慢。<br>**什么时候改**：追求稳定 → 200+；快速试跑 → 30~50。 |
| `max_depth` | 单树最大深度 | None（不限） | 3 ~ 30 | 4 / 6 / 10 / 16 / 30 | 控制每棵树的复杂度，越小越平滑。<br>**什么时候改**：过拟合（训练 R² 高、测试 R² 低）时调小。 |
| `min_samples_leaf` | 叶节点最小样本数 | 1 | 正整数，常用 1 ~ 20 | 1 / 2 / 4 / 8 / 16 | 越大越平滑、越抗过拟合。回归场景下它对「个别极端样本带偏预测」的抑制效果很明显。<br>**什么时候改**：训练 R² 高、测试 R² 低时调大；也是降低 RMSE 尾部的常用手段。 |
| `max_features` | 每棵树随机特征数 | 1.0（用全部特征） | 1.0（全部特征，默认） / 0.5（一半） / sqrt（开方） / log2（取对数） | — | 回归版 sklearn 默认用全部特征；调小能让树之间更不同、更抗过拟合，但单棵树更弱。<br>**什么时候改**：特征多且相关、过拟合时试 sqrt 或 0.5；欠拟合时保持 1.0。 |

进阶参数：`min_samples_split`、`bootstrap`、`oob_score`、`max_samples`、`max_leaf_nodes`、`ccp_alpha`

#### kmeans

划分式聚类，需指定簇数，适合近似球形、规模相近的簇。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_clusters` | 簇数量 k | 8 | 2 ~ min(样本数, 合理上限)，常用 2 ~ 10 | 2 / 3 / 4 / 5 / 8 / 10 | 要分成几类，是聚类里唯一真正决定结果的参数。k 太大会把本应同类的样本拆开（轮廓系数通常下降）；太小会把不同类混在一起。<br>**什么时候改**：先用业务含义定候选范围，再对比不同 k 的轮廓系数。注意轮廓系数会偏向 k=2，不能只看它。 |
| `init` | 初始质心选法 | k-means++ | k-means++（分散初始化，默认） / random（随机抽点） | — | k-means++ 让初始质心彼此尽量远，通常收敛更快、结果更好；random 容易陷入较差的局部最优。<br>**什么时候改**：基本不用改成 random。 |
| `n_init` | 初始化次数 | auto（sklearn 默认，等价于 10） | 正整数，常用 10 ~ 20 | 1 / 10 / 20 | 用多少组不同的初始质心各跑一遍，取最好的一组。越大结果越稳定但更慢。<br>**什么时候改**：同一数据换 seed 后簇划分明显不同（结果不稳定）时调大。 |

进阶参数：`max_iter`、`tol`、`algorithm`

#### dbscan

密度聚类，自动发现簇数，可识别噪声点（标签 -1），不支持对新样本预测。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `eps` | 邻域半径 | 0.5 | 正浮点数（**单位是标准化后的尺度**，通常 0.1 ~ 2） | 0.2 / 0.3 / 0.5 / 0.8 / 1.0 | 判断「两个点算不算邻居」的距离阈值，是最关键也最难定的参数。太大 → 所有点合并成一个大簇；太小 → 大量点被判为噪声（标签 -1）。<br>**什么时候改**：噪声点特别多 → 逐步调大；所有点挤成一簇 → 调小。建议以 0.05 为步长逐档试并观察簇数与噪声比例。 |
| `min_samples` | 核心点最小邻居数 | 5 | 正整数，常见 3 ~ 20 | 3 / 5 / 8 / 10 / 20 | 一个点周围至少要多少邻居才算「核心点」。越大越严格，越容易把稀疏区域的点判为噪声。<br>**什么时候改**：噪声过多 → 调小；簇过于碎（很多小簇）→ 调大。经验上可取「特征数 + 1」作为起点。 |

进阶参数：`metric`、`algorithm`、`leaf_size`

#### pca

主成分分析，降维与可视化，不预测标签。

| 参数 | 中文名 | 默认 | 范围 | 常用 | 影响与什么时候改 |
|---|---|---|---|---|---|
| `n_components` | 主成分数量 | 2 | 1 ~ min(样本数, 特征数) | 2 / 3 / 5 / 10 | 保留多少个主成分。取 2 可直接二维可视化；取值越大保留信息越多但降维意义越弱（取到特征数即等于没降维）。<br>**什么时候改**：看 explained_variance_ratio 的累计值，达到 85% 以上通常认为保留充分。只为画图则保持 2。 |
| `whiten` | 白化 | False | true / false | — | 开启后把各主成分除以自己的标准差，使输出各维度方差一致。对「看主成分重要性」是干扰，但作为下游模型的输入特征时有时更稳定。<br>**什么时候改**：把 PCA 当作下游模型的预处理特征（如配合聚类/KNN）时可试开启；只为解释方差与画图则保持关闭。 |

进阶参数：`svd_solver`、`tol`

### 3.4 参数不是彼此独立的：无效组合提示

有些参数必须配对使用，否则 sklearn 会直接抛错。界面会在**点训练之前**把这类组合标出来，
不必等训练失败再读英文报错：

| 模型 | 触发条件 | 同时必须满足 | 说明 |
|---|---|---|---|
| `logistic_regression` | `penalty=l1` | `solver` ∈ {liblinear, saga} | l1 只在 liblinear / saga 下可用 |
| `logistic_regression` | `penalty=elasticnet` | `solver=saga` | elasticnet 只支持 saga |
| `logistic_regression` | `solver=liblinear` | `penalty` ∈ {l1, l2} | liblinear 不支持 elasticnet / none |
| `logistic_regression` | `penalty=elasticnet` | 需给出 `l1_ratio` | 否则混合比例无法确定 |
| `random_forest_classifier` | `bootstrap=false` | `oob_score=false` 且 `max_samples` 为空 | 两者都依赖自助采样 |
| `random_forest_regressor` | `bootstrap=false` | `oob_score=false` 且 `max_samples` 为空 | 同上 |

> 判定口径：未显式设置的参数按**该参数的真实默认值**参与判断，
> 因此「什么都不动」不会误报（默认的 l2 + lbfgs 是合法组合）。

> **关于 11 个模型的口径**：注册表共 11 个模型 = 4 分类 + 4 回归 + 2 聚类 + 1 降维（PCA）。
> `GET /ml/models` 会**隐藏 `dimensionality` 任务**（PCA 不在训练下拉里，
> 它服务于降维与可视化），但 `GET /ml/catalog` 完整返回，用于教学讲解。

---

## 4. 指标解读表

| 指标 | 中文名 | 任务 | 含义 | 怎么读 | 方向 |
|---|---|---|---|---|---|
| `accuracy` | 准确率 | 分类 | 预测正确的样本占比 | **类别极不均衡时会严重虚高**（全猜多数类也能很高） | 越高越好 |
| `precision` | 精确率（macro） | 分类 | 预测为某类中，真正属于该类的比例，各类平均 | 关心"误报"代价时看它 | 越高越好 |
| `recall` | 召回率（macro） | 分类 | 真正属于某类的样本中，被成功找出的比例，各类平均 | 关心"漏报"代价时看它 | 越高越好 |
| `f1` | F1（macro） | 分类 | precision 与 recall 的调和平均 | 两指标冲突时看 F1；**类别不均衡时比 accuracy 更可信** | 越高越好 |
| `roc_auc` | ROC-AUC | 分类 | 把正类排在负类前面的能力，与阈值无关 | 0.5=随机猜，0.7 尚可，0.8+ 良好，0.9+ 优秀 | 越高越好，0.5 为无信息基线 |
| `mae` | 平均绝对误差 | 回归 | 误差绝对值的平均，单位与目标列一致 | 最直观："平均差多少个单位" | 越低越好 |
| `rmse` | 均方根误差 | 回归 | 误差平方平均后开方 | 对大误差惩罚更重，更能反映"偶尔差很多" | 越低越好 |
| `r2` | 决定系数 R² | 回归 | 模型解释了目标方差的多少比例 | 1=完美，0=与猜均值一样，**负数不如猜均值** | 越高越好，>0 才算有信息 |
| `cluster_count` | 簇数量 | 聚类 | 算法划分出的簇个数 | 看是否接近业务预期的分组数 | 无绝对好坏 |
| `silhouette` | 轮廓系数 | 聚类 | 簇内紧密 + 簇间分离的综合分 | -1~1：>0.5 结构清晰，0.25~0.5 一般，<0.25 弱，负值可能分错 | 越高越好 |

> **指标永不返回 NaN/Inf**：无法计算时返回 `None` 并附原因文字，
> 避免 JSON 序列化失败把整个结果吞掉。

---

## 5. 边界与错误提示对照表

模块把 sklearn / pandas 层难懂的英文异常统一转成可操作的中文提示：

| 场景 | 得到的提示 |
|---|---|
| 目标列不存在 | `目标列 'xxx' 不存在` |
| 排除列不存在 | `排除列不存在：['xxx']`（并列出可用列） |
| target 同时在排除列 | `目标列不能同时出现在排除列中` |
| 标签全部相同（回归） | `目标列无变化（所有取值相同），回归模型无法学习` |
| 类别数 < 2（分类） | `分类任务至少需要 2 个类别` |
| 训练特征列为空 | `训练特征列为空（排除后无可用列）` |
| 样本数 < 2 | `训练样本数 < 2，无法训练` |
| 测试集不足以覆盖类别 | `测试集样本不足以覆盖 N 个类别（当前 X 行、test_size=…，仅能划分 M 个测试样本）。请调大 test_size 或补充数据。` |
| `test_size` 越界 | `test_size 必须在 (0,1) 区间` |
| 推理缺列 | `推理数据缺少训练时的特征列：['xxx']`（并返回所需列清单） |
| 产物缺失 | 提示"请重新训练一次"，不静默产出结果 |

**目标列空值自动处理**：标签缺失的行会被**剔除**（记录在 `artifacts.dropped_rows`），
而不是让 sklearn 抛 "Input contains NaN"。这是数据不干净时最常见的阻塞点。

---

## 6. 教学接口：`GET /ml/catalog`

```bash
curl http://127.0.0.1:8000/api/v1/ml/catalog
```

返回结构（与本文第 2/4/5 节完全同源）：

```json
{
  "pipeline_steps":        [ ... 9 个环节，每个含 name/summary/input/output/key_behaviour/verify/code ],
  "preprocessing_params":  [ ... 5 项 ],
  "training_params":       [ ... 5 项 ],
  "models":                [ ... 11 个模型，含 task/note/params（73 项，每项带 type/tier/default/sklearn_default/range/options/step/typical/requires） ],
  "metrics":               { ... 10 个指标 },
  "tuning_playbook":       [ ... 6 类调参情形，见第 8 节：signal/title/detect/why/advice/escalate ],
  "param_combos":          [ ... 6 条无效参数组合约束，见第 3.4 节 ],
  "inference_params":      [ ... 2 项训练后才生效的旋钮（threshold / limit），每项带 affects_result / applies_to，见第 8.4 节 ],
  "conventions": {
    "target_naming": ["target", "label", "y", "class"],
    "default_test_size": 0.2,
    "default_seed": 42,
    "default_models": {
      "classification": "logistic_regression",
      "regression": "linear_regression",
      "clustering": "kmeans"
    }
  }
}
```

**Agent 侧对应工具** `ml.explain_config`，`topic` 取值：

| topic | 返回 | 可选参数 |
|---|---|---|
| `pipeline` | 9 个流程环节 | — |
| `preprocessing` | 5 项预处理参数 | — |
| `training` | 5 项训练参数 | — |
| `model` | 指定模型的可调参数 | `model`（必填） |
| `metrics` | 指标含义与好坏判断 | `metric`（可选，单个指标） |
| `tuning` | 调参手册 `{playbook, invalid_combos}`：指标不理想时该动哪些参数，以及哪些参数不能一起用 | — |
| `inference` | 训练后才生效的推理参数（决策阈值等），含 `affects_result` | — |

> 因此问 Agent「我的随机森林过拟合了怎么办」「KNN 的 k 怎么调」，
> 得到的答案与本文第 4/9 节同源，不会出现"文档和助手各说一套"。

### 6.1 前端如何呈现这些内容（「透明」看得见的地方）

同一份 catalog 在前端「机器学习」页被两处消费，因此**界面上的说明与本文表格口径一致**：

| 界面位置 | 展示什么 | 数据来源 |
|---|---|---|
| 模型下拉框下方 | 一句话模型说明 | `modelMeta.MODEL_DESC` |
| **「模型参数」面板** | 该模型**全部**可调参数：核心参数常显、进阶参数在「展开进阶参数」里按需启用；下拉/数字框按参数类型渲染 | `models[].params` 的 `type` / `tier` / `options` |
| 参数名旁的 **ⓘ** | 范围 · 默认 · 常用值 · 影响 · 什么时候改（悬停或聚焦展开，默认不占位） | 同上 |
| 参数已被改动时 | 名称旁出现 ↺，点一下只回退这一项；顶部「恢复推荐值」作用于全部 | 前端比对 `recommendedParams` |
| **「预处理」面板** | 缺失值填补 / 类别编码 / 数值缩放三项，留空即自动 | `preprocessing_params` |
| 无效参数组合 | 命中时在面板内以警示条说明（如 l1 需配 liblinear/saga） | `param_combos` |
| `test_size` / `seed` 输入框 | ⓘ 说明 + test_size 本地校验 (0,1) | `training_params.test_size` / `seed` |
| 「流程与参数说明」卡片 · **流程环节** tab | 9 个环节：名称 / 输入 / 输出 / 关键行为 / 可核验点 / 对应代码位置 | `pipeline_steps` |
| 同上 · **参数说明** tab | 训练参数表 + 预处理参数表 + **当前模型参数表置顶** + 推理阶段参数表 + 其他模型表（后两张表的行上会标「不影响结果」「仅二分类」这类例外标记） | `training_params` / `preprocessing_params` / `models` / `inference_params` |
| 同上 · **指标解读** tab | 10 个指标的含义 / 怎么读 / 方向 | `metrics` |
| 训练结果卡下方「训练过程透明化」 | 真实产物：参与训练的原始列、被排除的列、训练/测试行数、是否分层、剔除的空标签行数、预处理三段策略与实际生效配置、`fitted` 状态、编码后的模型输入列、实际传参字典、拟合耗时 | `run.artifacts`（**不是前端估算**） |
| 训练结果卡内「调参建议」 | 命中过拟合 / 欠拟合 / 几乎没学到东西 / 类别不均衡 / 聚类结构弱 / 成本偏高时，逐条给出参数、目标值与理由，可一键写回参数面板 | `run.artifacts`（含 `train_metrics` / `class_distribution`）+ `tuning_playbook` |
| **「模型推理」面板 · 决策阈值输入框** | 训练后可调、改它不用重训；带 ⓘ（范围 / 默认 / 影响 / 什么时候改 / 依据 / 是否影响结果）；非二分类时置灰并说明原因 | `inference_params.threshold` |
| 同上 · 推理结果下方的**阈值回执条** | 本次实际用的阈值 · 正类标签 · 标签变化比例（0 会额外标注「与默认行为等价」） | `POST /ml/predict` 返回值 |
| 同上 · **阈值取舍依据**表 | 19 档阈值的精确率 / 召回率 / F1 / 判正类比例，标出当前所处档位，并提供「采用 F1 最高阈值」 | `artifacts.threshold_curve`（首选）或推理返回的 `threshold_curve` |

> **口径保证**：界面上的「范围 / 默认 / 影响」全部来自同一份 `metadata.py`，
> 不存在「文档写一套、界面写另一套」的漂移。改参数说明只需改 `metadata.py` 一处。

**关键前端的查表入口**（供后续新增参数时复用）：

```ts
import { findParamSpec, paramLabel } from "../../features/ml/ParamGuide";

findParamSpec(catalog, "test_size")          // 训练参数 → 预处理参数 → 当前模型 → 任一模型
paramLabel(catalog, "n_estimators", "random_forest_classifier")  // 中原名，退回 PARAM_LABELS
```

新增一个可调参数时，只需在 `metadata.py` 的对应表里加一条，
`/ml/catalog`、`ml.explain_config`、本文档、**以及前端界面**会同时生效。


这样用户在对话里问"KNN 的 k 怎么调"，Agent 能给出与文档一字不差的答案。

---

## 7. 结果解读示例：同一份数据，五种做法

这一节展示**参数选择如何直接改变结论**。数据为 300 行、含真实规律的二分类集
（`income` 越高、`age` 越偏离中段 → 更可能 `label=1`，另加噪声）：

| 配置 | accuracy | F1 | ROC-AUC | 结论 |
|---|---|---|---|---|
| **A** 排除 `row_id` + 随机森林 | 0.9333 | 0.8800 | **0.9877** | ✅ 推荐基线 |
| **B** 把 `row_id` 也当特征 | 0.9467 | 0.9008 | 0.9892 | ⚠️ 指标更高，但**这是泄漏式虚高** |
| **C** 单棵决策树，不限深度 | 0.9200 | 0.8604 | 0.9692 | 尚可，但过拟合风险 |
| **D** 单棵决策树，`max_depth=4` | 0.9333 | 0.8800 | 0.9769 | 限制深度后不输随机森林 |
| **E** KNN 不标准化 | 0.9067 | 0.7678 | 0.9315 | ❌ 明显劣于 A |

**逐条解读**：

- **A vs B** —— B 的指标**全面更高**，但这**不代表 B 更好**。
  `row_id` 是行号，与标签无关；模型记住它可以略微降低训练损失，
  掩盖了真实特征的作用。**这是最危险的陷阱**：指标变好看，模型却更不可信。
  判断方法：看 `artifacts.features` 里有没有不该出现的列；
  或换一份数据推理，B 会明显崩掉。
- **C vs D** —— 不限深度的单树在训练集上几乎能到 100%，但测试集 AUC 反而低于限制深度 4 的版本。
  这是**过拟合最直观的证据**：更复杂的模型没有换来更好的泛化。
- **E** —— KNN 靠距离度量，`income`（量级十几）会完全压过 `age`（量级几十但尺度不同），
  不做标准化等于让一个特征主导距离计算。所以 KNN/KMeans 这类模型
  **标准化不是可选项而是必需项**。

### 回归与聚类的正常结果

```
linear_regression 预测 spend：
  {'mae': 0.908, 'mse': 1.198, 'rmse': 1.094, 'r2': 0.9967}
```

`R²=0.9967` 说明特征几乎完全解释了 `spend` 的方差；`MAE=0.908` 意味着
"平均预测偏差约 0.9 个货币单位"——**MAE 比 R² 更好向业务方解释**。

```
kmeans (k=3)：
  {'cluster_count': 3, 'silhouette': 0.2350}
```

轮廓系数 0.235 属于"结构较弱"（<0.25）。这时的正确动作**不是调参**，
而是回到业务：这三类在业务上说得通吗？说不通就换 k 或换特征，而不是死磕指标。

---

## 8. 调参：从「凭感觉试」到「有依据地试一次」

调参的困难通常不在"找不到参数"，而在**指标不理想时不知道该往哪个方向动**。
这一节把这件事拆成两个可执行的部分：**判定信号**（这次到底出了什么问题）与**对应动作**（动哪个参数、往哪边动）。

规则定义在 `metadata.py` 的 `TUNING_PLAYBOOK`，训练完成后由界面自动匹配并给出可一键应用的建议
（Agent 侧通过 `ml.explain_config` 的 `topic=tuning` 取同一份内容）。

### 8.1 六类判定信号

判定只用**训练时真实落库的产物**，不做前端估算。其中 `train_metrics` 是为此专门新增的产物：
它与 `run.metrics`（测试集口径）严格分开，两者相减才说明问题是"学过头"还是"还没学会"。

上表里的每个数字都由 `metadata.py` 的 `SIGNAL_RULES` 定义，经 `GET /ml/catalog` 的
`signal_rules` 字段透给前端 —— 前端判定与这里的说明文字读的是同一个数，不会各写一份 0.1。
`TUNING_PLAYBOOK` 的 `detect` 文案同样由它拼出，改阈值只需改一处。

| signal | 判定口径 | 对应的问题 |
|---|---|---|
| `overfit` | 分类 accuracy / f1 的 train−test > 0.10；回归 R² 的 train−test > 0.15 | 模型把训练数据的噪声也记住了 |
| `underfit` | 分类 test accuracy < 0.7 且 train−test ≤ 0.05；回归 test R² < 0.3 且 ≤ 0.1 | 模型表达能力不够 |
| `no_signal` | 分类 ROC-AUC ≤ 0.55；回归 R² ≤ 0；聚类轮廓系数 < 0.1 | 基本等于瞎猜，多半不是调参能救的 |
| `imbalanced` | `class_distribution.majority_ratio` ≥ 0.7（分类） | accuracy 被多数类撑高，少数类被忽略 |
| `cluster_weak` | 轮廓系数 < 0.25 或簇数 = 1 | 簇结构不清晰或簇数不对 |
| `costly` | 训练耗时 > 30s，或预处理后特征数 > 训练样本数 | 算力成本高 / 维度高于样本量 |

> `no_signal` 命中时会抑制 `underfit`：前者更精确，两条同时出现只会造成困惑。

### 8.2 每类情形的建议动作

界面会把下表中的动作按**当前模型确实拥有的参数**过滤后展示，并标注目标值。例如过拟合时
树模型看到的是深度/叶子相关项，线性模型看到的是 `C`——不会给出这个模型根本没有的参数。

| 情形 | 建议动作（方向） |
|---|---|
| `overfit` | `max_depth` 调小 · `min_samples_leaf` 调大 · `min_samples_split` 调大 · `ccp_alpha` 起步 0.001 · `max_features` 换 log2 · `n_estimators` 加到 300 · 线性模型 `C` 调小 · KNN `n_neighbors` 调大 · 聚类 `n_clusters` 调小 |
| `underfit` | `max_depth` 放宽到 12 · `min_samples_leaf` 回到 1 · `n_estimators` 加到 300 · `max_features` 用全部特征 · 线性模型 `C` 调大 · KNN `n_neighbors` 调小 · DBSCAN `min_samples` 调小 |
| `no_signal` | 不主张继续调参：先核对①目标列是否选对 ②特征里有没有漏掉关键列或混进 id ③特征与目标在业务上是否真有关系 |
| `imbalanced` | `class_weight = balanced`（最省事也最有效的第一步），配合 `n_estimators` 200 |
| `cluster_weak` | `n_clusters` 调小 · `n_init` 20 · DBSCAN `eps` 调大 / `min_samples` 调小 · PCA `whiten` 开启 |
| `costly` | `n_estimators` 减半 · `max_samples` 0.5 · `max_iter` 减半 · `leaf_size` 调大 · `min_samples_leaf` 调大 · `max_features` 换 sqrt |

每条规则还带一句 `escalate`——**什么时候该停止调参**。例如：
压到很浅后测试集仍无改善，问题多半在特征或数据量；特征数超过样本数时，靠调参只能缓解，
更有效的是回到数据处理阶段做筛选或降维。这比再多试几组参数更省时间。

### 8.3 人工排查顺序（先数据后模型）

1. **标签与特征真的相关吗？** —— 若 AUC≈0.5，先怀疑数据本身。
2. **有没有该排除的高基数列？** —— id/主键会让指标虚高（第 7 节 B）。
3. **训练 vs 测试指标差距大吗？** —— 差距大是过拟合（第 7 节 C）。
4. **树模型**：先试 `random_forest_*`（`n_estimators=100~200`），它几乎总是不差的基线。
5. **KNN / KMeans / 逻辑回归**：检查 `scaling` 是否生效。
6. **线性模型不收敛**：调大 `max_iter`。
7. **类别不均衡**：别只看 `accuracy`，看 `f1` 和 `recall`。
8. **数据量小（<200 行）**：`test_size` 提到 0.3 让评估更稳；
   同时注意分层切分的类别覆盖约束（第 4.1 节）。
9. **要可复现**：`seed` 固定，对比实验之间**不要改 seed**。

**改完参数后必查的产物**：

| 检查项 | 位置 | 看什么 |
|---|---|---|
| 实际用了哪些特征 | `artifacts.features` | 有无泄漏列（id、主键） |
| 预处理实际怎么做的 | `artifacts.preprocessing_report` | 你配的策略是否真的生效 |
| 切分是否分层、规模多少 | `artifacts.stratified` / `train_rows` / `test_rows` | 评估基数是否够 |
| 训练集 vs 测试集指标 | `artifacts.train_metrics` vs `run.metrics` | 差距大 = 过拟合；两边都低 = 欠拟合 |
| 分类的类别分布 | `artifacts.class_distribution` | 多数类占比 ≥70% 时 accuracy 会骗人 |

> 一个实际用法：调完参数后**保持 `seed` 不变再训一次**，然后用「实验历史」里勾选两次运行做对比
> （第 7 节的做法）——这样才能确定指标变化来自参数，而不是数据切分的随机性。

### 8.4 训练之后的旋钮：决策阈值

9.1~9.3 谈的都是**超参数**：改一个就得重训一次。但分类问题里还有一个参数
既影响结果、又完全不用重训 —— **决策阈值**。

**它是什么。** 模型输出的是「属于正类的概率」，标签则由一条线切出来：概率大于阈值 → 正类，
否则负类。sklearn 默认这条线在 0.5，所以「不设阈值」就等于「阈值 = 0.5」。
改它不碰模型本体，只改概率到标签的映射，**改一次立刻见效** —— 这是它与超参数最本质的区别。

**它对结果的影响有多大。** 同一组概率，阈值从 0.5 调到 0.7，精确率上升、召回率下降，
混淆矩阵里整行整列的数字都会变。接口返回 `label_changed_ratio`（标签发生变化的样本占比），
它就是「这次调整到底影响了多少样本」的直接答案；为 0 表示该阈值在此数据上等价于默认行为。

**该定多少：不要凭感觉，曲线已经算好了。** 训练时会在**留出测试集**上扫一遍
0.05~0.95（步长 0.05），把每档的指标写进 `artifacts.threshold_curve`，
推理面板直接展示这张表，并标出 F1 最优的那一档（可一键采用）。

| 字段 | 含义 |
|---|---|
| `threshold_curve.points[]` | 每档的 `threshold` / `precision` / `recall` / `f1` / `accuracy` / `positive_rate` |
| `threshold_curve.basis` | `holdout_test` = 留出测试集（选阈值的正当依据）；`inference_data` = 推理数据本身 |
| `threshold_curve.metric_average` | 固定为 `positive`，见下方口径提醒 |
| `threshold_curve.suggested_threshold` | F1 最高的那一档 |
| `threshold_curve.reason` | 曲线不可用时的原因（非二分类、真实标签缺正类等） |

> **口径提醒**：曲线上的 precision / recall / f1 是**正类**口径（`pos_label = classes_[-1]`），
> 而 `run.metrics` 用 **macro** 口径。两者算法不同，**不要横向比较** ——
> 调阈值看的本来就是「正类的召回换精确率」，用 macro 会把两类的变化平均掉、看不出取舍方向。

> **依据提醒**：只有 `holdout_test` 的曲线才代表泛化能力。若某次运行训练时还没有
> `threshold_curve`（例如改造前训练的运行），推理时会在推理数据上就地补算一条，
> 此时 `basis = inference_data` —— 若推理数据恰好就是训练数据，指标会明显偏乐观，只能当参考。

**两条边界（刻意不做，不是缺陷）**：

1. **只适用二分类。** 多分类下「一个阈值」没有明确语义（argmax 是对全部类别一起比较的）。
   与其给一个看着能调、含义不明的旋钮，不如不给；传了阈值会**显式报错**，不会静默忽略。
2. **比较用严格大于（`>`）。** sklearn 二分类走 `decision_function > 0`，
   概率恰好 0.5 时判**负类**。若用 `>=`，「阈值 0.5」就会对决策树这类概率能精确取到 0.5 的模型
   改变标签 —— 而阈值功能最基本的承诺正是「取默认值＝与原来行为完全一致」。
   这条口径由 `tests/test_ml_threshold.py::test_default_threshold_matches_argmax` 守着。

**类别不均衡时的优先级。** 9.1 节的 `imbalanced` 信号给出两条路：重训并设
`class_weight=balanced`，或直接调低决策阈值。两者目标相同（让少数类更多地被判出来），
但后者**不用重训**，所以先调阈值看效果、不行再回去重训 —— 调参手册里也会提示这条近路。

**相关文件**：`app/ml_engine/threshold.py`（语义与曲线）、
`app/experiments/service.py::_threshold_basis` / `_inference_threshold_curve`（曲线落点）、
`frontend/src/features/ml/PredictionPanel.tsx`（阈值入口与曲线展示）、
`tests/test_ml_threshold.py`（13 条语义回归用例，不训练模型）。

### 8.5 让"一次切分"变成"五个数"：交叉验证与自动超参搜索

8.1~8.4 的诊断都建立在**单次留出切分**上。数据少的时候，换个 seed 指标就能跳几个点 ——
此时「0.87」到底是模型的能力还是切分的运气，光看一个数分不出来。这一节给两个工具。

**交叉验证（`enable_cv=true`）。** 把**切分前的全量数据**分成 5 份，轮流拿 4 份训练、
1 份验证，得到 5 个数，报均值 ± 标准差。分类用 `StratifiedKFold`（每折类别比例与整体一致），
回归用 `KFold`。

> **防泄漏是这里唯一要紧的事。** 缺失值填补的均值、缩放用的方差、one-hot 的类别表，
> 都必须**只从当折的训练集里学**，不能先在整份数据上学完再切 —— 后者会把测试折的信息
> 提前喂给模型，指标虚高且不可复现。实现上，CV 拿到的是**未拟合**的
> `ColumnTransformer` 骨架（`PreprocessingPipeline.build`）与未拟合的 estimator
> （`ModelAdapter.new_estimator`），拟合发生在 `cross_validate` 的每一折内部。
> 这条契约由 `tests/test_cv_hpo.py::test_cv_as_matrix_does_not_fit_the_pipeline` 钉住。

| 字段 | 含义 |
|---|---|
| `artifacts.cv.folds` | 折数（默认 5） |
| `artifacts.cv.mean` / `std` | 各折指标的均值 / 标准差（`std` 大 = 结果对切分敏感） |
| `artifacts.cv.per_fold` | 每折原始值，形如 `[{metric, values:[5 个数]}]`（能看出是不是被某一折拖低） |
| `artifacts.cv.note` | 「5 折分层 CV，预处理在每折训练集内拟合」—— 口径写在结果里 |

**怎么用这个标准差。** 均值告诉你"大概多少"，标准差告诉你"能不能信"。
两个模型均值差 0.005 而标准差 0.02 时，这个差不值得写成结论。

**自动超参搜索（`POST /experiments/{id}/optimize`）。** 在**该实验当前配置的数据**
（同一套排除列、同一套切分前数据）上，用 `RandomizedSearchCV` 随机采样
`n_iter` 组参数（默认 20），每组做 5 折 CV，按主指标排名返回 Top10。
前端「调参建议」面板里一键搜索、一键把最佳参数填回参数面板。

| 字段 | 含义 |
|---|---|
| `best_params` | 最佳参数（**已去掉 `model__` 前缀**，可直接原样交给训练接口） |
| `best_score` / `top_results[]` | 最佳得分 / Top10（含 `rank`、`mean_score`、`std`、`params`） |
| `n_iter` | **实际**采样组数：候选网格比请求值小时自动收敛到网格大小（不重复采样、不丢警告在日志里） |
| `experiment_id` / `model` / `task` / `metric` | 这次搜索的落点，便于回查「这个分数是哪个实验、按哪个指标排的」 |

搜索空间按模型定义在 `app/ml_engine/cv.py::SEARCH_SPACES`，目前覆盖
随机森林 / HistGradientBoosting / 逻辑回归 / KNN 四个模型；
`GET /ml/catalog` 的 `search_spaces` 会透出「哪些参数在搜索范围内」，
**不在表里的模型不会静默搜索**，接口明确报错。

> **搜索不写回实验配置。** `optimize` 只返回结果，改不改由你决定 ——
> 搜索是在同一份数据上多看了很多次，那个 `best_score` 本身已经带一点乐观偏差，
> 它适合用来**缩小范围**，不适合直接当最终泛化指标。要报最终数字，请用搜到的参数
> **重训一次**，看留出测试集。

### 8.6 指标的可信度：95% 置信区间与配对差异

**单次的置信区间。** 训练结束时会把测试集的逐条预测留档
（`artifacts.test_predictions`），再用 Bootstrap 重采样算出指标的 95% 区间：
分类给 accuracy / f1 / roc_auc，回归给 r2 / rmse。界面在每个指标卡下方直接显示
`95% CI [下界, 上界]`，HTML 报告里多一列「95% CI」。

样本少于 30 行时不给区间（重采样已无意义），Bootstrap 次数按样本量自适应
（≤5000 行 1000 次、≤20000 行 400 次、更大 200 次），避免大数据上把训练拖慢。

> **上限提醒**：`test_predictions` 最多留 5 万行，超出时存**随机抽样版**并在
> `note` 里写明 —— 静默换成子集等于让区间建立在一份没交代的数据上，宁可写明。

**两个模型谁更好：看配对差异。** 模型对比页在两次运行**可比**时额外给出
`paired_delta`：对同一批测试样本**共用同一组重采样下标**，算「A − B」这个差值的区间。

| 字段 | 含义 |
|---|---|
| `delta` | A 减 B 的点估计 |
| `lower` / `upper` | 差值的 95% 区间 |
| `significant` | 区间是否**不含 0**（含 0 ⇒ 差异不显著） |
| `note` | 不可比时说明原因 |

> **为什么必须配对。** 两次独立 Bootstrap 各自带一份重采样噪声，相减会把两份噪声叠进去，
> 区间被人为撑宽。配对（同一组下标）抵消掉"这一批样本本身好不好分"的影响，
> 剩下的才是两个模型的真实差距。
>
> **判可比的前提**（缺一条就返回 `null`，而不是硬算）：两次运行都成功、
> `test_predictions.rows` 相同、且 `y_true` **逐行相等**、样本 ≥30 行。
> 不同切分上的两个 Δ 混着"样本不一样"和"模型不一样"两件事，配对前提不成立。

---

## 9. 已知限制（写入论文「局限与展望」）

以下问题**已知且刻意未在本阶段修复**，因为修复收益低于对既有约定的破坏风险：

1. **版本缓存与测试隔离** —— `VersionFrameCache` 是进程级单例，key 仅
   `(dataset_id, version)`。生产环境一个进程只服务一个存储后端，语义成立；
   但测试用例各自新建内存库（`dataset_id` 总从 1 开始）+ 临时 storage 时，
   会命中上一个用例的旧快照。**已通过 `tests/conftest.py` 的
   `_isolate_version_frame_cache` fixture 在测试侧隔离**，生产代码未改。
2. ~~**全量推理结果不可导出**~~ —— **已修复**：
   `GET /experiments/runs/{run_id}/predict-export?format=csv|parquet` 返回全量结果
   （原始列 + `prediction` + 各 `prob_*` 列），与预览共用同一次推理，口径必然一致。
3. **无鉴权** —— 与其他模块一致，接口不做身份校验（安全类问题统一写入论文展望）。
4. **pickle 产物无版本戳** —— `model.pkl` / `pipeline.pkl` 未记录 sklearn 版本；
   跨版本升级后反序列化可能失败，表现为"推理报错但模型文件存在"。
   完整性方面已补一层 HMAC 信封（`XLB1` + `hmac_sha256` + payload，
   密钥取自环境变量 `ML_MODEL_SIGNING_SECRET`，未配置时用内置常量）：
   文件被截断或被别的 pickle 顶替会在加载时明确报错，而不是给出一份错误的预测。
   **注意**：未配置环境变量时它不是安全边界——只防损坏与误替换，不防蓄意伪造。
5. **同步阻塞** —— 训练同步执行，长任务会占用请求线程（进程内单 worker 的既有约束）。
   交叉验证与超参搜索同样同步执行，**默认关闭**正是出于这一点。
6. **`test_predictions` 留档上限 5 万行** —— 超出时存随机抽样版并在 `note` 中写明，
   置信区间因此建立在抽样子集上。这是为了不让报告产物无限膨胀，需要精确区间时
   请自行在导出结果上重算。
7. **报告内嵌图只进 HTML** —— `ReportSection.images` 只有 HTML 导出器消费，
   Markdown 与 PDF 仍走表格/文本（PDF 渲染 base64 位图体积与兼容性都不划算）。
   这不是遗漏，是有意为之；代价是 HTML 报告依赖 `matplotlib`（新增依赖）。

---

## 附：相关文件索引

| 内容 | 文件 |
|---|---|
| 教学元数据（参数表来源，含 `search_spaces`） | `backend/app/ml_engine/metadata.py` |
| 预处理管道（`as_matrix` / `build`） | `backend/app/ml_engine/preprocessing.py` |
| **交叉验证与超参搜索** | `backend/app/ml_engine/cv.py`（`cv_evaluate` / `random_search` / `SEARCH_SPACES`） |
| **指标重采样工具** | `backend/app/ml_engine/evaluation.py`（`bootstrap_confidence_interval`） |
| 全流程服务 | `backend/app/experiments/service.py`（`_cross_validate` / `optimize` / `_paired_delta`） |
| 模型对比（配对差异） | `backend/app/experiments/comparator.py` |
| **训练报告图表（HTML 内嵌）** | `backend/app/experiments/run_report.py` + `backend/app/reports/html.py` |
| 教学接口 | `backend/app/api/v1/experiments.py`（`GET /ml/catalog`、`POST /experiments/{id}/optimize`） |
| Agent 教学工具 | `backend/app/tools/ml_tools.py`（`MlExplainConfigTool`） |
| 错误提示对照 | `backend/app/ml_engine/exceptions.py` + 各 `_preflight` |
| 模块审查报告 | `backend/ML_MODULE_REVIEW.md` |
| **前端 · 教学目录面板**（流程/参数/指标） | `frontend/src/features/ml/ParamGuide.tsx` |
| **前端 · 参数调整面板**（核心/进阶 + 类型化控件） | `frontend/src/features/ml/ParamTuner.tsx` |
| **前端 · 预处理面板**（填补/编码/缩放） | `frontend/src/features/ml/PreprocessPanel.tsx` |
| **前端 · 调参建议**（匹配 playbook + 一键应用 + 自动搜索） | `frontend/src/features/ml/TuningAdvice.tsx` |
| **前端 · 交叉验证表与 CI 展示** | `frontend/src/features/ml/ExperimentResult.tsx` |
| **前端 · 模型对比（配对差异）** | `frontend/src/features/ml/ModelComparison.tsx` |
| 回归用例（CV / 搜索 / 防泄漏） | `backend/tests/test_cv_hpo.py` |
| 回归用例（报告内嵌图表） | `backend/tests/test_run_report.py` |
| **前端 · 训练过程透明化**（读 `run.artifacts`） | `frontend/src/features/ml/TrainTrace.tsx` |
| **前端 · 页面接线**（含 `test_size` 输入） | `frontend/src/pages/ML/index.tsx` |
| **前端 · 类型镜像** | `frontend/src/types/ml.ts` |
| **前端 · 接口封装** | `frontend/src/api/ml.ts`（`getMlCatalog`） |
| **前端 · 样式** | `frontend/src/styles/visual.css`（「机器学习·教学透明化」小节） |
