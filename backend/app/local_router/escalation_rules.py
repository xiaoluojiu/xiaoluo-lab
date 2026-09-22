"""升级判定的 **L0 结构规则层** —— 用零参数规则回答「这句话平台接得住吗」。

为什么这一层必须是规则、而不是让 L1 小模型学
----------------------------------------------
「要不要升级」曾被当作一个 32 类分类任务里的 5 个标签去学，结果 escalate F1 只有
51.3%，漏升级率 65.5%。逐条看样本才发现：**五类升级原因各自都有结构性/词表性的
相关物**，而不是需要语义理解的判别 ——

  multi_step        连词序列（先/再/然后/最后/如果…就/并/同时）+ 句子必然偏长
  conflict          转折否定（但/不要/不用/前提/否则）
  param_dependency  平台参数空间之外的 ML 内部词（学习率/退火/warmup/梯度/显存…）
  ambiguous         极短 + 只有动作没有具体宾语（「优化一下」「按老规矩来一份」）
  out_of_scope      平台没有的外部系统 / 模态（S3/Kafka/Oracle/微信/3D/视频…）

让 char n-gram 词法模型去逼近一个**本质是结构统计**的边界，是把容量浪费在错的表述上。
实测（`scripts/router/escalation_rules_eval.py`，844 条样本，模板级 groupCV 口径）：

  escalate 精确率 100.0% / 召回率 96.6% / F1 98.2%   ← 本模块（fp=0, fn=1）
  escalate F1 51.3% / 漏升级率 65.5%                 ← 同数据上最好的纯学习基线

并且叠加后 **① 可执行决策 79.8%、工具选择 80.1% 完全不变** —— 证明「选工具」和
「该不该升级」是两条正交能力，可以各自用最便宜的手段解决。

适用边界（必须诚实标注）
------------------------
1. 上述数字是在**同一作者手写的 30 条升级语料**上测得的，存在**循环论证**风险：
   语料与规则出自同一人，规则必然贴合语料。真实泛化能力**必须在 AgentStore 采集到
   真实会话后重新验证**，本模块的数字不能直接当作线上指标。
2. 已知漏判 1 条：`conflict` 里「把这个数据集的行数改成 100 行然后保存」 —— 它结构上
   与普通单步调用无异，冲突点在**语义**（行数不可变）。这类残留才是真正需要 L1 的。
3. 词表是**短语级**的，绝不能退化成单字。实测把 `弄/搞/处理` 当单字标记后，
   「预**处理**参数」「想**搞**懂流程步骤」全部误伤，escalate F1 从 87.5% 掉到 83.6%。

本模块零依赖、零参数、零耗时（纯 `in` 判断），只读 `RouterRequest.utterance`。
"""

from __future__ import annotations

from app.local_router.contract import EscalationReason

__all__ = [
    "CONTRAST_MARKERS",
    "DOMAIN_CONTENT",
    "EXTERNAL_TERMS",
    "MAX_AMBIGUOUS_LEN",
    "MIN_MULTI_STEP_LEN",
    "ML_INTERNAL_TERMS",
    "SEQUENCE_MARKERS",
    "VAGUE_PHRASES",
    "detect_escalation",
]


# 多步编排的连词/条件标记。「就」「并」在单步句里也常见 ⇒ 要求合计 >= 2。
SEQUENCE_MARKERS: tuple[str, ...] = (
    "先", "再", "然后", "接着", "之后", "最后", "并", "同时", "如果", "就",
)

# 与平台约束冲突的转折否定。
CONTRAST_MARKERS: tuple[str, ...] = (
    "但", "不过", "却", "不要", "不能", "不用", "前提", "否则",
)

# 平台参数空间之外的 ML 内部词 —— 命中即说明用户下沉到了平台未建模的实现细节。
ML_INTERNAL_TERMS: tuple[str, ...] = (
    "学习率", "退火", "warmup", "超参数", "梯度", "损失函数", "显存", "batch", "早停",
    "正则", "权重衰减", "优化器", "激活函数", "网络结构", "层数", "神经元", "epoch",
    "微调", "蒸馏", "剪枝", "量化",
)

# 平台没有的外部系统 / 模态 / 数据源（小写匹配）。
EXTERNAL_TERMS: tuple[str, ...] = (
    "s3", "存储桶", "oracle", "kafka", "mysql", "redis", "hadoop", "hive", "es 集群",
    "微信", "钉钉", "飞书", "邮件", "短信", "3d", "视频", "照片", "图片", "语音",
    "股票", "实时流", "gpt", "机器人", "爬虫", "部署上线", "上线", "云函数",
)

# 「光有动作、没有宾语」的模糊说法。⚠️ 必须短语级，不能退化成单字（见模块 docstring）。
VAGUE_PHRASES: tuple[str, ...] = (
    "这个", "那个", "那个啥", "弄一下", "弄好点", "优化一下", "处理一下", "老规矩", "你懂的",
)

# 具体领域名词 —— 出现即说明宾语明确，指代/平台能接住，不该判模糊。
DOMAIN_CONTENT: tuple[str, ...] = (
    "数据", "模型", "报告", "图", "画", "聚合", "统计", "评估", "训练", "预测",
    "分布", "导出", "质量", "列", "表", "清洗", "特征", "可视化", "eda",
    "缺失", "异常", "相关", "平台",
)

MAX_AMBIGUOUS_LEN = 12
# 多步编排必然是一句里塞了多个任务 ⇒ 结构上必然更长。长度下限挡掉
# 「接着再统计一遍」这类含 2 个连词但其实是单步指代的话。
#
# ⚠️ 已知代价（2026-09-22 实测记录）：`先清洗缺失值然后再做一个分布图`（15 字、3 个连词、
# 确实是多步）被这里的下限漏判 ⇒ 交给 L1，L1 只会选出其中一个工具（data.clean）。
# 把下限下调到 15 能立刻收回这条，但会同时放进「接着再统计一遍」这类单步指代。
# **不要凭一句例子改这个常量** —— 先积累 shadow 真实样本，用
# `scripts/router/analyze_shadow.py` 里「会出错(工具不符)」的条数来权衡。
MIN_MULTI_STEP_LEN = 16


def _hits(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for w in words if w in text)


def detect_escalation(utterance: str) -> EscalationReason | None:
    """判定一句话是否超出平台能力；是则给出原因，否则 None。

    判定顺序即优先级。顺序不是任选的：多步编排的判断条件最强（连词密度 + 长度），
    放最前；模糊判定最弱（只靠长度 + 短语），放最后兜底。
    """
    text = (utterance or "").strip()
    if not text:
        return None
    low = text.lower()

    # 1) multi_step：连词密度 —— 一句话里同时出现多个编排标记，且够长
    if len(text) >= MIN_MULTI_STEP_LEN and _hits(text, SEQUENCE_MARKERS) >= 2:
        return EscalationReason.MULTI_STEP
    # 2) param_dependency：命中平台未建模的 ML 内部词
    if _hits(low, ML_INTERNAL_TERMS):
        return EscalationReason.PARAM_DEPENDENCY
    # 3) conflict：转折否定
    if _hits(text, CONTRAST_MARKERS):
        return EscalationReason.CONFLICT
    # 4) out_of_scope：外部系统 / 模态
    if _hits(low, EXTERNAL_TERMS):
        return EscalationReason.OUT_OF_SCOPE
    # 5) ambiguous：短 + 模糊说法 + 没有具体宾语（有领域名词 ⇒ 宾语明确，能接住）
    if (len(text) <= MAX_AMBIGUOUS_LEN
            and _hits(text, VAGUE_PHRASES)
            and not _hits(low, DOMAIN_CONTENT)):
        return EscalationReason.AMBIGUOUS
    return None
