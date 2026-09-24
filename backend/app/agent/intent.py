"""Agent 层 · 统一意图分类（标准化内核在 Agent 侧的落点）。

改造前的混乱
------------
「用户这句话属于哪类任务」这件事在平台里有 **三份各自维护的关键词表**：

===========================  ==============================================
位置                          用途
===========================  ==============================================
``runtime.AgentRuntime``      ``DATA_TERMS`` / ``GREETINGS``：决定走聊天还是工具
``context.ContextBuilder``    ``wants_train`` / ``wants_eda`` / …：决定注入哪些候选工具
``planner.AgentPlanner``      ``_rule_plan`` 里又写了一遍：决定降级时跑什么计划
===========================  ==============================================

后果是**同一个词在三处可能只改了两处**：例如「关联 / 拼接 / 宽表」早期只加进了
运行时路由，没加进候选工具集，于是请求进了工具流程、却因 ``data.merge`` 不在
候选集里被 ``_validate`` 判非法 —— 表现为「工具已注册但 Agent 不调用」。

本模块把意图判定收敛成一份：

    classify(text) -> Decision[Intent]

* ``Intent`` 复用 :mod:`app.local_router.contract` 的封闭枚举（能力域词表只有一份）；
* 关键词表只有一份，三处调用方共享；
* 返回 :class:`~app.core.contracts.Decision`，把「凭什么这么判」带回给调用方。
"""

from __future__ import annotations

from typing import Any

from app.core.contracts import Decision
from app.local_router.contract import Intent

__all__ = [
    "COMPREHENSIVE_KEYWORDS",
    "GREETINGS",
    "INTENT_KEYWORDS",
    "MERGE_KEYWORDS",
    "MODEL_HINTS",
    "QUALITY_KEYWORDS",
    "TOOL_DOMAINS",
    "classify",
    "hits",
    "is_modeling",
    "model_hint",
    "tool_domains_of",
    "wants_merge",
    "wants_modeling",
]

#: 问候/闲聊：命中即走普通对话，不消耗任何数据工具。
GREETINGS: frozenset[str] = frozenset(
    {
        "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey",
        "早上好", "下午好", "晚上好", "谢谢", "感谢", "在吗",
        "你是谁", "你叫什么", "再见", "拜拜",
    }
)

#: 能力域 -> 关键词。**唯一真源**：新增一种说法只改这里，三处调用方同时生效。
INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.DATASET: (
        "数据", "数据集", "csv", "excel", "xlsx", "表格", "字段", "列", "行",
        "读取", "导入", "预览", "画像", "profile", "inspect", "版本", "质量",
    ),
    Intent.DATA_TRANSFORM: (
        "处理", "清洗", "缺失", "重复", "去重", "转换", "合并", "筛选", "聚合",
        "排序", "关联", "join", "merge", "拼接", "宽表", "连接", "外键", "导出",
        "填充", "拆分",
    ),
    Intent.EDA: (
        "分析", "统计", "描述", "分布", "相关", "相关性", "可视化", "图表", "eda",
        "异常", "离群", "趋势", "探索", "直方图", "散点", "热力图", "分布图",
    ),
    Intent.ML: (
        "训练", "模型", "预测", "分类", "回归", "机器学习", "特征", "特征工程",
        "目标列", "建模", "评估", "准确率", "auc", "聚类", "实验", "python",
        "pandas", "监督",
    ),
    Intent.WORKFLOW: (
        "workflow", "工作流", "流程", "pipeline", "编排", "流水线", "节点",
    ),
    Intent.REPORT: (
        # ★「总结 / 结论」刻意**不在**这里。
        # 报告是**要产出一份文件**；「总结一下」「给个结论」要的是一段回答。
        # 早先两者共用一组关键词，于是「帮我总结一下分析结果」会静默产出一份 PDF
        # 写进报告中心 —— 用户没要文件，却多了一个交付物。
        # 判定标准只用「报告 / 汇报 / PDF / 导出」这类**明确指向文件**的说法。
        "报告", "report", "pdf", "汇报", "导出报告", "导出 pdf",
    ),
}

#: 规则规划器需要的**细分说法**。它们分属不同的 Intent（「重复」在 DATA_TRANSFORM、
#: 「异常」在 EDA），但规划时要按细分意图选工具，所以在这里集中定义 ——
#: 与 ``INTENT_KEYWORDS`` 一样只有一份，禁止在 planner / runtime / 前端各抄一遍。
QUALITY_KEYWORDS: tuple[str, ...] = ("质量", "quality", "缺失", "重复", "空值", "异常")
COMPREHENSIVE_KEYWORDS: tuple[str, ...] = (
    "智能分析", "全面分析", "完整分析", "关键统计", "问题摘要", "综合",
)

#: 能力域 -> 该能力需要的工具类目（供候选工具集构造使用）。
#: 与 :data:`INTENT_KEYWORDS` 一一对应，避免「路由放行了、工具没注入」的错配。
TOOL_DOMAINS: dict[Intent, tuple[str, ...]] = {
    Intent.DATASET: ("dataset",),
    Intent.DATA_TRANSFORM: ("data",),
    Intent.EDA: ("eda",),
    Intent.ML: ("ml",),
    Intent.WORKFLOW: ("workflow",),
    Intent.REPORT: ("report", "eda"),
}

#: 任何数据分析任务都必须能看到的能力核心（避免检索式召回把交付工具挡在外面）。
CORE_TOOLS: tuple[str, ...] = (
    "dataset.inspect", "dataset.schema", "dataset.quality", "dataset.profile",
    "eda.describe", "eda.correlation", "eda.visualize", "report.generate",
)


#: 多表关联的专用说法。它们同属 ``DATA_TRANSFORM``，但规则规划要为它走一条
#: 独立分支（``data.merge`` 需要两个 dataset_id），所以单独保留一份判定。
MERGE_KEYWORDS: tuple[str, ...] = (
    "合并", "关联", "拼接", "连接", "宽表", "join", "merge", "外键",
)

#: 自然语言 -> 模型名（仅当用户**明确**点名算法类型时才用，其余一律 "auto"）。
#: 硬编一个监督模型是事故来源：请求里既没说回归也没说分类时，写死
#: logistic_regression 会让 detect_task 判出的回归任务被静默换成聚类。
MODEL_HINTS: tuple[tuple[str, str], ...] = (
    ("回归", "linear_regression"),
    ("regression", "linear_regression"),
    ("分类", "logistic_regression"),
    ("classification", "logistic_regression"),
    ("聚类", "kmeans"),
    ("clustering", "kmeans"),
)


def wants_merge(text: str) -> bool:
    return any(k in (text or "").lower() for k in MERGE_KEYWORDS)


def wants_modeling(text: str) -> bool:
    return bool(hits(text).get(Intent.ML))


def model_hint(text: str) -> str:
    """用户明确点名算法类型时返回模型名，否则 ``"auto"``。"""
    low = (text or "").lower()
    for keyword, model in MODEL_HINTS:
        if keyword in low:
            return model
    return "auto"


def hits(text: str) -> dict[Intent, list[str]]:
    """全部命中的能力域（供「注入哪些候选工具」使用）。

    路由只需要**一个**意图（``classify``），但候选工具集需要**全部**命中的
    意图：用户说「建模并生成报告」时，两类工具都得在候选集里，
    否则规划器规划 report.generate 会被判非法。
    """
    low = (text or "").lower()
    out: dict[Intent, list[str]] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        matched = [k for k in keywords if k in low]
        if matched:
            out[intent] = matched
    return out


def classify(text: str, *, has_datasets: bool = False) -> Decision[Intent]:
    """判定一句话的能力域。

    判定顺序（先特判后通用，命中即返回）：

    1. 空文本 / 问候语 → ``Intent.CHAT``（source=``greeting``）
    2. 关键词命中 → 命中最多关键词的能力域（source=``keyword``）
    3. 会话已绑定数据集 → ``Intent.DATASET``（source=``bound_dataset``，
       这是「关键词漏召回」的兜底：用户说「关联一下」不该退化成一句闲聊）
    4. 都没命中 → ``Intent.CHAT``（source=``fallback``）

    并列时按 ``INTENT_KEYWORDS`` 的声明顺序取第一个（DSL 顺序即优先级）。
    """
    raw = (text or "").strip()
    low = raw.lower()
    if not low:
        return Decision.of(Intent.CHAT, source="empty", confidence=1.0,
                           reason="空输入，按普通对话处理")
    if low in GREETINGS:
        return Decision.of(Intent.CHAT, source="greeting", confidence=1.0,
                           reason=f"命中问候语「{raw}」")

    hits: dict[Intent, list[str]] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        matched = [k for k in keywords if k in low]
        if matched:
            hits[intent] = matched
    if hits:
        # 命中词多者优先；数量相同时按声明顺序（REPORT/WORKFLOW 优先于泛化的 DATASET）
        best = max(hits.items(), key=lambda kv: (len(kv[1]), -_order(kv[0])))
        intent, matched = best
        return Decision.of(
            intent, source="keyword", confidence=_confidence(len(matched)),
            reason=f"命中 {intent.value} 类关键词：{'、'.join(matched[:5])}"
                   f"{'…' if len(matched) > 5 else ''}",
            evidence={"matched": matched, "all_hits": {k.value: v for k, v in hits.items()}},
        )

    if has_datasets and len(raw) >= 4:
        return Decision.of(
            Intent.DATASET, source="bound_dataset", confidence=0.5,
            reason="会话已绑定数据集，关键词未命中时的兜底（按数据处理请求处理）",
        )
    return Decision.of(Intent.CHAT, source="fallback", confidence=0.4,
                       reason="未命中任何能力域关键词，按普通对话处理")


def _order(intent: Intent) -> int:
    try:
        return list(INTENT_KEYWORDS).index(intent)
    except ValueError:
        return len(INTENT_KEYWORDS)


def _confidence(n_matched: int) -> float:
    # 1 个词 0.6，2 个 0.75，3 个及以上 0.9 —— 命中越多越可信，但永远不到 1.0
    # （关键词法是启发式，不是语义理解）。
    return {1: 0.6, 2: 0.75}.get(n_matched, 0.9)


def is_modeling(decision: Decision[Intent] | Intent | None) -> bool:
    """该意图是否属于建模类（决定是否必须做目标列 Pre-flight 检查）。"""
    intent = decision.value if isinstance(decision, Decision) else decision
    return intent == Intent.ML


def tool_domains_of(decision: Decision[Intent] | Intent | None) -> tuple[str, ...]:
    """该意图需要注入的工具类目。"""
    intent = decision.value if isinstance(decision, Decision) else decision
    if intent is None:
        return ()
    return TOOL_DOMAINS.get(intent, ())


def explain(decision: Decision[Intent]) -> dict[str, Any]:
    """给事件 payload / 日志用的一行解释。"""
    return {
        "intent": str(decision.value),
        "source": decision.source,
        "confidence": round(decision.confidence, 3),
        "reason": "；".join(decision.reasons),
    }
