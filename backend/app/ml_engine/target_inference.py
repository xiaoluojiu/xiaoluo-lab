"""目标列推断：把「要预测什么」从诉求语义 + 列名语义 + 数据分布里推出来。

真实事故（2026-09-23，dataset 9「航空公司出发延误预测（回归）」）：

    规划器在规划阶段**看不到列名**（`ContextBuilder` 明确禁止读 Schema），
    而 `ml.detect_task` 又拒不替调用方选目标列 —— 只回传一个候选列表。
    于是链路死锁在「我不知道该预测哪一列」：
    步骤 6 `ml.detect_task` 失败 → 计划无法续接 → 整次运行空手而归。

本模块补上缺失的那一环：**按证据强度分级推断，并且永远给出理由**。

证据强度（从强到弱）：

    ① explicit            调用方显式指定        confidence 1.00
    ② naming_convention   target/label/y/class   confidence 0.95
    ③ goal_match          诉求/数据集名语义命中   confidence 0.85
    ④ single_candidate    排除日历·时间·标识列后唯一候选  confidence 0.60

两条关键设计：

* **语义命中优先于启发式排除**。列名叫 `ResponseTime` 时，它会被「时间分量」
  规则排除；但只要诉求里出现「响应时间」，语义命中就直接胜出 —— 排除规则
  只在**兜底排序**里生效，不会否掉有明确语义证据的列。
* **歧义就不猜**。命中多列且无法用任务类型收敛时返回 `column=None`，
  让上层继续走「回传候选 + 要用户确认」，绝不静默选中一个错的目标列。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl

__all__ = [
    "TargetGuess",
    "feature_like_reason",
    "goal_concepts",
    "infer_task_from_column",
    "name_tokens",
    "n_unique_map",
    "recommend_target",
    "target_candidates",
]


# --------------------------------------------------------------------------- #
# 列名分词
# --------------------------------------------------------------------------- #
# 复用 Quality 层的标准实现（app.quality.semantics.column_tokens）。
# 此前这里有一份逐字相同的副本：两处分词规则一旦漂移，
# 「目标列推断」与「字段语义判定」就会对同一个列名给出不同解释。
from app.quality.semantics import column_tokens as name_tokens  # noqa: F401  (保持既有导出名)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# --------------------------------------------------------------------------- #
# 类型 → 任务
# --------------------------------------------------------------------------- #


def infer_task_from_column(dtype: Any, n_unique: int) -> str | None:
    """由目标列的**类型与基数**判断任务类型（判不了返回 None）。"""
    if dtype == pl.String or dtype == pl.Boolean or dtype == pl.Categorical:
        return "classification"
    if dtype.is_integer() and n_unique <= 20:
        return "classification"
    if dtype.is_numeric():
        return "regression"
    return None


# --------------------------------------------------------------------------- #
# 目标列命名约定
# --------------------------------------------------------------------------- #

#: 目标列命名约定：ml.detect_task 未显式指定 target 时的第一顺位候选判定
_TARGET_COLUMN_NAMES = ("target", "label", "y", "class", "cls", "outcome", "response")
#: 形如 `is_fraud_target` / `target_price` / `y_` 的命名也会被认出来。
_TARGET_NAME_AFFIXES = ("_target", "_label", "_y", "target_", "label_")


def _convention_targets(columns: list[str]) -> list[str]:
    lowered = {str(c).lower(): str(c) for c in columns}
    return [
        lowered[k]
        for k in lowered
        if k in _TARGET_COLUMN_NAMES or k.endswith(_TARGET_NAME_AFFIXES)
    ]


# --------------------------------------------------------------------------- #
# 明显不该当目标的列
# --------------------------------------------------------------------------- #

#: 日历 / 时间分量。它们是**特征**，不是预测对象；`Month` / `DayofMonth` /
#: `DayOfWeek` / `CRSDepTime` 这类编码列经常被「类型 + 基数」规则误判成回归候选。
_CALENDAR_TOKENS = frozenset(
    {
        "month", "months", "day", "days", "date", "dates", "year", "years",
        "week", "weeks", "hour", "hours", "minute", "minutes", "second",
        "seconds", "time", "times", "timestamp", "quarter", "weekday",
        "weekend", "moy", "dow", "dom", "doy",
    }
)

#: 标识 / 编号列。列名分词后命中即排除（`cust_id` / `row_index` / `uuid`）。
_ID_TOKENS = frozenset(
    {"id", "ids", "index", "idx", "rowid", "row", "key", "uuid", "guid",
     "code", "seq", "serial", "pk", "fk"}
)


def _id_like(column: str) -> bool:
    low = str(column).lower()
    return low in _ID_TOKENS or low.endswith("_id") or bool(set(name_tokens(column)) & _ID_TOKENS)


def feature_like_reason(column: str) -> str | None:
    """列名看起来像「特征而非目标」时给出原因，否则 None。"""
    tokens = set(name_tokens(column))
    if tokens & _CALENDAR_TOKENS:
        return "日历/时间分量（是特征而非预测目标）"
    return None


# --------------------------------------------------------------------------- #
# 诉求语义 → 列名概念
# --------------------------------------------------------------------------- #

#: 概念 → 别名（中英混写）。中文按子串匹配，英文按词元匹配。
#: 只收录「经常被当成预测目标」的概念；刻意**不含**裸 "time"
#: （否则 `CRSDepTime` 会被「时间」概念命中，与真正的目标列抢票）。
_GOAL_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("delay", ("delay", "delays", "延误", "晚点", "迟延", "延迟")),
    ("price", ("price", "fare", "amount", "价格", "票价", "金额")),
    ("cost", ("cost", "expense", "成本", "花费", "费用")),
    ("sales", ("sales", "sale", "revenue", "turnover", "销量", "销售额", "营收", "收入")),
    ("duration", ("duration", "elapsed", "时长", "耗时", "历时", "持续时间")),
    ("churn", ("churn", "attrition", "流失", "离网")),
    ("fraud", ("fraud", "欺诈", "作弊", "刷单")),
    ("risk", ("risk", "default", "风险", "违约", "逾期")),
    ("score", ("score", "得分")),
    ("rating", ("rating", "review", "评级", "评分")),
    ("quantity", ("quantity", "qty", "count", "数量", "件数", "笔数")),
    ("age", ("age", "年龄")),
    ("demand", ("demand", "需求")),
    ("temperature", ("temperature", "temp", "温度", "气温")),
    ("profit", ("profit", "margin", "利润", "毛利")),
    ("conversion", ("conversion", "转化")),
    ("click", ("click", "ctr", "点击")),
    ("load", ("load", "traffic", "负载", "流量", "吞吐")),
    ("survival", ("survival", "存活")),
    ("sentiment", ("sentiment", "情感", "情绪")),
)

_TASK_LABELS = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def _alias_hits(alias: str, column: str, tokens: set[str]) -> bool:
    """别名是否命中列名。

    英文别名要求**词元级**相等（`age` 不能命中 `average`），
    长度 ≥ 4 时允许词元前后缀（`delay` 命中 `depdelay` / `totaldelay`）；
    中文别名按子串匹配（`流失` 命中 `是否流失`）。
    """
    if _CJK_RE.search(alias):
        return alias in str(column).lower()
    if alias in tokens:
        return True
    if len(alias) >= 4:
        return any(t.startswith(alias) or t.endswith(alias) for t in tokens)
    return False


def goal_concepts(text: str) -> set[str]:
    """从一段自然语言里读出涉及的概念集合（空集表示读不出）。"""
    low = str(text or "").lower()
    tokens = set(name_tokens(text))
    found: set[str] = set()
    for concept, aliases in _GOAL_CONCEPTS:
        for alias in aliases:
            if _CJK_RE.search(alias):
                if alias in low:
                    found.add(concept)
                    break
            elif alias in tokens or (
                len(alias) >= 4
                and any(t.startswith(alias) or t.endswith(alias) for t in tokens)
            ):
                found.add(concept)
                break
    return found


# --------------------------------------------------------------------------- #
# 列统计
# --------------------------------------------------------------------------- #


def n_unique_map(df: pl.DataFrame, columns: list[str]) -> dict[str, int]:
    """一次 select 批量取唯一值数（逐列扫描在千万行表上代价过高）。"""
    if not columns:
        return {}
    try:
        row = df.select([pl.col(c).n_unique().alias(c) for c in columns]).row(0, named=True)
    except Exception:  # noqa: BLE001 - 退化到逐列，宁可慢也不报错
        return {c: int(df[c].n_unique()) for c in columns}
    return {c: int(row[c]) for c in columns if row.get(c) is not None}


def target_candidates(df: pl.DataFrame) -> dict[str, list[str]]:
    """列出「类型上讲得通」的可选目标列（保留原始语义，供上层问用户）。

    刻意**不**在这里做排除（日历列、标识列照列）——这是给用户看的原始候选面，
    真正的排序与排除在 `recommend_target` 里做。
    """
    columns = [c for c in df.columns if not _id_like(c)]
    counts = n_unique_map(df, columns)
    rows = max(1, df.height)
    # 连续目标的门槛随数据规模收缩：30 行的表里不可能有「唯一值 > 50」的列，
    # 固定阈值会让小表一个候选都给不出。
    cont_threshold = min(50, max(2, rows // 2))
    regression: list[str] = []
    classification: list[str] = []
    for c in columns:
        n = counts.get(c, 0)
        numeric = df.schema[c].is_numeric()
        if numeric and n > cont_threshold:
            regression.append(c)
        elif 1 < n <= 200 and not (numeric and n == rows):
            classification.append(c)
    return {"regression": regression, "classification": classification}


# --------------------------------------------------------------------------- #
# 推断结果
# --------------------------------------------------------------------------- #


@dataclass
class TargetGuess:
    """一次目标列推断的完整结论（含理由与备选，便于回执与人工复核）。"""

    column: str | None = None
    task: str | None = None
    #: explicit / naming_convention / goal_match / single_candidate
    source: str | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    #: 其他值得考虑但没被选中的列
    alternatives: list[str] = field(default_factory=list)
    #: 被显式排除的列及原因
    excluded: list[dict[str, str]] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.column)

    def as_receipt(self) -> dict[str, Any]:
        """可序列化的回执（写进 ToolResult.data，供 Agent 转述与追溯）。"""
        return {
            "target": self.column,
            "task": self.task,
            "target_source": self.source,
            "target_confidence": round(self.confidence, 2),
            "reasons": list(self.reasons),
            "alternatives": list(self.alternatives),
            "excluded_columns": list(self.excluded),
        }


def recommend_target(
    df: pl.DataFrame,
    *,
    goal: str = "",
    dataset_name: str = "",
    dataset_hint: str | None = None,
    explicit: str | None = None,
) -> TargetGuess:
    """推断最可能的预测目标列。

    Args:
        df: 数据集（只需要 schema + 唯一值数，不需要全量消费）。
        goal: 用户诉求原文（用于语义匹配）。
        dataset_name: 数据集名称（名称里常直接写明预测对象，如「出发延误预测」）。
        dataset_hint: 数据集名称提示的任务类型（regression/classification/clustering）。
        explicit: 调用方显式指定的目标列（优先级最高）。
    """
    columns = [str(c) for c in df.columns]
    rows = int(df.height)
    schema = dict(df.schema)

    # ① 显式指定
    if explicit:
        if explicit not in df.columns:
            return TargetGuess(
                column=explicit,
                reasons=[f"调用方指定的目标列 {explicit!r} 不在数据中"],
            )
        counts = n_unique_map(df, [explicit])
        return TargetGuess(
            column=explicit,
            task=infer_task_from_column(schema[explicit], counts.get(explicit, 0)),
            source="explicit",
            confidence=1.0,
            reasons=[f"调用方显式指定目标列 {explicit}"],
        )

    # 唯一值数在下面每个分支都要用，只算一次（千万行表上这是主要开销）
    counts = n_unique_map(df, columns)

    def task_of(col: str) -> str | None:
        return infer_task_from_column(schema[col], counts.get(col, 0))

    def label_of(task: str | None) -> str:
        return _TASK_LABELS.get(task or "", task or "未知")

    reasons: list[str] = []
    excluded: list[dict[str, str]] = []

    def unique_key(col: str) -> bool:
        # 每行一个取值 ⇒ 是主键/hash，不能当目标（小表除外：n 小必然接近唯一）
        return rows > 20 and counts.get(col, 0) == rows

    # ② 命名约定
    convention = _convention_targets(columns)
    if len(convention) == 1:
        col = convention[0]
        reasons.append(f"按命名约定识别到唯一目标列 {col}")
        return TargetGuess(
            column=col,
            task=task_of(col),
            source="naming_convention",
            confidence=0.95,
            reasons=reasons,
        )
    if len(convention) > 1:
        reasons.append(f"命名约定命中多列 {convention}，继续按诉求语义与数据分布判断")
    else:
        reasons.append("未命中目标列命名约定（target/label/y/class 等）")

    # ③ 诉求语义命中（用户诉求 + 数据集名称一起当意图文本 ——
    #    数据集名称里常常直接写着预测对象：「航空公司出发延误预测」）
    intent = f"{goal or ''} {dataset_name or ''}"
    concepts = goal_concepts(intent)
    matched: list[str] = []
    if concepts:
        reasons.append(
            f"诉求/数据集名称中识别到预测对象语义：{'、'.join(sorted(concepts))}"
        )
        for col in columns:
            if unique_key(col):
                continue
            tokens = set(name_tokens(col))
            for concept, aliases in _GOAL_CONCEPTS:
                if concept not in concepts:
                    continue
                if any(_alias_hits(a, col, tokens) for a in aliases):
                    matched.append(col)
                    break
    else:
        reasons.append("诉求与数据集名称中未识别到明确的预测对象语义")

    if dataset_hint in ("regression", "classification"):
        consistent = [c for c in matched if task_of(c) == dataset_hint]
        conflicts = [c for c in matched if c not in consistent]
    else:
        consistent, conflicts = list(matched), []

    if len(consistent) == 1:
        col = consistent[0]
        reasons.append(
            f"语义命中的列中，唯一与{label_of(dataset_hint)}任务相符的是 {col}"
        )
        if conflicts:
            reasons.append(
                f"另有语义相近但类型不符的列 {conflicts}，未选为目标"
            )
        return TargetGuess(
            column=col,
            task=task_of(col),
            source="goal_match",
            confidence=0.85,
            reasons=reasons,
            alternatives=[c for c in (convention + conflicts) if c != col],
            excluded=excluded,
        )
    if len(consistent) > 1:
        reasons.append(
            f"语义命中多列 {consistent}，无法唯一确定目标列，需调用方指定"
        )
        return TargetGuess(
            column=None,
            task=dataset_hint if dataset_hint in ("regression", "classification") else None,
            reasons=reasons,
            alternatives=consistent + conflicts + convention,
            excluded=excluded,
        )
    if matched:
        # ★ 语义有证据、但与任务类型不符 ⇒ 到此为止。
        # 绝不能改用一个语义无关的列顶替：那会把「预测延误」悄悄换成「预测距离」，
        # 结果看起来完成了、其实答的是另一个问题 —— 比失败危险得多。
        reasons.append(
            f"语义命中的列 {conflicts} 的类型与数据集名称提示的"
            f"{label_of(dataset_hint)}任务不一致，无法据此确定目标列"
        )
        return TargetGuess(
            column=None,
            task=dataset_hint if dataset_hint in ("regression", "classification") else None,
            reasons=reasons,
            alternatives=matched + convention,
            excluded=excluded,
        )

    # ④ 兜底：**只有在完全没有语义证据时**才走。排除日历/时间/标识列后，
    #    若类型相符的候选只剩一个，就用它。
    if dataset_hint in ("regression", "classification"):
        pool_c: list[str] = []
        for col in columns:
            if unique_key(col):
                excluded.append({"column": col, "reason": "每行唯一，疑似主键/哈希"})
                continue
            if _id_like(col):
                excluded.append({"column": col, "reason": "标识/编号列"})
                continue
            reason = feature_like_reason(col)
            if reason:
                excluded.append({"column": col, "reason": reason})
                continue
            if task_of(col) == dataset_hint:
                pool_c.append(col)
        if len(pool_c) == 1:
            col = pool_c[0]
            reasons.append(
                f"排除日历/时间/标识列后，类型与{label_of(dataset_hint)}任务相符的候选"
                f"仅剩 {col}，据此选定"
            )
            return TargetGuess(
                column=col,
                task=task_of(col),
                source="single_candidate",
                confidence=0.6,
                reasons=reasons,
                excluded=excluded,
            )
        if len(pool_c) > 1:
            reasons.append(
                f"排除日历/时间/标识列后仍有多个候选 {pool_c}，无法唯一确定目标列"
            )
            return TargetGuess(
                column=None,
                task=dataset_hint,
                reasons=reasons,
                alternatives=pool_c + matched + convention,
                excluded=excluded,
            )
        reasons.append(
            f"未找到类型与{label_of(dataset_hint)}任务相符的候选列（可能目标列类型与任务不符）"
        )
    else:
        reasons.append("数据集名称未提示监督任务，且诉求未匹配到目标列")

    return TargetGuess(
        column=None,
        task=dataset_hint if dataset_hint in ("regression", "classification") else None,
        reasons=reasons,
        alternatives=matched + convention,
        excluded=excluded,
    )
