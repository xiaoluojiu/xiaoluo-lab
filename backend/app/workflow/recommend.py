"""基于数据集 schema 自动推荐 Workflow 方案。

为什么需要这个模块
------------------
★ 历史缺陷：用户说「帮我创建一个叫 客户流失预警 的 Workflow」，
  Agent 只会跑 ``workflow.list`` 然后反问「触发方式？流程内容？先做哪一步？」——
  它明明手上有数据集、有列名、有行数，却把设计的活推回给用户。
  用户反复被追问「要包含哪些步骤」，就是「挤牙膏」式体验。

规则（本模块的立场）
------------------
1. **先看数据再说话**：方案必须从真实列名、类型、唯一值数推出来，不写通用模板。
2. **一次给 1~3 个**，让用户用数字选，而不是让他从零描述。
3. **行列得能跑**：每个方案的节点类型与必填 config 都对齐
   ``app/workflow/runners`` 的契约（``dataset.read`` 要 ``dataset_id``，
   ``ml.train`` 要 ``target_column``）。推荐一组跑不通的节点，比不推荐更糟。
4. **诉求优先**：用户说「流失预警」时，分类建模方案排第一；
   没说就按数据本身能给的最有价值的信息排序。

本模块是**纯函数**：不碰数据库、不调服务，方便单测且不消耗 Token。
"""

from __future__ import annotations

import re
from typing import Any

#: 用户诉求里的关键词 → 方案族。命中就把对应方案排到前面。
_GOAL_HINTS: tuple[tuple[str, str], ...] = (
    ("流失", "classification"),
    ("churn", "classification"),
    ("预警", "classification"),
    ("风险", "classification"),
    ("分类", "classification"),
    ("是否", "classification"),
    ("预测", "regression"),
    ("回归", "regression"),
    ("销量", "regression"),
    ("金额", "regression"),
    ("质量", "quality"),
    ("清洗", "quality"),
    ("缺失", "quality"),
    ("重复", "quality"),
    ("异常", "quality"),
    ("分布", "explore"),
    ("探索", "explore"),
    ("概览", "explore"),
    ("相关性", "explore"),
    ("对比", "group"),
    ("分组", "group"),
)

#: 英文名线索：**词元级**匹配（见 :func:`_tokens`）。
#:
#: ★ 不能用子串匹配。``"y" in "monthly_charges"`` 恒为真 —— 实测
#:   monthly_charges 被当成了回归目标列，而真正的目标是 churn。
#:   「看起来能跑通但跑的是错的」，比直接说不知道更危险。
_TARGET_NAME_TOKENS: frozenset[str] = frozenset({
    "churn", "label", "target", "class", "y", "flag", "is",
    "default", "converted", "retained", "result", "outcome",
})
#: 中文线索用子串（中文没有词边界，且「是否流失」应命中「流失」）。
_TARGET_NAME_SUBSTRINGS: tuple[str, ...] = (
    "流失", "是否", "结果", "违约", "转化", "留存", "标签", "目标", "分类结果",
)

#: 看起来是主键/标识的列 —— 不能当目标列，也不该进建模特征。
_ID_NAME_TOKENS: frozenset[str] = frozenset({"id", "code", "no", "uuid", "key", "index"})
_ID_NAME_SUBSTRINGS: tuple[str, ...] = ("编号", "编码", "序号")

#: 分词用的分隔符（非字母数字一律切开）。
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
#: 驼峰 / 数字 / 中文块。
_TOKEN_PIECE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+|[\u4e00-\u9fff]+")

#: 低基数列的判定上限：唯一值不超过这个数才适合当分类目标。
_MAX_CLASS_UNIQUE = 12

_NUMERIC_PREFIXES = ("int", "uint", "float", "decimal")


def _is_numeric(dtype: str) -> bool:
    d = str(dtype or "").lower()
    return any(d.startswith(p) for p in _NUMERIC_PREFIXES)


def _tokens(name: str) -> set[str]:
    """把列名切成词元：先按非字母数字切，再切驼峰。

    ``DayofMonth`` 切不出 ``of``（小写连写），但这类词对判断目标列无关紧要；
    关键是 ``monthly_charges`` 必须切成 ``monthly`` / ``charges``，
    这样单数线索 ``y`` 才不会误命中（详见 _TARGET_NAME_TOKENS 的注释）。
    """
    parts = [p for p in _TOKEN_SPLIT_RE.split(str(name or "")) if p]
    tokens: set[str] = set()
    for part in parts:
        # 驼峰切分：customerId → customer, Id
        for piece in _TOKEN_PIECE_RE.findall(part):
            tokens.add(piece.lower())
    return tokens


def _is_id_like(name: str) -> bool:
    if not str(name or "").strip():
        return False
    tokens = _tokens(name)
    if tokens & _ID_NAME_TOKENS:
        return True
    return any(s in str(name) for s in _ID_NAME_SUBSTRINGS)


def _is_target_like(name: str) -> bool:
    if not str(name or "").strip():
        return False
    if _tokens(name) & _TARGET_NAME_TOKENS:
        return True
    return any(s in str(name) for s in _TARGET_NAME_SUBSTRINGS)


def _goal_family(goal: str) -> str:
    text = str(goal or "").lower()
    for key, family in _GOAL_HINTS:
        if key in text:
            return family
    return ""


def _classify_columns(
    columns: list[str],
    dtypes: dict[str, str],
    unique: dict[str, int] | None,
    row_count: int,
) -> dict[str, list[str]]:
    """把列分成：标识列 / 数值列 / 类别列 / 可能的分类目标 / 可能的回归目标。"""
    ids: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    # 两类分类目标候选**分开收**：名字像目标的优先，纯粹低基数的兜底。
    # 混在一个列表里按列顺序取第一个，实测会让 region（5 个唯一值、排在前）
    # 抢掉 churn（真正的目标列）—— 跑出来的是「预测地区」，不是「预测流失」。
    named_cls: list[str] = []
    low_card_cls: list[str] = []
    reg_targets: list[str] = []

    for col in columns:
        dtype = dtypes.get(col, "")
        if _is_id_like(col):
            ids.append(col)
            continue
        if _is_numeric(dtype):
            numeric.append(col)
            if _is_target_like(col):
                reg_targets.append(col)
            continue
        categorical.append(col)
        n = (unique or {}).get(col)
        # 唯一值数已知时必须真的低；未知（None）时只能靠列名判断。
        low_card = n is None or 1 < n <= _MAX_CLASS_UNIQUE
        if _is_target_like(col) and low_card:
            named_cls.append(col)
        elif n is not None and 1 < n <= _MAX_CLASS_UNIQUE and n < max(2, row_count // 20):
            low_card_cls.append(col)
    cls_targets = [*named_cls, *low_card_cls]
    return {
        "ids": ids,
        "numeric": numeric,
        "categorical": categorical,
        "classification_targets": cls_targets,
        "regression_targets": reg_targets,
    }


def _nodes(*specs: tuple[str, dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """把 [(type, config), ...] 变成链条节点与边。"""
    nodes = [
        {"id": f"n{i}", "type": t, "config": c}
        for i, (t, c) in enumerate(specs, start=1)
    ]
    edges = [
        {"source": nodes[i]["id"], "target": nodes[i + 1]["id"]}
        for i in range(len(nodes) - 1)
    ]
    return nodes, edges


def _plan(
    plan_id: int,
    family: str,
    name: str,
    steps: list[str],
    rationale: str,
    specs: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    nodes, edges = _nodes(*specs)
    return {
        "id": plan_id,
        "family": family,
        "name": name,
        "steps": steps,
        "rationale": rationale,
        "nodes": nodes,
        "edges": edges,
    }


def recommend_plans(
    *,
    dataset_id: int,
    columns: list[str],
    dtypes: dict[str, str] | None = None,
    unique: dict[str, int] | None = None,
    row_count: int = 0,
    goal: str = "",
    dataset_name: str = "",
) -> list[dict[str, Any]]:
    """返回 1~3 个可直接执行的 Workflow 方案。

    纯函数：同样的输入必然给出同样的方案 —— 这一点很重要，
    用户选了「2」之后，``workflow.build_and_run`` 会用同一份输入再算一次
    把方案还原出来，不需要跨步骤传递状态。
    """
    dtypes = dtypes or {}
    kinds = _classify_columns(list(columns or []), dtypes, unique, int(row_count or 0))
    numeric = kinds["numeric"]
    categorical = kinds["categorical"]
    cls_target = kinds["classification_targets"][0] if kinds["classification_targets"] else ""
    reg_target = kinds["regression_targets"][0] if kinds["regression_targets"] else ""
    # 没有点名的目标列时，退而求其次：二元类别列 > 低基数类别列。
    if not cls_target and categorical:
        cls_target = categorical[0]
    read = ("dataset.read", {"dataset_id": int(dataset_id)})

    candidates: list[dict[str, Any]] = []

    if cls_target:
        candidates.append(
            _plan(
                0, "classification",
                f"{cls_target} 预测建模",
                [
                    "读取数据集",
                    "统计各字段概况",
                    f"以 {cls_target} 为目标列训练分类模型",
                    "评估模型并输出指标",
                ],
                f"数据里有适合做分类目标的字段 {cls_target}"
                + (f"（唯一值 {(unique or {}).get(cls_target)} 个）" if (unique or {}).get(cls_target) else "")
                + "，可以直接训练并评估一个分类模型。",
                [
                    read,
                    ("data.statistics", {"dataset_id": int(dataset_id)}),
                    # ★ 建模节点的数据来自**上游**，config 里不能再带 dataset_id：
                    #   它会被整个塞进模型构造器，报「收到不认识的参数 dataset_id」。
                    ("ml.train", {"target_column": cls_target, "model": "random_forest_classifier"}),
                    ("ml.evaluate", {"target_column": cls_target}),
                ],
            )
        )

    if reg_target and reg_target != cls_target:
        candidates.append(
            _plan(
                0, "regression",
                f"{reg_target} 回归预测",
                [
                    "读取数据集",
                    "统计各字段概况",
                    f"以 {reg_target} 为目标列训练回归模型",
                    "评估模型并输出指标",
                ],
                f"{reg_target} 是数值字段且名字像预测目标，适合做回归。",
                [
                    read,
                    ("data.statistics", {"dataset_id": int(dataset_id)}),
                    ("ml.train", {"target_column": reg_target, "model": "random_forest_regressor"}),
                    ("ml.evaluate", {"target_column": reg_target}),
                ],
            )
        )

    if len(numeric) >= 2:
        picked = "、".join(numeric[:3])
        candidates.append(
            _plan(
                0, "explore",
                "分布与相关性探索",
                [
                    "读取数据集",
                    "计算描述性统计",
                    f"检查 {picked} 等数值列的分布",
                    "生成数据报告",
                ],
                f"有 {len(numeric)} 个数值列（{picked}），适合先看分布与相关性再决定建模方向。",
                [
                    read,
                    ("data.statistics", {"dataset_id": int(dataset_id)}),
                    ("data.quality_check", {"dataset_id": int(dataset_id)}),
                    ("report.summary", {"dataset_id": int(dataset_id)}),
                ],
            )
        )

    if categorical and numeric:
        group_col = categorical[0]
        value_col = numeric[0]
        candidates.append(
            _plan(
                0, "group",
                f"按 {group_col} 分组对比",
                [
                    "读取数据集",
                    f"按 {group_col} 分组",
                    f"聚合 {value_col}（均值/计数）",
                    "生成对比报告",
                ],
                f"{group_col} 是类别列、{value_col} 是数值列，分组聚合能直接看出组间差异。",
                [
                    read,
                    # ★ 聚合项的正确形状是 {"column": ..., "func": ...}。
                    #   写成 {value_col: "mean"} 时 runner 取到的 func 是 None，
                    #   报「unsupported aggregation func: None」—— 方案看着没问题，
                    #   一执行就挂在第 2 个节点上。
                    ("data.aggregate", {
                        "dataset_id": int(dataset_id),
                        "group_by": [group_col],
                        "aggregations": [{"column": value_col, "func": "mean"}],
                    }),
                    ("report.summary", {"dataset_id": int(dataset_id)}),
                ],
            )
        )

    candidates.append(
        _plan(
            0, "quality",
            "数据质量体检与清洗",
            [
                "读取数据集",
                "检查缺失值、重复行与异常值",
                "按发现的问题清洗数据",
                "生成数据质量报告",
            ],
            "任何分析之前都值得先确认数据能不能用：这一步能给出缺失、重复与异常的分布。",
            [
                read,
                ("data.quality_check", {"dataset_id": int(dataset_id)}),
                ("data.clean", {"dataset_id": int(dataset_id)}),
                ("report.summary", {"dataset_id": int(dataset_id)}),
            ],
        )
    )

    # 诉求优先：命中的那一族排到最前。没命中就保持上面的默认顺序
    # （建模 > 探索 > 分组 > 质量），它本身也是「信息量从高到低」。
    family = _goal_family(goal or dataset_name)
    if family:
        candidates.sort(key=lambda p: 0 if p["family"] == family else 1)

    plans: list[dict[str, Any]] = []
    for index, plan in enumerate(candidates[:3], start=1):
        item = dict(plan)
        item["id"] = index
        plans.append(item)
    return plans


def plan_options(plans: list[dict[str, Any]]) -> list[dict[str, str]]:
    """把方案整理成澄清面板可直接消费的选项（value/label/note）。"""
    options: list[dict[str, str]] = []
    for plan in plans:
        options.append(
            {
                "value": str(plan["id"]),
                "label": f"{plan['id']}. {plan['name']}",
                "note": " → ".join(plan["steps"]),
            }
        )
    return options


def plan_question(plans: list[dict[str, Any]], columns: list[str], row_count: int = 0) -> str:
    """给用户看的一句话：有哪些字段、设计了哪几个方案、怎么选。"""
    shown = "、".join(list(columns or [])[:8])
    more = f" 等 {len(columns)} 个字段" if len(columns or []) > 8 else ""
    scale = f"{row_count:,} 行" if row_count else ""
    head = f"基于当前数据集（{scale}{'，包含 ' if shown else ''}{shown}{more}），我为你设计了 {len(plans)} 个 Workflow 方案："
    body = []
    for plan in plans:
        # 澄清面板渲染的是纯文本，写 ** 只会让用户看到两颗星号。
        body.append(f"{plan['id']}. {plan['name']}：{' → '.join(plan['steps'])}。{plan['rationale']}")
    tail = "请回复数字 " + " / ".join(str(p["id"]) for p in plans) + " 选择，或告诉我你的目标，我会重新设计。"
    return "\n".join([head, *body, tail])


def resolve_plan(plans: list[dict[str, Any]], choice: Any) -> dict[str, Any] | None:
    """把用户的选择（数字 / 方案名 / 族名）还原成方案。

    认三种输入：序号「1」「方案1」、方案名（模糊包含）、以及族名。
    认不出就返回 None —— 让调用方明确提示，绝不偷偷选第一个。
    """
    if not plans:
        return None
    text = str(choice or "").strip()
    if not text:
        return None
    for plan in plans:
        if text == str(plan["id"]) or text in (f"方案{plan['id']}", f"{plan['id']}."):
            return plan
    for plan in plans:
        if text == str(plan.get("name")) or str(plan.get("name")) in text:
            return plan
    for plan in plans:
        if text.lower() == str(plan.get("family", "")).lower():
            return plan
    return None
