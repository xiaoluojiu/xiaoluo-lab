"""钉住 L0 升级规则层的不变式（`app.local_router.escalation_rules`）。

这层规则的价值全在「**零误判**」上：它一旦误判，就是把本地能做的话推给远程模型，
直接烧钱；漏判则降质量。所以这里把两个最容易回退的性质固定住：

1. **短语级标记**：`VAGUE_PHRASES` 里不允许出现单字标记。
   实测把 `弄/搞/处理` 当单字后，「预**处理**参数」「想**搞**懂流程步骤」全部误伤，
   escalate F1 由 87.5% 掉到 83.6%。这是一个**已被数据否掉**的直觉，必须防止它回来。

2. **数据集级零误判**：在 844 条样本上无谓升级率恒为 0（fp == 0）。
   若某次改词表引入了误伤，这条会直接失败，而不是等线上才发现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.local_router.contract import EscalationReason
from app.local_router.escalation_rules import (
    VAGUE_PHRASES,
    detect_escalation,
)

DATASET = Path(__file__).resolve().parents[1] / "scripts" / "router" / "dataset"


# --------------------------------------------------------------- 基础行为

def test_empty_and_blank_return_none():
    assert detect_escalation("") is None
    assert detect_escalation("   ") is None
    assert detect_escalation(None) is None  # type: ignore[arg-type]


def test_returns_closed_enum_member():
    """返回值必须落在封闭枚举内，否则上层无法约束解码。"""
    got = detect_escalation("预测一下明天的股票价格")
    assert isinstance(got, EscalationReason)


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        # multi_step：连词密度 + 足够长
        ("先做数据质量检查，有问题就自动清洗，然后跑 EDA，再根据结果决定要不要做特征工程",
         EscalationReason.MULTI_STEP),
        # param_dependency：平台未建模的 ML 内部词
        ("训练的时候让学习率随迭代次数衰减，配合 warmup", EscalationReason.PARAM_DEPENDENCY),
        # conflict：转折否定
        ("把这个数据集删掉，但它的历史版本都要保留", EscalationReason.CONFLICT),
        # out_of_scope：平台没有的外部系统
        ("把这个数据集同步到我们的 S3 存储桶", EscalationReason.OUT_OF_SCOPE),
        # ambiguous：短 + 模糊说法 + 无具体宾语
        ("优化一下", EscalationReason.AMBIGUOUS),
        # 平台能接住的，必须不升级
        ("这个数据干净吗", None),
        ("训练一个随机森林模型", None),
    ],
)
def test_reason_detection(utterance: str, expected: EscalationReason | None):
    assert detect_escalation(utterance) == expected


# --------------------------------------------------------------- 回归护栏

def test_vague_markers_must_be_phrases_not_single_chars():
    """单字标记会大面积误伤，这里禁止它回来（含实测证据）。"""
    assert not [w for w in VAGUE_PHRASES if len(w) == 1], (
        "VAGUE_PHRASES 出现单字标记：实测 弄/搞/处理 作单字时，「预处理参数」"
        "「想搞懂流程步骤」被误判为升级，escalate F1 由 87.5% 降至 83.6%"
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "预处理参数可以怎么设置",   # 含「处理」但是名词「预处理」
        "想搞懂流程步骤",           # 含「搞」但是「搞懂」
        "解释一下预处理参数",
        "这个数据干净吗",           # 有领域名词 ⇒ 宾语明确
        "接着再统计一遍",           # 含 2 个连词但是单步指代
        "刚才那个结果出份报告",
        "这个模型评估一下",
    ],
)
def test_known_traps_do_not_escalate(utterance: str):
    assert detect_escalation(utterance) is None, f"误判为升级：{utterance}"


def test_detection_is_pure_function():
    """同输入同输出，且不修改入参（规则层必须可重放）。"""
    u = "把这份数据用 3D 立体的方式展示出来"
    first = detect_escalation(u)
    assert detect_escalation(u) == first
    assert u == "把这份数据用 3D 立体的方式展示出来"


# --------------------------------------------------------------- 数据集级

def _load_samples() -> list[dict]:
    path = DATASET / "samples.jsonl"
    if not path.exists():
        pytest.skip("dataset/samples.jsonl 未生成（先跑 scripts/router/build_dataset.py）")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_rule_has_zero_false_escalation_on_dataset():
    """零误判是这层规则的全部价值所在 —— 必须恒成立。"""
    rows = _load_samples()
    fps = []
    tp = fn = 0
    for s in rows:
        gold_escalate = bool((s.get("target") or {}).get("escalate"))
        pred = detect_escalation(str((s.get("request") or {}).get("utterance") or ""))
        if gold_escalate and pred:
            tp += 1
        elif gold_escalate:
            fn += 1
        elif pred:
            fps.append((s.get("id"), (s.get("request") or {}).get("utterance")))

    assert not fps, f"出现无谓升级（会白烧远程调用）：{fps}"

    total_esc = tp + fn
    assert total_esc > 0, "数据集里没有升级样本，测试失去意义"
    recall = tp / total_esc
    # 已知残留 1 条（语义型 conflict：行数不可变），故允许到 0.93。
    assert recall >= 0.93, f"升级召回率回退到 {recall:.1%}（基线 96.6%）"


def test_dataset_rule_covers_all_five_reasons():
    """五类原因都要至少被识别出一个 —— 防止某条规则被误删。"""
    rows = _load_samples()
    detected = set()
    for s in rows:
        if not (s.get("target") or {}).get("escalate"):
            continue
        reason = detect_escalation(str((s.get("request") or {}).get("utterance") or ""))
        if reason:
            detected.add(reason)
    assert detected == set(EscalationReason), f"未覆盖的原因：{set(EscalationReason) - detected}"
