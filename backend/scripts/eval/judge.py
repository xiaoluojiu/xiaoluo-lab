"""主观维度评审：相关性 + 完整性。

为什么这两个维度交给 LLM
------------------------
「答的是不是所问」「该说的有没有说全」本质上是语义判断，
用正则穷举会退化成关键词匹配（把「我不确定，大概是华东」也算对）。
但为了不让评审漂移，这里做了三条约束：

1. **固定 rubric 锚点**：每档分数对应一句可判定的描述，不给模型自由发挥空间；
2. **temperature=0 + 强制 JSON 输出**：同一回答重复评审结果稳定；
3. **只评主观项**：准确性与格式由 graders.py 确定性判定，评审模型看不到也改不了。

评审模型与被测模型同为 deepseek-flash —— 这一点在报告里如实标注为已知局限。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.llm import LLMMessage, LLMProvider

_SYSTEM = (
    "你是严格的数据分析回复质量评审员。只根据下面给出的【用户问题】【参考答案】【待评回答】打分，"
    "不要考虑回答的文笔，不要因为回答礼貌而加分。\n"
    "输出必须是严格 JSON，形如 {\"relevance\": 20, \"completeness\": 22, \"reason\": \"…\"}。\n"
    "不要输出 JSON 以外的任何内容。"
)

_RUBRIC = """【评分标准】

相关性（0-25）：回答是否正面回应了用户所问
- 25：完全聚焦所问，没有跑题内容
- 18：基本聚焦，但夹带了少量无关信息（如不必要的背景说明、客套话）
- 10：部分跑题，回答了别的问题或擅自扩大了范围
- 0：答非所问，或拒绝回答

完整性（0-25）：被问到的每一项是否都覆盖
- 25：所问的每一项都答到了，集合类答案没有漏项，关键限定（排序/去重/口径）都体现
- 15：主体答到了但有遗漏（集合类漏项、少答一个被问到的方面）
- 0：严重缺漏，只答了一部分或完全没答

注意：
- 参考答案只用于判断对错，不要因为回答与参考答案表述不同就扣分（同义表述算对）
- 用户明确要求「只回答 X / 不要解释」时，简洁是合规的，不要因此扣完整性分
- 回答编造了数据里不存在的内容，两档都给 0
- ★ 反过来同样成立：**不得因为回答没有提供上下文里不存在的信息而扣分**。
  实测工具结果只有「10 行全部保留，未删除任何行」，评审却以
  「未说明具体清洗口径（去缺失值处理）」为由扣了完整性 5 分 ——
  那个信息根本不在事实里，要求它说出来等于逼模型编造。
  与「编造记 0」是同一条原则的两面：只能基于给定信息评判。
- ★ 如果用户请求的操作在纯对话场景下确实无法执行（例如"画个图"但没有工具结果），
  回答**如实说明做不到并给出可直接照说的替代指令**，属于完全、正确的回应，
  两档都按满分计；不得因为"没有真的产出图表"扣分。
  只有当回答既不做、也不说清该怎么说，才算不完整。
"""


def judge(
    provider: LLMProvider,
    *,
    question: str,
    gold: Any,
    reply: str,
    timeout: float = 60.0,
) -> tuple[int, int, str]:
    """返回 (相关性 0-25, 完整性 0-25, 理由)。解析失败时返回 (0, 0, 原因)。"""
    user = (
        f"{_RUBRIC}\n\n【用户问题】\n{question}\n\n"
        f"【参考答案】\n{gold}\n\n【待评回答】\n{reply}\n\n请打分："
    )
    try:
        resp = provider.chat(
            [LLMMessage(role="system", content=_SYSTEM), LLMMessage(role="user", content=user)],
            temperature=0.0,
            # ★ 第 1 轮给 400 不够用：模型在 reason 里写长句时被截断，
            #   输出半截 JSON → 解析失败 → 相关性/完整性双双记 0。
            #   那不是回答的问题，是评审器的预算问题。提到 800 并加兜底解析。
            max_tokens=800,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return 0, 0, f"评审调用失败：{exc}"

    raw = (resp.content or "").strip()
    return _parse(raw)


def _parse(raw: str) -> tuple[int, int, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            rel = int(data.get("relevance", 0))
            comp = int(data.get("completeness", 0))
            return max(0, min(25, rel)), max(0, min(25, comp)), str(data.get("reason", ""))[:300]
        except Exception:  # noqa: BLE001
            pass
    # 兜底：JSON 坏了也要把两个分值捞出来，不能因为格式问题把回答判成 0 分
    rel_m = re.search(r'"relevance"\s*[:：]\s*(-?\d+)', raw)
    comp_m = re.search(r'"completeness"\s*[:：]\s*(-?\d+)', raw)
    if rel_m and comp_m:
        return (
            max(0, min(25, int(rel_m.group(1)))),
            max(0, min(25, int(comp_m.group(1)))),
            "JSON 格式异常，已用正则兜底解析分值",
        )
    return 0, 0, f"评审未返回可解析的 JSON：{raw[:200]}"
