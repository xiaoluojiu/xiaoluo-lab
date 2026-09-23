"""上下文的 Token 估算。

为什么要估算而不是继续「纯按字符」
--------------------------------
上下文预算原先只有字符数（``AGENT_CONTEXT_MAX_CHARS=12000``）。字符与 token
的换算比例随语种剧烈变化：

* 中文约 **1 字 ≈ 1 token**（保守口径）；
* 英文约 **4 字符 ≈ 1 token**。

于是同一个 12000 字符的预算：
* 中文场景 ⇒ 约 12000 token，几乎吃满 ``AGENT_LLM_MAX_INPUT_TOKENS=24000``
  的一半，再叠加系统提示词与工具清单，很容易把整轮请求顶到上限；
* 纯英文场景 ⇒ 只有约 3000 token，预算又被白白浪费。

字符数无法表达「这段到底占多少 token」，所以这里补一个**低成本估算器**，
让上下文同时受「字符上限」与「token 上限」双重约束：哪一档先到就先截断。

不引入 tiktoken 之类依赖的理由：它要额外下载词表（离线环境不可用），
而这个估算只需要保守不低估即可——宁可少给一点上下文，也不要让请求超窗被拒。
"""

from __future__ import annotations

import re

# CJK（含日文假名、韩文）与全角标点：这类字符几乎一字符一个 token。
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)

# 其余字符（拉丁字母、数字、标点、空白）的经验换算：4 字符 ≈ 1 token
_CHARS_PER_TOKEN = 4.0

# 与上面的字符类同源，但用于单字符判定（clip_by_tokens 的线性扫描里逐字符用）
_CJK_SINGLE_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)


def count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(text))


def estimate_tokens(text: str) -> int:
    """保守估算一段文本的 token 数（只可能高估，不会低估）。"""
    if not text:
        return 0
    cjk = count_cjk(text)
    others = len(text) - cjk
    return cjk + int(others / _CHARS_PER_TOKEN + 0.999)


def fits_tokens(text: str, max_tokens: int) -> bool:
    return estimate_tokens(text) <= max_tokens


def clip_by_tokens(text: str, max_tokens: int) -> str:
    """按 token 上限截断，返回**不超过上限的最长前缀**。

    实现是 O(n)：逐字符累计代价（CJK=1，其余=0.25），再取最后一个累计值
    仍不超过 ``max_tokens`` 的位置。之所以不二分：二分每轮都要重新扫一遍全文
    数 CJK，多付 2~3 倍代价，而这里的输入本来就在几万字符量级，
    一次线性扫描更简单也更稳（且没有「估算不单调」这种隐含假设）。
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    limit = float(max_tokens)
    total = 0.0
    cut = 0
    for index, char in enumerate(text):
        if is_cjk_char(char):
            total += 1.0
        else:
            total += 1.0 / _CHARS_PER_TOKEN
        if total > limit:
            break
        cut = index + 1
    return text[:cut]


def is_cjk_char(char: str) -> bool:
    """单字符是否为 CJK / 全角类（这类字符按 1 token 计）。"""
    return bool(_CJK_SINGLE_RE.match(char))
