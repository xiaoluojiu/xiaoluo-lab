"""确定性评分器：准确性 + 格式合规性。

设计原则
--------
这一层**绝不调用 LLM**。准确性与格式必须是可复算、可审计的判定：
同一个回答今天评 25 分，明天再跑还是 25 分。
主观维度（相关性 / 完整性）交给 judge.py，两者互不污染。

判定口径移植自 MMTU 官方 evaluators/：
- 单值 / 标签 / 是否 → 精确匹配（归一化后）
- 数值 → 相对容差
- 集合类（Error-Detect / Schema-Matching / FD / 依赖发现…）→ F1（官方 summary_metric 里这些任务就是 f1）
- NL2SQL / Data-transform → 执行结果比对（官方也是执行 SQL / 代码判分）
"""

from __future__ import annotations

import io
import re
import sqlite3
import csv as _csv
from typing import Any

# ---------------------------------------------------------------- 工具


def _norm(text: str) -> str:
    """归一化：去空白、全角标点、大小写。用于标签类匹配。"""
    text = (text or "").strip().lower()
    text = text.replace("　", "")
    text = re.sub(r"\s+", "", text)
    for a, b in ("，", ","), ("。", "."), ("：", ":"), ("；", ";"), ("、", ","), ("（", "("), ("）", ")"):
        text = text.replace(a, b)
    return text


def _numbers(text: str) -> list[float]:
    """抽出文本里的所有数字（支持千分位与小数）。"""
    out: list[float] = []
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", (text or "").replace(",", "")):
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def build_sqlite(table: dict[str, Any], table_name: str = "table") -> sqlite3.Connection:
    """把结构化表定义建成带类型的 SQLite 表。

    为什么要显式类型：全按 TEXT 建表的话 ``SUM``/``WHERE 销售额>10000``
    会按字符串比较，gold 与模型 SQL 会同时算错，评测就失去意义。
    """
    con = sqlite3.connect(":memory:")
    cols = table["columns"]
    col_defs = ", ".join(f'"{name}" {typ}' for name, typ in cols)
    con.execute(f'CREATE TABLE "{table_name}" ({col_defs})')
    names = [c[0] for c in cols]
    placeholders = ", ".join("?" * len(names))
    con.executemany(
        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
        [tuple(r) for r in table["rows"]],
    )
    con.commit()
    return con


def run_sql(table: dict[str, Any], sql: str, table_name: str = "table") -> tuple[bool, Any]:
    """执行 SQL，返回 (是否可执行, 结果)。"""
    try:
        con = build_sqlite(table, table_name)
        cur = con.execute(sql)
        rows = cur.fetchall()
        con.close()
    except Exception:
        return False, None
    return True, rows


def table_to_csv(table: dict[str, Any]) -> str:
    """渲染成 CSV 文本（MMTU 官方的 csv 序列化形态）。"""
    buf = io.StringIO()
    names = [c[0] for c in table["columns"]]
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(names)
    writer.writerows(table["rows"])
    return buf.getvalue()


def table_to_markdown(table: dict[str, Any]) -> str:
    """渲染成 Markdown 表格（MMTU 官方的 markdown 序列化形态）。"""
    names = [c[0] for c in table["columns"]]
    head = "| " + " | ".join(names) + " |"
    sep = "| " + " | ".join("---" for _ in names) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table["rows"]]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------- 准确性


def _acc_exact(reply: str, gold: Any, **kw) -> float:
    return 1.0 if _norm(gold) in _norm(reply) else 0.0


def _acc_numeric(reply: str, gold: Any, rtol: float = 1e-3, atol: float = 1e-6, **kw) -> float:
    target = float(gold)
    for num in _numbers(reply):
        if abs(num - target) <= max(atol, rtol * abs(target)):
            return 1.0
    # 数量级正确但精度不足：给部分分，区分「算对了方向」和「完全算错」
    for num in _numbers(reply):
        if abs(num - target) <= max(atol, 0.1 * abs(target)):
            return 0.6
    return 0.0


def _acc_yesno(reply: str, gold: Any, **kw) -> float:
    want = str(gold).strip().lower()
    norm = _norm(reply)
    positive = ("是", "对", "正确", "true", "yes", "成立")
    negative = ("否", "不对", "错误", "false", "no", "不成立", "错")
    # 必须在答案里出现明确表态，且不能同时出现相反表态
    hit_pos = any(w in norm for w in positive)
    hit_neg = any(w in norm for w in negative)
    if hit_pos and hit_neg:
        return 0.0
    if want in ("是", "true", "yes"):
        return 1.0 if hit_pos else 0.0
    return 1.0 if hit_neg else 0.0


def _acc_set_f1(reply: str, gold: Any, universe: list[str] | None = None, **kw) -> float:
    """集合类任务的 F1（口径同 MMTU 的 f1 指标）。

    ``universe`` 是候选全集：预测集 = 全集中出现在回答里的项。
    这样既算召回（漏了没）也算精确率（多说了没）。
    """
    gold_set = {_norm(g) for g in gold}
    universe = universe or list(gold_set)
    pred_set = {_norm(u) for u in universe if _norm(u) in _norm(reply)}
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    tp = len(gold_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _acc_all_tokens(reply: str, gold: Any, **kw) -> float:
    """语义要点匹配：gold 是若干必须同时出现的语义要点。

    为什么不用 exact：像「这是什么字符串变换关系？用一句中文说明」这类题，
    正确答案是**一句话的自然语言描述**，不存在唯一标准串。
    用 exact 去卡「首字母大写」会把「把每个音节首字母改为大写」判成错误 ——
    这不是模型的错，是评分方式的错。
    """
    tokens = gold if isinstance(gold, (list, tuple)) else [gold]
    norm = _norm(reply)
    return 1.0 if all(_norm(t) in norm for t in tokens) else 0.0


def _acc_any_of(reply: str, gold: Any, **kw) -> float:
    """gold 是若干等价正确表述，命中任一即算对。

    与 :func:`_acc_all_tokens` 的分工：
    - ``all_tokens`` = 语义要点必须**同时**出现（描述题：既要说到「首字母」又要说到「大写」）
    - ``any_of``     = 若干表述**任一**命中即可（是非题：说「不存在」和说「没有这一列」都对）
    """
    options = gold if isinstance(gold, (list, tuple)) else [gold]
    norm = _norm(reply)
    return 1.0 if any(_norm(o) in norm for o in options) else 0.0


def _acc_sql_exec(reply: str, gold: Any, table: dict[str, Any] | None = None, **kw) -> float:
    """NL2SQL：把模型的 SQL 真跑一遍，与 gold 结果比对（MMTU NSEvaluator 口径）。"""
    blocks = re.findall(r"```(?:sql|SQL)?\s*(.*?)```", reply, re.DOTALL)
    sql = (blocks[0] if blocks else reply).strip().strip(";").strip()
    sql = re.sub(r"^(SELECT|select)", "SELECT", sql)
    ok, result = run_sql(table, sql)
    if not ok:
        return 0.0
    return 1.0 if _same_rows(result, gold) else 0.35


def _same_rows(a: Any, b: Any) -> bool:
    try:
        la = sorted([tuple(round(float(x), 6)) if isinstance(x, (int, float)) else str(x) for x in row] for row in a)
        lb = sorted([tuple(round(float(x), 6)) if isinstance(x, (int, float)) else str(x) for x in row] for row in b)
    except Exception:
        return str(a) == str(b)
    return la == lb


def _acc_py_exec(reply: str, gold: Any, py_input: Any = None, **kw) -> float:
    """Data-transform：执行模型写的代码，比对输出。

    契约在题目里写死：输入变量 ``s``，结果写进 ``result``。
    """
    blocks = re.findall(r"```(?:python|py|Python)?\s*(.*?)```", reply, re.DOTALL)
    code = blocks[0] if blocks else reply
    ns: dict[str, Any] = {"s": py_input, "re": re}
    try:
        exec(compile(code, "<model>", "exec"), ns)  # noqa: S102 - 评测需要执行模型产出的代码
    except Exception:
        return 0.0
    got = ns.get("result")
    if isinstance(got, str) and isinstance(gold, str):
        return 1.0 if got.strip() == gold.strip() else 0.3
    try:
        return 1.0 if abs(float(got) - float(gold)) < 1e-6 else 0.3
    except Exception:
        return 0.0


def _acc_formula(reply: str, gold: Any, **kw) -> float:
    """公式预测：抽 ``=...`` 表达式，归一化后比对。"""
    m = re.search(r"=\s*([A-Za-z_\u4e00-\u9fff][^\n，。；]*)", reply)
    expr = m.group(1) if m else reply
    return 1.0 if _norm(expr) == _norm(gold) else (0.5 if _norm(gold) in _norm(expr) else 0.0)


_MODES = {
    "exact": _acc_exact,
    "numeric": _acc_numeric,
    "yesno": _acc_yesno,
    "set_f1": _acc_set_f1,
    "sql_exec": _acc_sql_exec,
    "py_exec": _acc_py_exec,
    "formula": _acc_formula,
    "all_tokens": _acc_all_tokens,
    "any_of": _acc_any_of,
}


def score_accuracy(reply: str, spec: dict[str, Any]) -> tuple[float, str]:
    """返回 (0~1 的准确率, 判定说明)。"""
    mode = spec.get("mode", "exact")
    fn = _MODES[mode]
    gold = spec.get("gold")
    # 其余键作为关键字参数传给判定函数；先剔掉 gold 本身，否则与位置参数重复
    kwargs = {k: v for k, v in spec.items() if k not in ("mode", "gold")}
    ratio = float(fn(reply, gold, **kwargs))
    ratio = max(0.0, min(1.0, ratio))
    return ratio, f"{mode}={ratio:.2f}(gold={spec.get('gold')})"


# ---------------------------------------------------------------- 格式合规性

#: 内部口径红线：项目 prompt 明令禁止出现在用户可见答案里的措辞。
_INTERNAL_JARGON = ("事实摘要", "本次未获取", "未获取该信息", "没有执行", "我目前没有", "本次只")
#: 无工具路径下宣称「已执行」的措辞（answer.py 的 3.1 条硬要求）。
_FAKE_CLAIMS = ("已生成", "我画了", "我帮你算", "已跑完", "我直接按", "已经生成", "我已完成", "已绘制")
#: 否定前缀。为什么必须做上下文判断：实测模型答过
#: 「我这次没有实际执行绘图，所以**给不出已生成的**柱状图」——
#: 这是完全正确的行为（如实说明没做），但子串匹配会把「已生成」当成虚假宣称命中红线。
#: 判定虚假宣称必须看它前面是不是被否定了。
_NEGATIONS = ("没有", "并未", "不会", "无法", "不能", "未", "并非", "给不出", "算不出", "做不到", "不是", "谈不上")


def _has_fake_claim(text: str) -> bool:
    """断言式地宣称「已执行」。被否定语境包裹的不算。"""
    for word in _FAKE_CLAIMS:
        start = 0
        while True:
            idx = text.find(word, start)
            if idx < 0:
                break
            window = text[max(0, idx - 14):idx]
            if not any(neg in window for neg in _NEGATIONS):
                return True
            start = idx + len(word)
    return False
_NEGATIVE_OPEN = ("无法", "不能", "还没", "尚未", "没有", "不可以", "做不到")
_ENDING_ASK = ("请告诉我", "你希望我", "需要你补充", "请补充", "你想让我")


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    m = re.match(r"[^\n。！？!?]{0,40}", text)
    return (m.group(0) if m else text)[:40]


def _outside_codeblock(text: str) -> str:
    """去掉所有 ``` 代码块后剩下的正文。"""
    return re.sub(r"```.*?```", "", text or "", flags=re.DOTALL).strip()


_CHECKERS: dict[str, Any] = {
    "code_block": lambda r, p: bool(re.search(r"```" + re.escape(p.get("lang", "")) + r"\s*\n.*?```", r, re.DOTALL)),
    "no_code_block": lambda r, p: not re.search(r"```", r),
    "no_prose": lambda r, p: len(_outside_codeblock(r)) <= int(p.get("max_outside", 40)),
    "max_len": lambda r, p: len(r.strip()) <= int(p.get("n", 400)),
    "min_len": lambda r, p: len(r.strip()) >= int(p.get("n", 1)),
    "no_internal_jargon": lambda r, p: not any(w in r for w in _INTERNAL_JARGON),
    "no_fake_claim": lambda r, p: not _has_fake_claim(r),
    "contains_any": lambda r, p: any(w in r for w in p.get("items", [])),
    "first_sentence_positive": lambda r, p: not _first_sentence(r).startswith(_NEGATIVE_OPEN),
    "no_ending_ask": lambda r, p: not any(r.strip().endswith(w + "？") or r.strip().endswith(w + "?") or r.strip().endswith(w) for w in _ENDING_ASK),
    "contains_all": lambda r, p: all(w in r for w in p.get("items", [])),
    "contains_none": lambda r, p: not any(w in r for w in p.get("items", [])),
    "has_chinese": lambda r, p: bool(re.search(r"[\u4e00-\u9fff]", r)),
}


def score_format(reply: str, checks: list[dict[str, Any]]) -> tuple[float, list[tuple[str, bool]]]:
    """按检查项加权给分，返回 (0~1, [(检查名, 是否通过)])。"""
    if not checks:
        return 1.0, []
    total = sum(float(c.get("weight", 1)) for c in checks)
    got = 0.0
    detail: list[tuple[str, bool]] = []
    for check in checks:
        fn = _CHECKERS[check["type"]]
        try:
            ok = bool(fn(reply, check))
        except Exception:
            ok = False
        detail.append((check["type"], ok))
        if ok:
            got += float(check.get("weight", 1))
    return (got / total if total else 1.0), detail


# ---------------------------------------------------------------- 红线


def hit_redlines(reply: str, redlines: list[str]) -> list[str]:
    """红线一票否决检查。返回命中的红线名列表。"""
    hits: list[str] = []
    for name in redlines or []:
        if name == "internal_jargon" and any(w in reply for w in _INTERNAL_JARGON):
            hits.append("内部口径外泄")
        elif name == "fake_claim" and _has_fake_claim(reply):
            hits.append("无工具却宣称已执行")
        elif name == "negative_open" and _first_sentence(reply).startswith(_NEGATIVE_OPEN):
            hits.append("以否定/无法开头")
        elif name == "ending_ask" and any(reply.strip().endswith(w) or reply.strip().endswith(w + "？") for w in _ENDING_ASK):
            hits.append("以反问/索要补充结尾")
    return hits
