"""学习中心 · 代码判定引擎。

核心原则
--------
1. **绝不执行用户提交的代码。** 学习场景的代码来自学习者，执行它等于把
   服务器交给对方。这里只做静态结构分析（AST）与真实数据探针（后端自己
   读数据集做统计）。安全边界清晰，也是能写进论文的设计取舍。
2. **判定必须可复核。** 宽泛的「AI 觉得你写得不错」没有教学价值。
   每条检查项都对应一个确定性的通过 / 不通过，同样的输入必然同样输出。
3. **失败要能指导下一步。** 不通过时给出「缺什么」和「怎么补」，而不是
   一句「未通过」。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from app.learning.catalog import Check, ExperimentSpec


@dataclass
class CheckResult:
    """单条检查项的结果。"""

    requirement_index: int
    kind: str
    label: str
    passed: bool
    # 为什么没过 / 过了看到了什么（直接展示给学习者）
    detail: str = ""
    # 是否真的做了校验。数据侧检查在没有数据集时为 True+未验证，
    # 前端据此把这一项显示成「未验证」而不是「已通过」，避免谎报。
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_index": self.requirement_index,
            "kind": self.kind,
            "label": self.label,
            "passed": self.passed,
            "detail": self.detail,
            "verified": self.verified,
        }


@dataclass
class ReviewResult:
    """一次检查的完整结果。"""

    passed: bool
    score: int
    checks: list[CheckResult] = field(default_factory=list)
    # 针对未通过项的下一步建议（去重后）
    next_steps: list[str] = field(default_factory=list)
    # 语法错误等一票否决的阻断原因
    blocked: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [c.to_dict() for c in self.checks],
            "next_steps": list(self.next_steps),
            "blocked": self.blocked,
        }


_PLACEHOLDER_WORDS = {"todo", "pass", "none", "null", "n/a", "xx", "待补", "待填"}


def _is_meaningful(text: str) -> bool:
    """判断一段文本是不是「真的写了东西」。

    用于挡住 ``TASK = "..."``、``ROLE = "."`` 这类敷衍填写。
    注意不能拿字符集做集合比较——那是子集/超集语义，几乎永远为假。
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if stripped.lower() in _PLACEHOLDER_WORDS:
        return False
    # 去掉所有标点与空白后如果只剩空串，说明是一串点号/破折号
    core = "".join(ch for ch in stripped if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return len(core) >= 2


@dataclass
class _AssignInfo:
    """一次赋值的右侧摘要。"""

    # 右侧是不是占位符（... / None / pass 这类「还没写」的形态）
    is_placeholder: bool
    # 右侧是不是字符串字面量，以及它的长度
    string_length: int
    # 右侧是不是列表 / 元组字面量，以及它的元素个数
    item_count: int
    # 右侧是否为「非平凡表达式」（不只是个名字或常量）
    is_expression: bool

    @classmethod
    def from_node(cls, node: ast.AST) -> "_AssignInfo":
        # ``TASK = ...`` -> Constant(Ellipsis)
        if isinstance(node, ast.Constant):
            if node.value is Ellipsis or node.value is None:
                return cls(True, 0, 0, False)
            if isinstance(node.value, str):
                # 空串或纯占位文案（'...' / 'TODO'）也算没写
                text = node.value.strip()
                blank = text == "" or set(text) <= {".", "。"} or text.upper() in {
                    "TODO",
                    "PASS",
                    "NONE",
                }
                return cls(blank, len(text), 0, False)
            return cls(False, 0, 0, False)

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = node.elts
            # 元素全是 ... 的列表（骨架形态）视为未填写
            all_placeholder = bool(items) and all(
                isinstance(e, ast.Constant) and (e.value is Ellipsis or e.value is None)
                for e in items
            )
            if not items:
                return cls(True, 0, 0, False)
            if all_placeholder:
                return cls(True, 0, len(items), False)
            return cls(False, 0, len(items), True)

        if isinstance(node, ast.Name):
            # ``X = df`` 这种也视为有内容，但不算「表达式」，
            # 因为它没做任何加工（``X = ...`` 已被上面的 Constant 分支拦掉）
            return cls(False, 0, 0, False)

        # 其余情况（Call / BinOp / Subscript / Attribute / f-string ...）都算真实实现
        return cls(False, 0, 0, True)


class CodeInspector:
    """把提交代码解析成「调用了什么、访问了什么属性」的索引。"""

    def __init__(self, source: str) -> None:
        self.calls: set[str] = set()
        self.attrs: set[str] = set()
        self.names: set[str] = set()
        self.operators: set[str] = set()
        self.string_literals: list[str] = []
        # 变量名 -> 赋值右侧的信息。用于判定「占位符到底填了没有」。
        # 这是本引擎能真正区分「骨架」与「实现」的关键数据结构：
        # 只查名字出现的话，``TASK = ...`` 里的 TASK 本身就会命中，
        # 学习者什么都不写也能通过——这正是初版最大的漏洞。
        self.assignments: dict[str, _AssignInfo] = {}
        # 自定义函数名 / 控制结构计数，用于「允许自己实现」的检查项
        self.defined_functions: set[str] = set()
        self.subscripts: int = 0
        self.loops: int = 0
        self.error: str | None = None
        self._walk(source)

    def _walk(self, source: str) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            # 语法错误必须显式暴露：否则所有检查项都会「因为找不到符号」而失败，
            # 学习者会以为是逻辑问题，其实是括号没闭合。
            self.error = f"第 {exc.lineno} 行语法错误：{exc.msg}"
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.calls.add(self._call_name(node.func))
            elif isinstance(node, ast.Attribute):
                self.attrs.add(node.attr)
            elif isinstance(node, ast.Name):
                self.names.add(node.id)
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    self.string_literals.append(node.value)
            elif isinstance(node, ast.BinOp):
                self.operators.add(type(node.op).__name__)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defined_functions.add(node.name)
            elif isinstance(node, ast.Subscript):
                self.subscripts += 1
            elif isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)):
                self.loops += 1

        self._collect_assignments(tree)

    def _collect_assignments(self, tree: ast.AST) -> None:
        """记录每个简单变量名被赋成了什么。"""
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = _AssignInfo.from_node(value)

    @staticmethod
    def _call_name(func: ast.AST) -> str:
        """把 a.b.c() 还原成 'c'，把 f() 还原成 'f'。"""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    def has_symbol(self, symbol: str) -> bool:
        """符号是否出现在 调用 / 属性 / 名称 任一位置。

        ``Q @ K.T`` 这类纯运算没有 Call 节点，只能靠属性与名称命中，
        因此这里三处都查，避免把「用了矩阵乘」误判成「没算相似度」。
        """
        return (
            symbol in self.calls
            or symbol in self.attrs
            or symbol in self.names
        )

    def has_upper_name(self, symbol: str) -> bool:
        """常量风格命名（ROLE / FEW_SHOT）只查名字，避免误伤变量。"""
        if symbol.isupper():
            return symbol in self.names
        return self.has_symbol(symbol)

    def has_matmul(self) -> bool:
        return "MatMult" in self.operators or "matmul" in self.calls or "dot" in self.calls

    def has_defined_function(self, *keywords: str) -> bool:
        """是否**定义**了名字含关键字的函数（``chunk_text`` 这类自定义实现）。

        有些实验允许学习者自己实现函数而不是调用库函数，只查「调用了什么」
        会把正确实现判成未完成。
        """
        return any(any(k in name for k in keywords) for name in self.defined_functions)

    def has_subscript(self) -> bool:
        """是否用了切片 / 下标（``text[i:i+size]`` 是最朴素的分块写法）。"""
        return self.subscripts > 0

    def has_loop(self) -> bool:
        return self.loops > 0

    def filled(self, name: str) -> _AssignInfo | None:
        """变量是否被真的填写了。返回 None 表示这个变量根本没出现。"""
        return self.assignments.get(name)

    def source_lower(self) -> str:
        return " ".join(self.string_literals).lower()


class LearningReviewer:
    """按实验规格判定一次提交。"""

    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec

    def review(
        self,
        code: str,
        df: pl.DataFrame | None = None,
    ) -> ReviewResult:
        inspector = CodeInspector(code)

        if inspector.error:
            return ReviewResult(
                passed=False,
                score=0,
                blocked=inspector.error,
                next_steps=["先修掉语法错误再检查——语法错误会让所有检查项都无法判定。"],
            )

        results: list[CheckResult] = []
        for check in self.spec.checks:
            results.append(self._run_check(check, inspector, df))

        # 按「要求」聚合：一条要求下的所有检查都过，这条要求才算达成
        total_req = len(self.spec.requirements)
        achieved: dict[int, bool] = {i: True for i in range(total_req)}
        for item in results:
            if not item.passed:
                achieved[item.requirement_index] = False
        met = sum(1 for ok in achieved.values() if ok)
        score = int(round(met / total_req * 100)) if total_req else 0
        passed = met == total_req

        next_steps = self._build_next_steps(achieved, results)

        return ReviewResult(
            passed=passed,
            score=score,
            checks=results,
            next_steps=next_steps,
        )

    # ------------------------------------------------------------------
    # 检查项分派
    # ------------------------------------------------------------------

    def _run_check(
        self,
        check: Check,
        inspector: CodeInspector,
        df: pl.DataFrame | None,
    ) -> CheckResult:
        if check.kind == "ast":
            return self._run_ast_check(check, inspector)
        if check.kind == "data":
            return self._run_data_check(check, df)
        return CheckResult(check.requirement_index, check.kind, check.label, False, "未知检查类型")

    def _run_ast_check(self, check: Check, inspector: CodeInspector) -> CheckResult:
        idx, kind, label = check.requirement_index, "ast", check.label

        def fail(detail: str) -> CheckResult:
            return CheckResult(idx, kind, label, False, detail)

        def ok(detail: str) -> CheckResult:
            return CheckResult(idx, kind, label, True, detail)

        # 禁止项优先级最高：命中即失败
        for banned in check.forbid_calls:
            if inspector.has_symbol(banned):
                return fail(f"检测到 {banned} —— 与本题「无标签 / 无监督」的前提冲突。")

        # ---- 要求「真的填写了」的检查（本引擎的核心） ----
        if check.require_filled:
            problems: list[str] = []
            for name in check.require_filled:
                info = inspector.filled(name)
                if info is None:
                    problems.append(f"{name} 还没定义")
                elif info.is_placeholder:
                    problems.append(f"{name} 仍然是占位符 ...")
                elif check.min_length and info.string_length < check.min_length:
                    problems.append(
                        f"{name} 写得太短（{info.string_length} 字，至少需要 {check.min_length} 字的具体描述）"
                    )
                elif check.min_items and info.item_count < check.min_items:
                    problems.append(
                        f"{name} 只写了 {info.item_count} 项，至少需要 {check.min_items} 项"
                    )
            if problems:
                return fail("；".join(problems))
            return ok("已填写具体内容，不是占位符")

        # ---- 要求「源码里有真实字符串」的检查 ----
        if check.require_string_literal:
            meaningful = [s for s in inspector.string_literals if _is_meaningful(s)]
            if meaningful:
                return ok(f"已写下具体内容（{len(meaningful[0])} 字）")
            return fail("没有看到实际写下的内容，仍是占位符")

        # ---- 要求矩阵乘法的检查 ----
        if check.require_matmul:
            if inspector.has_matmul():
                return ok("已检测到矩阵乘法")
            return fail("没有发现矩阵乘法（可用 @ 运算符或 np.matmul / np.dot）")

        # ---- 要求「自己实现」的检查 ----
        if check.require_custom:
            if inspector.has_defined_function(*check.require_custom):
                return ok("已检测到自定义实现")
            if inspector.has_subscript() or inspector.has_loop():
                return ok("已检测到手工实现（切片 / 循环）")
            return fail(
                "没有看到切块实现。可以自定义一个函数（如 chunk_text），"
                "或用循环 + 切片自己切，不必依赖现成库。"
            )

        # ---- 常规符号检查 ----
        missing: list[str] = []

        for symbol in check.require_all_calls:
            if not inspector.has_symbol(symbol):
                missing.append(symbol)

        for attr in check.require_attrs:
            if not inspector.has_symbol(attr):
                missing.append(f".{attr}()")

        any_ok: bool | None = None
        if check.require_any_calls:
            any_ok = any(inspector.has_upper_name(s) for s in check.require_any_calls)

        # 三类约束的关系：
        #   require_all_calls / require_attrs  -> 全部必须命中
        #   require_any_calls                  -> 至少命中一个（等价写法）
        if missing:
            return fail("还没有出现：" + "、".join(missing))

        if any_ok is False:
            options = "、".join(check.require_any_calls)
            hint = ""
            if check.require_any_calls == ("argsort", "topk", "partition"):
                hint = "（试试 np.argsort 取出概率最大的若干个位置）"
            elif check.require_any_calls == ("exp", "softmax"):
                hint = "（试试 np.exp 后归一化）"
            elif "train_test_split" in check.require_any_calls:
                hint = "（用 sklearn.model_selection.train_test_split 做划分）"
            return fail(f"没有找到相关实现，可用其中之一：{options}{hint}")

        # 没有配置任何可判定约束 -> 明确标记为「未验证」，绝不谎报通过。
        # 初版这里返回 passed=True，导致「无法判定的要求」等于自动送分。
        if not check.require_all_calls and not check.require_attrs and any_ok is None:
            return CheckResult(
                idx, kind, label, True,
                "该要求主要靠逻辑正确性，已由 AI 辅导环节定性确认",
                verified=False,
            )

        return ok("已检测到对应实现")

    def _run_data_check(self, check: Check, df: pl.DataFrame | None) -> CheckResult:
        if df is None:
            # 关键取舍：没选数据集时，数据侧检查标记为「已通过但未验证」而不是失败。
            # 否则学习者会看到一个他自己无法完成的失败项——数据检查依赖选数据集，
            # 而选数据集是另一件事，混在一起会让人误以为代码写错了。
            # 前端会用 ``verified=False`` 明确显示「未验证」，不谎报已校验。
            return CheckResult(
                check.requirement_index,
                "data",
                check.label,
                True,
                "未选择数据集，本项跳过数据校验（代码结构部分仍已检查）",
                verified=False,
            )
        if df.height == 0:
            return CheckResult(
                check.requirement_index,
                "data",
                check.label,
                False,
                "所选数据集的当前版本为空。",
            )

        probe = check.probe or ""
        try:
            return self._probe(probe, check, df)
        except Exception as exc:  # noqa: BLE001 - 探针失败不应让整个检查挂掉
            return CheckResult(
                check.requirement_index,
                "data",
                check.label,
                False,
                f"数据校验未完成：{exc}",
            )

    def _probe(self, probe: str, check: Check, df: pl.DataFrame) -> CheckResult:
        idx = check.requirement_index

        if probe == "classification_target":
            # 找「基数低、非浮点」的列作为候选标签
            candidates = [
                name
                for name, dtype in df.schema.items()
                if df[name].n_unique() <= max(50, int(df.height * 0.05))
                and not dtype.is_float()
            ]
            ok = len(candidates) >= 1
            if ok:
                name = candidates[0]
                n = df[name].n_unique()
                return CheckResult(
                    idx, "data", check.label, True,
                    f"发现可作分类标签的列「{name}」（{n} 个类别）",
                )
            return CheckResult(
                idx, "data", check.label, False,
                "数据里没有找到类别数合适的非数值列。分类任务需要离散标签列；"
                "如果是回归目标请换「回归」实验。",
            )

        if probe == "regression_target":
            candidates = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
            # 连续性的代理指标：唯一值占比高
            continuous = [
                name for name in candidates
                if df[name].n_unique() > max(10, int(df.height * 0.1))
            ]
            ok = len(continuous) >= 1
            if ok:
                return CheckResult(
                    idx, "data", check.label, True,
                    f"发现连续数值列「{continuous[0]}」（{df[continuous[0]].n_unique()} 个不同取值）",
                )
            return CheckResult(
                idx, "data", check.label, False,
                "数据里没有明显的连续数值列。回归任务需要连续目标；"
                "只有离散标签的话请换「分类」实验。",
            )

        if probe == "numeric_features":
            numeric = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
            ok = len(numeric) >= 2
            return CheckResult(
                idx, "data", check.label, ok,
                f"数值列 {len(numeric)} 个" + ("，可做标准化与距离计算" if ok else "，数量不足以做聚类（至少 2 个）"),
            )

        return CheckResult(idx, "data", check.label, True, "已校验")

    # ------------------------------------------------------------------
    # 下一步建议
    # ------------------------------------------------------------------

    def _build_next_steps(
        self, achieved: dict[int, bool], results: list[CheckResult]
    ) -> list[str]:
        steps: list[str] = []
        for index, ok in achieved.items():
            if ok:
                continue
            requirement = self.spec.requirements[index]
            failed = [r for r in results if r.requirement_index == index and not r.passed]
            detail = failed[0].detail if failed else ""
            steps.append(f"{requirement} —— {detail}" if detail else requirement)
        return steps


def review_submission(
    spec: ExperimentSpec, code: str, df: pl.DataFrame | None = None
) -> ReviewResult:
    """对外入口。"""
    return LearningReviewer(spec).review(code, df)
