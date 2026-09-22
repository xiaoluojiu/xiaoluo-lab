"""Agent Turn 参数解析。

参考 Codex 的 item/function-call 生命周期：后续工具调用可以消费前一步的结构化输出，
但不把模型生成的自然语言当作数据传递协议。
"""
from __future__ import annotations

import re
from typing import Any

from app.agent.planner.models import PlanStep
from app.agent.runtime.models import AgentRun, AgentSession
from app.agent.context.models import AgentContext
from app.core.exceptions import ValidationException

_TEMPLATE_RE = re.compile(r"\{\{\s*(?:steps?\.)?step(\d+)\.([^{}]+?)\s*\}\}")

# 哨兵：引用解析失败（字段不存在）且该参数非必填时，标记为「该参数未提供」
_MISSING = object()


def _strip_missing(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_missing(v) for k, v in value.items() if v is not _MISSING}
    if isinstance(value, list):
        return [_strip_missing(v) for v in value if v is not _MISSING]
    return value


def _lookup(data: Any, path: str) -> Any:
    value = data
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(path)
    return value


def _outputs_by_step(run: AgentRun) -> dict[int, Any]:
    outputs: dict[int, Any] = {}
    for call in run.tool_calls:
        if call.result is not None and call.result.success:
            outputs[call.step_index + 1] = call.result.data
    return outputs


def resolve_arguments(step: PlanStep, run: AgentRun, session: AgentSession, context: AgentContext, schema: dict[str, Any]) -> PlanStep:
    """解析 {{step1.workflow_id}} 等依赖引用，并按工具 schema 做基础标量归一化。"""
    outputs = _outputs_by_step(run)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    def _missing(step_no: int, path: str) -> Any:
        """引用了不存在的输出字段时的降级策略。

        必填字段仍抛错（用户必须知道链路断了）；非必填字段直接视为「该参数未提供」，
        让工具用自身默认值继续跑。LLM 规划器经常凭直觉写字段名
        （如 {{step4.preprocessing}} 而实际叫 config），过去这类小偏差会让整步失败、
        触发重规划直至耗尽步数。
        """
        if path.split(".")[0] in required:
            raise ValidationException(f"工具 {step.tool} 引用了第 {step_no} 步不存在的输出字段：{path}")
        return _MISSING

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if not isinstance(value, str):
            return value
        matches = list(_TEMPLATE_RE.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            step_no = int(matches[0].group(1)); path = matches[0].group(2).strip()
            if step_no not in outputs:
                raise ValidationException(f"工具 {step.tool} 依赖第 {step_no} 步输出，但该步骤尚未产生成功结果。")
            try:
                return _lookup(outputs[step_no], path)
            except KeyError as exc:
                resolved_missing = _missing(step_no, path)
                if resolved_missing is _MISSING:
                    return _MISSING
                raise exc  # pragma: no cover - _missing 非必填时直接返回哨兵
        result = value
        for match in matches:
            step_no = int(match.group(1)); path = match.group(2).strip()
            if step_no not in outputs:
                raise ValidationException(f"工具 {step.tool} 依赖第 {step_no} 步输出，但该步骤尚未产生成功结果。")
            try:
                resolved = _lookup(outputs[step_no], path)
            except KeyError:
                resolved = _missing(step_no, path)
                if resolved is _MISSING:
                    # 字符串插值场景下没有「省略」语义，退化为填空串而不是整步失败
                    resolved = ""
            result = result.replace(match.group(0), str(resolved))
        return result

    args = resolve(dict(step.arguments or {}))
    if not isinstance(args, dict):
        raise ValidationException(f"工具 {step.tool} 的 arguments 解析后必须是对象。")

    # 引用未解析出值（字段不存在）时移除该参数（含嵌套结构）
    args = _strip_missing(args)

    # 引用解析为 None 时移除该参数（仅限非必填字段）。
    # 真实场景：ml.detect_task 判定为聚类后 target=None，下游 {{step4.target}} 会把 None
    # 塞进期望 string 的字段，预检报「$.target 类型错误：期望 string，实际 NoneType」，
    # 整步直接失败。工具内部普遍用 params.get(...) 取值，省略与传 None 语义一致，
    # 因此省略能让链路退化到「无目标列即聚类」而不是整条计划崩掉。
    for name in list(args):
        if args[name] is not None or name in required:
            continue
        raw = (step.arguments or {}).get(name)
        if isinstance(raw, str) and _TEMPLATE_RE.search(raw):
            args.pop(name, None)

    for name, prop in properties.items():
        if name not in args:
            continue
        typ = prop.get("type") if isinstance(prop, dict) else None
        if typ == "integer" and isinstance(args[name], str) and re.fullmatch(r"[-+]?\d+", args[name].strip()):
            args[name] = int(args[name].strip())
        elif typ == "number" and isinstance(args[name], str):
            try:
                args[name] = float(args[name].strip())
            except ValueError:
                pass

    return PlanStep(tool=step.tool, arguments=args, expected_output=step.expected_output, permission=step.permission)
