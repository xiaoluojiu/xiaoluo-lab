"""操作参数「必填维度」校验（用户输入不完整 → 422，而不是 400）。

与 `service._validate_operation_params` 的分工：
- 后者校验未知参数与旧有必填项，统一抛 TransformError（HTTP 400，
  语义是「数据转换失败」）。
- 本模块只处理「用户还没选 / 还没填」这类输入不完整场景，抛
  ValidationException（HTTP 422），并给出可执行的中文提示，
  前端可据此直接展示引导信息，而不是让用户看到一个含义不明的 400。

典型触发场景：透视 / 逆透视页面在未选择任何字段时点「预览」。
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import ValidationException


def validate_selector_params(op_type: str, params: dict[str, Any]) -> None:
    """校验需要用户显式选择字段的操作（当前为 pivot / melt）。"""
    if op_type == "pivot":
        _validate_pivot(params)
    elif op_type == "melt":
        _validate_melt(params)


def _validate_pivot(params: dict[str, Any]) -> None:
    index = params.get("index")
    if not isinstance(index, list) or not index:
        raise ValidationException(
            "透视操作需要先选择「行索引」字段（可多选）",
            details={"op_type": "pivot", "field": "index"},
        )

    if not params.get("columns"):
        raise ValidationException(
            "透视操作需要先选择「列维度」字段（会展开成列）",
            details={"op_type": "pivot", "field": "columns"},
        )

    if not params.get("values"):
        raise ValidationException(
            "透视操作需要先选择「值」字段（用于填充单元格）",
            details={"op_type": "pivot", "field": "values"},
        )


def _validate_melt(params: dict[str, Any]) -> None:
    if not params.get("id_vars") and not params.get("value_vars"):
        raise ValidationException(
            "逆透视操作需要至少选择「标识字段」（保持不动）"
            "或「值字段」（融化为 variable/value 两列）",
            details={"op_type": "melt"},
        )
