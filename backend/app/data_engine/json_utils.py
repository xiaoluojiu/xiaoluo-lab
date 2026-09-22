"""Data Engine JSON 序列化工具。"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import polars as pl


def json_safe(value: Any) -> Any:
    """将值转换成 JSON 安全类型。"""

    if value is None:
        return None

    if isinstance(
        value,
        (bool, int, str),
    ):
        return value

    if isinstance(value, float):
        return (
            None
            if math.isnan(value)
            or math.isinf(value)
            else value
        )

    if isinstance(value, Decimal):
        result = float(value)

        return (
            None
            if math.isnan(result)
            or math.isinf(result)
            else result
        )

    if isinstance(
        value,
        (datetime, date, time),
    ):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    try:
        return json_safe(value.item())
    except (AttributeError, ValueError, TypeError):
        return str(value)


def df_to_records(
    df: pl.DataFrame,
) -> list[dict[str, Any]]:
    return [
        {
            key: json_safe(value)
            for key, value in row.items()
        }
        for row in df.iter_rows(
            named=True
        )
    ]


def scalar_json_safe(
    value: Any,
) -> Any:
    return json_safe(value)
