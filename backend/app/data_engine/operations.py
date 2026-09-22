"""Operations：数据操作全集（过滤 / 预览 / 清洗 / 转换 / 聚合 / 透视 / 融化）。

全部操作返回新 DataFrame（不可变语义），输入不修改；校验失败统一抛 TransformError。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl

from app.data_engine.exceptions import TransformError
from app.data_engine.json_utils import df_to_records

# =========================================================
# Filter DSL（安全过滤，禁止执行用户代码）
# =========================================================

OPERATORS = {
    "eq": "等于",
    "neq": "不等于",
    "gt": "大于",
    "gte": "大于等于",
    "lt": "小于",
    "lte": "小于等于",
    "contains": "包含子串",
    "in": "在集合中",
    "is_null": "为空",
}

_LOGICS = ("and", "or")


def _validate_condition(
    columns: list[str],
    condition: dict[str, Any],
    index: int,
) -> None:
    if not isinstance(condition, dict):
        raise TransformError(
            f"condition #{index} must be an object",
            details={"index": index},
        )

    op = condition.get("op")

    if op not in OPERATORS:
        raise TransformError(
            f"unsupported filter operator: {op!r}",
            details={
                "op": op,
                "allowed": sorted(OPERATORS),
            },
        )

    column = condition.get("column")

    if not isinstance(column, str) or not column:
        raise TransformError(
            f"condition #{index} missing 'column'",
            details={"index": index},
        )

    if column not in columns:
        raise TransformError(
            f"filter column not found: {column!r}",
            details={
                "column": column,
                "available": columns,
            },
        )

    if op != "is_null" and "value" not in condition:
        raise TransformError(
            f"condition #{index} ({op}) missing 'value'",
            details={"index": index},
        )

    # in：接受真正的列表，也接受前端文本输入的逗号分隔串（在执行处拆分为列表）。
    if op == "in" and not isinstance(
        condition["value"],
        (list, tuple, set, str),
    ):
        raise TransformError(
            f"condition #{index} operator 'in' requires a list value "
            "or a comma-separated string",
            details={"index": index},
        )


def _coerce_filter_scalar(column: str, dtype: pl.DataType, value: Any) -> Any:
    """把前端传入的（通常为字符串）标量转换为与列类型匹配的值。

    这是筛选 400 的根因修复：前端所有输入都是文本，若直接与数值/时间/布尔列比较，
    Polars 会抛 InvalidOperationError。此处按列类型把值转换为正确的 Python 标量，
    转换失败时给出清晰可定位的错误信息。
    """
    if value is None:
        return None
    try:
        if dtype.is_numeric():
            if isinstance(value, bool):
                return 1 if value else 0
            if isinstance(value, (int, float)):
                return value
            text = str(value).strip()
            if text == "":
                raise ValueError("空字符串不能转换为数值")
            if dtype.is_integer():
                if "." in text or "e" in text.lower():
                    parsed = float(text)
                    if parsed != int(parsed):
                        raise ValueError(f"整数列不接受小数 {value!r}")
                    return int(parsed)
                return int(text)
            return float(text)
        if dtype == pl.Boolean:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes", "y", "是"):
                return True
            if text in ("false", "0", "no", "n", "否"):
                return False
            raise ValueError(f"无法解析为布尔值 {value!r}")
        if dtype.is_temporal():
            return _parse_temporal_scalar(value, dtype)
        # 字符串 / 其他类型：统一转字符串后比较
        return str(value)
    except (ValueError, TypeError) as exc:
        raise TransformError(
            f"筛选条件值 {value!r} 无法匹配字段 {column!r}（类型 {dtype}）：{exc}",
            details={"column": column, "dtype": str(dtype), "value": value},
        ) from exc


def _parse_temporal_scalar(value: Any, dtype: pl.DataType) -> date | datetime:
    text = str(value).strip()
    if dtype == pl.Date:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return date.fromisoformat(text)
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text)


def build_filter_expr(
    columns: list[str],
    conditions: list[dict[str, Any]],
    logic: str = "and",
    schema: dict[str, pl.DataType] | None = None,
) -> pl.Expr:
    if logic not in _LOGICS:
        raise TransformError(
            f"unsupported filter logic: {logic!r}",
            details={"allowed": list(_LOGICS)},
        )

    if not conditions:
        raise TransformError(
            "filter conditions must not be empty"
        )

    # 兼容历史调用：未传 schema 时不做类型转换（保持旧行为）
    schema = schema or {}

    expressions: list[pl.Expr] = []

    for index, condition in enumerate(conditions):
        column = condition.get("column")
        op = condition.get("op")

        # 跳过尚未选择字段的占位条件（如表单默认值 column=""），避免无意义的 400
        if not column:
            continue

        _validate_condition(
            columns,
            condition,
            index,
        )

        value = condition.get("value")
        dtype = schema.get(column)

        if op == "is_null":
            expr = pl.col(column).is_null()
        elif op == "contains":
            col_expr = (
                pl.col(column).cast(pl.String)
                if dtype is not None and dtype != pl.String
                else pl.col(column)
            )
            expr = col_expr.str.contains(str(value), literal=True)
        elif op == "in":
            # 兼容两种入参：真正的列表，或前端文本输入的逗号分隔串。
            raw_items = value.split(",") if isinstance(value, str) else list(value)
            items = [v.strip() for v in raw_items if (not isinstance(v, str) or v.strip())]
            coerced = (
                [_coerce_filter_scalar(column, dtype, v) for v in items]
                if dtype is not None
                else items
            )
            if not coerced:
                # 空集合无法构成有效条件，跳过（等价于恒真）
                continue
            expr = pl.col(column).is_in(coerced)
        else:
            coerced = _coerce_filter_scalar(column, dtype, value) if dtype is not None else value
            col = pl.col(column)
            if op == "eq":
                expr = col == coerced
            elif op == "neq":
                expr = col != coerced
            elif op == "gt":
                expr = col > coerced
            elif op == "gte":
                expr = col >= coerced
            elif op == "lt":
                expr = col < coerced
            else:  # lte
                expr = col <= coerced

        expressions.append(expr)

    if not expressions:
        # 所有条件都被跳过（例如仅含占位条件）：返回恒真，避免空表达式报错
        return pl.lit(True)

    result = expressions[0]

    for expr in expressions[1:]:
        result = (
            result & expr
            if logic == "and"
            else result | expr
        )

    return result


def apply_filter(
    df: pl.DataFrame,
    conditions: list[dict[str, Any]],
    logic: str = "and",
) -> pl.DataFrame:
    return df.filter(
        build_filter_expr(
            df.columns,
            conditions,
            logic,
            schema=dict(df.schema),
        )
    )


# =========================================================
# Preview（分页预览）
# =========================================================

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200


def _validate_pagination(
    page: int,
    page_size: int,
) -> None:
    if page < 1:
        raise TransformError(
            "page must be >= 1"
        )

    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise TransformError(
            f"page_size must be in [1, {MAX_PAGE_SIZE}]"
        )


def preview(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    columns: list[str] | None = None,
    sort: dict[str, Any] | None = None,
    filter_conditions: list[dict[str, Any]] | None = None,
    filter_logic: str = "and",
    known_total: int | None = None,
) -> dict[str, Any]:
    """分页预览。

    ``known_total``（M2 优化）：调用方已知整表行数且本次未做过滤时可直接传入，
    省掉一次 ``select(pl.len()).collect()`` 全表扫描——列表页翻页时收益明显。
    只在无 filter 时可信；带 filter 时行数会变，调用方不应传。
    """
    _validate_pagination(
        page,
        page_size,
    )

    frame = (
        df.lazy()
        if isinstance(df, pl.DataFrame)
        else df
    )

    schema = frame.collect_schema()
    available = list(schema.names())
    schema_map = dict(zip(schema.names(), schema.dtypes()))

    if columns:
        missing = [
            column
            for column in columns
            if column not in available
        ]

        if missing:
            raise TransformError(
                "preview columns not found",
                details={
                    "missing": missing,
                    "available": available,
                },
            )

    if filter_conditions:
        frame = frame.filter(
            build_filter_expr(
                available,
                filter_conditions,
                filter_logic,
                schema=schema_map,
            )
        )

    if sort:
        column = sort.get("column")

        if not column:
            raise TransformError(
                "sort requires 'column'"
            )

        if column not in available:
            raise TransformError(
                f"sort column not found: {column!r}"
            )

        frame = frame.sort(
            column,
            descending=bool(
                sort.get("desc", False)
            ),
            nulls_last=bool(
                sort.get("nulls_last", True)
            ),
        )

    if columns:
        frame = frame.select(columns)

    if (
        known_total is not None
        and not filter_conditions
        and known_total >= 0
    ):
        # 无过滤时行数不变，直接用已知值，避免整表 count。
        total = int(known_total)
    else:
        total = int(
            frame
            .select(pl.len())
            .collect()
            .item()
        )

    offset = (
        page - 1
    ) * page_size

    result = (
        frame
        .slice(offset, page_size)
        .collect()
    )

    return {
        "columns": result.columns,
        "items": df_to_records(result),
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (
            total + page_size - 1
        ) // page_size,
    }


# =========================================================
# Missing（缺失值处理）
# =========================================================

STRATEGIES = (
    "drop",
    "mean",
    "median",
    "mode",
    "constant",
)


def _resolve_columns(
    df: pl.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is None:
        return df.columns.copy()

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise TransformError(
            "missing-handling columns not found",
            details={
                "missing": missing,
                "available": df.columns,
            },
        )

    return columns


def handle_missing(
    df: pl.DataFrame,
    *,
    strategy: str,
    columns: list[str] | None = None,
    value: Any = None,
) -> pl.DataFrame:
    if strategy not in STRATEGIES:
        raise TransformError(
            f"unsupported missing strategy: {strategy!r}",
            details={
                "strategy": strategy,
                "allowed": list(STRATEGIES),
            },
        )

    columns = _resolve_columns(
        df,
        columns,
    )

    if strategy == "drop":
        return df.drop_nulls(
            subset=columns
        )

    if strategy == "constant":
        if value is None:
            raise TransformError(
                "strategy 'constant' requires a fill value"
            )

        return df.with_columns(
            pl.col(column).fill_null(value)
            for column in columns
        )

    expressions: list[pl.Expr] = []

    for column in columns:
        col = pl.col(column)
        dtype = df.schema[column]

        if strategy in ("mean", "median"):
            if not dtype.is_numeric():
                raise TransformError(
                    f"strategy {strategy!r} only applies to numeric columns",
                    details={
                        "column": column,
                        "dtype": str(dtype),
                    },
                )

            fill_value = (
                col.mean()
                if strategy == "mean"
                else col.median()
            )

        else:
            mode = df[column].mode()

            if mode.len() == 0:
                continue

            fill_value = mode[0]

        expressions.append(
            col.fill_null(fill_value)
        )

    return (
        df.with_columns(expressions)
        if expressions
        else df
    )


# =========================================================
# Duplicate（重复数据删除）
# =========================================================

KEEP_VALUES = ("first", "last")


def drop_duplicates(
    df: pl.DataFrame,
    *,
    subset: list[str] | None = None,
    keep: str = "first",
) -> pl.DataFrame:
    if keep not in KEEP_VALUES:
        raise TransformError(
            "keep must be 'first' or 'last'",
            details={
                "keep": keep,
                "allowed": list(KEEP_VALUES),
            },
        )

    if subset:
        missing = [
            column
            for column in subset
            if column not in df.columns
        ]

        if missing:
            raise TransformError(
                "duplicate subset columns not found",
                details={
                    "missing": missing,
                    "available": df.columns,
                },
            )

    return df.unique(
        subset=subset,
        keep=keep,
        maintain_order=True,
    )


# =========================================================
# Cast（类型转换）
# =========================================================

TARGET_TYPES = {
    "string": pl.String,
    "int": pl.Int64,
    "float": pl.Float64,
    "bool": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
}

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
)

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
)


def _parse_temporal(
    col: pl.Expr,
    *,
    target: str,
    fmt: str | None,
) -> pl.Expr:
    if fmt:
        if target == "date":
            return col.str.to_date(
                format=fmt,
                strict=False,
            )

        return col.str.to_datetime(
            format=fmt,
            strict=False,
        )

    formats = DATE_FORMATS if target == "date" else DATETIME_FORMATS

    def _parse(date_format: str) -> pl.Expr:
        if target == "date":
            return col.str.to_date(
                format=date_format,
                strict=False,
            )

        return col.str.to_datetime(
            format=date_format,
            strict=False,
        )

    result = pl.lit(
        None,
        dtype=TARGET_TYPES[target],
    )

    for date_format in formats:
        result = result.fill_null(
            _parse(date_format)
        )

    return result


def _cast_string(
    column: str,
    target: str,
    fmt: str | None,
) -> pl.Expr:
    col = pl.col(column)

    if target in ("date", "datetime"):
        return _parse_temporal(
            col,
            target=target,
            fmt=fmt,
        )

    if target == "bool":
        value = (
            col.str.strip_chars()
            .str.to_lowercase()
        )

        return (
            pl.when(
                value.is_in(
                    ["true", "1", "yes", "y", "是"]
                )
            )
            .then(True)
            .when(
                value.is_in(
                    ["false", "0", "no", "n", "否"]
                )
            )
            .then(False)
            .otherwise(None)
        )

    return col.cast(
        TARGET_TYPES[target],
        strict=False,
    )


def _bad_values(
    before: pl.Series,
    after: pl.Series,
    limit: int = 5,
) -> list[Any]:
    mask = (
        after.is_null()
        & before.is_not_null()
    )

    return (
        before
        .filter(mask)
        .head(limit)
        .to_list()
    )


def cast_columns(
    df: pl.DataFrame,
    types: dict[str, str],
    *,
    formats: dict[str, str] | None = None,
) -> pl.DataFrame:
    formats = formats or {}
    expressions: list[pl.Expr] = []

    for column, target in types.items():
        if column not in df.columns:
            raise TransformError(
                f"cast column not found: {column!r}"
            )

        target = target.lower()

        if target not in TARGET_TYPES:
            raise TransformError(
                f"unsupported cast target type: {target!r}",
                details={
                    "column": column,
                    "allowed": list(TARGET_TYPES),
                },
            )

        source = df.schema[column]
        fmt = formats.get(column)

        if (
            source == pl.String
            and target in (
                "date",
                "datetime",
                "bool",
                "int",
                "float",
            )
        ):
            expression = _cast_string(
                column,
                target,
                fmt,
            )

        elif (
            target in ("date", "datetime")
            and not source.is_temporal()
        ):
            raise TransformError(
                f"cannot cast column {column!r} from {source} to {target}",
                details={
                    "column": column,
                    "source_dtype": str(source),
                },
            )

        else:
            expression = pl.col(
                column
            ).cast(
                TARGET_TYPES[target],
                strict=False,
            )

        expressions.append(
            expression.alias(column)
        )

    result = df.with_columns(
        expressions
    )

    failures = []

    for column, target in types.items():
        before = df[column]
        after = result[column]

        failed_count = int(
            (
                after.is_null()
                & before.is_not_null()
            ).sum()
        )

        if failed_count:
            failures.append(
                {
                    "column": column,
                    "target": target.lower(),
                    "failed_count": failed_count,
                    "failed_values": _bad_values(
                        before,
                        after,
                    ),
                }
            )

    if failures:
        raise TransformError(
            "type cast failed for some columns",
            details={
                "failures": failures
            },
        )

    return result


# =========================================================
# String（字符串处理）
# =========================================================

STRING_OPERATIONS = (
    "trim",
    "lower",
    "upper",
    "replace",
    "regex",
)


def _require_string_column(
    df: pl.DataFrame,
    column: str,
) -> None:
    if column not in df.columns:
        raise TransformError(
            f"string column not found: {column!r}",
            details={
                "column": column,
                "available": df.columns,
            },
        )

    if df.schema[column] != pl.String:
        raise TransformError(
            f"column {column!r} is not a string column",
            details={
                "column": column,
                "dtype": str(df.schema[column]),
            },
        )


def apply_string_op(
    df: pl.DataFrame,
    *,
    column: str,
    op: str,
    params: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if op not in STRING_OPERATIONS:
        raise TransformError(
            f"unsupported string operation: {op!r}",
            details={
                "op": op,
                "allowed": list(STRING_OPERATIONS),
            },
        )

    _require_string_column(
        df,
        column,
    )

    params = params or {}
    col = pl.col(column)

    if op == "trim":
        chars = params.get("chars")
        expr = (
            col.str.strip_chars(chars)
            if chars
            else col.str.strip_chars()
        )

    elif op == "lower":
        expr = col.str.to_lowercase()

    elif op == "upper":
        expr = col.str.to_uppercase()

    elif op == "replace":
        old = params.get("old")
        new = params.get("new", "")

        if not old:
            raise TransformError(
                "replace requires params.old"
            )

        if not isinstance(new, str):
            raise TransformError(
                "replace params.new must be a string"
            )

        expr = (
            col.str.replace_all(
                old,
                new,
                literal=True,
            )
            if params.get("all", True)
            else col.str.replace(
                old,
                new,
                literal=True,
            )
        )

    else:
        pattern = params.get("pattern")
        replacement = params.get(
            "replacement",
            "",
        )

        if not pattern:
            raise TransformError(
                "regex requires params.pattern"
            )

        if not isinstance(
            replacement,
            str,
        ):
            raise TransformError(
                "regex params.replacement must be a string"
            )

        expr = col.str.replace_all(
            pattern,
            replacement,
            literal=bool(
                params.get(
                    "literal",
                    False,
                )
            ),
        )

    return df.with_columns(
        expr.alias(column)
    )


# =========================================================
# Aggregate（GroupBy 聚合）
# =========================================================

AGG_FUNCTIONS = (
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
)

NUMERIC_ONLY = {
    "sum",
    "mean",
    "median",
    "std",
}


def _validate_columns(
    df: pl.DataFrame,
    columns: list[str],
    *,
    field: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise TransformError(
            f"{field} columns not found",
            details={
                "missing": missing,
                "available": df.columns,
            },
        )


def aggregate(
    df: pl.DataFrame,
    *,
    group_by: list[str],
    aggregations: list[dict[str, Any]],
) -> pl.DataFrame:
    if not group_by:
        raise TransformError(
            "aggregate requires group_by columns"
        )

    _validate_columns(
        df,
        group_by,
        field="group_by",
    )

    if not aggregations:
        raise TransformError(
            "aggregate requires at least one aggregation"
        )

    output_names = set(group_by)
    expressions: list[pl.Expr] = []

    for index, item in enumerate(
        aggregations
    ):
        if not isinstance(item, dict):
            raise TransformError(
                f"aggregation #{index} must be an object",
                details={"index": index},
            )

        func = item.get("func")

        if func not in AGG_FUNCTIONS:
            raise TransformError(
                f"unsupported aggregation func: {func!r}",
                details={
                    "func": func,
                    "allowed": list(AGG_FUNCTIONS),
                },
            )

        column = item.get("column")

        if func == "count" and column is None:
            name = item.get(
                "alias",
                "count",
            )
            expression = pl.len()

        else:
            if not isinstance(
                column,
                str,
            ):
                raise TransformError(
                    f"aggregation #{index} ({func}) requires 'column'"
                )

            _validate_columns(
                df,
                [column],
                field="aggregation",
            )

            dtype = df.schema[column]

            if (
                func in NUMERIC_ONLY
                and not dtype.is_numeric()
            ):
                raise TransformError(
                    f"aggregation func {func!r} only applies to numeric columns",
                    details={
                        "column": column,
                        "dtype": str(dtype),
                    },
                )

            name = item.get(
                "alias"
            ) or f"{column}_{func}"

            expression = getattr(
                pl.col(column),
                func,
            )()

        if name in output_names:
            raise TransformError(
                f"duplicate output column name: {name!r}"
            )

        output_names.add(name)

        expressions.append(
            expression.alias(name)
        )

    return (
        df.group_by(group_by)
        .agg(expressions)
        .sort(group_by)
    )


# =========================================================
# Transform（结构化表达式新增 / 计算列）
# =========================================================
#
# 表达式 spec（递归定义）：
# - {"type": "column", "column": "a"}
# - {"type": "value", "value": 1}
# - {"type": "math", "op": "add|sub|mul|div", "left": spec, "right": spec}
# - {"type": "date_part", "part": "year|month|day|hour|minute|second", "column": "ts"}

MATH_OPS = ("add", "sub", "mul", "div")
DATE_PARTS = ("year", "month", "day", "hour", "minute", "second")
DATE_PART_METHODS = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
}


def _need_column(df: pl.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise TransformError(
            f"transform column not found: {column!r}",
            details={"column": column, "available": list(df.columns)},
        )


def _operand_expr(df: pl.DataFrame, spec: Any) -> pl.Expr:
    """把操作数 spec 编译为 Polars 表达式。"""
    if not isinstance(spec, dict):
        raise TransformError(
            f"operand must be an object, got {type(spec).__name__}",
            details={"operand": spec},
        )
    kind = spec.get("type")
    if kind == "column":
        column = spec.get("column")
        if not isinstance(column, str):
            raise TransformError("column operand requires 'column'", details={"operand": spec})
        _need_column(df, column)
        return pl.col(column)
    if kind == "value":
        if "value" not in spec:
            raise TransformError("value operand requires 'value'", details={"operand": spec})
        return pl.lit(spec["value"])
    if kind == "math":
        return _math_expr(df, spec)
    if kind == "date_part":
        return _date_part_expr(df, spec)
    raise TransformError(
        f"unsupported operand type: {kind!r}",
        details={"operand": spec, "allowed": ["column", "value", "math", "date_part"]},
    )


def _math_expr(df: pl.DataFrame, spec: dict[str, Any]) -> pl.Expr:
    op = spec.get("op")
    if op not in MATH_OPS:
        raise TransformError(
            f"unsupported math op: {op!r}", details={"op": op, "allowed": list(MATH_OPS)}
        )
    left = _operand_expr(df, spec.get("left"))
    right = _operand_expr(df, spec.get("right"))
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    # div：除零结果置 NULL，避免 inf
    return pl.when(right == 0).then(None).otherwise(left / right)


def _date_part_expr(df: pl.DataFrame, spec: dict[str, Any]) -> pl.Expr:
    part = spec.get("part")
    column = spec.get("column")
    if part not in DATE_PARTS:
        raise TransformError(
            f"unsupported date part: {part!r}", details={"part": part, "allowed": list(DATE_PARTS)}
        )
    if not isinstance(column, str):
        raise TransformError("date_part requires 'column'", details={"operand": spec})
    _need_column(df, column)
    dtype = df.schema[column]
    if not dtype.is_temporal():
        raise TransformError(
            f"date_part requires a temporal column, {column!r} is {dtype}",
            details={"column": column, "dtype": str(dtype)},
        )
    return getattr(pl.col(column).dt, DATE_PART_METHODS[part])()


def add_column(
    df: pl.DataFrame,
    *,
    name: str,
    expression: dict[str, Any],
    overwrite: bool = False,
) -> pl.DataFrame:
    """按结构化表达式新增列，返回新 DataFrame（不修改输入）。"""
    if not name or not isinstance(name, str):
        raise TransformError("new column name must be a non-empty string")
    if not overwrite and name in df.columns:
        raise TransformError(
            f"column already exists: {name!r}（overwrite=True 可覆盖）",
            details={"column": name},
        )
    expr = _operand_expr(df, expression)
    return df.with_columns(expr.alias(name))


# =========================================================
# Pivot（透视表）
# =========================================================

PIVOT_AGGREGATIONS = ("first", "sum", "mean", "min", "max", "count", "median", "last")


def pivot(
    df: pl.DataFrame,
    *,
    index: list[str],
    columns: str,
    values: str,
    aggregation: str = "first",
) -> pl.DataFrame:
    """透视表：index 为行维度，columns 为列维度，values 为值列。"""
    if not index:
        raise TransformError("pivot requires index columns")
    missing = [c for c in [*index, columns, values] if c not in df.columns]
    if missing:
        raise TransformError(
            "pivot columns not found",
            details={"missing": missing, "available": list(df.columns)},
        )
    if aggregation not in PIVOT_AGGREGATIONS:
        raise TransformError(
            f"unsupported pivot aggregation: {aggregation!r}",
            details={"aggregation": aggregation, "allowed": list(PIVOT_AGGREGATIONS)},
        )
    if df.schema[values].is_temporal():
        raise TransformError(
            "pivot values column must not be temporal",
            details={"column": values, "dtype": str(df.schema[values])},
        )
    if aggregation in ("sum", "mean", "median") and not df.schema[values].is_numeric():
        raise TransformError(
            f"pivot aggregation {aggregation!r} requires a numeric values column",
            details={"column": values, "dtype": str(df.schema[values])},
        )
    if df[columns].null_count() > 0:
        raise TransformError(
            f"pivot columns column {columns!r} contains null values", details={"column": columns}
        )
    if df[columns].n_unique() > 100:
        raise TransformError(
            f"pivot columns column {columns!r} has more than 100 distinct values",
            details={"column": columns, "distinct": int(df[columns].n_unique())},
        )
    try:
        return df.pivot(
            on=columns,
            index=index,
            values=values,
            aggregate_function=aggregation,
        )
    except Exception as exc:  # noqa: BLE001
        raise TransformError(
            f"pivot failed: {exc}", details={"reason": str(exc)}
        ) from exc


# =========================================================
# Melt（宽表 -> 长表）
# =========================================================


def melt(
    df: pl.DataFrame,
    *,
    id_vars: list[str] | None = None,
    value_vars: list[str] | None = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.DataFrame:
    """把宽表融化为长表。

    - id_vars: 保持不变的标识列（缺省 = 除 value_vars 外的全部列）
    - value_vars: 要融化的值列（缺省 = 除 id_vars 外的全部列）
    """
    for c in id_vars or []:
        if c not in df.columns:
            raise TransformError(
                f"melt id column not found: {c!r}",
                details={"column": c, "available": list(df.columns)},
            )
    for c in value_vars or []:
        if c not in df.columns:
            raise TransformError(
                f"melt value column not found: {c!r}",
                details={"column": c, "available": list(df.columns)},
            )
    if id_vars and value_vars:
        overlap = set(id_vars) & set(value_vars)
        if overlap:
            raise TransformError(
                "melt id_vars and value_vars must not overlap",
                details={"overlap": sorted(overlap)},
            )
    if not id_vars and not value_vars:
        raise TransformError("melt requires id_vars or value_vars")

    try:
        return df.unpivot(
            index=id_vars,
            on=value_vars,
            variable_name=variable_name,
            value_name=value_name,
        )
    except Exception as exc:  # noqa: BLE001
        raise TransformError(
            f"melt failed: {exc}", details={"reason": str(exc)}
        ) from exc
