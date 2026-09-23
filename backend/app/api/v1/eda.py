"""EDA API（Prompt 199）。

- GET /datasets/{id}/eda/descriptive   描述性统计
- GET /datasets/{id}/eda/correlation   相关性
- GET /datasets/{id}/eda/distribution  分布（column 必填）
- GET /datasets/{id}/eda/outlier       异常值
所有端点只读，不修改数据。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.analysis import (
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    DistributionAnalyzer,
    EdaOutlierAnalyzer,
    VisualizationBuilder,
)
from app.api.deps import get_data_engine_service
from app.core.exceptions import ValidationException
from app.data_engine.service import DataEngineService
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/datasets/{dataset_id}/eda", tags=["eda"])

try:  # pragma: no cover - 仅用于异常类型归一化
    import polars.exceptions as _pl_exceptions
except Exception:  # noqa: BLE001
    _pl_exceptions = None  # type: ignore[assignment]


def _load_df(
    service: DataEngineService,
    dataset_id: int,
    version: int | None,
    columns: list[str] | None = None,
) -> tuple[Any, int]:
    """加载版本快照。

    ``columns`` 用于 Parquet 列裁剪：只分析少数几列时不必解码整张表
    （DatasetService 内部会判断列裁剪是否值得，并在命中缓存时直接复用）。

    ``columns`` 里含不存在的列时**不做**裁剪：存储层抛出的
    ``ColumnNotFoundError`` 会变成 500，而「请求了不存在的字段」本该是 422。
    这里退化为读整表，把列校验留给 analysis 层的 ``require_columns``。
    """
    version_row = service.dataset_service.get_version_row(dataset_id, version)
    projection = _valid_projection(service, dataset_id, version_row.version, columns)
    try:
        df = service.dataset_service.load_version(
            dataset_id,
            version_row.version,
            columns=projection,
        )
    except _POLARS_INPUT_ERRORS as exc:
        # 兜底：任何「列名/投影」层面的 Polars 异常都归为 422 业务错误。
        raise ValidationException(
            f"无法读取所选字段：{exc}",
            details={
                "columns": list(projection or columns or []),
                "hint": "请刷新数据集字段列表后重试。",
            },
        ) from exc
    return df, version_row.version


def _valid_projection(
    service: DataEngineService,
    dataset_id: int,
    version: int,
    columns: list[str] | None,
) -> list[str] | None:
    """把裁剪列裁到「确实存在」的子集；有缺失列时返回 None（读整表）。

    返回 None 而不是抛异常，是为了让 analysis 层用统一的 422 文案
    （含 missing / available）报错，而不是漏成 500。
    """
    deduped = _dedupe(columns)
    if not deduped:
        return deduped
    available = _dataset_columns(service, dataset_id, version)
    if available is None:
        # 拿不到 schema 时保守处理：不裁剪，交给上层校验。
        return None
    if any(c not in available for c in deduped):
        return None
    return deduped


def _dataset_columns(
    service: DataEngineService,
    dataset_id: int,
    version: int,
) -> set[str] | None:
    """取该版本的列名集合；取不到返回 None（调用方退化为不裁剪）。

    用 ``DataEngineService.column_schema``：它只读 Parquet footer，
    不加载任何数据行，因此这次校验的代价可以忽略。
    """
    getter = getattr(service, "column_schema", None)
    if getter is None:
        return None
    try:
        schema = getter(dataset_id, version)
    except Exception:  # noqa: BLE001 - 探测式调用，失败即放弃裁剪
        return None
    names = _schema_names(schema)
    return names or None


def _schema_names(schema: Any) -> set[str] | None:
    """从多种 schema 表示里抽出列名集合。"""
    if not schema:
        return None
    if isinstance(schema, dict):
        return {str(k) for k in schema}
    if isinstance(schema, (list, tuple)):
        names: set[str] = set()
        for item in schema:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("column")
                if name:
                    names.add(str(name))
            else:
                name = getattr(item, "name", None) or getattr(item, "column", None)
                if name:
                    names.add(str(name))
        return names or None
    return None


def _columns(raw: str | None) -> list[str] | None:
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


# Polars 的读取期异常统一转成 422。
# 这些都是「请求的字段/参数不可处理」导致的，不该以 500 呈现给用户；
# 正常路径已由 _valid_projection + require_columns 拦在前面，这里是兜底，
# 避免任何遗漏的列名组合再退化回 Internal Server Error。
_POLARS_INPUT_ERRORS: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        getattr(_pl_exceptions, name, None)
        for name in ("DuplicateError", "ColumnNotFoundError", "SchemaError", "ComputeError")
    )
    if isinstance(exc, type) and issubclass(exc, BaseException)
) or (Exception,)


def _dedupe(columns: list[str] | None) -> list[str] | None:
    """按出现顺序去重，保持原顺序。

    Parquet 列裁剪的投影里出现同名列（如 ``x=DepDelay&y=DepDelay``）会直接让
    Polars 抛 ``DuplicateError`` —— 这是 500，而不是 422。这里先去重保证
    **读取**阶段不炸；「两个位置传了同一列」是否合法由 analysis 层再做业务校验。
    """
    if not columns:
        return columns
    return list(dict.fromkeys(columns))


def _chart_columns(
    chart: str,
    column: str | None,
    x: str | None,
    y: str | None,
    columns: str | None,
    group_by: str | None,
) -> list[str] | None:
    """按图表类型推断真正要读取的列，用于 Parquet 列裁剪。

    返回 ``None`` 表示无法安全推断（比如 heatmap 需要全部数值列），此时读完整个表。
    """
    cols = _columns(columns)
    if chart == "heatmap":
        # 不裁剪：heatmap 的失败提示要告诉用户「这个数据集还有哪些可用数值列」，
        # 若只加载用户点名的那几列（很可能全是分类列），提示里的可用列会变成空 ——
        # 用户照着提示重选依然失败（回归：heatmap 选 Month,DayofMonth 的报错曾
        # 显示「可用数值列 0 个」，而该数据集其实有 DepDelay/Distance 可用）。
        # 相关性计算本身就会选全部数值列，这里不裁剪也不损失什么。
        return None
    if chart in {"histogram", "bar", "qq", "area"}:
        return [column] if column else None
    if chart in {"line", "scatter"}:
        pairs = [c for c in (x, y) if c]
        # x 与 y 可能是同一列：去重后投影，避免读取阶段就 DuplicateError。
        return _dedupe(pairs) or None
    if chart == "boxplot":
        picked = [c for c in (column, group_by) if c]
        return _dedupe(picked) or None
    if chart == "grouped_bar":
        picked = [c for c in (column, y, group_by) if c]
        return _dedupe(picked) or None
    return None


@router.get("/descriptive", response_model=ApiResponse[dict])
def descriptive(
    dataset_id: int,
    version: int | None = Query(None),
    columns: str | None = Query(None, description="逗号分隔列名；缺省分析全部字段"),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    cols = _columns(columns)
    df, v = _load_df(service, dataset_id, version, cols)
    options: dict[str, Any] = {}
    if cols:
        options["columns"] = cols
    data = DescriptiveAnalyzer().analyze(df, **options)
    data["version"] = v
    return ApiResponse[dict](data=data)


@router.get("/correlation", response_model=ApiResponse[dict])
def correlation(
    dataset_id: int,
    version: int | None = Query(None),
    columns: str | None = Query(None, description="逗号分隔列名；缺省使用全部数值列"),
    method: str = Query(
        "pearson",
        pattern="^(pearson|spearman|auto)$",
        description="pearson=线性相关；spearman=秩相关（单调关系更稳健）；auto=按列类型自动选择",
    ),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    cols = _columns(columns)
    df, v = _load_df(service, dataset_id, version, cols)
    options: dict[str, Any] = {"method": method}
    if cols:
        options["columns"] = cols
    data = CorrelationAnalyzer().analyze(df, **options)
    data["version"] = v
    return ApiResponse[dict](data=data)


@router.get("/distribution", response_model=ApiResponse[dict])
def distribution(
    dataset_id: int,
    column: str = Query(..., description="要分析分布的字段"),
    version: int | None = Query(None),
    bins: int = Query(10, ge=1, le=100, description="数值列分桶数"),
    top_n: int = Query(20, ge=1, le=100, description="类别列 Top-N 取值数"),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    # 只用到单列：列裁剪把 I/O 降到 1/C。
    df, v = _load_df(service, dataset_id, version, [column])
    data = DistributionAnalyzer().analyze(df, column=column, bins=bins, top_n=top_n)
    data["version"] = v
    return ApiResponse[dict](data=data)


@router.get("/outlier", response_model=ApiResponse[dict])
def outlier(
    dataset_id: int,
    version: int | None = Query(None),
    columns: str | None = Query(None),
    method: str = Query("iqr", pattern="^(iqr|zscore)$"),
    k: float = Query(1.5, gt=0),
    z_threshold: float = Query(3.0, gt=0),
    sample_limit: int = Query(10, ge=1, le=100),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    cols = _columns(columns)
    # 指定列时只解码这些列，避免为几列异常值扫描整张表。
    df, v = _load_df(service, dataset_id, version, cols)
    options: dict[str, Any] = {
        "method": method,
        "k": k,
        "z_threshold": z_threshold,
        "sample_limit": sample_limit,
    }
    if cols:
        options["columns"] = cols
    data = EdaOutlierAnalyzer().analyze(df, **options)
    data["version"] = v
    return ApiResponse[dict](data=data)


@router.get("/visualize", response_model=ApiResponse[dict])
def visualize(
    dataset_id: int,
    chart: str = Query(
        ...,
        pattern=r"^(histogram|bar|line|scatter|boxplot|heatmap|qq|grouped_bar|area)$",
        description=(
            "图表类型。histogram/bar/qq/area/boxplot 需要 column；"
            "line/scatter 需要 x 与 y；"
            "grouped_bar 需要 column(分类)+y(数值)+group_by；"
            "heatmap 可用 columns 限定参与计算的数值列。"
        ),
    ),
    version: int | None = Query(None),
    column: str | None = Query(
        None,
        description="主字段：histogram/bar/qq/area/boxplot 用；grouped_bar 的分类列",
    ),
    x: str | None = Query(None, description="line / scatter 的 x 字段"),
    y: str | None = Query(None, description="line / scatter / grouped_bar 的数值字段"),
    columns: str | None = Query(None, description="逗号分隔列名（heatmap 用）"),
    group_by: str | None = Query(None, description="boxplot / grouped_bar 的分组字段"),
    method: str | None = Query(
        None,
        pattern="^(pearson|spearman|auto)$",
        description="heatmap 的相关性方法（原先该参数缺失，heatmap 只能走 auto）",
    ),
    agg: str | None = Query(
        None,
        pattern="^(mean|sum|count|median)$",
        description="grouped_bar 的聚合方式",
    ),
    k: float | None = Query(
        None,
        gt=0,
        le=5,
        description="boxplot 的 whisker 倍率（默认 1.5×IQR）",
    ),
    bins: int = Query(10, ge=1, le=100, description="histogram / area 的分桶数"),
    top_n: int = Query(20, ge=1, le=100, description="bar / grouped_bar 的分类基数上限"),
    max_points: int | None = Query(
        None,
        ge=10,
        le=5000,
        description="line 的最大返回点数（超出则等间隔抽稀）",
    ),
    sample_limit: int = Query(
        1000,
        ge=1,
        le=10000,
        description="scatter / qq 的抽样上限（上限由 10 万降到 1 万，避免超大响应）",
    ),
    seed: int = Query(42, description="scatter 抽样的随机种子（固定以保证可复现）"),
    service: DataEngineService = Depends(get_data_engine_service),
) -> ApiResponse[dict]:
    # 按图表类型推断实际需要的列，做最小列裁剪（做不到就不裁剪，交给 DatasetService 判断）。
    needed = _chart_columns(chart, column, x, y, columns, group_by)
    df, v = _load_df(service, dataset_id, version, needed)
    options: dict[str, Any] = {"chart": chart}
    if column:
        options["column"] = column
    if x:
        options["x"] = x
    if y:
        options["y"] = y
    if cols := _columns(columns):
        options["columns"] = cols
    if group_by:
        options["group_by"] = group_by
    if method:
        options["method"] = method
    if agg:
        options["agg"] = agg
    if k is not None:
        options["k"] = k
    if max_points is not None:
        options["max_points"] = max_points
    options["bins"] = bins
    options["top_n"] = top_n
    options["sample_limit"] = sample_limit
    options["seed"] = seed
    data = VisualizationBuilder().analyze(df, **options)
    data["version"] = v
    return ApiResponse[dict](data=data)
