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
from app.data_engine.service import DataEngineService
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/datasets/{dataset_id}/eda", tags=["eda"])


def _load_df(
    service: DataEngineService,
    dataset_id: int,
    version: int | None,
    columns: list[str] | None = None,
) -> tuple[Any, int]:
    """加载版本快照。

    ``columns`` 用于 Parquet 列裁剪：只分析少数几列时不必解码整张表
    （DatasetService 内部会判断列裁剪是否值得，并在命中缓存时直接复用）。
    """
    version_row = service.dataset_service.get_version_row(dataset_id, version)
    df = service.dataset_service.load_version(
        dataset_id,
        version_row.version,
        columns=columns,
    )
    return df, version_row.version


def _columns(raw: str | None) -> list[str] | None:
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None


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
        # 相关性默认吃全部数值列；只有显式给了 columns 才能裁剪。
        return cols
    if chart in {"histogram", "bar", "qq", "area"}:
        return [column] if column else None
    if chart in {"line", "scatter"}:
        pairs = [c for c in (x, y) if c]
        return pairs or None
    if chart == "boxplot":
        picked = [c for c in (column, group_by) if c]
        return picked or None
    if chart == "grouped_bar":
        picked = [c for c in (column, y, group_by) if c]
        return picked or None
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
    column: str | None = Query(None, description="主字段：histogram/bar/qq/area/boxplot 用；grouped_bar 的分类列"),
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
