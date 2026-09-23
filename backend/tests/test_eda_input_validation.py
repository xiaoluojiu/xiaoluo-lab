"""EDA 回归测试：重复列名 / 分类列相关性 / 数据集切换 / 错误透出。

覆盖 5 个已修 bug（均为用户可复现的真实缺陷）：

- bug1：折线图 x 与 y 同列 -> Polars DuplicateError -> 500（应为 422）
- bug2：heatmap 选分类列 -> 422 但提示不指明是哪几列、为什么
- bug3：数据集切换后仍带旧列名 -> 422（后端需给出 missing 列表）
- bug4：同类潜在风险（scatter / grouped_bar / 列裁剪投影的重复列名）
- bug5：前端错误展示（见 frontend/tests/analysisError.test.ts）

本文件全部为纯逻辑/接口测试，不调用 LLM，可进常规回归。
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analysis import (
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    EdaModule,
    VisualizationBuilder,
)
from app.api.deps import get_data_engine_service
from app.api.v1 import eda as eda_router
from app.core.exceptions import ValidationException
from app.core.middleware import RequestContextMiddleware

# ----------------------------------------------------------------------
# 夹具：规模足以让低基数列被判为「分类编码」
#   分类判定条件为 n_unique <= 50 且 n_unique * 2 < rows（见 classify_columns），
#   行数太少时 Month/DayOfWeek 会被当成连续变量，bug 反而不复现。
# ----------------------------------------------------------------------
ROWS = 2000

FLIGHT = pl.DataFrame(
    {
        "DepDelay": [float(i % 60) for i in range(ROWS)],
        "Month": [1 + (i % 12) for i in range(ROWS)],
        "DayofMonth": [1 + (i % 28) for i in range(ROWS)],
        "DayOfWeek": [1 + (i % 7) for i in range(ROWS)],
        "Distance": [float(100 + i % 500) for i in range(ROWS)],
    }
)

# 模拟「另一个数据集」：列名与 FLIGHT 完全不重叠（bug3 的场景）
OTHER = pl.DataFrame(
    {
        "ArrDelay": [float(i % 30) for i in range(ROWS)],
        "CRSArrTime": [float(600 + i % 900) for i in range(ROWS)],
    }
)

DATASETS: dict[int, pl.DataFrame] = {8: OTHER, 9: FLIGHT}


@pytest.fixture()
def builder() -> VisualizationBuilder:
    return VisualizationBuilder()


class _FakeDatasetService:
    """按 dataset_id 返回对应表，并**如实模拟 Parquet 投影**。

    ``load_version`` 用真实的 ``df.select(columns)``：重名列会像生产环境
    一样抛 ``DuplicateError`` —— 这样测试才能真正覆盖 bug1 的原始调用链
    （报错发生在读表投影阶段，而不是 analysis 内部）。
    """

    def get_version_row(self, dataset_id: int, version: int | None = None):
        if dataset_id not in DATASETS:
            raise KeyError(dataset_id)
        return type("VersionRow", (), {"version": 1})()

    def load_version(self, dataset_id: int, version: int, columns=None):
        table = DATASETS[dataset_id]
        return table.select(list(columns)) if columns else table


class _FakeService:
    dataset_service = _FakeDatasetService()

    def column_schema(self, dataset_id: int, version: int | None = None):
        """只读列结构（生产实现读 Parquet footer，不加载数据）。"""
        return [{"name": name, "dtype": str(dtype)} for name, dtype in DATASETS[dataset_id].schema.items()]


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(eda_router.router, prefix="/api/v1")
    app.dependency_overrides[get_data_engine_service] = lambda: _FakeService()
    return TestClient(app, raise_server_exceptions=False)


def _assert_4xx(response, *, code: str = "VALIDATION_ERROR") -> dict:
    """断言是明确的 4xx 业务错误（不是 500），并返回响应体。"""
    body = response.json()
    assert 400 <= response.status_code < 500, (
        f"期望 4xx 业务错误，实际 {response.status_code}：{body}"
    )
    if code:
        assert body.get("code") == code, body
    assert body.get("message"), "错误必须带可读 message"
    return body


# ----------------------------------------------------------------------
# bug1：折线图 x 与 y 同列
# ----------------------------------------------------------------------
class TestLineDuplicateColumn:
    def test_line_with_same_x_and_y_is_422_not_500(self, client):
        """原始报错请求：x=DepDelay&y=DepDelay 曾返回 500 DuplicateError。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "line", "x": "DepDelay", "y": "DepDelay", "sample_limit": 1000},
        )
        body = _assert_4xx(resp)
        assert "同一列" in body["message"]
        assert body["details"]["column"] == "DepDelay"
        assert set(body["details"]["fields"]) == {"x", "y"}

    def test_line_rejects_same_column_at_analysis_layer(self, builder):
        """不经过 HTTP 也要拦：analysis 层是唯一权威校验点。"""
        with pytest.raises(ValidationException) as exc:
            builder.analyze(FLIGHT, chart="line", x="Distance", y="Distance")
        assert exc.value.http_status == 422
        assert "同一列" in exc.value.message

    def test_line_never_leaks_duplicate_error(self, client):
        """无论走哪条路径，都不得出现 500/DuplicateError。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "line", "x": "Month", "y": "Month"},
        )
        assert resp.status_code != 500
        assert "DuplicateError" not in resp.text

    def test_line_normal_still_works(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "line", "x": "Month", "y": "DepDelay"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["x"]


# ----------------------------------------------------------------------
# bug4：同类潜在风险（scatter / grouped_bar / 投影去重）
# ----------------------------------------------------------------------
class TestDuplicateColumnAcrossCharts:
    def test_scatter_with_same_x_and_y_is_422(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "scatter", "x": "DepDelay", "y": "DepDelay"},
        )
        _assert_4xx(resp)

    def test_grouped_bar_column_equals_group_by_is_422(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={
                "chart": "grouped_bar",
                "column": "Month",
                "y": "Distance",
                "group_by": "Month",
            },
        )
        _assert_4xx(resp)

    def test_grouped_bar_column_equals_y_is_422(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={
                "chart": "grouped_bar",
                "column": "Distance",
                "y": "Distance",
                "group_by": "Month",
            },
        )
        _assert_4xx(resp)

    def test_require_distinct_columns_helper(self):
        """纯函数级：任一位置重复即抛 422，全不同则放行。"""
        EdaModule.require_distinct_columns(x="a", y="b")
        with pytest.raises(ValidationException):
            EdaModule.require_distinct_columns(x="a", y="a")
        # 允许传入 None（未指定的可选字段不参与判重）
        EdaModule.require_distinct_columns(column="a", y=None, group_by="b")

    def test_grouped_bar_same_column_rejected_not_crashed(self):
        """三列同名已被 require_distinct_columns 拦成 422，绝不能退化成 500。

        （投影别名 __cat/__val/__grp 是第二道防线：即使校验被绕过也不会炸。）
        """
        with pytest.raises(ValidationException) as exc:
            VisualizationBuilder().grouped_bar(
                FLIGHT, column="Month", y="Distance", group_by="Month"
            )
        assert exc.value.http_status == 422
        assert "同一列" in exc.value.message

    def test_grouped_bar_normal_path_unchanged(self, builder):
        result = builder.grouped_bar(
            FLIGHT, column="Month", y="Distance", group_by="DayOfWeek", agg="mean"
        )
        assert result["x"] and result["groups"]
        assert set(result["series"]) == set(result["groups"])

    def test_grouped_bar_all_aggs_work(self, builder):
        for agg in ("mean", "sum", "count", "median"):
            result = builder.grouped_bar(
                FLIGHT, column="Month", y="Distance", group_by="DayOfWeek", agg=agg
            )
            values = [v for series in result["series"].values() for v in series if v is not None]
            assert values, f"agg={agg} 未产出任何值"


# ----------------------------------------------------------------------
# bug2：heatmap 选分类列
# ----------------------------------------------------------------------
class TestHeatmapCategoricalColumns:
    @pytest.mark.parametrize(
        "columns",
        [
            "Month,DayofMonth",
            "Month,DayofMonth,DayOfWeek",
        ],
    )
    def test_categorical_only_selection_is_422_with_reason(self, client, columns):
        """原始报错请求：提示必须指明是哪几列、为什么不能用。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": columns, "sample_limit": 1000},
        )
        body = _assert_4xx(resp)
        message = body["message"]
        # 必须点出图表名，否则前端把它显示在「相关性」标签下会让人困惑
        assert "热力图" in message
        assert "分类" in message
        for name in columns.split(","):
            assert name in message, f"提示未指出不可用字段 {name}：{message}"

    def test_details_expose_rejected_columns(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": "Month,DayofMonth"},
        )
        details = _assert_4xx(resp)["details"]
        assert details["rejected_categorical_numeric"] == ["Month", "DayofMonth"]
        assert details["numeric_columns"] == []
        assert details["required"] == 2
        # 只列可用的连续数值列，不能把 Month/DayOfWeek 这类分类编码列算进去，
        # 否则用户照着提示重选依然会失败。
        assert "DepDelay" in details["available_numeric_columns"]
        assert "Month" not in details["available_numeric_columns"]
        assert "DayOfWeek" not in details["available_numeric_columns"]
        assert details["chart"] == "heatmap"

    def test_mixed_selection_reports_partial_availability(self, client):
        """含 1 个可用数值列时，提示要说明「可用 1 个，需要 2 个」。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": "Month,DepDelay"},
        )
        body = _assert_4xx(resp)
        assert "1 个" in body["message"]
        assert body["details"]["numeric_columns"] == ["DepDelay"]

    def test_missing_column_reported_distinctly(self, client):
        """不存在的列 vs 分类列，必须是两种不同提示。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": "DepDelay,NoSuchColumn"},
        )
        body = _assert_4xx(resp)
        assert "不存在" in body["message"]
        assert body["details"]["missing"] == ["NoSuchColumn"]

    def test_numeric_columns_still_succeed(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": "DepDelay,Distance"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["columns"] == ["DepDelay", "Distance"]

    def test_correlation_message_also_explains_categorical(self):
        """非 heatmap 的相关性接口同样要给出「为什么」而不是干巴巴一句。"""
        with pytest.raises(ValidationException) as exc:
            CorrelationAnalyzer().analyze(FLIGHT, columns=["Month", "DayofMonth"])
        assert "Month" in exc.value.message
        assert exc.value.details["rejected_categorical_numeric"] == ["Month", "DayofMonth"]

    def test_correlation_without_columns_keeps_plain_message(self):
        """未显式指定列时不该硬凑字段名（此时没有「请求了哪几列」可言）。"""
        # 6 行的三元表：n_unique=3、3*2 >= 6，低基数条件不成立，
        # 两列都按连续变量处理 -> 不会触发「字段不足」，也就无从测文案。
        # 因此改用行数足够、列全为低基数的表来触发「不足」分支。
        flat = pl.DataFrame(
            {"flag_a": [i % 2 for i in range(100)], "flag_b": [i % 3 for i in range(100)]}
        )
        with pytest.raises(ValidationException) as exc:
            CorrelationAnalyzer().analyze(flat)
        # 未指定 columns 时保持简短文案，不拼「以下列…不可用」
        assert exc.value.message == "相关性分析至少需要 2 个数值字段"
        assert exc.value.details["requested"] == []

    def test_correlation_with_enough_continuous_columns_succeeds(self):
        """护栏：有足够连续数值列时不得被新校验误伤。"""
        result = CorrelationAnalyzer().analyze(FLIGHT, columns=["DepDelay", "Distance"])
        assert result["columns"] == ["DepDelay", "Distance"]
        assert result["pair_count"] == 1


# ----------------------------------------------------------------------
# bug3：数据集切换后带旧数据集列名
# ----------------------------------------------------------------------
class TestStaleColumnsAcrossDatasets:
    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/descriptive", {"columns": "DepDelay,Month,DayofMonth"}),
            ("/correlation", {"columns": "DepDelay,Month,DayofMonth", "method": "pearson"}),
            ("/distribution", {"column": "DepDelay"}),
            ("/outlier", {"columns": "DepDelay,Month,DayofMonth"}),
        ],
    )
    def test_stale_columns_on_other_dataset_is_422_with_missing(self, client, path, params):
        """数据集 8 上没有数据集 9 的列：必须 422，且明确列出 missing。"""
        resp = client.get(f"/api/v1/datasets/8/eda{path}", params=params)
        body = _assert_4xx(resp)
        assert "不存在" in body["message"]
        expected = (
            [params["column"]] if path == "/distribution" else params["columns"].split(",")
        )
        assert body["details"]["missing"] == expected
        # available 要给出该数据集真实字段，方便前端/用户重新选择
        assert set(body["details"]["available"]) == set(OTHER.columns)

    def test_visualize_stale_columns_is_422(self, client):
        resp = client.get(
            "/api/v1/datasets/8/eda/visualize",
            params={"chart": "scatter", "x": "DepDelay", "y": "Month"},
        )
        body = _assert_4xx(resp)
        assert body["details"]["missing"] == ["DepDelay", "Month"]

    def test_same_columns_on_their_own_dataset_succeed(self, client):
        """同样的列名在「自己的」数据集上必须正常（证明不是一刀切拒绝）。"""
        assert client.get("/api/v1/datasets/9/eda/descriptive").status_code == 200
        resp = client.get(
            "/api/v1/datasets/9/eda/descriptive",
            params={"columns": "DepDelay,Month,DayofMonth"},
        )
        assert resp.status_code == 200

    def test_correlation_on_dataset_without_enough_numeric(self, client):
        """OTHER 只有 2 列且都低基数 -> 仍是 422，但理由应是「不够」而非「不存在」。"""
        resp = client.get("/api/v1/datasets/8/eda/correlation")
        body = _assert_4xx(resp)
        assert "不存在" not in body["message"]

    def test_projection_drops_nonexistent_columns_instead_of_500(self):
        """列裁剪不得把「列不存在」变成 500：应退化为读整表，交给上层报 422。"""
        projection = eda_router._valid_projection(
            _FakeService(), dataset_id=9, version=1, columns=["DepDelay", "NoSuch"]
        )
        assert projection is None, "含不存在列时应放弃裁剪（返回 None），而不是把错列传给 Polars"

    def test_projection_dedupes_repeated_columns(self):
        """x==y 时投影必须去重，否则读表阶段就 DuplicateError。"""
        projection = eda_router._valid_projection(
            _FakeService(), dataset_id=9, version=1, columns=["DepDelay", "DepDelay"]
        )
        assert projection == ["DepDelay"]

    def test_projection_keeps_valid_columns(self):
        projection = eda_router._valid_projection(
            _FakeService(), dataset_id=9, version=1, columns=["DepDelay", "Distance"]
        )
        assert projection == ["DepDelay", "Distance"]


# ----------------------------------------------------------------------
# bug5：错误响应结构（前端据此展示业务原因）
# ----------------------------------------------------------------------
class TestErrorPayloadShape:
    def test_error_payload_has_message_and_details(self, client):
        """前端要靠 message + details 拼出业务提示，字段缺一不可。"""
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "heatmap", "columns": "Month,DayofMonth"},
        )
        body = resp.json()
        assert set(body) >= {"code", "message", "details"}
        assert isinstance(body["details"], dict)

    def test_request_id_present_for_troubleshooting(self, client):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize",
            params={"chart": "line", "x": "Distance", "y": "Distance"},
        )
        assert resp.headers.get("x-request-id")

    def test_details_hint_guides_user(self, client):
        resp = client.get("/api/v1/datasets/8/eda/descriptive", params={"columns": "DepDelay"})
        details = _assert_4xx(resp)["details"]
        assert details.get("hint")

    def test_no_internal_error_for_any_bad_column_combination(self, client):
        """穷举各图表类型的风险参数组合，任何一个都不允许 500。"""
        risky = [
            {"chart": "line", "x": "Distance", "y": "Distance"},
            {"chart": "line", "x": "Month", "y": "Month"},
            {"chart": "scatter", "x": "DepDelay", "y": "DepDelay"},
            {"chart": "grouped_bar", "column": "Month", "y": "Distance", "group_by": "Month"},
            {"chart": "boxplot", "column": "Distance", "group_by": "Distance"},
            {"chart": "heatmap", "columns": "Month,DayofMonth"},
            {"chart": "heatmap", "columns": "DepDelay,DepDelay"},
            {"chart": "qq", "column": "Month"},
        ]
        for params in risky:
            resp = client.get("/api/v1/datasets/9/eda/visualize", params=params)
            assert resp.status_code != 500, f"{params} 触发了 500：{resp.text[:200]}"


# ----------------------------------------------------------------------
# 回归护栏：正常路径不受影响
# ----------------------------------------------------------------------
class TestNoRegressionOnHappyPath:
    @pytest.mark.parametrize(
        ("chart", "extra"),
        [
            ("histogram", {"column": "DepDelay"}),
            ("bar", {"column": "Month"}),
            ("line", {"x": "Month", "y": "DepDelay"}),
            ("scatter", {"x": "Month", "y": "DepDelay"}),
            ("boxplot", {"column": "Distance"}),
            ("boxplot", {"column": "Distance", "group_by": "Month"}),
            ("qq", {"column": "DepDelay"}),
            ("area", {"column": "DepDelay"}),
            ("heatmap", {"columns": "DepDelay,Distance"}),
            ("heatmap", {}),
            (
                "grouped_bar",
                {"column": "Month", "y": "Distance", "group_by": "DayOfWeek", "agg": "mean"},
            ),
        ],
    )
    def test_chart_still_renders(self, client, chart, extra):
        resp = client.get(
            "/api/v1/datasets/9/eda/visualize", params={"chart": chart, **extra}
        )
        assert resp.status_code == 200, resp.text[:300]

    def test_line_aggregates_duplicate_x(self, builder):
        """line 的业务语义（同 x 取均值）不能因为改投影而丢失。"""
        result = builder.line(FLIGHT, x="Month", y="DepDelay")
        assert len(result["x"]) == 12  # 12 个月
        assert result["original_count"] == 12

    def test_scatter_returns_aligned_axes(self, builder):
        result = builder.scatter(FLIGHT, x="Month", y="Distance", sample_limit=50)
        assert len(result["x"]) == len(result["y"])

    def test_descriptive_selected_columns_ok(self):
        result = DescriptiveAnalyzer().analyze(FLIGHT, columns=["DepDelay", "Distance"])
        assert [c["column"] for c in result["columns"]] == ["DepDelay", "Distance"]

    def test_deprecation_free_under_warnings_filter(self):
        """确保上述断言不是靠吞掉警告通过的（Polars 弃用 API 会显式暴露）。"""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            VisualizationBuilder().line(FLIGHT, x="Month", y="DepDelay")
