"""Prompt 030-035：Loader 抽象、CSV/JSON/Parquet/Excel Loader 与 Registry 测试。"""

from __future__ import annotations

import json

import polars as pl
import pytest
from app.data_engine.exceptions import LoadError, UnsupportedFormat
from app.data_engine.loaders import (
    REGISTRY,
    ArffLoader,
    CSVLoader,
    ExcelLoader,
    JSONLoader,
    LoaderRegistry,
    ParquetLoader,
)

CSV_DATA = b"name,age,score\nAlice,30,91.5\nBob,25,88.0\n"
CSV_SEMICOLON = b"name;age\nAlice;30\nBob;25\n"
CSV_TSV = b"name\tage\nAlice\t30\nBob\t25\n"
CSV_BOM = b"\xef\xbb\xbf" + CSV_DATA
CSV_GBK = "名字,年龄\n张三,20\n李四,25\n".encode("gbk")


class TestCSVLoader:
    def test_basic_utf8(self):
        result = CSVLoader().load(CSV_DATA)
        assert result.format == "csv"
        assert result.df.height == 2
        assert result.df.columns == ["name", "age", "score"]
        assert result.metadata["row_count"] == 2
        assert result.metadata["separator"] == ","

    def test_semicolon_sniffed(self):
        result = CSVLoader().load(CSV_SEMICOLON)
        assert result.df.columns == ["name", "age"]
        assert result.metadata["separator"] == ";"

    def test_tsv_sniffed(self):
        result = CSVLoader().load(CSV_TSV)
        assert result.df.columns == ["name", "age"]
        assert result.metadata["separator"] == "\t"

    def test_explicit_separator(self):
        result = CSVLoader().load(CSV_SEMICOLON, separator=";")
        assert result.df.height == 2

    def test_utf8_sig_bom(self):
        result = CSVLoader().load(CSV_BOM)
        assert "name" in result.df.columns  # BOM 不得污染首列名
        assert result.metadata["encoding"] == "utf-8"

    def test_gbk_auto_detected(self):
        result = CSVLoader().load(CSV_GBK)
        assert result.df.columns == ["名字", "年龄"]
        assert result.df["名字"][0] == "张三"

    def test_explicit_encoding(self):
        result = CSVLoader().load(CSV_GBK, encoding="gbk")
        assert result.df.height == 2

    def test_encoding_error_raises_load_error(self):
        with pytest.raises(LoadError) as exc_info:
            CSVLoader().load(CSV_GBK, encoding="utf-8")
        assert "encoding" in str(exc_info.value.message).lower()

    def test_unsupported_encoding_name(self):
        with pytest.raises(LoadError):
            CSVLoader().load(CSV_DATA, encoding="big5")

    def test_empty_csv(self):
        with pytest.raises(LoadError):
            CSVLoader().load(b"")

    def test_can_handle(self):
        loader = CSVLoader()
        assert loader.can_handle("a.csv")
        assert loader.can_handle("a.tsv")
        assert not loader.can_handle("a.json")


class TestJSONLoader:
    def test_records(self):
        data = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).encode()
        result = JSONLoader().load(data)
        assert result.df.height == 2
        assert result.metadata["json_type"] in ("records", "standard-object")

    def test_ndjson(self):
        data = b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'
        result = JSONLoader().load(data)
        assert result.df.height == 3
        assert result.metadata["json_type"] == "ndjson"

    def test_standard_object(self):
        data = b'{"a": 1, "b": 2}'
        result = JSONLoader().load(data)
        assert result.df.height == 1
        assert result.df.columns == ["a", "b"]

    def test_ndjson_extension_hint(self):
        result = JSONLoader().load(b'{"a": 1}\n{"a": 2}\n', json_type="ndjson")
        assert result.df.height == 2

    def test_invalid_json(self):
        with pytest.raises(LoadError):
            JSONLoader().load(b"{not valid")


class TestParquetLoader:
    def test_roundtrip_bytes(self):
        buf = io_bytes(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        result = ParquetLoader().load(buf)
        assert result.format == "parquet"
        assert result.df.height == 3
        assert result.df.schema["a"] == pl.Int64

    def test_metadata_lazy(self):
        buf = io_bytes(pl.DataFrame({"a": [1, 2, 3]}))
        meta = ParquetLoader().metadata(buf)
        assert meta["columns"] == ["a"]

    def test_invalid_parquet(self):
        with pytest.raises(LoadError):
            ParquetLoader().load(b"not parquet")


class TestExcelLoader:
    @pytest.fixture()
    def xlsx_path(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["name", "age"])
        ws.append(["Alice", 30])
        ws.append(["Bob", 25])
        ws2 = wb.create_sheet("Extra")
        ws2.append(["x"])
        ws2.append([1])
        path = tmp_path / "test.xlsx"
        wb.save(path)
        openpyxl  # noqa: B018 - 确认依赖存在
        return path

    def test_load_first_sheet(self, xlsx_path):
        result = ExcelLoader().load(xlsx_path)
        assert result.format == "excel"
        assert result.df.columns == ["name", "age"]
        assert result.df.height == 2
        assert result.metadata["sheet"] == "Sheet1"

    def test_sheet_by_name(self, xlsx_path):
        result = ExcelLoader().load(xlsx_path, sheet="Extra")
        assert result.df.columns == ["x"]

    def test_sheet_not_found(self, xlsx_path):
        with pytest.raises(LoadError) as exc_info:
            ExcelLoader().load(xlsx_path, sheet="Nope")
        assert "Nope" in str(exc_info.value.message)
        assert "available_sheets" in exc_info.value.details

    def test_sheet_names(self, xlsx_path):
        names = ExcelLoader().sheet_names(xlsx_path)
        assert names == ["Sheet1", "Extra"]

    def test_can_handle(self):
        assert ExcelLoader().can_handle("a.xlsx")
        assert ExcelLoader().can_handle("a.xls")
        assert not ExcelLoader().can_handle("a.csv")


ARFF_WEATHER = b"""% a comment line
@relation weather

@attribute outlook {sunny, overcast, rainy}
@attribute temperature real
@attribute humidity real
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}

@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,86,FALSE,yes
rainy,70,96,FALSE,yes
rainy,68,80,FALSE,yes
{0 overcast, 1 81, 2 75, 4 yes}
"""


class TestArffLoader:
    def test_load_dense_and_sparse(self):
        result = ArffLoader().load(ARFF_WEATHER)
        assert result.format == "arff"
        assert result.df.height == 6
        assert result.df.columns == [
            "outlook",
            "temperature",
            "humidity",
            "windy",
            "play",
        ]

    def test_relation_and_attribute_names(self):
        result = ArffLoader().load(ARFF_WEATHER)
        assert result.metadata["relation"] == "weather"
        assert result.metadata["columns"] == [
            "outlook",
            "temperature",
            "humidity",
            "windy",
            "play",
        ]

    def test_sparse_missing_values_filled(self):
        result = ArffLoader().load(ARFF_WEATHER)
        last_row = result.df[-1]
        # 稀疏行 {0 overcast, 1 81, 2 75, 4 yes}：windy 缺失应为 null
        assert last_row["outlook"][0] == "overcast"
        assert last_row["windy"][0] is None

    def test_numeric_type_inference(self):
        result = ArffLoader().load(ARFF_WEATHER)
        assert result.df.schema["temperature"] in (pl.Float64, pl.Float32)
        assert result.df.schema["humidity"] in (pl.Float64, pl.Float32)

    def test_nominal_string_type(self):
        result = ArffLoader().load(ARFF_WEATHER)
        assert result.df.schema["outlook"] == pl.Utf8

    def test_missing_attribute_def(self):
        bad = b"@relation x\n@data\n1,2\n"
        with pytest.raises(LoadError):
            ArffLoader().load(bad)

    def test_missing_data_section(self):
        bad = b"@relation x\n@attribute a numeric\n"
        with pytest.raises(LoadError):
            ArffLoader().load(bad)

    def test_can_handle(self):
        loader = ArffLoader()
        assert loader.can_handle("a.arff")
        assert not loader.can_handle("a.csv")

    def test_registry_routes_arff(self):
        assert isinstance(REGISTRY.get_loader("x.arff"), ArffLoader)
        assert ".arff" in REGISTRY.supported_extensions


class TestRegistry:
    def test_get_by_extension(self):
        loader = REGISTRY.get_loader("data/file.csv")
        assert isinstance(loader, CSVLoader)
        assert isinstance(REGISTRY.get_loader("x.xlsx"), ExcelLoader)
        assert isinstance(REGISTRY.get_loader("x.json"), JSONLoader)
        assert isinstance(REGISTRY.get_loader("x.parquet"), ParquetLoader)

    def test_unknown_extension(self):
        with pytest.raises(UnsupportedFormat):
            REGISTRY.get_loader("file.xyz")

    def test_get_by_name(self):
        assert REGISTRY.get("csv").name == "csv"
        with pytest.raises(UnsupportedFormat):
            REGISTRY.get("unknown-format")

    def test_load_convenience(self):
        result = REGISTRY.load("table.csv", CSV_DATA)
        assert result.df.height == 2

    def test_custom_registry_conflict(self):
        registry = LoaderRegistry()
        registry.register(CSVLoader())
        with pytest.raises(ValueError):
            registry.register(CSVLoader())

    def test_supported_extensions(self):
        assert ".csv" in REGISTRY.supported_extensions


def io_bytes(df: pl.DataFrame) -> bytes:
    import io

    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()
