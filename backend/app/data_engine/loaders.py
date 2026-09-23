"""统一加载器：CSV/Excel/JSON/Parquet + 注册表。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import polars as pl

from app.data_engine.exceptions import LoadError, UnsupportedFormat

# =========================================================
# 基类与统一接口
# =========================================================

LoadSource = bytes | str | PurePosixPath | PureWindowsPath


@dataclass
class LoadedTable:
    """统一 Loader 输出。"""

    df: pl.DataFrame
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("row_count", self.df.height)
        self.metadata.setdefault("column_count", self.df.width)
        self.metadata.setdefault("columns", self.df.columns)
        self.metadata.setdefault(
            "dtypes",
            {
                name: str(dtype)
                for name, dtype in self.df.schema.items()
            },
        )


class Loader(ABC):
    """数据加载器统一接口。"""

    name: str = ""
    extensions: tuple[str, ...] = ()

    def can_handle(
        self,
        source: str | LoadSource,
    ) -> bool:
        extension = _extension_of(source)
        return extension is not None and extension in self.extensions

    @abstractmethod
    def load(
        self,
        source: LoadSource,
        **options: Any,
    ) -> LoadedTable:
        """加载数据。"""

    @abstractmethod
    def metadata(
        self,
        source: LoadSource,
        **options: Any,
    ) -> dict[str, Any]:
        """获取数据元信息。"""


def _extension_of(
    source: str | LoadSource,
) -> str | None:
    """获取文件扩展名。"""
    if isinstance(source, bytes):
        return None

    filename = str(source).replace("\\", "/").rsplit("/", 1)[-1]

    if "." not in filename:
        return None

    return f".{filename.rsplit('.', 1)[-1].lower()}"


def read_bytes_or_path(
    source: LoadSource,
) -> bytes | str:
    """转换为 Polars 可直接读取的类型。"""
    if isinstance(source, bytes):
        return source

    return str(source)


def loader_error(
    message: str,
    **details: Any,
) -> LoadError:
    """创建标准 Loader 异常。"""
    return LoadError(
        message,
        details=details,
    )


# =========================================================
# CSV Loader
# =========================================================

COMMON_SEPARATORS = (
    ",",
    ";",
    "\t",
    "|",
)

SUPPORTED_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "gbk",
    "gb18030",
    "latin-1",
)


def _strip_bom(data: bytes) -> bytes:
    if data.startswith(
        b"\xef\xbb\xbf"
    ):
        return data[3:]

    return data


def _normalize_encoding(
    value: str | None,
) -> str:
    if not value or value == "auto":
        return "auto"

    encoding = (
        value
        .lower()
        .replace("_", "-")
    )

    if encoding not in SUPPORTED_ENCODINGS:
        raise LoadError(
            f"unsupported encoding {value!r}",
            details={
                "supported": SUPPORTED_ENCODINGS
            },
        )

    return encoding


def _decode_probe(
    data: bytes,
    encoding: str,
) -> bool:
    """判断 ``data`` 能否按 ``encoding`` 解码，容忍**结尾被截断**的多字节字符。

    为什么需要容忍：``_read_head`` 固定读前 64 000 字节，这个边界对 GBK/UTF-8
    来说有约一半概率落在一个多字节字符中间。若直接 ``data.decode(encoding)``，
    截断处必然抛 ``UnicodeDecodeError``，于是整份文件被误判成 latin-1 ——
    中文列名全部变成乱码，而且**文件越大越容易踩到**（小于 64 KB 才不受影响）。

    做法：解码失败时依次丢掉结尾 1~3 个字节重试（覆盖 UTF-8 的 4 字节序列与
    GB18030 的 4 字节序列）。只影响「探测」，真正解码仍由调用方按完整数据执行。
    """
    for trim in range(4):
        chunk = data if trim == 0 else data[: len(data) - trim]
        if not chunk:
            return False
        try:
            chunk.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def _detect_encoding(
    data: bytes,
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    for encoding in (
        "utf-8",
        "gb18030",
    ):
        if _decode_probe(data, encoding):
            return encoding

    return "latin-1"


def _detect_separator(
    data: bytes,
    encoding: str,
) -> str:
    text = data.decode(
        encoding,
        errors="replace",
    )

    line = next(
        (
            item
            for item in text.splitlines()
            if item.strip()
        ),
        "",
    )

    counts = {
        separator: line.count(separator)
        for separator in COMMON_SEPARATORS
    }

    separator = max(
        counts,
        key=counts.get,
    )

    return (
        separator
        if counts[separator] > 0
        else ","
    )


def _read_head(
    source: LoadSource,
) -> bytes:
    if isinstance(source, bytes):
        return _strip_bom(
            source[:64_000]
        )

    try:
        with open(
            source,
            "rb",
        ) as file:
            return _strip_bom(
                file.read(64_000)
            )
    except OSError as exc:
        raise LoadError(
            "cannot read file",
            details={
                "path": str(source),
                "reason": str(exc),
            },
        ) from exc


def _decode(
    data: bytes,
    encoding: str,
) -> bytes:
    data = _strip_bom(data)

    try:
        return (
            data.decode(encoding)
            .encode("utf-8")
        )
    except UnicodeDecodeError as exc:
        raise LoadError(
            f"failed to decode CSV with encoding {encoding}",
            details={
                "encoding": encoding,
                "reason": str(exc),
            },
        ) from exc


class CSVLoader(Loader):
    name = "csv"
    extensions = (
        ".csv",
        ".tsv",
        ".txt",
    )

    def load(
        self,
        source: LoadSource,
        **options: Any,
    ) -> LoadedTable:
        if not isinstance(
            source,
            (
                bytes,
                str,
                PurePosixPath,
                PureWindowsPath,
            ),
        ):
            raise UnsupportedFormat(
                f"unsupported source type: {type(source)!r}"
            )

        encoding = _normalize_encoding(
            options.pop(
                "encoding",
                "auto",
            )
        )

        separator = (
            options.pop(
                "separator",
                None,
            )
            or options.pop(
                "sep",
                None,
            )
        )

        head = _read_head(source)

        encoding = _detect_encoding(
            head,
            encoding,
        )

        separator = (
            separator
            or _detect_separator(
                head,
                encoding,
            )
        )

        if isinstance(source, bytes):
            payload = _decode(
                source,
                encoding,
            )
        elif encoding in {
            "utf-8",
            "utf-8-sig",
        }:
            payload = source
        else:
            try:
                with open(
                    source,
                    "rb",
                ) as file:
                    payload = _decode(
                        file.read(),
                        encoding,
                    )
            except OSError as exc:
                raise LoadError(
                    "cannot read CSV",
                    details={
                        "path": str(source),
                        "reason": str(exc),
                    },
                ) from exc

        try:
            df = pl.read_csv(
                payload,
                encoding="utf8",
                separator=separator,
            )
        except pl.exceptions.NoDataError as exc:
            raise LoadError(
                "CSV is empty"
            ) from exc
        except Exception as exc:
            raise LoadError(
                "failed to parse CSV",
                details={
                    "reason": str(exc)
                },
            ) from exc

        return LoadedTable(
            df=df,
            format="csv",
            metadata={
                "encoding": encoding,
                "separator": separator,
            },
        )

    def metadata(
        self,
        source: LoadSource,
        **options: Any,
    ) -> dict[str, Any]:
        return self.load(
            source,
            **options,
        ).metadata


# =========================================================
# Excel Loader
# =========================================================


class ExcelLoader(Loader):
    name = "excel"
    extensions = (".xlsx", ".xls", ".xlsm")

    # ---- 内部工具 ----
    @staticmethod
    def _open(source: LoadSource):  # noqa: ANN202 - fastexcel 对象无公开类型
        """打开工作簿，返回 fastexcel ExcelReader。"""
        try:
            import fastexcel
        except ImportError as exc:  # pragma: no cover
            raise LoadError(
                "excel support requires the 'fastexcel' package",
                details={"package": "fastexcel"},
            ) from exc
        try:
            if isinstance(source, bytes):
                return fastexcel.read_excel(source)
            return fastexcel.read_excel(str(source))
        except Exception as exc:  # noqa: BLE001 - 文件损坏 / 非法格式
            raise LoadError(
                f"cannot open excel file: {exc}", details={"reason": str(exc)}
            ) from exc

    def sheet_names(self, source: LoadSource) -> list[str]:
        """返回工作簿内全部 Sheet 名称。"""
        return list(self._open(source).sheet_names)

    def _resolve_sheet(self, reader: Any, sheet: str | int | None) -> str:
        names = list(reader.sheet_names)
        if sheet is None:
            return names[0]
        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(names):
                raise LoadError(
                    f"sheet index {sheet} out of range",
                    details={"sheet_count": len(names)},
                )
            return names[sheet]
        if sheet not in names:
            raise LoadError(
                f"sheet {sheet!r} not found",
                details={"available_sheets": names},
            )
        return sheet

    # ---- 接口实现 ----
    def load(self, source: LoadSource, **options: Any) -> LoadedTable:
        if not isinstance(source, (bytes, str, PurePosixPath, PureWindowsPath)):
            raise UnsupportedFormat(f"unsupported source type: {type(source)!r}")

        reader = self._open(source)
        sheet = options.pop("sheet", None)
        if sheet is None:
            sheet = options.pop("sheet_name", None)
        header_row = options.pop("header_row", None)  # 0-based；None 表示首行
        skip_rows = options.pop("skip_rows", 0)

        sheet_name = self._resolve_sheet(reader, sheet)
        try:
            header_row = int(header_row) if header_row is not None else 0
            arrow_tbl = reader.load_sheet(
                sheet_name,
                header_row=header_row,
                skip_rows=int(skip_rows) if skip_rows else 0,
            )
        except LoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LoadError(
                f"failed to read sheet {sheet_name!r}: {exc}",
                details={"sheet": sheet_name, "reason": str(exc)},
            ) from exc

        df = pl.from_arrow(arrow_tbl)
        if not isinstance(df, pl.DataFrame):  # pragma: no cover
            raise LoadError("excel sheet did not produce a DataFrame")

        return LoadedTable(
            df=df,
            format="excel",
            metadata={
                "sheet": sheet_name,
                "sheets": list(reader.sheet_names),
                "header_row": header_row,
            },
        )

    def metadata(self, source: LoadSource, **options: Any) -> dict[str, Any]:
        reader = self._open(source)
        sheets = list(reader.sheet_names)
        sheet_name = self._resolve_sheet(reader, options.pop("sheet", None))
        loaded = self.load(source, sheet=sheet_name, **options)
        return {
            "sheets": sheets,
            "sheet": sheet_name,
            **loaded.metadata,
        }


# =========================================================
# JSON Loader
# =========================================================


class JSONLoader(Loader):
    name = "json"
    extensions = (".json", ".jsonl", ".ndjson")

    def load(self, source: LoadSource, **options: Any) -> LoadedTable:
        if not isinstance(source, (bytes, str, PurePosixPath, PureWindowsPath)):
            raise UnsupportedFormat(f"unsupported source type: {type(source)!r}")

        if isinstance(source, bytes):
            raw = source
        elif isinstance(source, str) and source.strip().startswith(("{", "[")):
            # 直接传入 JSON 文本
            raw = source.encode("utf-8")
        else:
            try:
                with open(source, "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                raise LoadError(
                    f"cannot read file: {exc}", details={"path": str(source)}
                ) from exc

        # 扩展名优先决定模式；未显式指定时自动探测
        fmt_hint = options.pop("json_type", None) or options.pop("mode", None)
        ext = None
        if not isinstance(source, bytes) and isinstance(source, str) and "." in source:
            ext = source.rsplit(".", 1)[-1].lower()

        if fmt_hint:
            mode = fmt_hint
        elif ext in ("jsonl", "ndjson"):
            mode = "ndjson"
        else:
            mode = "auto"

        df, meta = self._parse(raw, mode)
        return LoadedTable(df=df, format="json", metadata=meta)

    def _parse(self, raw: bytes, mode: str) -> tuple[pl.DataFrame, dict[str, Any]]:
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]

        if mode == "auto":
            stripped = raw.lstrip()
            if stripped.startswith(b"[") or stripped.startswith(b"{"):
                # 单行以 { 开头且含多行 -> NDJSON；否则按标准/records 解析
                lines = [ln for ln in stripped.splitlines() if ln.strip()]
                if len(lines) > 1 and all(ln.lstrip().startswith(b"{") for ln in lines):
                    mode = "ndjson"
                else:
                    mode = "standard"
            else:
                mode = "ndjson"

        try:
            if mode == "ndjson":
                df = pl.read_ndjson(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
                return df, {"json_type": "ndjson"}
            obj = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - 统一转为 LoadError
            raise LoadError(
                f"failed to parse JSON: {exc}", details={"reason": str(exc)}
            ) from exc

        if mode == "records":
            if not isinstance(obj, list):
                raise LoadError(
                    "records JSON must be an array of objects",
                    details={"actual": type(obj).__name__},
                )
            return pl.DataFrame(obj), {"json_type": "records"}

        # standard：对象 -> 单行；数组 -> records；其他 -> 单值表
        if isinstance(obj, dict):
            return pl.DataFrame([obj]), {"json_type": "standard-object"}
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                return pl.DataFrame(obj), {"json_type": "records"}
            return pl.DataFrame({"value": obj}), {"json_type": "standard-list"}
        return pl.DataFrame({"value": [obj]}), {"json_type": "standard-scalar"}

    def metadata(self, source: LoadSource, **options: Any) -> dict[str, Any]:
        return self.load(source, **options).metadata


# =========================================================
# Parquet Loader
# =========================================================


class ParquetLoader(Loader):
    name = "parquet"
    extensions = (".parquet", ".pq")

    def load(self, source: LoadSource, **options: Any) -> LoadedTable:
        if not isinstance(source, (bytes, str, PurePosixPath, PureWindowsPath)):
            raise UnsupportedFormat(f"unsupported source type: {type(source)!r}")

        columns = options.pop("columns", None)

        try:
            df = pl.read_parquet(source, columns=columns)
        except FileNotFoundError as exc:
            raise LoadError(
                f"parquet file not found: {exc}", details={"path": str(source)}
            ) from exc
        except Exception as exc:  # noqa: BLE001 - 非法 parquet 等
            raise LoadError(
                f"failed to read parquet: {exc}", details={"reason": str(exc)}
            ) from exc

        return LoadedTable(df=df, format="parquet", metadata={"columns": list(df.columns)})

    def metadata(self, source: LoadSource, **options: Any) -> dict[str, Any]:
        """利用 parquet footer 的 schema 元数据，不物化数据。"""
        try:
            if isinstance(source, bytes):
                import io

                import pyarrow.parquet as pq

                schema = pq.read_schema(io.BytesIO(source))
                meta: dict[str, Any] = {
                    "columns": schema.names,
                    "dtypes": {name: str(schema.field(name).type) for name in schema.names},
                }
                return meta
            lf = pl.scan_parquet(source)
            schema = lf.collect_schema()
            return {
                "columns": list(schema.names()),
                "dtypes": {name: str(dtype) for name, dtype in schema.items()},
            }
        except Exception as exc:  # noqa: BLE001
            raise LoadError(
                f"failed to read parquet metadata: {exc}", details={"reason": str(exc)}
            ) from exc


# =========================================================
# ARFF Loader（Weka 属性-关系文件格式）
# =========================================================


class ArffLoader(Loader):
    """ARFF（Attribute-Relation File Format）加载器。

    支持 Weka 导出的稠密与稀疏两种数据表示：
    - 稠密：逗号分隔的普通行，缺失值用 ``?``；
    - 稀疏：``{索引 值, 索引 值}`` 形式，未列出的属性为缺失。
    属性类型支持 numeric / real / integer / string / date / 标称枚举
    （``{a,b,c}``）。标称值在数据段可被单引号包裹，支持含空格与逗号的取值。

    实现为自包含纯 Python 解析，不引入 scipy 等额外依赖，
    解析结果直接产出 Polars DataFrame，标称列保留为字符串（非字节串）。
    """

    name = "arff"
    extensions = (".arff",)

    _DATE_FORMATS = (
        "yyyy-MM-dd",
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd'T'HH:mm:ss",
        "yyyy/MM/dd",
        "yyyy/MM/dd HH:mm:ss",
    )

    def load(self, source: LoadSource, **options: Any) -> LoadedTable:
        if not isinstance(source, (bytes, str, PurePosixPath, PureWindowsPath)):
            raise UnsupportedFormat(f"unsupported source type: {type(source)!r}")
        text = self._read_text(source)
        parsed = self._parse(text)
        df = self._to_frame(parsed)
        return LoadedTable(
            df=df,
            format="arff",
            metadata={
                "relation": parsed["relation"],
                "columns": list(df.columns),
            },
        )

    def metadata(self, source: LoadSource, **options: Any) -> dict[str, Any]:
        text = self._read_text(source)
        parsed = self._parse(text)
        return {
            "relation": parsed["relation"],
            "columns": [a["name"] for a in parsed["attributes"]],
            "dtypes": {
                a["name"]: a["type"] for a in parsed["attributes"]
            },
        }

    # ---- 内部工具 ----

    @staticmethod
    def _read_text(source: LoadSource) -> str:
        if isinstance(source, bytes):
            raw = source
        else:
            try:
                with open(str(source), "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                raise LoadError(
                    "cannot read ARFF file",
                    details={"path": str(source), "reason": str(exc)},
                ) from exc
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        for enc in ("utf-8", "gb18030", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise LoadError("failed to decode ARFF file", details={"reason": "unsupported encoding"})

    @staticmethod
    def _strip_comment(line: str) -> str:
        # 仅在「行首」处理注释；% 若出现在引号内不应被误删，这里用引号感知的简单扫描
        in_quote = False
        for i, ch in enumerate(line):
            if ch == "'" and (i == 0 or line[i - 1] != "\\"):
                in_quote = not in_quote
            elif ch == "%" and not in_quote:
                return line[:i]
        return line

    def _parse(self, text: str) -> dict[str, Any]:
        relation = ""
        attributes: list[dict[str, Any]] = []
        data_rows: list[str] = []
        in_data = False

        for raw_line in text.splitlines():
            line = self._strip_comment(raw_line).strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("@relation"):
                relation = line.split(None, 1)[1].strip()
                continue
            if low.startswith("@attribute"):
                body = line.split(None, 1)[1].strip()
                name, spec = self._split_attribute(body)
                attributes.append(self._parse_attribute(name, spec))
                continue
            if low.startswith("@data"):
                in_data = True
                continue
            if in_data:
                data_rows.append(line)

        if not attributes:
            raise LoadError("ARFF 文件缺少 @attribute 定义")
        if not data_rows:
            raise LoadError("ARFF 文件缺少 @data 数据段")
        return {"relation": relation, "attributes": attributes, "rows": data_rows}

    @staticmethod
    def _split_attribute(body: str) -> tuple[str, str]:
        # 属性名可能带引号包裹（含空格），否则取第一个空白分隔
        if body.startswith("'") or body.startswith('"'):
            quote = body[0]
            end = body.find(quote, 1)
            if end == -1:
                raise LoadError("ARFF 属性名引号未闭合")
            return body[1:end], body[end + 1:].strip()
        parts = body.split(None, 1)
        if len(parts) != 2:
            raise LoadError(f"ARFF 属性定义非法：{body!r}")
        return parts[0], parts[1].strip()

    @staticmethod
    def _parse_attribute(name: str, spec: str) -> dict[str, Any]:
        spec = spec.strip()
        low = spec.lower()
        if low in ("numeric", "real", "integer"):
            return {"name": name, "type": "numeric"}
        if low == "string":
            return {"name": name, "type": "string"}
        if low.startswith("date"):
            # date 类型可能带格式串（引号包裹），这里统一按字符串日期列处理
            return {"name": name, "type": "string"}
        if spec.startswith("{") and spec.endswith("}"):
            # 标称枚举：{a,b,c}，取值可能带引号/空格
            inner = spec[1:-1].strip()
            values = ArffLoader._split_nominal(inner)
            return {"name": name, "type": "nominal", "values": values}
        raise LoadError(
            f"ARFF 属性类型不支持：{spec!r}",
            details={"attribute": name},
        )

    @staticmethod
    def _split_nominal(inner: str) -> list[str]:
        values: list[str] = []
        current = ""
        in_quote = False
        quote_char = ""
        for ch in inner:
            if in_quote:
                if ch == quote_char:
                    in_quote = False
                else:
                    current += ch
            elif ch in ("'", '"'):
                in_quote = True
                quote_char = ch
            elif ch == ",":
                values.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip() or (values and not in_quote):
            values.append(current.strip())
        return [v for v in values if v != ""]

    @staticmethod
    def _split_data_line(line: str) -> list[str]:
        """把一条稠密数据行拆成字段（引号感知，逗号分隔）。"""
        fields: list[str] = []
        current = ""
        in_quote = False
        quote_char = ""
        for ch in line:
            if in_quote:
                if ch == quote_char:
                    in_quote = False
                else:
                    current += ch
            elif ch in ("'", '"'):
                in_quote = True
                quote_char = ch
            elif ch == ",":
                fields.append(current.strip())
                current = ""
            else:
                current += ch
        fields.append(current.strip())
        return fields

    def _parse_sparse(self, line: str, n_attrs: int) -> list[str | None]:
        inner = line.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]
        out: list[str | None] = [None] * n_attrs
        for token in self._split_data_line(inner):
            token = token.strip()
            if not token:
                continue
            parts = token.split(None, 1)
            if len(parts) != 2:
                raise LoadError(f"ARFF 稀疏数据项非法：{token!r}")
            try:
                idx = int(parts[0])
            except ValueError as exc:
                raise LoadError(f"ARFF 稀疏索引非法：{parts[0]!r}") from exc
            if not 0 <= idx < n_attrs:
                raise LoadError(f"ARFF 稀疏索引越界：{idx}")
            out[idx] = parts[1].strip().strip("'\"")
        return out

    def _to_frame(self, parsed: dict[str, Any]) -> pl.DataFrame:
        attributes = parsed["attributes"]
        n_attrs = len(attributes)
        names = [a["name"] for a in attributes]
        columns: dict[str, list[Any]] = {n: [] for n in names}
        types: dict[str, str] = {a["name"]: a["type"] for a in attributes}

        for line in parsed["rows"]:
            if line.startswith("{"):
                fields = self._parse_sparse(line, n_attrs)
            else:
                fields = self._split_data_line(line)
            if len(fields) != n_attrs:
                raise LoadError(
                    "ARFF 数据行字段数与属性数不一致",
                    details={"expected": n_attrs, "actual": len(fields)},
                )
            for name, raw in zip(names, fields):
                columns[name].append(self._coerce(raw, types[name]))

        return pl.DataFrame(columns)

    @staticmethod
    def _coerce(raw: str | None, atype: str) -> Any:
        if raw is None:
            return None
        raw = raw.strip()
        if raw == "?" or raw == "":
            return None
        if atype == "numeric":
            try:
                return float(raw)
            except ValueError:
                return raw
        return raw


# =========================================================
# Loader 注册表
# =========================================================


class LoaderRegistry:
    """根据格式或扩展名选择 Loader。"""

    def __init__(self) -> None:
        self._loaders: dict[str, Loader] = {}
        self._extensions: dict[str, str] = {}

    def register(self, loader: Loader) -> None:
        if not loader.name:
            raise ValueError(
                "loader.name must not be empty"
            )

        if loader.name in self._loaders:
            raise ValueError(
                f"loader already registered: {loader.name}"
            )

        self._loaders[loader.name] = loader

        for extension in loader.extensions:
            if extension in self._extensions:
                raise ValueError(
                    f"extension conflict: {extension}"
                )

            self._extensions[extension] = loader.name

    def get(self, name: str) -> Loader:
        try:
            return self._loaders[name]
        except KeyError as exc:
            raise UnsupportedFormat(
                f"no loader registered for format {name!r}",
                details={
                    "format": name,
                    "available": sorted(self._loaders),
                },
            ) from exc

    def get_loader(
        self,
        source: str | LoadSource,
    ) -> Loader:
        if isinstance(source, bytes):
            raise UnsupportedFormat(
                "bytes source requires an explicit format"
            )

        filename = (
            str(source)
            .replace("\\", "/")
            .rsplit("/", 1)[-1]
        )

        if "." not in filename:
            raise UnsupportedFormat(
                "file extension is missing",
                details={"filename": filename},
            )

        extension = (
            f".{filename.rsplit('.', 1)[-1].lower()}"
        )

        loader_name = self._extensions.get(
            extension
        )

        if loader_name is None:
            raise UnsupportedFormat(
                f"unsupported file extension: {extension}",
                details={
                    "extension": extension,
                    "supported": self.supported_extensions,
                },
            )

        return self._loaders[loader_name]

    def load(
        self,
        source: str | LoadSource,
        data: bytes | None = None,
        **options: Any,
    ) -> LoadedTable:
        loader = self.get_loader(source)
        payload = data if data is not None else source

        return loader.load(
            payload,
            **options,
        )

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(self._extensions)


def default_registry() -> LoaderRegistry:
    registry = LoaderRegistry()

    for loader in (
        CSVLoader(),
        ExcelLoader(),
        JSONLoader(),
        ParquetLoader(),
        ArffLoader(),
    ):
        registry.register(loader)

    return registry


REGISTRY = default_registry()
