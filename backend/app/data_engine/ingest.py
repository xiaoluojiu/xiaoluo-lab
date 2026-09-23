"""流式入库引擎（数据规模上限与吞吐的核心）。

背景
----
改造前，一份文件要进到数据集版本里要经历三次「全量常驻内存」：

1. ``FileService.read()`` → 整个文件的 ``bytes``；
2. ``REGISTRY.load(..., data=content)`` → 完整 DataFrame（CSV 还要先 decode 成
   一个更大的 str 再 encode 回 utf-8，峰值是文件体积的数倍）；
3. ``DatasetService.create_version()`` → ``io.BytesIO()`` 里再攒一份 Parquet。

于是「上传上限」实际上被进程内存卡住，硬编码 100 MB 只是把 OOM 提前变成了
一个 422 错误码。真正要拆的不是那个常量，是上面这三段。

本模块的设计
------------
把「解析」和「落盘」合并成一次流式遍历，任何时刻常驻内存只与**块大小**有关，
与文件总体积无关。

两条路径，按体积自适应（见 ``_ingest_streaming``）：

- **sink**：``scan_*`` → ``sink_parquet``。快（实测边际吞吐约 730 MB/s），
  但 Polars 1.44 对 CSV 源并不真正流式，峰值内存 ≈ 3× 文件体积；
- **chunked**：按字节切块 → 逐块解析 → 增量写 Parquet 行组。内存恒定约 400 MB，
  实测「数据翻倍时内存倍率 0.97×」（即真正有界），代价是边际吞吐约 120 MB/s。

因此小文件走 sink、大文件走 chunked（``INGEST_STREAMING_THRESHOLD_BYTES`` 分界）。
**Chunked 是「突破 100 MB 上限」的关键**：它把内存从「随文件线性增长」变成常量。

- 可流式的格式：CSV / TSV / TXT、NDJSON、Parquet、**稠密 ARFF**
  （ARFF 的 ``@data`` 段就是逗号分隔文本，头部单独扫一遍即可定 schema）；
- 不可流式的格式：XLSX / XLS / 标准 JSON 对象、**稀疏 ARFF** → 只能整体物化
  （底层库不提供流式读），此时用 ``INGEST_MAX_MATERIALIZE_BYTES`` 明确拒绝
  而不是把进程撑爆；
- GBK / GB18030 等 Polars 不直接支持的编码：先流式转码为 UTF-8 临时文件
  （增量解码，正确处理跨块的多字节字符），再走流式路径。

产出统一是 Parquet：列式 + 带 footer 统计信息，让后续「只读 3 列」「按范围
过滤」能真正下推，而不是每次全表解码。

临时文件的生命周期
------------------
``build_lazy_frame`` 可能需要把源文件转码到 ``work_dir`` 下的临时文件，并让返回的
LazyFrame 指向它。**调用方必须在 collect/sink 完成后清理 work_dir**——
``ingest_to_parquet`` 用 ``TemporaryDirectory`` 包住整个过程，因此对外部调用者而言
这条约束是透明的。
"""

from __future__ import annotations

import codecs
import io
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from app.core.config import settings
from app.core.exceptions import AppException
from app.data_engine.exceptions import LoadError, UnsupportedFormat

# 复用 loaders 里的编码/分隔符探测与 ARFF 头部解析（同包工具，避免两份实现漂移）。
from app.data_engine.loaders import (
    SUPPORTED_ENCODINGS,
    ArffLoader,
    _detect_encoding,
    _detect_separator,
    _normalize_encoding,
    _read_head,
)

# ---------------------------------------------------------
# 格式分类
# ---------------------------------------------------------

CSV_FORMATS = ("csv", "tsv", "txt")
NDJSON_FORMATS = ("jsonl", "ndjson")
PARQUET_FORMATS = ("parquet", "pq")
# ARFF 的数据段（@data 之后）本质就是「逗号分隔 + 引号包裹标称值」的文本，
# 可以像 CSV 一样按字节分块流式解析；只有头部（@relation/@attribute）需要
# 单独扫描一遍。稀疏表示（{idx val,...}）不是定宽列，退回整体物化。
ARFF_FORMATS = ("arff",)
# 需要整体物化的格式：底层读取器不提供流式接口。
MATERIALIZE_FORMATS = ("xlsx", "xls", "json")

# ARFF @attribute 类型 → Polars dtype。
# 标称枚举与 string/date 都按字符串列处理（与 ArffLoader 的既有语义一致）。
_ARFF_DTYPE = {
    "numeric": pl.Float64,
    "string": pl.Utf8,
    "nominal": pl.Utf8,
}

# Polars scan_csv 能直接吃的编码；其余需要先转码。
_POLARS_NATIVE_ENCODINGS = {
    "utf-8": "utf8",
    "utf-8-sig": "utf8",
    "latin-1": "utf8-lossy",
}

# 物化格式 → LoaderRegistry 里的 loader 名（xls/xlsm 都归 excel）。
_MATERIALIZE_LOADER = {
    "xlsx": "excel",
    "xls": "excel",
    "arff": "arff",
    "json": "json",
}


def normalize_format(source: str | Path, explicit: str | None = None) -> str:
    """归一化格式名（小写、去点、统一别名）。"""
    if explicit:
        fmt = explicit.strip().lower().lstrip(".")
    else:
        fmt = Path(str(source)).suffix.lower().lstrip(".")

    aliases = {"pq": "parquet", "xlsm": "xlsx", "tdv": "tsv", "jsonl": "ndjson"}
    fmt = aliases.get(fmt, fmt)

    if (
        fmt in CSV_FORMATS
        or fmt in NDJSON_FORMATS
        or fmt in PARQUET_FORMATS
        or fmt in ARFF_FORMATS
    ):
        return fmt
    if fmt in MATERIALIZE_FORMATS:
        return fmt
    raise UnsupportedFormat(
        f"不支持的入库格式：{fmt or '(缺少扩展名)'}",
        details={
            "format": fmt,
            "streaming": list(
                CSV_FORMATS + NDJSON_FORMATS + PARQUET_FORMATS + ARFF_FORMATS
            ),
            "materialize": list(MATERIALIZE_FORMATS),
        },
    )


def supports_streaming(fmt: str) -> bool:
    """该格式能否走常量内存的流式路径。

    ARFF 也算：其 ``@data`` 段是定宽逗号分隔文本，头部单独扫描一遍即可
    确定 schema，因此稠密 ARFF 可以按块流式解析（见 ``_ingest_arff_streaming``）。
    稀疏 ARFF 不是定宽列，会在运行时退回整体物化。
    """
    return fmt in (*CSV_FORMATS, *NDJSON_FORMATS, *PARQUET_FORMATS, *ARFF_FORMATS)


# ---------------------------------------------------------
# 结果对象
# ---------------------------------------------------------


@dataclass
class IngestResult:
    """一次入库的结果与可观测指标。"""

    dest_path: Path
    format: str
    # chunked / chunked-utf8 / streaming / streaming-reinfer / streaming-utf8
    # / materialized / materialized(streaming-off)
    strategy: str
    row_count: int
    column_count: int
    schema: dict[str, str] = field(default_factory=dict)
    source_bytes: int = 0
    dest_bytes: int = 0
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if not self.source_bytes:
            return 1.0
        return round(self.dest_bytes / self.source_bytes, 4)

    @property
    def throughput_mb_s(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return round(self.source_bytes / 1048576 / self.elapsed_seconds, 2)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "strategy": self.strategy,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "source_bytes": self.source_bytes,
            "dest_bytes": self.dest_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_mb_s": self.throughput_mb_s,
            "compression_ratio": self.compression_ratio,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------
# 配置读取（配置层异常时退回保守默认，不让入库整体失败）
# ---------------------------------------------------------


def _row_group_rows() -> int:
    try:
        return max(1024, int(settings.INGEST_ROW_GROUP_ROWS))
    except Exception:  # pragma: no cover
        return 262144


def _max_materialize_bytes() -> int:
    try:
        return int(settings.INGEST_MAX_MATERIALIZE_BYTES)
    except Exception:  # pragma: no cover
        return 1024 * 1024 * 1024


def _streaming_enabled() -> bool:
    try:
        return bool(settings.INGEST_STREAMING_ENABLED)
    except Exception:  # pragma: no cover
        return True


def _infer_rows() -> int | None:
    try:
        value = int(settings.INGEST_SCHEMA_INFER_ROWS)
    except Exception:  # pragma: no cover
        return 50000
    return value if value > 0 else None


def _chunk_bytes() -> int:
    """分块入库的单块字节数。

    下限 64 KB 只是**防止 0/负数导致缓冲区永不落地**的安全网，不是性能下限 ——
    真实的性能拐点在 MB 级（见 ``INGEST_CHUNK_BYTES`` 的说明）。刻意不写死成
    「≥1 MB」：那会让「把块调小以便测试」变得不可能，而分块边界逻辑恰恰是最需要
    被测试覆盖的地方。
    """
    try:
        return max(64 * 1024, int(settings.INGEST_CHUNK_BYTES))
    except Exception:  # pragma: no cover
        return 32 * 1024 * 1024


def _streaming_threshold_bytes() -> int:
    """小于该体积走 sink（快），大于该体积走分块（内存有界）。"""
    try:
        return max(0, int(settings.INGEST_STREAMING_THRESHOLD_BYTES))
    except Exception:  # pragma: no cover
        return 256 * 1024 * 1024


# ---------------------------------------------------------
# 工具
# ---------------------------------------------------------


def count_rows(path: Path | str) -> int:
    """统计 Parquet 行数。

    Polars 对 ``select(pl.len())`` 有专门的优化：直接读 footer 元数据，
    不会真的解码数据页——所以这个「全表计数」是廉价的。
    """
    try:
        return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
    except Exception:  # noqa: BLE001 - 计数失败不应让入库整体失败
        return 0


def read_schema(path: Path | str) -> dict[str, str]:
    """读取 Parquet schema（同样只读 footer）。"""
    try:
        schema = pl.scan_parquet(str(path)).collect_schema()
        return {name: str(dtype) for name, dtype in schema.items()}
    except Exception:  # noqa: BLE001
        return {}


def transcode_to_utf8(
    src: Path | str,
    dst: Path | str,
    encoding: str,
) -> None:
    """把任意受支持编码的文本文件增量转码为 UTF-8。

    用 ``codecs.iterdecode`` 而不是「按块 decode」：GBK/GB18030 的一个汉字占
    2~4 字节，按固定块切会把多字节字符劈成两半，逐块 ``decode`` 会抛
    ``UnicodeDecodeError`` 或产出替换字符。``iterdecode`` 内部维护解码器状态，
    能正确处理跨块边界；同时 ``utf-8-sig`` 会自动吃掉 BOM。
    """
    src = Path(str(src))
    dst = Path(str(dst))

    try:
        with src.open("rb") as fin, dst.open("wb") as fout:
            for piece in codecs.iterdecode(fin, encoding):
                fout.write(piece.encode("utf-8"))
    except LookupError as exc:  # pragma: no cover - 编码名非法
        raise LoadError(
            f"不支持的编码：{encoding}",
            details={"supported": SUPPORTED_ENCODINGS},
        ) from exc
    except OSError as exc:
        raise LoadError(
            "转码失败",
            details={"path": str(src), "reason": str(exc)},
        ) from exc


class _StreamingNotApplicable(Exception):
    """流式路径对该文件不适用（如稀疏 ARFF）。

    调用方收到它应当**退回整体物化** —— 物化是原有的、已验证的语义。
    用它兜底的目的是保证「同一个文件无论走哪条路，结果都一致」，
    而不会出现「大文件走快路径、小文件走旧路径 → 结果不同」这种最难查的缺陷。
    """


@dataclass
class _ArffPlan:
    """ARFF 流式入库所需的全部信息（来自一次头部扫描）。"""

    scan_path: Path
    data_offset: int
    schema: dict[str, Any]
    columns: list[str]
    dense: bool


def _prepare_arff_source(
    source: Path,
    options: dict[str, Any],
    work_dir: Path | None,
    warnings: list[str],
) -> tuple[Path, str]:
    """确定 ARFF 的可扫描路径与编码（非 Polars 原生编码先转码为 UTF-8）。"""
    encoding = _normalize_encoding(options.get("encoding", "auto"))
    encoding = _detect_encoding(_read_head(str(source)), encoding)

    if _POLARS_NATIVE_ENCODINGS.get(encoding) is not None:
        return source, encoding

    if work_dir is None:
        raise LoadError(
            f"编码 {encoding} 需要转码，但未提供工作目录",
            details={"encoding": encoding},
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    scan_path = work_dir / f"{source.name}.utf8.tmp"
    transcode_to_utf8(source, scan_path, encoding)
    warnings.append(f"源编码 {encoding} 已转码为 UTF-8 后入库（多一次磁盘 I/O）")
    return scan_path, "utf-8"


def _scan_arff_header(scan_path: Path, encoding: str) -> _ArffPlan:
    """只扫头部（到 ``@data`` 为止），拿到属性定义与数据段字节偏移。

    用 ``for raw in fh`` 逐行读，而不是 ``fh.read()``：头部通常只有几十行，
    但文件可能有几百 MB —— 任何一次性读取都会让内存重新与文件体积挂钩，
    那正是本模块要消灭的东西。
    """

    def _decode(raw: bytes) -> str:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            return raw.decode(encoding, errors="replace")

    attributes: list[dict[str, Any]] = []
    data_offset: int | None = None
    dense = True

    with Path(scan_path).open("rb") as fh:
        for raw in fh:
            line = ArffLoader._strip_comment(_decode(raw)).strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                body = line.split(None, 1)[1].strip()
                name, spec = ArffLoader._split_attribute(body)
                attributes.append(ArffLoader._parse_attribute(name, spec))
                continue
            if low.startswith("@data"):
                data_offset = fh.tell()
                break
            # @relation 等其它指令行：忽略
        else:
            raise LoadError("ARFF 文件缺少 @data 标记", details={"path": str(scan_path)})

        if not attributes:
            raise LoadError(
                "ARFF 文件缺少 @attribute 定义", details={"path": str(scan_path)}
            )

        # 用第一条数据行判断稠密/稀疏：稀疏行形如 ``{0 1, 3 4}``，不是定宽列。
        for raw in fh:
            line = ArffLoader._strip_comment(_decode(raw)).strip()
            if not line:
                continue
            dense = not line.startswith("{")
            break

    schema: dict[str, Any] = {}
    for attr in attributes:
        dtype = _ARFF_DTYPE.get(attr["type"])
        if dtype is None:  # pragma: no cover - _parse_attribute 已收敛类型集合
            raise UnsupportedFormat(
                f"ARFF 属性类型无法流式映射：{attr['type']}",
                details={"attribute": attr["name"]},
            )
        schema[str(attr["name"])] = dtype

    return _ArffPlan(
        scan_path=Path(scan_path),
        data_offset=int(data_offset or 0),
        schema=schema,
        columns=[str(a["name"]) for a in attributes],
        dense=dense,
    )


def _csv_scan_kwargs(options: dict[str, Any], infer_len: int | None) -> dict[str, Any]:
    """把调用方选项映射成 scan_csv 参数。"""
    kwargs: dict[str, Any] = {
        "infer_schema_length": infer_len,
        "low_memory": True,
        "truncate_ragged_lines": True,
    }
    if options.get("has_header") is not None:
        kwargs["has_header"] = bool(options["has_header"])
    separator = options.get("separator") or options.get("sep")
    if separator:
        kwargs["separator"] = separator
    if options.get("null_values") is not None:
        kwargs["null_values"] = options["null_values"]
    if options.get("quote_char"):
        kwargs["quote_char"] = options["quote_char"]
    if options.get("comment_prefix"):
        kwargs["comment_prefix"] = options["comment_prefix"]
    if options.get("try_parse_dates"):
        kwargs["try_parse_dates"] = True
    return kwargs


def _prepare_csv_source(
    source: Path,
    options: dict[str, Any],
    work_dir: Path | None,
    warnings: list[str],
) -> tuple[Path, dict[str, Any]]:
    """探测 CSV 的编码/分隔符，必要时转码到工作目录。

    返回 ``(可扫描路径, scan_csv 附加参数)``。
    """
    encoding = _normalize_encoding(options.get("encoding", "auto"))
    head = _read_head(str(source))
    encoding = _detect_encoding(head, encoding)

    separator = options.get("separator") or options.get("sep")
    separator = separator or _detect_separator(head, encoding)

    scan_path = source
    polars_encoding = _POLARS_NATIVE_ENCODINGS.get(encoding)

    if polars_encoding is None:
        if work_dir is None:
            raise LoadError(
                f"编码 {encoding} 需要转码，但未提供工作目录",
                details={"encoding": encoding},
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        scan_path = work_dir / f"{source.name}.utf8.tmp"
        transcode_to_utf8(source, scan_path, encoding)
        warnings.append(f"源编码 {encoding} 已转码为 UTF-8 后入库（多一次磁盘 I/O）")
        polars_encoding = "utf8"

    extra: dict[str, Any] = {
        "encoding": polars_encoding,
        "separator": separator,
    }
    if options.get("has_header") is not None:
        extra["has_header"] = bool(options["has_header"])
    return scan_path, extra


def build_lazy_frame(
    source: Path | str,
    fmt: str | None = None,
    options: dict[str, Any] | None = None,
    *,
    work_dir: Path | None = None,
    warnings: list[str] | None = None,
) -> pl.LazyFrame:
    """构造懒执行帧（供谓词/投影下推使用，不物化数据）。

    只有 Parquet 与文本格式能真正下推；对 CSV 也能下推（Polars 在扫描期就应用
    filter/projection），但每次都要重新解析文本，因此「先入库成 Parquet 再分析」
    依然是更划算的路径。

    ``work_dir``：CSV 需要转码时的临时文件落点。返回的 LazyFrame 可能指向
    该目录下的文件 → **调用方必须在消费完 LazyFrame 之后再清理 work_dir**。
    """
    source = Path(str(source))
    options = dict(options or {})
    fmt = normalize_format(source, options.pop("format", None))
    warnings = warnings if warnings is not None else []
    infer_len = options.pop("infer_schema_length", _infer_rows())

    if fmt in CSV_FORMATS:
        scan_path, extra = _prepare_csv_source(source, options, work_dir, warnings)
        return pl.scan_csv(
            str(scan_path),
            **_csv_scan_kwargs({**options, **extra}, infer_len),
        )

    if fmt in NDJSON_FORMATS:
        return pl.scan_ndjson(str(source), infer_schema_length=infer_len)

    if fmt in PARQUET_FORMATS:
        # Polars 1.4x 起 scan_parquet 不再接受 columns 参数，投影一律走 select()：
        # 这同样是「下推」——优化器会把投影推进到 parquet 读取层。
        lf = pl.scan_parquet(str(source))
        columns = options.get("columns")
        if columns:
            lf = lf.select(columns)
        return lf

    raise UnsupportedFormat(
        f"格式 {fmt} 不支持懒加载（需要整体物化）",
        details={"format": fmt},
    )


def _iter_text_chunks(
    fh: Any,
    chunk_bytes: int,
    quote: int = 0x22,
) -> Iterator[bytes]:
    """按字节切分文本流，切点对齐到「不在引号内的换行」。

    为什么需要引号感知：RFC 4180 允许字段值里含换行（只要整体被引号包裹），
    直接按 ``\\n`` 切会把一条记录劈成两半，两块都解析失败。

    为什么用 ``bytes.rfind`` + ``bytes.count``：逐字节扫描 187 MB 在纯 Python 里
    要十几秒；``rfind``/``count`` 都在 C 层，整份文件只多花约 50 ms。
    ``""``（引号转义）成对出现，不改变奇偶性，因此只数奇偶是安全的。

    末尾没有换行的残块会在收尾时单独 yield（最后一行的换行符常被省略）。
    """
    buf = bytearray()

    while True:
        block = fh.read(chunk_bytes)
        if not block:
            break
        buf += block

        cut = buf.rfind(b"\n")
        # 若最后一个换行落在引号内，就往回找前一个换行；一直找到「引号外」的换行为止。
        while cut >= 0 and (buf.count(quote, 0, cut) & 1):
            cut = buf.rfind(b"\n", 0, cut)

        if cut < 0:
            # 病态输入（例如整个文件就是一个含换行的引号字段）：强制切，宁可让解析报错，
            # 也不让缓冲区无界增长。
            if len(buf) > chunk_bytes * 8:
                yield bytes(buf)
                buf.clear()
            continue

        yield bytes(buf[: cut + 1])
        del buf[: cut + 1]

    if buf:
        yield bytes(buf)


def _ingest_chunked(
    source: Path,
    dest: Path,
    fmt: str,
    options: dict[str, Any],
    work_dir: Path,
    warnings: list[str],
    *,
    all_strings: bool = False,
) -> int:
    """按字节分块解析 → 增量写 Parquet 行组，返回写入行数。

    内存 ≈ ``INGEST_CHUNK_BYTES``（单块 DataFrame）+ 写出缓冲，**与文件总体积无关**。

    为什么不用 ``scan_csv`` → ``sink_parquet``：实测该组合在 Polars 1.44 上并未真正
    流式 —— 峰值内存 ≈ 全量物化（只低约 10%），4 M 行 CSV 仍需 631 MB。
    要兑现「突破 100 MB 上限」，必须自己控制读取粒度。
    """
    import pyarrow.parquet as pq

    chunk_bytes = _chunk_bytes()
    row_group = _row_group_rows()

    if fmt in CSV_FORMATS:
        scan_path, extra = _prepare_csv_source(source, options, work_dir, warnings)
        read_kwargs: dict[str, Any] = {
            "separator": extra.get("separator") or ",",
            "encoding": extra.get("encoding") or "utf8",
            # ★ 必须与 ``_csv_scan_kwargs`` 保持一致。两条路径由「文件体积是否超过阈值」
            # 决定，若对参差行的处理不同，同一个文件会因大小不同而产出不同结果 ——
            # 这种不一致比「严格」或「宽松」本身危害更大。此处取宽松：真实数据集里
            # 偶尔多一个分隔符很常见，为一行畸形数据整体失败得不偿失。
            "truncate_ragged_lines": True,
        }
        has_header = bool(extra.get("has_header", True))
    else:  # NDJSON：每行一条记录，天然可按行切分
        scan_path = source
        read_kwargs = {}
        has_header = False

    # all_strings ⇒ 首块按「不推断」读，Polars 会把所有列当 Utf8。
    infer_len = 0 if all_strings else options.get("infer_schema_length", _infer_rows())
    projection = options.get("columns") or None

    writer = None
    schema: dict[str, Any] | None = None
    total = 0

    try:
        with Path(scan_path).open("rb") as fh:
            first = True
            for raw in _iter_text_chunks(fh, chunk_bytes):
                if fmt in CSV_FORMATS:
                    frame = pl.read_csv(
                        io.BytesIO(raw),
                        has_header=has_header if first else False,
                        schema=schema,
                        infer_schema_length=infer_len if first else None,
                        **read_kwargs,
                    )
                else:
                    frame = pl.read_ndjson(
                        io.BytesIO(raw),
                        schema=schema,
                        infer_schema_length=infer_len if first else None,
                    )

                if first:
                    schema = dict(frame.schema)

                if projection:
                    frame = frame.select(projection)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(dest), frame.to_arrow().schema, compression="zstd"
                    )

                # 切片是零拷贝的：用行组大小把大块拆成多个行组，兼顾下推粒度。
                pos = 0
                height = frame.height
                total += height
                while pos < height:
                    writer.write_table(frame.slice(pos, row_group).to_arrow())
                    pos += row_group

                first = False

        if writer is None:
            raise LoadError("源文件为空，没有可入库的记录", details={"path": str(source)})
    finally:
        if writer is not None:
            writer.close()

    return total


def _ingest_arff_streaming(
    source: Path,
    dest: Path,
    options: dict[str, Any],
    work_dir: Path,
    warnings: list[str],
) -> str:
    """稠密 ARFF 的流式入库：头部扫描一次 + ``@data`` 段按 CSV 分块解析。

    与 ``_ingest_chunked`` 只有两点不同：
    ① 数据段从 ``data_offset`` 开始、没有表头行；
    ② schema 由 ARFF 头部**显式给定**（不推断），因此跨块类型必然一致。

    为什么值得单独做：ARFF 走整体物化时是纯 Python 逐行逐格解析
    （``fh.read()`` → ``splitlines()`` → 每格一次 ``_coerce``），
    10 M 行 × 10 列实测要使整个请求耗时 140 s（4.04 亿字节）；
    走本路径则是 Polars 多线程解析 + 增量写行组，内存有界。

    返回 ``strategy`` 值 ``chunked-arff``。
    """
    import pyarrow.parquet as pq

    scan_path, encoding = _prepare_arff_source(source, options, work_dir, warnings)
    plan = _scan_arff_header(scan_path, encoding)

    if not plan.dense:
        raise _StreamingNotApplicable("稀疏 ARFF 不是定宽列")

    chunk_bytes = _chunk_bytes()
    row_group = _row_group_rows()
    projection = options.get("columns") or None

    writer = None
    fh = Path(plan.scan_path).open("rb")
    try:
        if plan.data_offset:
            fh.seek(plan.data_offset)

        # 切点按单引号判奇偶：ARFF 的标称值用 ' 包裹，虽然标准不允许字段内换行，
        # 但保持与 CSV 分块同样的引号感知逻辑更稳妥。
        for raw in _iter_text_chunks(fh, chunk_bytes, quote=0x27):
            frame = pl.read_csv(
                io.BytesIO(raw),
                has_header=False,
                # 显式 schema ⇒ 不推断，跨块类型恒定。
                schema=plan.schema,
                separator=",",
                quote_char="'",
                # ARFF 用 ? 表示缺失；不声明为空值的话 numeric 列会解析失败。
                null_values=["?"],
                # 字段数与属性数不一致时报错（与 ArffLoader 语义一致），
                # 不静默截断 —— 一旦抛错就退回整体物化。
                truncate_ragged_lines=False,
            )
            if projection:
                frame = frame.select(projection)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(dest), frame.to_arrow().schema, compression="zstd"
                )

            pos = 0
            height = frame.height
            while pos < height:
                writer.write_table(frame.slice(pos, row_group).to_arrow())
                pos += row_group

        if writer is None:
            raise LoadError(
                "ARFF 数据段为空，没有可入库的记录",
                details={"path": str(scan_path)},
            )
    finally:
        fh.close()
        if writer is not None:
            writer.close()

    return "chunked-arff"


def _materialize(
    source: Path,
    fmt: str,
    options: dict[str, Any],
) -> pl.DataFrame:
    """物化兜底路径（XLSX / ARFF / 标准 JSON）。"""
    size = source.stat().st_size if source.is_file() else 0
    limit = _max_materialize_bytes()
    if limit and size > limit:
        raise AppException(
            f"该格式（{fmt}）底层不支持流式读取，文件 {size / 1048576:.1f} MB 超过物化上限 "
            f"{limit / 1048576:.0f} MB。请先转为 CSV / Parquet 再导入。",
            code="INGEST_MATERIALIZE_TOO_LARGE",
            details={"format": fmt, "size": size, "max_materialize_bytes": limit},
        )

    from app.data_engine.loaders import REGISTRY

    loader = REGISTRY.get(_MATERIALIZE_LOADER.get(fmt, fmt))
    loaded = loader.load(str(source), **options)
    return loaded.df


# ---------------------------------------------------------
# 主入口
# ---------------------------------------------------------


def ingest_to_parquet(
    source: Path | str,
    dest_path: Path | str,
    *,
    fmt: str | None = None,
    options: dict[str, Any] | None = None,
) -> IngestResult:
    """把源文件流式转为 Parquet。

    ``dest_path`` 必须是**临时路径**：本函数直接写入，不做原子替换。
    由调用方（``DatasetService.create_version_from_source``）负责
    写完后 ``storage.promote()`` 到最终 key，保证存储层不会出现半截文件。
    """
    source = Path(str(source))
    dest = Path(str(dest_path))
    options = dict(options or {})
    fmt = normalize_format(source, options.pop("format", None))

    if not source.is_file():
        raise LoadError("源文件不存在", details={"path": str(source)})

    started = time.perf_counter()
    warnings: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 转码临时文件放在独立工作目录，随 with 块一起消失。
    with tempfile.TemporaryDirectory(prefix="xiaoluo_ingest_") as work_dir_raw:
        work_dir = Path(work_dir_raw)

        if supports_streaming(fmt) and _streaming_enabled():
            try:
                strategy = _ingest_streaming(
                    source, dest, fmt, options, work_dir, warnings
                )
            except _StreamingNotApplicable as exc:
                # 例如稀疏 ARFF：流式路径不适用，退回整体物化。
                # 物化是改造前就存在、已被测试覆盖的语义，用它兜底最稳。
                warnings.append(f"{exc}；已改用整体物化")
                dest.unlink(missing_ok=True)
                frame = _materialize(source, fmt, options)
                if options.get("columns"):
                    frame = frame.select(options["columns"])
                frame.write_parquet(dest, row_group_size=_row_group_rows())
                strategy = "materialized(streaming-unsupported)"
        elif supports_streaming(fmt):
            frame = build_lazy_frame(source, fmt, options, work_dir=work_dir, warnings=warnings).collect()
            if options.get("columns"):
                frame = frame.select(options["columns"])
            frame.write_parquet(dest, row_group_size=_row_group_rows())
            strategy = "materialized(streaming-off)"
            warnings.append("流式入库已被配置关闭（INGEST_STREAMING_ENABLED=false）")
        else:
            frame = _materialize(source, fmt, options)
            frame.write_parquet(dest, row_group_size=_row_group_rows())
            strategy = "materialized"

    row_count = count_rows(dest)
    schema = read_schema(dest)

    return IngestResult(
        dest_path=dest,
        format=fmt,
        strategy=strategy,
        row_count=row_count,
        column_count=len(schema),
        schema=schema,
        source_bytes=source.stat().st_size,
        dest_bytes=dest.stat().st_size,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        warnings=warnings,
    )


def _ingest_streaming(
    source: Path,
    dest: Path,
    fmt: str,
    options: dict[str, Any],
    work_dir: Path,
    warnings: list[str],
) -> str:
    """流式入库：在「快」与「内存有界」之间按文件体积自适应。

    两条实测过的路径：

    - ``sink``：``scan_csv`` → ``sink_parquet``。边际吞吐约 730 MB/s（Polars 多线程
      解析 + 并行压缩），但峰值内存 ≈ 3× 文件体积 —— 它并不真正流式。
    - ``chunked``：按字节分块 → 逐块解析 → 增量写行组。内存恒定约 400 MB，
      边际吞吐约 120 MB/s（少了并行压缩）。

    所以按 ``INGEST_STREAMING_THRESHOLD_BYTES`` 分界：装得下就走快的，装不下就走有界的。
    两者互为兜底 —— 任一路径失败都会落到另一条，最后才是「全字符串列」。

    返回的 strategy 值：``streaming`` / ``chunked`` / ``chunked-utf8`` /
    ``chunked-arff`` / ``streaming-reinfer`` / ``streaming-utf8``。
    """
    # ARFF：头部独立扫一遍，@data 段按 CSV 分块解析。
    # 任何异常（稀疏、字段数不齐、schema 冲突……）都退回整体物化 ——
    # 物化是原有的已验证语义，用它兜底可保证两条路径结果一致。
    if fmt in ARFF_FORMATS:
        try:
            return _ingest_arff_streaming(source, dest, options, work_dir, warnings)
        except _StreamingNotApplicable:
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底就是本分支的目的
            warnings.append(
                f"ARFF 流式入库失败（{type(exc).__name__}: {exc}），回退整体物化"
            )
            dest.unlink(missing_ok=True)
            raise _StreamingNotApplicable("ARFF 流式入库失败") from exc

    row_group = _row_group_rows()

    def _attempt_chunked(*, all_strings: bool, label: str) -> bool:
        try:
            _ingest_chunked(
                source,
                dest,
                fmt,
                options,
                work_dir,
                warnings,
                all_strings=all_strings,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - 分块路径异常种类多，统一降级
            warnings.append(f"{label}失败：{type(exc).__name__}: {exc}")
            # 中途失败会留下半截文件，必须清掉再换策略。
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _attempt_sink(infer_len: int | None, label: str) -> bool:
        try:
            lf = build_lazy_frame(
                source,
                fmt,
                {**options, "infer_schema_length": infer_len},
                work_dir=work_dir,
                warnings=warnings,
            )
            if options.get("columns"):
                lf = lf.select(options["columns"])
            lf.sink_parquet(str(dest), row_group_size=row_group, compression="zstd")
            return True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{label} 失败：{type(exc).__name__}: {exc}")
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    is_text = fmt in (*CSV_FORMATS, *NDJSON_FORMATS)
    size = source.stat().st_size
    # 大文件优先分块（内存有界）；小文件优先 sink（吞吐高约 6×）。
    chunked_first = is_text and size > _streaming_threshold_bytes()

    if is_text:
        if chunked_first:
            if _attempt_chunked(all_strings=False, label="分块入库"):
                return "chunked"
            if _attempt_chunked(all_strings=True, label="全字符串列分块入库"):
                warnings.append("已退化为全字符串列导入（数值列需在数据操作阶段显式转换）")
                return "chunked-utf8"
            warnings.append("分块入库失败，回退 scan + sink 路径")
        else:
            infer = options.get("infer_schema_length", _infer_rows())
            if _attempt_sink(infer, "流式入库"):
                return "streaming"
            warnings.append("scan + sink 失败，回退分块路径")
            if _attempt_chunked(all_strings=False, label="分块入库"):
                return "chunked"
            if _attempt_chunked(all_strings=True, label="全字符串列分块入库"):
                warnings.append("已退化为全字符串列导入（数值列需在数据操作阶段显式转换）")
                return "chunked-utf8"

    # Parquet 是原生流式源，sink 既快又省，无需分块。
    infer = options.get("infer_schema_length", _infer_rows())
    if _attempt_sink(infer, "流式入库"):
        return "streaming"
    if _attempt_sink(None, "全量推断 schema 后流式入库"):
        return "streaming-reinfer"
    if _attempt_sink(0, "全字符串列流式入库"):
        warnings.append("已退化为全字符串列导入（数值列需在数据操作阶段显式转换）")
        return "streaming-utf8"

    raise LoadError(
        "流式入库失败：所有策略均未成功",
        details={"path": str(source), "format": fmt, "warnings": warnings[-3:]},
    )
