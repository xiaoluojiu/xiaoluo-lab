"""数据规模与吞吐测试。

两件事需要被真正验证，而不是只看代码写没写：

1. **入库是流式的**：给一个明显大于「一次性物化会很吃力」的文件，断言走的是
   ``scan_* + sink_parquet`` 路径（strategy == "streaming"），并且结果与逐行
   对照一致。这里用 30 万行 CSV（约 10 MB+），既足以让三级降级逻辑真实生效，
   又不至于让测试套件变慢。
2. **读取能下推**：``scan_version`` 返回懒帧，列投影与谓词由 Polars 在 Parquet
   扫描层完成（断言只取出需要的列、行数正确）。

刻意不断言「内存峰值 < X MB」：那类断言在不同 Polars 版本/平台上噪声极大，
必然变成 flaky 测试。改为断言**策略标记**——它才是实现选择的稳定证据。
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from app.data_engine import ingest
from app.data_engine.exceptions import UnsupportedFormat
from app.services.dataset_service import DatasetService
from app.services.file_service import FileService, get_max_upload_size

LARGE_ROWS = 300_000


@pytest.fixture()
def large_csv(tmp_path) -> Path:
    """生成一份 30 万行 CSV（含中文列名与文本列）。"""
    path = tmp_path / "large.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,名称,分数,城市\n")
        for i in range(LARGE_ROWS):
            handle.write(f"{i},user_{i},{i % 1000}.25,{'北京' if i % 3 else '上海'}\n")
    return path


# =========================================================
# 上传限额：配置驱动
# =========================================================


def test_upload_limit_is_configurable_not_hardcoded():
    """上限必须来自配置——「大数据平台只收 100 MB」正是本次要拆掉的东西。"""
    from app.core.config import settings

    assert get_max_upload_size() == int(settings.MAX_UPLOAD_SIZE_BYTES)
    # 默认值必须显著高于 100 MB，否则「提升规模上限」这件事没有发生
    assert get_max_upload_size() > 100 * 1024 * 1024

    limits = settings.upload_limits()
    assert limits["max_size_bytes"] == get_max_upload_size()
    assert limits["streaming_ingest"] is True


def test_files_limits_endpoint_exposes_cap(client):
    """前端不再硬编码，因此必须有一个端点下发上限。"""
    resp = client.get("/api/v1/files/limits")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["max_size_bytes"] == get_max_upload_size()
    assert ".csv" in data["allowed_extensions"]


# =========================================================
# 流式入库
# =========================================================


def test_format_normalization_and_rejection():
    assert ingest.normalize_format("a.CSV") == "csv"
    assert ingest.normalize_format("b.pq") == "parquet"
    assert ingest.normalize_format("c.jsonl") == "ndjson"
    assert ingest.supports_streaming("csv") is True
    assert ingest.supports_streaming("xlsx") is False

    with pytest.raises(UnsupportedFormat):
        ingest.normalize_format("d.exe")


def test_large_csv_ingest_uses_streaming_strategy(large_csv: Path, tmp_path: Path):
    dest = tmp_path / "out.parquet"

    started = time.perf_counter()
    result = ingest.ingest_to_parquet(large_csv, dest)
    elapsed = time.perf_counter() - started

    assert result.strategy == "streaming"
    assert result.row_count == LARGE_ROWS
    assert result.column_count == 4
    assert result.schema["编号"] == "Int64"
    # 压缩后应明显小于源文件（列式 + zstd）
    assert result.compression_ratio < 1.0
    assert result.elapsed_seconds > 0
    assert result.throughput_mb_s > 0
    assert result.warnings == []

    # 落盘文件真的可读，且抽样数据正确
    frame = pl.read_parquet(dest)
    assert frame.height == LARGE_ROWS
    assert frame["城市"].n_unique() == 2
    assert frame.sort("编号").head(1)["名称"][0] == "user_0"

    # 计数与 schema 探测是「读 footer」而不是再解一遍全表
    assert ingest.count_rows(dest) == LARGE_ROWS
    assert ingest.read_schema(dest)["城市"] == "String"

    # 30 万行的入库不该慢到离谱（放宽到 60s，避免 CI 抖动导致假红）
    assert elapsed < 60


def test_gbk_csv_is_transcoded_then_streamed(tmp_path: Path):
    """GBK 是中文数据最常见的编码，Polars 不直接支持 ⇒ 必须先转码。"""
    path = tmp_path / "gbk.csv"
    with path.open("w", encoding="gbk", newline="") as handle:
        handle.write("编号,城市\n")
        for i in range(1000):
            handle.write(f"{i},广州\n")

    dest = tmp_path / "gbk.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "streaming"
    assert result.row_count == 1000
    assert any("转码" in item for item in result.warnings)

    frame = pl.read_parquet(dest)
    assert frame["城市"][0] == "广州"
    assert frame.columns == ["编号", "城市"]

    # 转码临时文件必须被清理（否则每导入一次就留一个临时文件）
    leftovers = [p.name for p in tmp_path.iterdir() if ".utf8.tmp" in p.name]
    assert leftovers == []


def test_materialize_format_is_size_capped(tmp_path: Path, monkeypatch):
    """不可流式格式（XLSX / 标准 JSON）超限时必须明确拒绝，而不是把进程 OOM 掉。

    注意这里**不能**再用 ARFF 当样本：ARFF 的 ``@data`` 段现已支持流式，
    不再受物化上限约束（见本文件末尾的 ARFF 用例）。
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "INGEST_MAX_MATERIALIZE_BYTES", 10, raising=False)

    payload = tmp_path / "big.json"
    payload.write_text('[{"a": 1}, {"a": 2}, {"a": 3}]', encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        ingest.ingest_to_parquet(payload, tmp_path / "out.parquet")

    assert "INGEST_MATERIALIZE_TOO_LARGE" in str(excinfo.value) or "物化上限" in str(
        excinfo.value
    )


# =========================================================
# 数据集版本：流式写 + 懒读
# =========================================================


def test_create_version_from_source_is_streaming(db, storage, large_csv: Path):
    service = DatasetService(db, storage)
    dataset = service.create(name="大表")

    version, result = service.create_version_from_source(dataset.id, large_csv)

    assert version.version == 1
    assert version.row_count == LARGE_ROWS
    assert version.column_count == 4
    assert result.strategy == "streaming"
    assert storage.exists(version.storage_path)

    # 关键：暂存文件已被原子提升，不该留下 .writing 残留
    leftovers = [item.key for item in storage.list(f"datasets/{dataset.id}/") if ".writing" in item.key]
    assert leftovers == []


def test_scan_version_pushes_down_projection_and_filter(db, storage, large_csv: Path):
    service = DatasetService(db, storage)
    dataset = service.create(name="大表")
    service.create_version_from_source(dataset.id, large_csv)

    lazy = service.scan_version(dataset.id)
    assert isinstance(lazy, pl.LazyFrame)

    # 只取两列 + 过滤：Polars 会把投影与谓词下推到 Parquet 扫描层
    projected = (
        lazy.select(["编号", "分数"]).filter(pl.col("编号") >= LARGE_ROWS - 10).collect()
    )
    assert projected.columns == ["编号", "分数"]
    assert projected.height == 10
    assert projected["编号"].min() == LARGE_ROWS - 10

    # 带 columns 的懒加载同样只产出请求的列
    lazy_cols = service.scan_version(dataset.id, columns=["城市"])
    assert lazy_cols.collect().columns == ["城市"]


def test_load_version_column_pruning_still_works(db, storage, large_csv: Path):
    """列裁剪是既有的优化（直接把列投影读进内存），改造后不能退化。"""
    service = DatasetService(db, storage)
    dataset = service.create(name="大表")
    service.create_version_from_source(dataset.id, large_csv)

    frame = service.load_version(dataset.id, columns=["编号", "城市"])

    assert frame.columns == ["编号", "城市"]
    assert frame.height == LARGE_ROWS


def test_file_service_stream_download_reads_by_chunks(db, storage):
    """流式下载：内存占用应恒为块大小，而不是整个文件。"""
    file_service = FileService(db, storage)
    payload = b"a,b,c\n" + b"1,2,3\n" * 5000
    record = file_service.upload("big.csv", payload)

    same, chunks = file_service.open_stream(record.id, chunk_size=1024)
    assert same.id == record.id

    collected = b"".join(chunks)
    assert collected == payload
    # 分块生效（1 KB 块 / 30 KB 数据 ⇒ 至少 30 个块）
    assert len(list(file_service.open_stream(record.id, chunk_size=1024)[1])) >= 30


def test_upload_promotes_temp_file_into_storage(db, storage):
    """上传后 Storage 里必须是完整文件（promote 走 rename 而非半截拷贝）。"""
    file_service = FileService(db, storage)
    record = file_service.upload("data.csv", b"x,y\n1,2\n")

    assert storage.exists(record.path)
    assert storage.read(record.path) == b"x,y\n1,2\n"
    assert storage.metadata(record.path).size == 8


# =========================================================
# 分块入库（常量内存路径）
# =========================================================


@pytest.fixture()
def force_chunked(monkeypatch):
    """把「大文件」阈值压到 0、块压到 64 KB，强制走分块路径。

    真实大文件测试要跑出几百 MB 才能验证内存有界，那不适合放进常规测试套件
    （实测数据见 docs/大数据规模优化与吞吐提升方案.md）。这里只验证**正确性**：
    路径被选中、边界切分不破坏记录。

    ★ 测试数据必须**大于实际块大小**，否则整份文件就是一个块，切分逻辑根本没被执行
      —— 第一版就踩了这个坑：块被写成 4 KB，但 `_chunk_bytes()` 的下限是 1 MB，
      补丁被静默钳掉，于是「跨块」用例全都只是「单块」用例的重复。
      下面显式断言钳制后的真实值，避免同类假绿再次发生。
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "INGEST_STREAMING_THRESHOLD_BYTES", 0, raising=False)
    monkeypatch.setattr(settings, "INGEST_CHUNK_BYTES", 64 * 1024, raising=False)

    assert ingest._chunk_bytes() == 64 * 1024, "块大小补丁被下限钳掉了，用例会退化为单块"
    assert ingest._streaming_threshold_bytes() == 0
    return settings


# 分块用例的数据体量：必须显著大于 64 KB 的块，才能真的产生多个切点。
CHUNKED_ROWS = 20_000


def test_threshold_routes_small_files_to_sink_and_large_to_chunked(
    tmp_path: Path, monkeypatch
):
    """自适应分界：同一份文件，阈值决定走哪条路——这是「快」与「有界」的取舍开关。"""
    from app.core.config import settings

    path = tmp_path / "small.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,城市\n")
        for i in range(200):
            handle.write(f"{i},北京\n")

    # 默认阈值（256 MB）⇒ 文件远小于阈值 ⇒ 走快的 sink 路径
    default_result = ingest.ingest_to_parquet(path, tmp_path / "a.parquet")
    assert default_result.strategy == "streaming"

    # 阈值压到 0 ⇒ 一律走有界的分块路径
    monkeypatch.setattr(settings, "INGEST_STREAMING_THRESHOLD_BYTES", 0, raising=False)
    chunked_result = ingest.ingest_to_parquet(path, tmp_path / "b.parquet")
    assert chunked_result.strategy == "chunked"

    # 两条路径的产出一致
    assert chunked_result.row_count == default_result.row_count == 200
    assert chunked_result.schema == default_result.schema


def test_chunked_ingest_matches_full_read(tmp_path: Path, force_chunked):
    """分块路径的产出一致，不能因为切块丢行/串列。"""
    rows = CHUNKED_ROWS
    path = tmp_path / "big.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,名称,分数,城市\n")
        for i in range(rows):
            handle.write(f"{i},user_{i},{i % 1000}.25,{'北京' if i % 3 else '上海'}\n")

    assert path.stat().st_size > 64 * 1024 * 2, "数据必须跨多个块，否则没测到切分逻辑"

    dest = tmp_path / "out.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "chunked"
    assert result.row_count == rows
    assert result.column_count == 4

    frame = pl.read_parquet(dest)
    expected = pl.read_csv(path, infer_schema_length=None)

    assert frame.height == expected.height == rows
    # 分块读必须与一次性全量读逐值相等（顺序也要一致）
    assert frame.equals(expected)
    assert frame["编号"].to_list() == list(range(rows))


def test_chunked_reader_does_not_split_quoted_newline(tmp_path: Path, force_chunked):
    """RFC 4180 允许引号字段内含换行；按 \\n 硬切会把一条记录劈成两半。

    这里刻意让**单条记录**（约 160 KB）远大于 64 KB 的块，使它必然跨越多个切点 ——
    只有这样才能证明切块器真的在追踪引号状态，而不是碰巧没切到换行。
    """
    path = tmp_path / "quoted.csv"
    left = "甲" * 30_000  # 引号内的前半段
    right = "乙" * 30_000  # 引号内的后半段（中间隔着换行）
    embedded = f"{left}\n{right}"

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,备注\n")
        handle.write(f'1,"{embedded}"\n')
        handle.write('2,"含""引号""的值"\n')
        handle.write("3,普通\n")

    # 60 000 个汉字 ≈ 180 KB ⇒ 至少 2 个 64 KB 块，记录必然跨越切点
    assert path.stat().st_size > 64 * 1024 * 2, "数据必须跨多个块，否则没测到切分逻辑"

    dest = tmp_path / "quoted.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "chunked"
    assert result.row_count == 3

    frame = pl.read_parquet(dest).sort("编号")
    assert frame["备注"][0] == embedded
    assert frame["备注"][0].count("\n") == 1
    assert frame["备注"].to_list()[1:] == ['含"引号"的值', "普通"]


def test_chunked_reader_handles_missing_trailing_newline(tmp_path: Path, force_chunked):
    """最后一行没有换行符是常见情况，不能在收尾时把它丢掉。"""
    path = tmp_path / "no_eol.csv"
    body = b"".join(f"{i},value_{i}\n".encode() for i in range(CHUNKED_ROWS))
    path.write_bytes(b"a,b\n" + body + b"999999,last_row")  # 结尾无 \n

    dest = tmp_path / "no_eol.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "chunked"
    assert result.row_count == CHUNKED_ROWS + 1
    frame = pl.read_parquet(dest)
    assert frame["b"][-1] == "last_row"
    assert frame["b"][0] == "value_0"


def test_iter_text_chunks_only_cuts_outside_quotes():
    """直接单测切块器：引号内的换行不能被当作切点。"""
    from io import BytesIO

    payload = b'1,"a\nb"\n2,c\n3,"d\ne"\n'
    chunks = list(ingest._iter_text_chunks(BytesIO(payload), 4))

    # 拼回去必须无损（既没吞字符，也没多插字符）
    assert b"".join(chunks) == payload
    # 每个块都必须是完整记录：以 \n 结尾，且引号数为偶数
    for chunk in chunks:
        assert chunk.count(b'"') % 2 == 0
        assert chunk.endswith(b"\n")
    # 含引号换行的两条记录不能被切开
    assert any(b'"a\nb"' in c for c in chunks)
    assert any(b'"d\ne"' in c for c in chunks)


def test_decode_probe_tolerates_truncated_multibyte_tail():
    """探测头可能把汉字劈成两半 —— 必须容忍，否则整份文件被误判成 latin-1。

    刻意用「掐掉结尾若干字节」而不是「造一个刚好超 64 KB 的文件」来构造截断：
    后者取决于内容排布，边界是不是落在汉字中间全凭运气（第一版就是这么写的，
    结果那份数据恰好对齐，用例变成空转，还得靠额外断言才发现）。
    """
    from app.data_engine.loaders import _decode_probe

    body = ("编号,城市\n" + "广州" * 200).encode("gbk")

    # trim=1 掐掉「州」的尾字节；trim=3 连「广」的尾字节一起掐掉 ⇒ 两者都留下悬空前导字节
    for trim in (1, 3):
        truncated = body[: len(body) - trim]
        with pytest.raises(UnicodeDecodeError):
            truncated.decode("gb18030")
        assert _decode_probe(truncated, "gb18030") is True

    # 未截断时当然也要为真
    assert _decode_probe(body, "gb18030") is True


def test_gbk_file_over_probe_head_is_not_misdetected_as_latin1(tmp_path: Path):
    """超过 64 KB 的 GBK 文件：端到端不允许再出现乱码。

    修复前 `_read_head` 的 64 000 字节固定切点一旦落在汉字中间，
    `data.decode("gb18030")` 就抛错并回落 latin-1 ⇒ 列名与内容全乱。
    """
    from app.data_engine.loaders import _detect_encoding, _read_head

    gbk_file = tmp_path / "gbk_big.csv"
    with gbk_file.open("w", encoding="gbk", newline="") as handle:
        handle.write("编号,城市,说明\n")
        for i in range(20_000):
            handle.write(f"{i},广州,中文说明{i}\n")
    assert gbk_file.stat().st_size > 64_000, "必须超过探测头长度"

    head = _read_head(str(gbk_file))
    assert len(head) == 64_000
    # 关键断言：不能是 latin-1
    assert _detect_encoding(head, "auto") == "gb18030"

    dest = tmp_path / "gbk_big.parquet"
    result = ingest.ingest_to_parquet(gbk_file, dest)

    assert result.row_count == 20_000
    assert any("转码" in item for item in result.warnings)

    frame = pl.read_parquet(dest)
    # 列名没被读成乱码
    assert frame.columns == ["编号", "城市", "说明"]
    assert frame["城市"].unique().to_list() == ["广州"]
    assert frame["说明"][-1] == "中文说明19999"


def test_chunked_gbk_uses_transcode(tmp_path: Path, force_chunked):
    """GBK 源在分块路径下同样要先转码，且中文不因跨块解码而乱码。"""
    path = tmp_path / "gbk_big.csv"
    with path.open("w", encoding="gbk", newline="") as handle:
        handle.write("编号,城市,说明\n")
        for i in range(CHUNKED_ROWS):
            handle.write(f"{i},广州,中文说明{i}\n")

    dest = tmp_path / "gbk_big.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "chunked"
    assert result.row_count == CHUNKED_ROWS
    assert any("转码" in item for item in result.warnings)

    frame = pl.read_parquet(dest)
    assert frame["城市"].unique().to_list() == ["广州"]
    assert frame["说明"][CHUNKED_ROWS - 1] == f"中文说明{CHUNKED_ROWS - 1}"
    # 转码临时文件必须清理
    assert [p.name for p in tmp_path.iterdir() if ".utf8.tmp" in p.name] == []


def test_chunked_falls_back_to_all_strings_when_types_conflict(
    tmp_path: Path, force_chunked
):
    """首块推断成整数列、后续块出现无法转换的值 ⇒ 不能整体失败，退化为字符串列导入。

    这是分块路径特有的风险：**采样只看得到第一块**。

    三个必须踩对的细节（前两版都写错了，留作记录）：
    1. 整数值的铺垫必须**超过一个块**（64 KB），确保畸形值落不到首块里 ——
       否则首块自己就把该列推断成字符串，压根走不到回退逻辑；
    2. 畸形值不能用 ``N/A`` / ``NA`` / ``null`` —— Polars 默认视其为空值，
       同样触发不了类型冲突。这里用 ``待补``；
    3. 块大小补丁会被 `_chunk_bytes()` 的下限钳制，故 `force_chunked` 里做了断言。
    """
    path = tmp_path / "conflict.csv"
    int_rows = CHUNKED_ROWS  # ≈ 180 KB ⇒ 前两块是纯整数
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,分数\n")
        for i in range(int_rows):
            handle.write(f"{i},{i}\n")
        handle.write("9001,待补\n")
        handle.write("9002,待补\n")

    dest = tmp_path / "conflict.parquet"
    result = ingest.ingest_to_parquet(path, dest)

    assert result.strategy == "chunked-utf8"
    assert result.row_count == int_rows + 2
    assert any("全字符串列" in item for item in result.warnings)

    frame = pl.read_parquet(dest)
    assert frame["分数"].dtype == pl.Utf8
    assert frame["分数"][-1] == "待补"
    # 退化后前面的整数仍是可读文本，没有丢行
    assert frame["分数"][0] == "0"


def test_chunked_honours_column_projection(tmp_path: Path, force_chunked):
    """``columns`` 投影在分块路径上同样生效。

    实现上有个易错点：首块的 schema 必须在 ``select`` **之前**取（后续块要按**完整**
    schema 解析文本，再投影），否则第二个块会因为列数对不上而报错。
    """
    path = tmp_path / "projected.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("编号,名称,分数,城市\n")
        for i in range(CHUNKED_ROWS):
            handle.write(f"{i},user_{i},{i % 1000}.25,北京\n")

    dest = tmp_path / "projected.parquet"
    result = ingest.ingest_to_parquet(path, dest, options={"columns": ["编号", "城市"]})

    assert result.strategy == "chunked"
    assert result.row_count == CHUNKED_ROWS
    assert result.column_count == 2

    frame = pl.read_parquet(dest)
    assert frame.columns == ["编号", "城市"]
    assert frame.height == CHUNKED_ROWS
    assert frame["编号"][0] == 0
    assert frame["编号"][-1] == CHUNKED_ROWS - 1


def test_both_ingest_paths_agree_on_ragged_rows(tmp_path: Path, monkeypatch):
    """同一个文件走 sink 与走 chunked，结果必须一致。

    两条路径由「体积是否超过阈值」自动选择，因此**行为差异会表现为「小文件能导入、
    大文件报错」这类极难排查的 bug**。这里用一份含参差行的 CSV 把不变量钉住。
    """
    from app.core.config import settings

    path = tmp_path / "ragged.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("a,b\n")
        for i in range(CHUNKED_ROWS):
            handle.write(f"{i},{i}\n")
        # 这一行**多一个字段**（真正的参差行）。注意别写成「两列但第二列不是数字」——
        # 那是类型冲突而非参差行，会走 chunked-utf8 分支，测不到本用例要钉的东西。
        handle.write("500000,500000,extra_field_here\n")

    sink_result = ingest.ingest_to_parquet(path, tmp_path / "sink.parquet")
    assert sink_result.strategy == "streaming"

    monkeypatch.setattr(settings, "INGEST_STREAMING_THRESHOLD_BYTES", 0, raising=False)
    monkeypatch.setattr(settings, "INGEST_CHUNK_BYTES", 64 * 1024, raising=False)
    chunked_result = ingest.ingest_to_parquet(path, tmp_path / "chunked.parquet")
    assert chunked_result.strategy == "chunked"

    # 行数、列名、schema 三者必须一致
    assert chunked_result.row_count == sink_result.row_count == CHUNKED_ROWS + 1
    assert chunked_result.schema == sink_result.schema

    sink_frame = pl.read_parquet(tmp_path / "sink.parquet")
    chunked_frame = pl.read_parquet(tmp_path / "chunked.parquet")
    assert chunked_frame.equals(sink_frame)


# =========================================================
# ARFF：头部扫描 + @data 段流式
# =========================================================


# 刻意覆盖 ARFF 的几个易错点：
#  - 行首注释 `%` 与空行（头部扫描要跳过，且不能算进数据段）
#  - 标称值用单引号包裹，且**取值内含逗号**（'b,c'）与空格（'d e'）
#  - 缺失值 `?` 与空字段（二者在旧路径都映射为 None）
DENSE_ARFF = """% 注释行，应被忽略
@relation demo

@attribute id numeric
@attribute name {'a','b,c','d e'}
@attribute score numeric
@attribute note string

@data
1,'a',3.5,hello
2,'b,c',?,world
3,'d e',10,
4,'a',2.25,'has space'
"""

SPARSE_ARFF = """@relation sp
@attribute x numeric
@attribute y numeric
@attribute z numeric
@data
{0 1, 2 3}
{1 5}
"""


def test_arff_dense_goes_through_streaming_path(tmp_path: Path, monkeypatch):
    """稠密 ARFF 必须走流式路径 —— 用「物化上限设为 1 字节」把它逼出来。

    若实现仍走 `_materialize`，会因超过 INGEST_MAX_MATERIALIZE_BYTES 直接报错；
    能成功即证明没有走整体物化（这是比断言 strategy 更强的证据）。
    """
    from app.core.config import settings

    path = tmp_path / "dense.arff"
    path.write_text(DENSE_ARFF, encoding="utf-8")

    monkeypatch.setattr(settings, "INGEST_MAX_MATERIALIZE_BYTES", 1, raising=False)

    result = ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    assert result.strategy == "chunked-arff"
    assert result.row_count == 4
    assert result.column_count == 4


def test_arff_streaming_matches_materialized(tmp_path: Path):
    """两条路径必须**逐值相等**。

    这是最重要的一条不变量：走哪条路由实现决定，若结果不同，
    就会出现「同一个文件今天导入和明天导入不一样」这种最难查的问题。
    """
    from app.data_engine.loaders import REGISTRY

    path = tmp_path / "dense.arff"
    path.write_text(DENSE_ARFF, encoding="utf-8")

    reference = REGISTRY.load(str(path)).df
    ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    streamed = pl.read_parquet(tmp_path / "out.parquet")

    # dtype 也要一致（numeric → Float64，标称/string → Utf8）
    assert dict(streamed.schema) == dict(reference.schema)
    assert streamed.equals(reference)

    rows = streamed.sort("id").to_dicts()
    # 引号内的逗号没被当成分隔符
    assert rows[1]["name"] == "b,c"
    # 引号内的空格保留
    assert rows[2]["name"] == "d e"
    # ? 与空字段都成为 null
    assert rows[1]["score"] is None
    assert rows[2]["note"] is None


def test_arff_sparse_falls_back_to_materialize(tmp_path: Path):
    """稀疏 ARFF 不是定宽列 → 必须整体物化，且给出可读告警。"""
    path = tmp_path / "sparse.arff"
    path.write_text(SPARSE_ARFF, encoding="utf-8")

    result = ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    assert result.strategy == "materialized(streaming-unsupported)"
    assert any("稀疏" in w for w in result.warnings)

    frame = pl.read_parquet(tmp_path / "out.parquet")
    assert frame.sort("x", nulls_last=True).to_dicts()[0] == {
        "x": 1.0,
        "y": None,
        "z": 3.0,
    }


def test_arff_header_scan_skips_comments_and_finds_data_offset(tmp_path: Path):
    """头部扫描要跳过注释/空行，并把偏移精确指到第一条数据。"""
    path = tmp_path / "dense.arff"
    path.write_text(DENSE_ARFF, encoding="utf-8")

    plan = ingest._scan_arff_header(path, "utf-8")
    assert plan.columns == ["id", "name", "score", "note"]
    assert plan.dense is True
    assert plan.schema["id"] == pl.Float64
    assert plan.schema["name"] == pl.Utf8
    # 偏移处必须正好是第一行数据（前面有注释行与空行，容易差一行）
    assert path.read_bytes()[plan.data_offset:].startswith(b"1,'a'")

    result = ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    assert result.strategy == "chunked-arff"
    assert result.row_count == 4


def test_arff_gbk_is_transcoded_then_streamed(tmp_path: Path):
    """GBK 的 ARFF（中文属性名）要先转码再流式，不能乱码。"""
    path = tmp_path / "gbk.arff"
    path.write_bytes(
        (
            "@relation 中文\n"
            "@attribute 城市 string\n"
            "@attribute 值 numeric\n"
            "@data\n"
            "北京,1.5\n上海,2.5\n"
        ).encode("gb18030")
    )

    result = ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    assert result.strategy == "chunked-arff"
    assert any("转码" in w for w in result.warnings)

    frame = pl.read_parquet(tmp_path / "out.parquet")
    assert list(frame.columns) == ["城市", "值"]
    assert frame["城市"].to_list() == ["北京", "上海"]


def test_arff_streaming_honours_column_projection(tmp_path: Path):
    """列投影在 ARFF 流式路径上同样生效。"""
    path = tmp_path / "dense.arff"
    path.write_text(DENSE_ARFF, encoding="utf-8")

    result = ingest.ingest_to_parquet(
        path, tmp_path / "out.parquet", fmt="arff", options={"columns": ["id", "score"]}
    )
    assert result.strategy == "chunked-arff"

    frame = pl.read_parquet(tmp_path / "out.parquet")
    assert list(frame.columns) == ["id", "score"]


def test_arff_streaming_is_correct_across_small_chunks(tmp_path: Path, monkeypatch):
    """把块调到下限，跨块仍然正确（钉住切点逻辑）。"""
    from app.core.config import settings

    path = tmp_path / "many.arff"
    rows = 5000
    lines = ["@relation many", "@attribute a numeric", "@attribute b {'x','y,z'}", "@data"]
    for i in range(rows):
        lines.append(f"{i},'{'y,z' if i % 2 else 'x'}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(settings, "INGEST_CHUNK_BYTES", 64 * 1024, raising=False)
    assert ingest._chunk_bytes() == 64 * 1024

    result = ingest.ingest_to_parquet(path, tmp_path / "out.parquet", fmt="arff")
    assert result.strategy == "chunked-arff"
    assert result.row_count == rows

    frame = pl.read_parquet(tmp_path / "out.parquet")
    assert frame["a"].to_list() == [float(i) for i in range(rows)]
    # 每隔一行是带逗号的标称值——若切点/引号处理错了这里会炸或错位
    assert frame["b"].to_list() == ["y,z" if i % 2 else "x" for i in range(rows)]
