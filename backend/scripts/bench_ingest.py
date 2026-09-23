"""最终基准：三种入库路径在多个数据量下的峰值内存与吞吐。

运行（cwd 需为 backend/）：
    ./.venv/Scripts/python.exe scripts/bench_ingest.py

路径：
  legacy   改造前的链路（整个文件 bytes → DataFrame → BytesIO → 写盘）
  sink     ingest_to_parquet 的「小文件快路径」（scan_csv → sink_parquet）
  chunked  ingest_to_parquet 的「大文件有界路径」（按字节分块 + 增量写行组）
  floor    只 import polars/pyarrow，用于扣除地基内存

每条在独立子进程里跑，父进程读子进程自报的 PeakWorkingSetSize —— 同进程比较会被
「内存不归还」污染，PeakWorkingSetSize 更是整进程生命周期峰值，跑过就降不下来。

用途：`docs/大数据规模优化与吞吐提升方案.md` §5 的基准表由此脚本产出，
验收指标「规模翻倍时内存倍率 < 1.3×」也由它跟踪。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent  # backend/（本脚本在 backend/scripts/ 下）
PY = str(BACKEND / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(BACKEND))

SIZES = (1_000_000, 2_000_000, 4_000_000, 8_000_000)


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_K32 = ctypes.WinDLL("kernel32", use_last_error=True)
_K32.GetCurrentProcess.restype = wt.HANDLE
try:
    _GPMI = _K32.K32GetProcessMemoryInfo
except AttributeError:  # pragma: no cover
    _GPMI = ctypes.WinDLL("psapi").GetProcessMemoryInfo
_GPMI.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
_GPMI.restype = wt.BOOL


def peak_mb() -> float:
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not _GPMI(_K32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error())
    return c.PeakWorkingSetSize / 1048576


def run_legacy(src: Path, dst: Path) -> tuple[int, str]:
    """改造前的链路：三次全量常驻内存。"""
    content = src.read_bytes()
    from app.data_engine.loaders import REGISTRY

    loaded = REGISTRY.load(src.name, data=content)
    buffer = io.BytesIO()
    loaded.df.write_parquet(buffer)
    dst.write_bytes(buffer.getvalue())
    return loaded.df.height, "materialize"


def run_ingest(src: Path, dst: Path, *, force: str) -> tuple[int, str]:
    """走真实入口 ingest_to_parquet，用配置把它逼到指定路径。"""
    from app.core.config import settings

    if force == "sink":
        settings.INGEST_STREAMING_THRESHOLD_BYTES = 1 << 62  # 一律走 sink
    else:
        settings.INGEST_STREAMING_THRESHOLD_BYTES = 0  # 一律走 chunked

    from app.data_engine.ingest import ingest_to_parquet

    result = ingest_to_parquet(src, dst)
    return result.row_count, result.strategy


def child(mode: str, src: Path, dst: Path) -> int:
    payload: dict[str, object] = {"mode": mode}
    gc.collect()
    t0 = time.perf_counter()
    try:
        if mode == "floor":
            import polars  # noqa: F401
            import pyarrow.parquet  # noqa: F401

            rows, strategy = -1, "-"
        elif mode == "legacy":
            rows, strategy = run_legacy(src, dst)
        elif mode in ("sink", "chunked"):
            rows, strategy = run_ingest(src, dst, force=mode)
        else:
            raise SystemExit(f"unknown mode {mode}")
        payload.update(rows=rows, strategy=strategy)
    except Exception as exc:  # noqa: BLE001 - 基准脚本要报告失败而不是崩掉
        payload.update(error=f"{type(exc).__name__}: {str(exc)[:200]}")

    payload["elapsed"] = time.perf_counter() - t0
    payload["peak_mb"] = peak_mb()
    payload["dest_mb"] = dst.stat().st_size / 1048576 if dst.exists() else 0.0
    print("RESULT " + json.dumps(payload, ensure_ascii=False))
    return 0


def run(mode: str, src: Path, dst: Path) -> dict[str, object]:
    p = subprocess.Popen(
        [PY, str(Path(__file__).resolve()), "--child", mode, str(src), str(dst)],
        cwd=str(BACKEND),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out, err = p.communicate()
    for line in (out or "").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"{mode} 无结果 rc={p.returncode}:\n{err[-2000:]}")


def make_csv(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("编号,名称,分数,城市,备注\n")
        for i in range(rows):
            fh.write(
                f"{i},user_{i},{i % 1000}.25,{'北京' if i % 3 else '上海'},备注文本{i % 50}\n"
            )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bench_final_"))
    files = {}
    print("生成测试数据…")
    for rows in SIZES:
        p = tmp / f"rows_{rows}.csv"
        make_csv(p, rows)
        files[rows] = p
    print()

    floor = float(run("floor", files[SIZES[0]], tmp / "x.parquet")["peak_mb"])
    print(f"地基 floor（只 import polars+pyarrow）= {floor:.1f} MB\n")

    hdr = (
        f"{'路径':>9} {'行数':>10} {'CSV(MB)':>8} {'耗时(s)':>8} "
        f"{'吞吐(MB/s)':>10} {'峰值(MB)':>9} {'数据相关(MB)':>12} {'策略':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    results: dict[tuple[str, int], dict[str, object]] = {}
    for mode, sizes in (
        ("legacy", (1_000_000, 2_000_000, 4_000_000)),
        ("sink", SIZES),
        ("chunked", SIZES),
    ):
        for rows in sizes:
            src = files[rows]
            size_mb = src.stat().st_size / 1048576
            m = run(mode, src, tmp / f"{mode}_{rows}.parquet")
            results[(mode, rows)] = m
            if m.get("error"):
                print(f"{mode:>9} {rows:>10,} {size_mb:>8.1f} FAILED: {str(m['error'])[:60]}")
                continue
            el = float(m["elapsed"])
            pm = float(m["peak_mb"])
            print(
                f"{mode:>9} {rows:>10,} {size_mb:>8.1f} {el:>8.2f} "
                f"{size_mb / el:>10.0f} {pm:>9.1f} {pm - floor:>12.1f} "
                f"{str(m['strategy']):>12}"
            )

    # 规模翻倍时的内存增长：有界 vs 线性
    print()
    print("== 4M → 8M 行（数据 ×2）时的峰值内存增长 ==")
    for mode in ("sink", "chunked"):
        a = results.get((mode, 4_000_000))
        b = results.get((mode, 8_000_000))
        if not a or not b or a.get("error") or b.get("error"):
            continue
        da, db = float(a["peak_mb"]) - floor, float(b["peak_mb"]) - floor
        verdict = "有界 ✅" if db / max(da, 1e-9) < 1.3 else "线性 ❌"
        print(
            f"   {mode:>8}: 数据相关 {da:7.1f} → {db:7.1f} MB "
            f"= {db / max(da, 1e-9):.2f}×  {verdict}"
        )

    print()
    print("== 8M 行（377 MB CSV）三方对比 ==")
    for mode in ("legacy", "sink", "chunked"):
        m = results.get((mode, 8_000_000))
        if not m:
            print(f"   {mode:>8}: （未测：改造前链路在 8M 行会吃掉过多内存）")
            continue
        if m.get("error"):
            print(f"   {mode:>8}: FAILED {str(m['error'])[:70]}")
            continue
        print(
            f"   {mode:>8}: 耗时 {float(m['elapsed']):6.2f}s  "
            f"峰值 {float(m['peak_mb']):7.1f} MB  吞吐 {377.0 / float(m['elapsed']):5.0f} MB/s"
        )

    # 下推验证
    import polars as pl

    dest = tmp / f"chunked_{SIZES[-1]}.parquet"
    if dest.exists():
        gc.collect()
        t = time.perf_counter()
        small = (
            pl.scan_parquet(str(dest))
            .select(["编号", "分数"])
            .filter(pl.col("编号") >= SIZES[-1] - 100)
            .collect()
        )
        t_push = time.perf_counter() - t
        gc.collect()
        t = time.perf_counter()
        full = pl.read_parquet(dest)
        t_full = time.perf_counter() - t
        print()
        print(
            f"== 投影 + 谓词下推（{SIZES[-1]:,} 行 Parquet）==\n"
            f"   只取 2 列 / 末 100 行：{t_push * 1000:6.1f} ms → {small.height} 行\n"
            f"   全表物化读取        ：{t_full * 1000:6.1f} ms（{full.height:,}×{full.width}）\n"
            f"   加速比 {t_full / max(t_push, 1e-9):.1f}×"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(child(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])))
    main()
